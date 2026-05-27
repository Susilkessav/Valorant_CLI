"""Post-game match analysis — Finding-based system.

Each analyzer inspects a single match from the perspective of one player
(``puuid``) and returns a list of ``Finding`` objects.  A Finding bundles
severity, a root-cause tag, a human-readable headline + detail, and the
raw evidence dict so the LLM can reason from numbers rather than paraphrasing.

Usage
-----
    match = await repo.get_post_game_match(session, puuid)
    findings = run_analyzers(match, puuid)
    top3    = select_top_findings(findings, n=3)
    block   = format_findings_block(top3)  # inject into LLM user message

Design principles
-----------------
- **Graceful degradation**: every analyzer declares ``required_fields``.
  ``run_analyzers()`` skips any analyzer whose fields are absent (NULL)
  on the loaded match, so pre-migration matches still produce useful output.
- **Root-cause grouping**: ``select_top_findings`` collapses multiple
  findings with the same ``root_cause_tag`` to the highest-severity one,
  preventing five different symptoms of "over-peeking" from flooding the prompt.
- **No LLM at analysis time**: all logic is deterministic Python.  The LLM
  receives structured evidence and writes the coaching narrative.

Severity scale
--------------
  positive  — a genuine strength to reinforce
  neutral   — informational, no strong judgment
  warning   — area that hurt this match
  critical  — likely primary driver of a loss / significant performance gap
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Literal

from valocoach.data.orm_models import Kill, Match, OrmMatchPlayer, Round, RoundPlayer

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Finding dataclass
# ---------------------------------------------------------------------------

Severity = Literal["positive", "neutral", "warning", "critical"]

_SEVERITY_RANK: dict[str, int] = {
    "critical": 3,
    "warning": 2,
    "neutral": 1,
    "positive": 0,
}


@dataclass
class Finding:
    """One coaching insight produced by a single post-game analyzer.

    ``required_fields`` lists ORM column names that must be non-NULL on at
    least one row in the match for this finding to be valid.  The runner
    checks these before invoking the analyzer.
    """

    severity: Severity
    category: str  # "duels" | "positioning" | "economy" | "utility" | "clutch"
    headline: str  # ≤ 80 chars — shown in the LLM prompt header
    detail: str  # full explanation with specific evidence
    evidence: dict  # raw numbers the LLM uses to reason
    root_cause_tag: str  # for redundancy grouping
    required_fields: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Agent role classification
# ---------------------------------------------------------------------------

_DUELISTS = frozenset(
    {
        "Jett",
        "Reyna",
        "Raze",
        "Phoenix",
        "Neon",
        "Iso",
        "Yoru",
        "Waylay",
    }
)
_CONTROLLERS = frozenset(
    {
        "Omen",
        "Brimstone",
        "Astra",
        "Harbor",
        "Viper",
        "Clove",
    }
)
_INITIATORS = frozenset(
    {
        "Sova",
        "Breach",
        "Fade",
        "KAY/O",
        "Gekko",
        "Skye",
        "Tejo",
    }
)
_SENTINELS = frozenset(
    {
        "Cypher",
        "Killjoy",
        "Sage",
        "Chamber",
        "Deadlock",
        "Vyse",
        "Miks",
    }
)

# Expected non-ultimate ability casts per round played (role baseline)
_UTIL_BASELINE: dict[str, float] = {
    "Initiator": 1.0,
    "Controller": 1.2,
    "Sentinel": 0.8,
    "Duelist": 0.6,
    "Unknown": 0.5,
}


def _agent_role(agent_name: str | None) -> str:
    if agent_name in _DUELISTS:
        return "Duelist"
    if agent_name in _CONTROLLERS:
        return "Controller"
    if agent_name in _INITIATORS:
        return "Initiator"
    if agent_name in _SENTINELS:
        return "Sentinel"
    return "Unknown"


# ---------------------------------------------------------------------------
# Match helpers
# ---------------------------------------------------------------------------


def _get_mp(match: Match, puuid: str) -> OrmMatchPlayer | None:
    """Return the MatchPlayer row for ``puuid`` in this match."""
    for mp in match.players:
        if mp.puuid == puuid:
            return mp
    return None


def _team_of(match: Match, puuid: str) -> str | None:
    """Return "Red" or "Blue" for ``puuid`` in this match, or None."""
    mp = _get_mp(match, puuid)
    return mp.team if mp else None


def _teammates(match: Match, puuid: str, my_team: str) -> set[str]:
    """Return puuids of all players on the same team (excluding self)."""
    return {p.puuid for p in match.players if p.team == my_team and p.puuid != puuid}


def _round_player(rnd: Round, puuid: str) -> RoundPlayer | None:
    for rp in rnd.round_players:
        if rp.puuid == puuid:
            return rp
    return None


def _my_kills_in_round(rnd: Round, puuid: str) -> list[Kill]:
    return [k for k in rnd.kills if k.killer_puuid == puuid]


def _my_deaths_in_round(rnd: Round, puuid: str) -> list[Kill]:
    return [k for k in rnd.kills if k.victim_puuid == puuid]


def _round_won(rnd: Round, my_team: str) -> bool:
    # Guard against pre-migration rows where winning_team may be NULL.
    return (rnd.winning_team or "") == my_team


def _has_field(match: Match, field_name: str) -> bool:
    """True when at least one kill or round row has a non-NULL value for ``field_name``."""
    if field_name in ("killer_x", "killer_y", "victim_x", "victim_y", "engagement_distance"):
        return any(
            getattr(k, field_name, None) is not None for rnd in match.rounds for k in rnd.kills
        )
    if field_name in ("plant_x", "plant_y", "defuse_x", "defuse_y"):
        return any(getattr(rnd, field_name, None) is not None for rnd in match.rounds)
    if field_name.startswith("ability_casts"):
        return any(
            getattr(rp, field_name, None) is not None
            for rnd in match.rounds
            for rp in rnd.round_players
        )
    return True  # structural fields are always present


def _can_run(match: Match, required: list[str]) -> bool:
    return all(_has_field(match, f) for f in required)


# ---------------------------------------------------------------------------
# Analyzer 1 — First contact (entry duel pattern)
# ---------------------------------------------------------------------------

_TRADE_WINDOW_MS = 5_000


def analyze_first_contact(match: Match, puuid: str) -> list[Finding]:
    """Who dies/frags first in each round?"""
    my_team = _team_of(match, puuid)
    if my_team is None:
        return []

    first_deaths = 0
    first_bloods = 0
    rounds_counted = 0

    for rnd in match.rounds:
        if not rnd.kills:
            continue
        ordered = sorted(rnd.kills, key=lambda k: k.time_in_round_ms or 0)
        first_kill = ordered[0]
        rounds_counted += 1
        if first_kill.victim_puuid == puuid:
            first_deaths += 1
        elif first_kill.killer_puuid == puuid:
            first_bloods += 1

    if rounds_counted == 0:
        return []

    findings: list[Finding] = []
    death_rate = first_deaths / rounds_counted
    blood_rate = first_bloods / rounds_counted

    if death_rate >= 0.30:
        sev: Severity = "critical" if death_rate >= 0.45 else "warning"
        findings.append(
            Finding(
                severity=sev,
                category="duels",
                headline=f"Died first in {first_deaths}/{rounds_counted} rounds ({death_rate:.0%})",
                detail=(
                    f"You were the first player to die in {death_rate:.0%} of rounds. "
                    "This typically means over-peeking, taking early contact on off-angles, "
                    "or pushing without utility. Your team loses map control and the round "
                    "momentum immediately. Focus on letting utility land before committing."
                ),
                evidence={
                    "first_deaths": first_deaths,
                    "first_bloods": first_bloods,
                    "rounds": rounds_counted,
                    "first_death_rate": round(death_rate, 3),
                },
                root_cause_tag="entry_failure",
            )
        )
    elif blood_rate >= 0.25:
        findings.append(
            Finding(
                severity="positive",
                category="duels",
                headline=f"First blood in {first_bloods}/{rounds_counted} rounds ({blood_rate:.0%})",
                detail=(
                    f"You opened {blood_rate:.0%} of rounds with a kill — strong entry fragging. "
                    "This creates 5v4 pressure for your team immediately."
                ),
                evidence={
                    "first_bloods": first_bloods,
                    "rounds": rounds_counted,
                    "first_blood_rate": round(blood_rate, 3),
                },
                root_cause_tag="entry_success",
            )
        )

    return findings


# ---------------------------------------------------------------------------
# Analyzer 2 — Economy decisions
# ---------------------------------------------------------------------------


def analyze_eco_decisions(match: Match, puuid: str) -> list[Finding]:
    """Detect force-buys against team save or saves against team buy."""
    my_team = _team_of(match, puuid)
    if my_team is None:
        return []

    force_vs_save = 0  # I bought, team saved
    save_vs_buy = 0  # I saved, team bought
    rounds_with_eco_data = 0

    for rnd in match.rounds:
        my_rp = _round_player(rnd, puuid)
        if my_rp is None or my_rp.loadout_value is None:
            continue

        team_rps = [
            rp
            for rp in rnd.round_players
            if rp.puuid != puuid and rp.team == my_team and rp.loadout_value is not None
        ]
        if not team_rps:
            continue

        rounds_with_eco_data += 1
        my_lv = my_rp.loadout_value
        team_avg = sum(rp.loadout_value for rp in team_rps) / len(team_rps)  # type: ignore[arg-type]

        # Force-buy: I spent significantly more than team average
        if my_lv > 2200 and team_avg < 1500:
            force_vs_save += 1
        # Under-buy: I spent significantly less than team
        elif my_lv < 1000 and team_avg > 2500:
            save_vs_buy += 1

    if rounds_with_eco_data == 0:
        return []

    findings: list[Finding] = []

    if force_vs_save >= 2:
        sev = "critical" if force_vs_save >= 4 else "warning"
        findings.append(
            Finding(
                severity=sev,
                category="economy",
                headline=f"Force-bought while team saved in {force_vs_save} rounds",
                detail=(
                    f"In {force_vs_save} rounds you spent significantly more than your teammates "
                    "who were saving. This breaks team economy synchronisation — you wasted "
                    "credits on rounds your team couldn't win as a unit and delayed the team "
                    "from hitting a full-buy together."
                ),
                evidence={
                    "force_buy_rounds": force_vs_save,
                    "rounds_analysed": rounds_with_eco_data,
                },
                root_cause_tag="bad_economy",
            )
        )

    if save_vs_buy >= 2:
        findings.append(
            Finding(
                severity="warning",
                category="economy",
                headline=f"Under-bought while team full-bought in {save_vs_buy} rounds",
                detail=(
                    f"In {save_vs_buy} rounds your team went full-buy but you saved credits. "
                    "Playing pistol/sheriff into a full-buy round mismatches your firepower "
                    "with the rest of your team's investment."
                ),
                evidence={
                    "under_buy_rounds": save_vs_buy,
                    "rounds_analysed": rounds_with_eco_data,
                },
                root_cause_tag="bad_economy",
            )
        )

    return findings


# ---------------------------------------------------------------------------
# Analyzer 3 — Utility efficiency
# ---------------------------------------------------------------------------


def analyze_utility_efficiency(match: Match, puuid: str) -> list[Finding]:
    """Compare ability casts per round to role baseline."""
    mp = _get_mp(match, puuid)
    if mp is None:
        return []

    role = _agent_role(mp.agent_name)
    baseline = _UTIL_BASELINE[role]

    total_non_ult = 0
    total_ult = 0
    rounds_with_data = 0

    for rnd in match.rounds:
        rp = _round_player(rnd, puuid)
        if rp is None or rp.ability_casts_ability1 is None:
            continue
        rounds_with_data += 1
        total_non_ult += (
            (rp.ability_casts_grenade or 0)
            + (rp.ability_casts_ability1 or 0)
            + (rp.ability_casts_ability2 or 0)
        )
        total_ult += rp.ability_casts_ultimate or 0

    if rounds_with_data < 3:
        return []

    avg_non_ult = total_non_ult / rounds_with_data
    findings: list[Finding] = []

    if avg_non_ult < baseline * 0.5:
        sev: Severity = "critical" if avg_non_ult < baseline * 0.25 else "warning"
        findings.append(
            Finding(
                severity=sev,
                category="utility",
                headline=(
                    f"Low utility usage: {avg_non_ult:.1f} casts/round "
                    f"(baseline {baseline:.1f} for {role})"
                ),
                detail=(
                    f"As a {role} on {mp.agent_name or 'your agent'}, you're expected to cast "
                    f"utilities ~{baseline:.1f}× per round. You averaged {avg_non_ult:.1f}×. "
                    "Under-using abilities means your kit isn't contributing to round outcomes — "
                    "smoke less, flash less, gather less info than your opponents expect."
                ),
                evidence={
                    "avg_non_ult_casts_per_round": round(avg_non_ult, 2),
                    "role_baseline": baseline,
                    "role": role,
                    "agent": mp.agent_name,
                    "total_ult_casts": total_ult,
                    "rounds_analysed": rounds_with_data,
                },
                root_cause_tag="low_utility",
                required_fields=["ability_casts_ability1"],
            )
        )
    elif total_ult == 0 and rounds_with_data >= 5:
        findings.append(
            Finding(
                severity="warning",
                category="utility",
                headline="Ultimate never used across the match",
                detail=(
                    f"You played {rounds_with_data} rounds with data and never used your ultimate. "
                    "Storing ult all game loses free value — use it early once it's ready, "
                    "then re-charge rather than holding indefinitely."
                ),
                evidence={
                    "ult_casts": 0,
                    "rounds_analysed": rounds_with_data,
                    "agent": mp.agent_name,
                },
                root_cause_tag="low_utility",
                required_fields=["ability_casts_ultimate"],
            )
        )

    return findings


# ---------------------------------------------------------------------------
# Analyzer 4 — Round timing (when do you die?)
# ---------------------------------------------------------------------------


def analyze_round_timing(match: Match, puuid: str) -> list[Finding]:
    """Detect systematic early deaths (over-peeking) or late deaths."""
    death_times: list[int] = []

    for rnd in match.rounds:
        for k in rnd.kills:
            if k.victim_puuid == puuid and k.time_in_round_ms is not None:
                death_times.append(k.time_in_round_ms)

    if len(death_times) < 4:
        return []

    median_ms = sorted(death_times)[len(death_times) // 2]
    early_deaths = sum(1 for t in death_times if t < 20_000)  # < 20 s
    early_pct = early_deaths / len(death_times)

    findings: list[Finding] = []

    if early_pct >= 0.45:
        sev: Severity = "critical" if early_pct >= 0.65 else "warning"
        findings.append(
            Finding(
                severity=sev,
                category="positioning",
                headline=f"Dying in first 20s in {early_pct:.0%} of death rounds",
                detail=(
                    f"{early_pct:.0%} of your deaths occurred within the first 20 seconds of the "
                    f"round (median death at {median_ms / 1000:.0f}s). Early deaths are almost always "
                    "overpeeks — pushing into uncleared angles or taking duels before your team "
                    "has planted util. Let smokes/flashes land before stepping onto site."
                ),
                evidence={
                    "early_deaths": early_deaths,
                    "total_deaths": len(death_times),
                    "early_death_pct": round(early_pct, 3),
                    "median_death_time_s": round(median_ms / 1000, 1),
                },
                root_cause_tag="over_peeking",
            )
        )

    return findings


# ---------------------------------------------------------------------------
# Analyzer 5 — Traded deaths
# ---------------------------------------------------------------------------


def analyze_traded_deaths(match: Match, puuid: str) -> list[Finding]:
    """How often are your deaths traded by a teammate?"""
    my_team = _team_of(match, puuid)
    if my_team is None:
        return []

    mates = _teammates(match, puuid, my_team)
    total_deaths = 0
    traded = 0

    for rnd in match.rounds:
        deaths = _my_deaths_in_round(rnd, puuid)
        if not deaths:
            continue
        total_deaths += len(deaths)
        kills_ordered = sorted(rnd.kills, key=lambda k: k.time_in_round_ms or 0)
        for death in deaths:
            if death.time_in_round_ms is None:
                continue
            killer = death.killer_puuid
            for k in kills_ordered:
                if k.killer_puuid not in mates or k.victim_puuid != killer:
                    continue
                if k.time_in_round_ms is None:
                    continue
                delta = k.time_in_round_ms - death.time_in_round_ms
                if 0 <= delta <= _TRADE_WINDOW_MS:
                    traded += 1
                    break

    if total_deaths < 4:
        return []

    trade_rate = traded / total_deaths
    findings: list[Finding] = []

    if trade_rate < 0.20:
        findings.append(
            Finding(
                severity="warning",
                category="duels",
                headline=f"Deaths traded only {trade_rate:.0%} of the time ({traded}/{total_deaths})",
                detail=(
                    f"Only {trade_rate:.0%} of your {total_deaths} deaths were traded by a teammate "
                    "within 5 seconds. Low trade rates suggest dying in isolated positions, far from "
                    "teammate support, or pushing ahead of your team. Try to take duels within "
                    "trading distance of at least one teammate."
                ),
                evidence={
                    "traded_deaths": traded,
                    "total_deaths": total_deaths,
                    "trade_rate": round(trade_rate, 3),
                },
                root_cause_tag="solo_hold",
            )
        )
    elif trade_rate >= 0.55:
        findings.append(
            Finding(
                severity="positive",
                category="duels",
                headline=f"Well-traded deaths — {trade_rate:.0%} traded ({traded}/{total_deaths})",
                detail=(
                    f"{trade_rate:.0%} of your deaths were traded quickly by teammates. "
                    "You're dying in good positions that your team can capitalise on."
                ),
                evidence={
                    "traded_deaths": traded,
                    "total_deaths": total_deaths,
                    "trade_rate": round(trade_rate, 3),
                },
                root_cause_tag="trade_success",
            )
        )

    return findings


# ---------------------------------------------------------------------------
# Analyzer 6 — Side split
# ---------------------------------------------------------------------------


def analyze_side_split(match: Match, puuid: str) -> list[Finding]:
    """Compare ATK vs DEF performance within this match.

    Side derivation uses round_analyzer._infer_attacker_team (scans for the
    first plant with a known planter_puuid) and the canonical get_side()
    helper which accounts for the half-swap at round 12 AND overtime
    alternation.  Falls back to the round_number < 12 heuristic only when
    no plant has a planter_puuid (pre-migration matches), and even then
    only if my_team starts on attack — which we can't know without a plant,
    so we skip side analysis entirely rather than mis-label half the rounds.
    """
    from valocoach.stats.round_analyzer import _infer_attacker_team, get_side

    mp = _get_mp(match, puuid)
    if mp is None:
        return []

    my_team = mp.team
    team_map = {p.puuid: p.team for p in match.players if p.puuid is not None}
    attacker_team = _infer_attacker_team(match, team_map)
    if attacker_team is None:
        # No plant data — we cannot determine which team started on attack.
        # Returning [] is safer than guessing my_team starts on attack
        # (the previous heuristic) which silently mis-labelled half of matches
        # where the player started on defense.
        return []

    atk_rounds_won = 0
    atk_rounds_played = 0
    def_rounds_won = 0
    def_rounds_played = 0

    for rnd in match.rounds:
        rp = _round_player(rnd, puuid)
        if rp is None:
            continue
        won = _round_won(rnd, my_team)
        side = get_side(rnd.round_number, attacker_team)
        # The player is on attack when their team matches the attacking team
        # for this round.  Compare lowercased to defend against case drift.
        on_attack = (side == "attack") == (my_team.lower() == attacker_team.lower())
        if on_attack:
            atk_rounds_played += 1
            if won:
                atk_rounds_won += 1
        else:
            def_rounds_played += 1
            if won:
                def_rounds_won += 1

    if atk_rounds_played < 3 or def_rounds_played < 3:
        return []

    atk_wr = atk_rounds_won / atk_rounds_played
    def_wr = def_rounds_won / def_rounds_played
    diff = abs(atk_wr - def_wr)

    if diff < 0.20:
        return []

    weak_side = "attack" if atk_wr < def_wr else "defense"
    strong_side = "defense" if weak_side == "attack" else "attack"
    weak_wr = atk_wr if weak_side == "attack" else def_wr
    strong_wr = def_wr if weak_side == "attack" else atk_wr

    sev: Severity = "critical" if diff >= 0.40 else "warning"
    return [
        Finding(
            severity=sev,
            category="positioning",
            headline=f"Strong {strong_side} ({strong_wr:.0%} WR) but weak {weak_side} ({weak_wr:.0%} WR)",
            detail=(
                f"Your {strong_side} win rate ({strong_wr:.0%}) significantly outperformed "
                f"your {weak_side} ({weak_wr:.0%}) this match. "
                f"{'Attack' if weak_side == 'attack' else 'Defense'} weaknesses often come from "
                f"{'over-holding angles and not adapting setups' if weak_side == 'defense' else 'poor site execute timing or lack of util coordination'}."
            ),
            evidence={
                "attack_win_rate": round(atk_wr, 3),
                "defense_win_rate": round(def_wr, 3),
                "attack_rounds": atk_rounds_played,
                "defense_rounds": def_rounds_played,
                "weak_side": weak_side,
            },
            root_cause_tag="side_imbalance",
        )
    ]


# ---------------------------------------------------------------------------
# Analyzer 7 — Clutch moments
# ---------------------------------------------------------------------------


def analyze_clutch_moments(match: Match, puuid: str) -> list[Finding]:
    """Find 1vN situations and compute the win rate."""
    my_team = _team_of(match, puuid)
    if my_team is None:
        return []

    mates = _teammates(match, puuid, my_team)
    clutch_opps = 0
    clutch_wins = 0

    for rnd in match.rounds:
        kills_ordered = sorted(rnd.kills, key=lambda k: k.time_in_round_ms or 0)
        team_alive = 5
        enemies_alive = 5
        player_alive = True
        entered_clutch = False
        won = _round_won(rnd, my_team)

        for k in kills_ordered:
            victim = k.victim_puuid
            killer = k.killer_puuid
            if victim == puuid:
                player_alive = False
                team_alive -= 1
            elif victim in mates:
                team_alive -= 1
            else:
                if killer == puuid or killer in mates:
                    enemies_alive -= 1
                else:
                    continue

            if not entered_clutch and player_alive and team_alive == 1 and enemies_alive >= 1:
                entered_clutch = True
                clutch_opps += 1
                if won:
                    clutch_wins += 1

            if not player_alive:
                break

    if clutch_opps < 2:
        return []

    clutch_wr = clutch_wins / clutch_opps
    findings: list[Finding] = []

    if clutch_wr < 0.30 and clutch_opps >= 3:
        findings.append(
            Finding(
                severity="warning",
                category="clutch",
                headline=f"Low clutch win rate: {clutch_wins}/{clutch_opps} ({clutch_wr:.0%})",
                detail=(
                    f"You entered {clutch_opps} clutch scenarios (last alive vs 1+ enemies) "
                    f"and converted only {clutch_wins}. In clutch situations: slow down, "
                    "use sound cues, reposition before peeking, and make the enemy take risks."
                ),
                evidence={
                    "clutch_wins": clutch_wins,
                    "clutch_opportunities": clutch_opps,
                    "clutch_win_rate": round(clutch_wr, 3),
                },
                root_cause_tag="clutch_failure",
            )
        )
    elif clutch_wr >= 0.60 and clutch_opps >= 2:
        findings.append(
            Finding(
                severity="positive",
                category="clutch",
                headline=f"Strong clutch performer: {clutch_wins}/{clutch_opps} ({clutch_wr:.0%})",
                detail=(
                    f"You converted {clutch_wr:.0%} of your clutch opportunities. "
                    "This is a significant strength — stay calm when isolated."
                ),
                evidence={
                    "clutch_wins": clutch_wins,
                    "clutch_opportunities": clutch_opps,
                    "clutch_win_rate": round(clutch_wr, 3),
                },
                root_cause_tag="clutch_success",
            )
        )

    return findings


# ---------------------------------------------------------------------------
# Analyzer 8 — Death location clusters (spatial, graceful degradation)
# ---------------------------------------------------------------------------

_CLUSTER_RADIUS = 300  # coordinate units
_CLUSTER_MIN = 3  # minimum deaths to call a hotspot


def _euclidean(x1: int, y1: int, x2: int, y2: int) -> float:
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def analyze_death_locations(match: Match, puuid: str) -> list[Finding]:
    """Cluster victim coordinates to find death hotspots."""
    coords: list[tuple[int, int]] = []
    for rnd in match.rounds:
        for k in rnd.kills:
            if k.victim_puuid == puuid and k.victim_x is not None and k.victim_y is not None:
                coords.append((k.victim_x, k.victim_y))

    if len(coords) < _CLUSTER_MIN:
        return []

    # Naive density clustering: find any point with >= MIN-1 neighbours within radius
    hotspots: list[tuple[tuple[int, int], int]] = []
    for i, (x, y) in enumerate(coords):
        neighbours = sum(
            1
            for j, (x2, y2) in enumerate(coords)
            if i != j and _euclidean(x, y, x2, y2) <= _CLUSTER_RADIUS
        )
        if neighbours >= _CLUSTER_MIN - 1:
            # Check this isn't already covered by a previous hotspot
            already = any(_euclidean(x, y, hx, hy) <= _CLUSTER_RADIUS for (hx, hy), _ in hotspots)
            if not already:
                hotspots.append(((x, y), neighbours + 1))

    if not hotspots:
        return []

    best_center, best_count = max(hotspots, key=lambda h: h[1])
    return [
        Finding(
            severity="warning",
            category="positioning",
            headline=f"Died {best_count}× in the same map area",
            detail=(
                f"You died {best_count} times within a tight cluster on the map "
                f"(centred around coordinates {best_center[0]}, {best_center[1]}). "
                "Repeatedly dying at the same spot indicates a predictable angle or "
                "an over-held position. Mix up your approach timing or try a different entry."
            ),
            evidence={
                "cluster_deaths": best_count,
                "cluster_center": list(best_center),
                "total_deaths_with_coords": len(coords),
                "cluster_radius_units": _CLUSTER_RADIUS,
            },
            root_cause_tag="over_peeking",
            required_fields=["victim_x", "victim_y"],
        )
    ]


# ---------------------------------------------------------------------------
# Analyzer 9 — Engagement distance (spatial, graceful degradation)
# ---------------------------------------------------------------------------


def analyze_engagement_distances(match: Match, puuid: str) -> list[Finding]:
    """Compare median engagement distance in wins vs losses."""
    won_distances: list[float] = []
    lost_distances: list[float] = []

    for rnd in match.rounds:
        for k in rnd.kills:
            if k.engagement_distance is None:
                continue
            try:
                d = float(k.engagement_distance)
            except (ValueError, TypeError):
                continue
            if k.killer_puuid == puuid:
                won_distances.append(d)
            elif k.victim_puuid == puuid:
                lost_distances.append(d)

    if len(won_distances) < 3 or len(lost_distances) < 3:
        return []

    median_won = sorted(won_distances)[len(won_distances) // 2]
    median_lost = sorted(lost_distances)[len(lost_distances) // 2]
    ratio = median_won / max(median_lost, 1)

    findings: list[Finding] = []

    if ratio < 0.55:
        # Winning at long range, losing at close range
        findings.append(
            Finding(
                severity="warning",
                category="duels",
                headline=f"Losing close-range duels (won at {median_won:.0f}u, lost at {median_lost:.0f}u)",
                detail=(
                    f"You win duels at median distance {median_won:.0f} units but lose duels at "
                    f"{median_lost:.0f} units — a significantly closer range. This can mean enemies "
                    "are surprising you around corners. Crosshair placement and pre-aiming corners "
                    "improve close-range outcomes substantially."
                ),
                evidence={
                    "median_win_distance": round(median_won, 1),
                    "median_loss_distance": round(median_lost, 1),
                    "kills_with_distance": len(won_distances),
                    "deaths_with_distance": len(lost_distances),
                },
                root_cause_tag="crosshair_placement",
                required_fields=["engagement_distance"],
            )
        )
    elif ratio > 1.8:
        # Winning at close range, losing at long range
        findings.append(
            Finding(
                severity="warning",
                category="duels",
                headline=f"Losing long-range duels (won at {median_won:.0f}u, lost at {median_lost:.0f}u)",
                detail=(
                    "You win close-range fights but lose at longer distances. "
                    "Consider playing tighter angles or using cover to reduce enemy ADS time. "
                    "Op/Vandal players will punish wide long-range peeks."
                ),
                evidence={
                    "median_win_distance": round(median_won, 1),
                    "median_loss_distance": round(median_lost, 1),
                    "kills_with_distance": len(won_distances),
                    "deaths_with_distance": len(lost_distances),
                },
                root_cause_tag="crosshair_placement",
                required_fields=["engagement_distance"],
            )
        )

    return findings


# ---------------------------------------------------------------------------
# Analyzer 10 — Plant/defuse site distribution (E3)
# ---------------------------------------------------------------------------


def analyze_plant_defuse_sites(match: Match, puuid: str) -> list[Finding]:
    """E3 — plant and defuse contributions per site this match.

    Surfaces predictable planting patterns (always A) and surfaces
    whether the player is contributing to objective work.

    required_fields: ["planter_puuid"] — needs Phase A migration.
    """
    from collections import defaultdict

    plants_per_site: dict[str, int] = defaultdict(int)
    defuses = 0

    for rnd in match.rounds:
        planter = getattr(rnd, "planter_puuid", None)
        if planter == puuid and rnd.plant_site:
            plants_per_site[rnd.plant_site] += 1
        defuser = getattr(rnd, "defuser_puuid", None)
        if defuser == puuid:
            defuses += 1

    total_plants = sum(plants_per_site.values())
    if total_plants == 0 and defuses == 0:
        return []

    # Check for single-site over-concentration on attack (≥ 75% on one site)
    severity: Severity = "neutral"
    detail_lines = []
    if plants_per_site:
        site_str = ", ".join(f"{site}: {n}×" for site, n in sorted(plants_per_site.items()))
        detail_lines.append(f"Plant distribution: {site_str}.")
        top_site, top_count = max(plants_per_site.items(), key=lambda kv: kv[1])
        if total_plants >= 4 and top_count / total_plants >= 0.75:
            severity = "warning"
            detail_lines.append(
                f"Over-concentrating on {top_site} ({top_count}/{total_plants} plants) "
                "makes your attack pattern predictable."
            )
    if defuses:
        detail_lines.append(f"Defused {defuses} spike(s).")

    headline = f"Objective: {total_plants} plant(s)"
    if plants_per_site:
        top_site_str = max(plants_per_site, key=lambda s: plants_per_site[s])
        headline += f" (mostly {top_site_str})" if len(plants_per_site) > 1 else ""
    if defuses:
        headline += f", {defuses} defuse(s)"

    return [
        Finding(
            severity=severity,
            category="objective",
            headline=headline,
            detail=" ".join(detail_lines),
            evidence={
                "plants_total": total_plants,
                "plants_per_site": dict(plants_per_site),
                "defuses": defuses,
            },
            root_cause_tag="objective_work",
            required_fields=["planter_puuid"],
        )
    ]


# ---------------------------------------------------------------------------
# Analyzer 11 — MMR trend (E4, standalone — needs MMR data from separate query)
# ---------------------------------------------------------------------------


def analyze_mmr_trend(mmr_rows: list) -> list[Finding]:
    """E4 — detect significant MMR decline across recent tracked games.

    Args:
        mmr_rows: MMRHistory ORM rows, most-recent first (repo.get_mmr_history
                  returns them in that order). Pass the last 5-10 rows.

    Returns a Finding when either:
    - 3 or more consecutive losses (mmr_change < 0 in a row), or
    - total RR delta is <= -50 across the last 5 entries.

    Never raises — on any data issue returns [].
    """
    if not mmr_rows or len(mmr_rows) < 2:
        return []

    try:
        last5 = mmr_rows[:5]

        # Consecutive loss streak (from most recent)
        streak = 0
        for row in last5:
            if (row.mmr_change or 0) < 0:
                streak += 1
            else:
                break

        total_delta = sum(row.mmr_change or 0 for row in last5)

        if streak < 3 and total_delta > -50:
            return []

        severity: Severity = "critical" if total_delta <= -80 else "warning"
        current = f"{mmr_rows[0].tier_patched} {mmr_rows[0].rr}RR"
        detail = (
            f"Currently {current}. "
            f"{abs(total_delta)} RR lost across the last {len(last5)} tracked games"
            + (f" ({streak} consecutive losses)" if streak >= 3 else "")
            + ". Consider reviewing your agent/map pool or queuing after a break."
        )
        return [
            Finding(
                severity=severity,
                category="momentum",
                headline=f"RR decline: {total_delta:+d} across last {len(last5)} games",
                detail=detail,
                evidence={
                    "total_delta": total_delta,
                    "consecutive_losses": streak,
                    "current_rr": mmr_rows[0].rr,
                    "current_tier": mmr_rows[0].tier_patched,
                },
                root_cause_tag="mmr_decline",
            )
        ]
    except Exception:
        log.debug("analyze_mmr_trend failed", exc_info=True)
        return []


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

_ANALYZERS = [
    analyze_first_contact,
    analyze_eco_decisions,
    analyze_utility_efficiency,
    analyze_round_timing,
    analyze_traded_deaths,
    analyze_side_split,
    analyze_clutch_moments,
    analyze_death_locations,
    analyze_engagement_distances,
    analyze_plant_defuse_sites,
]


def run_analyzers(match: Match, puuid: str) -> list[Finding]:
    """Run all analyzers, skipping those whose required fields are absent."""
    all_findings: list[Finding] = []
    for fn in _ANALYZERS:
        try:
            results = fn(match, puuid)
            # Post-filter: drop findings whose required_fields are not present
            for f in results:
                if _can_run(match, f.required_fields):
                    all_findings.append(f)
        except Exception:
            # One analyzer's failure shouldn't break the command, but it should
            # be visible at debug log so we can diagnose silent drops.
            log.debug("post-game analyzer %s raised", fn.__name__, exc_info=True)
    return all_findings


# ---------------------------------------------------------------------------
# Selector
# ---------------------------------------------------------------------------


def select_top_findings(findings: list[Finding], n: int = 3) -> list[Finding]:
    """Collapse by root_cause_tag, keep highest severity per group, return top N."""
    by_tag: dict[str, Finding] = {}
    for f in findings:
        existing = by_tag.get(f.root_cause_tag)
        if existing is None or _SEVERITY_RANK[f.severity] > _SEVERITY_RANK.get(
            existing.severity, 0
        ):
            by_tag[f.root_cause_tag] = f
    collapsed = sorted(by_tag.values(), key=lambda f: _SEVERITY_RANK[f.severity], reverse=True)
    return collapsed[:n]


# ---------------------------------------------------------------------------
# Formatter
# ---------------------------------------------------------------------------


def format_findings_block(findings: list[Finding]) -> str:
    """Format findings as a structured block for LLM injection.

    Example output::

        POST-GAME FINDINGS (3 areas — use these as the basis for your analysis):

        [CRITICAL] entry_failure — Died first in 8/18 rounds (44%)
        Evidence: first_deaths=8, rounds=18, first_death_rate=0.444
        Died first in 44% of rounds. This typically means over-peeking...

        [WARNING] bad_economy — Force-bought while team saved in 3 rounds
        ...
    """
    if not findings:
        return "POST-GAME FINDINGS: No significant patterns detected."

    lines = [f"POST-GAME FINDINGS ({len(findings)} area(s) — base your analysis on these):"]
    lines.append("")
    for f in findings:
        sev_label = f.severity.upper()
        lines.append(f"[{sev_label}] {f.root_cause_tag} — {f.headline}")
        evidence_str = ", ".join(f"{k}={v}" for k, v in f.evidence.items())
        lines.append(f"Evidence: {evidence_str}")
        lines.append(f.detail)
        lines.append("")
    return "\n".join(lines).rstrip()


__all__ = [
    "Finding",
    "Severity",
    "analyze_mmr_trend",
    "analyze_plant_defuse_sites",
    "format_findings_block",
    "run_analyzers",
    "select_top_findings",
]

"""Round-level stats: KAST, clutches, trades, multi-kills.

This is the hardest code in the stats engine because the metrics it
produces can't be derived from match aggregates — they need a round-by-
round walk over kill events with timing, team mapping, and survivor
tracking. Every metric here is a per-round classification, then summed.

Input contract:
    analyze_rounds(matches, puuid) takes a sequence of fully-loaded Match
    rows — `match.rounds` must be populated, each `round.kills` must be
    populated, and `match.players` must be populated (so we can map
    puuid → team). Callers drive the preload via SQLAlchemy
    `selectinload`; this module does no I/O.

Output contract:
    Returns a `RoundAnalysis` with raw counts (safe to aggregate further)
    and convenience rate properties. Empty input returns an all-zeros
    analysis — the caller decides whether to render "no data" or hide
    unreliable metrics via SAMPLE_THRESHOLDS.

Algorithms at a glance:

    KAST per round (Kill/Assist/Survive/Trade)
        K: player appears as killer at least once
        A: player appears in any kill's assistant list
        S: player is never a victim that round
        T: player dies AND a teammate kills the killer within TRADE_WINDOW_MS

    Clutch per round
        Walk kills in timestamp order, track alive/dead per puuid.
        A "clutch entered" event fires the first time the player is the
        sole survivor on their team with at least one enemy alive. A
        "clutch won" requires that AND the round winner is the player's
        team. We ignore multi-stage clutches (1v3 → 1v1) — one clutch
        per round, won or lost.

    Trade per death
        For every death of the player, scan forward for a teammate kill
        of the killer within TRADE_WINDOW_MS. Symmetrically, every
        teammate death the player avenges within that window counts as
        "trade given" (coverage metric).

    Multi-kills
        Count rounds where the player got 2/3/4/5 kills. Highest tier per
        round only — a 3-kill round is one triple, not one double + one
        triple.

Missing data:
    `Kill.time_in_round_ms` is nullable. When a kill's timestamp is None
    we fall back to insertion order for sequence, and we exclude the
    kill from trade-window arithmetic (can't measure what we can't
    time). This is documented at each use site.

    Economy/econ-rating is NOT computed here. The schema does not
    persist round-level economy (`api_models.economy` is an unparsed
    dict on the API boundary), so there is no source-of-truth data
    for a credits-based rating. Adding it requires a schema migration
    and mapper change — out of scope for this module.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from valocoach.data.orm_models import Kill, Match, Round
from valocoach.stats.calculator import StatResult, _check_threshold
from valocoach.stats.constants import (
    FIRST_HALF_END,
    MAX_REGULATION_ROUNDS,
    SECOND_HALF_END,
    TRADE_WINDOW_MS,
)

# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RoundAnalysis:
    """Per-puuid round-level aggregates across one or more matches.

    All fields are raw counts so callers can re-aggregate (e.g. split by
    agent) without losing information to premature rate-math. Rates are
    exposed as read-only properties and use safe division — empty input
    returns 0.0 rather than raising.
    """

    # Denominators ------------------------------------------------------
    rounds: int
    """Total rounds across all matches (sum of len(match.rounds))."""
    deaths: int
    """Rounds the player was a victim in (== player deaths across rounds)."""
    teammate_deaths: int
    """Teammate deaths — denominator for the 'trades given' rate."""
    clutch_opportunities: int
    """Rounds where the player ended up sole survivor with ≥1 enemy alive."""

    # KAST components (each is a round count) ---------------------------
    rounds_with_kill: int
    rounds_with_assist: int
    rounds_survived: int
    rounds_traded_death: int
    rounds_kast: int
    """Rounds where K, A, S, or T was true — KAST's numerator."""

    # Clutch ------------------------------------------------------------
    clutches_won: int

    # Trades ------------------------------------------------------------
    traded_deaths: int
    """Own deaths that a teammate avenged within TRADE_WINDOW_MS."""
    trades_given: int
    """Teammate deaths the player avenged within TRADE_WINDOW_MS."""

    # Multi-kills (exclusive tiers — one per round at the highest tier) -
    double_kills: int
    triple_kills: int
    quadra_kills: int
    aces: int

    # Breakdown dict — default empty, declared last to keep ordering clean.
    clutch_breakdown: dict[int, tuple[int, int]] = field(default_factory=dict)
    """Opponents-remaining → (won, total). E.g. {3: (1, 2)} = 1/2 1v3s won."""

    # Attack / defense split -------------------------------------------
    # None means side data was unavailable for this match set (no plant
    # events with a planter_puuid — e.g. data synced before migration).
    attack_rounds: int | None = None
    attack_wins: int | None = None
    defense_rounds: int | None = None
    defense_wins: int | None = None

    # ---- Rates ----

    @property
    def kast_pct(self) -> float:
        """Fraction of rounds with Kill/Assist/Survive/Trade in [0.0, 1.0]."""
        return _safe_div(self.rounds_kast, self.rounds)

    @property
    def clutch_rate(self) -> float:
        """Clutches won ÷ clutches entered. 0.0 when no opportunities."""
        return _safe_div(self.clutches_won, self.clutch_opportunities)

    @property
    def trade_efficiency(self) -> float:
        """Your deaths that got traded back ÷ your deaths. In [0.0, 1.0]."""
        return _safe_div(self.traded_deaths, self.deaths)

    @property
    def trade_participation(self) -> float:
        """Teammate deaths you avenged ÷ teammate deaths. Coverage metric."""
        return _safe_div(self.trades_given, self.teammate_deaths)

    @property
    def attack_win_rate(self) -> float | None:
        """Attack win rate in [0.0, 1.0], or None if side data unavailable."""
        if self.attack_rounds is None or self.attack_wins is None:
            return None
        return _safe_div(self.attack_wins, self.attack_rounds)

    @property
    def defense_win_rate(self) -> float | None:
        """Defense win rate in [0.0, 1.0], or None if side data unavailable."""
        if self.defense_rounds is None or self.defense_wins is None:
            return None
        return _safe_div(self.defense_wins, self.defense_rounds)


def _safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


# ---------------------------------------------------------------------------
# Side assignment
# ---------------------------------------------------------------------------


def get_side(round_number: int, attacker_team: str) -> str:
    """Return "attack" or "defense" for this round given the known attacking team.

    Args:
        round_number:   0-indexed round number (as stored in Round.round_number).
        attacker_team:  The team that started on attack in the first half
                        ("Red" or "Blue"). Derive this per-match via
                        _infer_attacker_team() before calling here.

    Round structure:
        0..FIRST_HALF_END    — first half (attacker_team attacks)
        FIRST_HALF_END+1..SECOND_HALF_END — second half (sides swap)
        SECOND_HALF_END+1+  — overtime, alternating every round
    """
    attacker_norm = attacker_team.lower()
    if round_number <= FIRST_HALF_END:
        round_attacker = attacker_norm
    elif round_number <= SECOND_HALF_END:
        round_attacker = "blue" if attacker_norm == "red" else "red"
    else:
        ot_index = round_number - MAX_REGULATION_ROUNDS
        # OT starts with the SAME swap that happened at half, then flips each round.
        second_half_attacker = "blue" if attacker_norm == "red" else "red"
        round_attacker = second_half_attacker if ot_index % 2 == 0 else attacker_norm
    return "attack" if round_attacker == attacker_norm else "defense"


def _infer_attacker_team(match: Match, team_map: dict[str, str]) -> str | None:
    """Scan match rounds for the first plant event with a known planter puuid.

    Returns the planter's team ("Red" | "Blue") — that team was the attacker
    in the round the plant occurred, so we know which team started on attack
    in the first half.

    Returns None when no plant event has a planter_puuid (pre-migration data
    or a match with no successful plants), in which case side tracking is
    skipped for the match.
    """
    for rnd in match.rounds:
        puuid = getattr(rnd, "planter_puuid", None)
        if puuid and puuid in team_map:
            team = team_map[puuid]
            # The planter is always an attacker. Back-calculate the first-half
            # attacker based on which half this round was in.
            if rnd.round_number <= FIRST_HALF_END:
                return team  # planted in first half → team started attack
            elif rnd.round_number <= SECOND_HALF_END:
                return (
                    "Blue" if team == "Red" else "Red"
                )  # planted in 2nd half → team swapped to attack
            else:
                # OT is ambiguous — skip, try next round with a plant.
                continue
    return None


# ---------------------------------------------------------------------------
# Main analyzer
# ---------------------------------------------------------------------------


def analyze_rounds(matches: Iterable[Match], puuid: str) -> RoundAnalysis:
    """Aggregate round-level stats for `puuid` across `matches`.

    Matches must have rounds, kills, and players eagerly loaded — this
    function does no DB work. Matches where the puuid is not on either
    team are silently skipped (defensive, not expected in practice).
    """
    # Accumulators — one big bag; we build a RoundAnalysis at the end so
    # the dataclass stays frozen.
    acc = _Acc()

    for match in matches:
        team_map = _build_team_map(match)
        player_team = team_map.get(puuid)
        if player_team is None:
            continue  # puuid not in this match; skip rather than crash
        teammates = {p for p, t in team_map.items() if t == player_team and p != puuid}
        attacker_team = _infer_attacker_team(match, team_map)

        for rnd in match.rounds:
            _tally_round(rnd, puuid, player_team, teammates, acc, attacker_team=attacker_team)

    return acc.freeze()


# ---------------------------------------------------------------------------
# Per-round tally — the meat of the module
# ---------------------------------------------------------------------------


def _tally_round(
    rnd: Round,
    puuid: str,
    player_team: str,
    teammates: set[str],
    acc: _Acc,
    *,
    attacker_team: str | None,
) -> None:
    """Classify one round's events and update accumulators in place."""
    acc.rounds += 1
    won = rnd.winning_team.lower() == player_team.lower()

    # Side accounting — only when we know which team started on attack.
    if attacker_team is not None:
        side = get_side(rnd.round_number, attacker_team)
        on_attack = (side == "attack") == (player_team.lower() == attacker_team.lower())
        if on_attack:
            acc.attack_rounds = (acc.attack_rounds or 0) + 1
            if won:
                acc.attack_wins = (acc.attack_wins or 0) + 1
        else:
            acc.defense_rounds = (acc.defense_rounds or 0) + 1
            if won:
                acc.defense_wins = (acc.defense_wins or 0) + 1
    kills = _ordered_kills(rnd.kills)
    if not kills:
        # No combat events — treat as survive (KAST S), no clutch, no trades.
        acc.rounds_survived += 1
        acc.rounds_kast += 1
        return

    # Per-round scratch state ------------------------------------------------
    player_kill_count = 0
    has_assist = False
    player_death: Kill | None = None

    # Teammate-deaths denominator is counted here (per-round attribution).
    for k in kills:
        if k.victim_puuid in teammates:
            acc.teammate_deaths += 1

    # First pass: K / A / victim detection.
    for k in kills:
        if k.killer_puuid == puuid:
            player_kill_count += 1
        if puuid in _assistants(k):
            has_assist = True
        if k.victim_puuid == puuid and player_death is None:
            player_death = k  # first (and only) death

    survived = player_death is None

    # Multi-kill classification (exclusive tiers, highest only).
    if player_kill_count == 2:
        acc.double_kills += 1
    elif player_kill_count == 3:
        acc.triple_kills += 1
    elif player_kill_count == 4:
        acc.quadra_kills += 1
    elif player_kill_count >= 5:
        acc.aces += 1

    # ---- Trade classification on the player's death ----
    traded = False
    if player_death is not None:
        acc.deaths += 1
        traded = _was_death_traded(
            death=player_death,
            kills=kills,
            teammates=teammates,
        )
        if traded:
            acc.traded_deaths += 1

    # ---- Trades given (player avenges teammate deaths) ----
    # For each teammate death, scan forward for a kill by the player of
    # the killer within the trade window.
    for i, k in enumerate(kills):
        if k.victim_puuid not in teammates:
            continue
        killer = k.killer_puuid
        t0 = k.time_in_round_ms
        for j in range(i + 1, len(kills)):
            k2 = kills[j]
            if k2.killer_puuid != puuid or k2.victim_puuid != killer:
                continue
            if _within_trade_window(t0, k2.time_in_round_ms):
                acc.trades_given += 1
                break  # one trade per teammate death, even if multi-avenged

    # ---- KAST round classification ----
    if player_kill_count > 0:
        acc.rounds_with_kill += 1
    if has_assist:
        acc.rounds_with_assist += 1
    if survived:
        acc.rounds_survived += 1
    if traded:
        acc.rounds_traded_death += 1
    if player_kill_count > 0 or has_assist or survived or traded:
        acc.rounds_kast += 1

    # ---- Clutch detection ----
    _tally_clutch(
        kills=kills,
        puuid=puuid,
        teammates=teammates,
        round_won=won,
        acc=acc,
    )


# ---------------------------------------------------------------------------
# Clutch simulation
# ---------------------------------------------------------------------------


def _tally_clutch(
    kills: Sequence[Kill],
    puuid: str,
    teammates: set[str],
    round_won: bool,
    acc: _Acc,
) -> None:
    """Detect 1vN scenarios and whether the player clutched them.

    Start with 5v5 and walk kills in order, marking victims dead. The
    clutch fires the first time the player is alive, all teammates are
    dead, and ≥1 enemy remains. We record the enemy count at that moment
    — multi-stage clutches (e.g. 1v3 → 1v1) still count as one entered;
    the N-breakdown uses the count at entry.
    """
    # Team sizes start at 5 — Valorant is always 5v5 at round start.
    # We don't look up actual rosters because it doesn't matter; all we
    # need is alive counts for the player's team (including self) and
    # the enemy team, which shift down by 1 on each relevant kill.
    team_alive = 5  # includes the player
    enemies_alive = 5
    player_alive = True
    clutch_opponents: int | None = None  # enemy count at 1vN entry

    for k in kills:
        killer = k.killer_puuid
        victim = k.victim_puuid
        # Update alive counts for the relevant team.
        if victim == puuid:
            player_alive = False
            team_alive -= 1
        elif victim in teammates:
            team_alive -= 1
        else:
            # Victim is on the enemy team (or unknown — treat as enemy
            # only if killer is on our team; otherwise ignore to avoid
            # double-counting when puuid isn't in team_map on either side).
            if killer == puuid or killer in teammates:
                enemies_alive -= 1
            else:
                # Unknown victim AND unknown killer — can't attribute.
                # Shouldn't happen for properly-loaded matches; skip.
                continue

        # Check for clutch entry (only fire once per round).
        if (
            clutch_opponents is None
            and player_alive
            and team_alive == 1  # only the player left
            and enemies_alive >= 1
        ):
            clutch_opponents = enemies_alive

        # Early exit once the player dies — no further clutch state changes
        # on the player's side matter (they can't clutch from the grave).
        if not player_alive:
            break

    if clutch_opponents is None:
        return

    acc.clutch_opportunities += 1
    won_this_clutch = round_won and player_alive
    if won_this_clutch:
        acc.clutches_won += 1

    prev_won, prev_total = acc.clutch_breakdown.get(clutch_opponents, (0, 0))
    acc.clutch_breakdown[clutch_opponents] = (
        prev_won + (1 if won_this_clutch else 0),
        prev_total + 1,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_team_map(match: Match) -> dict[str, str]:
    """Map each match participant's puuid to their team ("Red" | "Blue").

    Participants with a NULL puuid (historical data gap — see
    OrmMatchPlayer.puuid) are skipped; they can't be mapped back to a
    specific player so any kill involving them falls through the clutch
    simulation's "unknown" branch.
    """
    return {mp.puuid: mp.team for mp in match.players if mp.puuid is not None}


def _ordered_kills(kills: Iterable[Kill]) -> list[Kill]:
    """Return kills sorted by time_in_round_ms, NULLs preserved in input order.

    Stable sort with a sentinel for None pushes undated kills to the end
    while keeping any dated kills strictly chronological.
    """
    kill_list = list(kills)
    return sorted(
        kill_list,
        key=lambda k: (k.time_in_round_ms is None, k.time_in_round_ms or 0),
    )


def _assistants(kill: Kill) -> list[str]:
    """Parse the assistants JSON blob, tolerating malformed rows.

    Returns [] for missing/invalid data rather than raising — one bad
    row shouldn't null out a 20-match analysis.
    """
    raw = kill.assistants_json or "[]"
    try:
        parsed = json.loads(raw)
    except (ValueError, TypeError):
        return []
    return [p for p in parsed if isinstance(p, str)]


def _was_death_traded(
    *,
    death: Kill,
    kills: Sequence[Kill],
    teammates: set[str],
) -> bool:
    """True if a teammate killed `death.killer_puuid` within TRADE_WINDOW_MS.

    If either timestamp is missing we cannot measure the window, so we
    return False — under-counting trades is safer than over-counting.
    """
    if death.time_in_round_ms is None:
        return False
    killer = death.killer_puuid
    for k in kills:
        if k.killer_puuid not in teammates or k.victim_puuid != killer:
            continue
        if _within_trade_window(death.time_in_round_ms, k.time_in_round_ms):
            return True
    return False


def _within_trade_window(t_death: int | None, t_revenge: int | None) -> bool:
    """True if `t_revenge` is after `t_death` and within TRADE_WINDOW_MS.

    Both timestamps must be non-None. Using 0 <= delta <= WINDOW is
    inclusive on both ends — a same-ms revenge counts (engine rounding)
    and an exactly-5000-ms revenge counts (boundary convention).
    """
    if t_death is None or t_revenge is None:
        return False
    delta = t_revenge - t_death
    return 0 <= delta <= TRADE_WINDOW_MS


# ---------------------------------------------------------------------------
# Internal accumulator
# ---------------------------------------------------------------------------


@dataclass
class _Acc:
    """Mutable scratch state used during aggregation; frozen into RoundAnalysis."""

    rounds: int = 0
    deaths: int = 0
    teammate_deaths: int = 0
    clutch_opportunities: int = 0

    rounds_with_kill: int = 0
    rounds_with_assist: int = 0
    rounds_survived: int = 0
    rounds_traded_death: int = 0
    rounds_kast: int = 0

    clutches_won: int = 0
    clutch_breakdown: dict[int, tuple[int, int]] = field(
        default_factory=lambda: defaultdict(lambda: (0, 0))
    )

    traded_deaths: int = 0
    trades_given: int = 0

    double_kills: int = 0
    triple_kills: int = 0
    quadra_kills: int = 0
    aces: int = 0

    attack_rounds: int | None = None
    attack_wins: int | None = None
    defense_rounds: int | None = None
    defense_wins: int | None = None

    def freeze(self) -> RoundAnalysis:
        return RoundAnalysis(
            rounds=self.rounds,
            deaths=self.deaths,
            teammate_deaths=self.teammate_deaths,
            clutch_opportunities=self.clutch_opportunities,
            rounds_with_kill=self.rounds_with_kill,
            rounds_with_assist=self.rounds_with_assist,
            rounds_survived=self.rounds_survived,
            rounds_traded_death=self.rounds_traded_death,
            rounds_kast=self.rounds_kast,
            clutches_won=self.clutches_won,
            clutch_breakdown=dict(self.clutch_breakdown),
            traded_deaths=self.traded_deaths,
            trades_given=self.trades_given,
            double_kills=self.double_kills,
            triple_kills=self.triple_kills,
            quadra_kills=self.quadra_kills,
            aces=self.aces,
            attack_rounds=self.attack_rounds,
            attack_wins=self.attack_wins,
            defense_rounds=self.defense_rounds,
            defense_wins=self.defense_wins,
        )


# ---------------------------------------------------------------------------
# Per-metric StatResult wrappers  (their design, our correct internals)
# ---------------------------------------------------------------------------
# Each function takes the already-computed RoundAnalysis (pure counts) and
# the match count from the aggregate layer, returning a StatResult that
# carries value + label + reliability + warning in one object.
#
# The presentation layer (CLI, coach context) calls these instead of
# manually re-deriving thresholds — one place to change if a threshold moves.


def kast_stat(analysis: RoundAnalysis, matches: int) -> StatResult:
    """KAST% as a StatResult — Kill/Assist/Survive/Trade per round."""
    rel, warn = _check_threshold("kast", matches, analysis.rounds)
    return StatResult(
        value=analysis.kast_pct * 100,
        label="KAST%",
        format="pct",
        matches_used=matches,
        rounds_used=analysis.rounds,
        is_reliable=rel,
        warning=warn,
    )


def clutch_stat(analysis: RoundAnalysis, matches: int) -> StatResult:
    """Clutch win% as a StatResult — clutches_won / clutch_opportunities.

    When the player faced zero clutch situations the value is 0.0 and
    ``is_reliable`` is False regardless of sample size, because the
    denominator is zero.
    """
    rel, warn = _check_threshold("clutch_rate", matches, analysis.rounds)
    no_opps = analysis.clutch_opportunities == 0
    return StatResult(
        value=analysis.clutch_rate * 100,
        label="Clutch%",
        format="pct",
        matches_used=matches,
        rounds_used=analysis.rounds,
        is_reliable=rel and not no_opps,
        warning=warn or ("⚠ no clutch situations in sample" if no_opps else None),
    )


def trade_efficiency_stat(analysis: RoundAnalysis, matches: int) -> StatResult:
    """Trade efficiency as a StatResult — own deaths that were avenged."""
    rel, warn = _check_threshold("trade_efficiency", matches, analysis.rounds)
    no_deaths = analysis.deaths == 0
    return StatResult(
        value=analysis.trade_efficiency * 100,
        label="Trade Eff%",
        format="pct",
        matches_used=matches,
        rounds_used=analysis.rounds,
        is_reliable=rel and not no_deaths,
        warning=warn or ("⚠ no deaths in sample" if no_deaths else None),
    )


def trade_participation_stat(analysis: RoundAnalysis, matches: int) -> StatResult:
    """Trade participation as a StatResult — teammate deaths the player avenged."""
    rel, warn = _check_threshold("trade_efficiency", matches, analysis.rounds)
    no_teammate_deaths = analysis.teammate_deaths == 0
    return StatResult(
        value=analysis.trade_participation * 100,
        label="Trade Part%",
        format="pct",
        matches_used=matches,
        rounds_used=analysis.rounds,
        is_reliable=rel and not no_teammate_deaths,
        warning=warn or ("⚠ no teammate deaths in sample" if no_teammate_deaths else None),
    )


def multi_kill_summary(analysis: RoundAnalysis) -> str:
    """Short multi-kill string for display, e.g. '2xAce, 5x3K, 12x2K'.

    Returns an empty string when there are no multi-kills so callers can
    do a simple ``if summary:`` guard without special-casing None.
    """
    parts: list[str] = []
    if analysis.aces:
        parts.append(f"{analysis.aces}xAce")
    if analysis.quadra_kills:
        parts.append(f"{analysis.quadra_kills}x4K")
    if analysis.triple_kills:
        parts.append(f"{analysis.triple_kills}x3K")
    if analysis.double_kills:
        parts.append(f"{analysis.double_kills}x2K")
    return ", ".join(parts)

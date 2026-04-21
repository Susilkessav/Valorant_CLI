"""Pure stats computation over MatchPlayer rows.

No I/O — callers fetch MatchPlayer objects (via valocoach.data.repository)
and pass them in. The same functions drive the `stats` and `profile`
CLIs and feed context into the coach system prompt.

Aggregation philosophy:
    ACS, ADR, HS% and the first-blood rates are all *per-round* quantities.
    We sum the numerator and denominator across matches and divide once —
    round-weighted — rather than taking a mean of per-match means. A 30-round
    match counts more than a 14-round stomp, which is the right behavior.

    Win rate is per-match (not per-round).

Shape:
    compute_player_stats(rows)      -> PlayerStats          (overall)
    compute_per_agent(rows)         -> list[AgentStats]     (sorted: most played first)
    compute_per_map(rows)           -> list[MapStats]       (sorted: most played first)

Zero-safety:
    Every ratio goes through _safe_div, which returns 0.0 on a zero
    denominator. Empty input returns a PlayerStats of all zeros —
    callers can render a "no data yet" panel instead of crashing.

KAST% is intentionally omitted: it requires kill-timeline reconstruction
and teammate-trade detection from the Kill table. Worth adding later as a
dedicated pass over rounds + kills; not a calculator.py concern.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

from valocoach.data.orm_models import MatchPlayer

# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PlayerStats:
    """Aggregate stats over a set of MatchPlayer rows.

    Rates (win_rate, hs_pct, fb_rate, fd_rate) are ratios in [0.0, 1.0] —
    multiply by 100 at the presentation layer. Keeps arithmetic on these
    values (e.g. weighting, averaging) free of unit-conversion bugs.
    """

    matches: int
    rounds: int

    # Outcomes
    wins: int
    losses: int
    win_rate: float  # 0.0-1.0

    # Per-round combat
    acs: float  # score / rounds
    adr: float  # damage_dealt / rounds

    # KDA
    kills: int
    deaths: int
    assists: int
    kd: float  # kills / deaths  (deaths==0 → 0.0)
    kda: float  # (kills + assists) / deaths

    # Accuracy
    headshots: int
    bodyshots: int
    legshots: int
    hs_pct: float  # headshots / (hs+bs+ls), 0.0-1.0

    # Impact
    first_bloods: int
    first_deaths: int
    fb_rate: float  # first_bloods / rounds
    fd_rate: float  # first_deaths / rounds
    fb_diff: int  # first_bloods - first_deaths

    # Objective
    plants: int
    defuses: int


@dataclass(frozen=True, slots=True)
class AgentStats:
    agent: str
    stats: PlayerStats


@dataclass(frozen=True, slots=True)
class MapStats:
    map_name: str
    stats: PlayerStats


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_div(num: float, den: float) -> float:
    """Division that returns 0.0 on a zero denominator.

    Used for every rate/ratio in this module — a player with 0 deaths
    (or 0 rounds, or 0 shots landed) should render as 0.0, not crash.
    """
    return num / den if den else 0.0


def _zero_stats() -> PlayerStats:
    """Empty-input stats — all zeros. Callers can render 'no data yet'."""
    return PlayerStats(
        matches=0,
        rounds=0,
        wins=0,
        losses=0,
        win_rate=0.0,
        acs=0.0,
        adr=0.0,
        kills=0,
        deaths=0,
        assists=0,
        kd=0.0,
        kda=0.0,
        headshots=0,
        bodyshots=0,
        legshots=0,
        hs_pct=0.0,
        first_bloods=0,
        first_deaths=0,
        fb_rate=0.0,
        fd_rate=0.0,
        fb_diff=0,
        plants=0,
        defuses=0,
    )


# ---------------------------------------------------------------------------
# Core aggregation
# ---------------------------------------------------------------------------


def compute_player_stats(match_players: Iterable[MatchPlayer]) -> PlayerStats:
    """Round-weighted aggregation across an iterable of MatchPlayer rows.

    Pass the rows for a single puuid — this function does not filter. If
    you hand it rows for multiple players it will happily aggregate them
    all into one blob (useful for "team average", meaningless otherwise).
    """
    rows = list(match_players)
    if not rows:
        return _zero_stats()

    matches = len(rows)
    rounds = sum(mp.rounds_played for mp in rows)

    wins = sum(1 for mp in rows if mp.won)
    losses = matches - wins

    score = sum(mp.score for mp in rows)
    damage = sum(mp.damage_dealt for mp in rows)

    kills = sum(mp.kills for mp in rows)
    deaths = sum(mp.deaths for mp in rows)
    assists = sum(mp.assists for mp in rows)

    headshots = sum(mp.headshots for mp in rows)
    bodyshots = sum(mp.bodyshots for mp in rows)
    legshots = sum(mp.legshots for mp in rows)
    total_shots = headshots + bodyshots + legshots

    first_bloods = sum(mp.first_bloods for mp in rows)
    first_deaths = sum(mp.first_deaths for mp in rows)

    plants = sum(mp.plants for mp in rows)
    defuses = sum(mp.defuses for mp in rows)

    return PlayerStats(
        matches=matches,
        rounds=rounds,
        wins=wins,
        losses=losses,
        win_rate=_safe_div(wins, matches),
        acs=_safe_div(score, rounds),
        adr=_safe_div(damage, rounds),
        kills=kills,
        deaths=deaths,
        assists=assists,
        kd=_safe_div(kills, deaths),
        kda=_safe_div(kills + assists, deaths),
        headshots=headshots,
        bodyshots=bodyshots,
        legshots=legshots,
        hs_pct=_safe_div(headshots, total_shots),
        first_bloods=first_bloods,
        first_deaths=first_deaths,
        fb_rate=_safe_div(first_bloods, rounds),
        fd_rate=_safe_div(first_deaths, rounds),
        fb_diff=first_bloods - first_deaths,
        plants=plants,
        defuses=defuses,
    )


# ---------------------------------------------------------------------------
# Groupings
# ---------------------------------------------------------------------------


def compute_per_agent(match_players: Iterable[MatchPlayer]) -> list[AgentStats]:
    """Per-agent breakdown, sorted by matches played (descending).

    Ties broken alphabetically so the ordering is stable across runs.
    """
    buckets: dict[str, list[MatchPlayer]] = defaultdict(list)
    for mp in match_players:
        buckets[mp.agent_name].append(mp)

    results = [
        AgentStats(agent=agent, stats=compute_player_stats(rows)) for agent, rows in buckets.items()
    ]
    results.sort(key=lambda a: (-a.stats.matches, a.agent))
    return results


def compute_per_map(match_players: Iterable[MatchPlayer]) -> list[MapStats]:
    """Per-map breakdown, sorted by matches played (descending).

    Reads mp.match.map_name — the Match relationship is loaded eagerly
    (lazy='selectin' on OrmMatchPlayer.match), so no extra query fires
    inside this loop when the rows come from repository.get_recent_matches.
    """
    buckets: dict[str, list[MatchPlayer]] = defaultdict(list)
    for mp in match_players:
        map_name = mp.match.map_name if mp.match is not None else "Unknown"
        buckets[map_name].append(mp)

    results = [
        MapStats(map_name=name, stats=compute_player_stats(rows)) for name, rows in buckets.items()
    ]
    results.sort(key=lambda m: (-m.stats.matches, m.map_name))
    return results

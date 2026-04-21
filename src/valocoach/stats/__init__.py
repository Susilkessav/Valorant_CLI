"""valocoach.stats — pure stats computation, no I/O.

Fetch MatchPlayer rows via valocoach.data.repository, hand them here.
The same result types feed the `stats` CLI, the `profile` CLI, and the
coach system-prompt context builder.

    from valocoach.data import get_recent_matches, session_scope
    from valocoach.stats import compute_player_stats, compute_per_agent

    async with session_scope() as session:
        rows = await get_recent_matches(session, puuid, limit=20)
    overall = compute_player_stats(rows)
    by_agent = compute_per_agent(rows)
"""

from __future__ import annotations

from valocoach.stats.calculator import (
    MIN_MATCHES_ACS,
    MIN_MATCHES_ADR,
    MIN_MATCHES_FB,
    MIN_MATCHES_HS,
    MIN_MATCHES_KD,
    MIN_MATCHES_WIN_RATE_SPLIT,
    AgentStats,
    MapStats,
    PlayerStats,
    compute_per_agent,
    compute_per_map,
    compute_player_stats,
    reliability_flags,
)

__all__ = [
    "MIN_MATCHES_ACS",
    "MIN_MATCHES_ADR",
    "MIN_MATCHES_FB",
    "MIN_MATCHES_HS",
    "MIN_MATCHES_KD",
    "MIN_MATCHES_WIN_RATE_SPLIT",
    "AgentStats",
    "MapStats",
    "PlayerStats",
    "compute_per_agent",
    "compute_per_map",
    "compute_player_stats",
    "reliability_flags",
]

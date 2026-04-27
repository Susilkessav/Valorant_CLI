"""valocoach.stats — pure stats computation, no I/O.

Fetch MatchPlayer rows via valocoach.data.repository, hand them here.
The same result types feed the `stats` CLI, the `profile` CLI, and the
coach system-prompt context builder.

    from valocoach.data import get_recent_matches, session_scope
    from valocoach.stats import compute_player_stats, compute_per_agent
    from valocoach.stats import apply_filters, split_by_result, compare_baseline

    async with session_scope() as session:
        rows = await get_recent_matches(session, puuid, limit=20)
    filtered = apply_filters(rows, period="30d", agent="Jett")
    overall = compute_player_stats(filtered)
    by_agent = compute_per_agent(filtered)
    wins, losses = split_by_result(filtered)
    comparison = compare_baseline(filtered)   # form anomalies vs. baseline
"""

from __future__ import annotations

from valocoach.stats.baseline import (
    Anomaly,
    BaselineComparison,
    compare_baseline,
    detect_anomalies,
)
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
    StatResult,
    _check_threshold,
    compute_per_agent,
    compute_per_map,
    compute_player_stats,
    reliability_flags,
)
from valocoach.stats.filters import (
    apply_filters,
    filter_by_agent,
    filter_by_map,
    filter_by_period,
    filter_by_queue,
    filter_by_result,
    filter_by_tier_range,
    parse_period,
    recent_form,
    split_by_result,
)

__all__ = [
    # calculator
    "MIN_MATCHES_ACS",
    "MIN_MATCHES_ADR",
    "MIN_MATCHES_FB",
    "MIN_MATCHES_HS",
    "MIN_MATCHES_KD",
    "MIN_MATCHES_WIN_RATE_SPLIT",
    "AgentStats",
    # baseline
    "Anomaly",
    "BaselineComparison",
    "MapStats",
    "PlayerStats",
    "StatResult",
    "_check_threshold",
    # filters
    "apply_filters",
    "compare_baseline",
    "compute_per_agent",
    "compute_per_map",
    "compute_player_stats",
    "detect_anomalies",
    "filter_by_agent",
    "filter_by_map",
    "filter_by_period",
    "filter_by_queue",
    "filter_by_result",
    "filter_by_tier_range",
    "parse_period",
    "recent_form",
    "reliability_flags",
    "split_by_result",
]

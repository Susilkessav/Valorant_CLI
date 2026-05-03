"""Pure filter predicates and partition helpers over MatchPlayer rows.

All functions are pure — they receive a list[MatchPlayer] and return a
filtered or partitioned list. No I/O, no CLI dependency, no side effects.
The same predicates feed:
  - ``valocoach stats``  (--agent, --map, --period, --result flags)
  - coach context builder
  - profile deep-dive commands

Filter vs. Partition:
    filter_*()     reduce the row set. Same schema, fewer rows.
    split_*()      return both halves as separate lists (used by split tables).
    apply_filters() convenience combinator for the common multi-flag case.

Period parsing:
    parse_period()  converts the CLI string ('30d', 'all') to an ISO8601
    cutoff timestamp. Raises ValueError (not typer.BadParameter) — the CLI
    layer wraps that at the boundary so this module stays dependency-free.

Ordering note:
    ISO8601 UTC timestamps sort lexicographically == chronologically, so
    all started_at comparisons use plain string ordering — no datetime parsing.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from valocoach.data.orm_models import MatchPlayer

# ---------------------------------------------------------------------------
# Period parsing
# ---------------------------------------------------------------------------


def parse_period(period: str) -> str | None:
    """Convert a CLI period string to an ISO8601 cutoff timestamp.

    Args:
        period: ``'all'`` → no filter (returns None).
                ``'Nd'``  → ISO timestamp of *N* days ago (e.g. ``'30d'``).

    Returns:
        ISO8601 string (UTC) when a cutoff applies, or None for no cutoff.

    Raises:
        ValueError: on unrecognised format or non-positive N.

    ISO8601 strings sort correctly as plain strings, so callers can use
    ``mp.started_at >= cutoff`` without datetime parsing.
    """
    p = period.strip().lower()
    if p == "all":
        return None
    if not p.endswith("d") or not p[:-1].isdigit():
        raise ValueError(f"period must be 'Nd' (e.g. 7d, 30d) or 'all'; got {period!r}")
    days = int(p[:-1])
    if days <= 0:
        raise ValueError(f"period must be positive; got {period!r}")
    cutoff = datetime.now(UTC) - timedelta(days=days)
    return cutoff.isoformat()


# ---------------------------------------------------------------------------
# Individual filter predicates
# ---------------------------------------------------------------------------


def filter_by_period(
    rows: list[MatchPlayer],
    cutoff_iso: str | None,
) -> list[MatchPlayer]:
    """Keep rows where started_at >= cutoff_iso.

    ``None`` cutoff → passthrough (no filter applied).
    ISO8601 string comparison is safe without datetime parsing.
    """
    if cutoff_iso is None:
        return rows
    return [mp for mp in rows if mp.started_at >= cutoff_iso]


def filter_by_agent(
    rows: list[MatchPlayer],
    agent: str | None,
) -> list[MatchPlayer]:
    """Keep only rows where the player used *agent* (case-insensitive).

    ``None`` → passthrough.

    Comparison is case-insensitive — CLI users type "jett", the DB stores
    "Jett", both should match.
    """
    if agent is None:
        return rows
    agent_lc = agent.lower()
    return [mp for mp in rows if mp.agent_name.lower() == agent_lc]


def filter_by_map(
    rows: list[MatchPlayer],
    map_name: str | None,
) -> list[MatchPlayer]:
    """Keep only rows played on *map_name* (case-insensitive).

    ``None`` → passthrough. Rows without a linked Match are dropped when
    a map filter is active (can't verify the map they were played on).
    """
    if map_name is None:
        return rows
    map_lc = map_name.lower()
    return [mp for mp in rows if mp.match is not None and mp.match.map_name.lower() == map_lc]


def filter_by_result(
    rows: list[MatchPlayer],
    won: bool | None,
) -> list[MatchPlayer]:
    """Keep only wins (won=True) or losses (won=False).

    ``None`` → passthrough (keep both outcomes).
    """
    if won is None:
        return rows
    return [mp for mp in rows if mp.won is won]


def filter_by_queue(
    rows: list[MatchPlayer],
    queue_id: str | None,
) -> list[MatchPlayer]:
    """Keep only rows from *queue_id* matches (case-insensitive).

    ``None`` → passthrough.

    Common values: ``'competitive'``, ``'unrated'``, ``'spikerush'``,
    ``'deathmatch'``. Rows without a linked Match are dropped.
    """
    if queue_id is None:
        return rows
    q_lc = queue_id.lower()
    return [mp for mp in rows if mp.match is not None and mp.match.queue_id.lower() == q_lc]


def filter_by_tier_range(
    rows: list[MatchPlayer],
    *,
    min_tier: int | None,
    max_tier: int | None,
) -> list[MatchPlayer]:
    """Keep rows where competitive_tier is in [min_tier, max_tier].

    Rows with competitive_tier=None are excluded when either bound is set —
    they can't be verified as in-range, and including them would inflate
    a rank-gated sample with unverified data.

    Tier integers follow Riot's encoding: 0=Unranked, 3=Iron 1, …, 27=Radiant.
    ``None`` on either bound means "no lower/upper limit".
    """
    if min_tier is None and max_tier is None:
        return rows
    result = []
    for mp in rows:
        t = mp.competitive_tier
        if t is None:
            continue
        if min_tier is not None and t < min_tier:
            continue
        if max_tier is not None and t > max_tier:
            continue
        result.append(mp)
    return result


# ---------------------------------------------------------------------------
# Convenience combinator
# ---------------------------------------------------------------------------


def apply_filters(
    rows: list[MatchPlayer],
    *,
    period: str = "all",
    agent: str | None = None,
    map_name: str | None = None,
    won: bool | None = None,
    queue_id: str | None = None,
    min_tier: int | None = None,
    max_tier: int | None = None,
) -> list[MatchPlayer]:
    """Apply all filters in one call.

    Ordering: period → queue → agent → map → result → tier.
    Period and queue narrow the most aggressively (time-bound and mode-bound),
    so they run first to reduce work in subsequent predicates.

    Args:
        period:    ``'Nd'`` or ``'all'`` — passed to :func:`parse_period`.
        agent:     Case-insensitive agent name, or ``None`` for no filter.
        map_name:  Case-insensitive map name, or ``None`` for no filter.
        won:       ``True`` = wins only, ``False`` = losses only, ``None`` = both.
        queue_id:  Queue string (e.g. ``'competitive'``), or ``None``.
        min_tier:  Inclusive lower bound on competitive_tier, or ``None``.
        max_tier:  Inclusive upper bound on competitive_tier, or ``None``.

    Raises:
        ValueError: when *period* is malformed.
    """
    cutoff = parse_period(period)
    result = filter_by_period(rows, cutoff)
    result = filter_by_queue(result, queue_id)
    result = filter_by_agent(result, agent)
    result = filter_by_map(result, map_name)
    result = filter_by_result(result, won)
    result = filter_by_tier_range(result, min_tier=min_tier, max_tier=max_tier)
    return result


# ---------------------------------------------------------------------------
# Partition helpers  (return both halves — used for split tables)
# ---------------------------------------------------------------------------


def split_by_result(
    rows: list[MatchPlayer],
) -> tuple[list[MatchPlayer], list[MatchPlayer]]:
    """Partition into (wins, losses).

    Neither half is re-sorted — the original ordering (newest-first from
    the repository) is preserved. Callers pass each half to
    :func:`valocoach.stats.compute_player_stats` for independent aggregation.
    """
    wins = [mp for mp in rows if mp.won]
    losses = [mp for mp in rows if not mp.won]
    return wins, losses


def recent_form(rows: list[MatchPlayer], n: int) -> list[MatchPlayer]:
    """The *n* most-recent rows, sorted newest-first by started_at.

    ISO8601 timestamps sort lexicographically == chronologically, so plain
    string comparison is correct without datetime parsing.

    When len(rows) < n, all rows are returned — no truncation warning needed;
    the caller can check ``len(result)`` to detect a thin window.
    """
    if n <= 0:
        raise ValueError(f"n must be positive; got {n!r}")
    sorted_rows = sorted(rows, key=lambda mp: mp.started_at, reverse=True)
    return sorted_rows[:n]

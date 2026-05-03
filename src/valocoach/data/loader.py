"""Data loader — bridge between the DB and the stats calculator.

The calculator (valocoach.stats.calculator) is pure math: it takes
list[MatchPlayer] and returns PlayerStats. The repository
(valocoach.data.repository) is pure DB: it takes a session and a puuid and
returns rows. Callers need both, which currently means repeating the same
setup at every call site::

    await ensure_db(...)
    async with session_scope() as session:
        player = await get_player_by_name(session, ...)
        rows   = await get_recent_matches(session, ...)

This module absorbs that boilerplate into one call and hands back a detached
``PlayerData`` bundle — rows already off the session, ready for the
calculator and filter functions.

Two entry points:
    load_player_data(settings, ...)        — sync, for CLI commands.
    load_player_data_async(settings, ...)  — async, for callers already
                                             inside an event loop
                                             (e.g. context.py).

Both return ``PlayerData | None``.  ``None`` means "no data available" — the
caller should degrade gracefully (prompt the user to run ``valocoach sync``,
skip context personalisation, etc.).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Final

from valocoach.core.config import Settings
from valocoach.data.database import ensure_db, session_scope
from valocoach.data.orm_models import Match, MatchPlayer, Player
from valocoach.data.repository import (
    DEFAULT_QUEUE,
    get_player_by_name,
    get_recent_matches,
    get_recent_matches_full,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_LOAD_LIMIT: Final[int] = 200
"""Default match window.

200 rows ≈ 3 months of heavy ranked play. In-Python filters (period, agent,
map) narrow this down after the fetch, so fetching wide is cheap — one DB
round-trip instead of one per filter combination.
"""

# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PlayerData:
    """Player identity + match rows, detached from the DB session.

    Safe to hold across async boundaries — the session is already closed
    before callers receive this object, so there is no lazy-load risk.

    Attributes:
        player:        Identity and current rank snapshot from the
                       ``players`` table.
        rows:          ``MatchPlayer`` rows for the tracked player, newest
                       first.  May be empty when the player exists in the DB
                       but no matches have been synced yet.
        full_matches:  Full ``Match`` trees with rounds and kills eagerly
                       loaded.  Populated only when ``include_rounds=True``
                       is passed to the loader; an empty list otherwise.
                       Required for round-level analysis (KAST, clutch,
                       trade efficiency via ``valocoach.stats.round_analyzer``).
    """

    player: Player
    rows: list[MatchPlayer]
    full_matches: list[Match]


# ---------------------------------------------------------------------------
# Async worker
# ---------------------------------------------------------------------------


async def load_player_data_async(
    settings: Settings,
    *,
    limit: int = DEFAULT_LOAD_LIMIT,
    queue_id: str | None = DEFAULT_QUEUE,
    include_rounds: bool = False,
    name: str | None = None,
    tag: str | None = None,
) -> PlayerData | None:
    """Fetch player identity + match rows from the local DB.

    Async variant — use this when the call site is already inside an event
    loop (for example, inside ``_build_stats_context_async`` in
    ``valocoach.coach.context``).  For CLI entry points use the sync
    :func:`load_player_data` instead.

    Args:
        settings:       App configuration.  Provides ``data_dir`` and the
                        default ``riot_name`` / ``riot_tag`` identity.
        limit:          Maximum number of ``MatchPlayer`` rows to fetch.
                        In-Python filters apply afterwards, so fetching wide
                        (the default 200) is preferred over multiple narrow
                        DB queries.
        queue_id:       Queue filter forwarded to
                        :func:`~valocoach.data.repository.get_recent_matches`.
                        Defaults to ``"competitive"``; pass ``None`` to
                        include all game modes.
        include_rounds: When ``True``, also fire the heavier
                        :func:`~valocoach.data.repository.get_recent_matches_full`
                        query to populate ``PlayerData.full_matches`` with
                        rounds and kills.  Only enable this when the caller
                        needs round-level analysis.
        name:           Override for the Riot display name.  When supplied,
                        takes precedence over ``settings.riot_name``.  The
                        ``profile`` command uses this to look up arbitrary
                        players (not just the configured default).
        tag:            Override for the Riot tag.  Must be given together
                        with ``name``; ignored otherwise.

    Returns:
        :class:`PlayerData` when the player is found in the local DB.
        ``None`` when:

        - The resolved name or tag is empty / unset.
        - No ``Player`` row exists for the given name+tag (the player has
          never been synced — caller should prompt ``valocoach sync``).
    """
    resolved_name = name or settings.riot_name
    resolved_tag = tag or settings.riot_tag
    if not resolved_name or not resolved_tag:
        return None

    await ensure_db(settings.data_dir / "valocoach.db")

    async with session_scope() as session:
        player = await get_player_by_name(session, resolved_name, resolved_tag)
        if player is None:
            return None

        rows = await get_recent_matches(session, player.puuid, limit=limit, queue_id=queue_id)

        # Full-match fetch is a separate, heavier selectin query (rounds +
        # kills).  Only worth firing when the caller will actually use the
        # round data, and only when there are rows to annotate.
        full_matches: list[Match] = (
            await get_recent_matches_full(session, player.puuid, limit=limit, queue_id=queue_id)
            if include_rounds and rows
            else []
        )

    return PlayerData(player=player, rows=rows, full_matches=full_matches)


# ---------------------------------------------------------------------------
# Sync entry point
# ---------------------------------------------------------------------------


def load_player_data(
    settings: Settings,
    *,
    limit: int = DEFAULT_LOAD_LIMIT,
    queue_id: str | None = DEFAULT_QUEUE,
    include_rounds: bool = False,
    name: str | None = None,
    tag: str | None = None,
) -> PlayerData | None:
    """Fetch player identity + match rows from the local DB.

    Sync entry point — safe to call from CLI commands or any non-async
    context.  Internally wraps :func:`load_player_data_async` in
    ``asyncio.run()``.

    .. warning::
        Do **not** call this from inside an already-running event loop —
        use :func:`load_player_data_async` instead.  Calling
        ``asyncio.run()`` from within a running loop raises
        ``RuntimeError: This event loop is already running``.

    Args:
        settings:       App configuration.  Provides ``data_dir`` and
                        fallback identity.
        limit:          Maximum ``MatchPlayer`` rows to fetch (default 200).
        queue_id:       Queue filter (default ``"competitive"``; ``None``
                        for all modes).
        include_rounds: Fetch full ``Match`` trees with rounds + kills.
                        Triggers the heavier ``get_recent_matches_full``
                        query; only enable when round-level analysis is
                        needed.
        name:           Override for the Riot display name (optional).
        tag:            Override for the Riot tag (optional).

    Returns:
        :class:`PlayerData` on success, ``None`` when the resolved
        identity is missing or the player has never been synced.
    """
    return asyncio.run(
        load_player_data_async(
            settings,
            limit=limit,
            queue_id=queue_id,
            include_rounds=include_rounds,
            name=name,
            tag=tag,
        )
    )

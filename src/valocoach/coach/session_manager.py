"""Sync bridge between the interactive REPL and the async coaching DB layer.

The REPL (prompt_toolkit) runs in a synchronous context.  These wrappers each
do ``asyncio.run(ensure_db(...) + session_scope() + repo call)`` so slash
commands can read and write ``coaching_sessions`` / ``coaching_notes`` without
spreading asyncio boilerplate across ``interactive.py``.

All public functions degrade gracefully — they return ``None`` / ``[]`` /
``False`` on any DB or config error so the REPL never crashes on DB issues.
A ``REPLCoachState`` dataclass is the single mutable bag passed through the
REPL loop; other modules may import it for type annotations.

Public API
----------
    get_player_puuid(settings)                          -> str | None
    open_coaching_session(settings, puuid, ...)         -> int | None
    close_coaching_session(settings, session_id)        -> None
    add_coaching_note(settings, session_id, puuid, ...) -> int | None
    list_open_notes(settings, puuid, ...)               -> list[NoteInfo]
    resolve_coaching_note(settings, note_id)            -> bool
    list_coaching_sessions(settings, puuid, ...)        -> list[SessionInfo]
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Plain-data transfer objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MMRHistoryInfo:
    """Detached representation of one ``MMRHistory`` row."""

    tier_patched: str
    rr: int
    elo: int
    mmr_change: int | None
    recorded_at: str


@dataclass(frozen=True)
class LastMatchInfo:
    """Compact snapshot of the player's most recent match — detached from ORM.

    Computed fields (acs, hs_pct, adr) are pre-calculated by the loader so
    the formatter can be a pure function with no DB access.
    """

    match_id: str
    map_name: str
    agent: str
    won: bool
    own_score: int   # rounds won by player's team
    opp_score: int   # rounds won by opponent's team
    kills: int
    deaths: int
    assists: int
    acs: int         # average combat score per round
    hs_pct: float    # headshot percentage (0.0-100.0)
    adr: float       # average damage per round
    started_at: str  # ISO8601 UTC


@dataclass(frozen=True)
class NoteInfo:
    """Detached representation of a ``CoachingNote`` row."""

    id: int
    body: str
    category: str
    priority: int  # 1 (high) … 3 (low)
    created_at: str
    match_id: str | None = None


@dataclass(frozen=True)
class SessionInfo:
    """Detached representation of a ``CoachingSession`` row."""

    id: int
    title: str | None
    started_at: str
    ended_at: str | None
    focus_agent: str | None
    focus_map: str | None

    @property
    def is_open(self) -> bool:
        return self.ended_at is None


# ---------------------------------------------------------------------------
# REPL state
# ---------------------------------------------------------------------------


# Maximum open notes injected per coaching turn.  Notes arrive priority-DESC
# then age-ASC from the repo, so the most urgent items always surface first.
# Injecting more than five rarely adds value and eats into the token budget.
_NOTES_INJECTION_LIMIT: int = 5


def format_open_notes_context(notes: list[NoteInfo]) -> str | None:
    """Format *notes* as a compact 'COACHING FOCUS' block for the system prompt.

    Returns ``None`` when the list is empty so callers can guard with
    ``if context:``.

    Example output::

        COACHING FOCUS (3 open notes — address these when relevant):
        • [aim] Stop peeking wide on A ramp — use util first.
        • [positioning] Activate Jett's updraft before duelling.
        • [economy] Don't full-buy after losing pistol into eco.
    """
    if not notes:
        return None
    capped = notes[:_NOTES_INJECTION_LIMIT]
    n = len(capped)
    plural = "notes" if n != 1 else "note"
    lines = [f"COACHING FOCUS ({n} open {plural} — address these when relevant):"]
    for note in capped:
        lines.append(f"• [{note.category}] {note.body}")
    return "\n".join(lines)


def format_last_match_context(info: LastMatchInfo) -> str:
    """Format *info* as a compact one-liner for injection into the coaching prompt.

    Example output::

        LAST MATCH: Jett on Ascent · W 13-7 · 18/8/4 · ACS 225 · HS 28% · ADR 142

    The string is intentionally short (~100 chars) so it adds minimal token cost.
    """
    result = "W" if info.won else "L"
    return (
        f"LAST MATCH: {info.agent} on {info.map_name} · "
        f"{result} {info.own_score}-{info.opp_score} · "
        f"{info.kills}/{info.deaths}/{info.assists} · "
        f"ACS {info.acs} · HS {info.hs_pct:.0f}% · ADR {info.adr:.0f}"
    )


@dataclass
class REPLCoachState:
    """Mutable coaching-session state carried by the REPL for its lifetime.

    ``active`` is True only when *both* a puuid and a DB session id are known.
    The REPL creates one instance at startup and passes it to ``_handle_slash``
    so notes slash commands can read/write the session without global state.
    """

    puuid: str | None = None
    coaching_session_id: int | None = None

    @property
    def active(self) -> bool:
        """True when there is both a known player and an open DB session."""
        return self.puuid is not None and self.coaching_session_id is not None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _db_path(settings):
    """Return the SQLite DB path from settings.

    Raises ``TypeError`` early when ``settings.data_dir`` is not a real
    :class:`~pathlib.Path` or string — this prevents aiosqlite from creating
    files whose names are MagicMock repr strings when tests pass bare
    ``MagicMock()`` objects for settings without setting ``data_dir``.
    The TypeError propagates up to each sync wrapper's ``except Exception``
    guard and causes it to return ``None`` / ``[]`` safely.
    """
    from pathlib import Path

    d = settings.data_dir
    if not isinstance(d, (Path, str)):
        raise TypeError(f"settings.data_dir must be a Path or str, got {type(d).__name__!r}")
    return Path(d) / "valocoach.db"


# ---------------------------------------------------------------------------
# Public sync wrappers
# ---------------------------------------------------------------------------


def get_player_puuid(settings) -> str | None:
    """Look up the configured player's puuid from the local DB.

    Returns ``None`` when the player has never been synced (no row in
    ``players`` table) or when any error occurs.
    """

    async def _run() -> str | None:
        from valocoach.data.database import ensure_db, session_scope
        from valocoach.data.repository import get_player_by_name

        await ensure_db(_db_path(settings))
        async with session_scope() as db:
            player = await get_player_by_name(db, settings.riot_name, settings.riot_tag)
            return player.puuid if player else None

    try:
        return asyncio.run(_run())
    except Exception as exc:
        logger.debug("get_player_puuid failed: %s", exc)
        return None


def open_coaching_session(
    settings,
    puuid: str,
    *,
    title: str | None = None,
    focus_agent: str | None = None,
    focus_map: str | None = None,
) -> int | None:
    """Create a new coaching session for *puuid*.  Returns its integer id."""

    async def _run() -> int | None:
        from valocoach.data.database import ensure_db, session_scope
        from valocoach.data.repository import create_coaching_session

        await ensure_db(_db_path(settings))
        async with session_scope() as db:
            cs = await create_coaching_session(
                db,
                puuid,
                title=title,
                focus_agent=focus_agent,
                focus_map=focus_map,
            )
            return cs.id

    try:
        return asyncio.run(_run())
    except Exception as exc:
        logger.debug("open_coaching_session failed: %s", exc)
        return None


def close_coaching_session(settings, session_id: int) -> None:
    """End an open coaching session (set ``ended_at``)."""

    async def _run() -> None:
        from valocoach.data.database import ensure_db, session_scope
        from valocoach.data.repository import end_coaching_session

        await ensure_db(_db_path(settings))
        async with session_scope() as db:
            await end_coaching_session(db, session_id)

    try:
        asyncio.run(_run())
    except Exception as exc:
        logger.debug("close_coaching_session failed: %s", exc)


def add_coaching_note(
    settings,
    session_id: int,
    puuid: str,
    body: str,
    *,
    category: str = "general",
    priority: int = 2,
) -> int | None:
    """Append a note to a coaching session.  Returns the new note's id."""

    async def _run() -> int | None:
        from valocoach.data.database import ensure_db, session_scope
        from valocoach.data.repository import add_coaching_note as _add

        await ensure_db(_db_path(settings))
        async with session_scope() as db:
            note = await _add(
                db,
                session_id,
                body,
                puuid=puuid,
                category=category,
                priority=priority,
            )
            return note.id

    try:
        return asyncio.run(_run())
    except Exception as exc:
        logger.debug("add_coaching_note failed: %s", exc)
        return None


def list_open_notes(
    settings,
    puuid: str,
    *,
    limit: int = 20,
) -> list[NoteInfo]:
    """Return open (unresolved) notes for *puuid* as :class:`NoteInfo` objects."""

    async def _run() -> list[NoteInfo]:
        from valocoach.data.database import ensure_db, session_scope
        from valocoach.data.repository import get_open_notes

        await ensure_db(_db_path(settings))
        async with session_scope() as db:
            rows = await get_open_notes(db, puuid, resolved=False, limit=limit)
            return [
                NoteInfo(
                    id=n.id,
                    body=n.body,
                    category=n.category,
                    priority=n.priority,
                    created_at=n.created_at,
                    match_id=n.match_id,
                )
                for n in rows
            ]

    try:
        return asyncio.run(_run())
    except Exception as exc:
        logger.debug("list_open_notes failed: %s", exc)
        return []


def resolve_coaching_note(settings, note_id: int) -> bool:
    """Mark note *note_id* as resolved.  Returns ``True`` on success."""

    async def _run() -> bool:
        from valocoach.data.database import ensure_db, session_scope
        from valocoach.data.repository import resolve_note

        await ensure_db(_db_path(settings))
        async with session_scope() as db:
            note = await resolve_note(db, note_id)
            return note is not None

    try:
        return asyncio.run(_run())
    except Exception as exc:
        logger.debug("resolve_coaching_note failed: %s", exc)
        return False


def get_or_open_coaching_session(
    settings,
    puuid: str,
) -> int | None:
    """Return the id of the current open coaching session, creating one if needed.

    Checks for an existing open session first (ended_at IS NULL) — re-uses it
    so that notes added from the CLI land in the same session that the REPL may
    have started.  Creates a fresh session only when none is open.

    Returns ``None`` when the operation fails (DB unavailable, etc.).
    """

    async def _run() -> int | None:
        from valocoach.data.database import ensure_db, session_scope
        from valocoach.data.repository import create_coaching_session, get_open_coaching_session

        await ensure_db(_db_path(settings))
        async with session_scope() as db:
            existing = await get_open_coaching_session(db, puuid)
            if existing is not None:
                return existing.id
            cs = await create_coaching_session(db, puuid)
            return cs.id

    try:
        return asyncio.run(_run())
    except Exception as exc:
        logger.debug("get_or_open_coaching_session failed: %s", exc)
        return None


def get_mmr_trend(
    settings,
    puuid: str,
    *,
    limit: int = 20,
) -> list[MMRHistoryInfo]:
    """Return up to *limit* rank snapshots newest-first as :class:`MMRHistoryInfo` objects.

    Returns an empty list when no MMR history exists (player never synced or
    the DB is inaccessible).
    """

    async def _run() -> list[MMRHistoryInfo]:
        from valocoach.data.database import ensure_db, session_scope
        from valocoach.data.repository import get_mmr_history

        await ensure_db(_db_path(settings))
        async with session_scope() as db:
            rows = await get_mmr_history(db, puuid, limit=limit)
            return [
                MMRHistoryInfo(
                    tier_patched=r.tier_patched,
                    rr=r.rr,
                    elo=r.elo,
                    mmr_change=r.mmr_change,
                    recorded_at=r.recorded_at,
                )
                for r in rows
            ]

    try:
        return asyncio.run(_run())
    except Exception as exc:
        logger.debug("get_mmr_trend failed: %s", exc)
        return []


def list_coaching_sessions(
    settings,
    puuid: str,
    *,
    limit: int = 5,
) -> list[SessionInfo]:
    """Return recent coaching sessions as :class:`SessionInfo` objects."""

    async def _run() -> list[SessionInfo]:
        from valocoach.data.database import ensure_db, session_scope
        from valocoach.data.repository import get_coaching_sessions

        await ensure_db(_db_path(settings))
        async with session_scope() as db:
            rows = await get_coaching_sessions(db, puuid, limit=limit)
            return [
                SessionInfo(
                    id=s.id,
                    title=s.session_title,
                    started_at=s.started_at,
                    ended_at=s.ended_at,
                    focus_agent=s.focus_agent,
                    focus_map=s.focus_map,
                )
                for s in rows
            ]

    try:
        return asyncio.run(_run())
    except Exception as exc:
        logger.debug("list_coaching_sessions failed: %s", exc)
        return []


def get_last_match(settings) -> LastMatchInfo | None:
    """Return the player's most recent competitive match as a :class:`LastMatchInfo`.

    Looks up the player from ``settings.riot_name`` / ``settings.riot_tag``,
    fetches the single most recent competitive ``MatchPlayer`` row, and returns
    a frozen detached snapshot with pre-computed ACS / HS% / ADR fields.

    Returns ``None`` when:
      - The configured Riot name/tag is unset.
      - No ``Player`` row exists (player has never been synced).
      - The player has no stored matches.
      - Any DB or config error occurs.

    This function is non-fatal — callers must treat ``None`` as "no data"
    and degrade gracefully.
    """

    async def _run() -> LastMatchInfo | None:
        from valocoach.data.database import ensure_db, session_scope
        from valocoach.data.repository import get_player_by_name, get_recent_matches

        name = settings.riot_name
        tag = settings.riot_tag
        if not name or not tag:
            return None

        await ensure_db(_db_path(settings))
        async with session_scope() as db:
            player = await get_player_by_name(db, name, tag)
            if player is None:
                return None
            rows = await get_recent_matches(db, player.puuid, limit=1)
            if not rows:
                return None
            mp = rows[0]
            m = mp.match
            own_score = m.blue_score if mp.team == "Blue" else m.red_score
            opp_score = m.red_score if mp.team == "Blue" else m.blue_score
            total_shots = mp.headshots + mp.bodyshots + mp.legshots
            hs_pct = round(mp.headshots / max(total_shots, 1) * 100.0, 1)
            rounds = max(mp.rounds_played, 1)
            return LastMatchInfo(
                match_id=mp.match_id,
                map_name=m.map_name,
                agent=mp.agent_name,
                won=mp.won,
                own_score=own_score,
                opp_score=opp_score,
                kills=mp.kills,
                deaths=mp.deaths,
                assists=mp.assists,
                acs=round(mp.score / rounds),
                hs_pct=hs_pct,
                adr=round(mp.damage_dealt / rounds, 1),
                started_at=mp.started_at,
            )

    try:
        return asyncio.run(_run())
    except Exception as exc:
        logger.debug("get_last_match failed: %s", exc)
        return None


__all__ = [
    "LastMatchInfo",
    "MMRHistoryInfo",
    "NoteInfo",
    "REPLCoachState",
    "SessionInfo",
    "add_coaching_note",
    "close_coaching_session",
    "format_last_match_context",
    "format_open_notes_context",
    "get_last_match",
    "get_mmr_trend",
    "get_or_open_coaching_session",
    "get_player_puuid",
    "list_coaching_sessions",
    "list_open_notes",
    "open_coaching_session",
    "resolve_coaching_note",
]

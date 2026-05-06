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
    """Return the SQLite DB path from settings."""
    return settings.data_dir / "valocoach.db"


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


__all__ = [
    "NoteInfo",
    "SessionInfo",
    "REPLCoachState",
    "get_player_puuid",
    "open_coaching_session",
    "close_coaching_session",
    "add_coaching_note",
    "list_open_notes",
    "resolve_coaching_note",
    "list_coaching_sessions",
]

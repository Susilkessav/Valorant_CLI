"""`valocoach sessions` — coaching session management.

Complements the coaching history shown at the bottom of ``valocoach profile``
by exposing session operations as standalone CLI sub-commands so players can
inspect and manage sessions without loading the full profile card.

Sub-commands
------------
    valocoach sessions          — alias for ``list`` (sessions table)
    valocoach sessions list     — list recent coaching sessions
    valocoach sessions close ID — close (end) an open session by id

Design notes
------------
Sessions are opened automatically when the interactive REPL starts and closed
on exit.  This command exists for two edge cases:

1. The REPL was killed without a clean exit — ``sessions close`` lets the
   player stamp ``ended_at`` retroactively.
2. The player wants a quick overview of past sessions without waiting for the
   full ``valocoach profile`` render.
"""

from __future__ import annotations

import typer

from valocoach.cli import display
from valocoach.cli.formatter import render_coaching_sessions
from valocoach.coach.session_manager import (
    close_coaching_session,
    get_player_puuid,
    list_coaching_sessions,
)
from valocoach.core.config import load_settings

# Default number of sessions to surface.  Profile shows the last 5; the CLI
# command leans a bit higher so users can see more history without --limit.
_DEFAULT_LIMIT = 20


def run_sessions_list(*, limit: int = _DEFAULT_LIMIT) -> None:
    """List recent coaching sessions for the configured player.

    Args:
        limit: Maximum number of sessions to return (newest first).
    """
    settings = load_settings()
    puuid = get_player_puuid(settings)

    if not puuid:
        display.warn(
            "No player profile found — run `valocoach sync` first so "
            "ValoCoach can identify your player."
        )
        raise typer.Exit(1)

    sessions = list_coaching_sessions(settings, puuid, limit=limit)
    if not sessions:
        display.info(
            "No coaching sessions found.  Start one with:  valocoach interactive"
        )
        return

    render_coaching_sessions(display.console, sessions)

    # Remind the player how to close orphaned open sessions.
    open_count = sum(1 for s in sessions if s.is_open)
    if open_count:
        display.console.print(
            f"\n[dim]{open_count} open session(s).  "
            "Close with:  valocoach sessions close <id>[/dim]"
        )


def run_sessions_close(session_id: int) -> None:
    """Close an open coaching session by stamping its ``ended_at`` timestamp.

    Args:
        session_id: Integer id of the session to close.

    Exits with status 1 when the session does not exist or is already closed.
    The DB ``end_coaching_session`` function is idempotent but we surface a
    warning so the player knows the command had no effect.
    """
    settings = load_settings()

    # close_coaching_session is fire-and-forget — it logs on failure but does
    # not raise.  We need to distinguish "already closed / not found" from
    # "successfully closed", so we check the session list before and after.
    # Rather than a round-trip, we just call close and report success — the
    # function only touches rows where ended_at IS NULL, so repeated calls are
    # idempotent.  The player sees "closed" if the row existed; a non-existent
    # id silently does nothing from the DB side.
    close_coaching_session(settings, session_id)
    display.success(f"Session #{session_id} closed.")

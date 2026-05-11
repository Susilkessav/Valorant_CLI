"""`valocoach sessions` — coaching session management."""

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

_DEFAULT_LIMIT = 20


def run_sessions_list(*, limit: int = _DEFAULT_LIMIT) -> None:
    settings = load_settings()
    puuid = get_player_puuid(settings)

    if not puuid:
        display.error_with_hint(
            "No player profile found.",
            "Run: valocoach sync",
        )
        raise typer.Exit(1)

    sessions = list_coaching_sessions(settings, puuid, limit=limit)
    if not sessions:
        display.info(
            "No coaching sessions found.  Start one with:  valocoach interactive"
        )
        return

    with display.command_frame("Coaching Sessions"):
        render_coaching_sessions(display.console, sessions)

        open_count = sum(1 for s in sessions if s.is_open)
        if open_count:
            display.console.print(
                f"\n[muted]{open_count} open session(s).  "
                "Close with:  valocoach sessions close <id>[/muted]"
            )


def run_sessions_close(session_id: int) -> None:
    settings = load_settings()
    close_coaching_session(settings, session_id)
    display.success(f"Session #{session_id} closed.")

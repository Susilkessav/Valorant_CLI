"""`valocoach notes` — standalone coaching-note management."""

from __future__ import annotations

import logging

import typer

log = logging.getLogger(__name__)

from valocoach.cli import display
from valocoach.cli.formatter import render_open_notes
from valocoach.coach.session_manager import (
    add_coaching_note,
    get_or_open_coaching_session,
    get_player_puuid,
    list_open_notes,
    resolve_coaching_note,
)
from valocoach.core.config import load_settings

_INTENT_TO_CATEGORY: dict[str, str] = {
    "clutch": "tactical",
    "post_plant": "tactical",
    "retake": "tactical",
    "tactical": "tactical",
    "economy": "economy",
    "agent_info": "agent",
    "meta": "meta",
    "stat_analysis": "general",
    "general": "general",
}


def _infer_category(text: str) -> str:
    try:
        from valocoach.coach.intent import classify_intent
        from valocoach.core.parser import parse_situation

        parsed = parse_situation(text)
        intent = classify_intent(parsed, text)
        return _INTENT_TO_CATEGORY.get(intent, "general")
    except Exception:
        log.debug("intent classification failed, defaulting to general", exc_info=True)
        return "general"


def run_notes_list() -> None:
    settings = load_settings()
    puuid = get_player_puuid(settings)

    if not puuid:
        display.error_with_hint(
            "No player profile found.",
            "Run: valocoach sync",
        )
        raise typer.Exit(1)

    notes = list_open_notes(settings, puuid)
    if not notes:
        display.info("No open coaching notes.  Add one with:  valocoach notes add <text>")
        return

    with display.command_frame("Coaching Notes"):
        render_open_notes(display.console, notes)


def run_notes_add(text: str, *, priority: int = 2) -> None:
    if not text.strip():
        display.warn("Note text cannot be empty.")
        raise typer.Exit(1)

    if priority not in (1, 2, 3):
        display.warn(f"Priority must be 1, 2, or 3 — got {priority}")
        raise typer.Exit(1)

    settings = load_settings()
    puuid = get_player_puuid(settings)

    if not puuid:
        display.error_with_hint(
            "No player profile found.",
            "Run: valocoach sync",
        )
        raise typer.Exit(1)

    session_id = get_or_open_coaching_session(settings, puuid)
    if session_id is None:
        display.warn("Couldn't open a coaching session — check DB access.")
        raise typer.Exit(1)

    category = _infer_category(text)
    note_id = add_coaching_note(
        settings,
        session_id,
        puuid,
        text,
        category=category,
        priority=priority,
    )

    if note_id is None:
        display.warn("Couldn't save note — check logs.")
        raise typer.Exit(1)

    display.success(f"Note #{note_id} saved  [muted](category: {category})[/muted]")


def run_notes_resolve(note_id: int) -> None:
    settings = load_settings()

    ok = resolve_coaching_note(settings, note_id)
    if ok:
        display.success(f"Note #{note_id} marked as resolved.")
    else:
        display.warn(f"Note #{note_id} not found or already resolved.")
        raise typer.Exit(1)

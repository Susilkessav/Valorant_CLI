"""`valocoach notes` — standalone coaching-note management.

Complements the ``/note``, ``/notes``, and ``/resolve`` REPL slash commands
by exposing the same operations as regular CLI sub-commands so players can
manage notes without entering interactive mode.

Sub-commands
------------
    valocoach notes          — alias for ``list`` (open notes table)
    valocoach notes list     — list all open (unresolved) notes
    valocoach notes add TEXT — add a new note; category auto-inferred from text
    valocoach notes resolve ID — mark note <ID> as resolved

Category inference
------------------
Note text is run through the intent classifier so that "work on eco" lands in
the ``economy`` category and "improve clutch" lands in ``tactical`` — without
the player needing to specify a category manually.  The inferred category is
shown in the success message so the player can verify it.

The mapping from intent to category is intentionally coarse:

    clutch / post_plant / retake / tactical  →  tactical
    economy                                  →  economy
    agent_info                               →  agent
    meta                                     →  meta
    stat_analysis / general                  →  general
"""

from __future__ import annotations

import typer

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

# ---------------------------------------------------------------------------
# Intent → note category mapping
# ---------------------------------------------------------------------------

#: Maps intent classifier output to the short category tag stored in the DB.
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
    """Classify *text* with the intent classifier and map to a note category.

    Falls back to ``"general"`` on any error so the caller never has to handle
    a failure from the classifier.
    """
    try:
        from valocoach.coach.intent import classify_intent
        from valocoach.core.parser import parse_situation

        parsed = parse_situation(text)
        intent = classify_intent(parsed, text)
        return _INTENT_TO_CATEGORY.get(intent, "general")
    except Exception:
        return "general"


# ---------------------------------------------------------------------------
# Command implementations (called by the Typer app in app.py)
# ---------------------------------------------------------------------------


def run_notes_list() -> None:
    """List all open (unresolved) coaching notes."""
    settings = load_settings()
    puuid = get_player_puuid(settings)

    if not puuid:
        display.warn(
            "No player profile found — run `valocoach sync` first so "
            "ValoCoach can identify your player."
        )
        raise typer.Exit(1)

    notes = list_open_notes(settings, puuid)
    if not notes:
        display.info("No open coaching notes.  Add one with:  valocoach notes add <text>")
        return

    render_open_notes(display.console, notes)


def run_notes_add(text: str, *, priority: int = 2) -> None:
    """Add a new coaching note with auto-inferred category.

    Args:
        text:     Note body text.
        priority: 1 (high), 2 (medium — default), or 3 (low).
    """
    if not text.strip():
        display.warn("Note text cannot be empty.")
        raise typer.Exit(1)

    if priority not in (1, 2, 3):
        display.warn(f"Priority must be 1, 2, or 3 — got {priority}")
        raise typer.Exit(1)

    settings = load_settings()
    puuid = get_player_puuid(settings)

    if not puuid:
        display.warn(
            "No player profile found — run `valocoach sync` first so "
            "ValoCoach can identify your player."
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

    display.success(f"Note #{note_id} saved  [dim](category: {category})[/dim]")


def run_notes_resolve(note_id: int) -> None:
    """Mark coaching note *note_id* as resolved."""
    settings = load_settings()

    ok = resolve_coaching_note(settings, note_id)
    if ok:
        display.success(f"Note #{note_id} marked as resolved.")
    else:
        display.warn(f"Note #{note_id} not found or already resolved.")
        raise typer.Exit(1)

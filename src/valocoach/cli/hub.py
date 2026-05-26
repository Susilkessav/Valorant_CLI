"""Hub dashboard — shown when `valocoach` is run with no arguments on a TTY.

Renders a compact at-a-glance card with:
  - Current patch and meta freshness
  - Match history status (synced / not synced)
  - Quick-nav hints pointing to the most useful next commands

All data is read from local files / DB — no network calls, loads in < 200 ms.
Falls back gracefully when any section's data is unavailable.
"""

from __future__ import annotations

import logging

from valocoach.cli import display

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Section builders — each returns a list of Rich markup lines or []
# ---------------------------------------------------------------------------


def _patch_lines() -> list[str]:
    """Return patch + meta-freshness info from meta.json."""
    try:
        from valocoach.retrieval.meta import get_meta

        meta = get_meta()
        patch = meta.get("patch", "?")
        updated = meta.get("updated", "?")
        sync_flag = (
            "  [warning]⚠ refresh in progress[/warning]"
            if meta.get("sync_in_progress")
            else ""
        )
        return [
            f"  [heading]Patch[/heading] [info]{patch}[/info]"
            f"  [muted]·  meta updated {updated}{sync_flag}[/muted]"
        ]
    except Exception:
        log.debug("hub: patch info unavailable", exc_info=True)
        return []


def _player_lines(settings) -> list[str]:
    """Return player identity + quick match-count summary, or setup hint."""
    try:
        from valocoach.coach.session_manager import get_player_puuid
        from valocoach.core.config import load_settings

        if settings is None:
            return []

        # Read identity from config (fast — no DB)
        riot_id = getattr(settings, "riot_id", None) or getattr(settings, "valorant_name", None)
        tag = getattr(settings, "riot_tag", None) or getattr(settings, "valorant_tag", None)
        if riot_id and tag:
            identity = f"[info]{riot_id}[/info][muted]#{tag}[/muted]"
        else:
            return [
                "  [muted]No player configured — run [info]valocoach config init[/info] to set up.[/muted]"
            ]

        # Quick match count via the loader (reads from SQLite, fast)
        try:
            from valocoach.data.database import init_engine
            from valocoach.data.loader import load_player_data

            init_engine(settings.data_dir / "valocoach.db")
            data = load_player_data(settings, limit=1, include_rounds=False)
            if data and data.rows:
                # Use total row count as a proxy — loader caps at `limit` but
                # we only need to know "any data exists" for the hub line.
                match_note = "[muted]match history synced — run [info]valocoach stats[/info] to view[/muted]"
            else:
                match_note = "[muted]no match history — run [info]valocoach sync[/info][/muted]"
        except Exception:
            match_note = "[muted]run [info]valocoach sync[/info] to pull match history[/muted]"

        return [f"  [heading]Player[/heading] {identity}  ·  {match_note}"]

    except Exception:
        log.debug("hub: player info unavailable", exc_info=True)
        return []


def _nav_lines() -> list[str]:
    return [
        "",
        "  [heading]Quick navigation[/heading]",
        "    [info]valocoach coach[/info]            [muted]→  Interactive coaching session[/muted]",
        '    [info]valocoach coach[/info] [muted]"..."[/muted]       [muted]→  One-shot situational advice[/muted]',
        "    [info]valocoach post-game[/info]         [muted]→  Debrief your last match[/muted]",
        "    [info]valocoach stats[/info]             [muted]→  Performance dashboard[/muted]",
        "    [info]valocoach meta[/info]              [muted]→  Current tier list[/muted]",
        "    [info]valocoach sync[/info]              [muted]→  Pull latest match history[/muted]",
        "",
        "  [muted]valocoach --help  for all commands[/muted]",
    ]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def show_hub(settings=None) -> None:
    """Render the hub dashboard to the terminal."""
    if settings is None:
        try:
            from valocoach.core.config import load_settings
            settings = load_settings()
        except Exception:
            settings = None

    display.render_banner()
    display.console.print()

    patch_lines = _patch_lines()
    player_lines = _player_lines(settings)

    if patch_lines or player_lines:
        for line in patch_lines:
            display.console.print(line)
        for line in player_lines:
            display.console.print(line)

    for line in _nav_lines():
        display.console.print(line)

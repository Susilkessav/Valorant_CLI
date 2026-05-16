"""G4 — ``valocoach lineup`` command handler.

Queries the LIVE ChromaDB collection for lineup chunks matching the
supplied agent/map/site filters and displays results with video timestamps.
"""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)


def run_lineup(
    agent: str | None,
    map_name: str | None,
    site: str | None,
    query: str | None,
    n_results: int,
) -> None:
    """Display lineup suggestions matching the given filters."""
    from valocoach.cli import display
    from valocoach.core.config import load_settings
    from valocoach.retrieval.lineups import format_lineup_results, search_lineups

    settings = load_settings()

    # Build a natural language query from the filters if no explicit query given
    if query:
        search_query = query
    else:
        parts: list[str] = []
        if agent:
            parts.append(agent)
        if map_name:
            parts.append(map_name)
        if site:
            parts.append(f"site {site}")
        parts.append("ability lineup throw spot")
        search_query = " ".join(parts)

    title_parts: list[str] = ["Lineups"]
    if agent:
        title_parts.append(agent)
    if map_name:
        title_parts.append(map_name)
    if site:
        title_parts.append(f"{site} site")
    panel_title = " · ".join(title_parts)

    with display.command_frame(panel_title):
        # Rich's Console.status (not display.status — that doesn't exist).
        with display.console.status("[info]Searching lineup database…[/info]"):
            hits = search_lineups(
                settings.data_dir,
                search_query,
                agent=agent,
                map_name=map_name,
                site=site,
                n_results=n_results,
            )

        if not hits:
            display.console.print(
                "[muted]No lineups found. Try:[/muted]\n"
                "  [info]valocoach ingest --seed[/info]  [muted]to load the bundled lineup database[/muted]\n"
                "  [info]valocoach ingest --youtube <video_id>[/info]  [muted]to ingest a YouTube lineup guide[/muted]"
            )
            return

        formatted = format_lineup_results(hits)
        display.console.print(formatted)
        display.console.print()
        display.console.print(
            f"[muted]{len(hits)} lineup(s) found. "
            "Use --agent / --map / --site to narrow results.[/muted]"
        )

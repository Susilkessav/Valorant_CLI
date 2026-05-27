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
    from valocoach.retrieval.lineups import search_lineups

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
                "  [info]valocoach ingest --youtube <url>[/info]  [muted]to ingest a YouTube lineup guide[/muted]"
            )
            return

        _print_lineup_hits(hits)
        display.console.print(
            f"[muted]{len(hits)} lineup(s) matched. "
            "Refine with [bold]--agent[/bold] / [bold]--map[/bold] / [bold]--site[/bold].[/muted]"
        )


def _print_lineup_hits(hits: list[dict]) -> None:
    """Render lineup search results with Rich formatting."""
    from valocoach.cli import display

    c = display.console

    for i, hit in enumerate(hits, 1):
        meta = hit["metadata"]
        agent = meta.get("agent", "?")
        ability = meta.get("ability") or "ability"
        map_name = meta.get("map", "?")
        site = meta.get("site")
        side = meta.get("side")
        purpose = meta.get("purpose")
        channel = meta.get("channel", "?")
        title = meta.get("title", "?")
        start = int(meta.get("start_seconds", 0))
        mins, secs = divmod(start, 60)
        score = hit.get("distance", 1.0)
        relevance = max(0, round((1 - score) * 100))

        # ── Header line ───────────────────────────────────────────────────
        header_parts = [f"[val.red]{agent}[/val.red]", f"[bold]{ability}[/bold]"]
        if map_name and map_name != "?":
            header_parts.append(f"[stat.label]{map_name}[/stat.label]")
        if site:
            header_parts.append(f"[stat.value]{site} site[/stat.value]")
        if side:
            header_parts.append(f"[muted]{side}[/muted]")

        c.print(f"  [stat.value]{i}.[/stat.value]  {' · '.join(header_parts)}", end="")
        if purpose:
            c.print(f"  [muted][{purpose}][/muted]")
        else:
            c.print()

        # ── Transcript snippet ────────────────────────────────────────────
        snippet = hit["text"][:160].replace("\n", " ").strip()
        if len(hit["text"]) > 160:
            snippet += "…"
        c.print(f'       [muted]"{snippet}"[/muted]')

        # ── Source line ───────────────────────────────────────────────────
        if channel != "seed":
            c.print(
                f"       [info]▶ {channel}[/info]  [muted]{title}  @{mins}:{secs:02d}[/muted]"
                f"  [muted]({relevance}% match)[/muted]"
            )
        c.print()

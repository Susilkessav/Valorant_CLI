from __future__ import annotations

import logging

from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from valocoach.cli import display
from valocoach.core.config import load_settings
from valocoach.retrieval import (
    format_agent_context,
    format_map_context,
    get_agent,
    get_map,
    get_meta,
    list_map_names,
)

log = logging.getLogger(__name__)


def _tier_table() -> Table:
    meta = get_meta()
    tier_list = meta.get("tier_list", {})

    table = Table(show_header=True, header_style="bold", box=None, padding=(0, 1))
    table.add_column("Tier", style="bold", width=6)
    table.add_column("Agents")

    colors = {"S": "tier.s", "A": "tier.a", "B": "tier.b", "C": "tier.c"}
    for tier in ("S", "A", "B", "C"):
        agents = tier_list.get(tier, [])
        if agents:
            table.add_row(
                Text(tier, style=colors.get(tier, "white")),
                ", ".join(agents),
            )
    return table


def _eco_line() -> str:
    eco = get_meta().get("economy", {})
    return (
        f"Full buy: [stat.good]{eco.get('full_buy', 3900)} cr[/stat.good]  "
        f"Half buy: [warning]{eco.get('half_buy', 2400)} cr[/warning]  "
        f"Eco/save: [muted]<{eco.get('eco_save', 1600)} cr[/muted]"
    )


def _meta_header() -> str:
    meta = get_meta()
    return f"[heading]Patch {meta['patch']}[/heading]  [muted]·  updated {meta['updated']}  ·  {meta.get('notes', '')}[/muted]"


def _try_get_live_patch(settings) -> str | None:
    try:
        import asyncio

        from valocoach.data.database import init_engine
        from valocoach.retrieval.patch_tracker import get_current_patch

        init_engine(settings.data_dir / "valocoach.db")
        return asyncio.run(get_current_patch())
    except Exception as exc:
        log.warning("Could not read live patch from local DB: %s", exc)
        return None


def run_meta(agent: str | None, map_: str | None) -> None:
    settings = load_settings()
    meta = get_meta()

    live_patch = _try_get_live_patch(settings)
    json_patch = meta.get("patch", "")
    if live_patch and live_patch != json_patch:
        display.warn(
            f"New patch detected: [bold]{live_patch}[/bold] "
            f"(meta data still reflects {json_patch} — run `valocoach meta-refresh` to update)"
        )

    # --- agent-specific view ---
    if agent:
        agent_data = get_agent(agent)
        if not agent_data:
            display.error(f"Unknown agent: '{agent}'. Check spelling.")
            return

        resolved = agent_data["name"]
        agent_meta = meta.get("agent_meta", {}).get(resolved, {})

        # Build body with abilities + meta standing combined
        ability_text = format_agent_context(resolved) or ""
        body_parts = [ability_text]

        if agent_meta:
            tier = agent_meta.get("tier", "?")
            pick = agent_meta.get("pick_rate", "N/A")
            win = agent_meta.get("win_rate", "N/A")
            reason = agent_meta.get("reason", "")
            tier_color = {"S": "tier.s", "A": "tier.a", "B": "tier.b", "C": "tier.c"}.get(
                tier, "white"
            )
            body_parts.append("")
            body_parts.append(
                f"Tier: [{tier_color}]{tier}[/{tier_color}]  ·  "
                f"Pick rate: [val.blue]{pick}[/val.blue]  ·  "
                f"Win rate: [val.blue]{win}[/val.blue]"
            )
            if reason:
                body_parts.append(f"[muted]{reason}[/muted]")

        with display.command_frame("Agent Intel", subtitle=resolved):
            display.console.print(
                Panel(
                    "\n".join(body_parts),
                    title=f"[heading]{resolved}[/heading] [muted]— {agent_data['role']}[/muted]",
                    border_style="border.dim",
                    padding=(1, 2),
                )
            )
            if not agent_meta:
                display.warn(f"No meta stats available for {resolved} in the current dataset.")
            display.console.print()
            display.console.print(_eco_line())
        return

    # --- map-specific view ---
    if map_:
        map_data = get_map(map_)
        if not map_data:
            known = ", ".join(list_map_names())
            display.error(f"Unknown map: '{map_}'. Known maps: {known}")
            return

        resolved = map_data["name"]
        map_meta = meta.get("map_meta", {}).get(resolved, {})

        callout_text = format_map_context(resolved) or ""
        body_parts = [callout_text]

        if map_meta:
            top = ", ".join(map_meta.get("top_agents", []))
            body_parts.append("")
            body_parts.append(f"[heading]Top agents:[/heading] {top}")
            if map_meta.get("notes"):
                body_parts.append(f"[muted]{map_meta['notes']}[/muted]")

        with display.command_frame("Map Intel", subtitle=resolved):
            display.console.print(
                Panel(
                    "\n".join(body_parts),
                    title=f"[heading]{resolved}[/heading] [muted]— Callouts[/muted]",
                    border_style="border.dim",
                    padding=(1, 2),
                )
            )
            if not map_meta:
                display.warn(f"No map-specific meta available for {resolved}.")
            display.console.print()
            display.console.print(_eco_line())
        return

    # --- global overview ---
    with display.command_frame("Current Meta", subtitle=f"Patch {meta.get('patch', '?')}"):
        display.console.print(_meta_header())
        display.console.print()
        display.console.print(
            Panel(
                _tier_table(),
                title="[heading]Agent Tier List[/heading]",
                border_style="border",
                padding=(1, 2),
            )
        )
        display.console.print()
        display.console.print(_eco_line())
        display.console.print()
        display.info("Use [bold]--agent <name>[/bold] or [bold]--map <name>[/bold] for detailed info.")

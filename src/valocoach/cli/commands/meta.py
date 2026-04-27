from __future__ import annotations

from rich.columns import Columns
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from valocoach.cli import display
from valocoach.retrieval import (
    format_agent_context,
    format_map_context,
    get_agent,
    get_map,
    get_meta,
    list_map_names,
)


def _tier_table() -> Table:
    meta = get_meta()
    tier_list = meta.get("tier_list", {})

    table = Table(show_header=True, header_style="bold cyan", box=None, padding=(0, 1))
    table.add_column("Tier", style="bold", width=6)
    table.add_column("Agents")

    colors = {"S": "bold green", "A": "green", "B": "yellow", "C": "dim white"}
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
        f"Full buy: [bold green]{eco.get('full_buy', 3900)} cr[/bold green]  "
        f"Half buy: [yellow]{eco.get('half_buy', 2400)} cr[/yellow]  "
        f"Eco/save: [dim]<{eco.get('eco_save', 1600)} cr[/dim]"
    )


def _meta_header() -> str:
    meta = get_meta()
    return f"[bold]Patch {meta['patch']}[/bold]  ·  updated {meta['updated']}  ·  {meta.get('notes', '')}"


def run_meta(agent: str | None, map_: str | None) -> None:
    meta = get_meta()

    # --- agent-specific view ---
    if agent:
        agent_data = get_agent(agent)
        if not agent_data:
            display.error(f"Unknown agent: '{agent}'. Check spelling.")
            return

        resolved = agent_data["name"]
        agent_meta = meta.get("agent_meta", {}).get(resolved, {})

        # Ability block
        ability_text = format_agent_context(resolved) or ""
        display.console.print(
            Panel(
                ability_text,
                title=f"[bold cyan]{resolved}[/bold cyan] — {agent_data['role']}",
                border_style="cyan",
                padding=(1, 2),
            )
        )

        # Meta standing
        if agent_meta:
            tier = agent_meta.get("tier", "?")
            pick = agent_meta.get("pick_rate", "N/A")
            win = agent_meta.get("win_rate", "N/A")
            reason = agent_meta.get("reason", "")
            tier_color = {"S": "bold green", "A": "green", "B": "yellow", "C": "dim white"}.get(
                tier, "white"
            )
            display.console.print(
                f"  Tier: [{tier_color}]{tier}[/{tier_color}]  ·  "
                f"Pick rate: [cyan]{pick}[/cyan]  ·  "
                f"Win rate: [cyan]{win}[/cyan]"
            )
            if reason:
                display.console.print(f"  [dim]{reason}[/dim]")
        else:
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

        # Callout block
        callout_text = format_map_context(resolved) or ""
        display.console.print(
            Panel(
                callout_text,
                title=f"[bold cyan]{resolved}[/bold cyan] — Callouts",
                border_style="cyan",
                padding=(1, 2),
            )
        )

        # Map meta
        if map_meta:
            top = ", ".join(map_meta.get("top_agents", []))
            display.console.print(f"  [bold]Top agents on {resolved}:[/bold] {top}")
            if map_meta.get("notes"):
                display.console.print(f"  [dim]{map_meta['notes']}[/dim]")
        else:
            display.warn(f"No map-specific meta available for {resolved}.")

        display.console.print()
        display.console.print(_eco_line())
        return

    # --- global overview ---
    display.console.print(_meta_header())
    display.console.print()
    display.console.print(
        Panel(
            _tier_table(),
            title="[bold]Agent Tier List[/bold]",
            border_style="cyan",
            padding=(1, 2),
        )
    )
    display.console.print()
    display.console.print(_eco_line())
    display.console.print()
    display.info("Use [bold]--agent <name>[/bold] or [bold]--map <name>[/bold] for detailed info.")

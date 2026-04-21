"""`valocoach profile` — player identity + compact recent-performance card.

Stats and profile share machinery but answer different questions:

    stats   — "how am I actually playing?"     period-filtered, per-map breakdown
    profile — "who is this player, at a glance?"  identity + rank + last-N summary

Argument resolution:
    valocoach profile                        → uses settings.riot_name / riot_tag
    valocoach profile --name X --tag Y       → looks up that player
    valocoach profile --name X               → error (name and tag go together)

Either way, we only consult the local DB — no live API calls. If the target
player has never been synced, we nudge the user toward ``valocoach sync``
rather than silently fetching from HenrikDev here.
"""

from __future__ import annotations

import asyncio

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from valocoach.cli import display

# Reach into stats.py for the shared per-agent breakdown renderer and the
# percent formatter — they're private-by-convention to discourage drive-by
# imports, but a sibling CLI command is a legitimate reuse case.
from valocoach.cli.commands.stats import _fmt_pct, _render_breakdown
from valocoach.core.config import load_settings
from valocoach.data.database import ensure_db, session_scope
from valocoach.data.orm_models import MatchPlayer, Player
from valocoach.data.repository import get_player_by_name, get_recent_matches
from valocoach.stats import compute_per_agent, compute_player_stats

# Default number of recent matches to summarise. Profile is "at a glance" —
# a larger N starts looking like the stats dashboard.
DEFAULT_LIMIT = 20

# Top agents to surface in the profile card.
TOP_AGENTS = 3


# ---------------------------------------------------------------------------
# Argument resolution  (pure — tested directly)
# ---------------------------------------------------------------------------


def _resolve_identity(
    *,
    name: str | None,
    tag: str | None,
    settings_name: str,
    settings_tag: str,
) -> tuple[str, str]:
    """Work out which player to look up.

    Rules:
        - Both CLI args given    → use them.
        - Neither CLI arg given  → fall back to settings.
        - One but not the other  → typer.BadParameter.
        - Fallback with empty settings → typer.BadParameter.

    Raises:
        typer.BadParameter: surfaces as a clean Typer error (exit 2).
    """
    if (name is None) != (tag is None):
        raise typer.BadParameter(
            "--name and --tag must be given together (or both omitted to use your configured identity)."
        )

    if name is not None and tag is not None:
        return name, tag

    if not settings_name or not settings_tag:
        raise typer.BadParameter(
            "No --name/--tag given and no configured identity — "
            "run `valocoach config init` or pass --name/--tag."
        )
    return settings_name, settings_tag


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _render_identity_panel(console: Console, player: Player) -> None:
    """Top panel: name, rank, peak, region, account level, last match."""
    last_match = player.last_match_at or "never"

    body_lines = [
        f"[bold]Current[/bold]   {player.current_tier_patched}"
        f"  [dim]({player.current_rr} RR · elo {player.elo})[/dim]",
        f"[bold]Peak[/bold]      {player.peak_tier_patched}",
        f"[bold]Region[/bold]    {player.region.upper()}   "
        f"[dim]· level {player.account_level}[/dim]",
        f"[bold]Last match[/bold]  [dim]{last_match}[/dim]",
    ]
    title = f"[bold]{player.riot_name}#{player.riot_tag}[/bold]"
    console.print(Panel("\n".join(body_lines), title=title, border_style="cyan", padding=(0, 2)))


def _render_summary_card(
    console: Console,
    rows: list[MatchPlayer],
    *,
    limit: int,
) -> None:
    """Compact "last N matches" summary. One table, dense numbers.

    Different shape from the stats dashboard's Overall card — tighter, meant
    to fit alongside the identity panel rather than dominate the screen.
    """
    if not rows:
        console.print("[dim]No matches in the local DB yet.[/dim]")
        return

    stats = compute_player_stats(rows)
    shown = min(len(rows), limit)

    table = Table(
        title=f"Last {shown} match(es)",
        show_header=False,
        box=None,
        pad_edge=False,
    )
    table.add_column("Label", style="dim")
    table.add_column("Value", justify="right")
    table.add_column("Label", style="dim")
    table.add_column("Value", justify="right")

    table.add_row(
        "Record", f"{stats.wins}-{stats.losses}  ({_fmt_pct(stats.win_rate)})",
        "ACS", f"{stats.acs:.0f}",
    )
    table.add_row(
        "K/D", f"{stats.kd:.2f}",
        "KDA", f"{stats.kda:.2f}",
    )
    table.add_row(
        "HS%", _fmt_pct(stats.hs_pct),
        "ADR", f"{stats.adr:.0f}",
    )
    table.add_row(
        "FB diff", f"{stats.fb_diff:+d}",
        "Rounds", str(stats.rounds),
    )
    console.print(table)


# ---------------------------------------------------------------------------
# Async worker
# ---------------------------------------------------------------------------


async def _fetch_profile_data(
    *,
    db_path,
    name: str,
    tag: str,
    limit: int,
) -> tuple[Player, list[MatchPlayer]] | None:
    """Resolve player + fetch recent rows. None if player is unknown locally."""
    await ensure_db(db_path)

    async with session_scope() as session:
        player = await get_player_by_name(session, name, tag)
        if player is None:
            return None
        rows = await get_recent_matches(session, player.puuid, limit=limit)

    return player, rows


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run_profile(
    *,
    name: str | None = None,
    tag: str | None = None,
    limit: int = DEFAULT_LIMIT,
    console: Console | None = None,
) -> None:
    """CLI entry: ``valocoach profile`` dispatches here."""
    con = console or display.console

    if limit <= 0:
        raise typer.BadParameter(f"--limit must be positive; got {limit}")

    settings = load_settings()
    resolved_name, resolved_tag = _resolve_identity(
        name=name,
        tag=tag,
        settings_name=settings.riot_name,
        settings_tag=settings.riot_tag,
    )

    fetched = asyncio.run(
        _fetch_profile_data(
            db_path=settings.data_dir / "valocoach.db",
            name=resolved_name,
            tag=resolved_tag,
            limit=limit,
        )
    )
    if fetched is None:
        display.warn(
            f"No local data for {resolved_name}#{resolved_tag}. "
            "Run `valocoach sync` first to pull match history."
        )
        raise typer.Exit(1)

    player, rows = fetched

    _render_identity_panel(con, player)
    _render_summary_card(con, rows, limit=limit)

    # Top-N agents — reuses the stats breakdown renderer. Skip if only one
    # agent shows up; the single row would be redundant with the summary.
    per_agent = compute_per_agent(rows)
    if len(per_agent) >= 2:
        con.print()
        _render_breakdown(
            con,
            title="Top agents",
            group_col="Agent",
            rows=per_agent,
            top_n=TOP_AGENTS,
        )

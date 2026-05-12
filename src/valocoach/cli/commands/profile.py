"""`valocoach profile` — player identity + compact recent-performance card.

Supports two modes:
  1. **Local profile** (default) — renders from DB-synced match history.
     Full stats, round analysis, coaching notes.
  2. **Live lookup** — when ``--name``/``--tag`` targets a player not in the
     local DB, fetches account + MMR + last 10 matches from the HenrikDev
     API and renders an ephemeral card.  Nothing is stored; data lives only
     in memory for the duration of the command.
"""

from __future__ import annotations

import asyncio

import typer
from rich.console import Console

from valocoach.cli import display
from valocoach.cli.formatter import (
    render_breakdown,
    render_coaching_sessions,
    render_identity_panel,
    render_lookup_identity_panel,
    render_lookup_summary,
    render_open_notes,
    render_rank_trend,
    render_round_stats,
    render_summary_card,
    render_trend,
    render_warn_legend,
)
from valocoach.coach.session_manager import (
    get_mmr_trend,
    list_coaching_sessions,
    list_open_notes,
)
from valocoach.core.config import load_settings
from valocoach.data.loader import load_player_data
from valocoach.stats import compute_per_agent
from valocoach.stats.round_analyzer import analyze_rounds

DEFAULT_LIMIT = 20
TOP_AGENTS = 3


def _resolve_identity(
    *,
    name: str | None,
    tag: str | None,
    settings_name: str,
    settings_tag: str,
) -> tuple[str, str]:
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


def run_profile(
    *,
    name: str | None = None,
    tag: str | None = None,
    limit: int = DEFAULT_LIMIT,
    console: Console | None = None,
) -> None:
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

    data = load_player_data(
        settings,
        name=resolved_name,
        tag=resolved_tag,
        limit=limit,
        include_rounds=True,
    )

    if data is None:
        if name is not None and tag is not None:
            run_lookup(
                name=resolved_name,
                tag=resolved_tag,
                region=settings.riot_region,
                console=con,
            )
            return
        display.error_with_hint(
            f"No local data for {resolved_name}#{resolved_tag}.",
            "Run: valocoach sync",
        )
        raise typer.Exit(1)

    player, rows = data.player, data.rows

    with display.command_frame("Player Profile", subtitle=f"{resolved_name}#{resolved_tag}", con=con):
        # Identity panel
        render_identity_panel(con, player)

        # Rank trend
        mmr_history = get_mmr_trend(settings, player.puuid, limit=20)
        if mmr_history:
            display.render_section(con, "Rank Progression")
            render_rank_trend(con, mmr_history)

        # Summary card
        display.render_section(con, f"Last {min(len(rows), limit)} Matches")
        any_warn = render_summary_card(con, rows, limit=limit)

        # Round stats
        if data.full_matches:
            round_analysis = analyze_rounds(data.full_matches, player.puuid)
            if round_analysis.rounds > 0:
                display.render_section(con, "Round Mastery")
                any_warn |= render_round_stats(con, round_analysis, len(rows))

        # Top agents
        per_agent = compute_per_agent(rows)
        if len(per_agent) >= 2:
            display.render_section(con, "Top Agents")
            any_warn |= render_breakdown(
                con,
                title="Top Agents",
                group_col="Agent",
                rows=per_agent,
                top_n=TOP_AGENTS,
            )

        # Trend
        render_trend(con, rows)

        if any_warn:
            con.print()
            render_warn_legend(con)

        # Coaching section — sessions + notes combined
        try:
            sessions = list_coaching_sessions(settings, player.puuid, limit=5)
            notes = list_open_notes(settings, player.puuid, limit=10)
        except Exception:
            sessions = []
            notes = []

        if sessions or notes:
            display.render_section(con, "Coaching")
            if sessions:
                render_coaching_sessions(con, sessions)
            if notes:
                con.print()
                render_open_notes(con, notes)


# ---------------------------------------------------------------------------
# Live lookup — ephemeral, no DB persistence
# ---------------------------------------------------------------------------


def run_lookup(
    *,
    name: str,
    tag: str,
    region: str = "na",
    console: Console | None = None,
) -> None:
    """Fetch a player's profile from the HenrikDev API and render it ephemerally."""
    from valocoach.core.exceptions import APIError, ConfigError
    from valocoach.data.api_client import HenrikClient
    from valocoach.stats.calculator import (
        compute_per_agent_from_api,
        compute_stats_from_api,
    )

    con = console or display.console
    settings = load_settings()

    async def _fetch():
        async with HenrikClient(settings) as client:
            snapshot = await client.fetch_player_snapshot(region, name, tag)
            try:
                mmr_history = await client.get_mmr_history(region, name, tag)
            except Exception:
                mmr_history = []
            return (*snapshot, mmr_history)

    display.info(f"Looking up [heading]{name}#{tag}[/heading] via API…")

    try:
        account, mmr, matches, mmr_history = asyncio.run(_fetch())
    except ConfigError as exc:
        display.error(str(exc))
        raise typer.Exit(1) from exc
    except APIError as exc:
        display.error_with_hint(
            f"API lookup failed for {name}#{tag}: {exc}",
            "Check the name/tag spelling and your API key.",
        )
        raise typer.Exit(1) from exc

    stats = compute_stats_from_api(matches, name, tag)
    per_agent = compute_per_agent_from_api(matches, name, tag)

    # Normalize API MMRHistoryEntry → shape render_rank_trend expects
    from valocoach.coach.session_manager import MMRHistoryInfo

    rank_history = [
        MMRHistoryInfo(
            tier_patched=h.currenttierpatched,
            rr=h.ranking_in_tier,
            elo=h.elo,
            mmr_change=h.mmr_change_to_last_game or None,
            recorded_at=h.date or "",
        )
        for h in mmr_history
    ]

    with display.command_frame("Player Lookup", subtitle=f"{name}#{tag}", con=con):
        # Identity panel
        render_lookup_identity_panel(con, account, mmr)

        # Rank trend
        if rank_history:
            display.render_section(con, "Rank Progression")
            render_rank_trend(con, rank_history)

        # Summary card
        display.render_section(con, f"Last {stats.matches} Matches")
        any_warn = render_lookup_summary(con, stats)

        # Top agents
        if len(per_agent) >= 2:
            display.render_section(con, "Top Agents")
            any_warn |= render_breakdown(
                con,
                title="Top Agents",
                group_col="Agent",
                rows=per_agent,
                top_n=TOP_AGENTS,
            )

        if any_warn:
            con.print()
            render_warn_legend(con)

        # Footer
        con.print()
        con.print(
            "[muted]Live lookup — data is ephemeral and not stored locally. "
            "Free-tier API limits results to 10 matches.[/muted]"
        )

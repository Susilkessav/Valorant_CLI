"""`valocoach stats` — performance dashboard from locally synced matches."""

from __future__ import annotations

import typer
from rich.console import Console

from valocoach.cli import display
from valocoach.cli.formatter import (
    render_breakdown,
    render_header,
    render_overall,
    render_round_stats,
    render_trend,
    render_warn_legend,
    render_win_loss_split,
)
from valocoach.core.config import load_settings
from valocoach.data.loader import load_player_data
from valocoach.stats import (
    compute_per_agent,
    compute_per_map,
    compute_player_stats,
    parse_period,
    split_by_result,
)
from valocoach.stats.round_analyzer import analyze_rounds

FETCH_LIMIT = 200
TOP_N = 5


def _period_to_cutoff_iso(period: str) -> str | None:
    try:
        return parse_period(period)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc


def run_stats(
    *,
    agent: str | None = None,
    map_: str | None = None,
    period: str = "30d",
    result: str | None = None,
    console: Console | None = None,
) -> None:
    con = console or display.console

    # Fire the patch-staleness warning at most once per process so users on
    # the stats / profile flow are still told their meta data is stale,
    # without spamming the warning on every command.
    try:
        from valocoach.cli.commands.coach import warn_stale_meta_once
        from valocoach.core.config import load_settings as _load

        warn_stale_meta_once(_load())
    except Exception:
        pass

    won: bool | None
    if result is None:
        won = None
    elif result.lower() == "win":
        won = True
    elif result.lower() in ("loss", "lose"):
        won = False
    else:
        raise typer.BadParameter(
            f"--result must be 'win' or 'loss'; got {result!r}", param_hint="'--result'"
        )

    cutoff_iso = _period_to_cutoff_iso(period)

    settings = load_settings()
    if not settings.riot_name or not settings.riot_tag:
        display.error_with_hint(
            "riot_name / riot_tag not configured.",
            "Run: valocoach config init",
        )
        raise typer.Exit(1)

    data = load_player_data(settings, limit=FETCH_LIMIT, include_rounds=True)
    if data is None:
        display.error_with_hint(
            f"No local data for {settings.riot_name}#{settings.riot_tag}.",
            "Run: valocoach sync",
        )
        raise typer.Exit(1)

    player, rows = data.player, data.rows

    from valocoach.stats.filters import (
        filter_by_agent,
        filter_by_map,
        filter_by_period,
        filter_by_result,
    )

    filtered = filter_by_period(rows, cutoff_iso)
    filtered = filter_by_agent(filtered, agent)
    filtered = filter_by_map(filtered, map_)
    filtered = filter_by_result(filtered, won)

    if not filtered:
        display.warn(
            "No matches after filters"
            + (f" (period={period}" if period != "all" else " (period=all")
            + (f", agent={agent}" if agent else "")
            + (f", map={map_}" if map_ else "")
            + (f", result={result}" if result else "")
            + ")."
        )
        # Helpful hint when the window itself is the likely cause — point at
        # the obvious wider window before they go looking through --help.
        if period != "all" and not agent and not map_ and not result:
            display.console.print(
                "[muted]Tip: try [info]--period 365d[/info] or "
                "[info]--period all[/info] for a wider window.[/muted]"
            )
        raise typer.Exit(0)

    overall = compute_player_stats(filtered)
    per_agent = compute_per_agent(filtered)
    per_map = compute_per_map(filtered)

    # Build subtitle from active filters
    filter_parts = [f"period={period}"]
    if agent:
        filter_parts.append(f"agent={agent}")
    if map_:
        filter_parts.append(f"map={map_}")
    if result:
        filter_parts.append(f"result={result}")
    subtitle = " · ".join(filter_parts)

    with display.command_frame("Stats Dashboard", subtitle=subtitle, con=con):
        # Identity line
        render_header(
            con,
            name=player.riot_name,
            tag=player.riot_tag,
            tier=player.current_tier_patched,
            region=player.region,
            matches_shown=overall.matches,
            period=period,
            agent_filter=agent,
            map_filter=map_,
            result_filter=result,
        )

        # Core Performance
        display.render_section(con, "Core Performance")
        any_warn = render_overall(con, overall)

        # Recent Form — silent when stable
        from valocoach.stats.baseline import compare_baseline
        comparison = compare_baseline(filtered)
        if comparison is not None and comparison.has_anomalies:
            display.render_section(con, "Recent Form")
            render_trend(con, filtered)

        # Win vs Loss — only when no result filter
        if won is None:
            wins_rows, losses_rows = split_by_result(filtered)
            if wins_rows and losses_rows:
                display.render_section(con, "Win vs Loss")
                any_warn |= render_win_loss_split(con, wins_rows, losses_rows)

        # Round Mastery — moved up from bottom
        if data.full_matches:
            filtered_match_ids = {mp.match_id for mp in filtered}
            filtered_full_matches = [m for m in data.full_matches if m.match_id in filtered_match_ids]
            if filtered_full_matches:
                round_analysis = analyze_rounds(filtered_full_matches, player.puuid)
                if round_analysis.rounds > 0:
                    display.render_section(con, "Round Mastery")
                    any_warn |= render_round_stats(con, round_analysis, overall.matches)

        # Agent Breakdown
        if agent is None and per_agent:
            display.render_section(con, "Agent Breakdown")
            any_warn |= render_breakdown(
                con, title="By Agent", group_col="Agent", rows=per_agent, top_n=TOP_N
            )

        # Map Breakdown
        if map_ is None and per_map:
            display.render_section(con, "Map Breakdown")
            any_warn |= render_breakdown(
                con, title="By Map", group_col="Map", rows=per_map, top_n=TOP_N
            )

        if any_warn:
            con.print()
            render_warn_legend(con)

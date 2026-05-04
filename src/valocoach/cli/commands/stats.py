"""`valocoach stats` — performance dashboard from locally synced matches.

Flow:
    1. Load settings → validate riot_name/riot_tag are configured.
    2. load_player_data() → ensure_db + session_scope + fetch rows (via loader).
    3. Apply --period / --agent / --map / --result filters (pure, cheap).
    4. Call formatter renderers → overall + trend + win/loss split + per-agent + per-map.

The DB-facing phase is inside the loader (async internally, sync entry point).
By the time we render, every row is detached from the session.  All rendering
logic lives in ``valocoach.cli.formatter`` so this module is pure orchestration:
parse args → load data → filter → render.
"""

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

# Fetch plenty of rows — the period/agent/map filters narrow from here in Python.
# 200 covers ~3 months of ranked for a heavy player; raise when we track >1 player.
FETCH_LIMIT = 200

# How many rows to show in the per-agent and per-map tables.
TOP_N = 5


# ---------------------------------------------------------------------------
# Period parsing  (thin CLI wrapper around the pure parse_period helper)
# ---------------------------------------------------------------------------


def _period_to_cutoff_iso(period: str) -> str | None:
    """Translate a ``--period`` string into an ISO8601 cutoff timestamp.

    Delegates to :func:`valocoach.stats.filters.parse_period` and converts
    ``ValueError`` to ``typer.BadParameter`` so the CLI surfaces a clean
    error message instead of a traceback.
    """
    try:
        return parse_period(period)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run_stats(
    *,
    agent: str | None = None,
    map_: str | None = None,
    period: str = "30d",
    result: str | None = None,
    console: Console | None = None,
) -> None:
    """CLI entry: ``valocoach stats`` dispatches here.

    Args:
        agent:   Case-insensitive agent filter (e.g. ``'Jett'``).
        map_:    Case-insensitive map filter (e.g. ``'Ascent'``).
        period:  Time window — ``'Nd'`` or ``'all'``.
        result:  ``'win'`` to show wins only, ``'loss'`` for losses only,
                 ``None`` to show both (default).
        console: Injectable for testing; defaults to the shared theme.
    """
    con = console or display.console

    # Parse and validate result filter before any I/O — fast fail on bad input.
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

    # Validate period before I/O too.
    cutoff_iso = _period_to_cutoff_iso(period)

    settings = load_settings()
    if not settings.riot_name or not settings.riot_tag:
        display.error(
            "riot_name / riot_tag not configured — run `valocoach config init` and edit the file"
        )
        raise typer.Exit(1)

    data = load_player_data(settings, limit=FETCH_LIMIT, include_rounds=True)
    if data is None:
        display.warn(
            f"No local data for {settings.riot_name}#{settings.riot_tag}. "
            "Run `valocoach sync` first to pull match history."
        )
        raise typer.Exit(1)

    player, rows = data.player, data.rows

    # Apply all filters in one pass. period is already parsed to a cutoff.
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
        raise typer.Exit(0)

    overall = compute_player_stats(filtered)
    per_agent = compute_per_agent(filtered)
    per_map = compute_per_map(filtered)

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
    any_warn = render_overall(con, overall)
    con.print()

    # Trend section — silent when stable or window is too thin.
    render_trend(con, filtered)

    # Win/Loss split — only when no result filter is active (splits are
    # meaningless if the user already filtered to wins or losses only).
    if won is None:
        wins_rows, losses_rows = split_by_result(filtered)
        any_warn |= render_win_loss_split(con, wins_rows, losses_rows)
        con.print()

    # Skip per-agent breakdown when the user already filtered to one agent —
    # the single-row table would be redundant with the overall card.
    if agent is None:
        any_warn |= render_breakdown(
            con, title="By agent", group_col="Agent", rows=per_agent, top_n=TOP_N
        )
        con.print()
    if map_ is None:
        any_warn |= render_breakdown(
            con, title="By map", group_col="Map", rows=per_map, top_n=TOP_N
        )

    # Round-level stats (KAST, clutch, trade, side win rates).
    # Only shown when full match data was loaded (include_rounds=True) and the
    # round analyzer finds round events for the player. Absent data renders
    # nothing — users with pre-round-migration history see a clean stats card.
    #
    # Filter scope: the round analyzer must run on the SAME match set the
    # aggregate card was computed from.  Otherwise ``--agent Jett --map Ascent``
    # shows ACS for filtered Jett-on-Ascent matches but KAST/clutch/trade for
    # every loaded match — silently mixing scopes (FINDINGS P1).
    if data.full_matches:
        filtered_match_ids = {mp.match_id for mp in filtered}
        filtered_full_matches = [m for m in data.full_matches if m.match_id in filtered_match_ids]
        if filtered_full_matches:
            round_analysis = analyze_rounds(filtered_full_matches, player.puuid)
            if round_analysis.rounds > 0:
                con.print()
                any_warn |= render_round_stats(con, round_analysis, overall.matches)

    if any_warn:
        con.print()
        render_warn_legend(con)

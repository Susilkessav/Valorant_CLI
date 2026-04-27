"""`valocoach profile` — player identity + compact recent-performance card.

Stats and profile share machinery but answer different questions:

    stats   — "how am I actually playing?"     period-filtered, per-map breakdown
    profile — "who is this player, at a glance?"  identity + rank + last-N summary

Argument resolution:
    valocoach profile                        → uses settings.riot_name / riot_tag
    valocoach profile --name X --tag Y       → looks up that player
    valocoach profile --name X               → error (name and tag go together)

Data flow (stats engine):
    1. _resolve_identity()             → (name, tag)
    2. load_player_data_async()        → PlayerData  (DB bridge, include_rounds=True)
    3. compute_per_agent(rows)         → per-agent breakdown
    4. analyze_rounds(full_matches, …) → KAST / clutch / trade  (when available)
    5. compare_baseline(rows)          → recent-form anomalies  (when enough history)
    6. Formatter renderers             → display
"""

from __future__ import annotations

import asyncio

import typer
from rich.console import Console

from valocoach.cli import display
from valocoach.cli.formatter import (
    render_breakdown,
    render_identity_panel,
    render_round_stats,
    render_summary_card,
    render_trend,
    render_warn_legend,
)
from valocoach.core.config import load_settings
from valocoach.data.loader import load_player_data_async
from valocoach.stats import compute_per_agent
from valocoach.stats.round_analyzer import analyze_rounds

# Default number of recent matches to summarise.  Profile is "at a glance" —
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

    # Use the shared async loader with name/tag overrides so profile can look
    # up any player, not just the Settings-configured one.  include_rounds=True
    # so we can show KAST / clutch / trade when round data is available.
    data = asyncio.run(
        load_player_data_async(
            settings,
            name=resolved_name,
            tag=resolved_tag,
            limit=limit,
            include_rounds=True,
        )
    )

    if data is None:
        display.warn(
            f"No local data for {resolved_name}#{resolved_tag}. "
            "Run `valocoach sync` first to pull match history."
        )
        raise typer.Exit(1)

    player, rows = data.player, data.rows

    # ── Identity panel ─────────────────────────────────────────────────────
    render_identity_panel(con, player)

    # ── Compact summary (last N matches) ───────────────────────────────────
    any_warn = render_summary_card(con, rows, limit=limit)

    # ── Round-level stats: KAST / clutch / trade ───────────────────────────
    # Silently absent when rounds weren't synced (pre-migration data) or when
    # no round events exist for this player in the DB.
    if data.full_matches:
        round_analysis = analyze_rounds(data.full_matches, player.puuid)
        if round_analysis.rounds > 0:
            con.print()
            any_warn |= render_round_stats(con, round_analysis, len(rows))

    # ── Top agents ─────────────────────────────────────────────────────────
    # Skip when every game was on the same agent — a single-row table adds
    # no information beyond what the summary card already shows.
    per_agent = compute_per_agent(rows)
    if len(per_agent) >= 2:
        con.print()
        any_warn |= render_breakdown(
            con,
            title="Top agents",
            group_col="Agent",
            rows=per_agent,
            top_n=TOP_AGENTS,
        )

    # ── Recent form anomalies ──────────────────────────────────────────────
    # compare_baseline returns None when the window is too thin; render_trend
    # is silent when there are no anomalies.  Neither clutters a clean card.
    render_trend(con, rows)

    if any_warn:
        con.print()
        render_warn_legend(con)

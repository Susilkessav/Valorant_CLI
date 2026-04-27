"""Rich rendering for all CLI output — the single source of truth for display.

Separation of concerns
----------------------
Commands own: argument parsing, data loading, filter application, error handling.
This module owns: every Rich table, panel, and formatted value that ends up on screen.

Design goals
------------
- Pure render layer — no I/O, no DB, no API calls. Functions take computed
  objects (PlayerStats, AgentStats, Player, …) and a Console; they write output.
- Public names — nothing private-by-convention. Callers import exactly what
  they need without reaching into each other's command modules.
- Testable — each render function returns a ``bool`` (``True`` = at least one
  ⚠️ cell was shown) so tests can assert on the warn/no-warn contract without
  parsing the rendered string, and integration tests that DO parse the string
  can be explicit about what they're looking for.
- One legend — ``render_warn_legend`` is the only place the ⚠️ footer text
  lives. Both ``stats`` and ``profile`` call it; it cannot drift.

Formatting primitives
---------------------
- ``fmt_pct``      — 0.27 → "27.0%"
- ``fmt_delta``    — signed difference between two floats
- ``WARN_PREFIX``  — the ⚠️ glyph prepended to unreliable cells
- ``warn_cell``    — conditionally prepend WARN_PREFIX

Rich renderers
--------------
- ``render_warn_legend``       — the ⚠️ explanation footer
- ``render_header``            — top panel (name, rank, region, active filters)
- ``render_overall``           — overall stats two-column table  (stats command)
- ``render_breakdown``         — per-agent or per-map table  (stats + profile)
- ``render_win_loss_split``    — wins vs losses comparison table  (stats command)
- ``render_trend``             — recent-form anomaly block  (stats command)
- ``render_round_stats``       — KAST / clutch / trade / side split  (stats command)
- ``render_identity_panel``    — player identity panel  (profile command)
- ``render_summary_card``      — compact "last N matches" table  (profile command)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from valocoach.data.orm_models import MatchPlayer, Player
from valocoach.stats.baseline import compare_baseline
from valocoach.stats.calculator import AgentStats, MapStats, PlayerStats, reliability_flags

if TYPE_CHECKING:
    pass

# ---------------------------------------------------------------------------
# Formatting primitives
# ---------------------------------------------------------------------------

WARN_PREFIX: Final[str] = "⚠️ "
"""Glyph prepended to unreliable metric cells.

Exposed as a constant so tests can assert ``WARN_PREFIX in output`` without
duplicating the Unicode literal, and so any future theming swap stays in one
place.
"""


def fmt_pct(ratio: float) -> str:
    """Render a [0.0, 1.0] ratio as a percentage string.

    >>> fmt_pct(0.2734)
    '27.3%'
    >>> fmt_pct(1.0)
    '100.0%'
    """
    return f"{ratio * 100:.1f}%"


def fmt_delta(a: float, b: float, *, fmt: str = ".1f") -> str:
    """Signed difference ``a - b`` with a leading ``+`` when non-negative.

    Used in the win/loss split table to show "how much better/worse in wins".

    >>> fmt_delta(210.0, 170.0, fmt=".0f")
    '+40'
    >>> fmt_delta(0.95, 1.10, fmt=".2f")
    '-0.15'
    """
    d = a - b
    sign = "+" if d >= 0 else ""
    return f"{sign}{d:{fmt}}"


def warn_cell(value: str, reliable: bool) -> str:
    """Prepend ``WARN_PREFIX`` when ``reliable`` is ``False``.

    Pure presentation concern — the reliability truth-value comes from
    ``calculator.reliability_flags``, never re-derived here.

    >>> warn_cell("245", True)
    '245'
    >>> warn_cell("245", False).startswith("⚠️")
    True
    """
    return value if reliable else f"{WARN_PREFIX}{value}"


# ---------------------------------------------------------------------------
# Shared footer
# ---------------------------------------------------------------------------


def render_warn_legend(console: Console) -> None:
    """Print the ⚠️ explanation footer.

    Call this after rendering when at least one ``warn_cell`` fired (i.e.
    when a render function returned ``True``). The footer teaches the user
    what ⚠️ means so it's never a mystery.

    This is the single source of truth for that text — both ``stats`` and
    ``profile`` call this function; the message cannot drift between them.
    """
    console.print(
        f"[dim]{WARN_PREFIX.strip()}  = below the sample-size threshold for this metric; "
        "treat as indicative, not reliable "
        "(see BUILD_PLAN.md \u00a7 sample-size thresholds).[/dim]"
    )


# ---------------------------------------------------------------------------
# stats command renderers
# ---------------------------------------------------------------------------


def render_header(
    console: Console,
    *,
    name: str,
    tag: str,
    tier: str,
    region: str,
    matches_shown: int,
    period: str,
    agent_filter: str | None,
    map_filter: str | None,
    result_filter: str | None = None,
) -> None:
    """Top panel: who we're showing, over what window."""
    filter_bits = [f"period={period}"]
    if agent_filter:
        filter_bits.append(f"agent={agent_filter}")
    if map_filter:
        filter_bits.append(f"map={map_filter}")
    if result_filter:
        filter_bits.append(f"result={result_filter}")

    title = f"[bold]{name}#{tag}[/bold]  [dim]·[/dim]  {tier}  [dim]·[/dim]  {region.upper()}"
    subtitle = f"{matches_shown} match(es) after filters · " + " · ".join(filter_bits)
    console.print(Panel(subtitle, title=title, border_style="cyan", padding=(0, 2)))


def render_overall(console: Console, stats: PlayerStats) -> bool:
    """Overall stats — two-column layout of the numbers that matter most.

    Returns ``True`` if any metric was tagged as unreliable (⚠️). Callers
    roll that up to decide whether to show the footer legend.
    """
    flags = reliability_flags(stats)
    any_warn = not all(flags.values())

    table = Table(title="Overall", show_header=False, box=None, pad_edge=False)
    table.add_column("Label", style="dim")
    table.add_column("Value", justify="right")
    table.add_column("Label", style="dim")
    table.add_column("Value", justify="right")

    table.add_row(
        "Matches",
        str(stats.matches),
        "Win rate",
        warn_cell(
            f"{stats.wins}-{stats.losses}  ({fmt_pct(stats.win_rate)})",
            flags["win_rate"],
        ),
    )
    table.add_row(
        "Rounds",
        str(stats.rounds),
        "ACS",
        warn_cell(f"{stats.acs:.1f}", flags["acs"]),
    )
    table.add_row(
        "K / D / A",
        f"{stats.kills} / {stats.deaths} / {stats.assists}",
        "ADR",
        warn_cell(f"{stats.adr:.1f}", flags["adr"]),
    )
    table.add_row(
        "K/D",
        warn_cell(f"{stats.kd:.2f}", flags["kd"]),
        "KDA",
        warn_cell(f"{stats.kda:.2f}", flags["kda"]),
    )
    table.add_row(
        "HS%",
        warn_cell(fmt_pct(stats.hs_pct), flags["hs_pct"]),
        "FB / FD (diff)",
        # FB and FD share a threshold (same rarity). Either being thin
        # warrants a ⚠️ on the combined cell.
        warn_cell(
            f"{stats.first_bloods} / {stats.first_deaths}  ({stats.fb_diff:+d})",
            flags["fb_rate"] and flags["fd_rate"],
        ),
    )
    console.print(table)
    return any_warn


def render_breakdown(
    console: Console,
    *,
    title: str,
    group_col: str,
    rows: list[AgentStats] | list[MapStats],
    top_n: int,
) -> bool:
    """Per-agent or per-map table, top ``top_n`` rows by matches.

    Returns ``True`` if any cell was tagged as unreliable. Per-split rows
    use the stricter ``is_split=True`` floor — a 4-game Jett split should
    ⚠️ even when the overall sample is reliable.

    Used by both ``valocoach stats`` (agent + map breakdown) and
    ``valocoach profile`` (top agents).
    """
    if not rows:
        return False

    table = Table(title=title, show_header=True, header_style="bold", pad_edge=False)
    table.add_column(group_col, style="cyan")
    table.add_column("G", justify="right")
    table.add_column("W-L", justify="right")
    table.add_column("Win%", justify="right")
    table.add_column("ACS", justify="right")
    table.add_column("K/D", justify="right")
    table.add_column("HS%", justify="right")

    any_warn = False
    for row in rows[:top_n]:
        label = row.agent if isinstance(row, AgentStats) else row.map_name
        s = row.stats
        flags = reliability_flags(s, is_split=True)
        if not all(flags.values()):
            any_warn = True
        table.add_row(
            label,
            # Mark the G column when *any* split-metric is thin — this is
            # the user's "don't trust this row" canary, keyed off the
            # sample size that drives everything else.
            warn_cell(str(s.matches), all(flags.values())),
            warn_cell(f"{s.wins}-{s.losses}", flags["win_rate"]),
            warn_cell(fmt_pct(s.win_rate), flags["win_rate"]),
            warn_cell(f"{s.acs:.0f}", flags["acs"]),
            warn_cell(f"{s.kd:.2f}", flags["kd"]),
            warn_cell(fmt_pct(s.hs_pct), flags["hs_pct"]),
        )
    console.print(table)
    return any_warn


def render_win_loss_split(
    console: Console,
    wins_rows: list[MatchPlayer],
    losses_rows: list[MatchPlayer],
) -> bool:
    """Win/Loss performance comparison table.

    Shows ACS, K/D, ADR, HS%, and FB diff split by outcome, with a delta
    column so the user immediately sees "I play better/worse when winning".
    Both halves are computed independently as PlayerStats so every metric
    is correctly round-weighted within each subset.

    Returns ``True`` when *either* half is below reliability threshold.

    Skipped (returns ``False``) when either half is empty — a 0-game split
    tells the coach nothing useful.
    """
    from valocoach.stats.calculator import compute_player_stats

    if not wins_rows or not losses_rows:
        return False

    ws = compute_player_stats(wins_rows)
    ls = compute_player_stats(losses_rows)

    # Use split-level reliability (stricter) for each half.
    wf = reliability_flags(ws, is_split=True)
    lf = reliability_flags(ls, is_split=True)
    any_warn = not all(wf.values()) or not all(lf.values())

    table = Table(
        title="Win / Loss split",
        show_header=True,
        header_style="bold",
        pad_edge=False,
    )
    table.add_column("Metric")
    table.add_column(f"Wins ({ws.matches}g)", justify="right", style="green")
    table.add_column(f"Losses ({ls.matches}g)", justify="right", style="red")
    table.add_column("Delta (W\u2212L)", justify="right")

    table.add_row(
        "ACS",
        warn_cell(f"{ws.acs:.0f}", wf["acs"]),
        warn_cell(f"{ls.acs:.0f}", lf["acs"]),
        fmt_delta(ws.acs, ls.acs, fmt=".0f"),
    )
    table.add_row(
        "K/D",
        warn_cell(f"{ws.kd:.2f}", wf["kd"]),
        warn_cell(f"{ls.kd:.2f}", lf["kd"]),
        fmt_delta(ws.kd, ls.kd, fmt=".2f"),
    )
    table.add_row(
        "ADR",
        warn_cell(f"{ws.adr:.0f}", wf["adr"]),
        warn_cell(f"{ls.adr:.0f}", lf["adr"]),
        fmt_delta(ws.adr, ls.adr, fmt=".0f"),
    )
    table.add_row(
        "HS%",
        warn_cell(fmt_pct(ws.hs_pct), wf["hs_pct"]),
        warn_cell(fmt_pct(ls.hs_pct), lf["hs_pct"]),
        # Percentage-point delta — append 'pp' so it's clear it's not a ratio.
        fmt_delta(ws.hs_pct * 100, ls.hs_pct * 100, fmt=".1f") + "pp",
    )
    table.add_row(
        "FB diff",
        f"{ws.fb_diff:+d}",
        f"{ls.fb_diff:+d}",
        f"{ws.fb_diff - ls.fb_diff:+d}",
    )

    console.print(table)
    return any_warn


def render_trend(
    console: Console,
    rows: list[MatchPlayer],
) -> None:
    """Recent-form anomaly block — shown only when anomalies are detected.

    Silent when the row window is too thin for a meaningful comparison or
    when performance is stable. The section never shows a "no changes"
    placeholder — absence means stability.
    """
    comparison = compare_baseline(rows)
    if comparison is None or not comparison.has_anomalies:
        return

    console.print(
        f"[bold]Trend[/bold] [dim](last {comparison.form_matches}g"
        f" vs {comparison.baseline_matches}g baseline)[/dim]"
    )
    for a in comparison.anomalies:
        if a.severity == "significant":
            style = "bold red" if not a.is_improvement else "bold green"
            tag = "  [dim](!!)[/dim]"
        else:
            style = "yellow" if not a.is_improvement else "green"
            tag = "  [dim](!)[/dim]"
        console.print(f"  [{style}]{a.one_liner()}[/{style}]{tag}")
    console.print()


# ---------------------------------------------------------------------------
# Round-level stats renderer  (stats command)
# ---------------------------------------------------------------------------


def render_round_stats(
    console: Console,
    analysis: object,  # RoundAnalysis — imported lazily to avoid circular deps
    matches: int,
) -> bool:
    """KAST%, clutch rate, trade stats, and attack/defense win rates.

    This section is only shown when round-level data is available (the loader
    was called with ``include_rounds=True`` and the DB has rounds for the
    filtered match set). It is a supplement to ``render_overall``, not a
    replacement — the two tables together give a complete picture.

    Returns ``True`` if any cell was tagged ⚠️ (drives the legend footer).
    """
    from valocoach.stats.round_analyzer import (
        RoundAnalysis,
        clutch_stat,
        kast_stat,
        trade_efficiency_stat,
        trade_participation_stat,
    )

    if not isinstance(analysis, RoundAnalysis):
        return False

    ks = kast_stat(analysis, matches)
    cs = clutch_stat(analysis, matches)
    te = trade_efficiency_stat(analysis, matches)
    tp = trade_participation_stat(analysis, matches)

    any_warn = not all([ks.is_reliable, cs.is_reliable, te.is_reliable, tp.is_reliable])

    table = Table(title="Round-level Stats", show_header=False, box=None, pad_edge=False)
    table.add_column("Label", style="dim")
    table.add_column("Value", justify="right")
    table.add_column("Label", style="dim")
    table.add_column("Value", justify="right")

    table.add_row(
        "KAST%",
        warn_cell(f"{ks.value:.1f}%", ks.is_reliable),
        "Clutch rate",
        warn_cell(f"{cs.value:.1f}%", cs.is_reliable),
    )
    table.add_row(
        "Trade eff%",
        warn_cell(f"{te.value:.1f}%", te.is_reliable),
        "Trade part%",
        warn_cell(f"{tp.value:.1f}%", tp.is_reliable),
    )

    # Multi-kill summary — compact, no reliability gate (raw counts).
    mk_parts = []
    if analysis.double_kills:
        mk_parts.append(f"2K×{analysis.double_kills}")
    if analysis.triple_kills:
        mk_parts.append(f"3K×{analysis.triple_kills}")
    if analysis.quadra_kills:
        mk_parts.append(f"4K×{analysis.quadra_kills}")
    if analysis.aces:
        mk_parts.append(f"ACE×{analysis.aces}")
    table.add_row(
        "Multi-kills",
        "  ".join(mk_parts) if mk_parts else "—",
        "Clutch opps",
        str(analysis.clutch_opportunities),
    )

    # Attack / defense win rates — only shown when side data is available.
    # Side tracking requires plant-event data; pre-migration rows may lack it.
    atk_wr = analysis.attack_win_rate
    def_wr = analysis.defense_win_rate
    if atk_wr is not None and def_wr is not None:
        table.add_row(
            "Attack W%",
            fmt_pct(atk_wr),
            "Defense W%",
            fmt_pct(def_wr),
        )

    console.print(table)
    return any_warn


# ---------------------------------------------------------------------------
# profile command renderers
# ---------------------------------------------------------------------------


def render_identity_panel(console: Console, player: Player) -> None:
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


def render_summary_card(
    console: Console,
    rows: list[MatchPlayer],
    *,
    limit: int,
) -> bool:
    """Compact "last N matches" summary. One table, dense numbers.

    Different shape from ``render_overall`` — tighter, meant to fit
    alongside the identity panel rather than dominate the screen.

    Returns ``True`` if any cell was tagged ⚠️; ``False`` on the
    empty-rows branch (nothing to warn about → no legend needed).
    """
    from valocoach.stats.calculator import compute_player_stats

    if not rows:
        console.print("[dim]No matches in the local DB yet.[/dim]")
        return False

    stats = compute_player_stats(rows)
    flags = reliability_flags(stats)
    any_warn = not all(flags.values())
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
        "Record",
        warn_cell(
            f"{stats.wins}-{stats.losses}  ({fmt_pct(stats.win_rate)})",
            flags["win_rate"],
        ),
        "ACS",
        warn_cell(f"{stats.acs:.0f}", flags["acs"]),
    )
    table.add_row(
        "K/D",
        warn_cell(f"{stats.kd:.2f}", flags["kd"]),
        "KDA",
        warn_cell(f"{stats.kda:.2f}", flags["kda"]),
    )
    table.add_row(
        "HS%",
        warn_cell(fmt_pct(stats.hs_pct), flags["hs_pct"]),
        "ADR",
        warn_cell(f"{stats.adr:.0f}", flags["adr"]),
    )
    table.add_row(
        "FB diff",
        # FB diff meaning hinges on the underlying rates being reliable —
        # tag when either rate is thin (same rule as in render_overall).
        warn_cell(f"{stats.fb_diff:+d}", flags["fb_rate"] and flags["fd_rate"]),
        "Rounds",
        str(stats.rounds),
    )
    console.print(table)
    return any_warn

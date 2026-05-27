"""Rich rendering for all CLI output — the single source of truth for display.

Commands own: argument parsing, data loading, filter application, error handling.
This module owns: every Rich table, panel, and formatted value on screen.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from rich import box
from rich.columns import Columns
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

WARN_PREFIX: Final[str] = "! "

def fmt_pct(ratio: float) -> str:
    return f"{ratio * 100:.1f}%"


def fmt_delta(a: float, b: float, *, fmt: str = ".1f") -> str:
    d = a - b
    sign = "+" if d >= 0 else ""
    return f"{sign}{d:{fmt}}"


def warn_cell(value: str, reliable: bool) -> str:
    return value if reliable else f"{WARN_PREFIX}{value}"


# ---------------------------------------------------------------------------
# Shared footer
# ---------------------------------------------------------------------------


def render_warn_legend(console: Console) -> None:
    console.print(
        f"[muted]{WARN_PREFIX.strip()} Metric has a small sample size. "
        "Treat as directional, not definitive.[/muted]"
    )


# ---------------------------------------------------------------------------
# Rank tier color helper
# ---------------------------------------------------------------------------

def _rank_style(tier_patched: str) -> str:
    t = tier_patched.lower()
    if "radiant" in t or "immortal" in t:
        return "val.red"
    if "ascendant" in t or "diamond" in t:
        return "val.blue"
    if "platinum" in t or "gold" in t:
        return "stat.good"
    return "stat.value"


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
    rank_style = _rank_style(tier)
    console.print(
        f"  [heading]{name}#{tag}[/heading]  [muted]·[/muted]  "
        f"[{rank_style}]{tier}[/{rank_style}]  [muted]·[/muted]  "
        f"[muted]{region.upper()}[/muted]  [muted]·[/muted]  "
        f"[stat.value]{matches_shown}[/stat.value] [muted]match(es)[/muted]"
    )


def render_overall(console: Console, stats: PlayerStats) -> bool:
    flags = reliability_flags(stats)
    any_warn = not all(flags.values())

    # Combat stats group
    combat = Table(
        title="[heading]Combat[/heading]",
        show_header=False,
        box=box.SIMPLE,
        pad_edge=True,
        padding=(0, 1),
    )
    combat.add_column("Label", style="stat.label")
    combat.add_column("Value", justify="right", style="stat.value")

    combat.add_row(
        "K / D / A",
        f"{stats.kills} / {stats.deaths} / {stats.assists}",
    )
    combat.add_row("K/D", warn_cell(f"{stats.kd:.2f}", flags["kd"]))
    combat.add_row("KDA", warn_cell(f"{stats.kda:.2f}", flags["kda"]))
    combat.add_row("HS%", warn_cell(fmt_pct(stats.hs_pct), flags["hs_pct"]))
    combat.add_row("ACS", warn_cell(f"{stats.acs:.1f}", flags["acs"]))
    combat.add_row("ADR", warn_cell(f"{stats.adr:.1f}", flags["adr"]))

    # Match stats group
    match_t = Table(
        title="[heading]Match[/heading]",
        show_header=False,
        box=box.SIMPLE,
        pad_edge=True,
        padding=(0, 1),
    )
    match_t.add_column("Label", style="stat.label")
    match_t.add_column("Value", justify="right", style="stat.value")

    match_t.add_row(
        "Record",
        warn_cell(
            f"{stats.wins}-{stats.losses}  ({fmt_pct(stats.win_rate)})",
            flags["win_rate"],
        ),
    )
    match_t.add_row("Matches", str(stats.matches))
    match_t.add_row("Rounds", str(stats.rounds))
    match_t.add_row(
        "FB / FD",
        warn_cell(
            f"{stats.first_bloods} / {stats.first_deaths}  ({stats.fb_diff:+d})",
            flags["fb_rate"] and flags["fd_rate"],
        ),
    )

    console.print(Columns([combat, match_t], padding=(0, 4)))
    return any_warn


def render_breakdown(
    console: Console,
    *,
    title: str,
    group_col: str,
    rows: list[AgentStats] | list[MapStats],
    top_n: int,
) -> bool:
    if not rows:
        return False

    table = Table(
        title=f"[heading]{title}[/heading]",
        show_header=True,
        header_style="bold",
        box=box.HEAVY_HEAD,
        pad_edge=True,
    )
    table.add_column(group_col, style="heading")
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
    from valocoach.stats.calculator import compute_player_stats

    if not wins_rows or not losses_rows:
        return False

    ws = compute_player_stats(wins_rows)
    ls = compute_player_stats(losses_rows)

    wf = reliability_flags(ws, is_split=True)
    lf = reliability_flags(ls, is_split=True)
    any_warn = not all(wf.values()) or not all(lf.values())

    table = Table(
        title="[heading]Win / Loss Split[/heading]",
        show_header=True,
        header_style="bold",
        box=box.HEAVY_HEAD,
        pad_edge=True,
    )
    table.add_column("Metric")
    table.add_column(f"Wins ({ws.matches}g)", justify="right", style="stat.good")
    table.add_column(f"Losses ({ls.matches}g)", justify="right", style="stat.bad")
    table.add_column("Delta (W−L)", justify="right")

    def _delta_style(a: float, b: float) -> str:
        return "stat.good" if a >= b else "stat.bad"

    acs_delta = fmt_delta(ws.acs, ls.acs, fmt=".0f")
    kd_delta = fmt_delta(ws.kd, ls.kd, fmt=".2f")
    adr_delta = fmt_delta(ws.adr, ls.adr, fmt=".0f")
    hs_delta = fmt_delta(ws.hs_pct * 100, ls.hs_pct * 100, fmt=".1f") + "pp"

    table.add_row(
        "ACS",
        warn_cell(f"{ws.acs:.0f}", wf["acs"]),
        warn_cell(f"{ls.acs:.0f}", lf["acs"]),
        f"[{_delta_style(ws.acs, ls.acs)}]{acs_delta}[/{_delta_style(ws.acs, ls.acs)}]",
    )
    table.add_row(
        "K/D",
        warn_cell(f"{ws.kd:.2f}", wf["kd"]),
        warn_cell(f"{ls.kd:.2f}", lf["kd"]),
        f"[{_delta_style(ws.kd, ls.kd)}]{kd_delta}[/{_delta_style(ws.kd, ls.kd)}]",
    )
    table.add_row(
        "ADR",
        warn_cell(f"{ws.adr:.0f}", wf["adr"]),
        warn_cell(f"{ls.adr:.0f}", lf["adr"]),
        f"[{_delta_style(ws.adr, ls.adr)}]{adr_delta}[/{_delta_style(ws.adr, ls.adr)}]",
    )
    table.add_row(
        "HS%",
        warn_cell(fmt_pct(ws.hs_pct), wf["hs_pct"]),
        warn_cell(fmt_pct(ls.hs_pct), lf["hs_pct"]),
        f"[{_delta_style(ws.hs_pct, ls.hs_pct)}]{hs_delta}[/{_delta_style(ws.hs_pct, ls.hs_pct)}]",
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
    comparison = compare_baseline(rows)
    if comparison is None or not comparison.has_anomalies:
        return

    console.print(
        f"  [heading]Trend[/heading] [muted](last {comparison.form_matches}g"
        f" vs {comparison.baseline_matches}g baseline)[/muted]"
    )

    for a in comparison.anomalies:
        if a.severity == "significant":
            style = "stat.bad" if not a.is_improvement else "stat.good"
            badge = "[error] CRIT [/error]"
        else:
            style = "stat.bad" if not a.is_improvement else "stat.good"
            badge = "[warning] WATCH [/warning]"
        console.print(f"  {badge}  [{style}]{a.one_liner()}[/{style}]")
    console.print()


# ---------------------------------------------------------------------------
# Round-level stats renderer  (stats command)
# ---------------------------------------------------------------------------


def render_round_stats(
    console: Console,
    analysis: object,
    matches: int,
) -> bool:
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

    table = Table(
        title="[heading]Round Stats[/heading]",
        show_header=False,
        box=box.SIMPLE,
        pad_edge=True,
    )
    table.add_column("Label", style="stat.label")
    table.add_column("Value", justify="right", style="stat.value")
    table.add_column("Label", style="stat.label")
    table.add_column("Value", justify="right", style="stat.value")

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
    last_match = player.last_match_at or "never"
    rank_style = _rank_style(player.current_tier_patched)

    body = Table(show_header=False, box=None, pad_edge=False, padding=(0, 2))
    body.add_column("Label", style="stat.label", width=12)
    body.add_column("Value", style="stat.value")

    body.add_row(
        "Rank",
        f"[{rank_style}]{player.current_tier_patched}[/{rank_style}]"
        f"  [muted]({player.current_rr} RR · elo {player.elo})[/muted]",
    )
    body.add_row("Peak", f"[{_rank_style(player.peak_tier_patched)}]{player.peak_tier_patched}[/{_rank_style(player.peak_tier_patched)}]")
    body.add_row(
        "Region",
        f"{player.region.upper()}  [muted]· level {player.account_level}[/muted]",
    )
    body.add_row("Last match", f"[muted]{last_match}[/muted]")

    title = f"[heading]{player.riot_name}#{player.riot_tag}[/heading]"
    console.print(Panel(body, title=title, border_style="border", padding=(1, 2)))


def render_lookup_identity_panel(
    console: Console,
    account: object,
    mmr: object,
) -> None:
    """Render identity panel from live API data — mirrors render_identity_panel."""
    from valocoach.data.api_models import AccountResponse, MMRData as ApiMMR

    if not isinstance(account, AccountResponse) or not isinstance(mmr, ApiMMR):
        return

    cd = mmr.current_data
    hr = mmr.highest_rank
    rank_style = _rank_style(cd.currenttierpatched)
    peak_style = _rank_style(hr.patched_tier)

    body = Table(show_header=False, box=None, pad_edge=False, padding=(0, 2))
    body.add_column("Label", style="stat.label", width=12)
    body.add_column("Value", style="stat.value")

    rr_delta = cd.mmr_change_to_last_game
    delta_markup = ""
    if rr_delta and rr_delta > 0:
        delta_markup = f"  [rank.up]{rr_delta:+d} last game[/rank.up]"
    elif rr_delta and rr_delta < 0:
        delta_markup = f"  [rank.down]{rr_delta:+d} last game[/rank.down]"

    body.add_row(
        "Rank",
        f"[{rank_style}]{cd.currenttierpatched}[/{rank_style}]"
        f"  [muted]({cd.ranking_in_tier} RR · elo {cd.elo})[/muted]{delta_markup}",
    )
    peak_label = hr.patched_tier
    if hr.season:
        peak_label += f"  [muted]({hr.season})[/muted]"
    body.add_row("Peak", f"[{peak_style}]{peak_label}[/{peak_style}]")
    body.add_row(
        "Region",
        f"{account.region.upper()}  [muted]· level {account.account_level}[/muted]",
    )
    if cd.old:
        body.add_row("Status", "[warning]Rank decayed[/warning]")

    title = f"[heading]{account.name}#{account.tag}[/heading]"
    console.print(Panel(body, title=title, border_style="border", padding=(1, 2)))


def render_lookup_summary(
    console: Console,
    stats: object,
) -> bool:
    """Render compact summary card — mirrors render_summary_card layout."""
    from valocoach.stats.calculator import PlayerStats as CalcStats

    if not isinstance(stats, CalcStats) or stats.matches == 0:
        console.print("[muted]No recent competitive matches found.[/muted]")
        return False

    flags = reliability_flags(stats)
    any_warn = not all(flags.values())

    table = Table(
        title=f"[heading]Last {stats.matches} Match(es)[/heading]",
        show_header=False,
        box=box.SIMPLE,
        pad_edge=True,
    )
    table.add_column("Label", style="stat.label")
    table.add_column("Value", justify="right", style="stat.value")
    table.add_column("Label", style="stat.label")
    table.add_column("Value", justify="right", style="stat.value")

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
        "K / D / A",
        f"{stats.kills} / {stats.deaths} / {stats.assists}",
        "Rounds",
        str(stats.rounds),
    )
    console.print(table)
    return any_warn


def render_summary_card(
    console: Console,
    rows: list[MatchPlayer],
    *,
    limit: int,
) -> bool:
    from valocoach.stats.calculator import compute_player_stats

    if not rows:
        console.print("[muted]No matches in the local DB yet.[/muted]")
        return False

    stats = compute_player_stats(rows)
    flags = reliability_flags(stats)
    any_warn = not all(flags.values())
    shown = min(len(rows), limit)

    table = Table(
        title=f"[heading]Last {shown} Match(es)[/heading]",
        show_header=False,
        box=box.SIMPLE,
        pad_edge=True,
    )
    table.add_column("Label", style="stat.label")
    table.add_column("Value", justify="right", style="stat.value")
    table.add_column("Label", style="stat.label")
    table.add_column("Value", justify="right", style="stat.value")

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
        warn_cell(f"{stats.fb_diff:+d}", flags["fb_rate"] and flags["fd_rate"]),
        "Rounds",
        str(stats.rounds),
    )
    console.print(table)
    return any_warn


# ---------------------------------------------------------------------------
# Rank trend renderer  (profile command)
# ---------------------------------------------------------------------------


def render_rank_trend(
    console: Console,
    history: list,
) -> None:
    if len(history) < 2:
        return

    newest = history[0]
    oldest = history[-1]

    elo_delta = newest.elo - oldest.elo
    delta_str = (f"+{elo_delta}" if elo_delta >= 0 else str(elo_delta)) + " elo"
    if elo_delta > 0:
        delta_markup = f"[rank.up]{delta_str}[/rank.up]"
        arrow = "[rank.up]↑[/rank.up]"
    elif elo_delta < 0:
        delta_markup = f"[rank.down]{delta_str}[/rank.down]"
        arrow = "[rank.down]↓[/rank.down]"
    else:
        delta_markup = f"[muted]{delta_str}[/muted]"
        arrow = "[muted]→[/muted]"

    spark_chars: list[str] = []
    for h in reversed(history):
        if h.mmr_change is None:
            spark_chars.append("·")
        elif h.mmr_change > 0:
            spark_chars.append("▲")
        elif h.mmr_change < 0:
            spark_chars.append("▼")
        else:
            spark_chars.append("─")
    sparkline = "".join(spark_chars[-15:])

    n = len(history)
    rank_range = (
        f"{oldest.tier_patched}"
        if oldest.tier_patched == newest.tier_patched
        else f"{oldest.tier_patched} → {newest.tier_patched}"
    )

    console.print(
        f"  [heading]Rank Trend[/heading] [muted]({n} snapshot(s))[/muted]  "
        f"{rank_range}  {delta_markup}  {arrow}  [muted]{sparkline}[/muted]"
    )


# ---------------------------------------------------------------------------
# Coaching sessions / notes renderers  (profile command)
# ---------------------------------------------------------------------------

_PRIORITY_ICON: dict[int, str] = {
    1: "[val.red]●[/val.red]",
    2: "[warning]●[/warning]",
    3: "[muted]●[/muted]",
}


def render_coaching_sessions(
    console: Console,
    sessions: list,
) -> None:
    if not sessions:
        return

    table = Table(
        title=f"[heading]Recent Coaching Sessions ({len(sessions)})[/heading]",
        show_header=True,
        header_style="bold",
        box=box.HEAVY_HEAD,
        pad_edge=True,
        show_edge=False,
    )
    table.add_column("#", style="muted", justify="right", width=4)
    table.add_column("Title", min_width=18, max_width=28)
    table.add_column("Started", width=10)
    table.add_column("Status", width=7)
    table.add_column("Focus", min_width=8, max_width=14)

    for s in sessions:
        started = s.started_at[:10] if s.started_at else "—"
        status = "[muted]closed[/muted]" if not s.is_open else "[stat.good]open[/stat.good]"
        focus_parts = [p for p in (s.focus_agent, s.focus_map) if p]
        focus = " · ".join(focus_parts) if focus_parts else "—"
        title = (s.title or "—")[:26]
        table.add_row(str(s.id), title, started, status, focus)

    console.print(table)


def render_open_notes(
    console: Console,
    notes: list,
) -> None:
    if not notes:
        return

    table = Table(
        title=f"[heading]Open Coaching Notes ({len(notes)})[/heading]",
        show_header=True,
        header_style="bold",
        box=box.HEAVY_HEAD,
        pad_edge=True,
        show_edge=False,
    )
    table.add_column("#", style="muted", justify="right", width=4)
    table.add_column("P", justify="center", width=2)
    table.add_column("Category", width=9)
    table.add_column("Note")

    for n in notes:
        icon = _PRIORITY_ICON.get(n.priority, "●")
        cat = n.category[:8]
        body_text = n.body[:72] + ("…" if len(n.body) > 72 else "")
        table.add_row(str(n.id), icon, cat, body_text)

    console.print(table)

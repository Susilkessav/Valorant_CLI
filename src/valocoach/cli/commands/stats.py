"""`valocoach stats` — performance dashboard from locally synced matches.

Flow:
    1. Load settings → validate riot_name/riot_tag are configured.
    2. ensure_db()   → open the SQLite engine, create tables if missing.
    3. Look up the tracked Player row by name+tag.
       Missing Player → nudge the user to run `valocoach sync` first.
    4. Fetch recent MatchPlayer rows for that puuid (up to FETCH_LIMIT).
    5. Apply --period / --agent / --map filters in Python (cheap at this size).
    6. Run the calculator → render three tables.

The DB-facing phase is async; rendering is sync. All DB access sits behind a
single ``session_scope()``; by the time we render, every row is detached from
the session and the connection is back in the pool.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from valocoach.cli import display
from valocoach.core.config import Settings, load_settings
from valocoach.data.database import ensure_db, session_scope
from valocoach.data.orm_models import MatchPlayer
from valocoach.data.repository import get_player_by_name, get_recent_matches
from valocoach.stats import (
    AgentStats,
    MapStats,
    PlayerStats,
    compute_per_agent,
    compute_per_map,
    compute_player_stats,
)

# Fetch plenty of rows — the period/agent/map filters narrow from here in Python.
# 200 covers ~3 months of ranked for a heavy player; raise when we track >1 player.
FETCH_LIMIT = 200

# How many rows to show in the per-agent and per-map tables.
TOP_N = 5


# ---------------------------------------------------------------------------
# Period parsing
# ---------------------------------------------------------------------------


def _period_to_cutoff_iso(period: str) -> str | None:
    """Translate a ``--period`` string into an ISO8601 cutoff timestamp.

    ``'30d'`` → ISO of "30 days ago" (filter rows with started_at >= cutoff).
    ``'all'`` → ``None`` (no filter).

    ISO8601 UTC strings sort lexicographically the same as chronologically,
    so callers can use ``row.started_at >= cutoff`` directly — no parsing.

    Raises:
        typer.BadParameter: on unrecognised input. Surfaces in the CLI as a
            clean error instead of a traceback.
    """
    p = period.strip().lower()
    if p == "all":
        return None
    if not p.endswith("d") or not p[:-1].isdigit():
        raise typer.BadParameter(
            f"--period must be 'Nd' (e.g. 7d, 30d) or 'all'; got {period!r}"
        )
    days = int(p[:-1])
    if days <= 0:
        raise typer.BadParameter(f"--period must be positive; got {period!r}")
    cutoff = datetime.now(UTC) - timedelta(days=days)
    return cutoff.isoformat()


# ---------------------------------------------------------------------------
# Row filtering  (pure — tested directly)
# ---------------------------------------------------------------------------


def _filter_rows(
    rows: list[MatchPlayer],
    *,
    cutoff_iso: str | None,
    agent: str | None,
    map_name: str | None,
) -> list[MatchPlayer]:
    """Apply --period / --agent / --map filters to the row set.

    All comparisons are case-insensitive on agent/map because CLI users
    will type "jett" and the DB has "Jett".
    """
    result = rows
    if cutoff_iso is not None:
        result = [mp for mp in result if mp.started_at >= cutoff_iso]
    if agent is not None:
        agent_lc = agent.lower()
        result = [mp for mp in result if mp.agent_name.lower() == agent_lc]
    if map_name is not None:
        map_lc = map_name.lower()
        result = [
            mp for mp in result if mp.match is not None and mp.match.map_name.lower() == map_lc
        ]
    return result


# ---------------------------------------------------------------------------
# Rendering  (pure — takes computed stats, writes to a Console)
# ---------------------------------------------------------------------------


def _fmt_pct(ratio: float) -> str:
    """0.2734 → '27.3%'. Rate fields live as 0.0-1.0 ratios in PlayerStats."""
    return f"{ratio * 100:.1f}%"


def _render_header(
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
) -> None:
    """Top panel: who we're showing, over what window."""
    filter_bits = [f"period={period}"]
    if agent_filter:
        filter_bits.append(f"agent={agent_filter}")
    if map_filter:
        filter_bits.append(f"map={map_filter}")

    title = f"[bold]{name}#{tag}[/bold]  [dim]·[/dim]  {tier}  [dim]·[/dim]  {region.upper()}"
    subtitle = f"{matches_shown} match(es) after filters · " + " · ".join(filter_bits)
    console.print(Panel(subtitle, title=title, border_style="cyan", padding=(0, 2)))


def _render_overall(console: Console, stats: PlayerStats) -> None:
    """Overall stats — two-column layout of the numbers that matter most."""
    table = Table(title="Overall", show_header=False, box=None, pad_edge=False)
    table.add_column("Label", style="dim")
    table.add_column("Value", justify="right")
    table.add_column("Label", style="dim")
    table.add_column("Value", justify="right")

    table.add_row(
        "Matches", str(stats.matches),
        "Win rate", f"{stats.wins}-{stats.losses}  ({_fmt_pct(stats.win_rate)})",
    )
    table.add_row(
        "Rounds", str(stats.rounds),
        "ACS", f"{stats.acs:.1f}",
    )
    table.add_row(
        "K / D / A",
        f"{stats.kills} / {stats.deaths} / {stats.assists}",
        "ADR", f"{stats.adr:.1f}",
    )
    table.add_row(
        "K/D", f"{stats.kd:.2f}",
        "KDA", f"{stats.kda:.2f}",
    )
    table.add_row(
        "HS%", _fmt_pct(stats.hs_pct),
        "FB / FD (diff)", f"{stats.first_bloods} / {stats.first_deaths}  ({stats.fb_diff:+d})",
    )
    console.print(table)


def _render_breakdown(
    console: Console,
    *,
    title: str,
    group_col: str,
    rows: list[AgentStats] | list[MapStats],
    top_n: int,
) -> None:
    """Per-agent or per-map table, top_n rows by matches."""
    if not rows:
        return

    table = Table(title=title, show_header=True, header_style="bold", pad_edge=False)
    table.add_column(group_col, style="cyan")
    table.add_column("G", justify="right")
    table.add_column("W-L", justify="right")
    table.add_column("Win%", justify="right")
    table.add_column("ACS", justify="right")
    table.add_column("K/D", justify="right")
    table.add_column("HS%", justify="right")

    for row in rows[:top_n]:
        label = row.agent if isinstance(row, AgentStats) else row.map_name
        s = row.stats
        table.add_row(
            label,
            str(s.matches),
            f"{s.wins}-{s.losses}",
            _fmt_pct(s.win_rate),
            f"{s.acs:.0f}",
            f"{s.kd:.2f}",
            _fmt_pct(s.hs_pct),
        )
    console.print(table)


# ---------------------------------------------------------------------------
# Async worker
# ---------------------------------------------------------------------------


async def _fetch_stats_data(
    settings: Settings,
) -> tuple[object, list[MatchPlayer]] | None:
    """Resolve player + fetch recent MatchPlayer rows.

    Returns:
        (player, rows) on success.
        None if the player has never been synced — caller prints guidance.
    """
    await ensure_db(settings.data_dir / "valocoach.db")

    async with session_scope() as session:
        player = await get_player_by_name(session, settings.riot_name, settings.riot_tag)
        if player is None:
            return None
        rows = await get_recent_matches(session, player.puuid, limit=FETCH_LIMIT)

    return player, rows


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run_stats(
    *,
    agent: str | None = None,
    map_: str | None = None,
    period: str = "30d",
    console: Console | None = None,
) -> None:
    """CLI entry: ``valocoach stats`` dispatches here.

    ``console`` is injectable for testing — defaults to the shared themed
    console so output matches the rest of the CLI.
    """
    con = console or display.console

    # Parse the period string before we do anything expensive — fail fast on bad input.
    cutoff_iso = _period_to_cutoff_iso(period)

    settings = load_settings()
    if not settings.riot_name or not settings.riot_tag:
        display.error(
            "riot_name / riot_tag not configured — run `valocoach config init` and edit the file"
        )
        raise typer.Exit(1)

    fetched = asyncio.run(_fetch_stats_data(settings))
    if fetched is None:
        display.warn(
            f"No local data for {settings.riot_name}#{settings.riot_tag}. "
            "Run `valocoach sync` first to pull match history."
        )
        raise typer.Exit(1)

    player, rows = fetched

    filtered = _filter_rows(rows, cutoff_iso=cutoff_iso, agent=agent, map_name=map_)
    if not filtered:
        display.warn(
            f"No matches after filters (period={period}"
            + (f", agent={agent}" if agent else "")
            + (f", map={map_}" if map_ else "")
            + ")."
        )
        raise typer.Exit(0)

    overall = compute_player_stats(filtered)
    per_agent = compute_per_agent(filtered)
    per_map = compute_per_map(filtered)

    _render_header(
        con,
        name=player.riot_name,
        tag=player.riot_tag,
        tier=player.current_tier_patched,
        region=player.region,
        matches_shown=overall.matches,
        period=period,
        agent_filter=agent,
        map_filter=map_,
    )
    _render_overall(con, overall)
    con.print()
    # Skip per-agent breakdown when the user already filtered to one agent —
    # the single-row table would be redundant with the overall card.
    if agent is None:
        _render_breakdown(
            con, title="By agent", group_col="Agent", rows=per_agent, top_n=TOP_N
        )
        con.print()
    if map_ is None:
        _render_breakdown(con, title="By map", group_col="Map", rows=per_map, top_n=TOP_N)

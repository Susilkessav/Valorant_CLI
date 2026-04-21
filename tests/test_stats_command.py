"""Tests for `valocoach stats` — period parsing, row filtering, rendering.

The DB-facing layer is already tested by test_repository.py; here we focus
on the CLI glue: parsing, pure filtering, and that the renderer produces
output containing the key numbers.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from io import StringIO

import pytest
import typer
from rich.console import Console

from valocoach.cli.commands.stats import (
    _filter_rows,
    _period_to_cutoff_iso,
    _render_breakdown,
    _render_header,
    _render_overall,
)
from valocoach.data.orm_models import Match, MatchPlayer
from valocoach.stats import (
    compute_per_agent,
    compute_per_map,
    compute_player_stats,
)

# ---------------------------------------------------------------------------
# Helpers (shared with test_stats_calculator.py in spirit — kept local for
# clarity; the helper is small and re-declaring avoids a cross-test import.)
# ---------------------------------------------------------------------------


def _mp(
    *,
    agent: str = "Jett",
    map_name: str = "Ascent",
    started_at: str = "2026-04-19T18:00:00+00:00",
    won: bool = True,
    rounds_played: int = 20,
    score: int = 5000,
    kills: int = 20,
    deaths: int = 10,
    assists: int = 5,
    headshots: int = 30,
    bodyshots: int = 60,
    legshots: int = 10,
    damage_dealt: int = 3000,
    first_bloods: int = 3,
    first_deaths: int = 1,
    match_id: str = "m-1",
) -> MatchPlayer:
    match = Match(
        match_id=match_id,
        map_name=map_name,
        queue_id="competitive",
        is_ranked=True,
        game_length_secs=0,
        rounds_played=rounds_played,
        red_score=0,
        blue_score=0,
        started_at=started_at,
    )
    mp = MatchPlayer(
        match_id=match_id,
        puuid="p-tracked",
        agent_name=agent,
        team="Blue",
        won=won,
        score=score,
        kills=kills,
        deaths=deaths,
        assists=assists,
        rounds_played=rounds_played,
        headshots=headshots,
        bodyshots=bodyshots,
        legshots=legshots,
        damage_dealt=damage_dealt,
        damage_received=0,
        first_bloods=first_bloods,
        first_deaths=first_deaths,
        plants=0,
        defuses=0,
        afk_rounds=0,
        rounds_in_spawn=0,
        started_at=started_at,
    )
    mp.match = match
    return mp


def _capture_console() -> Console:
    """A Console that writes to a string buffer, no ANSI, stable width."""
    return Console(file=StringIO(), force_terminal=False, width=120)


# ---------------------------------------------------------------------------
# _period_to_cutoff_iso
# ---------------------------------------------------------------------------


def test_period_all_returns_none() -> None:
    assert _period_to_cutoff_iso("all") is None
    assert _period_to_cutoff_iso("ALL") is None


def test_period_days_returns_iso_in_past() -> None:
    before = datetime.now(UTC) - timedelta(days=7)
    cutoff_iso = _period_to_cutoff_iso("7d")
    after = datetime.now(UTC) - timedelta(days=7)
    assert cutoff_iso is not None
    parsed = datetime.fromisoformat(cutoff_iso)
    # Cutoff should land in the interval we measured around it.
    assert before - timedelta(seconds=1) <= parsed <= after + timedelta(seconds=1)


def test_period_days_arithmetic() -> None:
    """30d cutoff should be ~23 days earlier than 7d cutoff."""
    iso_30 = datetime.fromisoformat(_period_to_cutoff_iso("30d"))
    iso_7 = datetime.fromisoformat(_period_to_cutoff_iso("7d"))
    delta_days = (iso_7 - iso_30).days
    assert delta_days == 23  # 30 - 7


@pytest.mark.parametrize("bad", ["abc", "30", "7days", "-5d", "0d", "", "d"])
def test_period_bad_input_raises_bad_parameter(bad: str) -> None:
    with pytest.raises(typer.BadParameter):
        _period_to_cutoff_iso(bad)


# ---------------------------------------------------------------------------
# _filter_rows
# ---------------------------------------------------------------------------


def test_filter_by_period_drops_old_rows() -> None:
    rows = [
        _mp(started_at="2026-04-19T18:00:00+00:00", match_id="new"),
        _mp(started_at="2025-01-01T00:00:00+00:00", match_id="old"),
    ]
    cutoff = "2026-01-01T00:00:00+00:00"
    out = _filter_rows(rows, cutoff_iso=cutoff, agent=None, map_name=None)
    assert [mp.match_id for mp in out] == ["new"]


def test_filter_by_agent_is_case_insensitive() -> None:
    rows = [
        _mp(agent="Jett", match_id="a"),
        _mp(agent="Reyna", match_id="b"),
        _mp(agent="jett", match_id="c"),  # defensive — shouldn't happen in real data
    ]
    out = _filter_rows(rows, cutoff_iso=None, agent="JETT", map_name=None)
    assert {mp.match_id for mp in out} == {"a", "c"}


def test_filter_by_map_is_case_insensitive() -> None:
    rows = [
        _mp(map_name="Ascent", match_id="a"),
        _mp(map_name="Lotus", match_id="b"),
    ]
    out = _filter_rows(rows, cutoff_iso=None, agent=None, map_name="ascent")
    assert [mp.match_id for mp in out] == ["a"]


def test_filter_combines_all_three() -> None:
    rows = [
        _mp(agent="Jett", map_name="Ascent", started_at="2026-04-01T00:00:00+00:00", match_id="keep"),
        _mp(agent="Jett", map_name="Lotus", started_at="2026-04-01T00:00:00+00:00", match_id="wrong_map"),
        _mp(agent="Reyna", map_name="Ascent", started_at="2026-04-01T00:00:00+00:00", match_id="wrong_agent"),
        _mp(agent="Jett", map_name="Ascent", started_at="2020-01-01T00:00:00+00:00", match_id="too_old"),
    ]
    out = _filter_rows(
        rows,
        cutoff_iso="2026-01-01T00:00:00+00:00",
        agent="Jett",
        map_name="Ascent",
    )
    assert [mp.match_id for mp in out] == ["keep"]


def test_filter_no_args_is_identity() -> None:
    rows = [_mp(match_id="a"), _mp(match_id="b")]
    out = _filter_rows(rows, cutoff_iso=None, agent=None, map_name=None)
    assert out == rows  # same list instance semantics — equal contents


# ---------------------------------------------------------------------------
# Rendering smoke tests
# ---------------------------------------------------------------------------


def test_render_overall_includes_key_numbers() -> None:
    """Overall render must surface matches, win rate, ACS, K/D."""
    rows = [_mp()]
    stats = compute_player_stats(rows)
    con = _capture_console()
    _render_overall(con, stats)
    out = con.file.getvalue()
    assert "Matches" in out
    assert "1" in out  # match count
    assert "ACS" in out
    assert "250" in out  # 5000/20
    assert "K/D" in out


def test_render_breakdown_top_n_limits_rows() -> None:
    rows = [_mp(agent=f"Agent{i}", match_id=f"m-{i}") for i in range(10)]
    per_agent = compute_per_agent(rows)
    con = _capture_console()
    _render_breakdown(
        con, title="By agent", group_col="Agent", rows=per_agent, top_n=3
    )
    out = con.file.getvalue()
    # Only 3 agents rendered
    assert out.count("Agent0") == 1
    assert out.count("Agent1") == 1
    assert out.count("Agent2") == 1
    # Agent9 shouldn't appear in the limited table
    assert "Agent9" not in out


def test_render_breakdown_empty_is_noop() -> None:
    con = _capture_console()
    _render_breakdown(con, title="By map", group_col="Map", rows=[], top_n=5)
    assert con.file.getvalue() == ""


def test_render_header_shows_filters() -> None:
    con = _capture_console()
    _render_header(
        con,
        name="Yoursaviour01",
        tag="SK04",
        tier="Gold 1",
        region="na",
        matches_shown=12,
        period="30d",
        agent_filter="Jett",
        map_filter=None,
    )
    out = con.file.getvalue()
    assert "Yoursaviour01" in out
    assert "SK04" in out
    assert "Gold 1" in out
    assert "NA" in out
    assert "period=30d" in out
    assert "agent=Jett" in out


def test_render_per_map_labels_map_column() -> None:
    rows = [_mp(map_name="Ascent", match_id="a"), _mp(map_name="Lotus", match_id="b")]
    per_map = compute_per_map(rows)
    con = _capture_console()
    _render_breakdown(con, title="By map", group_col="Map", rows=per_map, top_n=5)
    out = con.file.getvalue()
    assert "Ascent" in out
    assert "Lotus" in out

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
    _period_to_cutoff_iso,
    run_stats,
)
from valocoach.cli.formatter import (
    WARN_PREFIX,
)
from valocoach.cli.formatter import (
    render_breakdown as _render_breakdown,
)
from valocoach.cli.formatter import (
    render_header as _render_header,
)
from valocoach.cli.formatter import (
    render_overall as _render_overall,
)
from valocoach.data.loader import PlayerData
from valocoach.data.orm_models import Match, MatchPlayer
from valocoach.stats import (
    compute_per_agent,
    compute_per_map,
    compute_player_stats,
)
from valocoach.stats.filters import (
    filter_by_agent,
    filter_by_map,
    filter_by_period,
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
# Row filtering (via public filter_by_* from valocoach.stats.filters)
# These tests live here because they exercise the same data the stats
# command uses; comprehensive filter-module tests are in test_filters.py.
# ---------------------------------------------------------------------------


def test_filter_by_period_drops_old_rows() -> None:
    rows = [
        _mp(started_at="2026-04-19T18:00:00+00:00", match_id="new"),
        _mp(started_at="2025-01-01T00:00:00+00:00", match_id="old"),
    ]
    out = filter_by_period(rows, "2026-01-01T00:00:00+00:00")
    assert [mp.match_id for mp in out] == ["new"]


def test_filter_by_agent_is_case_insensitive() -> None:
    rows = [
        _mp(agent="Jett", match_id="a"),
        _mp(agent="Reyna", match_id="b"),
        _mp(agent="jett", match_id="c"),  # defensive — shouldn't happen in real data
    ]
    out = filter_by_agent(rows, "JETT")
    assert {mp.match_id for mp in out} == {"a", "c"}


def test_filter_by_map_is_case_insensitive() -> None:
    rows = [
        _mp(map_name="Ascent", match_id="a"),
        _mp(map_name="Lotus", match_id="b"),
    ]
    out = filter_by_map(rows, "ascent")
    assert [mp.match_id for mp in out] == ["a"]


def test_filter_combines_period_agent_map() -> None:
    rows = [
        _mp(
            agent="Jett", map_name="Ascent", started_at="2026-04-01T00:00:00+00:00", match_id="keep"
        ),
        _mp(
            agent="Jett",
            map_name="Lotus",
            started_at="2026-04-01T00:00:00+00:00",
            match_id="wrong_map",
        ),
        _mp(
            agent="Reyna",
            map_name="Ascent",
            started_at="2026-04-01T00:00:00+00:00",
            match_id="wrong_agent",
        ),
        _mp(
            agent="Jett",
            map_name="Ascent",
            started_at="2020-01-01T00:00:00+00:00",
            match_id="too_old",
        ),
    ]
    out = filter_by_period(rows, "2026-01-01T00:00:00+00:00")
    out = filter_by_agent(out, "Jett")
    out = filter_by_map(out, "Ascent")
    assert [mp.match_id for mp in out] == ["keep"]


def test_filter_no_cutoff_is_identity() -> None:
    rows = [_mp(match_id="a"), _mp(match_id="b")]
    out = filter_by_period(rows, None)
    assert out == rows


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
    _render_breakdown(con, title="By agent", group_col="Agent", rows=per_agent, top_n=3)
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


# ---------------------------------------------------------------------------
# Reliability warnings (⚠️) — Phase B
# ---------------------------------------------------------------------------
#
# The calculator already has threshold tests; these tests verify the CLI
# renderers actually *surface* that signal instead of silently dropping it
# on the floor. That's the failure mode we're guarding against — stats
# being technically available but visually indistinguishable from reliable
# ones.


def _make_rows(n: int, **kwargs: object) -> list[MatchPlayer]:
    """n matches with distinct match_ids. Keeps match_id unique so the
    ORM relationship set in _mp() doesn't collapse rows."""
    return [_mp(match_id=f"m-{i}", **kwargs) for i in range(n)]  # type: ignore[arg-type]


def test_render_overall_warns_on_thin_sample() -> None:
    """Under the smallest threshold (MIN_MATCHES_ACS=15), every derived
    metric should render with ⚠️ and the function returns True so the
    caller can print the legend."""
    stats = compute_player_stats(_make_rows(5))
    con = _capture_console()
    any_warn = _render_overall(con, stats)
    out = con.file.getvalue()
    assert any_warn is True
    assert WARN_PREFIX in out
    # Spot-check: the ACS cell should be tagged. Exact column layout is
    # Rich's business; we just assert the glyph co-occurs with the value.
    assert "⚠️" in out and "ACS" in out


def test_render_overall_does_not_warn_on_thick_sample() -> None:
    """At 30 matches every metric clears its threshold → no ⚠️ anywhere."""
    stats = compute_player_stats(_make_rows(30))
    con = _capture_console()
    any_warn = _render_overall(con, stats)
    out = con.file.getvalue()
    assert any_warn is False
    assert WARN_PREFIX not in out


def test_render_breakdown_warns_on_thin_split_even_when_overall_is_fat() -> None:
    """A player can have 30 overall matches (overall reliable) but only
    3 games on a specific agent — that per-agent row MUST still ⚠️.
    Guards against the bug where we'd use the overall flag for splits."""
    # 27 Jett games + 3 Reyna games → 30 overall, reliable;
    # but Reyna split has 3 games, below every threshold.
    rows = _make_rows(27, agent="Jett") + [_mp(agent="Reyna", match_id=f"r-{i}") for i in range(3)]
    per_agent = compute_per_agent(rows)
    con = _capture_console()
    any_warn = _render_breakdown(con, title="By agent", group_col="Agent", rows=per_agent, top_n=5)
    out = con.file.getvalue()
    assert any_warn is True
    # Reyna row is thin — must carry ⚠️. Use the row-level canary: the G cell.
    # We locate Reyna's line in the output and assert the warn glyph on it.
    reyna_line = next(line for line in out.splitlines() if "Reyna" in line)
    assert "⚠️" in reyna_line, f"Reyna row missing warn glyph: {reyna_line!r}"


def test_render_breakdown_split_needs_30_for_win_rate() -> None:
    """Split win rate uses the stricter 30-match floor. At exactly 30
    (`MIN_MATCHES_WIN_RATE_SPLIT`), the row clears all flags → no ⚠️."""
    rows = _make_rows(30, agent="Jett")
    per_agent = compute_per_agent(rows)
    con = _capture_console()
    any_warn = _render_breakdown(con, title="By agent", group_col="Agent", rows=per_agent, top_n=5)
    assert any_warn is False
    assert WARN_PREFIX not in con.file.getvalue()


def test_run_stats_shows_legend_only_when_warnings_fired(tmp_path, monkeypatch) -> None:
    """Integration: legend text appears iff any ⚠️ was rendered. Two
    cases from one fixture — thin (warns, legend) and thick (no warn,
    no legend) — proves the footer wiring tracks the render calls."""
    from unittest.mock import patch

    # Thin path: 5 matches → warnings fire → legend appears.
    thin_rows = _make_rows(5)
    # Thick path: 30 matches → no warnings → no legend.
    thick_rows = _make_rows(30)

    # Minimal stub Player with the attributes _render_header reads.
    class _P:
        riot_name = "Tester"
        riot_tag = "NA1"
        current_tier_patched = "Gold 1"
        region = "na"
        puuid = "p-tracked"

    from valocoach.core.config import Settings

    fake_settings = Settings(
        riot_name="Tester",
        riot_tag="NA1",
        riot_region="na",
        henrikdev_api_key="fake",
        data_dir=tmp_path,
    )

    def _fetch_thin(_settings, **_kw):
        return PlayerData(player=_P(), rows=thin_rows, full_matches=[])

    def _fetch_thick(_settings, **_kw):
        return PlayerData(player=_P(), rows=thick_rows, full_matches=[])

    for fetch, rows, expect_legend in [
        (_fetch_thin, thin_rows, True),
        (_fetch_thick, thick_rows, False),
    ]:
        con = _capture_console()
        with (
            patch("valocoach.cli.commands.stats.load_settings", return_value=fake_settings),
            patch("valocoach.cli.commands.stats.load_player_data", side_effect=fetch),
        ):
            run_stats(period="all", console=con)
        out = con.file.getvalue()
        # Footer is keyed on a stable phrase, not the full sentence, to
        # avoid brittle wording coupling.
        has_legend = "sample-size threshold" in out
        assert has_legend is expect_legend, (
            f"legend expectation mismatch (rows={len(rows)}): "
            f"got has_legend={has_legend}, expected {expect_legend}"
        )


def test_legend_mentions_the_warn_glyph() -> None:
    """The footer must contain the same ⚠️ glyph we use to tag cells,
    otherwise the user sees the warning on rows but no explanation of
    what it means."""
    stats = compute_player_stats(_make_rows(3))  # forces warnings
    con = _capture_console()

    # Drive the full render path by calling run_stats with mocked IO.
    from unittest.mock import patch

    from valocoach.core.config import Settings

    class _P:
        riot_name = "T"
        riot_tag = "X"
        current_tier_patched = "Iron 1"
        region = "na"
        puuid = "p"

    fake_settings = Settings(riot_name="T", riot_tag="X", riot_region="na", henrikdev_api_key="f")

    def _fetch(_s, **_kw):
        return PlayerData(player=_P(), rows=_make_rows(3), full_matches=[])

    with (
        patch("valocoach.cli.commands.stats.load_settings", return_value=fake_settings),
        patch("valocoach.cli.commands.stats.load_player_data", side_effect=_fetch),
    ):
        run_stats(period="all", console=con)
    out = con.file.getvalue()
    assert WARN_PREFIX.strip() in out  # glyph appears in body
    # And the legend line carries it too, so the user can map cell→meaning.
    legend_lines = [line for line in out.splitlines() if "sample-size threshold" in line]
    assert legend_lines, "no legend found"
    assert any(WARN_PREFIX.strip() in line for line in legend_lines)
    # Ensure `stats` imported above is actually used (keeps ruff happy on
    # this explicitly-structured test without suppressing a lint).
    assert stats.matches == 3


# ---------------------------------------------------------------------------
# run_stats — previously-uncovered branches
# ---------------------------------------------------------------------------

# Shared infra (mirrors the block in test_run_stats_shows_legend_only_…)
_FAKE_SETTINGS_PATCH = "valocoach.cli.commands.stats.load_settings"
_FAKE_DATA_PATCH = "valocoach.cli.commands.stats.load_player_data"


class _FakePlayer:
    riot_name = "Tester"
    riot_tag = "NA1"
    current_tier_patched = "Gold 1"
    region = "na"
    puuid = "p-tracked"


def _fake_settings(tmp_path=None) -> object:
    from valocoach.core.config import Settings

    return Settings(
        riot_name="Tester",
        riot_tag="NA1",
        riot_region="na",
        henrikdev_api_key="fake",
    )


def _run(
    *,
    rows: list,
    full_matches: list | None = None,
    result: str | None = None,
    agent: str | None = None,
    map_: str | None = None,
    period: str = "all",
) -> str:
    """Helper: run_stats with mocked settings+data; return captured output."""
    import contextlib
    from unittest.mock import patch

    import click

    from valocoach.data.loader import PlayerData

    con = _capture_console()

    def _fetch(_s, **_kw):
        return PlayerData(player=_FakePlayer(), rows=rows, full_matches=full_matches or [])

    with (
        patch(_FAKE_SETTINGS_PATCH, return_value=_fake_settings()),
        patch(_FAKE_DATA_PATCH, side_effect=_fetch),
        contextlib.suppress(SystemExit, click.exceptions.Exit),
    ):
        run_stats(period=period, result=result, agent=agent, map_=map_, console=con)

    return con.file.getvalue()


def test_run_stats_result_win_covers_won_true_branch():
    """result='win' assigns won=True (line 97) and skips the win/loss split."""
    rows = _make_rows(10, won=True) + _make_rows(5, won=False)
    out = _run(rows=rows, result="win")
    # Only wins in the output — can assert we at least got output without error.
    assert "Tester" in out
    # The split section is skipped when won is not None — "Win" and "Loss"
    # sub-headers should NOT both appear (no side-by-side split).
    # (A crude check: the split renders two W% rows; one would still appear
    # in the overall. We just verify no crash + the command produced output.)
    assert len(out) > 0


def test_run_stats_result_loss_covers_won_false_branch():
    """result='loss' assigns won=False (line 99)."""
    # Rows must have won=False so the loss filter keeps them (not empties them).
    rows = _make_rows(10, won=False)
    out = _run(rows=rows, result="loss")
    assert "Tester" in out


def test_run_stats_result_lose_alias():
    """'lose' is an accepted alias for 'loss' (also line 99)."""
    rows = _make_rows(10, won=False)
    out = _run(rows=rows, result="lose")
    assert "Tester" in out


def test_run_stats_no_matches_after_filter_exits():
    """Lines 139-147: warn + Exit(0) when every row is filtered out."""
    from unittest.mock import MagicMock, patch

    import click

    from valocoach.data.loader import PlayerData

    con = _capture_console()
    warn_mock = MagicMock()

    def _fetch(_s, **_kw):
        return PlayerData(player=_FakePlayer(), rows=_make_rows(5), full_matches=[])

    exited = False
    with (
        patch(_FAKE_SETTINGS_PATCH, return_value=_fake_settings()),
        patch(_FAKE_DATA_PATCH, side_effect=_fetch),
        patch("valocoach.cli.commands.stats.display.warn", warn_mock),
    ):
        try:
            # Agent filter "Killjoy" matches nothing — empty result triggers warn.
            run_stats(period="all", agent="Killjoy", console=con)
        except (SystemExit, click.exceptions.Exit) as exc:
            exited = True
            assert getattr(exc, "code", 0) == 0 or getattr(exc, "exit_code", 0) == 0

    assert exited, "Expected Exit to be raised"
    warn_mock.assert_called_once()
    assert "No matches after filters" in warn_mock.call_args[0][0]


def test_run_stats_agent_filter_skips_per_agent_table():
    """Lines 180→185: per-agent breakdown omitted when agent filter active."""
    rows = _make_rows(20, agent="Jett")
    out = _run(rows=rows, agent="Jett")
    # With agent filter active, the "By agent" table must NOT appear.
    assert "By agent" not in out
    # But the per-map table should still appear (map_ is None).
    assert "By map" in out


def test_run_stats_map_filter_skips_per_map_table():
    """Lines 185→194: per-map breakdown omitted when map filter active."""
    rows = _make_rows(20, map_name="Ascent")
    out = _run(rows=rows, map_="Ascent")
    # With map filter active, "By map" table must NOT appear.
    assert "By map" not in out
    # Per-agent table should still appear (agent is None).
    assert "By agent" in out


def test_run_stats_round_stats_rendered_when_full_matches_available():
    """Lines 195-198: round analysis section shown when full_matches has data."""
    from unittest.mock import patch

    from valocoach.data.loader import PlayerData
    from valocoach.stats.round_analyzer import RoundAnalysis

    rows = _make_rows(20)
    # Build a stub RoundAnalysis with rounds > 0 so render_round_stats fires.
    stub_analysis = RoundAnalysis(
        rounds=40,
        deaths=10,
        teammate_deaths=15,
        clutch_opportunities=3,
        rounds_with_kill=20,
        rounds_with_assist=5,
        rounds_survived=25,
        rounds_traded_death=4,
        rounds_kast=30,
        clutches_won=1,
        traded_deaths=4,
        trades_given=6,
        double_kills=3,
        triple_kills=1,
        quadra_kills=0,
        aces=0,
    )

    con = _capture_console()

    def _fetch(_s, **_kw):
        # full_matches must be non-empty to enter the if-block.
        fake_match = object()  # analyze_rounds is mocked — shape doesn't matter.
        return PlayerData(player=_FakePlayer(), rows=rows, full_matches=[fake_match])

    with (
        patch(_FAKE_SETTINGS_PATCH, return_value=_fake_settings()),
        patch(_FAKE_DATA_PATCH, side_effect=_fetch),
        patch("valocoach.cli.commands.stats.analyze_rounds", return_value=stub_analysis),
    ):
        run_stats(period="all", console=con)

    out = con.file.getvalue()
    # render_round_stats should have added KAST or round-related content.
    assert "KAST" in out or "Round" in out or "Clutch" in out

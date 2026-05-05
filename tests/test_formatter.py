"""Tests for valocoach.cli.formatter.

Coverage strategy
-----------------
Formatting primitives (fmt_pct, fmt_delta, warn_cell) are tested as pure
functions — no Console needed, assertions on return values.

Rich render functions are tested by injecting a capture Console (StringIO
buffer, no ANSI, stable width) and asserting on the text output.  Each test
checks the *contract* — a number that must appear, a glyph that must be
absent — rather than the exact table layout, which is Rich's business.

Return-value contract:
    Every render function that can show ⚠️ returns bool (True = at least one
    warn cell was emitted).  Tests assert this separately from the string
    content so the contract is explicit.
"""

from __future__ import annotations

from io import StringIO

from rich.console import Console

from unittest.mock import patch

from valocoach.cli.formatter import (
    WARN_PREFIX,
    fmt_delta,
    fmt_pct,
    render_breakdown,
    render_header,
    render_identity_panel,
    render_overall,
    render_round_stats,
    render_summary_card,
    render_trend,
    render_warn_legend,
    render_win_loss_split,
    warn_cell,
)
from valocoach.data.orm_models import Match, MatchPlayer, Player
from valocoach.stats import compute_per_agent, compute_per_map, compute_player_stats
from valocoach.stats.round_analyzer import RoundAnalysis

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _con() -> Console:
    """Capture console — no ANSI escapes, stable width, writes to StringIO."""
    return Console(file=StringIO(), force_terminal=False, width=120)


def _out(console: Console) -> str:
    return console.file.getvalue()  # type: ignore[union-attr]


def _mp(
    *,
    agent: str = "Jett",
    map_name: str = "Ascent",
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
        started_at="2026-04-19T18:00:00+00:00",
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
        started_at="2026-04-19T18:00:00+00:00",
    )
    mp.match = match
    return mp


def _make_rows(n: int, **kwargs: object) -> list[MatchPlayer]:
    return [_mp(match_id=f"m-{i}", **kwargs) for i in range(n)]  # type: ignore[arg-type]


def _player(
    *,
    name: str = "Yoursaviour01",
    tag: str = "SK04",
    region: str = "na",
    level: int = 240,
    current_tier_patched: str = "Gold 1",
    current_rr: int = 25,
    elo: int = 925,
    peak_tier_patched: str = "Gold 3",
    last_match_at: str | None = "2026-04-18T18:00:00+00:00",
) -> Player:
    return Player(
        puuid="p-tracked",
        riot_name=name,
        riot_tag=tag,
        region=region,
        platform="pc",
        account_level=level,
        current_tier=12,
        current_tier_patched=current_tier_patched,
        current_rr=current_rr,
        elo=elo,
        peak_tier=14,
        peak_tier_patched=peak_tier_patched,
        last_match_at=last_match_at,
    )


# ---------------------------------------------------------------------------
# Formatting primitives
# ---------------------------------------------------------------------------


class TestFmtPct:
    def test_typical_value(self) -> None:
        assert fmt_pct(0.2734) == "27.3%"

    def test_zero(self) -> None:
        assert fmt_pct(0.0) == "0.0%"

    def test_one(self) -> None:
        assert fmt_pct(1.0) == "100.0%"

    def test_half(self) -> None:
        assert fmt_pct(0.5) == "50.0%"


class TestFmtDelta:
    def test_positive_delta_has_plus_sign(self) -> None:
        assert fmt_delta(210.0, 170.0, fmt=".0f") == "+40"

    def test_negative_delta_has_minus_sign(self) -> None:
        assert fmt_delta(0.95, 1.10, fmt=".2f") == "-0.15"

    def test_zero_delta_has_plus_sign(self) -> None:
        # zero is non-negative → + prefix
        assert fmt_delta(100.0, 100.0, fmt=".0f") == "+0"

    def test_default_fmt_is_one_decimal(self) -> None:
        result = fmt_delta(1.5, 1.0)
        assert result == "+0.5"


class TestWarnCell:
    def test_reliable_passes_through(self) -> None:
        assert warn_cell("245", True) == "245"

    def test_unreliable_prepends_warn_prefix(self) -> None:
        result = warn_cell("245", False)
        assert result.startswith(WARN_PREFIX)
        assert "245" in result

    def test_warn_prefix_constant_matches_glyph(self) -> None:
        """Changing WARN_PREFIX in formatter must be reflected here."""
        assert WARN_PREFIX.strip() == "⚠️"


# ---------------------------------------------------------------------------
# render_warn_legend
# ---------------------------------------------------------------------------


class TestRenderWarnLegend:
    def test_contains_warn_glyph(self) -> None:
        con = _con()
        render_warn_legend(con)
        out = _out(con)
        assert "⚠️" in out

    def test_contains_stable_phrase(self) -> None:
        """'sample-size threshold' is what integration tests grep for."""
        con = _con()
        render_warn_legend(con)
        assert "sample-size threshold" in _out(con)

    def test_same_text_as_stats_legend(self) -> None:
        """Single source — calling it twice produces identical output."""
        con1, con2 = _con(), _con()
        render_warn_legend(con1)
        render_warn_legend(con2)
        assert _out(con1) == _out(con2)


# ---------------------------------------------------------------------------
# render_header
# ---------------------------------------------------------------------------


class TestRenderHeader:
    def test_shows_name_tag_tier_region(self) -> None:
        con = _con()
        render_header(
            con,
            name="Yoursaviour01",
            tag="SK04",
            tier="Gold 1",
            region="na",
            matches_shown=15,
            period="30d",
            agent_filter=None,
            map_filter=None,
        )
        out = _out(con)
        assert "Yoursaviour01" in out
        assert "SK04" in out
        assert "Gold 1" in out
        assert "NA" in out
        assert "period=30d" in out

    def test_optional_filters_appear_when_set(self) -> None:
        con = _con()
        render_header(
            con,
            name="T",
            tag="X",
            tier="Iron 1",
            region="eu",
            matches_shown=5,
            period="7d",
            agent_filter="Jett",
            map_filter="Ascent",
            result_filter="win",
        )
        out = _out(con)
        assert "agent=Jett" in out
        assert "map=Ascent" in out
        assert "result=win" in out

    def test_optional_filters_absent_when_none(self) -> None:
        con = _con()
        render_header(
            con,
            name="T",
            tag="X",
            tier="Iron 1",
            region="eu",
            matches_shown=5,
            period="all",
            agent_filter=None,
            map_filter=None,
        )
        out = _out(con)
        assert "agent=" not in out
        assert "map=" not in out
        assert "result=" not in out


# ---------------------------------------------------------------------------
# render_overall
# ---------------------------------------------------------------------------


class TestRenderOverall:
    def test_shows_key_stats(self) -> None:
        stats = compute_player_stats(_make_rows(1))
        con = _con()
        render_overall(con, stats)
        out = _out(con)
        assert "Matches" in out
        assert "ACS" in out
        assert "K/D" in out

    def test_returns_false_on_thick_sample(self) -> None:
        stats = compute_player_stats(_make_rows(30))
        con = _con()
        assert render_overall(con, stats) is False
        assert WARN_PREFIX not in _out(con)

    def test_returns_true_on_thin_sample(self) -> None:
        stats = compute_player_stats(_make_rows(5))
        con = _con()
        assert render_overall(con, stats) is True
        assert WARN_PREFIX in _out(con)

    def test_acs_value_matches_calculation(self) -> None:
        """ACS = score / rounds = 5000 / 20 = 250."""
        stats = compute_player_stats([_mp(score=5000, rounds_played=20)])
        con = _con()
        render_overall(con, stats)
        assert "250" in _out(con)


# ---------------------------------------------------------------------------
# render_breakdown
# ---------------------------------------------------------------------------


class TestRenderBreakdown:
    def test_empty_rows_is_noop_and_returns_false(self) -> None:
        con = _con()
        assert render_breakdown(con, title="By agent", group_col="Agent", rows=[], top_n=5) is False
        assert _out(con) == ""

    def test_top_n_limits_output(self) -> None:
        rows = _make_rows(10, agent="AgentX")
        # Give each row a distinct agent so we get 10 unique agent rows
        for i, mp in enumerate(rows):
            mp.agent_name = f"Agent{i}"
        per_agent = compute_per_agent(rows)
        con = _con()
        render_breakdown(con, title="By agent", group_col="Agent", rows=per_agent, top_n=3)
        out = _out(con)
        assert "Agent0" in out
        assert "Agent2" in out
        assert "Agent9" not in out

    def test_thin_split_warns_even_with_fat_overall(self) -> None:
        """3-game split must ⚠️ even when 30 total games are reliable."""
        rows = _make_rows(27, agent="Jett") + [
            _mp(agent="Reyna", match_id=f"r-{i}") for i in range(3)
        ]
        per_agent = compute_per_agent(rows)
        con = _con()
        any_warn = render_breakdown(
            con, title="By agent", group_col="Agent", rows=per_agent, top_n=5
        )
        out = _out(con)
        assert any_warn is True
        reyna_line = next(ln for ln in out.splitlines() if "Reyna" in ln)
        assert "⚠️" in reyna_line

    def test_map_breakdown_shows_map_names(self) -> None:
        rows = [
            _mp(map_name="Ascent", match_id="a"),
            _mp(map_name="Lotus", match_id="b"),
        ]
        per_map = compute_per_map(rows)
        con = _con()
        render_breakdown(con, title="By map", group_col="Map", rows=per_map, top_n=5)
        out = _out(con)
        assert "Ascent" in out
        assert "Lotus" in out


# ---------------------------------------------------------------------------
# render_win_loss_split
# ---------------------------------------------------------------------------


class TestRenderWinLossSplit:
    def test_skips_when_wins_empty(self) -> None:
        losses = _make_rows(5, won=False)
        con = _con()
        assert render_win_loss_split(con, [], losses) is False
        assert _out(con) == ""

    def test_skips_when_losses_empty(self) -> None:
        wins = _make_rows(5, won=True)
        con = _con()
        assert render_win_loss_split(con, wins, []) is False
        assert _out(con) == ""

    def test_shows_both_halves(self) -> None:
        wins = [_mp(won=True, score=8000, match_id=f"w-{i}") for i in range(5)]
        losses = [_mp(won=False, score=3000, match_id=f"l-{i}") for i in range(5)]
        con = _con()
        render_win_loss_split(con, wins, losses)
        out = _out(con)
        assert "ACS" in out
        assert "K/D" in out
        assert "Delta" in out


# ---------------------------------------------------------------------------
# render_round_stats
# ---------------------------------------------------------------------------


def _round_analysis(
    *,
    rounds: int = 600,  # 30 matches x 20 rounds -- clears all round-floor thresholds
    deaths: int = 350,
    teammate_deaths: int = 240,
    clutch_opportunities: int = 30,
    rounds_with_kill: int = 330,
    rounds_with_assist: int = 120,
    rounds_survived: int = 250,
    rounds_traded_death: int = 60,
    rounds_kast: int = 444,  # 74.0%
    clutches_won: int = 12,
    traded_deaths: int = 150,
    trades_given: int = 108,
    double_kills: int = 70,
    triple_kills: int = 18,
    quadra_kills: int = 5,
    aces: int = 1,
    attack_rounds: int | None = 300,
    attack_wins: int | None = 168,  # 56.0%
    defense_rounds: int | None = 300,
    defense_wins: int | None = 132,  # 44.0%
) -> RoundAnalysis:
    return RoundAnalysis(
        rounds=rounds,
        deaths=deaths,
        teammate_deaths=teammate_deaths,
        clutch_opportunities=clutch_opportunities,
        rounds_with_kill=rounds_with_kill,
        rounds_with_assist=rounds_with_assist,
        rounds_survived=rounds_survived,
        rounds_traded_death=rounds_traded_death,
        rounds_kast=rounds_kast,
        clutches_won=clutches_won,
        traded_deaths=traded_deaths,
        trades_given=trades_given,
        double_kills=double_kills,
        triple_kills=triple_kills,
        quadra_kills=quadra_kills,
        aces=aces,
        attack_rounds=attack_rounds,
        attack_wins=attack_wins,
        defense_rounds=defense_rounds,
        defense_wins=defense_wins,
    )


class TestRenderRoundStats:
    def test_shows_kast_and_clutch(self) -> None:
        con = _con()
        render_round_stats(con, _round_analysis(), matches=30)
        out = _out(con)
        assert "KAST" in out
        assert "Clutch" in out
        assert "Trade" in out

    def test_kast_value_matches_calculation(self) -> None:
        """444 KAST rounds / 600 total = 74.0%."""
        con = _con()
        render_round_stats(con, _round_analysis(rounds=600, rounds_kast=444), matches=30)
        assert "74.0%" in _out(con)

    def test_shows_multi_kill_summary(self) -> None:
        con = _con()
        render_round_stats(
            con, _round_analysis(double_kills=12, triple_kills=3, aces=1), matches=30
        )
        out = _out(con)
        assert "2K×12" in out
        assert "3K×3" in out
        assert "ACE×1" in out

    def test_no_multi_kills_shows_dash(self) -> None:
        con = _con()
        render_round_stats(
            con,
            _round_analysis(double_kills=0, triple_kills=0, quadra_kills=0, aces=0),
            matches=30,
        )
        assert "—" in _out(con)

    def test_side_win_rates_shown_when_available(self) -> None:
        """168/300 = 56.0% attack, 132/300 = 44.0% defense (fixture defaults)."""
        con = _con()
        render_round_stats(con, _round_analysis(), matches=30)
        out = _out(con)
        assert "Attack" in out
        assert "Defense" in out
        assert "56.0%" in out
        assert "44.0%" in out

    def test_side_win_rates_hidden_when_unavailable(self) -> None:
        """No crash or junk output when side data is None (pre-migration rows)."""
        con = _con()
        render_round_stats(
            con,
            _round_analysis(
                attack_rounds=None, attack_wins=None, defense_rounds=None, defense_wins=None
            ),
            matches=30,
        )
        out = _out(con)
        assert "KAST" in out  # still renders the rest
        assert "Attack W%" not in out

    def test_returns_false_on_thick_sample(self) -> None:
        """At 30 matches every round-level threshold is met — no ⚠️."""
        con = _con()
        result = render_round_stats(con, _round_analysis(), matches=30)
        assert result is False
        assert WARN_PREFIX not in _out(con)

    def test_returns_true_on_thin_sample(self) -> None:
        """At 5 matches with 100 rounds, all round-level thresholds fail → ⚠️."""
        con = _con()
        # Use thin fixture: 100 rounds / 5 matches — below all round-floor thresholds
        thin = _round_analysis(rounds=100, rounds_kast=74)
        result = render_round_stats(con, thin, matches=5)
        assert result is True
        assert WARN_PREFIX in _out(con)

    def test_returns_false_for_non_analysis_input(self) -> None:
        """Guard: passing a non-RoundAnalysis returns False without crashing."""
        con = _con()
        result = render_round_stats(con, object(), matches=10)
        assert result is False
        assert _out(con) == ""


# ---------------------------------------------------------------------------
# render_identity_panel
# ---------------------------------------------------------------------------


class TestRenderIdentityPanel:
    def test_shows_all_identity_fields(self) -> None:
        con = _con()
        render_identity_panel(con, _player())
        out = _out(con)
        assert "Yoursaviour01" in out
        assert "SK04" in out
        assert "NA" in out
        assert "Gold 1" in out
        assert "25 RR" in out
        assert "Gold 3" in out
        assert "240" in out

    def test_never_synced_shows_never(self) -> None:
        con = _con()
        render_identity_panel(con, _player(last_match_at=None))
        assert "never" in _out(con).lower()


# ---------------------------------------------------------------------------
# render_summary_card
# ---------------------------------------------------------------------------


class TestRenderSummaryCard:
    def test_empty_rows_prints_no_matches_message(self) -> None:
        con = _con()
        result = render_summary_card(con, [], limit=20)
        assert result is False
        assert "No matches" in _out(con)

    def test_shows_record_and_key_stats(self) -> None:
        con = _con()
        render_summary_card(con, [_mp()], limit=20)
        out = _out(con)
        assert "1-0" in out  # record
        assert "250" in out  # ACS 5000/20
        assert "K/D" in out

    def test_limit_clamps_shown_count(self) -> None:
        rows = _make_rows(3)
        con = _con()
        render_summary_card(con, rows, limit=20)
        out = _out(con)
        assert "Last 3 match" in out
        assert "Last 20 match" not in out

    def test_fb_diff_has_explicit_sign(self) -> None:
        rows = [_mp(first_bloods=5, first_deaths=2)]
        con = _con()
        render_summary_card(con, rows, limit=20)
        assert "+3" in _out(con)

    def test_warns_on_thin_sample(self) -> None:
        rows = _make_rows(5)
        con = _con()
        assert render_summary_card(con, rows, limit=20) is True
        assert WARN_PREFIX in _out(con)

    def test_no_warn_on_thick_sample(self) -> None:
        rows = _make_rows(30)
        con = _con()
        assert render_summary_card(con, rows, limit=30) is False
        assert WARN_PREFIX not in _out(con)

    def test_hs_warns_before_fb_warns(self) -> None:
        """At 20 matches: HS% (needs 30) should ⚠️ but ACS (needs 15) should not."""
        rows = _make_rows(20)
        con = _con()
        render_summary_card(con, rows, limit=20)
        out = _out(con)
        hs_line = next(ln for ln in out.splitlines() if "HS%" in ln)
        acs_line = next(ln for ln in out.splitlines() if "ACS" in ln)
        assert "⚠️" in hs_line
        assert "⚠️" not in acs_line


# ---------------------------------------------------------------------------
# render_trend — lines 363-375
# ---------------------------------------------------------------------------


def _make_anomaly(*, severity: str = "significant", is_improvement: bool = False):
    """Build a real Anomaly dataclass for render_trend tests."""
    from valocoach.stats.baseline import Anomaly

    return Anomaly(
        metric="acs",
        label="ACS",
        baseline_val=250.0,
        form_val=200.0,
        delta=-50.0,
        pct_delta=-0.2,
        direction="down",
        severity=severity,  # type: ignore[arg-type]
        is_improvement=is_improvement,
        fmt=".0f",
        z_score=-2.1,
        baseline_stddev=30.0,
    )


def _make_comparison(anomalies):
    """Build a real BaselineComparison for render_trend tests."""
    from valocoach.stats.baseline import BaselineComparison
    from valocoach.stats.calculator import PlayerStats

    ps = PlayerStats(
        matches=20,
        rounds=400,
        wins=12,
        losses=8,
        win_rate=0.6,
        acs=250.0,
        adr=150.0,
        kills=400,
        deaths=200,
        assists=100,
        kd=2.0,
        kda=2.5,
        headshots=120,
        bodyshots=240,
        legshots=40,
        hs_pct=0.30,
        first_bloods=40,
        first_deaths=20,
        fb_rate=0.10,
        fd_rate=0.05,
        fb_diff=20,
        plants=10,
        defuses=5,
        econ_rating=None,
    )
    return BaselineComparison(
        baseline=ps,
        form=ps,
        baseline_matches=15,
        form_matches=5,
        anomalies=anomalies,
    )


class TestRenderTrend:
    """Coverage for formatter.render_trend (lines 363-375)."""

    def test_silent_when_comparison_is_none(self) -> None:
        """No output when compare_baseline returns None (line 360 early-return)."""
        con = _con()
        with patch("valocoach.cli.formatter.compare_baseline", return_value=None):
            render_trend(con, [])
        assert _out(con) == ""

    def test_silent_when_no_anomalies(self) -> None:
        """No output when has_anomalies is False (line 360 second guard)."""
        comp = _make_comparison(anomalies=[])
        con = _con()
        with patch("valocoach.cli.formatter.compare_baseline", return_value=comp):
            render_trend(con, [])
        assert _out(con) == ""

    def test_significant_slump_shows_double_bang(self) -> None:
        """Significant non-improvement → '(!!)' tag (lines 368-370).

        Coverage target: lines 363-375 — the whole trend block executes.
        """
        a = _make_anomaly(severity="significant", is_improvement=False)
        comp = _make_comparison([a])
        con = _con()
        with patch("valocoach.cli.formatter.compare_baseline", return_value=comp):
            render_trend(con, [])
        out = _out(con)
        assert "Trend" in out
        assert "ACS" in out
        assert "!!" in out  # (!!)-tag for significant

    def test_significant_improvement_also_shows_double_bang(self) -> None:
        """Significant improvement → 'bold green' branch (line 369)."""
        a = _make_anomaly(severity="significant", is_improvement=True)
        comp = _make_comparison([a])
        con = _con()
        with patch("valocoach.cli.formatter.compare_baseline", return_value=comp):
            render_trend(con, [])
        out = _out(con)
        assert "Trend" in out
        assert "!!" in out

    def test_notable_slump_shows_single_bang(self) -> None:
        """Notable non-improvement → '(!)' tag (lines 371-373)."""
        a = _make_anomaly(severity="notable", is_improvement=False)
        comp = _make_comparison([a])
        con = _con()
        with patch("valocoach.cli.formatter.compare_baseline", return_value=comp):
            render_trend(con, [])
        out = _out(con)
        assert "Trend" in out
        # notable uses "(!)" — double-bang must NOT appear
        assert "(!)" in out
        assert "(!!" not in out

    def test_notable_improvement_shows_single_bang(self) -> None:
        """Notable improvement → 'green' branch (line 372)."""
        a = _make_anomaly(severity="notable", is_improvement=True)
        comp = _make_comparison([a])
        con = _con()
        with patch("valocoach.cli.formatter.compare_baseline", return_value=comp):
            render_trend(con, [])
        out = _out(con)
        assert "Trend" in out
        assert "(!)" in out
        assert "(!!" not in out

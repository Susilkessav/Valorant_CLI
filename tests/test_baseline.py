"""Tests for valocoach.stats.baseline -- rolling baseline and anomaly detection.

Covers:
  - _per_match_metrics: correct per-match decomposition
  - _mean_stddev: population statistics
  - _check_one: z-score threshold logic (notable / significant, up / down)
  - detect_anomalies: multi-metric, sorting, improvement vs. decline
  - compare_baseline: reliability gates, form/baseline relationship
  - Anomaly display properties (arrow, delta_display, one_liner)
  - Coach-context _format_baseline_lines integration
"""

from __future__ import annotations

import math

from valocoach.data.orm_models import Match, MatchPlayer
from valocoach.stats.baseline import (
    FORM_WINDOW,
    MIN_BASELINE_MATCHES,
    MIN_FORM_MATCHES,
    Z_NOTABLE,
    Anomaly,
    BaselineComparison,
    _check_one,
    _mean_stddev,
    _per_match_metrics,
    compare_baseline,
    detect_anomalies,
)
from valocoach.stats.calculator import PlayerStats

# ---------------------------------------------------------------------------
# Stub builders
# ---------------------------------------------------------------------------


def _mp(
    *,
    match_id: str = "m-1",
    agent: str = "Jett",
    map_name: str = "Ascent",
    started_at: str = "2026-04-20T12:00:00+00:00",
    won: bool = True,
    rounds_played: int = 20,
    score: int = 4000,        # ACS = 200
    kills: int = 14,
    deaths: int = 10,
    assists: int = 3,
    headshots: int = 20,      # HS% = 20/90 = 22.2%
    bodyshots: int = 60,
    legshots: int = 10,
    damage_dealt: int = 3000, # ADR = 150
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
        first_bloods=1,
        first_deaths=0,
        plants=0,
        defuses=0,
        afk_rounds=0,
        rounds_in_spawn=0,
        started_at=started_at,
    )
    mp.match = match
    return mp


def _make_rows(
    n: int,
    *,
    score: int = 4000,
    kills: int = 14,
    deaths: int = 10,
    damage_dealt: int = 3000,
    headshots: int = 20,
    won: bool = True,
) -> list[MatchPlayer]:
    """n matches with distinct match_ids and decreasing started_at (index 0 = newest)."""
    return [
        _mp(
            match_id=f"m-{i}",
            started_at=f"2026-04-{20 - i:02d}T12:00:00+00:00",
            score=score,
            kills=kills,
            deaths=deaths,
            damage_dealt=damage_dealt,
            headshots=headshots,
            won=won,
        )
        for i in range(n)
    ]


def _rows_mixed(*, old_score: int, new_score: int, n_old: int = 15, n_new: int = 5) -> list[MatchPlayer]:
    """Build a mixed window: n_old older rows with old_score, n_new newer with new_score."""
    old = [
        _mp(
            match_id=f"old-{i}",
            started_at=f"2026-03-{i + 1:02d}T00:00:00+00:00",
            score=old_score,
        )
        for i in range(n_old)
    ]
    new = [
        _mp(
            match_id=f"new-{i}",
            started_at=f"2026-04-{15 + i:02d}T00:00:00+00:00",
            score=new_score,
        )
        for i in range(n_new)
    ]
    return old + new


# ---------------------------------------------------------------------------
# _per_match_metrics
# ---------------------------------------------------------------------------


class TestPerMatchMetrics:
    def test_acs_is_score_over_rounds_per_match(self) -> None:
        rows = [_mp(score=4000, rounds_played=20)]
        m = _per_match_metrics(rows)
        assert m["acs"] == [200.0]

    def test_kd_is_kills_over_deaths_per_match(self) -> None:
        rows = [_mp(kills=14, deaths=10)]
        m = _per_match_metrics(rows)
        assert m["kd"] == [1.4]

    def test_adr_is_damage_over_rounds_per_match(self) -> None:
        rows = [_mp(damage_dealt=3000, rounds_played=20)]
        m = _per_match_metrics(rows)
        assert m["adr"] == [150.0]

    def test_hs_pct_is_headshots_over_total_shots(self) -> None:
        rows = [_mp(headshots=20, bodyshots=60, legshots=10)]  # 20/90
        m = _per_match_metrics(rows)
        assert abs(m["hs_pct"][0] - 20 / 90) < 1e-9

    def test_win_rate_is_1_for_win_0_for_loss(self) -> None:
        rows = [_mp(won=True), _mp(match_id="m-2", won=False)]
        m = _per_match_metrics(rows)
        assert m["win_rate"] == [1.0, 0.0]

    def test_n_matches_gives_n_values(self) -> None:
        rows = _make_rows(7)
        m = _per_match_metrics(rows)
        for key in ("acs", "kd", "adr", "hs_pct", "win_rate"):
            assert len(m[key]) == 7


# ---------------------------------------------------------------------------
# _mean_stddev
# ---------------------------------------------------------------------------


class TestMeanStddev:
    def test_constant_values_zero_stddev(self) -> None:
        mean, stddev = _mean_stddev([5.0, 5.0, 5.0])
        assert mean == 5.0
        assert stddev == 0.0

    def test_known_mean(self) -> None:
        mean, _ = _mean_stddev([1.0, 2.0, 3.0, 4.0, 5.0])
        assert mean == 3.0

    def test_known_population_stddev(self) -> None:
        # Population stddev of [2, 4, 4, 4, 5, 5, 7, 9] = 2.0
        _, stddev = _mean_stddev([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
        assert abs(stddev - 2.0) < 1e-9

    def test_single_value_zero_stddev(self) -> None:
        mean, stddev = _mean_stddev([42.0])
        assert mean == 42.0
        assert stddev == 0.0


# ---------------------------------------------------------------------------
# _check_one -- z-score threshold logic
# ---------------------------------------------------------------------------


class TestCheckOne:
    def test_no_change_returns_none(self) -> None:
        assert _check_one("acs", [200.0] * 5, 200.0, 10.0) is None

    def test_unknown_metric_returns_none(self) -> None:
        assert _check_one("fantasy_stat", [100.0], 100.0, 10.0) is None

    def test_empty_form_vals_returns_none(self) -> None:
        assert _check_one("acs", [], 200.0, 10.0) is None

    def test_below_z_notable_returns_none(self) -> None:
        # z = (190 - 200) / 10 = -1.0, below Z_NOTABLE=1.5
        assert _check_one("acs", [190.0], 200.0, 10.0) is None

    def test_at_z_notable_returns_anomaly(self) -> None:
        # form = 200 - 1.5*10 = 185 => z = -1.5 exactly
        a = _check_one("acs", [185.0], 200.0, 10.0)
        assert a is not None
        assert a.severity == "notable"
        assert a.direction == "down"
        assert a.metric == "acs"

    def test_at_z_significant_returns_significant(self) -> None:
        # form = 200 - 2.0*10 = 180 => z = -2.0 exactly
        a = _check_one("acs", [180.0], 200.0, 10.0)
        assert a is not None
        assert a.severity == "significant"

    def test_above_z_significant(self) -> None:
        # z = -3.0 => still significant (>= Z_SIGNIFICANT)
        a = _check_one("acs", [170.0], 200.0, 10.0)
        assert a is not None
        assert a.severity == "significant"

    def test_improvement_flagged_correctly(self) -> None:
        # ACS up 2.0 sigma => notable improvement
        a = _check_one("acs", [220.0], 200.0, 10.0)
        assert a is not None
        assert a.is_improvement is True
        assert a.direction == "up"

    def test_kd_flagged(self) -> None:
        # K/D drops 2.25 sigma: (0.75 - 1.20) / 0.20 = -2.25 -> significant
        a = _check_one("kd", [0.75], 1.20, 0.20)
        assert a is not None
        assert a.severity == "significant"

    def test_hs_pct_format_is_pct(self) -> None:
        # HS% drops 2.5 sigma
        a = _check_one("hs_pct", [0.10], 0.22, 0.048)
        assert a is not None
        assert a.fmt == "pct"

    def test_stddev_floor_prevents_huge_z_on_consistent_player(self) -> None:
        # baseline_mean=200, baseline_stddev=0 (perfectly consistent).
        # floor = max(0, 200*0.02, 1e-6) = 4.0
        # z = (150 - 200) / 4.0 = -12.5 => significant, but finite
        a = _check_one("acs", [150.0], 200.0, 0.0)
        assert a is not None
        assert a.severity == "significant"
        assert math.isfinite(a.z_score)

    def test_z_score_stored_on_anomaly(self) -> None:
        a = _check_one("acs", [180.0], 200.0, 10.0)
        assert a is not None
        assert abs(a.z_score - (-2.0)) < 1e-9

    def test_baseline_stddev_stored_on_anomaly(self) -> None:
        a = _check_one("acs", [180.0], 200.0, 10.0)
        assert a is not None
        assert a.baseline_stddev == 10.0

    def test_pct_delta_none_when_baseline_zero(self) -> None:
        # win_rate baseline = 0 (impossible in practice but safe to handle)
        a = _check_one("win_rate", [0.5], 0.0, 0.0)
        assert a is not None
        assert a.pct_delta is None


# ---------------------------------------------------------------------------
# Anomaly display properties
# ---------------------------------------------------------------------------


class TestAnomalyDisplay:
    def _acs_decline(self) -> Anomaly:
        a = _check_one("acs", [168.0], 215.0, 23.5)
        assert a is not None
        return a

    def _hs_gain(self) -> Anomaly:
        a = _check_one("hs_pct", [0.28], 0.22, 0.03)
        assert a is not None
        return a

    def test_arrow_down(self) -> None:
        assert self._acs_decline().arrow == "\u2193"

    def test_arrow_up(self) -> None:
        assert self._hs_gain().arrow == "\u2191"

    def test_baseline_display_acs(self) -> None:
        assert self._acs_decline().baseline_display == "215"

    def test_form_display_acs(self) -> None:
        assert self._acs_decline().form_display == "168"

    def test_delta_display_pct_metric(self) -> None:
        # HS% gain of 6pp -> "+6pp"
        assert self._hs_gain().delta_display == "+6pp"

    def test_delta_display_relative_metric(self) -> None:
        dd = self._acs_decline().delta_display
        assert dd.startswith("-") and "%" in dd

    def test_one_liner_contains_sigma(self) -> None:
        line = self._acs_decline().one_liner()
        assert "\u03c3" in line  # sigma character

    def test_one_liner_contains_key_info(self) -> None:
        line = self._acs_decline().one_liner()
        assert "ACS" in line
        assert "\u2193" in line  # down arrow
        assert "168" in line
        assert "215" in line
        assert "slump" in line

    def test_one_liner_improvement(self) -> None:
        line = self._hs_gain().one_liner()
        assert "improvement" in line
        assert "\u2191" in line


# ---------------------------------------------------------------------------
# detect_anomalies -- multi-metric, sorting, direction
# ---------------------------------------------------------------------------


class TestDetectAnomalies:
    """detect_anomalies(baseline_rows, form_rows) excludes form_rows from the
    z-score baseline so it compares "recent form" against "historical only"."""

    def test_identical_rows_returns_empty(self) -> None:
        rows = _make_rows(20)
        form = rows[:5]
        # Identical data -- no anomalies even with uncontaminated baseline.
        assert detect_anomalies(rows, form) == []

    def test_detects_significant_acs_drop(self) -> None:
        # 15 old rows at ACS=250, 5 form rows at ACS=100.
        # Historical baseline: mean=250, stddev=0, eff_stddev=250*0.02=5
        # z = (100 - 250) / 5 = -30 -> significant
        rows = _rows_mixed(old_score=5000, new_score=2000)
        form = [r for r in rows if r.match_id.startswith("new-")]
        anomalies = detect_anomalies(rows, form)
        acs = next((a for a in anomalies if a.metric == "acs"), None)
        assert acs is not None
        assert acs.severity == "significant"
        assert not acs.is_improvement

    def test_significant_before_notable(self) -> None:
        rows = _rows_mixed(old_score=5000, new_score=2000)
        form = [r for r in rows if r.match_id.startswith("new-")]
        anomalies = detect_anomalies(rows, form)
        sig = [a for a in anomalies if a.severity == "significant"]
        nota = [a for a in anomalies if a.severity == "notable"]
        if sig and nota:
            last_sig_idx = max(anomalies.index(a) for a in sig)
            first_nota_idx = min(anomalies.index(a) for a in nota)
            assert last_sig_idx < first_nota_idx

    def test_declines_before_improvements_within_tier(self) -> None:
        # old: ACS 200 + HS% 10%; new: ACS 100 + HS% 40%.
        # Both are anomalies -- ACS down (decline), HS% up (improvement).
        old = [_mp(match_id=f"o-{i}", started_at=f"2026-03-{i+1:02d}T00:00:00+00:00",
                   score=4000, headshots=9, bodyshots=81) for i in range(15)]
        new = [_mp(match_id=f"n-{i}", started_at=f"2026-04-{15+i:02d}T00:00:00+00:00",
                   score=2000, headshots=36, bodyshots=54) for i in range(5)]
        rows = old + new
        anomalies = detect_anomalies(rows, new)
        decline_indices = [i for i, a in enumerate(anomalies) if not a.is_improvement]
        improve_indices = [i for i, a in enumerate(anomalies) if a.is_improvement]
        if decline_indices and improve_indices:
            assert max(decline_indices) < min(improve_indices)

    def test_too_few_historical_rows_skips_metric(self) -> None:
        # 4 rows total, 3 as form -> only 1 historical row (< MIN_BASELINE_MATCHES=5).
        rows = _make_rows(4)
        form = rows[:3]
        assert detect_anomalies(rows, form) == []

    def test_all_rows_are_form_returns_empty(self) -> None:
        rows = _make_rows(5)
        # form == all rows -> no historical rows left -> empty
        assert detect_anomalies(rows, rows) == []

    def test_win_rate_tracked(self) -> None:
        # 15 wins (historical), 5 losses (form) -- huge drop.
        old = [_mp(match_id=f"o-{i}", started_at=f"2026-03-{i+1:02d}T00:00:00+00:00",
                   won=True) for i in range(15)]
        new = [_mp(match_id=f"n-{i}", started_at=f"2026-04-{15+i:02d}T00:00:00+00:00",
                   won=False) for i in range(5)]
        rows = old + new
        anomalies = detect_anomalies(rows, new)
        wr = next((a for a in anomalies if a.metric == "win_rate"), None)
        assert wr is not None
        assert wr.severity == "significant"
        assert not wr.is_improvement


# ---------------------------------------------------------------------------
# compare_baseline -- reliability gates and structure
# ---------------------------------------------------------------------------


class TestCompareBaseline:
    def test_too_few_rows_returns_none(self) -> None:
        rows = _make_rows(MIN_BASELINE_MATCHES - 1)
        assert compare_baseline(rows) is None

    def test_form_equals_baseline_returns_none(self) -> None:
        rows = _make_rows(FORM_WINDOW)  # form_n = 5 = total rows
        assert compare_baseline(rows) is None

    def test_returns_comparison_for_sufficient_rows(self) -> None:
        rows = _make_rows(FORM_WINDOW + 1)
        result = compare_baseline(rows)
        assert result is not None
        assert result.baseline_matches == FORM_WINDOW + 1
        assert result.form_matches == FORM_WINDOW

    def test_thin_form_window_no_anomalies(self) -> None:
        # form_n = 2 < MIN_FORM_MATCHES=3 -- anomalies skipped
        rows = _make_rows(MIN_BASELINE_MATCHES + 2)
        result = compare_baseline(rows, form_n=MIN_FORM_MATCHES - 1)
        assert result is not None
        assert result.anomalies == []

    def test_stable_performance_no_anomalies(self) -> None:
        rows = _make_rows(20)
        result = compare_baseline(rows)
        assert result is not None
        assert result.anomalies == []  # identical rows -> z = 0

    def test_detects_acs_drop_in_recent_form(self) -> None:
        rows = _rows_mixed(old_score=5000, new_score=2000)  # 250 vs 100 ACS
        result = compare_baseline(rows)
        assert result is not None
        acs = next((a for a in result.anomalies if a.metric == "acs"), None)
        assert acs is not None
        assert acs.direction == "down"
        assert not acs.is_improvement

    def test_has_anomalies_property(self) -> None:
        rows = _make_rows(20)
        result = compare_baseline(rows)
        assert result is not None
        assert result.has_anomalies is False

    def test_declines_and_improvements_partition(self) -> None:
        rows = _rows_mixed(old_score=5000, new_score=2000)
        result = compare_baseline(rows)
        assert result is not None
        all_ids = {id(a) for a in result.anomalies}
        dec_ids = {id(a) for a in result.declines}
        imp_ids = {id(a) for a in result.improvements}
        assert dec_ids | imp_ids == all_ids
        assert dec_ids & imp_ids == set()

    def test_baseline_form_stats_are_player_stats(self) -> None:
        rows = _make_rows(10)
        result = compare_baseline(rows)
        assert result is not None
        assert isinstance(result.baseline, PlayerStats)
        assert isinstance(result.form, PlayerStats)

    def test_anomaly_carries_z_score(self) -> None:
        rows = _rows_mixed(old_score=5000, new_score=2000)
        result = compare_baseline(rows)
        assert result is not None
        acs = next((a for a in result.anomalies if a.metric == "acs"), None)
        assert acs is not None
        assert math.isfinite(acs.z_score)
        assert abs(acs.z_score) >= Z_NOTABLE


# ---------------------------------------------------------------------------
# Coach context integration -- _format_baseline_lines
# ---------------------------------------------------------------------------


class TestFormatBaselineLines:
    def _comparison_no_anomalies(self) -> BaselineComparison:
        rows = _make_rows(20)
        result = compare_baseline(rows)
        assert result is not None
        return result

    def _comparison_with_acs_drop(self) -> BaselineComparison:
        rows = _rows_mixed(old_score=5000, new_score=2000)
        result = compare_baseline(rows)
        assert result is not None
        return result

    def test_no_anomalies_returns_empty_list(self) -> None:
        from valocoach.coach.context import _format_baseline_lines
        assert _format_baseline_lines(self._comparison_no_anomalies()) == []

    def test_anomaly_lines_start_with_dash(self) -> None:
        from valocoach.coach.context import _format_baseline_lines
        lines = _format_baseline_lines(self._comparison_with_acs_drop())
        body = [ln for ln in lines if ln.startswith("-")]
        assert body

    def test_significant_anomaly_double_exclamation(self) -> None:
        from valocoach.coach.context import _format_baseline_lines
        lines = _format_baseline_lines(self._comparison_with_acs_drop())
        body = [ln for ln in lines if ln.startswith("-")]
        assert any("(!!)" in ln for ln in body)

    def test_header_contains_match_counts(self) -> None:
        from valocoach.coach.context import _format_baseline_lines
        comp = self._comparison_with_acs_drop()
        lines = _format_baseline_lines(comp)
        assert lines
        header = lines[0]
        assert f"{comp.form_matches}g" in header
        assert f"{comp.baseline_matches}g" in header

    def test_one_liner_sigma_in_body(self) -> None:
        from valocoach.coach.context import _format_baseline_lines
        lines = _format_baseline_lines(self._comparison_with_acs_drop())
        body = [ln for ln in lines if ln.startswith("-")]
        assert any("\u03c3" in ln for ln in body)  # sigma

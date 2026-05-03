"""Tests for valocoach.stats.filters — pure filter predicates.

Covers every public function: parse_period, filter_by_*, apply_filters,
split_by_result, recent_form.

No DB, no I/O — all tests build MatchPlayer stubs directly.
"""

from __future__ import annotations

import pytest

from valocoach.data.orm_models import Match, MatchPlayer
from valocoach.stats.filters import (
    apply_filters,
    filter_by_agent,
    filter_by_map,
    filter_by_period,
    filter_by_queue,
    filter_by_result,
    filter_by_tier_range,
    parse_period,
    recent_form,
    split_by_result,
)

# ---------------------------------------------------------------------------
# Shared stub builder
# ---------------------------------------------------------------------------


def _mp(
    *,
    match_id: str = "m-1",
    agent: str = "Jett",
    map_name: str = "Ascent",
    queue_id: str = "competitive",
    started_at: str = "2026-04-20T12:00:00+00:00",
    won: bool = True,
    competitive_tier: int | None = 12,  # Gold 1
    rounds_played: int = 20,
    score: int = 4000,
    kills: int = 15,
    deaths: int = 10,
    assists: int = 3,
    headshots: int = 20,
    bodyshots: int = 60,
    legshots: int = 10,
    damage_dealt: int = 2500,
) -> MatchPlayer:
    match = Match(
        match_id=match_id,
        map_name=map_name,
        queue_id=queue_id,
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
        competitive_tier=competitive_tier,
        started_at=started_at,
    )
    mp.match = match
    return mp


# ---------------------------------------------------------------------------
# parse_period
# ---------------------------------------------------------------------------


def test_parse_period_all_returns_none() -> None:
    assert parse_period("all") is None
    assert parse_period("ALL") is None
    assert parse_period("  All  ") is None


def test_parse_period_days_returns_iso_string() -> None:
    from datetime import UTC, datetime, timedelta

    before = datetime.now(UTC) - timedelta(days=30)
    result = parse_period("30d")
    after = datetime.now(UTC) - timedelta(days=30)

    assert result is not None
    parsed = datetime.fromisoformat(result)
    # Allow 1s tolerance for test execution time.
    assert before - __import__("datetime").timedelta(seconds=1) <= parsed
    assert parsed <= after + __import__("datetime").timedelta(seconds=1)


def test_parse_period_case_insensitive_days() -> None:
    # "7D" should be accepted.
    result = parse_period("7D")
    assert result is not None


@pytest.mark.parametrize(
    "bad",
    ["", "d", "abc", "30", "7days", "-5d", "0d", "3.5d"],
)
def test_parse_period_bad_input_raises_value_error(bad: str) -> None:
    with pytest.raises(ValueError, match="period must"):
        parse_period(bad)


def test_parse_period_ordering() -> None:
    """7d cutoff is later (more recent) than 30d cutoff."""
    from datetime import datetime

    iso7 = parse_period("7d")
    iso30 = parse_period("30d")
    assert iso7 is not None and iso30 is not None
    assert datetime.fromisoformat(iso7) > datetime.fromisoformat(iso30)


# ---------------------------------------------------------------------------
# filter_by_period
# ---------------------------------------------------------------------------


def test_filter_by_period_none_is_passthrough() -> None:
    rows = [_mp(match_id="a"), _mp(match_id="b")]
    assert filter_by_period(rows, None) == rows


def test_filter_by_period_keeps_rows_on_cutoff() -> None:
    """Exact cutoff timestamp should be kept (>= comparison)."""
    cutoff = "2026-01-01T00:00:00+00:00"
    rows = [
        _mp(started_at=cutoff, match_id="on-cutoff"),
        _mp(started_at="2025-12-31T23:59:59+00:00", match_id="just-before"),
    ]
    out = filter_by_period(rows, cutoff)
    assert [mp.match_id for mp in out] == ["on-cutoff"]


def test_filter_by_period_drops_old_and_keeps_new() -> None:
    rows = [
        _mp(started_at="2026-04-20T12:00:00+00:00", match_id="new"),
        _mp(started_at="2025-06-01T00:00:00+00:00", match_id="mid"),
        _mp(started_at="2020-01-01T00:00:00+00:00", match_id="old"),
    ]
    out = filter_by_period(rows, "2026-01-01T00:00:00+00:00")
    assert [mp.match_id for mp in out] == ["new"]


def test_filter_by_period_empty_input() -> None:
    assert filter_by_period([], "2026-01-01T00:00:00+00:00") == []


# ---------------------------------------------------------------------------
# filter_by_agent
# ---------------------------------------------------------------------------


def test_filter_by_agent_none_is_passthrough() -> None:
    rows = [_mp(agent="Jett"), _mp(agent="Reyna")]
    assert filter_by_agent(rows, None) == rows


def test_filter_by_agent_exact_match() -> None:
    rows = [_mp(agent="Jett", match_id="j"), _mp(agent="Reyna", match_id="r")]
    out = filter_by_agent(rows, "Jett")
    assert [mp.match_id for mp in out] == ["j"]


def test_filter_by_agent_case_insensitive() -> None:
    rows = [_mp(agent="Jett", match_id="j"), _mp(agent="Reyna", match_id="r")]
    assert [mp.match_id for mp in filter_by_agent(rows, "JETT")] == ["j"]
    assert [mp.match_id for mp in filter_by_agent(rows, "jett")] == ["j"]
    assert [mp.match_id for mp in filter_by_agent(rows, "Jett")] == ["j"]


def test_filter_by_agent_no_match_returns_empty() -> None:
    rows = [_mp(agent="Jett"), _mp(agent="Reyna")]
    assert filter_by_agent(rows, "Sage") == []


def test_filter_by_agent_multiple_matches() -> None:
    rows = [
        _mp(agent="Jett", match_id="j1"),
        _mp(agent="Jett", match_id="j2"),
        _mp(agent="Reyna", match_id="r"),
    ]
    out = filter_by_agent(rows, "Jett")
    assert {mp.match_id for mp in out} == {"j1", "j2"}


# ---------------------------------------------------------------------------
# filter_by_map
# ---------------------------------------------------------------------------


def test_filter_by_map_none_is_passthrough() -> None:
    rows = [_mp(map_name="Ascent"), _mp(map_name="Lotus")]
    assert filter_by_map(rows, None) == rows


def test_filter_by_map_exact_match() -> None:
    rows = [_mp(map_name="Ascent", match_id="a"), _mp(map_name="Lotus", match_id="b")]
    out = filter_by_map(rows, "Ascent")
    assert [mp.match_id for mp in out] == ["a"]


def test_filter_by_map_case_insensitive() -> None:
    rows = [_mp(map_name="Ascent", match_id="a"), _mp(map_name="Lotus", match_id="b")]
    assert [mp.match_id for mp in filter_by_map(rows, "ascent")] == ["a"]
    assert [mp.match_id for mp in filter_by_map(rows, "ASCENT")] == ["a"]


def test_filter_by_map_no_match() -> None:
    rows = [_mp(map_name="Ascent"), _mp(map_name="Lotus")]
    assert filter_by_map(rows, "Haven") == []


# ---------------------------------------------------------------------------
# filter_by_result
# ---------------------------------------------------------------------------


def test_filter_by_result_none_is_passthrough() -> None:
    rows = [_mp(won=True, match_id="w"), _mp(won=False, match_id="l")]
    assert filter_by_result(rows, None) == rows


def test_filter_by_result_wins_only() -> None:
    rows = [_mp(won=True, match_id="w"), _mp(won=False, match_id="l")]
    out = filter_by_result(rows, won=True)
    assert [mp.match_id for mp in out] == ["w"]


def test_filter_by_result_losses_only() -> None:
    rows = [_mp(won=True, match_id="w"), _mp(won=False, match_id="l")]
    out = filter_by_result(rows, won=False)
    assert [mp.match_id for mp in out] == ["l"]


def test_filter_by_result_all_wins() -> None:
    rows = [_mp(won=True, match_id=f"w{i}") for i in range(5)]
    assert filter_by_result(rows, won=False) == []
    assert len(filter_by_result(rows, won=True)) == 5


# ---------------------------------------------------------------------------
# filter_by_queue
# ---------------------------------------------------------------------------


def test_filter_by_queue_none_is_passthrough() -> None:
    rows = [_mp(queue_id="competitive"), _mp(queue_id="unrated")]
    assert filter_by_queue(rows, None) == rows


def test_filter_by_queue_exact_match() -> None:
    rows = [
        _mp(queue_id="competitive", match_id="c"),
        _mp(queue_id="unrated", match_id="u"),
    ]
    out = filter_by_queue(rows, "competitive")
    assert [mp.match_id for mp in out] == ["c"]


def test_filter_by_queue_case_insensitive() -> None:
    rows = [
        _mp(queue_id="competitive", match_id="c"),
        _mp(queue_id="unrated", match_id="u"),
    ]
    assert [mp.match_id for mp in filter_by_queue(rows, "COMPETITIVE")] == ["c"]


# ---------------------------------------------------------------------------
# filter_by_tier_range
# ---------------------------------------------------------------------------


def test_filter_by_tier_range_no_bounds_is_passthrough() -> None:
    rows = [_mp(competitive_tier=12), _mp(competitive_tier=18)]
    assert filter_by_tier_range(rows, min_tier=None, max_tier=None) == rows


def test_filter_by_tier_range_min_only() -> None:
    rows = [
        _mp(competitive_tier=6, match_id="iron"),  # below
        _mp(competitive_tier=12, match_id="gold"),  # at floor
        _mp(competitive_tier=18, match_id="plat"),  # above
    ]
    out = filter_by_tier_range(rows, min_tier=12, max_tier=None)
    assert {mp.match_id for mp in out} == {"gold", "plat"}


def test_filter_by_tier_range_max_only() -> None:
    rows = [
        _mp(competitive_tier=6, match_id="iron"),
        _mp(competitive_tier=12, match_id="gold"),
        _mp(competitive_tier=18, match_id="plat"),
    ]
    out = filter_by_tier_range(rows, min_tier=None, max_tier=12)
    assert {mp.match_id for mp in out} == {"iron", "gold"}


def test_filter_by_tier_range_both_bounds() -> None:
    rows = [
        _mp(competitive_tier=6, match_id="iron"),
        _mp(competitive_tier=12, match_id="gold"),
        _mp(competitive_tier=15, match_id="plat"),
        _mp(competitive_tier=21, match_id="diamond"),
    ]
    out = filter_by_tier_range(rows, min_tier=12, max_tier=15)
    assert {mp.match_id for mp in out} == {"gold", "plat"}


def test_filter_by_tier_range_drops_none_tier() -> None:
    """Rows with no tier data are excluded when any bound is set."""
    rows = [
        _mp(competitive_tier=None, match_id="no-tier"),
        _mp(competitive_tier=12, match_id="gold"),
    ]
    out = filter_by_tier_range(rows, min_tier=10, max_tier=None)
    assert [mp.match_id for mp in out] == ["gold"]


def test_filter_by_tier_range_none_rows_pass_when_no_bounds() -> None:
    rows = [
        _mp(competitive_tier=None, match_id="no-tier"),
        _mp(competitive_tier=12, match_id="gold"),
    ]
    out = filter_by_tier_range(rows, min_tier=None, max_tier=None)
    assert {mp.match_id for mp in out} == {"no-tier", "gold"}


# ---------------------------------------------------------------------------
# apply_filters — combinator
# ---------------------------------------------------------------------------


def test_apply_filters_defaults_are_passthrough() -> None:
    rows = [_mp(match_id="a"), _mp(match_id="b")]
    out = apply_filters(rows)
    assert {mp.match_id for mp in out} == {"a", "b"}


def test_apply_filters_period_and_agent() -> None:
    rows = [
        _mp(agent="Jett", started_at="2026-04-20T00:00:00+00:00", match_id="keep"),
        _mp(agent="Reyna", started_at="2026-04-20T00:00:00+00:00", match_id="wrong-agent"),
        _mp(agent="Jett", started_at="2020-01-01T00:00:00+00:00", match_id="too-old"),
    ]
    out = apply_filters(rows, period="all", agent="Jett")
    # period=all → no cutoff; only agent filter should fire.
    assert {mp.match_id for mp in out} == {"keep", "too-old"}


def test_apply_filters_all_filters_combined() -> None:
    """Each of the non-period filters eliminates exactly one row.

    period="all" is intentional — period filtering is tested separately.
    The combined test proves all other predicates compose correctly without
    interference. One row ("keep") satisfies every constraint; each other
    row fails exactly one filter so we know every branch is exercised.
    """
    rows = [
        _mp(
            agent="Jett",
            map_name="Ascent",
            queue_id="competitive",
            won=True,
            competitive_tier=12,
            started_at="2026-04-20T00:00:00+00:00",
            match_id="keep",
        ),
        _mp(
            agent="Reyna",
            map_name="Ascent",
            queue_id="competitive",
            won=True,
            competitive_tier=12,
            started_at="2026-04-20T00:00:00+00:00",
            match_id="wrong-agent",
        ),
        _mp(
            agent="Jett",
            map_name="Haven",
            queue_id="competitive",
            won=True,
            competitive_tier=12,
            started_at="2026-04-20T00:00:00+00:00",
            match_id="wrong-map",
        ),
        _mp(
            agent="Jett",
            map_name="Ascent",
            queue_id="unrated",
            won=True,
            competitive_tier=12,
            started_at="2026-04-20T00:00:00+00:00",
            match_id="wrong-queue",
        ),
        _mp(
            agent="Jett",
            map_name="Ascent",
            queue_id="competitive",
            won=False,
            competitive_tier=12,
            started_at="2026-04-20T00:00:00+00:00",
            match_id="loss",
        ),
        _mp(
            agent="Jett",
            map_name="Ascent",
            queue_id="competitive",
            won=True,
            competitive_tier=6,
            started_at="2026-04-20T00:00:00+00:00",
            match_id="low-tier",
        ),
    ]
    out = apply_filters(
        rows,
        period="all",  # "all" → no period cutoff (tested separately)
        agent="Jett",
        map_name="Ascent",
        won=True,
        queue_id="competitive",
        min_tier=12,
        max_tier=24,
    )
    assert [mp.match_id for mp in out] == ["keep"]


def test_apply_filters_bad_period_raises_value_error() -> None:
    with pytest.raises(ValueError):
        apply_filters([_mp()], period="bad")


# ---------------------------------------------------------------------------
# split_by_result
# ---------------------------------------------------------------------------


def test_split_by_result_both_sides() -> None:
    rows = [
        _mp(won=True, match_id="w1"),
        _mp(won=False, match_id="l1"),
        _mp(won=True, match_id="w2"),
    ]
    wins, losses = split_by_result(rows)
    assert {mp.match_id for mp in wins} == {"w1", "w2"}
    assert [mp.match_id for mp in losses] == ["l1"]


def test_split_by_result_all_wins() -> None:
    rows = [_mp(won=True, match_id=f"w{i}") for i in range(3)]
    wins, losses = split_by_result(rows)
    assert len(wins) == 3
    assert losses == []


def test_split_by_result_all_losses() -> None:
    rows = [_mp(won=False, match_id=f"l{i}") for i in range(4)]
    wins, losses = split_by_result(rows)
    assert wins == []
    assert len(losses) == 4


def test_split_by_result_empty() -> None:
    wins, losses = split_by_result([])
    assert wins == []
    assert losses == []


def test_split_by_result_preserves_order() -> None:
    """Neither half should be re-sorted — original order is preserved."""
    rows = [
        _mp(won=True, started_at="2026-04-01T00:00:00+00:00", match_id="w-old"),
        _mp(won=True, started_at="2026-04-20T00:00:00+00:00", match_id="w-new"),
    ]
    wins, _ = split_by_result(rows)
    assert [mp.match_id for mp in wins] == ["w-old", "w-new"]


# ---------------------------------------------------------------------------
# recent_form
# ---------------------------------------------------------------------------


def test_recent_form_returns_n_most_recent() -> None:
    rows = [
        _mp(started_at="2026-04-20T00:00:00+00:00", match_id="newest"),
        _mp(started_at="2026-04-18T00:00:00+00:00", match_id="middle"),
        _mp(started_at="2026-04-10T00:00:00+00:00", match_id="oldest"),
    ]
    out = recent_form(rows, 2)
    assert [mp.match_id for mp in out] == ["newest", "middle"]


def test_recent_form_n_larger_than_rows_returns_all() -> None:
    rows = [_mp(match_id=f"m{i}") for i in range(3)]
    out = recent_form(rows, 10)
    assert len(out) == 3


def test_recent_form_sorted_newest_first() -> None:
    """Input is in arbitrary order; output must be newest-first."""
    rows = [
        _mp(started_at="2026-04-10T00:00:00+00:00", match_id="old"),
        _mp(started_at="2026-04-20T00:00:00+00:00", match_id="new"),
        _mp(started_at="2026-04-15T00:00:00+00:00", match_id="mid"),
    ]
    out = recent_form(rows, 3)
    assert [mp.match_id for mp in out] == ["new", "mid", "old"]


def test_recent_form_n1_returns_single_newest() -> None:
    rows = [
        _mp(started_at="2026-04-20T00:00:00+00:00", match_id="new"),
        _mp(started_at="2026-04-10T00:00:00+00:00", match_id="old"),
    ]
    out = recent_form(rows, 1)
    assert [mp.match_id for mp in out] == ["new"]


def test_recent_form_zero_raises() -> None:
    with pytest.raises(ValueError, match="n must be positive"):
        recent_form([_mp()], 0)


def test_recent_form_negative_raises() -> None:
    with pytest.raises(ValueError):
        recent_form([_mp()], -5)

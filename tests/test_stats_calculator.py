"""Stats calculator — pure-function aggregation tests.

We build MatchPlayer objects in memory (no DB) with the `_mp` helper.
SQLAlchemy declarative classes accept arbitrary kwargs at construction
time; column defaults only fire on flush, so we set every field the
calculator reads. Anything the calculator doesn't touch (competitive_tier,
afk_rounds, plants, defuses on tests that don't care) is set to 0.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from valocoach.data.orm_models import Match, MatchPlayer
from valocoach.stats.calculator import (
    AgentStats,
    MapStats,
    PlayerStats,
    compute_per_agent,
    compute_per_map,
    compute_player_stats,
)

# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------


def _mp(
    *,
    agent: str = "Jett",
    map_name: str = "Ascent",
    won: bool = True,
    score: int = 0,
    rounds_played: int = 20,
    kills: int = 0,
    deaths: int = 0,
    assists: int = 0,
    headshots: int = 0,
    bodyshots: int = 0,
    legshots: int = 0,
    damage_dealt: int = 0,
    damage_received: int = 0,
    first_bloods: int = 0,
    first_deaths: int = 0,
    plants: int = 0,
    defuses: int = 0,
    match_id: str = "m-1",
    puuid: str = "p-tracked",
) -> MatchPlayer:
    """Build an in-memory MatchPlayer with its Match relationship set.

    The calculator reads mp.match.map_name in compute_per_map, so we
    always attach a Match — even for tests that don't group by map.
    """
    match = Match(
        match_id=match_id,
        map_name=map_name,
        map_id=None,
        queue_id="competitive",
        is_ranked=True,
        game_version=None,
        game_length_secs=0,
        season_short=None,
        region="na",
        rounds_played=rounds_played,
        red_score=0,
        blue_score=0,
        winning_team=None,
        started_at="2026-04-19T18:00:00+00:00",
    )
    mp = MatchPlayer(
        match_id=match_id,
        puuid=puuid,
        agent_name=agent,
        agent_id=None,
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
        damage_received=damage_received,
        first_bloods=first_bloods,
        first_deaths=first_deaths,
        plants=plants,
        defuses=defuses,
        afk_rounds=0,
        rounds_in_spawn=0,
        competitive_tier=None,
        started_at="2026-04-19T18:00:00+00:00",
    )
    # selectin relationships aren't populated on in-memory instances;
    # assign explicitly so compute_per_map works.
    mp.match = match
    return mp


# ---------------------------------------------------------------------------
# compute_player_stats — edge cases
# ---------------------------------------------------------------------------


def test_empty_input_returns_all_zeros() -> None:
    """No rows → all-zero PlayerStats, no crash on zero division."""
    stats = compute_player_stats([])
    assert stats.matches == 0
    assert stats.rounds == 0
    assert stats.win_rate == 0.0
    assert stats.kd == 0.0
    assert stats.hs_pct == 0.0
    assert stats.acs == 0.0
    assert stats.adr == 0.0


def test_single_match_basic_fields() -> None:
    stats = compute_player_stats(
        [
            _mp(
                won=True,
                rounds_played=20,
                score=5000,
                kills=20,
                deaths=10,
                assists=5,
                headshots=30,
                bodyshots=60,
                legshots=10,
                damage_dealt=3000,
                damage_received=2000,
                first_bloods=3,
                first_deaths=1,
                plants=2,
                defuses=1,
            )
        ]
    )
    assert stats.matches == 1
    assert stats.rounds == 20
    assert stats.wins == 1
    assert stats.losses == 0
    assert stats.win_rate == 1.0
    assert stats.acs == 250.0  # 5000/20
    assert stats.adr == 150.0  # 3000/20
    assert stats.kd == 2.0
    assert stats.kda == 2.5  # (20+5)/10
    assert stats.hs_pct == pytest.approx(0.3)  # 30 / (30+60+10)
    assert stats.fb_rate == pytest.approx(0.15)
    assert stats.fd_rate == pytest.approx(0.05)
    assert stats.fb_diff == 2
    assert stats.plants == 2
    assert stats.defuses == 1


# ---------------------------------------------------------------------------
# Round weighting — the key design decision
# ---------------------------------------------------------------------------


def test_acs_is_round_weighted_not_mean_of_means() -> None:
    """A 30-round match should count more than a 14-round stomp.

    Match A: 6000 score over 30 rounds → 200 ACS
    Match B: 4200 score over 14 rounds → 300 ACS

    Mean of means would be (200 + 300) / 2 = 250.
    Round-weighted: (6000 + 4200) / (30 + 14) = 10200 / 44 ≈ 231.82.

    The right answer is round-weighted.
    """
    stats = compute_player_stats(
        [
            _mp(score=6000, rounds_played=30, match_id="a"),
            _mp(score=4200, rounds_played=14, match_id="b"),
        ]
    )
    assert stats.acs == pytest.approx(10200 / 44)
    assert stats.acs != 250.0


def test_adr_is_round_weighted() -> None:
    stats = compute_player_stats(
        [
            _mp(damage_dealt=3000, rounds_played=30, match_id="a"),
            _mp(damage_dealt=2800, rounds_played=14, match_id="b"),
        ]
    )
    assert stats.adr == pytest.approx(5800 / 44)


# ---------------------------------------------------------------------------
# Zero-division guards
# ---------------------------------------------------------------------------


def test_zero_deaths_does_not_crash() -> None:
    stats = compute_player_stats(
        [_mp(kills=20, deaths=0, assists=3, rounds_played=13)]
    )
    assert stats.kd == 0.0
    assert stats.kda == 0.0


def test_zero_shots_does_not_crash() -> None:
    stats = compute_player_stats(
        [_mp(headshots=0, bodyshots=0, legshots=0, rounds_played=13)]
    )
    assert stats.hs_pct == 0.0


def test_zero_rounds_does_not_crash() -> None:
    """Forfeited / null match with 0 rounds — per-round stats go to 0."""
    stats = compute_player_stats([_mp(score=100, damage_dealt=200, rounds_played=0)])
    assert stats.acs == 0.0
    assert stats.adr == 0.0
    assert stats.fb_rate == 0.0


# ---------------------------------------------------------------------------
# Win rate
# ---------------------------------------------------------------------------


def test_win_rate_across_matches() -> None:
    stats = compute_player_stats(
        [
            _mp(won=True, match_id="a"),
            _mp(won=True, match_id="b"),
            _mp(won=False, match_id="c"),
            _mp(won=False, match_id="d"),
        ]
    )
    assert stats.matches == 4
    assert stats.wins == 2
    assert stats.losses == 2
    assert stats.win_rate == 0.5


# ---------------------------------------------------------------------------
# First-blood differential
# ---------------------------------------------------------------------------


def test_fb_diff_can_be_negative() -> None:
    """Entry-frag negative — should surface, not clamp at zero."""
    stats = compute_player_stats(
        [
            _mp(first_bloods=1, first_deaths=4, match_id="a"),
            _mp(first_bloods=0, first_deaths=3, match_id="b"),
        ]
    )
    assert stats.first_bloods == 1
    assert stats.first_deaths == 7
    assert stats.fb_diff == -6


# ---------------------------------------------------------------------------
# compute_per_agent
# ---------------------------------------------------------------------------


def test_per_agent_groups_and_sorts_by_match_count() -> None:
    rows = [
        _mp(agent="Jett", match_id="a"),
        _mp(agent="Jett", match_id="b"),
        _mp(agent="Jett", match_id="c"),
        _mp(agent="Reyna", match_id="d"),
        _mp(agent="Omen", match_id="e"),
        _mp(agent="Omen", match_id="f"),
    ]
    result = compute_per_agent(rows)
    assert [a.agent for a in result] == ["Jett", "Omen", "Reyna"]
    assert result[0].stats.matches == 3
    assert result[1].stats.matches == 2
    assert result[2].stats.matches == 1
    assert all(isinstance(a, AgentStats) for a in result)


def test_per_agent_tie_sorts_alphabetically() -> None:
    """Equal match counts — break ties by agent name for stable ordering."""
    rows = [
        _mp(agent="Sova", match_id="a"),
        _mp(agent="Jett", match_id="b"),
        _mp(agent="Reyna", match_id="c"),
    ]
    result = compute_per_agent(rows)
    assert [a.agent for a in result] == ["Jett", "Reyna", "Sova"]


def test_per_agent_stats_are_independent() -> None:
    """Per-agent aggregation shouldn't leak data across agents."""
    rows = [
        _mp(agent="Jett", kills=20, deaths=5, rounds_played=20, match_id="a"),
        _mp(agent="Reyna", kills=10, deaths=15, rounds_played=20, match_id="b"),
    ]
    result = compute_per_agent(rows)
    jett = next(a for a in result if a.agent == "Jett")
    reyna = next(a for a in result if a.agent == "Reyna")
    assert jett.stats.kills == 20
    assert jett.stats.kd == 4.0
    assert reyna.stats.kills == 10
    assert reyna.stats.kd == pytest.approx(10 / 15)


# ---------------------------------------------------------------------------
# compute_per_map
# ---------------------------------------------------------------------------


def test_per_map_groups_and_sorts() -> None:
    rows = [
        _mp(map_name="Ascent", match_id="a"),
        _mp(map_name="Ascent", match_id="b"),
        _mp(map_name="Lotus", match_id="c"),
    ]
    result = compute_per_map(rows)
    assert [m.map_name for m in result] == ["Ascent", "Lotus"]
    assert result[0].stats.matches == 2
    assert result[1].stats.matches == 1
    assert all(isinstance(m, MapStats) for m in result)


def test_per_map_handles_missing_match_relationship() -> None:
    """Defensive: if mp.match is None, bucket under 'Unknown' rather than crash."""
    mp = _mp(match_id="a")
    mp.match = None
    result = compute_per_map([mp])
    assert result[0].map_name == "Unknown"


# ---------------------------------------------------------------------------
# Return type sanity
# ---------------------------------------------------------------------------


def test_player_stats_is_frozen() -> None:
    """Dataclass is frozen — accidental mutation should raise."""
    stats = compute_player_stats([_mp()])
    with pytest.raises(FrozenInstanceError):
        stats.matches = 99  # type: ignore[misc]


def test_returns_playerstats_instance() -> None:
    assert isinstance(compute_player_stats([]), PlayerStats)

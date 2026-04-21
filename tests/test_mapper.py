"""Unit tests for mapper.py — pure Python, no DB required.

These tests exercise the API-shape → ORM-shape logic directly.  No session,
no fixtures from conftest beyond the shared Pydantic model fixtures.

Because mapper functions are pure (no I/O, no async), every test is synchronous.
"""

from __future__ import annotations

import json

from valocoach.data.mapper import (
    _compute_impact,
    _tier_int,
    match_from_details,
    player_from_account_mmr,
)
from valocoach.data.orm_models import Kill, Match, OrmMatchPlayer, Player

# Mirror the constants defined in conftest — avoids importing from tests package
PUUID = "20905543-1b42-5f6f-8435-ab284a0094f8"
MATCH_ID = "b0c012f7-9a68-46d1-a527-32783a190a5c"
ENEMY_PUUID = "enemy-puuid-0001"


# ---------------------------------------------------------------------------
# _tier_int
# ---------------------------------------------------------------------------


def test_tier_int_numeric_string():
    assert _tier_int("13") == 13


def test_tier_int_zero():
    assert _tier_int("0") == 0


def test_tier_int_empty_string():
    assert _tier_int("") is None


def test_tier_int_non_numeric():
    assert _tier_int("unranked") is None


# ---------------------------------------------------------------------------
# _compute_impact
# ---------------------------------------------------------------------------


def test_compute_impact_first_bloods_and_deaths(match_details):
    impact = _compute_impact(match_details)
    # Fixture: round 0 first kill = dipp kills Yoursaviour01 (10 000 ms)
    #          round 1 first kill = Yoursaviour01 kills dipp  (8 000 ms)
    assert impact.first_bloods[ENEMY_PUUID] == 1
    assert impact.first_bloods[PUUID] == 1
    assert impact.first_deaths[PUUID] == 1
    assert impact.first_deaths[ENEMY_PUUID] == 1


def test_compute_impact_plants_and_defuses(match_details):
    impact = _compute_impact(match_details)
    # round 0 plant → dipp; round 1 defuse → Yoursaviour01
    assert impact.plants[ENEMY_PUUID] == 1
    assert impact.plants[PUUID] == 0
    assert impact.defuses[PUUID] == 1
    assert impact.defuses[ENEMY_PUUID] == 0


def test_compute_impact_unknown_puuid_is_zero(match_details):
    impact = _compute_impact(match_details)
    assert impact.first_bloods["no-such-puuid"] == 0


# ---------------------------------------------------------------------------
# match_from_details — Match row
# ---------------------------------------------------------------------------


def test_match_from_details_returns_match(match_details):
    assert isinstance(match_from_details(match_details), Match)


def test_match_metadata_fields(match_details):
    m = match_from_details(match_details)
    assert m.match_id == MATCH_ID
    assert m.map_name == "Lotus"
    assert m.map_id == "map-lotus-id"
    assert m.queue_id == "competitive"
    assert m.is_ranked is True
    assert m.game_length_secs == 1462  # 1_462_000 ms ÷ 1000
    assert m.rounds_played == 2
    assert m.red_score == 9
    assert m.blue_score == 8
    assert m.winning_team == "Red"
    assert m.started_at == "2026-04-19T18:00:00+00:00"
    assert m.season_short == "EPISODE 9 ACT 1"
    assert m.region == "na"


def test_match_null_started_at_becomes_empty_string(match_details):
    match_details.metadata.started_at = None
    assert match_from_details(match_details).started_at == ""


# ---------------------------------------------------------------------------
# match_from_details — players
# ---------------------------------------------------------------------------


def test_match_player_count(match_details):
    assert len(match_from_details(match_details).players) == 2


def test_match_player_stats(match_details):
    m = match_from_details(match_details)
    me = next(p for p in m.players if p.puuid == PUUID)

    assert isinstance(me, OrmMatchPlayer)
    assert me.agent_name == "Jett"
    assert me.agent_id == "jett-id"
    assert me.team == "Blue"
    assert me.won is False
    assert me.score == 3811
    assert me.kills == 14
    assert me.deaths == 12
    assert me.assists == 2
    assert me.headshots == 16
    assert me.damage_dealt == 2400
    assert me.damage_received == 1800
    assert me.rounds_played == 2
    assert me.competitive_tier == 12
    assert me.afk_rounds == 0
    assert me.rounds_in_spawn == 1


def test_match_player_impact_stats(match_details):
    m = match_from_details(match_details)
    me = next(p for p in m.players if p.puuid == PUUID)
    enemy = next(p for p in m.players if p.puuid == ENEMY_PUUID)

    assert me.first_bloods == 1
    assert me.first_deaths == 1
    assert me.defuses == 1
    assert me.plants == 0

    assert enemy.first_bloods == 1
    assert enemy.first_deaths == 1
    assert enemy.plants == 1
    assert enemy.defuses == 0


def test_match_player_won_assigned_correctly(match_details):
    m = match_from_details(match_details)
    enemy = next(p for p in m.players if p.puuid == ENEMY_PUUID)
    assert enemy.won is True  # Red team won


# ---------------------------------------------------------------------------
# match_from_details — rounds
# ---------------------------------------------------------------------------


def test_round_count(match_details):
    assert len(match_from_details(match_details).rounds) == 2


def test_round_fields(match_details):
    m = match_from_details(match_details)
    r0, r1 = sorted(m.rounds, key=lambda r: r.round_number)

    assert r0.round_number == 0
    assert r0.winning_team == "Red"
    assert r0.bomb_planted is True
    assert r0.plant_site == "A"
    assert r0.bomb_defused is False

    assert r1.round_number == 1
    assert r1.winning_team == "Blue"
    assert r1.bomb_planted is False
    assert r1.bomb_defused is True


# ---------------------------------------------------------------------------
# match_from_details — kills
# ---------------------------------------------------------------------------


def test_kill_count(match_details):
    m = match_from_details(match_details)
    assert sum(len(r.kills) for r in m.rounds) == 3


def test_kill_fields(match_details):
    m = match_from_details(match_details)
    r0 = next(r for r in m.rounds if r.round_number == 0)
    first = min(r0.kills, key=lambda k: k.time_in_round_ms)

    assert isinstance(first, Kill)
    assert first.killer_puuid == ENEMY_PUUID
    assert first.victim_puuid == PUUID
    assert first.weapon_name == "Vandal"
    assert first.is_headshot is False
    assert json.loads(first.assistants_json) == []


def test_kill_with_assistants(match_details):
    m = match_from_details(match_details)
    r1 = next(r for r in m.rounds if r.round_number == 1)
    assert len(r1.kills) == 1
    assert "ally-puuid" in json.loads(r1.kills[0].assistants_json)


def test_ability_kill_weapon_name_is_none(match_details):
    """Kills with null weapon.name (ability / ultimate) are stored as NULL."""
    from valocoach.data.api_models import MatchDetailsKill, _PlayerRef, _Ref

    match_details.kills.append(
        MatchDetailsKill(
            round=0,
            time_in_round_in_ms=99_000,
            time_in_match_in_ms=99_000,
            killer=_PlayerRef(puuid="x", team="Red"),
            victim=_PlayerRef(puuid="y", team="Blue"),
            weapon=_Ref(id="", name=None),  # ability kill — no weapon name
        )
    )
    m = match_from_details(match_details)
    all_kills = [k for r in m.rounds for k in r.kills]
    ability_kill = next(k for k in all_kills if k.killer_puuid == "x")
    assert ability_kill.weapon_name is None


# ---------------------------------------------------------------------------
# player_from_account_mmr
# ---------------------------------------------------------------------------


def test_player_from_account_mmr(account_data, mmr_data):
    p = player_from_account_mmr(account_data, mmr_data)

    assert isinstance(p, Player)
    assert p.puuid == PUUID
    assert p.riot_name == "Yoursaviour01"
    assert p.riot_tag == "SK04"
    assert p.region == "na"
    assert p.account_level == 240
    assert p.current_tier == 12
    assert p.current_tier_patched == "Gold 1"
    assert p.current_rr == 0
    assert p.elo == 900
    assert p.peak_tier == 14
    assert p.peak_tier_patched == "Gold 3"

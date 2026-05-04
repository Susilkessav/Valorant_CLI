"""Unit tests for mapper.py — pure Python, no DB required.

These tests exercise the API-shape → ORM-shape logic directly.  No session,
no fixtures from conftest beyond the shared Pydantic model fixtures.

Because mapper functions are pure (no I/O, no async), every test is synchronous.
"""

from __future__ import annotations

import json

import pytest

from valocoach.core.exceptions import MapperError
from valocoach.data.mapper import (
    _compute_impact,
    _econ_int,
    _tier_int,
    match_from_details,
    player_from_account_mmr,
)
from valocoach.data.orm_models import (
    Kill,
    Match,
    MetaCache,
    OrmMatchPlayer,
    PatchVersion,
    Player,
    Round,
    SyncLog,
)

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


def test_match_null_started_at_raises_mapper_error(match_details):
    """A match with no started_at must be rejected, not stored as ``""``.

    Empty string sorts lexicographically before every valid ISO timestamp,
    which would silently corrupt ``ORDER BY started_at`` queries.  The
    mapper raises :class:`MapperError`; the sync pipeline catches it and
    skips the offending match without poisoning the rest of the batch.
    """
    match_details.metadata.started_at = None
    with pytest.raises(MapperError, match="started_at"):
        match_from_details(match_details)


def test_match_empty_started_at_raises_mapper_error(match_details):
    """An empty-string started_at is also rejected (defence in depth)."""
    match_details.metadata.started_at = ""
    with pytest.raises(MapperError, match="started_at"):
        match_from_details(match_details)


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


# ---------------------------------------------------------------------------
# _econ_int — line 73 else-None branch
# ---------------------------------------------------------------------------


def test_econ_int_returns_int_for_numeric_leaf():
    """_econ_int returns int when leaf is a number (line 73 True branch)."""
    assert _econ_int({"spent": {"overall": 3900}}, "spent", "overall") == 3900


def test_econ_int_returns_none_for_string_leaf():
    """_econ_int returns None when leaf is a non-numeric string (line 73 else-None branch).

    Coverage target: mapper.py line 73 — `else None` when leaf not int|float.
    """
    result = _econ_int({"spent": {"overall": "not_a_number"}}, "spent", "overall")
    assert result is None


def test_econ_int_returns_none_for_none_leaf():
    """_econ_int returns None when the key is missing (leaf=None)."""
    assert _econ_int({"spent": {}}, "spent", "overall") is None


# ---------------------------------------------------------------------------
# match_from_details — kill in unknown round (lines 200-201)
# ---------------------------------------------------------------------------


def test_kill_in_unknown_round_is_skipped(match_details) -> None:
    """A kill that references a round ID not in the rounds list is silently skipped.

    Coverage target: mapper.py lines 200-201 — `log.debug("kill in unknown round")` +
    `continue`.

    Approach: append a kill referencing round 99 which doesn't exist in the rounds list.
    """
    from valocoach.data.api_models import MatchDetailsKill, _PlayerRef, _Ref

    # Add a kill that references round 99 — not present in match_details.rounds (IDs 0, 1)
    phantom_kill = MatchDetailsKill(
        round=99,  # out-of-range round ID
        time_in_round_in_ms=5_000,
        time_in_match_in_ms=500_000,
        killer=_PlayerRef(puuid=ENEMY_PUUID, name="dipp", tag="100T", team="Red"),
        victim=_PlayerRef(puuid=PUUID, name="Yoursaviour01", tag="SK04", team="Blue"),
        weapon=_Ref(id="vandal-id", name="Vandal"),
    )
    # Pydantic models are immutable — rebuild kills list with the extra kill
    modified = match_details.model_copy(
        update={"kills": list(match_details.kills) + [phantom_kill]}
    )

    # Should not raise; the unknown-round kill is simply skipped
    result = match_from_details(modified)
    all_kills = [k for r in result.rounds for k in r.kills]

    # The phantom kill (round=99) must NOT appear in the ORM kill list
    phantom_kill_in_orm = [k for k in all_kills if k.round_number == 99]
    assert phantom_kill_in_orm == []

    # The regular kills (round 0 and 1) are still present
    assert len(all_kills) >= 1


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


# ---------------------------------------------------------------------------
# ORM model __repr__ methods (lines 71, 115, 201, 240, 271, 296, 329, 347)
# ---------------------------------------------------------------------------
#
# None of the ORM __repr__ methods are exercised elsewhere.
# These tests call repr() on in-memory instances — no DB session required.


class TestOrmRepr:
    """Cover __repr__ on every ORM model class (orm_models.py)."""

    def test_player_repr(self) -> None:
        """Player.__repr__ (line 71)."""
        p = Player(
            puuid="abc123",
            riot_name="Yoursaviour01",
            riot_tag="SK04",
            region="na",
            current_tier_patched="Gold 1",
            elo=925,
        )
        r = repr(p)
        assert "Player" in r
        assert "Yoursaviour01" in r

    def test_match_repr(self) -> None:
        """Match.__repr__ (line 115)."""
        m = Match(
            match_id="aaaabbbbcccc",
            map_name="Ascent",
            queue_id="competitive",
            is_ranked=True,
            game_length_secs=1800,
            rounds_played=20,
            red_score=13,
            blue_score=7,
            started_at="2026-04-19T18:00:00+00:00",
        )
        r = repr(m)
        assert "Match" in r
        assert "Ascent" in r

    def test_orm_match_player_repr(self) -> None:
        """OrmMatchPlayer.__repr__ (line 201)."""
        mp = OrmMatchPlayer(
            match_id="x",
            puuid="p",
            agent_name="Jett",
            team="Blue",
            won=True,
            score=5000,
            kills=20,
            deaths=10,
            assists=5,
            rounds_played=20,
            headshots=30,
            bodyshots=60,
            legshots=10,
            damage_dealt=3000,
            damage_received=2000,
            first_bloods=3,
            first_deaths=1,
            plants=0,
            defuses=0,
            afk_rounds=0,
            rounds_in_spawn=0,
            started_at="2026-04-19T18:00:00+00:00",
        )
        r = repr(mp)
        assert "MatchPlayer" in r
        assert "Jett" in r

    def test_round_repr(self) -> None:
        """Round.__repr__ (line 240)."""
        rnd = Round(
            match_id="m1",
            round_number=5,
            winning_team="Blue",
            result_code="Elimination",
            bomb_planted=False,
            bomb_defused=False,
        )
        r = repr(rnd)
        assert "Round" in r
        assert "5" in r

    def test_kill_repr(self) -> None:
        """Kill.__repr__ (line 271)."""
        k = Kill(
            match_id="m1",
            round_number=3,
            time_in_round_ms=5000,
            killer_puuid="killer-puuid-0001",
            victim_puuid="victim-puuid-0002",
            weapon_name="Vandal",
            is_headshot=True,
            assistants_json="[]",
        )
        r = repr(k)
        assert "Kill" in r
        assert "round=3" in r

    def test_sync_log_repr(self) -> None:
        """SyncLog.__repr__ (line 296)."""
        sl = SyncLog(
            puuid="synced-puuid-0001",
            matches_fetched=10,
            matches_new=3,
            error=None,
        )
        r = repr(sl)
        assert "SyncLog" in r
        assert "synced-p" in r  # puuid[:8]

    def test_meta_cache_repr(self) -> None:
        """MetaCache.__repr__ (line 329)."""
        mc = MetaCache(
            url="https://valorant.fandom.com/wiki/Jett",
            source="web",
            content_hash="abc12345",
            ttl_tier="stable",
            expires_at="2026-06-01T00:00:00+00:00",
            content_text="Jett is a duelist...",
        )
        r = repr(mc)
        assert "MetaCache" in r
        assert "stable" in r

    def test_patch_version_repr(self) -> None:
        """PatchVersion.__repr__ (line 347)."""
        pv = PatchVersion(
            game_version="release-09.00",
            detected_at="2026-04-20T00:00:00+00:00",
        )
        r = repr(pv)
        assert "PatchVersion" in r
        assert "release-09.00" in r

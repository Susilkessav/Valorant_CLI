"""Tests for the async repository layer using an isolated SQLite DB."""

from __future__ import annotations

from sqlalchemy import select

from valocoach.data.models import (
    MatchData,
    MatchMetadata,
    MatchPlayers,
    MatchTeams,
    PlayerStats,
    TeamResult,
)
from valocoach.data.models import (
    MatchPlayer as MatchPlayerModel,
)
from valocoach.data.orm_models import Match, MatchPlayer, Player
from valocoach.data.repository import (
    close_stale_syncs,
    complete_sync,
    get_match,
    get_player,
    get_player_by_name,
    get_recent_matches,
    get_recent_matches_full,
    match_exists,
    start_sync,
    upsert_match,
    upsert_match_details,
    upsert_player,
)

PUUID = "20905543-1b42-5f6f-8435-ab284a0094f8"
MATCH_ID = "b0c012f7-9a68-46d1-a527-32783a190a5c"
ENEMY_PUUID = "enemy-puuid-0001"


async def _all_players(db_session, match_id: str) -> list[MatchPlayer]:
    result = await db_session.scalars(select(MatchPlayer).where(MatchPlayer.match_id == match_id))
    return list(result.all())


async def _match_player(db_session, *, match_id: str, puuid: str) -> MatchPlayer:
    result = await db_session.scalars(
        select(MatchPlayer).where(MatchPlayer.match_id == match_id, MatchPlayer.puuid == puuid)
    )
    return result.one()


# ---------------------------------------------------------------------------
# upsert_player
# ---------------------------------------------------------------------------


async def test_upsert_player_creates_row(db_session, account_data, mmr_data):
    player = await upsert_player(db_session, account_data, mmr_data)
    await db_session.flush()

    assert isinstance(player, Player)
    assert player.puuid == PUUID
    assert player.riot_name == "Yoursaviour01"
    assert player.riot_tag == "SK04"
    assert player.current_tier_patched == "Gold 1"
    assert player.elo == 900
    assert player.current_rr == 0
    assert player.peak_tier_patched == "Gold 3"


async def test_upsert_player_is_idempotent(db_session, account_data, mmr_data):
    """Calling upsert_player twice should not create a duplicate."""
    await upsert_player(db_session, account_data, mmr_data)
    await db_session.flush()

    mmr_data.current_data.elo = 950
    await upsert_player(db_session, account_data, mmr_data)
    await db_session.flush()

    result = await db_session.scalars(select(Player).where(Player.puuid == PUUID))
    rows = list(result.all())
    assert len(rows) == 1
    assert rows[0].elo == 950


# ---------------------------------------------------------------------------
# get_player / get_player_by_name
# ---------------------------------------------------------------------------


async def test_get_player_returns_none_when_missing(db_session):
    assert await get_player(db_session, "nonexistent-puuid") is None


async def test_get_player_returns_correct_row(db_session, account_data, mmr_data):
    await upsert_player(db_session, account_data, mmr_data)
    await db_session.flush()

    player = await get_player(db_session, PUUID)
    assert player is not None
    assert player.riot_name == "Yoursaviour01"


async def test_get_player_by_name_case_insensitive(db_session, account_data, mmr_data):
    await upsert_player(db_session, account_data, mmr_data)
    await db_session.flush()

    player = await get_player_by_name(db_session, "yoursaviour01", "sk04")
    assert player is not None
    assert player.puuid == PUUID


async def test_get_player_by_name_not_found(db_session):
    assert await get_player_by_name(db_session, "ghost", "0000") is None


# ---------------------------------------------------------------------------
# upsert_match / match_exists
# ---------------------------------------------------------------------------


async def test_upsert_match_creates_match_and_players(db_session, match_data):
    result = await upsert_match(db_session, match_data)
    await db_session.flush()

    assert result is not None
    assert isinstance(result, Match)
    assert result.match_id == MATCH_ID
    assert result.map_name == "Lotus"
    assert result.rounds_played == 17
    assert result.is_ranked is True
    assert result.winning_team == "Red"
    assert result.red_score == 9
    assert result.blue_score == 8

    players = await _all_players(db_session, MATCH_ID)
    assert len(players) == 2


async def test_upsert_match_stores_correct_stats(db_session, match_data):
    await upsert_match(db_session, match_data)
    await db_session.flush()

    me = await _match_player(db_session, match_id=MATCH_ID, puuid=PUUID)
    assert me.kills == 14
    assert me.deaths == 14
    assert me.agent_name == "Jett"
    assert me.won is False
    assert me.kd_ratio == 1.0
    assert me.headshot_pct > 0
    assert me.afk_rounds == 1
    assert me.rounds_in_spawn == 2


async def test_upsert_match_stores_won_correctly(db_session, match_data):
    await upsert_match(db_session, match_data)
    await db_session.flush()

    result = await db_session.scalars(
        select(MatchPlayer).where(
            MatchPlayer.match_id == MATCH_ID,
            MatchPlayer.agent_name == "Neon",
        )
    )
    red = result.one()
    assert red.won is True


async def test_upsert_match_stores_started_at_as_iso_string(db_session, match_data):
    await upsert_match(db_session, match_data)
    await db_session.flush()

    match = await db_session.get(Match, MATCH_ID)
    assert match is not None
    assert isinstance(match.started_at, str)
    assert match.started_at.startswith("2026-")
    assert "+00:00" in match.started_at


async def test_upsert_match_is_idempotent(db_session, match_data):
    first = await upsert_match(db_session, match_data)
    await db_session.flush()

    second = await upsert_match(db_session, match_data)

    assert first is not None
    assert second is None

    result = await db_session.scalars(select(Match).where(Match.match_id == MATCH_ID))
    rows = list(result.all())
    assert len(rows) == 1


async def test_match_exists_true(db_session, match_data):
    await upsert_match(db_session, match_data)
    await db_session.flush()
    assert await match_exists(db_session, MATCH_ID) is True


async def test_match_exists_false(db_session):
    assert await match_exists(db_session, "does-not-exist") is False


# ---------------------------------------------------------------------------
# get_recent_matches
# ---------------------------------------------------------------------------


async def test_get_recent_matches_returns_empty_for_unknown_puuid(db_session):
    assert await get_recent_matches(db_session, "unknown") == []


async def test_get_recent_matches_ordered_newest_first(db_session, account_data, mmr_data):
    await upsert_player(db_session, account_data, mmr_data)
    await db_session.flush()

    def _make_match(match_id: str, game_start: int) -> MatchData:
        return MatchData(
            metadata=MatchMetadata(
                match_id=match_id,
                map_name="Ascent",
                mode="Competitive",
                queue_id="competitive",
                rounds_played=13,
                game_length_secs=1200,
                game_start=game_start,
                region="na",
            ),
            players=MatchPlayers(
                all_players=[
                    MatchPlayerModel(
                        puuid=PUUID,
                        name="Yoursaviour01",
                        tag="SK04",
                        team="Blue",
                        character="Sage",
                        stats=PlayerStats(kills=10, deaths=8, assists=2),
                    )
                ]
            ),
            teams=MatchTeams(
                red=TeamResult(has_won=False),
                blue=TeamResult(has_won=True),
            ),
        )

    await upsert_match(db_session, _make_match("older-match", game_start=1_000_000))
    await upsert_match(db_session, _make_match("newer-match", game_start=2_000_000))
    await db_session.flush()

    rows = await get_recent_matches(db_session, PUUID, limit=10)
    assert len(rows) == 2
    assert rows[0].match.started_at > rows[1].match.started_at


# ---------------------------------------------------------------------------
# MatchPlayer computed properties
# ---------------------------------------------------------------------------


async def test_participant_acs(db_session, match_data):
    await upsert_match(db_session, match_data)
    await db_session.flush()

    me = await _match_player(db_session, match_id=MATCH_ID, puuid=PUUID)
    assert me.acs == round(4352 / 17)


async def test_get_match(db_session, match_data):
    await upsert_match(db_session, match_data)
    await db_session.flush()

    match = await get_match(db_session, MATCH_ID)
    assert match is not None
    assert match.map_name == "Lotus"
    assert len(match.players) == 2


# ---------------------------------------------------------------------------
# SyncLog
# ---------------------------------------------------------------------------


async def test_sync_log_start_and_complete(db_session):
    from valocoach.data.orm_models import SyncLog

    log = await start_sync(db_session, PUUID)
    assert log.id is not None
    assert log.completed_at is None

    complete_sync(db_session, log, matches_fetched=5, matches_new=3)
    await db_session.flush()

    row = await db_session.get(SyncLog, log.id)
    assert row.matches_fetched == 5
    assert row.matches_new == 3
    assert isinstance(row.completed_at, str)
    assert "T" in row.completed_at
    assert row.error is None


async def test_sync_log_records_error(db_session):
    from valocoach.data.orm_models import SyncLog

    log = await start_sync(db_session, PUUID)
    complete_sync(db_session, log, matches_fetched=0, matches_new=0, error="timeout")
    await db_session.flush()

    row = await db_session.get(SyncLog, log.id)
    assert row.error == "timeout"


# ---------------------------------------------------------------------------
# upsert_match_details (v4)
# ---------------------------------------------------------------------------


async def test_upsert_match_details_creates_match(db_session, match_details):
    result = await upsert_match_details(db_session, match_details)
    await db_session.flush()

    assert result is not None
    assert isinstance(result, Match)
    assert result.match_id == MATCH_ID
    assert result.map_name == "Lotus"
    assert result.queue_id == "competitive"
    assert result.is_ranked is True
    assert result.game_length_secs == 1462
    assert result.rounds_played == 2
    assert result.red_score == 9
    assert result.blue_score == 8
    assert result.winning_team == "Red"
    assert result.started_at == "2026-04-19T18:00:00+00:00"
    assert result.season_short == "EPISODE 9 ACT 1"
    assert result.region == "na"


async def test_upsert_match_details_player_stats(db_session, match_details):
    await upsert_match_details(db_session, match_details)
    await db_session.flush()

    me = await _match_player(db_session, match_id=MATCH_ID, puuid=PUUID)

    assert me.agent_name == "Jett"
    assert me.agent_id == "jett-id"
    assert me.team == "Blue"
    assert me.won is False
    assert me.score == 3811
    assert me.kills == 14
    assert me.deaths == 12
    assert me.assists == 2
    assert me.headshots == 16
    assert me.bodyshots == 40
    assert me.legshots == 1
    assert me.damage_dealt == 2400
    assert me.damage_received == 1800
    assert me.rounds_played == 2
    assert me.competitive_tier == 12
    assert me.afk_rounds == 0
    assert me.rounds_in_spawn == 1


async def test_upsert_match_details_enemy_won(db_session, match_details):
    await upsert_match_details(db_session, match_details)
    await db_session.flush()

    enemy = await _match_player(db_session, match_id=MATCH_ID, puuid=ENEMY_PUUID)
    assert enemy.won is True
    assert enemy.agent_name == "Neon"
    assert enemy.competitive_tier == 13


async def test_upsert_match_details_first_bloods(db_session, match_details):
    """dipp gets FB in round 0; Yoursaviour01 gets FB in round 1."""
    await upsert_match_details(db_session, match_details)
    await db_session.flush()

    me = await _match_player(db_session, match_id=MATCH_ID, puuid=PUUID)
    enemy = await _match_player(db_session, match_id=MATCH_ID, puuid=ENEMY_PUUID)

    # dipp: 1 first_blood (round 0), 1 first_death (round 1)
    assert enemy.first_bloods == 1
    assert enemy.first_deaths == 1
    # Yoursaviour01: 1 first_blood (round 1), 1 first_death (round 0)
    assert me.first_bloods == 1
    assert me.first_deaths == 1


async def test_upsert_match_details_plants_and_defuses(db_session, match_details):
    """dipp plants in round 0; Yoursaviour01 defuses in round 1."""
    await upsert_match_details(db_session, match_details)
    await db_session.flush()

    me = await _match_player(db_session, match_id=MATCH_ID, puuid=PUUID)
    enemy = await _match_player(db_session, match_id=MATCH_ID, puuid=ENEMY_PUUID)

    assert enemy.plants == 1
    assert enemy.defuses == 0
    assert me.plants == 0
    assert me.defuses == 1


async def test_upsert_match_details_rounds_stored(db_session, match_details):
    from sqlalchemy import select

    from valocoach.data.orm_models import Round

    await upsert_match_details(db_session, match_details)
    await db_session.flush()

    result = await db_session.scalars(
        select(Round).where(Round.match_id == MATCH_ID).order_by(Round.round_number)
    )
    rounds = list(result.all())

    assert len(rounds) == 2
    assert rounds[0].round_number == 0
    assert rounds[0].winning_team == "Red"
    assert rounds[0].bomb_planted is True
    assert rounds[0].plant_site == "A"
    assert rounds[0].bomb_defused is False

    assert rounds[1].round_number == 1
    assert rounds[1].winning_team == "Blue"
    assert rounds[1].bomb_planted is False
    assert rounds[1].bomb_defused is True


async def test_upsert_match_details_kills_stored(db_session, match_details):
    import json

    from sqlalchemy import select

    from valocoach.data.orm_models import Kill

    await upsert_match_details(db_session, match_details)
    await db_session.flush()

    result = await db_session.scalars(
        select(Kill)
        .where(Kill.match_id == MATCH_ID)
        .order_by(Kill.round_number, Kill.time_in_round_ms)
    )
    kills = list(result.all())

    assert len(kills) == 3

    # First kill in round 0 (10 000 ms): dipp kills Yoursaviour01
    first = kills[0]
    assert first.round_number == 0
    assert first.killer_puuid == ENEMY_PUUID
    assert first.victim_puuid == PUUID
    assert first.weapon_name == "Vandal"
    assert first.is_headshot is False
    assert json.loads(first.assistants_json) == []

    # First kill in round 1 has an assistant
    third = kills[2]
    assert third.round_number == 1
    assert third.killer_puuid == PUUID
    assistants = json.loads(third.assistants_json)
    assert "ally-puuid" in assistants


async def test_upsert_match_details_is_idempotent(db_session, match_details):
    first = await upsert_match_details(db_session, match_details)
    await db_session.flush()

    second = await upsert_match_details(db_session, match_details)

    assert first is not None
    assert second is None

    result = await db_session.scalars(select(Match).where(Match.match_id == MATCH_ID))
    assert len(list(result.all())) == 1


async def test_upsert_match_details_match_exists(db_session, match_details):
    assert await match_exists(db_session, MATCH_ID) is False
    await upsert_match_details(db_session, match_details)
    await db_session.flush()
    assert await match_exists(db_session, MATCH_ID) is True


# ---------------------------------------------------------------------------
# upsert_match — uncovered branches
# ---------------------------------------------------------------------------


async def test_upsert_match_skips_unavailable(db_session, match_data):
    """Lines 126-127: is_available=False returns None immediately."""
    match_data.is_available = False
    result = await upsert_match(db_session, match_data)
    assert result is None
    # Nothing should have been persisted.
    assert await match_exists(db_session, MATCH_ID) is False


async def test_upsert_match_integrity_error_path(db_session, match_data):
    """Lines 184-186: IntegrityError guard — duplicate insert returns None.

    We insert the match directly first, then patch session.get to return None
    so upsert_match passes the existence pre-check and tries to flush again.
    The resulting IntegrityError is caught and None is returned.
    """
    from unittest.mock import AsyncMock, patch

    # Store the match legitimately so the row exists in the DB.
    await upsert_match(db_session, match_data)
    await db_session.flush()

    # Now call again but trick the pre-check into thinking the row is absent.
    with patch.object(db_session, "get", new=AsyncMock(return_value=None)):
        result = await upsert_match(db_session, match_data)

    assert result is None


# ---------------------------------------------------------------------------
# upsert_match_details — IntegrityError path
# ---------------------------------------------------------------------------


async def test_upsert_match_details_integrity_error_path(db_session, match_details):
    """Lines 222-224: IntegrityError guard in upsert_match_details."""
    from unittest.mock import AsyncMock, patch

    # Store the match legitimately first.
    await upsert_match_details(db_session, match_details)
    await db_session.flush()

    # Bypass the pre-check to force the flush to hit the unique constraint.
    with patch.object(db_session, "get", new=AsyncMock(return_value=None)):
        result = await upsert_match_details(db_session, match_details)

    assert result is None


# ---------------------------------------------------------------------------
# get_recent_matches — queue_id=None branch (line 264→266)
# ---------------------------------------------------------------------------


async def test_get_recent_matches_no_queue_filter(db_session, match_data):
    """queue_id=None skips the queue_id WHERE clause — all modes returned."""
    await upsert_match(db_session, match_data)
    await db_session.flush()

    rows = await get_recent_matches(db_session, PUUID, queue_id=None)
    assert len(rows) == 1  # our competitive match is still returned


# ---------------------------------------------------------------------------
# get_recent_matches_full (lines 297-314)
# ---------------------------------------------------------------------------


async def test_get_recent_matches_full_returns_matches(db_session, match_details):
    """get_recent_matches_full returns Match trees with rounds pre-loaded."""
    await upsert_match_details(db_session, match_details)
    await db_session.flush()

    matches = await get_recent_matches_full(db_session, PUUID)
    assert len(matches) == 1
    assert matches[0].match_id == MATCH_ID
    # Rounds should be eagerly loaded.
    assert len(matches[0].rounds) == 2


async def test_get_recent_matches_full_no_queue_filter(db_session, match_details):
    """queue_id=None variant still returns the match."""
    await upsert_match_details(db_session, match_details)
    await db_session.flush()

    matches = await get_recent_matches_full(db_session, PUUID, queue_id=None)
    assert len(matches) == 1


async def test_get_recent_matches_full_empty(db_session):
    """Returns empty list for an unknown puuid."""
    matches = await get_recent_matches_full(db_session, "unknown-puuid")
    assert matches == []


# ---------------------------------------------------------------------------
# close_stale_syncs (repository-level unit — complements test_sync_cursor.py)
# ---------------------------------------------------------------------------


async def test_close_stale_syncs_closes_incomplete_rows(db_session):
    """close_stale_syncs marks all NULL-completed rows (except current) done."""
    from valocoach.data.orm_models import SyncLog

    # Two stale syncs.
    stale1 = await start_sync(db_session, PUUID)
    stale2 = await start_sync(db_session, PUUID)
    # Current sync.
    current = await start_sync(db_session, PUUID)
    await db_session.flush()

    count = await close_stale_syncs(db_session, PUUID, exclude_id=current.id)
    await db_session.flush()

    assert count == 2
    row1 = await db_session.get(SyncLog, stale1.id)
    row2 = await db_session.get(SyncLog, stale2.id)
    assert row1.error == "interrupted"
    assert row2.error == "interrupted"
    # Current must be untouched.
    cur_row = await db_session.get(SyncLog, current.id)
    assert cur_row.completed_at is None

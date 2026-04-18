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
    complete_sync,
    get_match,
    get_player,
    get_player_by_name,
    get_recent_matches,
    match_exists,
    start_sync,
    upsert_match,
    upsert_player,
)

PUUID = "20905543-1b42-5f6f-8435-ab284a0094f8"
MATCH_ID = "b0c012f7-9a68-46d1-a527-32783a190a5c"


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
                matchid=match_id,
                map="Ascent",
                mode="Competitive",
                mode_id="competitive",
                rounds_played=13,
                game_length=1200,
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

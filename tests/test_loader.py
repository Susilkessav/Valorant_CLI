"""Tests for valocoach.data.loader.

Two layers tested here:

  PlayerData                   — dataclass contract
  load_player_data_async(...)  — async core logic (mocked DB layer)
  load_player_data(...)        — sync wrapper (calls asyncio.run)

The DB is fully mocked:  ensure_db, session_scope, get_player_by_name,
get_recent_matches, get_recent_matches_full.  No SQLite file is created.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from valocoach.core.config import Settings
from valocoach.data.loader import (
    DEFAULT_LOAD_LIMIT,
    PlayerData,
    load_player_data,
    load_player_data_async,
)
from valocoach.data.orm_models import Match, MatchPlayer, Player

# ---------------------------------------------------------------------------
# Shared test fixtures / builders
# ---------------------------------------------------------------------------

PUUID = "test-puuid-0001"


def _settings(*, riot_name: str = "TestUser", riot_tag: str = "NA1") -> Settings:
    return Settings(
        riot_name=riot_name,
        riot_tag=riot_tag,
        riot_region="na",
        henrikdev_api_key="fake-key",
    )


def _player() -> Player:
    p = Player(
        puuid=PUUID,
        riot_name="TestUser",
        riot_tag="NA1",
        region="na",
        current_tier_patched="Gold 1",
    )
    return p


def _mp(match_id: str = "m-1") -> MatchPlayer:
    match = Match(
        match_id=match_id,
        map_name="Ascent",
        queue_id="competitive",
        is_ranked=True,
        game_length_secs=0,
        rounds_played=20,
        red_score=0,
        blue_score=0,
        started_at="2026-04-19T18:00:00+00:00",
    )
    mp = MatchPlayer(
        match_id=match_id,
        puuid=PUUID,
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
    mp.match = match
    return mp


def _fake_session_scope(session: object):
    """Build an async context manager that yields *session*."""

    @asynccontextmanager
    async def _scope():
        yield session

    return _scope


# ---------------------------------------------------------------------------
# PlayerData dataclass
# ---------------------------------------------------------------------------


class TestPlayerData:
    def test_is_frozen(self) -> None:
        p = _player()
        data = PlayerData(player=p, rows=[], full_matches=[])
        from dataclasses import FrozenInstanceError

        with pytest.raises(FrozenInstanceError):
            data.rows = []  # type: ignore[misc]

    def test_fields_accessible(self) -> None:
        p = _player()
        rows = [_mp()]
        fm: list[Match] = []
        data = PlayerData(player=p, rows=rows, full_matches=fm)
        assert data.player is p
        assert data.rows is rows
        assert data.full_matches is fm


# ---------------------------------------------------------------------------
# Guard: settings without identity → None (no DB access)
# ---------------------------------------------------------------------------


class TestEarlyReturn:
    """Returns None before touching the DB when settings lack identity."""

    def test_no_riot_name_sync(self) -> None:
        result = load_player_data(_settings(riot_name=""))
        assert result is None

    def test_no_riot_tag_sync(self) -> None:
        result = load_player_data(_settings(riot_tag=""))
        assert result is None

    async def test_no_riot_name_async(self) -> None:
        result = await load_player_data_async(_settings(riot_name=""))
        assert result is None

    async def test_no_riot_tag_async(self) -> None:
        result = await load_player_data_async(_settings(riot_tag=""))
        assert result is None


# ---------------------------------------------------------------------------
# Player not found in DB → None
# ---------------------------------------------------------------------------


class TestPlayerNotFound:
    async def test_returns_none_when_no_player_row(self) -> None:
        session = MagicMock()

        with (
            patch("valocoach.data.loader.ensure_db", AsyncMock()),
            patch("valocoach.data.loader.session_scope", _fake_session_scope(session)),
            patch("valocoach.data.loader.get_player_by_name", AsyncMock(return_value=None)),
        ):
            result = await load_player_data_async(_settings())
        assert result is None


# ---------------------------------------------------------------------------
# Happy path: player found, rows returned
# ---------------------------------------------------------------------------


class TestHappyPath:
    async def test_returns_player_data(self) -> None:
        player = _player()
        rows = [_mp("m-1"), _mp("m-2")]
        session = MagicMock()

        with (
            patch("valocoach.data.loader.ensure_db", AsyncMock()),
            patch("valocoach.data.loader.session_scope", _fake_session_scope(session)),
            patch("valocoach.data.loader.get_player_by_name", AsyncMock(return_value=player)),
            patch("valocoach.data.loader.get_recent_matches", AsyncMock(return_value=rows)),
            patch("valocoach.data.loader.get_recent_matches_full", AsyncMock(return_value=[])),
        ):
            result = await load_player_data_async(_settings())

        assert isinstance(result, PlayerData)
        assert result.player is player
        assert result.rows is rows
        assert result.full_matches == []

    async def test_empty_rows_still_returns_player_data(self) -> None:
        """Player exists in DB but has never had a match synced."""
        player = _player()
        session = MagicMock()

        with (
            patch("valocoach.data.loader.ensure_db", AsyncMock()),
            patch("valocoach.data.loader.session_scope", _fake_session_scope(session)),
            patch("valocoach.data.loader.get_player_by_name", AsyncMock(return_value=player)),
            patch("valocoach.data.loader.get_recent_matches", AsyncMock(return_value=[])),
            patch("valocoach.data.loader.get_recent_matches_full", AsyncMock(return_value=[])),
        ):
            result = await load_player_data_async(_settings())

        assert result is not None
        assert result.player is player
        assert result.rows == []


# ---------------------------------------------------------------------------
# include_rounds flag
# ---------------------------------------------------------------------------


class TestIncludeRounds:
    async def test_false_skips_full_match_fetch(self) -> None:
        """get_recent_matches_full must not be called when include_rounds=False."""
        player = _player()
        rows = [_mp()]
        session = MagicMock()
        mock_full = AsyncMock(return_value=[])

        with (
            patch("valocoach.data.loader.ensure_db", AsyncMock()),
            patch("valocoach.data.loader.session_scope", _fake_session_scope(session)),
            patch("valocoach.data.loader.get_player_by_name", AsyncMock(return_value=player)),
            patch("valocoach.data.loader.get_recent_matches", AsyncMock(return_value=rows)),
            patch("valocoach.data.loader.get_recent_matches_full", mock_full),
        ):
            result = await load_player_data_async(_settings(), include_rounds=False)

        mock_full.assert_not_called()
        assert result is not None
        assert result.full_matches == []

    async def test_true_fetches_full_matches(self) -> None:
        player = _player()
        rows = [_mp()]
        full_match = Match(
            match_id="m-full",
            map_name="Ascent",
            queue_id="competitive",
            is_ranked=True,
            game_length_secs=0,
            rounds_played=20,
            red_score=0,
            blue_score=0,
            started_at="2026-04-19T18:00:00+00:00",
        )
        session = MagicMock()

        with (
            patch("valocoach.data.loader.ensure_db", AsyncMock()),
            patch("valocoach.data.loader.session_scope", _fake_session_scope(session)),
            patch("valocoach.data.loader.get_player_by_name", AsyncMock(return_value=player)),
            patch("valocoach.data.loader.get_recent_matches", AsyncMock(return_value=rows)),
            patch(
                "valocoach.data.loader.get_recent_matches_full",
                AsyncMock(return_value=[full_match]),
            ),
        ):
            result = await load_player_data_async(_settings(), include_rounds=True)

        assert result is not None
        assert result.full_matches == [full_match]

    async def test_empty_rows_skips_full_fetch_even_with_include_rounds(self) -> None:
        """No point querying full matches if there are no aggregate rows."""
        player = _player()
        session = MagicMock()
        mock_full = AsyncMock(return_value=[])

        with (
            patch("valocoach.data.loader.ensure_db", AsyncMock()),
            patch("valocoach.data.loader.session_scope", _fake_session_scope(session)),
            patch("valocoach.data.loader.get_player_by_name", AsyncMock(return_value=player)),
            patch("valocoach.data.loader.get_recent_matches", AsyncMock(return_value=[])),
            patch("valocoach.data.loader.get_recent_matches_full", mock_full),
        ):
            result = await load_player_data_async(_settings(), include_rounds=True)

        mock_full.assert_not_called()
        assert result is not None
        assert result.full_matches == []


# ---------------------------------------------------------------------------
# Parameter forwarding
# ---------------------------------------------------------------------------


class TestParameterForwarding:
    async def test_limit_forwarded_to_get_recent_matches(self) -> None:
        player = _player()
        session = MagicMock()
        mock_rows = AsyncMock(return_value=[])

        with (
            patch("valocoach.data.loader.ensure_db", AsyncMock()),
            patch("valocoach.data.loader.session_scope", _fake_session_scope(session)),
            patch("valocoach.data.loader.get_player_by_name", AsyncMock(return_value=player)),
            patch("valocoach.data.loader.get_recent_matches", mock_rows),
            patch("valocoach.data.loader.get_recent_matches_full", AsyncMock(return_value=[])),
        ):
            await load_player_data_async(_settings(), limit=42)

        mock_rows.assert_called_once()
        _, kwargs = mock_rows.call_args
        assert kwargs["limit"] == 42

    async def test_queue_id_forwarded(self) -> None:
        player = _player()
        session = MagicMock()
        mock_rows = AsyncMock(return_value=[])

        with (
            patch("valocoach.data.loader.ensure_db", AsyncMock()),
            patch("valocoach.data.loader.session_scope", _fake_session_scope(session)),
            patch("valocoach.data.loader.get_player_by_name", AsyncMock(return_value=player)),
            patch("valocoach.data.loader.get_recent_matches", mock_rows),
            patch("valocoach.data.loader.get_recent_matches_full", AsyncMock(return_value=[])),
        ):
            await load_player_data_async(_settings(), queue_id=None)

        _, kwargs = mock_rows.call_args
        assert kwargs["queue_id"] is None

    async def test_default_limit_is_200(self) -> None:
        player = _player()
        session = MagicMock()
        mock_rows = AsyncMock(return_value=[])

        with (
            patch("valocoach.data.loader.ensure_db", AsyncMock()),
            patch("valocoach.data.loader.session_scope", _fake_session_scope(session)),
            patch("valocoach.data.loader.get_player_by_name", AsyncMock(return_value=player)),
            patch("valocoach.data.loader.get_recent_matches", mock_rows),
            patch("valocoach.data.loader.get_recent_matches_full", AsyncMock(return_value=[])),
        ):
            await load_player_data_async(_settings())

        _, kwargs = mock_rows.call_args
        assert kwargs["limit"] == DEFAULT_LOAD_LIMIT == 200


# ---------------------------------------------------------------------------
# Sync entry point
# ---------------------------------------------------------------------------


class TestSyncEntryPoint:
    def test_returns_none_for_unknown_player(self) -> None:
        session = MagicMock()

        with (
            patch("valocoach.data.loader.ensure_db", AsyncMock()),
            patch("valocoach.data.loader.session_scope", _fake_session_scope(session)),
            patch("valocoach.data.loader.get_player_by_name", AsyncMock(return_value=None)),
        ):
            result = load_player_data(_settings())
        assert result is None

    def test_returns_player_data_for_known_player(self) -> None:
        player = _player()
        rows = [_mp("m-1")]
        session = MagicMock()

        with (
            patch("valocoach.data.loader.ensure_db", AsyncMock()),
            patch("valocoach.data.loader.session_scope", _fake_session_scope(session)),
            patch("valocoach.data.loader.get_player_by_name", AsyncMock(return_value=player)),
            patch("valocoach.data.loader.get_recent_matches", AsyncMock(return_value=rows)),
            patch("valocoach.data.loader.get_recent_matches_full", AsyncMock(return_value=[])),
        ):
            result = load_player_data(_settings())

        assert isinstance(result, PlayerData)
        assert result.player.puuid == PUUID
        assert len(result.rows) == 1
        assert result.full_matches == []

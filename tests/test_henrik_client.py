"""Tests for HenrikClient — all HTTP calls are mocked with pytest-httpx."""

from __future__ import annotations

import pytest

from valocoach.data.henrik_client import HenrikAPIError, HenrikClient

BASE = "https://api.henrikdev.xyz"
API_KEY = "HDEV-test-key"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _account_payload(puuid: str = "test-puuid") -> dict:
    return {
        "status": 200,
        "data": {
            "puuid": puuid,
            "region": "na",
            "account_level": 240,
            "name": "Yoursaviour01",
            "tag": "SK04",
            "platforms": ["PC"],
        },
    }


def _mmr_payload() -> dict:
    return {
        "status": 200,
        "data": {
            "name": "Yoursaviour01",
            "tag": "SK04",
            "puuid": "test-puuid",
            "current_data": {
                "currenttier": 12,
                "currenttierpatched": "Gold 1",
                "ranking_in_tier": 0,
                "mmr_change_to_last_game": -13,
                "elo": 900,
                "games_needed_for_rating": 0,
                "old": False,
            },
            "highest_rank": {
                "old": False,
                "tier": 14,
                "patched_tier": "Gold 3",
                "season": "e6a2",
            },
            "by_season": {},
        },
    }


def _match_payload(match_id: str = "match-001") -> dict:
    return {
        "status": 200,
        "data": [
            {
                "is_available": True,
                "metadata": {
                    "matchid": match_id,
                    "map": "Lotus",
                    "mode": "Competitive",
                    "mode_id": "competitive",
                    "queue": "Standard",
                    "rounds_played": 17,
                    "game_length": 1462,
                    "game_start": 1775285695,
                    "region": "na",
                },
                "players": {
                    "all_players": [
                        {
                            "puuid": "test-puuid",
                            "name": "Yoursaviour01",
                            "tag": "SK04",
                            "team": "Blue",
                            "character": "Jett",
                            "level": 240,
                            "currenttier": 12,
                            "currenttier_patched": "Gold 1",
                            "stats": {
                                "score": 4352,
                                "kills": 14,
                                "deaths": 14,
                                "assists": 1,
                                "bodyshots": 40,
                                "headshots": 16,
                                "legshots": 1,
                            },
                            "behavior": {
                                "afkRounds": 1,
                                "stayedInSpawnRounds": 2,
                            },
                        }
                    ]
                },
                "teams": {
                    "red": {"has_won": True, "rounds_won": 9, "rounds_lost": 8},
                    "blue": {"has_won": False, "rounds_won": 8, "rounds_lost": 9},
                },
            }
        ],
    }


# ---------------------------------------------------------------------------
# get_account
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_account_returns_account_data(httpx_mock):
    httpx_mock.add_response(
        url=f"{BASE}/valorant/v2/account/Yoursaviour01/SK04",
        json=_account_payload("my-puuid"),
    )
    async with HenrikClient(API_KEY) as client:
        account = await client.get_account("Yoursaviour01", "SK04")

    assert account.name == "Yoursaviour01"
    assert account.tag == "SK04"
    assert account.puuid == "my-puuid"
    assert account.region == "na"


@pytest.mark.asyncio
async def test_get_account_raises_on_http_error(httpx_mock):
    httpx_mock.add_response(
        url=f"{BASE}/valorant/v2/account/Bad/User",
        status_code=404,
        text="Not found",
    )
    with pytest.raises(HenrikAPIError) as exc_info:
        async with HenrikClient(API_KEY) as client:
            await client.get_account("Bad", "User")

    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_get_account_raises_on_api_level_error(httpx_mock):
    """API returns 200 HTTP but status!=200 in the JSON body."""
    httpx_mock.add_response(
        url=f"{BASE}/valorant/v2/account/Bad/User",
        json={"status": 403, "errors": [{"message": "Forbidden"}]},
    )
    with pytest.raises(HenrikAPIError) as exc_info:
        async with HenrikClient(API_KEY) as client:
            await client.get_account("Bad", "User")

    assert exc_info.value.status_code == 403


# ---------------------------------------------------------------------------
# get_mmr
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_mmr_returns_mmr_data(httpx_mock):
    httpx_mock.add_response(
        url=f"{BASE}/valorant/v2/mmr/na/Yoursaviour01/SK04",
        json=_mmr_payload(),
    )
    async with HenrikClient(API_KEY) as client:
        mmr = await client.get_mmr("na", "Yoursaviour01", "SK04")

    assert mmr.current_data.currenttierpatched == "Gold 1"
    assert mmr.current_data.elo == 900
    assert mmr.highest_rank.patched_tier == "Gold 3"


# ---------------------------------------------------------------------------
# get_matches
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_matches_returns_list(httpx_mock):
    httpx_mock.add_response(
        url=f"{BASE}/valorant/v3/matches/na/Yoursaviour01/SK04?size=3&mode=competitive",
        json=_match_payload("match-abc"),
    )
    async with HenrikClient(API_KEY) as client:
        matches = await client.get_matches("na", "Yoursaviour01", "SK04", size=3)

    assert len(matches) == 1
    assert matches[0].metadata.map_name == "Lotus"
    assert matches[0].metadata.match_id == "match-abc"


@pytest.mark.asyncio
async def test_get_matches_player_stats(httpx_mock):
    httpx_mock.add_response(
        url=f"{BASE}/valorant/v3/matches/na/Yoursaviour01/SK04?size=3&mode=competitive",
        json=_match_payload(),
    )
    async with HenrikClient(API_KEY) as client:
        matches = await client.get_matches("na", "Yoursaviour01", "SK04", size=3)

    player = matches[0].player_by_puuid("test-puuid")
    assert player is not None
    assert player.character == "Jett"
    assert player.stats.kills == 14
    assert player.stats.kd_ratio == 1.0
    assert player.behavior.afk_rounds == 1
    assert player.behavior.rounds_in_spawn == 2


@pytest.mark.asyncio
async def test_get_matches_raises_on_http_error(httpx_mock):
    """Non-retryable 401 should surface immediately as HenrikAPIError."""
    httpx_mock.add_response(
        url=f"{BASE}/valorant/v3/matches/na/Yoursaviour01/SK04?size=5&mode=competitive",
        status_code=401,
        text="Unauthorized",
    )
    with pytest.raises(HenrikAPIError) as exc_info:
        async with HenrikClient(API_KEY) as client:
            await client.get_matches("na", "Yoursaviour01", "SK04", size=5)

    assert exc_info.value.status_code == 401


@pytest.mark.asyncio
async def test_get_matches_retries_on_429(httpx_mock):
    """429 is retryable — register 3 responses to cover all attempts, then verify error."""
    for _ in range(3):
        httpx_mock.add_response(
            url=f"{BASE}/valorant/v3/matches/na/Yoursaviour01/SK04?size=5&mode=competitive",
            status_code=429,
            text="Rate limited",
        )
    with pytest.raises(HenrikAPIError) as exc_info:
        async with HenrikClient(API_KEY) as client:
            await client.get_matches("na", "Yoursaviour01", "SK04", size=5)

    assert exc_info.value.status_code == 429


# ---------------------------------------------------------------------------
# get_account — force refresh
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_account_force_refresh(httpx_mock):
    """force=True should append ?force=true to the request URL."""
    httpx_mock.add_response(
        url=f"{BASE}/valorant/v2/account/Yoursaviour01/SK04?force=true",
        json=_account_payload("force-puuid"),
    )
    async with HenrikClient(API_KEY) as client:
        account = await client.get_account("Yoursaviour01", "SK04", force=True)

    assert account.puuid == "force-puuid"


# ---------------------------------------------------------------------------
# get_mmr_history
# ---------------------------------------------------------------------------


def _mmr_history_payload() -> dict:
    return {
        "status": 200,
        "data": [
            {
                "currenttier": 12,
                "currenttierpatched": "Gold 1",
                "ranking_in_tier": 45,
                "mmr_change_to_last_game": -13,
                "elo": 945,
                "date": "2026-04-15T20:00:00.000Z",
                "date_raw": 1744747200,
                "match_id": "hist-match-001",
            },
            {
                "currenttier": 12,
                "currenttierpatched": "Gold 1",
                "ranking_in_tier": 58,
                "mmr_change_to_last_game": 20,
                "elo": 958,
                "date": "2026-04-14T19:00:00.000Z",
                "date_raw": 1744657200,
                "match_id": "hist-match-002",
            },
        ],
    }


@pytest.mark.asyncio
async def test_get_mmr_history_returns_list(httpx_mock):
    httpx_mock.add_response(
        url=f"{BASE}/valorant/v2/mmr-history/na/Yoursaviour01/SK04",
        json=_mmr_history_payload(),
    )
    async with HenrikClient(API_KEY) as client:
        history = await client.get_mmr_history("na", "Yoursaviour01", "SK04")

    assert len(history) == 2
    assert history[0].currenttierpatched == "Gold 1"
    assert history[0].ranking_in_tier == 45
    assert history[0].mmr_change_to_last_game == -13
    assert history[0].match_id == "hist-match-001"
    assert history[1].elo == 958


# ---------------------------------------------------------------------------
# fetch_player_snapshot
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fetch_player_snapshot_concurrent(httpx_mock):
    """fetch_player_snapshot should return (AccountData, MMRData, list[MatchData])
    gathered from three concurrent requests."""
    httpx_mock.add_response(
        url=f"{BASE}/valorant/v2/account/Yoursaviour01/SK04",
        json=_account_payload("snap-puuid"),
    )
    httpx_mock.add_response(
        url=f"{BASE}/valorant/v2/mmr/na/Yoursaviour01/SK04",
        json=_mmr_payload(),
    )
    httpx_mock.add_response(
        url=f"{BASE}/valorant/v3/matches/na/Yoursaviour01/SK04?size=10&mode=competitive",
        json=_match_payload("snap-match-001"),
    )

    async with HenrikClient(API_KEY) as client:
        account, mmr, matches = await client.fetch_player_snapshot("na", "Yoursaviour01", "SK04")

    assert account.puuid == "snap-puuid"
    assert mmr.current_data.currenttierpatched == "Gold 1"
    assert len(matches) == 1
    assert matches[0].metadata.match_id == "snap-match-001"

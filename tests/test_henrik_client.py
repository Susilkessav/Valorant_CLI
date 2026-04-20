"""Tests for api_client.HenrikClient — all HTTP calls are mocked with pytest-httpx.

Client API:
    HenrikClient(settings, *, throttle_seconds=0)  ← throttle_seconds=0 in all tests
    Raises APIError / RateLimitError / ServerError (from valocoach.core.exceptions).

Endpoints covered:
    get_account          GET /valorant/v2/account/{name}/{tag}
    get_mmr              GET /valorant/v2/mmr/{region}/{name}/{tag}
    get_mmr_history      GET /valorant/v2/mmr-history/{region}/{name}/{tag}
    get_matches          GET /valorant/v3/matches/{region}/{name}/{tag}
    get_stored_matches   GET /valorant/v1/stored-matches/{region}/{name}/{tag}
    get_match_details    GET /valorant/v4/match/{region}/{match_id}
    fetch_player_snapshot  concurrent gather of account + MMR + matches
"""

from __future__ import annotations

import pytest

from valocoach.core.config import Settings
from valocoach.core.exceptions import APIError, RateLimitError
from valocoach.data.api_client import HenrikClient

BASE = "https://api.henrikdev.xyz"
API_KEY = "HDEV-test-key"

# Use throttle_seconds=0 so sequential calls in a single test don't sleep.
_SETTINGS = Settings(
    riot_name="",
    riot_tag="",
    riot_region="na",
    henrikdev_api_key=API_KEY,
    ollama_model="qwen3:8b",
    ollama_host="http://localhost:11434",
)


# ---------------------------------------------------------------------------
# Response payload helpers
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


def _matches_payload(match_id: str = "match-001") -> dict:
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


def _stored_matches_payload(match_id: str = "stored-001") -> dict:
    return {
        "status": 200,
        "data": [
            {
                "meta": {"id": match_id},
                "stats": {
                    "puuid": "test-puuid",
                    "team": "Blue",
                    "level": 240,
                    "character": {"id": "jett-id", "name": "Jett"},
                    "score": 4352,
                    "kills": 14,
                    "deaths": 14,
                    "assists": 1,
                    "shots": {"head": 16, "body": 40, "leg": 1},
                    "damage": {"made": 2400, "received": 1800},
                },
                "teams": {"red": 9, "blue": 8},
            }
        ],
    }


def _match_details_payload(match_id: str = "v4-match-001") -> dict:
    """Minimal but valid v4 match payload (no players/rounds/kills — all default to [])."""
    return {
        "status": 200,
        "data": {
            "metadata": {
                "match_id": match_id,
                "map": {"id": "map-lotus", "name": "Lotus"},
                "game_version": "release-09.00",
                "game_length_in_ms": 1_462_000,
                "started_at": "2026-04-19T18:00:00+00:00",
                "is_completed": True,
                "queue": {
                    "id": "competitive",
                    "name": "Competitive",
                    "mode_type": "Standard",
                },
                "season": {"id": "e9a1", "name": "EPISODE 9 ACT 1"},
                "region": "na",
            },
            "players": [],
            "teams": [],
            "rounds": [],
            "kills": [],
        },
    }


# ---------------------------------------------------------------------------
# get_account
# ---------------------------------------------------------------------------


async def test_get_account_returns_account_data(httpx_mock):
    httpx_mock.add_response(
        url=f"{BASE}/valorant/v2/account/Yoursaviour01/SK04",
        json=_account_payload("my-puuid"),
    )
    async with HenrikClient(_SETTINGS, throttle_seconds=0) as client:
        account = await client.get_account("Yoursaviour01", "SK04")

    assert account.name == "Yoursaviour01"
    assert account.tag == "SK04"
    assert account.puuid == "my-puuid"
    assert account.region == "na"


async def test_get_account_raises_on_http_error(httpx_mock):
    httpx_mock.add_response(
        url=f"{BASE}/valorant/v2/account/Bad/User",
        status_code=404,
        text="Not found",
    )
    with pytest.raises(APIError) as exc_info:
        async with HenrikClient(_SETTINGS, throttle_seconds=0) as client:
            await client.get_account("Bad", "User")

    assert "404" in str(exc_info.value)


async def test_get_account_raises_on_body_level_error(httpx_mock):
    """API returns HTTP 200 but status!=200 inside the JSON body."""
    httpx_mock.add_response(
        url=f"{BASE}/valorant/v2/account/Bad/User",
        json={"status": 403, "errors": [{"message": "Forbidden"}]},
    )
    with pytest.raises(APIError) as exc_info:
        async with HenrikClient(_SETTINGS, throttle_seconds=0) as client:
            await client.get_account("Bad", "User")

    assert "403" in str(exc_info.value)


async def test_get_account_force_refresh(httpx_mock):
    """force=True appends ?force=true to the URL."""
    httpx_mock.add_response(
        url=f"{BASE}/valorant/v2/account/Yoursaviour01/SK04?force=true",
        json=_account_payload("force-puuid"),
    )
    async with HenrikClient(_SETTINGS, throttle_seconds=0) as client:
        account = await client.get_account("Yoursaviour01", "SK04", force=True)

    assert account.puuid == "force-puuid"


# ---------------------------------------------------------------------------
# get_mmr
# ---------------------------------------------------------------------------


async def test_get_mmr_returns_mmr_data(httpx_mock):
    httpx_mock.add_response(
        url=f"{BASE}/valorant/v2/mmr/na/Yoursaviour01/SK04",
        json=_mmr_payload(),
    )
    async with HenrikClient(_SETTINGS, throttle_seconds=0) as client:
        mmr = await client.get_mmr("na", "Yoursaviour01", "SK04")

    assert mmr.current_data.currenttierpatched == "Gold 1"
    assert mmr.current_data.elo == 900
    assert mmr.highest_rank.patched_tier == "Gold 3"


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


async def test_get_mmr_history_returns_list(httpx_mock):
    httpx_mock.add_response(
        url=f"{BASE}/valorant/v2/mmr-history/na/Yoursaviour01/SK04",
        json=_mmr_history_payload(),
    )
    async with HenrikClient(_SETTINGS, throttle_seconds=0) as client:
        history = await client.get_mmr_history("na", "Yoursaviour01", "SK04")

    assert len(history) == 2
    assert history[0].currenttierpatched == "Gold 1"
    assert history[0].ranking_in_tier == 45
    assert history[0].mmr_change_to_last_game == -13
    assert history[0].match_id == "hist-match-001"
    assert history[1].elo == 958


# ---------------------------------------------------------------------------
# get_matches (v3)
# ---------------------------------------------------------------------------


async def test_get_matches_returns_list(httpx_mock):
    httpx_mock.add_response(
        url=f"{BASE}/valorant/v3/matches/na/Yoursaviour01/SK04?size=3&mode=competitive",
        json=_matches_payload("match-abc"),
    )
    async with HenrikClient(_SETTINGS, throttle_seconds=0) as client:
        matches = await client.get_matches("na", "Yoursaviour01", "SK04", size=3)

    assert len(matches) == 1
    assert matches[0].metadata.map_name == "Lotus"
    assert matches[0].metadata.match_id == "match-abc"


async def test_get_matches_player_stats(httpx_mock):
    httpx_mock.add_response(
        url=f"{BASE}/valorant/v3/matches/na/Yoursaviour01/SK04?size=3&mode=competitive",
        json=_matches_payload(),
    )
    async with HenrikClient(_SETTINGS, throttle_seconds=0) as client:
        matches = await client.get_matches("na", "Yoursaviour01", "SK04", size=3)

    player = matches[0].player_by_puuid("test-puuid")
    assert player is not None
    assert player.character == "Jett"
    assert player.stats.kills == 14
    assert player.stats.kd_ratio == 1.0
    assert player.behavior.afk_rounds == 1
    assert player.behavior.rounds_in_spawn == 2


async def test_get_matches_raises_on_401(httpx_mock):
    """Non-retryable 401 raises APIError immediately (no retry)."""
    httpx_mock.add_response(
        url=f"{BASE}/valorant/v3/matches/na/Yoursaviour01/SK04?size=5&mode=competitive",
        status_code=401,
        text="Unauthorized",
    )
    with pytest.raises(APIError) as exc_info:
        async with HenrikClient(_SETTINGS, throttle_seconds=0) as client:
            await client.get_matches("na", "Yoursaviour01", "SK04", size=5)

    assert "401" in str(exc_info.value)


async def test_get_matches_raises_rate_limit_on_429(httpx_mock):
    """429 raises RateLimitError after all retry attempts are exhausted."""
    for _ in range(3):
        httpx_mock.add_response(
            url=f"{BASE}/valorant/v3/matches/na/Yoursaviour01/SK04?size=5&mode=competitive",
            status_code=429,
            text="Rate limited",
        )
    with pytest.raises(RateLimitError):
        async with HenrikClient(_SETTINGS, throttle_seconds=0) as client:
            await client.get_matches("na", "Yoursaviour01", "SK04", size=5)


# ---------------------------------------------------------------------------
# get_stored_matches (v1) — used by the sync pipeline
# ---------------------------------------------------------------------------


async def test_get_stored_matches_returns_list(httpx_mock):
    httpx_mock.add_response(
        url=f"{BASE}/valorant/v1/stored-matches/na/Yoursaviour01/SK04?size=20&mode=competitive",
        json=_stored_matches_payload("stored-abc"),
    )
    async with HenrikClient(_SETTINGS, throttle_seconds=0) as client:
        matches = await client.get_stored_matches("na", "Yoursaviour01", "SK04")

    assert len(matches) == 1
    assert matches[0].match_id == "stored-abc"


async def test_get_stored_matches_empty_data_returns_empty_list(httpx_mock):
    """API returning data:[] (no matches) yields an empty list, not an error."""
    httpx_mock.add_response(
        url=f"{BASE}/valorant/v1/stored-matches/na/Yoursaviour01/SK04?size=20&mode=competitive",
        json={"status": 200, "data": []},
    )
    async with HenrikClient(_SETTINGS, throttle_seconds=0) as client:
        matches = await client.get_stored_matches("na", "Yoursaviour01", "SK04")

    assert matches == []


async def test_get_stored_matches_custom_size_and_mode(httpx_mock):
    """Custom size and mode are forwarded as query params."""
    httpx_mock.add_response(
        url=f"{BASE}/valorant/v1/stored-matches/na/Yoursaviour01/SK04?size=5&mode=unrated",
        json=_stored_matches_payload("unrated-match"),
    )
    async with HenrikClient(_SETTINGS, throttle_seconds=0) as client:
        matches = await client.get_stored_matches(
            "na", "Yoursaviour01", "SK04", size=5, mode="unrated"
        )

    assert matches[0].match_id == "unrated-match"


# ---------------------------------------------------------------------------
# get_match_details (v4) — used by the sync pipeline
# ---------------------------------------------------------------------------


async def test_get_match_details_returns_match_details(httpx_mock):
    httpx_mock.add_response(
        url=f"{BASE}/valorant/v4/match/na/v4-match-001",
        json=_match_details_payload("v4-match-001"),
    )
    async with HenrikClient(_SETTINGS, throttle_seconds=0) as client:
        details = await client.get_match_details("na", "v4-match-001")

    from valocoach.data.api_models import MatchDetails

    assert isinstance(details, MatchDetails)
    assert details.metadata.match_id == "v4-match-001"
    assert details.metadata.map.name == "Lotus"
    assert details.metadata.region == "na"


async def test_get_match_details_raises_on_404(httpx_mock):
    httpx_mock.add_response(
        url=f"{BASE}/valorant/v4/match/na/bad-id",
        status_code=404,
        text="Not found",
    )
    with pytest.raises(APIError) as exc_info:
        async with HenrikClient(_SETTINGS, throttle_seconds=0) as client:
            await client.get_match_details("na", "bad-id")

    assert "404" in str(exc_info.value)


# ---------------------------------------------------------------------------
# fetch_player_snapshot — concurrent gather
# ---------------------------------------------------------------------------


async def test_fetch_player_snapshot_concurrent(httpx_mock):
    """Returns (AccountData, MMRData, list[MatchData]) from three concurrent requests."""
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
        json=_matches_payload("snap-match-001"),
    )

    async with HenrikClient(_SETTINGS, throttle_seconds=0) as client:
        account, mmr, matches = await client.fetch_player_snapshot("na", "Yoursaviour01", "SK04")

    assert account.puuid == "snap-puuid"
    assert mmr.current_data.currenttierpatched == "Gold 1"
    assert len(matches) == 1
    assert matches[0].metadata.match_id == "snap-match-001"


# ---------------------------------------------------------------------------
# ConfigError on missing API key
# ---------------------------------------------------------------------------


async def test_raises_config_error_when_api_key_missing():
    """HenrikClient raises ConfigError immediately if the API key is not set."""
    from valocoach.core.exceptions import ConfigError

    bad_settings = Settings(
        riot_name="",
        riot_tag="",
        riot_region="na",
        henrikdev_api_key="",
        ollama_model="qwen3:8b",
        ollama_host="http://localhost:11434",
    )
    with pytest.raises(ConfigError):
        HenrikClient(bad_settings)

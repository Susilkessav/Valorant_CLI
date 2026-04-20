"""Unit tests for Pydantic API models (no I/O)."""

from __future__ import annotations

from valocoach.data.models import (
    AccountData,
    HenrikResponse,
    MatchPlayer,
    PlayerBehavior,
    PlayerStats,
)

# ---------------------------------------------------------------------------
# HenrikResponse envelope
# ---------------------------------------------------------------------------


def test_henrik_response_wraps_account(account_data):
    payload = {"status": 200, "data": account_data.model_dump()}
    response = HenrikResponse[AccountData].model_validate(payload)
    assert response.status == 200
    assert response.data.puuid == account_data.puuid


def test_henrik_response_non_200_still_parses():
    """The envelope model doesn't enforce status — repository does."""
    payload = {"status": 429, "data": {}}
    response = HenrikResponse[dict].model_validate(payload)
    assert response.status == 429


# ---------------------------------------------------------------------------
# AccountData
# ---------------------------------------------------------------------------


def test_account_data_fields(account_data):
    assert account_data.name == "Yoursaviour01"
    assert account_data.tag == "SK04"
    assert account_data.region == "na"
    assert account_data.account_level == 240
    assert account_data.puuid == "20905543-1b42-5f6f-8435-ab284a0094f8"


def test_account_data_optional_fields_default_none():
    acc = AccountData(
        puuid="abc",
        region="na",
        account_level=1,
        name="Test",
        tag="T1",
    )
    assert acc.card is None
    assert acc.title is None
    assert acc.platforms == []


# ---------------------------------------------------------------------------
# MMRData
# ---------------------------------------------------------------------------


def test_mmr_current_rank(mmr_data):
    assert mmr_data.current_data.currenttierpatched == "Gold 1"
    assert mmr_data.current_data.elo == 900
    assert mmr_data.current_data.ranking_in_tier == 0


def test_mmr_highest_rank(mmr_data):
    assert mmr_data.highest_rank.patched_tier == "Gold 3"
    assert mmr_data.highest_rank.tier == 14


def test_mmr_defaults_to_unranked():
    from valocoach.data.models import CurrentRankData

    rank = CurrentRankData()
    assert rank.currenttierpatched == "Unranked"
    assert rank.elo == 0


# ---------------------------------------------------------------------------
# PlayerStats computed properties
# ---------------------------------------------------------------------------


def test_player_stats_kd_ratio():
    stats = PlayerStats(kills=14, deaths=14)
    assert stats.kd_ratio == 1.0


def test_player_stats_kd_ratio_zero_deaths():
    """Deaths=0 should not raise ZeroDivisionError."""
    stats = PlayerStats(kills=10, deaths=0)
    assert stats.kd_ratio == 10.0


def test_player_stats_headshot_pct():
    stats = PlayerStats(headshots=16, bodyshots=40, legshots=1)
    total = 16 + 40 + 1
    expected = round(16 / total * 100, 1)
    assert stats.headshot_pct == expected


def test_player_stats_headshot_pct_no_shots():
    stats = PlayerStats()
    assert stats.headshot_pct == 0.0


def test_player_behavior_accepts_henrik_camel_case():
    behavior = PlayerBehavior.model_validate(
        {
            "afkRounds": 2,
            "stayedInSpawnRounds": 3,
        }
    )
    assert behavior.afk_rounds == 2
    assert behavior.rounds_in_spawn == 3


def test_match_player_nested_behavior_accepts_henrik_camel_case():
    player = MatchPlayer.model_validate(
        {
            "puuid": "test-puuid",
            "name": "Test",
            "tag": "T1",
            "team": "Blue",
            "character": "Jett",
            "behavior": {
                "afkRounds": 1,
                "stayedInSpawnRounds": 4,
            },
        }
    )
    assert player.behavior.afk_rounds == 1
    assert player.behavior.rounds_in_spawn == 4


# ---------------------------------------------------------------------------
# MatchData helpers
# ---------------------------------------------------------------------------


def test_match_metadata_fields(match_data):
    meta = match_data.metadata
    assert meta.map_name == "Lotus"
    assert meta.queue_id == "competitive"
    assert meta.mode == "Competitive"
    assert meta.rounds_played == 17
    assert meta.game_length_secs == 1462


def test_player_by_puuid_found(match_data, account_data):
    player = match_data.player_by_puuid(account_data.puuid)
    assert player is not None
    assert player.name == "Yoursaviour01"
    assert player.character == "Jett"


def test_player_by_puuid_not_found(match_data):
    player = match_data.player_by_puuid("does-not-exist")
    assert player is None


def test_match_teams(match_data):
    assert match_data.teams.red.has_won is True
    assert match_data.teams.blue.has_won is False
    assert match_data.teams.red.rounds_won == 9

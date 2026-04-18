from __future__ import annotations

from pathlib import Path

import pytest

from valocoach.core.config import Settings
from valocoach.data.database import Base, init_engine
from valocoach.data.models import (
    AccountData,
    CurrentRankData,
    HighestRank,
    MatchData,
    MatchMetadata,
    MatchPlayer,
    MatchPlayers,
    MatchTeams,
    MMRData,
    PlayerBehavior,
    PlayerStats,
    TeamResult,
)

# ---------------------------------------------------------------------------
# App settings
# ---------------------------------------------------------------------------


@pytest.fixture
def settings() -> Settings:
    return Settings(
        riot_name="TestUser",
        riot_tag="TEST",
        riot_region="na",
        henrikdev_api_key="fake-key",
        ollama_model="qwen3:8b",
        ollama_host="http://localhost:11434",
    )


# ---------------------------------------------------------------------------
# Pydantic model fixtures  (reused across model + repository tests)
# ---------------------------------------------------------------------------

PUUID = "20905543-1b42-5f6f-8435-ab284a0094f8"
MATCH_ID = "b0c012f7-9a68-46d1-a527-32783a190a5c"


@pytest.fixture
def account_data() -> AccountData:
    return AccountData(
        puuid=PUUID,
        region="na",
        account_level=240,
        name="Yoursaviour01",
        tag="SK04",
    )


@pytest.fixture
def mmr_data() -> MMRData:
    return MMRData(
        name="Yoursaviour01",
        tag="SK04",
        puuid=PUUID,
        current_data=CurrentRankData(
            currenttier=12,
            currenttierpatched="Gold 1",
            ranking_in_tier=0,
            elo=900,
            mmr_change_to_last_game=-13,
        ),
        highest_rank=HighestRank(
            tier=14,
            patched_tier="Gold 3",
            season="e6a2",
        ),
    )


@pytest.fixture
def match_data() -> MatchData:
    """One competitive match on Lotus with two players."""
    return MatchData(
        metadata=MatchMetadata(
            matchid=MATCH_ID,
            map="Lotus",
            mode="Competitive",
            mode_id="competitive",
            queue="Standard",
            rounds_played=17,
            game_length=1462,
            game_start=1775285695,
            region="na",
        ),
        players=MatchPlayers(
            all_players=[
                MatchPlayer(
                    puuid=PUUID,
                    name="Yoursaviour01",
                    tag="SK04",
                    team="Blue",
                    character="Jett",
                    currenttier=12,
                    currenttier_patched="Gold 1",
                    stats=PlayerStats(
                        score=4352,
                        kills=14,
                        deaths=14,
                        assists=1,
                        headshots=16,
                        bodyshots=40,
                        legshots=1,
                    ),
                    behavior=PlayerBehavior(afk_rounds=1, rounds_in_spawn=2),
                ),
                MatchPlayer(
                    puuid="enemy-puuid-0001",
                    name="dipp",
                    tag="100T",
                    team="Red",
                    character="Neon",
                    currenttier=13,
                    currenttier_patched="Gold 2",
                    stats=PlayerStats(
                        score=3811,
                        kills=12,
                        deaths=10,
                        assists=6,
                        headshots=6,
                        bodyshots=49,
                        legshots=2,
                    ),
                ),
            ]
        ),
        teams=MatchTeams(
            red=TeamResult(has_won=True, rounds_won=9, rounds_lost=8),
            blue=TeamResult(has_won=False, rounds_won=8, rounds_lost=9),
        ),
    )


# ---------------------------------------------------------------------------
# Database fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def db_engine(tmp_path: Path):
    """Fresh SQLite engine + schema for each test."""
    engine = init_engine(tmp_path / "test.db")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


@pytest.fixture
async def db_session(db_engine):
    """Open session that rolls back after each test (keeps tests isolated)."""
    from valocoach.data.database import _SessionLocal

    async with _SessionLocal() as session:
        yield session
        await session.rollback()

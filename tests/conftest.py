from __future__ import annotations

from pathlib import Path

import pytest

from valocoach.core.config import Settings
from valocoach.data.api_models import (
    MatchDetails,
    MatchDetailsKill,
    MatchDetailsMetadata,
    MatchDetailsPlayer,
    MatchDetailsPlayerBehavior,
    MatchDetailsPlayerStats,
    MatchDetailsRound,
    MatchDetailsTeam,
    _PlayerRef,
    _Ref,
    _V4BombEvent,
    _V4PlayerDamage,
    _V4Queue,
    _V4TeamRounds,
)
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
            match_id=MATCH_ID,
            map_name="Lotus",
            mode="Competitive",
            queue_id="competitive",
            queue="Standard",
            rounds_played=17,
            game_length_secs=1462,
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
# v4 MatchDetails fixture
# ---------------------------------------------------------------------------

ENEMY_PUUID = "enemy-puuid-0001"


@pytest.fixture
def match_details() -> MatchDetails:
    """Minimal but complete v4 MatchDetails with 2 players, 2 rounds, 3 kills."""
    return MatchDetails(
        metadata=MatchDetailsMetadata(
            match_id=MATCH_ID,
            map=_Ref(id="map-lotus-id", name="Lotus"),
            game_version="release-09.00",
            game_length_in_ms=1_462_000,
            started_at="2026-04-19T18:00:00+00:00",
            is_completed=True,
            queue=_V4Queue(id="competitive", name="Competitive", mode_type="Standard"),
            season=_Ref(id="e9a1", name="EPISODE 9 ACT 1"),
            region="na",
        ),
        players=[
            MatchDetailsPlayer(
                puuid=PUUID,
                name="Yoursaviour01",
                tag="SK04",
                team_id="Blue",
                agent=_Ref(id="jett-id", name="Jett"),
                stats=MatchDetailsPlayerStats(
                    score=3811,
                    kills=14,
                    deaths=12,
                    assists=2,
                    headshots=16,
                    bodyshots=40,
                    legshots=1,
                    damage=_V4PlayerDamage(dealt=2400, received=1800),
                ),
                tier=_Ref(id="12", name="Gold 1"),
                behavior=MatchDetailsPlayerBehavior(afk_rounds=0.0, rounds_in_spawn=1.0),
            ),
            MatchDetailsPlayer(
                puuid=ENEMY_PUUID,
                name="dipp",
                tag="100T",
                team_id="Red",
                agent=_Ref(id="neon-id", name="Neon"),
                stats=MatchDetailsPlayerStats(
                    score=4200,
                    kills=15,
                    deaths=10,
                    assists=5,
                    headshots=8,
                    bodyshots=55,
                    legshots=3,
                    damage=_V4PlayerDamage(dealt=3100, received=2100),
                ),
                tier=_Ref(id="13", name="Gold 2"),
                behavior=MatchDetailsPlayerBehavior(afk_rounds=0.0, rounds_in_spawn=0.0),
            ),
        ],
        teams=[
            MatchDetailsTeam(
                team_id="Red",
                won=True,
                rounds=_V4TeamRounds(won=9, lost=8),
            ),
            MatchDetailsTeam(
                team_id="Blue",
                won=False,
                rounds=_V4TeamRounds(won=8, lost=9),
            ),
        ],
        rounds=[
            MatchDetailsRound(
                id=0,
                winning_team="Red",
                ceremony="Default",
                plant=_V4BombEvent(
                    round_time_in_ms=45_000,
                    site="A",
                    player=_PlayerRef(puuid=ENEMY_PUUID, name="dipp", tag="100T", team="Red"),
                ),
            ),
            MatchDetailsRound(
                id=1,
                winning_team="Blue",
                ceremony="Default",
                defuse=_V4BombEvent(
                    round_time_in_ms=50_000,
                    site="B",
                    player=_PlayerRef(puuid=PUUID, name="Yoursaviour01", tag="SK04", team="Blue"),
                ),
            ),
        ],
        kills=[
            # Round 0 — first blood: dipp kills Yoursaviour01
            MatchDetailsKill(
                round=0,
                time_in_round_in_ms=10_000,
                time_in_match_in_ms=10_000,
                killer=_PlayerRef(puuid=ENEMY_PUUID, name="dipp", tag="100T", team="Red"),
                victim=_PlayerRef(puuid=PUUID, name="Yoursaviour01", tag="SK04", team="Blue"),
                weapon=_Ref(id="vandal-id", name="Vandal"),
            ),
            # Round 0 — second kill: dipp kills another player
            MatchDetailsKill(
                round=0,
                time_in_round_in_ms=25_000,
                time_in_match_in_ms=25_000,
                killer=_PlayerRef(puuid=ENEMY_PUUID, name="dipp", tag="100T", team="Red"),
                victim=_PlayerRef(puuid="other-puuid", name="other", tag="EX", team="Blue"),
                weapon=_Ref(id="vandal-id", name="Vandal"),
            ),
            # Round 1 — first blood: Yoursaviour01 kills dipp
            MatchDetailsKill(
                round=1,
                time_in_round_in_ms=8_000,
                time_in_match_in_ms=108_000,
                killer=_PlayerRef(puuid=PUUID, name="Yoursaviour01", tag="SK04", team="Blue"),
                victim=_PlayerRef(puuid=ENEMY_PUUID, name="dipp", tag="100T", team="Red"),
                weapon=_Ref(id="phantom-id", name="Phantom"),
                assistants=[
                    _PlayerRef(puuid="ally-puuid", name="ally", tag="AA", team="Blue")
                ],
            ),
        ],
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

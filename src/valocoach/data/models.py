"""Pydantic models for HenrikDev Valorant API responses.

Shapes are derived from the v2/account, v2/mmr, and v3/matches endpoints.
All fields are optional-friendly — the API occasionally omits keys on
unranked / incomplete records.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Generic, TypeVar

from pydantic import AliasChoices, BaseModel, ConfigDict, Field

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Generic envelope
# ---------------------------------------------------------------------------


class HenrikResponse(BaseModel, Generic[T]):
    """Every Henrik endpoint wraps its payload in {status, data}."""

    status: int
    data: T


# ---------------------------------------------------------------------------
# Account  (/v2/account/{name}/{tag})
# ---------------------------------------------------------------------------


class AccountData(BaseModel):
    puuid: str
    region: str
    account_level: int
    name: str
    tag: str
    card: str | None = None
    title: str | None = None
    platforms: list[str] = Field(default_factory=list)
    updated_at: datetime | None = None


# ---------------------------------------------------------------------------
# MMR / Rank  (/v2/mmr/{region}/{name}/{tag})
# ---------------------------------------------------------------------------


class RankImages(BaseModel):
    small: str | None = None
    large: str | None = None
    triangle_down: str | None = None
    triangle_up: str | None = None


class CurrentRankData(BaseModel):
    currenttier: int = 0
    currenttierpatched: str = "Unranked"
    images: RankImages | None = None
    ranking_in_tier: int = 0  # 0-99 RR
    mmr_change_to_last_game: int = 0  # ± RR delta
    elo: int = 0  # absolute MMR
    games_needed_for_rating: int = 0
    old: bool = False


class HighestRank(BaseModel):
    tier: int = 0
    patched_tier: str = "Unranked"
    season: str | None = None
    old: bool = False


class MMRData(BaseModel):
    name: str
    tag: str
    puuid: str | None = None
    current_data: CurrentRankData = Field(default_factory=CurrentRankData)
    highest_rank: HighestRank = Field(default_factory=HighestRank)
    by_season: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Match  (/v3/matches/{region}/{name}/{tag})
# ---------------------------------------------------------------------------


class MatchMetadata(BaseModel):
    matchid: str
    map: str
    mode: str
    mode_id: str
    queue: str | None = None
    rounds_played: int = 0
    game_length: int = 0  # seconds
    game_start: int = 0  # unix timestamp
    game_start_patched: str | None = None
    game_version: str | None = None
    season_id: str | None = None
    region: str | None = None
    cluster: str | None = None


class PlayerBehavior(BaseModel):
    """AFK and spawn-camping signals returned by the API."""

    model_config = ConfigDict(populate_by_name=True)

    afk_rounds: int = Field(
        default=0,
        validation_alias=AliasChoices("afk_rounds", "afkRounds"),
    )
    rounds_in_spawn: int = Field(
        default=0,
        validation_alias=AliasChoices(
            "rounds_in_spawn",
            "roundsInSpawn",
            "stayed_in_spawn_rounds",
            "stayedInSpawnRounds",
        ),
    )


class PlayerStats(BaseModel):
    score: int = 0
    kills: int = 0
    deaths: int = 0
    assists: int = 0
    bodyshots: int = 0
    headshots: int = 0
    legshots: int = 0

    @property
    def kd_ratio(self) -> float:
        return round(self.kills / max(self.deaths, 1), 2)

    @property
    def headshot_pct(self) -> float:
        total = self.headshots + self.bodyshots + self.legshots
        return round(self.headshots / max(total, 1) * 100, 1)


class MatchPlayer(BaseModel):
    puuid: str
    name: str
    tag: str
    team: str  # "Red" | "Blue"
    character: str  # agent name
    level: int = 0
    currenttier: int = 0
    currenttier_patched: str = "Unranked"
    stats: PlayerStats = Field(default_factory=PlayerStats)
    behavior: PlayerBehavior = Field(default_factory=PlayerBehavior)
    party_id: str | None = None


class TeamResult(BaseModel):
    has_won: bool = False
    rounds_won: int = 0
    rounds_lost: int = 0


class MatchPlayers(BaseModel):
    all_players: list[MatchPlayer] = Field(default_factory=list)
    red: list[MatchPlayer] = Field(default_factory=list)
    blue: list[MatchPlayer] = Field(default_factory=list)


class MatchTeams(BaseModel):
    red: TeamResult = Field(default_factory=TeamResult)
    blue: TeamResult = Field(default_factory=TeamResult)


class MatchData(BaseModel):
    is_available: bool = True
    metadata: MatchMetadata
    players: MatchPlayers = Field(default_factory=MatchPlayers)
    teams: MatchTeams = Field(default_factory=MatchTeams)

    def player_by_puuid(self, puuid: str) -> MatchPlayer | None:
        """Return the MatchPlayer entry for a given PUUID."""
        return next(
            (p for p in self.players.all_players if p.puuid == puuid),
            None,
        )

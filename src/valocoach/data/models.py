"""Pydantic models for HenrikDev Valorant API responses.

All models inherit _Base which applies two settings to every subclass:
  - extra="ignore"        — unknown API fields are silently dropped instead of
                            raising ValidationError. Protects against patch-day
                            API additions.
  - populate_by_name=True — models can be constructed with Python field names
                            even when a validation_alias is set. Keeps test
                            fixtures readable while still accepting API JSON.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Generic, TypeVar

from pydantic import AliasChoices, BaseModel, ConfigDict, Field

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Shared base
# ---------------------------------------------------------------------------


class _Base(BaseModel):
    """Shared config for all API response models."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True)


# ---------------------------------------------------------------------------
# Generic envelope
# ---------------------------------------------------------------------------


class HenrikResponse(_Base, Generic[T]):
    """Every Henrik endpoint wraps its payload in {status, data}."""

    status: int
    data: T


# ---------------------------------------------------------------------------
# Account  (/v2/account/{name}/{tag})
# ---------------------------------------------------------------------------


class AccountData(_Base):
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


class RankImages(_Base):
    small: str | None = None
    large: str | None = None
    triangle_down: str | None = None
    triangle_up: str | None = None


class CurrentRankData(_Base):
    currenttier: int = 0
    currenttierpatched: str = "Unranked"
    images: RankImages | None = None
    ranking_in_tier: int = 0  # 0-99 RR
    mmr_change_to_last_game: int = 0  # +/- RR delta
    elo: int = 0  # absolute MMR
    games_needed_for_rating: int = 0
    old: bool = False


class HighestRank(_Base):
    tier: int = 0
    patched_tier: str = "Unranked"
    season: str | None = None
    old: bool = False


class MMRData(_Base):
    name: str
    tag: str
    puuid: str | None = None
    current_data: CurrentRankData = Field(default_factory=CurrentRankData)
    highest_rank: HighestRank = Field(default_factory=HighestRank)
    by_season: dict[str, Any] = Field(default_factory=dict)


class MMRHistoryEntry(_Base):
    """One entry in a player's rank history (/v2/mmr-history).

    Each entry corresponds to one ranked game and shows the rank snapshot
    at the end of that game — useful for coaching trend analysis.
    """

    currenttier: int = 0
    currenttierpatched: str = "Unranked"
    ranking_in_tier: int = 0  # RR within tier after this game
    mmr_change_to_last_game: int = 0  # +/- RR delta
    elo: int = 0  # absolute MMR after this game
    date: str | None = None  # ISO8601 string from API
    date_raw: int | None = None  # unix timestamp
    match_id: str | None = None


# ---------------------------------------------------------------------------
# Match  (/v3/matches/{region}/{name}/{tag})
# ---------------------------------------------------------------------------


class MatchMetadata(_Base):
    # Field renames: Python name on the left, API key as validation_alias.
    # populate_by_name=True (from _Base) means both forms work for construction.
    match_id: str = Field(validation_alias="matchid")
    map_name: str = Field(validation_alias="map")  # "map" in API is the name, not an ID
    mode: str | None = None  # display name, e.g. "Competitive"
    queue_id: str = Field(validation_alias="mode_id")  # slug, e.g. "competitive"
    queue: str | None = None  # e.g. "Standard", "Swiftplay"
    rounds_played: int = 0
    game_length_secs: int = Field(default=0, validation_alias="game_length")  # API sends seconds
    game_start: int = 0  # unix timestamp — converted to ISO8601 in repository
    game_start_patched: str | None = None
    game_version: str | None = None
    season_id: str | None = None
    region: str | None = None
    cluster: str | None = None


class PlayerBehavior(_Base):
    """AFK and spawn-camping signals returned by the API.

    AliasChoices accepts both snake_case (Python construction) and
    camelCase (live Henrik API JSON) for each field.
    populate_by_name=True is inherited from _Base, so no override needed.
    """

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


class PlayerStats(_Base):
    score: int = 0
    kills: int = 0
    deaths: int = 0
    assists: int = 0
    bodyshots: int = 0
    headshots: int = 0
    legshots: int = 0
    damage_dealt: int = 0  # added — already a column in the ORM
    damage_received: int = 0  # added — already a column in the ORM

    @property
    def kd_ratio(self) -> float:
        return round(self.kills / max(self.deaths, 1), 2)

    @property
    def headshot_pct(self) -> float:
        total = self.headshots + self.bodyshots + self.legshots
        return round(self.headshots / max(total, 1) * 100, 1)


class MatchPlayer(_Base):
    puuid: str
    name: str
    tag: str
    team: str  # "Red" | "Blue"
    character: str  # agent name, e.g. "Jett"
    level: int = 0
    currenttier: int = 0
    currenttier_patched: str = "Unranked"
    stats: PlayerStats = Field(default_factory=PlayerStats)
    behavior: PlayerBehavior = Field(default_factory=PlayerBehavior)
    party_id: str | None = None


class TeamResult(_Base):
    has_won: bool = False
    rounds_won: int = 0
    rounds_lost: int = 0


class MatchPlayers(_Base):
    all_players: list[MatchPlayer] = Field(default_factory=list)
    red: list[MatchPlayer] = Field(default_factory=list)
    blue: list[MatchPlayer] = Field(default_factory=list)


class MatchTeams(_Base):
    red: TeamResult = Field(default_factory=TeamResult)
    blue: TeamResult = Field(default_factory=TeamResult)


# ---------------------------------------------------------------------------
# Kill events (returned inline in the /v3/matches response)
# ---------------------------------------------------------------------------


class KillAssistant(_Base):
    """One assistant credit within a kill event."""

    puuid: str
    team: str | None = None
    display_name: str | None = None


class KillEvent(_Base):
    """One kill — flat fields mirror the API shape.

    killer/victim are kept as flat fields (killer_puuid, killer_team) rather
    than nested objects because the Henrik v3 API returns them that way.
    """

    round: int
    time_in_round_ms: int | None = Field(default=None, alias="time_in_round_in_ms")
    killer_puuid: str
    killer_team: str | None = None
    victim_puuid: str
    victim_team: str | None = None
    weapon_name: str | None = None
    is_headshot: bool = False
    secondary_fire_mode: bool = False
    assistants: list[KillAssistant] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Round data (returned inline in the /v3/matches response)
# ---------------------------------------------------------------------------


class BombEvent(_Base):
    """Plant or defuse event within a round."""

    site: str | None = None
    time_in_round_ms: int | None = None


class RoundPlayerStats(_Base):
    """Per-player breakdown for a single round."""

    puuid: str
    kills: int = 0
    damage: int = 0
    score: int = 0
    headshots: int = 0
    bodyshots: int = 0
    legshots: int = 0
    was_afk: bool = False
    stayed_in_spawn: bool = False


class RoundData(_Base):
    """One round — outcome, bomb events, and per-player breakdown."""

    winning_team: str
    end_type: str  # "Elimination" | "Defuse" | "Detonate" | "Surrendered"
    bomb_planted: bool = False
    bomb_defused: bool = False
    plant: BombEvent | None = None
    defuse: BombEvent | None = None
    player_stats: list[RoundPlayerStats] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Full match shape
# ---------------------------------------------------------------------------


class MatchData(_Base):
    """Full match response from /v3/matches.

    rounds and kills default to empty lists — the endpoint may return them
    or omit them for unavailable / still-processing matches.
    """

    is_available: bool = True
    metadata: MatchMetadata
    players: MatchPlayers = Field(default_factory=MatchPlayers)
    teams: MatchTeams = Field(default_factory=MatchTeams)
    rounds: list[RoundData] = Field(default_factory=list)
    kills: list[KillEvent] = Field(default_factory=list)

    def player_by_puuid(self, puuid: str) -> MatchPlayer | None:
        """Return the MatchPlayer entry for a given PUUID."""
        return next(
            (p for p in self.players.all_players if p.puuid == puuid),
            None,
        )

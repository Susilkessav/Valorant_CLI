"""Pydantic models for all HenrikDev Valorant API responses.

All models inherit _Base which applies two settings to every subclass:
  - extra="ignore"        — unknown API fields are silently dropped instead of
                            raising ValidationError. Protects against patch-day
                            API additions.
  - populate_by_name=True — models can be constructed with Python field names
                            even when a validation_alias is set. Keeps test
                            fixtures readable while still accepting API JSON.

API versions covered:
  v2/account           AccountData, HenrikResponse
  v2/mmr               MMRData, CurrentRankData, HighestRank, MMRHistoryEntry
  v3/matches           MatchData (legacy v3 shape)
  v1/stored-matches    StoredMatch
  v4/match             MatchDetails (current preferred shape)

Key v4 vs v3 differences (do NOT alias MatchDetails to MatchData):
  metadata.match_id      plain key (v3 used matchid)
  metadata.started_at    ISO8601 string (v3 sent game_start unix int)
  metadata.game_length_in_ms  ms (v3 sent game_length in seconds)
  metadata has no mode/mode_id — use queue.id instead
  players   flat list   (v3: {all_players, red, blue} dict)
  teams     list        (v3: {red, blue} dict)
  teams[].rounds        {won, lost} object (v3: rounds_won/rounds_lost)
  kill.killer/victim    nested {puuid,name,tag,team} (v3: flat killer_puuid)
  kill assistants       flat player list (v3 had wrapper object)
  kill has no is_headshot field
  round player list key  "stats"   (v3: "player_stats")
  player.stats.damage   {dealt, received} nested (v3: damage_dealt flat)
  player.tier           {id, name} object (v3: int)
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Generic, TypeVar

from pydantic import AliasChoices, BaseModel, BeforeValidator, ConfigDict, Field

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


AccountResponse = AccountData  # v2/account shape unchanged in newer API versions

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
# Match v3  (/v3/matches/{region}/{name}/{tag})
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


# ---------------------------------------------------------------------------
# Shared helpers for v4 models
# ---------------------------------------------------------------------------


def _coerce_str(v: object) -> str:
    """Coerce int/float API values to str; map None → empty string.

    Used on _Ref.id because the tier object sends {"id": 13, "name": "Gold 2"}
    where id is a JSON integer, not a string.
    """
    return "" if v is None else str(v)


class _Ref(_Base):
    """Generic {id, name} reference (map, agent, weapon, season, tier …).

    id  — always stored as str even when the API sends an int (e.g. tier.id=13).
    name — nullable: ability-kill weapon objects arrive as {"name": null, …}.
           Callers use `ref.name or ""` / `ref.name or "Unknown"` for display.
    """

    id: Annotated[str, BeforeValidator(_coerce_str)] = ""
    name: str | None = None


class _PlayerRef(_Base):
    """Minimal player reference used inside kill / round objects."""

    puuid: str = ""
    name: str | None = None
    tag: str = ""
    team: str = ""


# ---------------------------------------------------------------------------
# v1/stored-matches  — confirmed 2026-04-19
# ---------------------------------------------------------------------------


class _StoredSeason(_Base):
    id: str = ""
    short: str = ""


class StoredMatchMeta(_Base):
    match_id: str = Field(default="", validation_alias=AliasChoices("id", "match_id"))
    map: _Ref = Field(default_factory=_Ref)  # {"id": "...", "name": "Lotus"}
    mode: str | None = None  # "Competitive"
    version: str | None = None
    started_at: str | None = None  # ISO8601
    season: _StoredSeason = Field(default_factory=_StoredSeason)
    region: str | None = None
    cluster: str | None = None

    @property
    def map_name(self) -> str:
        return self.map.name or "Unknown"


class _StoredShots(_Base):
    head: int = 0
    body: int = 0
    leg: int = 0


class _StoredDamage(_Base):
    made: int = 0
    received: int = 0


class StoredMatchStats(_Base):
    puuid: str = ""
    name: str = ""
    tag: str = ""
    team: str | None = None
    level: int = 0
    character: _Ref = Field(default_factory=_Ref)  # {"id": "...", "name": "Waylay"}
    tier: int = 0
    score: int = 0
    kills: int = 0
    deaths: int = 0
    assists: int = 0
    shots: _StoredShots = Field(default_factory=_StoredShots)
    damage: _StoredDamage = Field(default_factory=_StoredDamage)

    @property
    def agent_name(self) -> str:
        return self.character.name or ""

    @property
    def headshots(self) -> int:
        return self.shots.head

    @property
    def bodyshots(self) -> int:
        return self.shots.body

    @property
    def legshots(self) -> int:
        return self.shots.leg

    @property
    def damage_made(self) -> int:
        return self.damage.made

    @property
    def damage_received(self) -> int:
        return self.damage.received


class _StoredTeams(_Base):
    red: int = 0
    blue: int = 0


class StoredMatch(_Base):
    """One entry from GET /valorant/v1/stored-matches/{region}/{name}/{tag}."""

    meta: StoredMatchMeta = Field(default_factory=StoredMatchMeta)
    stats: StoredMatchStats = Field(default_factory=StoredMatchStats)
    teams: _StoredTeams = Field(default_factory=_StoredTeams)

    @property
    def match_id(self) -> str:
        return self.meta.match_id


# ---------------------------------------------------------------------------
# v4/match — full match detail
# ---------------------------------------------------------------------------


class _V4Queue(_Base):
    id: str = ""  # "competitive"
    name: str = ""  # "Competitive"
    mode_type: str = ""  # "Standard"


class MatchDetailsMetadata(_Base):
    """Metadata from /v4/match. Fields confirmed from sample_match.json."""

    match_id: str = ""  # key is literally "match_id" in v4
    map: _Ref = Field(default_factory=_Ref)
    game_version: str | None = None
    game_length_in_ms: int = 0  # milliseconds (NOT seconds like v3 game_length)
    started_at: str | None = None  # ISO8601 (NOT a unix int like v3 game_start)
    is_completed: bool = False
    queue: _V4Queue = Field(default_factory=_V4Queue)
    season: _Ref = Field(default_factory=_Ref)
    platform: str | None = None
    region: str | None = None
    cluster: str | None = None
    # rounds_played is NOT in v4 metadata — compute as len(match.rounds)

    @property
    def map_name(self) -> str:
        return self.map.name or "Unknown"

    @property
    def queue_id(self) -> str:
        return self.queue.id

    @property
    def game_length_secs(self) -> float:
        return self.game_length_in_ms / 1000


class _V4PlayerDamage(_Base):
    """Damage breakdown nested inside player stats."""

    dealt: int = 0
    received: int = 0


class MatchDetailsPlayerStats(_Base):
    score: int = 0
    kills: int = 0
    deaths: int = 0
    assists: int = 0
    headshots: int = 0
    bodyshots: int = 0
    legshots: int = 0
    damage: _V4PlayerDamage = Field(default_factory=_V4PlayerDamage)

    @property
    def damage_dealt(self) -> int:
        return self.damage.dealt

    @property
    def damage_received(self) -> int:
        return self.damage.received


class MatchDetailsPlayerAbility(_Base):
    """ability_casts block on a player (match-level, uses camelCase keys in v4)."""

    grenade: int = 0
    ability1: int = 0
    ability2: int = 0
    ultimate: int = 0


class MatchDetailsPlayerBehavior(_Base):
    afk_rounds: float = 0.0
    rounds_in_spawn: float = 0.0
    friendly_fire: dict[str, Any] = Field(default_factory=dict)


class MatchDetailsPlayer(_Base):
    """One player from the flat players list in a v4 match."""

    puuid: str
    name: str = ""
    tag: str = ""
    team_id: str = ""  # "Red" | "Blue"
    platform: str | None = None
    party_id: str | None = None
    agent: _Ref = Field(default_factory=_Ref)
    stats: MatchDetailsPlayerStats = Field(default_factory=MatchDetailsPlayerStats)
    ability_casts: MatchDetailsPlayerAbility = Field(default_factory=MatchDetailsPlayerAbility)
    tier: _Ref = Field(default_factory=_Ref)  # {"id": 13, "name": "Gold 2"}
    customization: dict[str, Any] = Field(default_factory=dict)
    account_level: int = 0
    session_playtime_in_ms: int = 0
    behavior: MatchDetailsPlayerBehavior = Field(default_factory=MatchDetailsPlayerBehavior)
    economy: dict[str, Any] = Field(default_factory=dict)

    @property
    def agent_name(self) -> str:
        return self.agent.name or ""

    @property
    def team(self) -> str:
        return self.team_id

    @property
    def current_tier(self) -> int:
        return self.tier.id if self.tier.id else 0


class _V4TeamRounds(_Base):
    won: int = 0
    lost: int = 0


class MatchDetailsTeam(_Base):
    team_id: str = ""
    won: bool = False
    rounds: _V4TeamRounds = Field(default_factory=_V4TeamRounds)
    premier_roster: Any = None

    @property
    def rounds_won(self) -> int:
        return self.rounds.won

    @property
    def rounds_lost(self) -> int:
        return self.rounds.lost


class MatchDetailsKill(_Base):
    """Kill event from the kills list in a v4 match.

    killer/victim are nested {puuid,name,tag,team} objects in v4.
    assistants is a flat list of the same player-reference shape.
    There is no is_headshot field at the kill level in v4.
    """

    round: int = 0
    time_in_round_in_ms: int = 0
    time_in_match_in_ms: int = 0
    killer: _PlayerRef = Field(default_factory=_PlayerRef)
    victim: _PlayerRef = Field(default_factory=_PlayerRef)
    assistants: list[_PlayerRef] = Field(default_factory=list)
    weapon: _Ref = Field(default_factory=_Ref)
    secondary_fire_mode: bool = False
    location: dict[str, Any] = Field(default_factory=dict)
    player_locations: list[dict[str, Any]] = Field(default_factory=list)

    @property
    def killer_puuid(self) -> str:
        return self.killer.puuid

    @property
    def victim_puuid(self) -> str:
        return self.victim.puuid

    @property
    def killer_team(self) -> str:
        return self.killer.team

    @property
    def victim_team(self) -> str:
        return self.victim.team

    @property
    def weapon_name(self) -> str | None:
        return self.weapon.name


class _RoundPlayerInnerStats(_Base):
    score: int = 0
    kills: int = 0
    headshots: int = 0
    bodyshots: int = 0
    legshots: int = 0


class _RoundAbilityCasts(_Base):
    """Round-level ability_casts — uses ability_1/ability_2 (underscores)."""

    grenade: int | None = None
    ability_1: int | None = None
    ability_2: int | None = None
    ultimate: int | None = None


class MatchDetailsRoundPlayerStats(_Base):
    """Per-player breakdown for one round. Key is 'stats' in v4 (not 'player_stats')."""

    player: _PlayerRef = Field(default_factory=_PlayerRef)
    ability_casts: _RoundAbilityCasts = Field(default_factory=_RoundAbilityCasts)
    damage_events: list[dict[str, Any]] = Field(default_factory=list)
    stats: _RoundPlayerInnerStats = Field(default_factory=_RoundPlayerInnerStats)
    economy: dict[str, Any] = Field(default_factory=dict)
    was_afk: bool = False
    received_penalty: bool = False
    stayed_in_spawn: bool = False

    @property
    def puuid(self) -> str:
        return self.player.puuid

    @property
    def score(self) -> int:
        return self.stats.score

    @property
    def kills(self) -> int:
        return self.stats.kills


class _V4BombEvent(_Base):
    round_time_in_ms: int | None = None
    site: str | None = None
    location: dict[str, Any] = Field(default_factory=dict)
    player: _PlayerRef = Field(default_factory=_PlayerRef)
    player_locations: list[dict[str, Any]] = Field(default_factory=list)


class MatchDetailsRound(_Base):
    """One round from a v4 match. Player list key is 'stats' (not 'player_stats')."""

    id: int = 0
    result: str = ""  # team-relative: "Win" | "Loss"
    ceremony: str | None = None
    winning_team: str = ""
    plant: _V4BombEvent | None = None
    defuse: _V4BombEvent | None = None
    stats: list[MatchDetailsRoundPlayerStats] = Field(default_factory=list)

    @property
    def player_stats(self) -> list[MatchDetailsRoundPlayerStats]:
        """Alias for .stats — keeps calling code readable."""
        return self.stats


class MatchDetails(_Base):
    """Full match from GET /valorant/v4/match/{region}/{match_id}.

    Confirmed field names from sample_match.json (match b0c012f7, 2026-04-19).
    """

    metadata: MatchDetailsMetadata = Field(default_factory=MatchDetailsMetadata)
    players: list[MatchDetailsPlayer] = Field(default_factory=list)
    observers: list[dict[str, Any]] = Field(default_factory=list)
    coaches: list[dict[str, Any]] = Field(default_factory=list)
    teams: list[MatchDetailsTeam] = Field(default_factory=list)
    rounds: list[MatchDetailsRound] = Field(default_factory=list)
    kills: list[MatchDetailsKill] = Field(default_factory=list)

    def player_by_puuid(self, puuid: str) -> MatchDetailsPlayer | None:
        return next((p for p in self.players if p.puuid == puuid), None)

    def team_result(self, team_id: str) -> MatchDetailsTeam | None:
        return next((t for t in self.teams if t.team_id == team_id), None)

    @property
    def rounds_played(self) -> int:
        return len(self.rounds)

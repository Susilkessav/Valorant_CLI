"""Pydantic models for newer Henrik API endpoints.

AccountResponse — alias of AccountData (v2/account shape unchanged).
MatchDetails    — v4/match full shape, confirmed from sample_match.json
                  (2026-04-19, match b0c012f7).
StoredMatch     — v1/stored-matches shape, confirmed same session.

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

from typing import Annotated, Any

from pydantic import AliasChoices, BeforeValidator, Field

from valocoach.data.models import AccountData, _Base

# ---------------------------------------------------------------------------
# v2/account  — unchanged
# ---------------------------------------------------------------------------

AccountResponse = AccountData


# ---------------------------------------------------------------------------
# Shared helpers
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

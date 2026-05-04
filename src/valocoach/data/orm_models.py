"""SQLAlchemy ORM table definitions.

Six tables:
  players       — identity + current/peak rank snapshot
  matches       — match metadata + team scores
  match_players — one row per player per match (stats junction table)
  rounds        — per-round outcome + bomb events
  kills         — per-kill events within a round
  sync_log      — operational audit trail for each data sync

Design decisions:
  - Timestamps stored as UTC ISO8601 TEXT (e.g. "2026-04-18T18:00:00+00:00").
    SQLite has no native TIMESTAMP type; storing as TEXT keeps the value
    human-readable, sortable, and timezone-unambiguous. Use _now_iso() /
    _unix_to_iso() helpers everywhere — never store naive datetimes.
  - match_players.puuid has NO FK to players — all 10 participants per match
    are stored for context, but only the tracked player has a players row.
  - Computed properties (kd_ratio, headshot_pct, acs) live on OrmMatchPlayer.
  - Indexes on (puuid, started_at) and (puuid, agent_name, started_at) for
    the most common coaching queries.
"""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from valocoach.data.database import Base


def _now_iso() -> str:
    """Current UTC time as an ISO8601 string — used as column default."""
    return datetime.now(UTC).isoformat()


# ---------------------------------------------------------------------------
# players
# ---------------------------------------------------------------------------


class Player(Base):
    """One row per tracked Riot account."""

    __tablename__ = "players"

    puuid: Mapped[str] = mapped_column(String, primary_key=True)
    riot_name: Mapped[str] = mapped_column(String, nullable=False)
    riot_tag: Mapped[str] = mapped_column(String, nullable=False)
    region: Mapped[str] = mapped_column(String, nullable=False)
    platform: Mapped[str] = mapped_column(String, default="pc")
    account_level: Mapped[int] = mapped_column(Integer, default=0)

    # Current rank snapshot (updated on each MMR fetch)
    current_tier: Mapped[int] = mapped_column(Integer, default=0)
    current_tier_patched: Mapped[str] = mapped_column(String, default="Unranked")
    current_rr: Mapped[int] = mapped_column(Integer, default=0)  # 0-99 RR
    elo: Mapped[int] = mapped_column(Integer, default=0)  # absolute MMR

    # Peak rank
    peak_tier: Mapped[int] = mapped_column(Integer, default=0)
    peak_tier_patched: Mapped[str] = mapped_column(String, default="Unranked")

    # ISO8601 UTC strings
    last_match_at: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[str] = mapped_column(String, default=_now_iso)
    updated_at: Mapped[str] = mapped_column(String, default=_now_iso, onupdate=_now_iso)

    def __repr__(self) -> str:
        return (
            f"<Player {self.riot_name}#{self.riot_tag} {self.current_tier_patched} ELO={self.elo}>"
        )


# ---------------------------------------------------------------------------
# matches
# ---------------------------------------------------------------------------


class Match(Base):
    """One row per unique match."""

    __tablename__ = "matches"

    match_id: Mapped[str] = mapped_column(String, primary_key=True)
    map_name: Mapped[str] = mapped_column(String, nullable=False)
    map_id: Mapped[str | None] = mapped_column(String, nullable=True)
    queue_id: Mapped[str] = mapped_column(String, nullable=False)  # "competitive" | "unrated" …
    is_ranked: Mapped[bool] = mapped_column(Boolean, default=False)
    game_version: Mapped[str | None] = mapped_column(String, nullable=True)
    game_length_secs: Mapped[int] = mapped_column(Integer, default=0)
    season_short: Mapped[str | None] = mapped_column(String, nullable=True)
    region: Mapped[str | None] = mapped_column(String, nullable=True)
    rounds_played: Mapped[int] = mapped_column(Integer, default=0)

    # Team scores
    red_score: Mapped[int] = mapped_column(Integer, default=0)
    blue_score: Mapped[int] = mapped_column(Integer, default=0)
    winning_team: Mapped[str | None] = mapped_column(String, nullable=True)  # "Red" | "Blue"

    # ISO8601 UTC strings — lexicographic order == chronological order
    started_at: Mapped[str] = mapped_column(String, nullable=False)
    synced_at: Mapped[str] = mapped_column(String, default=_now_iso)

    # Relationships
    players: Mapped[list[OrmMatchPlayer]] = relationship(
        back_populates="match", cascade="all, delete-orphan", lazy="selectin"
    )
    rounds: Mapped[list[Round]] = relationship(
        back_populates="match", cascade="all, delete-orphan", lazy="selectin"
    )

    def __repr__(self) -> str:
        return f"<Match {self.match_id[:8]}… {self.map_name} {self.queue_id}>"


# ---------------------------------------------------------------------------
# match_players
# ---------------------------------------------------------------------------


class OrmMatchPlayer(Base):
    """One row per player per match — core stats table.

    Named OrmMatchPlayer to avoid collision with the Pydantic MatchPlayer
    in valocoach.data.models. Re-exported as MatchPlayer from this module.
    """

    __tablename__ = "match_players"
    __table_args__ = (
        UniqueConstraint("match_id", "puuid", name="uq_match_player"),
        Index("idx_mp_puuid_started", "puuid", "started_at"),
        Index("idx_mp_puuid_agent", "puuid", "agent_name", "started_at"),
        # team must be exactly "Red" or "Blue" — silent typos in the mapper
        # would otherwise corrupt every side-split calculation.
        CheckConstraint("team IN ('Red', 'Blue')", name="ck_match_player_team"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    match_id: Mapped[str] = mapped_column(
        ForeignKey("matches.match_id", ondelete="CASCADE"), nullable=False
    )
    # Plain string — no FK to players; all 10 participants are stored
    puuid: Mapped[str | None] = mapped_column(String, nullable=True)

    agent_name: Mapped[str] = mapped_column(String, nullable=False)
    agent_id: Mapped[str | None] = mapped_column(String, nullable=True)
    team: Mapped[str] = mapped_column(String, nullable=False)  # "Red" | "Blue"
    won: Mapped[bool] = mapped_column(Boolean, default=False)

    # Combat stats
    score: Mapped[int] = mapped_column(Integer, default=0)
    kills: Mapped[int] = mapped_column(Integer, default=0)
    deaths: Mapped[int] = mapped_column(Integer, default=0)
    assists: Mapped[int] = mapped_column(Integer, default=0)
    rounds_played: Mapped[int] = mapped_column(Integer, default=0)
    headshots: Mapped[int] = mapped_column(Integer, default=0)
    bodyshots: Mapped[int] = mapped_column(Integer, default=0)
    legshots: Mapped[int] = mapped_column(Integer, default=0)
    damage_dealt: Mapped[int] = mapped_column(Integer, default=0)
    damage_received: Mapped[int] = mapped_column(Integer, default=0)

    # Impact stats
    first_bloods: Mapped[int] = mapped_column(Integer, default=0)
    first_deaths: Mapped[int] = mapped_column(Integer, default=0)
    plants: Mapped[int] = mapped_column(Integer, default=0)
    defuses: Mapped[int] = mapped_column(Integer, default=0)

    # Economy — match-level aggregates from player.economy API field.
    # Populated on v4 syncs; NULL for rows synced before this migration.
    credits_spent: Mapped[int | None] = mapped_column(Integer, nullable=True)
    avg_loadout: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Behavior / metadata
    afk_rounds: Mapped[int] = mapped_column(Integer, default=0)
    rounds_in_spawn: Mapped[int] = mapped_column(Integer, default=0)
    competitive_tier: Mapped[int | None] = mapped_column(Integer, nullable=True)
    started_at: Mapped[str] = mapped_column(String, nullable=False)  # ISO8601 UTC

    # Relationship — loaded eagerly via Match.players selectin
    match: Mapped[Match] = relationship(back_populates="players", lazy="selectin")

    # ------------------------------------------------------------------
    # Computed properties
    # ------------------------------------------------------------------

    @property
    def kd_ratio(self) -> float:
        return round(self.kills / max(self.deaths, 1), 2)

    @property
    def headshot_pct(self) -> float:
        total = self.headshots + self.bodyshots + self.legshots
        return round(self.headshots / max(total, 1) * 100, 1)

    @property
    def acs(self) -> int:
        """Average combat score per round (score ÷ rounds_played)."""
        return round(self.score / max(self.rounds_played, 1))

    def __repr__(self) -> str:
        return (
            f"<MatchPlayer {self.agent_name} "
            f"K={self.kills} D={self.deaths} A={self.assists} won={self.won}>"
        )


# Public alias — callers import MatchPlayer from this module
MatchPlayer = OrmMatchPlayer


# ---------------------------------------------------------------------------
# rounds
# ---------------------------------------------------------------------------


class Round(Base):
    """One row per round — outcome + bomb events."""

    __tablename__ = "rounds"
    __table_args__ = (
        UniqueConstraint("match_id", "round_number", name="uq_match_round"),
        # winning_team must be "Red" or "Blue" (no draws at the round level).
        CheckConstraint("winning_team IN ('Red', 'Blue')", name="ck_round_winning_team"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    match_id: Mapped[str] = mapped_column(
        ForeignKey("matches.match_id", ondelete="CASCADE"), nullable=False
    )
    round_number: Mapped[int] = mapped_column(Integer, nullable=False)
    winning_team: Mapped[str] = mapped_column(String, nullable=False)  # "Red" | "Blue"
    result_code: Mapped[str] = mapped_column(String, nullable=False)  # e.g. "Elimination"
    bomb_planted: Mapped[bool] = mapped_column(Boolean, default=False)
    plant_site: Mapped[str | None] = mapped_column(String, nullable=True)
    planter_puuid: Mapped[str | None] = mapped_column(String, nullable=True)
    bomb_defused: Mapped[bool] = mapped_column(Boolean, default=False)
    defuser_puuid: Mapped[str | None] = mapped_column(String, nullable=True)

    # Relationships
    match: Mapped[Match] = relationship(back_populates="rounds")
    kills: Mapped[list[Kill]] = relationship(back_populates="round", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<Round {self.round_number} won={self.winning_team} {self.result_code}>"


# ---------------------------------------------------------------------------
# kills
# ---------------------------------------------------------------------------


class Kill(Base):
    """One row per kill event within a round."""

    __tablename__ = "kills"
    __table_args__ = (
        Index("idx_kills_killer", "killer_puuid", "match_id"),
        # round_id is the FK loaded by `selectinload(Round.kills)` for every
        # round during round-level analysis.  Without this index, that query
        # degrades to a full kills-table scan per round.
        Index("idx_kills_round", "round_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    round_id: Mapped[int] = mapped_column(
        ForeignKey("rounds.id", ondelete="CASCADE"), nullable=False
    )
    match_id: Mapped[str] = mapped_column(String, nullable=False)  # denormalised
    round_number: Mapped[int] = mapped_column(Integer, nullable=False)  # denormalised
    time_in_round_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    killer_puuid: Mapped[str] = mapped_column(String, nullable=False)
    victim_puuid: Mapped[str] = mapped_column(String, nullable=False)
    weapon_name: Mapped[str | None] = mapped_column(String, nullable=True)
    is_headshot: Mapped[bool] = mapped_column(Boolean, default=False)
    assistants_json: Mapped[str] = mapped_column(Text, default="[]")  # JSON array of PUUIDs

    # Relationship
    round: Mapped[Round] = relationship(back_populates="kills")

    def __repr__(self) -> str:
        return (
            f"<Kill round={self.round_number} "
            f"{self.killer_puuid[:8]}→{self.victim_puuid[:8]} hs={self.is_headshot}>"
        )


# ---------------------------------------------------------------------------
# sync_log
# ---------------------------------------------------------------------------


class SyncLog(Base):
    """Audit trail for every sync operation — one row per sync attempt."""

    __tablename__ = "sync_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    puuid: Mapped[str] = mapped_column(String, nullable=False)
    started_at: Mapped[str] = mapped_column(String, default=_now_iso)  # ISO8601 UTC
    completed_at: Mapped[str | None] = mapped_column(String, nullable=True)
    matches_fetched: Mapped[int] = mapped_column(Integer, default=0)
    matches_new: Mapped[int] = mapped_column(Integer, default=0)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)

    def __repr__(self) -> str:
        return (
            f"<SyncLog puuid={self.puuid[:8]} "
            f"new={self.matches_new}/{self.matches_fetched} err={self.error is not None}>"
        )


# ---------------------------------------------------------------------------
# meta_cache
# ---------------------------------------------------------------------------


class MetaCache(Base):
    """Cache for scraped external content (patch notes, articles, etc.).

    TTL tiers: stable (30d), semi_stable (5d), volatile (12h).
    expires_at is ISO8601 UTC — compare lexicographically since SQLite has no
    native TIMESTAMP. content_hash (SHA256 prefix) detects unchanged re-fetches.
    """

    __tablename__ = "meta_cache"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    url: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    source: Mapped[str] = mapped_column(String, nullable=False)  # "patch_note", "youtube", "web", …
    content_hash: Mapped[str] = mapped_column(String, nullable=False)  # SHA256[:16]
    ttl_tier: Mapped[str] = mapped_column(
        String, nullable=False
    )  # "stable" | "semi_stable" | "volatile"
    fetched_at: Mapped[str] = mapped_column(String, default=_now_iso)
    expires_at: Mapped[str] = mapped_column(String, nullable=False)
    content_text: Mapped[str] = mapped_column(Text, nullable=False)  # extracted text, not raw HTML

    def __repr__(self) -> str:
        return f"<MetaCache {self.url[:40]} tier={self.ttl_tier} expires={self.expires_at[:10]}>"


# ---------------------------------------------------------------------------
# patch_versions
# ---------------------------------------------------------------------------


class PatchVersion(Base):
    """Records each unique game version detected during a sync."""

    __tablename__ = "patch_versions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    game_version: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    detected_at: Mapped[str] = mapped_column(String, default=_now_iso)

    def __repr__(self) -> str:
        return f"<PatchVersion {self.game_version} at {self.detected_at[:10]}>"

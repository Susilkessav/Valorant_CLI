"""SQLAlchemy ORM table definitions.

Ten tables:
  players            — identity + current/peak rank snapshot
  matches            — match metadata + team scores
  match_players      — one row per player per match (stats junction table)
  rounds             — per-round outcome + bomb events
  round_players      — per-player per-round stats (score, kills, economy, survival)
  kills              — per-kill events within a round
  mmr_history        — rank snapshot per sync (enables progression timeline)
  sync_log           — operational audit trail for each data sync
  coaching_sessions  — one row per coaching conversation / focus block
  coaching_notes     — individual takeaways / action items from a session

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
    in valocoach.data.api_models. Re-exported as MatchPlayer from this module.
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
    round_players: Mapped[list[RoundPlayer]] = relationship(
        back_populates="round", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Round {self.round_number} won={self.winning_team} {self.result_code}>"


# ---------------------------------------------------------------------------
# round_players
# ---------------------------------------------------------------------------


class RoundPlayer(Base):
    """Per-player, per-round stats — economy, combat score, and survival flag.

    One row per (round_id, puuid).  Populated by the mapper from the
    ``round.stats`` list in a v4 MatchDetails response.

    Why a separate table:
      - Per-round loadout_value / remaining_credits unlock a correct econ
        rating that the match-level aggregate (credits_spent) approximates.
      - survived=False rows identify deaths without scanning the kills table.
      - Round-level score variance is not derivable from match_players alone.
    """

    __tablename__ = "round_players"
    __table_args__ = (
        UniqueConstraint("round_id", "puuid", name="uq_round_player"),
        Index("idx_rp_match_puuid", "match_id", "puuid"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    round_id: Mapped[int] = mapped_column(
        ForeignKey("rounds.id", ondelete="CASCADE"), nullable=False
    )
    match_id: Mapped[str] = mapped_column(String, nullable=False)  # denormalised
    puuid: Mapped[str] = mapped_column(String, nullable=False)
    team: Mapped[str] = mapped_column(String, nullable=False)  # "Red" | "Blue"
    score: Mapped[int] = mapped_column(Integer, default=0)
    kills: Mapped[int] = mapped_column(Integer, default=0)
    headshots: Mapped[int] = mapped_column(Integer, default=0)
    bodyshots: Mapped[int] = mapped_column(Integer, default=0)
    legshots: Mapped[int] = mapped_column(Integer, default=0)
    damage_dealt: Mapped[int] = mapped_column(Integer, default=0)  # sum(damage_events[*].damage)
    loadout_value: Mapped[int | None] = mapped_column(Integer, nullable=True)
    remaining_credits: Mapped[int | None] = mapped_column(Integer, nullable=True)
    survived: Mapped[bool] = mapped_column(Boolean, default=True)
    was_afk: Mapped[bool] = mapped_column(Boolean, default=False)
    stayed_in_spawn: Mapped[bool] = mapped_column(Boolean, default=False)

    # Relationship
    round: Mapped[Round] = relationship(back_populates="round_players")

    def __repr__(self) -> str:
        return (
            f"<RoundPlayer round={self.round_id} "
            f"puuid={self.puuid[:8]}… "
            f"kills={self.kills} survived={self.survived}>"
        )


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
# mmr_history
# ---------------------------------------------------------------------------


class MMRHistory(Base):
    """Rank snapshot recorded at each successful sync.

    One row is inserted per sync, but only when the player's ELO has changed
    since the previous snapshot — purely identical re-syncs produce no row.

    This table is the source of truth for:
      - Rank progression charts (tier/RR over time)
      - Session delta ("you gained +47 RR in tonight's session")
      - Peak-rank detection within a chosen time window
      - Coach context: "currently Diamond 2 after climbing from Platinum 1"

    recorded_at is the sync timestamp (not the game timestamp).  It
    correlates closely with when the player's last match ended but is not
    exact — use match.started_at for per-game granularity.
    """

    __tablename__ = "mmr_history"
    __table_args__ = (Index("idx_mmr_history_puuid", "puuid", "recorded_at"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    puuid: Mapped[str] = mapped_column(
        ForeignKey("players.puuid", ondelete="CASCADE"), nullable=False
    )
    recorded_at: Mapped[str] = mapped_column(String, nullable=False, default=_now_iso)
    tier: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    tier_patched: Mapped[str] = mapped_column(String, nullable=False, default="Unranked")
    rr: Mapped[int] = mapped_column(Integer, nullable=False, default=0)  # ranking_in_tier (0-99)
    elo: Mapped[int] = mapped_column(Integer, nullable=False, default=0)  # absolute MMR
    mmr_change: Mapped[int | None] = mapped_column(Integer, nullable=True)  # +/- RR from last game

    def __repr__(self) -> str:
        sign = "+" if (self.mmr_change or 0) >= 0 else ""
        change = f" ({sign}{self.mmr_change})" if self.mmr_change is not None else ""
        return (
            f"<MMRHistory {self.tier_patched} {self.rr}RR{change} "
            f"elo={self.elo} at={self.recorded_at[:10]}>"
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
# coaching_sessions
# ---------------------------------------------------------------------------


class CoachingSession(Base):
    """One row per coaching conversation or focus block.

    A session groups related coaching notes and provides temporal context
    (when did the player seek feedback, what were they focusing on?).

    Design decisions:
      - puuid FK with CASCADE — sessions belong to a player; deleting the
        player purges their session history.
      - ended_at is NULL while the session is in progress.  The CLI sets it
        when the user closes the coaching conversation or runs `coach close`.
      - agent / map are optional focus hints, not enforcement — a session
        can cover multiple agents or maps (notes carry their own context).
      - session_title defaults to the started_at date to give humans a
        recognisable anchor without mandatory input.
    """

    __tablename__ = "coaching_sessions"
    __table_args__ = (Index("idx_cs_puuid", "puuid", "started_at"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    puuid: Mapped[str] = mapped_column(
        ForeignKey("players.puuid", ondelete="CASCADE"), nullable=False
    )
    started_at: Mapped[str] = mapped_column(String, nullable=False, default=_now_iso)
    ended_at: Mapped[str | None] = mapped_column(String, nullable=True)
    session_title: Mapped[str | None] = mapped_column(String, nullable=True)
    # Optional focus context for the session
    focus_agent: Mapped[str | None] = mapped_column(String, nullable=True)
    focus_map: Mapped[str | None] = mapped_column(String, nullable=True)

    # Relationship
    notes: Mapped[list[CoachingNote]] = relationship(
        back_populates="session", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        status = "open" if self.ended_at is None else "closed"
        return (
            f"<CoachingSession id={self.id} puuid={self.puuid[:8]}… "
            f"title={self.session_title!r} {status}>"
        )


# ---------------------------------------------------------------------------
# coaching_notes
# ---------------------------------------------------------------------------


class CoachingNote(Base):
    """One row per coaching takeaway or action item within a session.

    Notes are the primary output of a coaching session — concrete,
    actionable observations the player can review and act on.

    Design decisions:
      - session_id FK with CASCADE — deleting a session removes all its notes.
      - match_id is a plain string (no FK) — the same rationale as
        match_players.puuid: not every note is tied to a stored match, and
        adding a FK would block notes for matches not yet synced.
      - category is free-form TEXT (no CHECK constraint) — coaching categories
        evolve; validation lives in the CLI / repository layer.
        Suggested values: "aim", "positioning", "economy", "rotation",
        "mindset", "agent_usage", "comms", "general".
      - priority: 1 (low), 2 (medium, default), 3 (high).
      - resolved: False until the player marks the note as addressed.
    """

    __tablename__ = "coaching_notes"
    __table_args__ = (
        Index("idx_cn_session", "session_id"),
        Index("idx_cn_puuid_resolved", "puuid", "resolved"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[int] = mapped_column(
        ForeignKey("coaching_sessions.id", ondelete="CASCADE"), nullable=False
    )
    # Denormalised — allows querying all notes for a player without joining sessions
    puuid: Mapped[str] = mapped_column(String, nullable=False)
    body: Mapped[str] = mapped_column(Text, nullable=False)
    category: Mapped[str] = mapped_column(String, nullable=False, default="general")
    priority: Mapped[int] = mapped_column(Integer, nullable=False, default=2)  # 1-3
    resolved: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    # Optional reference to a specific match this note was generated from
    match_id: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[str] = mapped_column(String, nullable=False, default=_now_iso)
    resolved_at: Mapped[str | None] = mapped_column(String, nullable=True)

    # Relationship
    session: Mapped[CoachingSession] = relationship(back_populates="notes")

    def __repr__(self) -> str:
        state = "✓" if self.resolved else "○"
        prio = {1: "low", 2: "med", 3: "high"}.get(self.priority, "?")
        return f"<CoachingNote {state} [{prio}] [{self.category}] {self.body[:40]!r}>"


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

"""Repository layer — converts Pydantic API models to ORM rows and back.

All functions take an open SQLAlchemy AsyncSession. Callers are responsible for
wrapping calls in session_scope() from database.py.

Public API:
    await upsert_player(session, account, mmr)                -> Player
    await upsert_match(session, match_data)                   -> Match | None
    await get_player(session, puuid)                          -> Player | None
    await get_player_by_name(session, name, tag)              -> Player | None
    await get_recent_matches(session, puuid, limit, queue_id) -> list[MatchPlayer]
    await match_exists(session, match_id)                     -> bool
    await get_match(session, match_id)                        -> Match | None
    await start_sync(session, puuid)                          -> SyncLog
    complete_sync(session, log, ...)                          -> None
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from valocoach.data.models import AccountData, MatchData, MMRData
from valocoach.data.orm_models import Match, MatchPlayer, Player, SyncLog

logger = logging.getLogger(__name__)

# Default queue for all stats queries — keeps coaching data clean
DEFAULT_QUEUE = "competitive"


def _unix_to_iso(ts: int) -> str:
    """Convert a unix timestamp (seconds) to a UTC ISO8601 string."""
    return datetime.fromtimestamp(ts, tz=UTC).isoformat()


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


# ---------------------------------------------------------------------------
# Player
# ---------------------------------------------------------------------------


async def upsert_player(
    session: AsyncSession,
    account: AccountData,
    mmr: MMRData,
) -> Player:
    """Insert or update a Player row from fresh account + MMR data.

    Uses session.merge() - safe to call repeatedly; second call updates
    rank/level without creating a duplicate row.
    """
    player = Player(
        puuid=account.puuid,
        riot_name=account.name,
        riot_tag=account.tag,
        region=account.region,
        account_level=account.account_level,
        current_tier=mmr.current_data.currenttier,
        current_tier_patched=mmr.current_data.currenttierpatched,
        current_rr=mmr.current_data.ranking_in_tier,
        elo=mmr.current_data.elo,
        peak_tier=mmr.highest_rank.tier,
        peak_tier_patched=mmr.highest_rank.patched_tier,
    )
    merged = await session.merge(player)
    logger.debug(
        "upsert_player %s#%s tier=%s elo=%d",
        account.name,
        account.tag,
        mmr.current_data.currenttierpatched,
        mmr.current_data.elo,
    )
    return merged


async def get_player(session: AsyncSession, puuid: str) -> Player | None:
    """Fetch a player by PUUID."""
    return await session.get(Player, puuid)


async def get_player_by_name(session: AsyncSession, name: str, tag: str) -> Player | None:
    """Fetch a player by Riot name + tag.

    Uses func.lower() for case-insensitive exact matching — avoids ilike()
    which treats % and _ as SQL wildcards and could match unintended rows.
    """
    stmt = select(Player).where(
        func.lower(Player.riot_name) == name.lower(),
        func.lower(Player.riot_tag) == tag.lower(),
    )
    result = await session.scalars(stmt)
    return result.first()


# ---------------------------------------------------------------------------
# Match
# ---------------------------------------------------------------------------


async def match_exists(session: AsyncSession, match_id: str) -> bool:
    """Return True if this match is already stored."""
    stmt = select(Match.match_id).where(Match.match_id == match_id)
    result = await session.scalars(stmt)
    return result.first() is not None


async def upsert_match(session: AsyncSession, match_data: MatchData) -> Match | None:
    """Persist a MatchData (metadata + all participants) to the DB.

    Returns None (silently) if:
      - The match is already stored (idempotent - safe to call on every sync)
      - match_data.is_available is False (incomplete/pending match)

    Uses IntegrityError as the deduplication guard for concurrency safety -
    a pre-check + insert race condition could store duplicates under concurrent
    syncs; catching IntegrityError makes the operation atomic.
    """
    if not match_data.is_available:
        logger.debug("match not available — skip")
        return None

    meta = match_data.metadata
    if await session.get(Match, meta.matchid) is not None:
        logger.debug("match %s already stored - skip", meta.matchid[:8])
        return None

    red_won = match_data.teams.red.has_won
    blue_won = match_data.teams.blue.has_won
    winning_team = "Red" if red_won else ("Blue" if blue_won else None)
    started_at = _unix_to_iso(meta.game_start)

    match = Match(
        match_id=meta.matchid,
        map_name=meta.map,
        queue_id=meta.mode_id,
        is_ranked=meta.mode_id.lower() == "competitive",
        game_version=meta.game_version,
        game_length_secs=meta.game_length,
        season_short=meta.season_id,
        region=meta.region,
        rounds_played=meta.rounds_played,
        red_score=match_data.teams.red.rounds_won,
        blue_score=match_data.teams.blue.rounds_won,
        winning_team=winning_team,
        started_at=started_at,
    )

    for player in match_data.players.all_players:
        won = red_won if player.team.lower() == "red" else blue_won
        mp = MatchPlayer(
            match_id=meta.matchid,
            puuid=player.puuid or None,
            agent_name=player.character,
            team=player.team,
            won=won,
            score=player.stats.score,
            kills=player.stats.kills,
            deaths=player.stats.deaths,
            assists=player.stats.assists,
            rounds_played=meta.rounds_played,
            headshots=player.stats.headshots,
            bodyshots=player.stats.bodyshots,
            legshots=player.stats.legshots,
            afk_rounds=player.behavior.afk_rounds,
            rounds_in_spawn=player.behavior.rounds_in_spawn,
            competitive_tier=player.currenttier or None,
            started_at=started_at,
        )
        match.players.append(mp)

    try:
        async with session.begin_nested():  # SAVEPOINT rolls back only this insert on failure
            session.add(match)
            await session.flush()
    except IntegrityError:
        logger.debug("match %s already stored - skip (IntegrityError)", meta.matchid[:8])
        return None

    logger.debug(
        "stored match %s  %s  %s  players=%d",
        meta.matchid[:8],
        meta.map,
        meta.mode_id,
        len(match_data.players.all_players),
    )
    return match


# ---------------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------------


async def get_recent_matches(
    session: AsyncSession,
    puuid: str,
    limit: int = 10,
    queue_id: str | None = DEFAULT_QUEUE,
) -> list[MatchPlayer]:
    """Return a player's most recent MatchPlayer rows, newest first.

    Defaults to competitive only - prevents unrated/deathmatch/swift rows
    from polluting coaching stats. Pass queue_id=None to return all modes.

    Each row has `.match` pre-loaded (no lazy-load DetachedInstanceError risk).
    """
    stmt = (
        select(MatchPlayer)
        .join(MatchPlayer.match)
        .where(MatchPlayer.puuid == puuid)
        .order_by(Match.started_at.desc())
        .limit(limit)
        .options(selectinload(MatchPlayer.match))
    )
    if queue_id is not None:
        stmt = stmt.where(Match.queue_id == queue_id)
    result = await session.scalars(stmt)
    return list(result.all())


async def get_match(session: AsyncSession, match_id: str) -> Match | None:
    """Fetch a single match with players and rounds pre-loaded."""
    stmt = (
        select(Match)
        .where(Match.match_id == match_id)
        .options(
            selectinload(Match.players),
            selectinload(Match.rounds),
        )
    )
    result = await session.scalars(stmt)
    return result.first()


# ---------------------------------------------------------------------------
# SyncLog
# ---------------------------------------------------------------------------


async def start_sync(session: AsyncSession, puuid: str) -> SyncLog:
    """Open a new sync log entry. Call complete_sync() when done."""
    log = SyncLog(puuid=puuid)
    session.add(log)
    await session.flush()
    return log


def complete_sync(
    session: AsyncSession,
    log: SyncLog,
    *,
    matches_fetched: int,
    matches_new: int,
    error: str | None = None,
) -> None:
    """Close a sync log entry with results."""
    log.completed_at = _now_iso()
    log.matches_fetched = matches_fetched
    log.matches_new = matches_new
    log.error = error

"""Repository layer — DB operations only.  No API-shape knowledge lives here.

All functions take an open SQLAlchemy AsyncSession. Callers are responsible for
wrapping calls in session_scope() from database.py.

API-shape-to-ORM mapping is in mapper.py.  Update that file when HenrikDev
changes their schema; this file stays stable.

Public API:
    await upsert_player(session, account, mmr)                -> Player
    await record_mmr_snapshot(session, puuid, mmr)            -> MMRHistory | None
    await get_mmr_history(session, puuid, limit)              -> list[MMRHistory]
    await upsert_match(session, match_data)                   -> Match | None
    await upsert_match_details(session, details)              -> Match | None  (v4)
    await get_player(session, puuid)                          -> Player | None
    await get_player_by_name(session, name, tag)              -> Player | None
    await get_recent_matches(session, puuid, limit, queue_id) -> list[MatchPlayer]
    await get_recent_matches_full(session, puuid, limit, ...) -> list[Match]
    await match_exists(session, match_id)                     -> bool
    await get_match(session, match_id)                        -> Match | None
    await get_post_game_match(session, match_id)              -> Match | None  (full tree)
    await start_sync(session, puuid)                          -> SyncLog
    complete_sync(session, log, ...)                          -> None
    await close_stale_syncs(session, puuid, exclude_id)      -> int
    await create_coaching_session(session, puuid, ...)        -> CoachingSession
    await end_coaching_session(session, session_id)           -> CoachingSession | None
    await get_coaching_sessions(session, puuid, limit)        -> list[CoachingSession]
    await get_open_coaching_session(session, puuid)           -> CoachingSession | None
    await add_coaching_note(session, session_id, body, ...)   -> CoachingNote
    await get_coaching_notes(session, session_id)             -> list[CoachingNote]
    await get_open_notes(session, puuid, ...)                 -> list[CoachingNote]
    await resolve_note(session, note_id)                      -> CoachingNote | None
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from valocoach.data.api_models import AccountData, MatchData, MatchDetails, MMRData
from valocoach.data.mapper import match_from_details, player_from_account_mmr
from valocoach.data.orm_models import (
    CoachingNote,
    CoachingSession,
    Kill,
    Match,
    MatchPlayer,
    MMRHistory,
    Player,
    Round,
    RoundPlayer,
    SyncLog,
)

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

    Uses session.merge() — safe to call repeatedly; the second call updates
    rank/level without creating a duplicate row.

    Mapping from API shapes to the ORM row is handled by mapper.player_from_account_mmr.
    """
    player = player_from_account_mmr(account, mmr)
    merged = await session.merge(player)
    logger.debug(
        "upsert_player %s#%s tier=%s elo=%d",
        account.name,
        account.tag,
        mmr.current_data.currenttierpatched,
        mmr.current_data.elo,
    )
    return merged


async def record_mmr_snapshot(
    session: AsyncSession,
    puuid: str,
    mmr: MMRData,
) -> MMRHistory | None:
    """Insert a rank snapshot if ELO has changed since the last recorded one.

    Skips insertion when the player's ELO matches the most recent row —
    avoids duplicate rows when the user syncs multiple times without playing.

    Returns the new MMRHistory row, or None if skipped (no change).
    """
    cd = mmr.current_data
    new_elo = cd.elo

    # Look up the most recent snapshot for this puuid.
    stmt = (
        select(MMRHistory)
        .where(MMRHistory.puuid == puuid)
        .order_by(MMRHistory.recorded_at.desc())
        .limit(1)
    )
    result = await session.scalars(stmt)
    latest = result.first()

    if latest is not None and latest.elo == new_elo:
        logger.debug("mmr_history skip — elo unchanged (%d) for %s…", new_elo, puuid[:8])
        return None

    snapshot = MMRHistory(
        puuid=puuid,
        recorded_at=_now_iso(),
        tier=cd.currenttier,
        tier_patched=cd.currenttierpatched,
        rr=cd.ranking_in_tier,
        elo=new_elo,
        mmr_change=cd.mmr_change_to_last_game or None,
    )
    session.add(snapshot)
    await session.flush()
    logger.debug(
        "mmr_history insert %s elo=%d rr=%d tier=%s",
        puuid[:8],
        new_elo,
        cd.ranking_in_tier,
        cd.currenttierpatched,
    )
    return snapshot


async def get_mmr_history(
    session: AsyncSession,
    puuid: str,
    limit: int = 50,
) -> list[MMRHistory]:
    """Return up to *limit* rank snapshots for *puuid*, newest first.

    Typical callers:
      - Coaching context builder: last 10 rows for a rank-progression summary.
      - Stats CLI profile panel: last 50 rows for a sparkline.
    """
    stmt = (
        select(MMRHistory)
        .where(MMRHistory.puuid == puuid)
        .order_by(MMRHistory.recorded_at.desc())
        .limit(limit)
    )
    result = await session.scalars(stmt)
    return list(result.all())


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
    if await session.get(Match, meta.match_id) is not None:
        logger.debug("match %s already stored - skip", meta.match_id[:8])
        return None

    red_won = match_data.teams.red.has_won
    blue_won = match_data.teams.blue.has_won
    winning_team = "Red" if red_won else ("Blue" if blue_won else None)
    started_at = _unix_to_iso(meta.game_start)

    match = Match(
        match_id=meta.match_id,
        map_name=meta.map_name,
        queue_id=meta.queue_id,
        is_ranked=meta.queue_id.lower() == "competitive",
        game_version=meta.game_version,
        game_length_secs=meta.game_length_secs,
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
            match_id=meta.match_id,
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
            damage_dealt=player.stats.damage_dealt,
            damage_received=player.stats.damage_received,
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
        logger.debug("match %s already stored - skip (IntegrityError)", meta.match_id[:8])
        return None

    logger.debug(
        "stored match %s  %s  %s  players=%d",
        meta.match_id[:8],
        meta.map_name,
        meta.queue_id,
        len(match_data.players.all_players),
    )
    return match


async def upsert_match_details(
    session: AsyncSession,
    details: MatchDetails,
) -> Match | None:
    """Persist a v4 MatchDetails (full match + players + rounds + kills) to the DB.

    Idempotent — returns None silently if the match is already stored.

    All API-to-ORM field mapping is handled by mapper.match_from_details.
    This function only handles DB-level concerns: existence check, add, flush,
    and IntegrityError guard for concurrency safety.
    """
    match_id = details.metadata.match_id

    if await session.get(Match, match_id) is not None:
        logger.debug("match %s already stored — skip", match_id[:8])
        return None

    match = match_from_details(details)

    try:
        async with session.begin_nested():  # SAVEPOINT — rolls back only this insert on conflict
            session.add(match)
            await session.flush()
    except IntegrityError:
        logger.debug("match %s already stored — skip (IntegrityError)", match_id[:8])
        return None

    logger.debug(
        "stored v4 match %s  %s  %s  players=%d rounds=%d kills=%d",
        match_id[:8],
        details.metadata.map.name or "",
        details.metadata.queue.id,
        len(details.players),
        len(details.rounds),
        len(details.kills),
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


async def get_post_game_match(
    session: AsyncSession,
    match_id: str,
) -> Match | None:
    """Load a single match with the full round-level tree for post-game analysis.

    Eagerly loads all relationships needed by ``run_analyzers()``:

      - ``Match.players``           — agent/team per participant
      - ``Match.rounds → kills``    — per-kill timing + spatial data
      - ``Match.rounds → round_players`` — economy + ability cast counts

    All three levels use ``selectinload`` so that accessing the loaded
    collections after the session is closed never triggers a lazy-load
    (``expire_on_commit=False`` is set on the engine, so attributes survive
    the ``session.commit()`` inside ``session_scope()``).

    Returns ``None`` when ``match_id`` is not stored.
    """
    stmt = (
        select(Match)
        .where(Match.match_id == match_id)
        .options(
            selectinload(Match.players),
            selectinload(Match.rounds).selectinload(Round.kills),
            selectinload(Match.rounds).selectinload(Round.round_players),
        )
    )
    result = await session.scalars(stmt)
    return result.first()


async def get_recent_matches_full(
    session: AsyncSession,
    puuid: str,
    limit: int = 10,
    queue_id: str | None = DEFAULT_QUEUE,
) -> list[Match]:
    """Return full Match trees (players + rounds + kills) for a puuid.

    Heavier than get_recent_matches: fires additional selectin queries
    for rounds and kills. Use only when the caller needs round-level
    data (the round analyzer for KAST / clutch / trade) — aggregate-only
    callers (stats CLI, profile card) should keep using get_recent_matches.
    """
    stmt = (
        select(Match)
        .join(Match.players)
        .where(MatchPlayer.puuid == puuid)
        .order_by(Match.started_at.desc())
        .limit(limit)
        .options(
            selectinload(Match.players),
            selectinload(Match.rounds).selectinload(Round.kills),
        )
    )
    if queue_id is not None:
        stmt = stmt.where(Match.queue_id == queue_id)
    result = await session.scalars(stmt)
    # Distinct: a Match can surface twice if the join picks more than one
    # MatchPlayer row (shouldn't happen with our puuid filter, but cheap
    # defense — .unique() is a no-op on a clean result).
    return list(result.unique().all())


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


async def close_stale_syncs(
    session: AsyncSession,
    puuid: str,
    exclude_id: int,
) -> int:
    """Mark all incomplete SyncLog rows (other than *exclude_id*) as interrupted.

    Returns the number of rows closed.  A row is "incomplete" when its
    completed_at is NULL — meaning the previous sync process was killed or
    crashed before _finalise() could run.
    """
    stmt = (
        select(SyncLog)
        .where(SyncLog.puuid == puuid)
        .where(SyncLog.completed_at.is_(None))
        .where(SyncLog.id != exclude_id)
    )
    result = await session.scalars(stmt)
    stale = list(result.all())
    now = _now_iso()
    for log in stale:
        log.completed_at = now
        log.error = "interrupted"
    return len(stale)


# ---------------------------------------------------------------------------
# Coaching sessions
# ---------------------------------------------------------------------------


async def create_coaching_session(
    session: AsyncSession,
    puuid: str,
    *,
    title: str | None = None,
    focus_agent: str | None = None,
    focus_map: str | None = None,
) -> CoachingSession:
    """Open a new coaching session for *puuid*.

    title defaults to the current date (YYYY-MM-DD) when omitted so every
    session has a human-readable anchor even without explicit input.

    Returns the persisted CoachingSession with its auto-assigned id.
    """
    now = _now_iso()
    coaching_session = CoachingSession(
        puuid=puuid,
        started_at=now,
        session_title=title or now[:10],  # YYYY-MM-DD prefix
        focus_agent=focus_agent,
        focus_map=focus_map,
    )
    session.add(coaching_session)
    await session.flush()
    logger.debug(
        "coaching_session create id=%d puuid=%s… title=%r",
        coaching_session.id,
        puuid[:8],
        coaching_session.session_title,
    )
    return coaching_session


async def end_coaching_session(
    session: AsyncSession,
    coaching_session_id: int,
) -> CoachingSession | None:
    """Close an open coaching session by setting ended_at to now.

    Returns the updated CoachingSession, or None if the id is not found.
    Does nothing (returns the row unchanged) if ended_at is already set.
    """
    cs = await session.get(CoachingSession, coaching_session_id)
    if cs is None:
        logger.debug("end_coaching_session: id=%d not found", coaching_session_id)
        return None
    if cs.ended_at is None:
        cs.ended_at = _now_iso()
        logger.debug("coaching_session close id=%d", coaching_session_id)
    return cs


async def get_coaching_sessions(
    session: AsyncSession,
    puuid: str,
    limit: int = 20,
) -> list[CoachingSession]:
    """Return up to *limit* coaching sessions for *puuid*, newest first."""
    stmt = (
        select(CoachingSession)
        .where(CoachingSession.puuid == puuid)
        .order_by(CoachingSession.started_at.desc())
        .limit(limit)
    )
    result = await session.scalars(stmt)
    return list(result.all())


async def get_open_coaching_session(
    session: AsyncSession,
    puuid: str,
) -> CoachingSession | None:
    """Return the most recent still-open session for *puuid*, or None.

    "Open" means ended_at IS NULL.  If the user starts the CLI without
    closing a previous session, this lets the CLI offer to resume it.
    """
    stmt = (
        select(CoachingSession)
        .where(CoachingSession.puuid == puuid)
        .where(CoachingSession.ended_at.is_(None))
        .order_by(CoachingSession.started_at.desc())
        .limit(1)
    )
    result = await session.scalars(stmt)
    return result.first()


# ---------------------------------------------------------------------------
# Coaching notes
# ---------------------------------------------------------------------------


async def add_coaching_note(
    session: AsyncSession,
    coaching_session_id: int,
    body: str,
    *,
    puuid: str,
    category: str = "general",
    priority: int = 2,
    match_id: str | None = None,
) -> CoachingNote:
    """Append a coaching note to *coaching_session_id*.

    Args:
        coaching_session_id: Parent session id — must exist (caller's responsibility).
        body:                 The coaching observation / action item text.
        puuid:                Player PUUID (denormalised so notes can be
                              queried without joining coaching_sessions).
        category:             Coaching category tag (e.g. "aim", "economy").
        priority:             1 (low) / 2 (medium) / 3 (high).
        match_id:             Optional reference to a specific stored match.

    Returns the persisted CoachingNote with its auto-assigned id.
    """
    note = CoachingNote(
        session_id=coaching_session_id,
        puuid=puuid,
        body=body,
        category=category,
        priority=max(1, min(3, priority)),  # clamp to 1-3
        match_id=match_id,
        created_at=_now_iso(),
    )
    session.add(note)
    await session.flush()
    logger.debug(
        "coaching_note add id=%d session=%d category=%s priority=%d",
        note.id,
        coaching_session_id,
        category,
        priority,
    )
    return note


async def get_coaching_notes(
    session: AsyncSession,
    coaching_session_id: int,
) -> list[CoachingNote]:
    """Return all notes for *coaching_session_id*, oldest first."""
    stmt = (
        select(CoachingNote)
        .where(CoachingNote.session_id == coaching_session_id)
        .order_by(CoachingNote.created_at.asc())
    )
    result = await session.scalars(stmt)
    return list(result.all())


async def get_open_notes(
    session: AsyncSession,
    puuid: str,
    *,
    resolved: bool = False,
    category: str | None = None,
    limit: int = 50,
) -> list[CoachingNote]:
    """Return coaching notes for *puuid* filtered by resolved status.

    Args:
        resolved:  When False (default) returns unresolved notes only.
                   Pass True to retrieve already-resolved notes.
        category:  Optional filter; pass None to return all categories.
        limit:     Maximum rows returned (default 50).

    Notes are ordered by priority DESC then created_at ASC so the most
    urgent unresolved items surface first.
    """
    stmt = (
        select(CoachingNote)
        .where(CoachingNote.puuid == puuid)
        .where(CoachingNote.resolved == resolved)
        .order_by(CoachingNote.priority.desc(), CoachingNote.created_at.asc())
        .limit(limit)
    )
    if category is not None:
        stmt = stmt.where(CoachingNote.category == category)
    result = await session.scalars(stmt)
    return list(result.all())


async def resolve_note(
    session: AsyncSession,
    note_id: int,
) -> CoachingNote | None:
    """Mark a coaching note as resolved.

    Sets resolved=True and resolved_at=now.  Idempotent — calling again on
    an already-resolved note updates resolved_at but leaves resolved=True.

    Returns the updated CoachingNote, or None if note_id is not found.
    """
    note = await session.get(CoachingNote, note_id)
    if note is None:
        logger.debug("resolve_note: id=%d not found", note_id)
        return None
    note.resolved = True
    note.resolved_at = _now_iso()
    logger.debug("coaching_note resolve id=%d", note_id)
    return note

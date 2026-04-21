"""valocoach.data — data layer public API.

Import from here; internal module structure is an implementation detail.

Typical startup sequence:
    from valocoach.data import init_engine, Base
    engine = init_engine(settings.data_dir / "valocoach.db")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

Fetching and storing data:
    from valocoach.data import HenrikClient, session_scope
    from valocoach.data import upsert_player, upsert_match, get_recent_matches
    # API errors live in valocoach.core.exceptions: APIError, RateLimitError, ServerError
    async with session_scope() as session:
        await upsert_match(session, match_data)

Name note:
    MatchPlayer here refers to the ORM table class (valocoach.data.orm_models).
    The Pydantic API shape is valocoach.data.models.MatchPlayer — import it
    directly from that module to avoid ambiguity.
"""

from __future__ import annotations

# HTTP client (Settings-based; raises APIError/RateLimitError/ServerError from core.exceptions)
from valocoach.data.api_client import HenrikClient

# Database setup
from valocoach.data.database import Base, ensure_db, get_engine, init_engine, session_scope

# Mapper functions (API shape → ORM shape — update mapper.py when schema changes)
from valocoach.data.mapper import match_from_details, player_from_account_mmr

# Pydantic API models (MatchPlayer intentionally NOT re-exported here
# to avoid collision with the ORM MatchPlayer — import from .models directly)
from valocoach.data.models import AccountData, MatchData, MMRData, MMRHistoryEntry

# ORM models
from valocoach.data.orm_models import Kill, Match, MatchPlayer, Player, Round, SyncLog

# Repository functions (DB operations only — no API-shape knowledge)
from valocoach.data.repository import (
    complete_sync,
    get_match,
    get_player,
    get_player_by_name,
    get_recent_matches,
    match_exists,
    start_sync,
    upsert_match,
    upsert_match_details,
    upsert_player,
)

__all__ = [
    # database
    "Base",
    "init_engine",
    "ensure_db",
    "get_engine",
    "session_scope",
    # client
    "HenrikClient",
    # pydantic models
    "AccountData",
    "MMRData",
    "MMRHistoryEntry",
    "MatchData",
    # orm models  (MatchPlayer = ORM table class)
    "Player",
    "Match",
    "MatchPlayer",
    "Round",
    "Kill",
    "SyncLog",
    # mapper (API shape → ORM shape)
    "match_from_details",
    "player_from_account_mmr",
    # repository (DB operations)
    "upsert_player",
    "upsert_match",
    "upsert_match_details",
    "match_exists",
    "get_player",
    "get_player_by_name",
    "get_recent_matches",
    "get_match",
    "start_sync",
    "complete_sync",
]

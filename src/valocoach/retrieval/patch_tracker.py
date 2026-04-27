from __future__ import annotations

import logging

from sqlalchemy import select

from valocoach.core.config import Settings
from valocoach.data.api_client import HenrikClient
from valocoach.data.database import session_scope
from valocoach.data.orm_models import PatchVersion
from valocoach.retrieval.cache import invalidate_volatile

log = logging.getLogger(__name__)


async def check_patch_update(settings: Settings) -> tuple[str, bool]:
    """Check whether the game version has changed since the last known patch.

    Fetches the current version from Henrik, compares against the most recent
    PatchVersion row, and — if it's new — records the row and evicts volatile
    cache entries so stale pick-rate data isn't served after a patch.

    Returns (current_version, is_new_patch).
    """
    # Resolve the current version from the API first, outside any DB session,
    # so we don't hold an open transaction while waiting on an HTTP response.
    async with HenrikClient(settings) as client:
        version_data = await client.get_version(settings.riot_region)

    current = version_data.get("version", "unknown")

    async with session_scope() as s:
        latest = await s.scalar(
            select(PatchVersion).order_by(PatchVersion.detected_at.desc())
        )

        if latest is None or latest.game_version != current:
            s.add(PatchVersion(game_version=current))

            if latest is not None:
                # Patch changed — volatile cache is immediately stale.
                # Run invalidation inside a fresh session after this one commits.
                pass

            log.info(
                "New patch detected: %s (previous: %s)",
                current,
                latest.game_version if latest else "none",
            )
            is_new = True
        else:
            log.debug("Patch unchanged: %s", current)
            is_new = False

    # Invalidate volatile entries after the PatchVersion row is committed,
    # so the two writes don't share a transaction.
    if is_new and latest is not None:
        count = await invalidate_volatile()
        log.info(
            "Patch %s → %s: invalidated %d volatile cache entries.",
            latest.game_version,
            current,
            count,
        )

    return current, is_new


async def get_current_patch() -> str | None:
    """Return the most recently detected game version from the DB.

    Returns None if no patch has been recorded yet (i.e., check_patch_update
    has never been called successfully).
    """
    async with session_scope() as s:
        latest = await s.scalar(
            select(PatchVersion).order_by(PatchVersion.detected_at.desc())
        )
        return latest.game_version if latest else None

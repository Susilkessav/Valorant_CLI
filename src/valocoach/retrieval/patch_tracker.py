from __future__ import annotations

import asyncio
import logging
import re
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import select

from valocoach.core.config import Settings
from valocoach.data.api_client import HenrikClient
from valocoach.data.database import session_scope
from valocoach.data.orm_models import PatchVersion
from valocoach.retrieval.cache import invalidate_volatile

log = logging.getLogger(__name__)

# HenrikDev's /v1/version response uses multiple version-like fields:
#   - "version"            → build number only (e.g. "25") — NOT user-facing
#   - "riotClientVersion"  → full string like "release-10.09-shipping-25-12345"
#   - "branch"             → e.g. "release-10.09"
#   - "build_ver"          → same shape as riotClientVersion (newer schemas)
# Earlier code grabbed "version" alone, which surfaced the build number to the
# user and broke patch-notes URL construction.  Extract X.YY from the first
# field that contains it; fall back to the raw "version" so we never store
# the literal string "unknown" if the response shape changes again.
_PATCH_NUMBER_RE = re.compile(r"(\d+)\.(\d{2})")


def _extract_patch_number(data: dict) -> str:
    """Pull "X.YY" from a HenrikDev /v1/version payload.

    Tries the X.YY-yielding fields in priority order; returns the raw
    "version" field (or "unknown") if none of them parse.

    Regional variants (e.g. Korean/JP "10.08-kr", "10.08.0-rc") match the
    leading numeric portion and intentionally drop any suffix — downstream
    consumers (patch-notes URL builder, meta_sync) only need the X.YY
    canonical form, and stripping the regional tag lets us share scraped
    notes across regions.
    """
    for key in ("riotClientVersion", "build_ver", "branch"):
        v = data.get(key)
        if not v:
            continue
        m = _PATCH_NUMBER_RE.search(str(v))
        if m:
            return f"{m.group(1)}.{m.group(2)}"
    # Last-ditch: "version" itself may already be X.YY in some regions.
    raw = data.get("version")
    if raw:
        m = _PATCH_NUMBER_RE.search(str(raw))
        if m:
            return f"{m.group(1)}.{m.group(2)}"
        return str(raw)
    return "unknown"


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

    # Extract user-facing X.YY (e.g. "10.09") rather than the bare build
    # number HenrikDev returns in the "version" field.
    current = _extract_patch_number(version_data)

    async with session_scope() as s:
        latest = await s.scalar(select(PatchVersion).order_by(PatchVersion.detected_at.desc()))

        if latest is None or latest.game_version != current:
            s.add(PatchVersion(game_version=current))
            # Volatile cache invalidation runs *after* this session commits
            # (see below) so the two writes never share a transaction.
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
    # so the two writes don't share a transaction.  Passing ``data_dir``
    # also nukes the corresponding live ChromaDB documents — patch-day
    # invalidation must clear BOTH halves of the cache or stale meta keeps
    # leaking into coach prompts via vector search.
    if is_new and latest is not None:
        count = await invalidate_volatile(settings.data_dir)
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
        latest = await s.scalar(select(PatchVersion).order_by(PatchVersion.detected_at.desc()))
        return latest.game_version if latest else None


def get_patch_staleness_days(data_dir: Path) -> float | None:
    """Return how many days have elapsed since the last patch check.

    Reads the most recent ``PatchVersion.detected_at`` timestamp from the local
    DB and computes ``(now - detected_at)`` in fractional days.

    Returns:
        float: days since last check (≥ 0.0).
        None:  no patch version recorded yet, or any DB / import error.

    This function is intentionally non-fatal — callers use the return value to
    decide whether to show a staleness warning, but never depend on it for
    correctness.  A ``None`` return is treated conservatively as "never checked"
    by the coach so the warning still fires.
    """

    async def _run() -> float | None:
        from valocoach.data.database import ensure_db

        await ensure_db(data_dir / "valocoach.db")
        async with session_scope() as s:
            latest = await s.scalar(select(PatchVersion).order_by(PatchVersion.detected_at.desc()))
            if latest is None:
                return None
            # detected_at is stored as an ISO string; parse it tolerantly.
            raw = latest.detected_at
            # Normalise trailing 'Z' (UTC shorthand) to '+00:00' for fromisoformat.
            if raw.endswith("Z"):
                raw = raw[:-1] + "+00:00"
            detected = datetime.fromisoformat(raw)
            if detected.tzinfo is None:
                detected = detected.replace(tzinfo=UTC)
            now = datetime.now(UTC)
            return max(0.0, (now - detected).total_seconds() / 86_400.0)

    try:
        return asyncio.run(_run())
    except Exception as exc:
        log.warning("Could not compute patch staleness: %s", exc)
        return None

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime
from pathlib import Path

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


async def run_patch_watcher(
    settings: "Settings",
    *,
    check_interval_hours: int = 24,
    on_new_patch: "Callable[[str], None] | None" = None,
) -> None:
    """Run a continuous async loop that checks for a new patch periodically.

    This is the scheduled-trigger half of the automation pipeline.
    When a new patch is detected it calls ``on_new_patch(version)`` —
    the default callback kicks off the full meta sync pipeline.

    Args:
        settings:              Application settings.
        check_interval_hours:  How often to poll HenrikDev (default 24 h).
        on_new_patch:          Optional async or sync callable invoked with
                               the new version string.  Defaults to running
                               :func:`~valocoach.retrieval.meta_sync.run_meta_sync`.

    The loop runs forever and should be cancelled externally (e.g. via
    ``asyncio.Task.cancel()`` or Ctrl-C in watch mode).
    """
    from collections.abc import Callable

    interval_secs = check_interval_hours * 3_600

    log.info(
        "Patch watcher started — checking every %dh (interval=%ds).",
        check_interval_hours,
        interval_secs,
    )

    while True:
        try:
            version, is_new = await check_patch_update(settings)
            if is_new:
                log.info("New patch detected by watcher: %s — triggering meta sync.", version)
                if on_new_patch is not None:
                    if asyncio.iscoroutinefunction(on_new_patch):
                        await on_new_patch(version)  # type: ignore[arg-type]
                    else:
                        on_new_patch(version)
                else:
                    # Default: run the full meta sync pipeline.
                    from valocoach.retrieval.meta_sync import run_meta_sync

                    result = await run_meta_sync(settings, force=True)
                    if result.ok:
                        log.info(
                            "Meta sync complete for patch %s — "
                            "meta_written=%s, meta_ingested=%s.",
                            version,
                            result.meta_written,
                            result.meta_ingested,
                        )
                    else:
                        log.warning(
                            "Meta sync finished with errors: %s",
                            "; ".join(result.errors),
                        )
            else:
                log.debug("Patch watcher: no change (%s), sleeping %dh.", version, check_interval_hours)
        except asyncio.CancelledError:
            log.info("Patch watcher cancelled.")
            return
        except Exception as exc:
            log.error("Patch watcher encountered an error: %s — will retry in %dh.", exc, check_interval_hours)

        await asyncio.sleep(interval_secs)


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
            latest = await s.scalar(
                select(PatchVersion).order_by(PatchVersion.detected_at.desc())
            )
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
    except Exception:
        return None

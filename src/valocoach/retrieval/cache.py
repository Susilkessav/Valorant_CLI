from __future__ import annotations

import hashlib
import logging
from datetime import UTC, datetime, timedelta

from sqlalchemy import select

from valocoach.data.database import session_scope
from valocoach.data.orm_models import MetaCache

log = logging.getLogger(__name__)

TTL_HOURS: dict[str, int] = {
    "stable": 24 * 30,  # 30 days — corpus/static knowledge
    "semi_stable": 24 * 5,  # 5 days  — patch notes, meta articles
    "volatile": 12,  # 12 hours — live standings, pick rates
}


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


async def get_cached(url: str) -> str | None:
    """Return cached content if it exists and hasn't expired, else None.

    Deletes the entry on expiry so the next call re-scrapes cleanly.
    """
    async with session_scope() as s:
        entry = await s.scalar(select(MetaCache).where(MetaCache.url == url))
        if entry is None:
            return None
        if entry.expires_at < _now_iso():
            log.debug("Cache expired for %s — evicting.", url)
            await s.delete(entry)
            return None
        log.debug("Cache hit for %s (expires %s)", url, entry.expires_at[:10])
        return entry.content_text


async def store_cached(
    url: str,
    text: str,
    source: str,
    ttl_tier: str = "semi_stable",
) -> None:
    """Upsert scraped content with a TTL derived from the tier.

    If an entry already exists for this URL, its text, hash, and expiry are
    refreshed in place. If the content hash is unchanged, the write still
    updates fetched_at and expires_at so the TTL window slides forward.
    """
    hours = TTL_HOURS.get(ttl_tier, 24)
    expires = (datetime.now(UTC) + timedelta(hours=hours)).isoformat()
    content_hash = hashlib.sha256(text.encode()).hexdigest()[:16]

    async with session_scope() as s:
        existing = await s.scalar(select(MetaCache).where(MetaCache.url == url))
        if existing is not None:
            existing.content_text = text
            existing.content_hash = content_hash
            existing.fetched_at = _now_iso()
            existing.expires_at = expires
            log.debug("Cache updated for %s (tier=%s)", url, ttl_tier)
        else:
            s.add(
                MetaCache(
                    url=url,
                    source=source,
                    content_hash=content_hash,
                    ttl_tier=ttl_tier,
                    fetched_at=_now_iso(),
                    expires_at=expires,
                    content_text=text,
                )
            )
            log.debug("Cache stored for %s (tier=%s, expires %s)", url, ttl_tier, expires[:10])


async def invalidate_volatile() -> int:
    """Delete all volatile-tier cache entries.

    Call this when a new patch version is detected — volatile content (live
    pick rates, standings) becomes stale immediately on patch day.
    """
    async with session_scope() as s:
        entries = (await s.scalars(select(MetaCache).where(MetaCache.ttl_tier == "volatile"))).all()
        count = len(entries)
        for entry in entries:
            await s.delete(entry)
    log.info("Invalidated %d volatile cache entries.", count)
    return count


async def purge_expired() -> int:
    """Delete all entries whose expires_at has passed.

    Useful as a periodic housekeeping call — the app doesn't auto-purge
    on read (only the specific URL is evicted on a cache miss).
    """
    now = _now_iso()
    async with session_scope() as s:
        entries = (await s.scalars(select(MetaCache).where(MetaCache.expires_at < now))).all()
        count = len(entries)
        for entry in entries:
            await s.delete(entry)
    log.info("Purged %d expired cache entries.", count)
    return count

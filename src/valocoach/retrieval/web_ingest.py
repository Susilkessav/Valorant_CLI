"""Smart URL ingest — classifies web-article chunks and routes lineup content correctly.

When ``valocoach ingest --url <article>`` is used, this module:

1. Scrapes the URL via the existing web scraper.
2. Chunks the article text with ``chunk_markdown`` (same chunker as the rest of the pipeline).
3. Runs every chunk through the anchor-based classifier (same as YouTube ingest, D4).
4. Lineup-classified chunks → LLM metadata extraction → LIVE_COLLECTION as ``type=lineup``.
5. All other relevant chunks → LIVE_COLLECTION as ``type=web`` (not STATIC, so TTL applies).
6. Off-topic chunks (intro/sponsor/noise) are dropped silently.

This means a Dotesports or Gameleap lineup guide ingested via ``--url`` will surface
in ``valocoach lineup Sova --map Haven`` just like a YouTube video would.
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from valocoach.core.config import Settings

log = logging.getLogger(__name__)

# Web-scraped content stays live for 30 days (shorter than YouTube's 60 — articles
# go stale faster as the meta shifts).
_WEB_TTL_SECONDS = 30 * 24 * 3600

# Minimum chunk character length — below this the chunk is almost certainly a
# navigation element, heading, or caption, not tactical content.
_MIN_CHUNK_CHARS = 80


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class WebIngestResult:
    """Summary of what ``ingest_web_url`` did — mirrors YouTubeIngestPlan structure."""

    url: str
    title: str
    domain: str
    skipped_reason: str | None = None   # "scrape_failed" | "no_chunks"
    fetched_count: int = 0
    kept_count: int = 0
    lineup_count: int = 0
    web_count: int = 0
    dropped_counts: dict[str, int] = field(default_factory=dict)
    # Each entry: {"text": ..., "category": ..., "score": ..., "lineup_metadata": ..., "drop_reason": ...}
    chunks: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def ingest_web_url(
    data_dir: Path,
    url: str,
    settings: "Settings",
    *,
    force: bool = False,
) -> WebIngestResult:
    """Scrape *url*, classify all chunks, and upsert into the LIVE collection.

    Lineup chunks are extracted with LLM metadata and stored as ``type=lineup``
    so they appear in ``valocoach lineup`` searches.  All other relevant chunks
    are stored as ``type=web``.  Off-topic chunks are dropped.

    Returns a :class:`WebIngestResult` the caller can use to display a summary.
    """
    from valocoach.retrieval.chunker import chunk_markdown
    from valocoach.retrieval.scrapers.web import scrape_url
    from valocoach.retrieval.youtube_ingest import (
        RELEVANCE_THRESHOLD,
        _keyword_lineup_boost,
        _should_keep_as_lineup,
        get_classifier,
    )

    # ── 1. Scrape ─────────────────────────────────────────────────────────
    try:
        content = scrape_url(url, source="web")
    except Exception as exc:
        log.warning("web_ingest: scrape failed for %s: %s", url, exc)
        return WebIngestResult(url=url, title="", domain=_domain(url), skipped_reason="scrape_failed")

    if content is None:
        log.warning("web_ingest: no content extracted from %s", url)
        return WebIngestResult(url=url, title="", domain=_domain(url), skipped_reason="scrape_failed")

    title = content.title or url
    domain = _domain(url)

    # ── 2. Chunk ──────────────────────────────────────────────────────────
    raw_chunks = chunk_markdown(content.text, source=url, max_tokens=300)
    # Filter out tiny heading/nav fragments
    raw_chunks = [c for c in raw_chunks if len(c.text.strip()) >= _MIN_CHUNK_CHARS]

    if not raw_chunks:
        return WebIngestResult(url=url, title=title, domain=domain, skipped_reason="no_chunks")

    # ── 3. Classify ───────────────────────────────────────────────────────
    classifier = get_classifier()
    dropped_counts: dict[str, int] = {"off_topic": 0, "low_score": 0, "unknown": 0}
    chunk_records: list[dict] = []

    for chunk in raw_chunks:
        text = chunk.text.strip()
        category, score = classifier.classify(text)
        drop_reason: str | None = None

        if category == "off_topic":
            drop_reason = "off_topic"
        elif category == "unknown":
            # Degraded embedder — keep the chunk rather than silently lose it
            drop_reason = None
        elif not _should_keep_as_lineup(category, score, text) and score < RELEVANCE_THRESHOLD:
            drop_reason = "low_score"

        # Keyword rescue: bump borderline chunks to lineup if keywords fire
        if drop_reason is None and category != "lineups":
            boost = _keyword_lineup_boost(text)
            if boost >= 0.7:
                category = "lineups"

        if drop_reason:
            dropped_counts[drop_reason] = dropped_counts.get(drop_reason, 0) + 1

        chunk_records.append({
            "text": text,
            "category": category,
            "score": score,
            "drop_reason": drop_reason,
            "lineup_metadata": None,
            "chunk_index": chunk.chunk_index,
        })

    kept = [r for r in chunk_records if r["drop_reason"] is None]
    lineup_records = [r for r in kept if r["category"] == "lineups"]
    web_records    = [r for r in kept if r["category"] != "lineups"]

    # ── 4. LLM metadata for lineup chunks ─────────────────────────────────
    if lineup_records:
        from valocoach.retrieval.lineups import extract_lineup_metadata

        for rec in lineup_records:
            try:
                rec["lineup_metadata"] = extract_lineup_metadata(
                    rec["text"], settings, video_title=title
                )
            except Exception as exc:
                log.debug("web_ingest: metadata extraction failed: %s", exc)
                rec["lineup_metadata"] = {}

    # ── 5. Upsert ─────────────────────────────────────────────────────────
    expires_at = int(time.time()) + _WEB_TTL_SECONDS
    url_hash = hashlib.sha1(url.encode()).hexdigest()[:12]
    total_kept = len(kept)

    lineup_upserted = _upsert_lineup_chunks(
        data_dir, lineup_records, url=url, title=title, domain=domain,
        url_hash=url_hash, expires_at=expires_at, total_kept=total_kept,
    )
    web_upserted = _upsert_web_chunks(
        data_dir, web_records, url=url, title=title, domain=domain,
        url_hash=url_hash, expires_at=expires_at, total_kept=total_kept,
    )

    log.info(
        "web_ingest: %s — %d lineup + %d web chunk(s) from %r",
        domain, lineup_upserted, web_upserted, title,
    )

    return WebIngestResult(
        url=url,
        title=title,
        domain=domain,
        fetched_count=len(raw_chunks),
        kept_count=total_kept,
        lineup_count=lineup_upserted,
        web_count=web_upserted,
        dropped_counts={k: v for k, v in dropped_counts.items() if v > 0},
        chunks=chunk_records,
    )


# ---------------------------------------------------------------------------
# Internal upsert helpers
# ---------------------------------------------------------------------------


def _upsert_lineup_chunks(
    data_dir: Path,
    records: list[dict],
    *,
    url: str,
    title: str,
    domain: str,
    url_hash: str,
    expires_at: int,
    total_kept: int,
) -> int:
    """Embed and upsert lineup-classified chunks as ``type=lineup``."""
    from valocoach.retrieval.embedder import embed_one
    from valocoach.retrieval.vector_store import LIVE_COLLECTION, get_collection

    upserted = 0
    for rec in records:
        meta_fields = rec.get("lineup_metadata") or {}
        doc_id = f"lineup:web:{url_hash}:{rec['chunk_index']}"
        metadata: dict = {
            "type": "lineup",
            "video_id": f"web:{url_hash}",
            "title": title,
            "channel": domain,
            "source": url,
            "start_seconds": 0,
            "expires_at_unix": expires_at,
            "expected_chunks": total_kept,
            **{k: v for k, v in meta_fields.items() if v is not None},
        }
        try:
            vec  = embed_one(rec["text"])
            coll = get_collection(data_dir, LIVE_COLLECTION)
            coll.upsert(ids=[doc_id], documents=[rec["text"]], embeddings=[vec], metadatas=[metadata])
            upserted += 1
            log.info(
                "web_ingest: lineup chunk %s (agent=%s map=%s)",
                doc_id, meta_fields.get("agent"), meta_fields.get("map"),
            )
        except Exception as exc:
            log.warning("web_ingest: failed to upsert lineup chunk %s: %s", doc_id, exc)

    return upserted


def _upsert_web_chunks(
    data_dir: Path,
    records: list[dict],
    *,
    url: str,
    title: str,
    domain: str,
    url_hash: str,
    expires_at: int,
    total_kept: int,
) -> int:
    """Embed and upsert non-lineup relevant chunks as ``type=web``."""
    from valocoach.retrieval.ingester import _upsert_batch
    from valocoach.retrieval.vector_store import LIVE_COLLECTION

    if not records:
        return 0

    texts     = [r["text"] for r in records]
    ids       = [f"web:{url_hash}:{r['chunk_index']}" for r in records]
    metadatas = [
        {
            "type": "web",
            "name": title,
            "source": url,
            "domain": domain,
            "category": r["category"],
            "anchor_score": round(r["score"], 4),
            "expires_at_unix": expires_at,
            "expected_chunks": total_kept,
        }
        for r in records
    ]
    return _upsert_batch(data_dir, texts, ids, metadatas, collection_name=LIVE_COLLECTION)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def _domain(url: str) -> str:
    """Extract a readable domain label from a URL."""
    import re as _re
    m = _re.search(r"https?://(?:www\.)?([^/]+)", url)
    return m.group(1) if m else url


__all__ = ["WebIngestResult", "ingest_web_url"]

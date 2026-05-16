"""YouTube ingest pipeline — Phase D (D2–D5).

Orchestrates deduplication, time-window chunking, anchor-based relevance
filtering, optional LLM summarisation, and vector-store upsert for a single
YouTube video.

Pipeline (called once per video):
  1. Validate / extract video ID
  2. Check ChromaDB LIVE collection for existing video_id    (D2 — dedup)
  3. Fetch time-windowed transcript chunks + oEmbed metadata (D3/D1)
  4. Classify each chunk via embedding-based anchor vectors  (D4)
  5. Drop ``off_topic`` chunks and those below RELEVANCE_THRESHOLD (D4)
  6. Optionally run LLM to produce a 1-sentence summary      (D5, opt-in)
  7. Upsert into LIVE collection with rich metadata

Entry point: :func:`ingest_youtube_video`.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from valocoach.core.config import Settings

log = logging.getLogger(__name__)

# D4 — minimum cosine similarity to retain a chunk regardless of category.
# 0.25 per §7.1 of the improvement plan — slightly aggressive on purpose;
# easier to lower than to undo once polluted chunks are in ChromaDB.
RELEVANCE_THRESHOLD = 0.25

# YouTube chunks stay live for 60 days (same TTL as other scraped content).
_YT_TTL_SECONDS = 60 * 24 * 3600


# ---------------------------------------------------------------------------
# D4 — Anchor-based relevance classifier
# ---------------------------------------------------------------------------

# Seven topic categories with representative phrases (§7.1 step 2).
# Each list is averaged into a single anchor embedding vector.
ANCHOR_PHRASES: dict[str, list[str]] = {
    "agent_strategy": [
        "which agents are strong this patch",
        "how to play this agent effectively",
        "agent kit abilities and usage tips",
        "agent role in team composition meta",
        "ability synergies combos this meta",
    ],
    "map_tactics": [
        "map specific strategies and site setups",
        "site execute and retake how to",
        "map control mid pressure positions",
        "attacking and defending site entry",
        "rotations through mid and connector timing",
    ],
    "lineups": [
        "utility lineup throw spot coordinates",
        "smoke placement from this corner position",
        "recon bolt lineup bounces off wall",
        "stand here aim there ability placement",
        "post-plant molotov incendiary lineup spot",
    ],
    "economy": [
        "round economy buy decisions credit management",
        "when to save or force buy this round",
        "pistol round economy bonus round strategy",
        "team economy spending and credit management",
        "gun upgrade or save decision making",
    ],
    "aim_mechanics": [
        "aim training crosshair placement headshots",
        "shooting mechanics recoil control spray pattern",
        "counter-strafing movement and accuracy",
        "flick aim tracking improvement drills",
        "headshot percentage accuracy practice",
    ],
    "game_sense": [
        "positioning rotations reading the round",
        "when to push aggressive or hold passive",
        "information gathering map awareness intel",
        "reading enemy patterns and tendencies",
        "late round clutch decisions game sense",
    ],
    "off_topic": [
        "like subscribe comment notification bell",
        "sponsor promotion advertisement discount code",
        "intro outro greeting farewell end screen",
        "merchandise giveaway social media follow",
        "twitch stream schedule highlights funny moments",
    ],
}


class AnchorClassifier:
    """Classifies text chunks by cosine similarity to per-category anchor vectors.

    Anchor vectors are computed lazily on first classification call by embedding
    all ANCHOR_PHRASES and averaging them per category.  If the embedder is
    unavailable (Ollama not running) the classifier degrades gracefully:
    ``classify()`` returns ``("unknown", 0.0)`` which the caller treats as
    *keep the chunk* rather than wrongly dropping it.
    """

    def __init__(self) -> None:
        self._anchor_vecs: dict[str, list[float]] | None = None

    def _build(self) -> None:
        import numpy as np

        from valocoach.retrieval.embedder import embed

        vecs: dict[str, list[float]] = {}
        for category, phrases in ANCHOR_PHRASES.items():
            embs = np.array(embed(phrases), dtype=float)  # (n_phrases, dim)
            avg = embs.mean(axis=0)
            norm = np.linalg.norm(avg)
            vecs[category] = (avg / norm).tolist() if norm > 0 else avg.tolist()

        self._anchor_vecs = vecs
        log.debug("AnchorClassifier: built %d anchor vectors", len(vecs))

    def classify(self, text: str) -> tuple[str, float]:
        """Return ``(category, cosine_score)`` for a text chunk.

        ``("unknown", 0.0)`` means the embedder was unavailable — treat as
        *keep* so we never silently drop chunks due to infrastructure failure.
        """
        if self._anchor_vecs is None:
            try:
                self._build()
            except Exception as exc:
                log.debug("AnchorClassifier build failed: %s — bypassing filter", exc)
                return ("unknown", 0.0)

        try:
            import numpy as np

            from valocoach.retrieval.embedder import embed_one

            vec = np.array(embed_one(text), dtype=float)
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm

            best_cat = "unknown"
            best_score = -1.0
            for category, anchor in self._anchor_vecs.items():
                score = float(np.dot(vec, np.array(anchor, dtype=float)))
                if score > best_score:
                    best_score = score
                    best_cat = category

            return (best_cat, best_score)

        except Exception as exc:
            log.debug("AnchorClassifier.classify failed: %s — keeping chunk", exc)
            return ("unknown", 0.0)


# Module-level singleton — built lazily, shared across all ingest calls in a
# process so anchor vectors are only embedded once.
_classifier: AnchorClassifier | None = None


def get_classifier() -> AnchorClassifier:
    global _classifier
    if _classifier is None:
        _classifier = AnchorClassifier()
    return _classifier


# ---------------------------------------------------------------------------
# D2 — Deduplication
# ---------------------------------------------------------------------------


def is_video_ingested(data_dir: Path, video_id: str) -> bool:
    """Return True if any chunk for *video_id* already exists in the LIVE collection.

    Queries by ``video_id`` metadata field so even a partial previous ingest
    (e.g. one that crashed mid-batch) registers as "already ingested" and is
    not re-processed unless ``force=True``.
    """
    try:
        from valocoach.retrieval.vector_store import LIVE_COLLECTION, get_collection

        coll = get_collection(data_dir, LIVE_COLLECTION)
        results = coll.get(
            where={"video_id": {"$eq": video_id}},
            limit=1,
            include=[],
        )
        return len(results.get("ids", [])) > 0
    except Exception as exc:
        # If the check itself fails, treat as not ingested so we don't silently
        # skip content due to a transient ChromaDB error.
        log.debug("Dedup check failed for %s: %s — treating as not ingested", video_id, exc)
        return False


# ---------------------------------------------------------------------------
# D5 — Optional LLM summarisation
# ---------------------------------------------------------------------------

_SUMMARISE_SYSTEM = (
    "You are a Valorant coaching assistant. "
    "Write a single, clear sentence capturing the key tactical teaching from "
    "the transcript excerpt below. Focus on specific mechanics, strategies, or "
    "setups — omit filler, channel promotion, and off-topic content. "
    "Output ONLY the sentence, no preamble."
)


def summarise_chunk(settings: Settings, text: str) -> str:
    """Run a single LLM pass to produce a 1-sentence tactical summary (D5).

    The summary replaces raw transcript as the embedding target, producing
    cleaner vector representations from messy auto-captions.  Falls back to
    the original *text* on any LLM failure so ingestion always has something
    to embed.
    """
    try:
        from valocoach.llm.provider import stream_completion

        tokens: list[str] = []
        for token in stream_completion(
            settings,
            system_prompt=_SUMMARISE_SYSTEM,
            user_message=text[:2_000],
        ):
            tokens.append(token)
        summary = "".join(tokens).strip()
        if summary:
            return summary
        log.debug("LLM returned empty summary — using original text")
    except Exception as exc:
        log.debug("LLM summarise failed: %s — using original text", exc)
    return text


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def ingest_youtube_video(
    data_dir: Path,
    url: str,
    settings: Settings,
    *,
    force: bool = False,
    summarize: bool = False,
) -> int:
    """Full Phase-D ingest pipeline for a single YouTube video.

    Args:
        data_dir:  Root data directory (ChromaDB lives under ``data_dir/chroma``).
        url:       YouTube URL or bare 11-char video ID.
        settings:  App settings (needed by D5 LLM summarisation).
        force:     Re-ingest even if the video is already in ChromaDB (D2).
        summarize: Run LLM summarisation on each retained chunk (D5).

    Returns:
        Number of chunks upserted (0 on skip or failure).
    """
    from valocoach.retrieval.ingester import _upsert_batch
    from valocoach.retrieval.scrapers.youtube import _extract_video_id, fetch_transcript_chunks
    from valocoach.retrieval.vector_store import LIVE_COLLECTION

    # Step 1: validate video ID
    try:
        video_id = _extract_video_id(url)
    except ValueError as exc:
        log.warning("YouTube ingest skipped — invalid URL/ID: %s", exc)
        return 0

    # D2: dedup check
    if not force and is_video_ingested(data_dir, video_id):
        log.info(
            "YouTube %s already ingested — skipping (pass force=True to re-ingest)", video_id
        )
        return 0

    # D3 + D1: time-windowed chunks with real title + channel
    chunks = fetch_transcript_chunks(url)
    if not chunks:
        log.warning("No transcript chunks retrieved for %s", video_id)
        return 0

    title = chunks[0].title
    channel = chunks[0].channel

    # D4: anchor-based relevance filter
    classifier = get_classifier()
    kept: list[tuple] = []
    dropped = 0

    for chunk in chunks:
        category, score = classifier.classify(chunk.text)
        is_off_topic = category == "off_topic"
        below_threshold = category != "unknown" and score < RELEVANCE_THRESHOLD

        if is_off_topic or below_threshold:
            dropped += 1
            log.debug(
                "D4 drop — start=%ds category=%s score=%.3f",
                chunk.start_seconds,
                category,
                score,
            )
            continue
        kept.append((chunk, category, score))

    log.info(
        "D4 anchor filter: %d kept / %d dropped from %s (%r)",
        len(kept),
        dropped,
        video_id,
        title,
    )

    if not kept:
        log.warning("All chunks filtered for %s — nothing to ingest", video_id)
        return 0

    # Build batch for upsert
    expires_at_unix = int(time.time()) + _YT_TTL_SECONDS
    texts: list[str] = []
    ids: list[str] = []
    metadatas: list[dict] = []

    for chunk, category, score in kept:
        # D5: optional LLM summarisation — summary becomes the embedding target
        embed_text = summarise_chunk(settings, chunk.text) if summarize else chunk.text

        # G2 — lineup chunks get an additional LLM metadata extraction pass.
        # The chunk is still stored in the LIVE collection with type=lineup
        # rather than type=youtube so the lineup retriever can filter on it.
        if category == "lineups":
            try:
                from valocoach.retrieval.lineups import ingest_lineup_chunk

                ingest_lineup_chunk(
                    data_dir,
                    embed_text,
                    video_id=video_id,
                    title=title,
                    channel=channel,
                    start_seconds=chunk.start_seconds,
                    url=chunk.url,
                    settings=settings,
                )
            except Exception as exc:
                log.warning("G2: lineup ingest failed for %s @ %ds: %s", video_id, chunk.start_seconds, exc)
            # Don't also add to the standard YouTube batch (avoid duplicate)
            continue

        texts.append(embed_text)
        ids.append(f"youtube:{video_id}:{chunk.start_seconds}")
        metadatas.append(
            {
                "type": "youtube",
                "video_id": video_id,
                "title": title,
                "channel": channel,
                "source": chunk.url,
                "start_seconds": chunk.start_seconds,
                "category": category,
                "anchor_score": round(score, 4),
                "summarized": summarize,
                # Always store raw text so D6 citations can display it
                "raw_text": chunk.text[:500],
                "expires_at_unix": expires_at_unix,
            }
        )

    n = _upsert_batch(data_dir, texts, ids, metadatas, collection_name=LIVE_COLLECTION)
    log.info(
        "Ingested %d YouTube chunk(s) from %r (channel=%r, force=%s, summarize=%s)",
        n,
        title,
        channel,
        force,
        summarize,
    )
    return n


__all__ = [
    "ANCHOR_PHRASES",
    "AnchorClassifier",
    "RELEVANCE_THRESHOLD",
    "get_classifier",
    "ingest_youtube_video",
    "is_video_ingested",
    "summarise_chunk",
]

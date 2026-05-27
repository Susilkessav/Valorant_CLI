"""YouTube ingest pipeline -- Phase D (D2-D5).

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

Two-phase public API:
  - :func:`analyze_youtube_video`  — classify + extract metadata, no writes
  - :func:`apply_youtube_ingest_plan` — upsert the pre-analysed plan
  - :func:`ingest_youtube_video`   — backward-compat wrapper (analyse + apply)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from valocoach.core.config import Settings

log = logging.getLogger(__name__)

# D4 — minimum cosine similarity to retain a chunk regardless of category.
# 0.20 gives borderline lineup chunks a chance; keyword boost provides a
# second gate so lowering this doesn't flood the store with noise.
RELEVANCE_THRESHOLD = 0.20

# YouTube chunks stay live for 60 days (same TTL as other scraped content).
_YT_TTL_SECONDS = 60 * 24 * 3600


# ---------------------------------------------------------------------------
# Result dataclasses (two-phase analyse → apply API)
# ---------------------------------------------------------------------------


@dataclass
class CandidateChunk:
    """One time-windowed transcript chunk after classification (and optional LLM extraction)."""

    start_seconds: int
    text: str
    category: str  # e.g. "lineups", "map_tactics", "off_topic"
    score: float  # cosine similarity to best anchor
    lineup_metadata: dict | None = None
    # None = kept.  Set to one of "off_topic" | "low_score" | "unknown" when dropped.
    drop_reason: str | None = None


@dataclass
class YouTubeIngestPlan:
    """Everything analyze_youtube_video() learned, without writing anything to ChromaDB."""

    video_id: str
    title: str
    channel: str
    already_ingested: bool = False
    # Set when the plan cannot proceed: "invalid_url" | "already_ingested" | "no_transcript" | "ip_blocked" | "no_captions" | "no_language"
    skipped_reason: str | None = None
    fetched_count: int = 0
    kept_count: int = 0
    lineup_count: int = 0
    youtube_count: int = 0
    # {"off_topic": N, "low_score": N, "unknown": N}
    dropped_counts: dict[str, int] = field(default_factory=dict)
    candidates: list[CandidateChunk] = field(default_factory=list)


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
        # Natural speech patterns found in real lineup guides:
        "stand here and throw the lineup right here",
        "aim at the box on the roof bounces off",
        "from this position throw your dart or smoke",
        "this lineup reveals the entire site defenders",
        "for this smoke stand on the ledge and aim",
        "one bounce lineup lands perfectly on site",
        "throw from here and it will land right there",
        "this is where you stand for the lineup spot",
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
# D4b — Deterministic keyword boost for lineup detection
# ---------------------------------------------------------------------------

# Tactical words that are strong evidence of lineup content.  These are
# checked after the embedding classifier — a chunk that scores just below
# RELEVANCE_THRESHOLD but contains several of these keywords gets a second
# chance rather than being silently dropped.
_LINEUP_KEYWORDS: frozenset[str] = frozenset(
    {
        "lineup",
        "lineups",
        "stand here",
        "aim here",
        "aim at",
        "aim there",
        "one bounce",
        "two bounce",
        "post plant",
        "post-plant",
        "molly",
        "molotov",
        "incendiary",
        "dart",
        "recon bolt",
        "shock bolt",
        "owl drone",
        "seeker",
        "alarmbot",
        "killjoy",
        "cypher",
        "chamber",
        "smoke",
        "wall",
        "barrier",
        "turret",
        "tripwire",
        "trap",
        "bounce",
        "bounces",
        "throw from",
        "land on",
        "lands on",
        "right here",
        "stand on",
        "position",
        "ledge",
        "corner",
        "pre plant",
        "retake",
        "site clear",
        "revealing",
        "reveal",
        "scan",
    }
)

# Agent names — presence strongly suggests tactical lineup content
_AGENT_NAMES: frozenset[str] = frozenset(
    {
        "sova",
        "viper",
        "brimstone",
        "omen",
        "astra",
        "harbor",
        "killjoy",
        "cypher",
        "sage",
        "chamber",
        "deadlock",
        "vyse",
        "breach",
        "skye",
        "fade",
        "gekko",
        "kayo",
        "kay/o",
        "jett",
        "phoenix",
        "reyna",
        "raze",
        "neon",
        "yoru",
        "iso",
        "clove",
        "tejo",
        "waylay",
    }
)


def _keyword_lineup_boost(text: str) -> float:
    """Return 0.0-1.0 based on tactical keyword density.

    3 or more keyword hits → score of 1.0 (full boost).  Used as a safety
    net for chunks that embed slightly below RELEVANCE_THRESHOLD but are
    clearly about lineups based on word choice.
    """
    text_lower = text.lower()
    hits = sum(1 for kw in _LINEUP_KEYWORDS if kw in text_lower)
    hits += sum(1 for agent in _AGENT_NAMES if agent in text_lower)
    return min(hits / 3.0, 1.0)


def _should_keep_as_lineup(category: str, score: float, text: str) -> bool:
    """Return True when a chunk should be treated as a lineup candidate.

    A chunk qualifies if:
    - The embedding classifier said "lineups" and the score is above threshold, OR
    - The keyword boost is strong (≥ 0.7), regardless of embedding score — this
      rescues chunks whose signal was diluted by noise that survived cleaning.
    """
    if category == "off_topic" or category == "unknown":
        return False
    if category == "lineups" and score >= RELEVANCE_THRESHOLD:
        return True
    boost = _keyword_lineup_boost(text)
    return boost >= 0.7


# ---------------------------------------------------------------------------
# D2 — Deduplication
# ---------------------------------------------------------------------------


def is_video_ingested(data_dir: Path, video_id: str) -> bool:
    """Return True iff *video_id* was ingested to completion.

    The completeness check compares the count of chunks currently in the
    LIVE collection for *video_id* against ``expected_chunks`` stamped on
    each chunk at write time.  This lets a re-run recover from a partial
    crash (50 of 200 chunks made it before the process died) by
    re-ingesting the rest instead of treating the half-finished video as
    "done" forever.

    Backwards compatibility: chunks ingested before ``expected_chunks``
    was added carry no field at all.  We treat those as complete to avoid
    re-scraping every legacy video on the next run.
    """
    try:
        from valocoach.retrieval.vector_store import LIVE_COLLECTION, get_collection

        coll = get_collection(data_dir, LIVE_COLLECTION)
        results = coll.get(
            where={"video_id": {"$eq": video_id}},
            include=["metadatas"],
        )
        ids = results.get("ids", []) or []
        if not ids:
            return False

        # Any chunk's metadata tells us how many were expected — they all
        # carry the same ``expected_chunks`` value.
        metas = results.get("metadatas", []) or []
        expected: int | None = None
        for meta in metas:
            if isinstance(meta, dict) and "expected_chunks" in meta:
                expected = int(meta["expected_chunks"])
                break

        if expected is None:
            # Legacy chunk with no completeness stamp — treat as complete.
            return True

        complete = len(ids) >= expected
        if not complete:
            log.warning(
                "YouTube %s ingest is partial (%d/%d chunks) — re-ingesting.",
                video_id,
                len(ids),
                expected,
            )
        return complete
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
# Public API — two-phase: analyse (no writes) then apply (upsert)
# ---------------------------------------------------------------------------


def analyze_youtube_video(
    data_dir: Path,
    url: str,
    settings: Settings,
    *,
    force: bool = False,
) -> YouTubeIngestPlan:
    """Classify a YouTube video and extract lineup metadata without writing to ChromaDB.

    Runs steps 1-4 of the full pipeline (validate, dedup check, transcript
    fetch, anchor classification) plus LLM metadata extraction for any chunk
    classified as ``lineups``.  Nothing is written to ChromaDB.

    Returns a :class:`YouTubeIngestPlan` the caller can inspect and then pass
    to :func:`apply_youtube_ingest_plan` to actually persist.
    """
    from valocoach.retrieval.scrapers.youtube import _extract_video_id, fetch_transcript_chunks

    # Step 1: validate
    try:
        video_id = _extract_video_id(url)
    except ValueError as exc:
        log.warning("YouTube analyze skipped — invalid URL/ID: %s", exc)
        return YouTubeIngestPlan(
            video_id="",
            title="",
            channel="",
            skipped_reason="invalid_url",
        )

    # D2: dedup check
    already = is_video_ingested(data_dir, video_id)
    if already and not force:
        log.info(
            "YouTube %s already ingested — skipping analyze (pass force=True to re-analyse)",
            video_id,
        )
        return YouTubeIngestPlan(
            video_id=video_id,
            title="",
            channel="",
            already_ingested=True,
            skipped_reason="already_ingested",
        )

    # D3 + D1: fetch chunks
    chunks, fail_reason = fetch_transcript_chunks(url)
    if not chunks:
        skipped = fail_reason if fail_reason else "no_transcript"
        if fail_reason == "ip_blocked":
            log.warning("YouTube transcript blocked for %s — IP rate-limited", video_id)
        else:
            log.warning("No transcript chunks retrieved for %s (%s)", video_id, skipped)
        return YouTubeIngestPlan(
            video_id=video_id,
            title="",
            channel="",
            already_ingested=already,
            skipped_reason=skipped,
        )

    title = chunks[0].title
    channel = chunks[0].channel

    # D4: classify — embedding anchor + keyword boost rescue
    classifier = get_classifier()
    candidates: list[CandidateChunk] = []
    dropped_counts: dict[str, int] = {"off_topic": 0, "low_score": 0, "unknown": 0}

    for chunk in chunks:
        category, score = classifier.classify(chunk.text)
        drop_reason: str | None = None

        if category == "off_topic":
            drop_reason = "off_topic"
        elif category == "unknown":
            drop_reason = "unknown"
        elif (
            not _should_keep_as_lineup(category, score, chunk.text) and score < RELEVANCE_THRESHOLD
        ):
            # Neither the embedding threshold nor the keyword boost saved it
            drop_reason = "low_score"

        # Keyword-rescued lineup: force category to "lineups" so the G2 path runs
        if drop_reason is None and category != "lineups":
            boost = _keyword_lineup_boost(chunk.text)
            if boost >= 0.7:
                category = "lineups"

        if drop_reason:
            dropped_counts[drop_reason] = dropped_counts.get(drop_reason, 0) + 1
            candidates.append(
                CandidateChunk(
                    start_seconds=chunk.start_seconds,
                    text=chunk.text,
                    category=category,
                    score=score,
                    drop_reason=drop_reason,
                )
            )
        else:
            candidates.append(
                CandidateChunk(
                    start_seconds=chunk.start_seconds,
                    text=chunk.text,
                    category=category,
                    score=score,
                )
            )

    kept = [c for c in candidates if c.drop_reason is None]
    lineup_chunks = [c for c in kept if c.category == "lineups"]
    youtube_chunks = [c for c in kept if c.category != "lineups"]

    # G2: LLM metadata extraction for lineup chunks (preview shows what will be stored)
    if lineup_chunks:
        from valocoach.retrieval.lineups import extract_lineup_metadata

        for chunk in lineup_chunks:
            try:
                chunk.lineup_metadata = extract_lineup_metadata(
                    chunk.text, settings, video_title=title
                )
            except Exception as exc:
                log.debug(
                    "G2: metadata extraction failed for %s@%ds: %s",
                    video_id,
                    chunk.start_seconds,
                    exc,
                )
                chunk.lineup_metadata = {}

    if dropped_counts.get("unknown", 0):
        log.warning(
            "D4: %d/%d chunks classifier returned 'unknown' for %s — "
            "embedding service may be degraded.",
            dropped_counts["unknown"],
            len(chunks),
            video_id,
        )

    return YouTubeIngestPlan(
        video_id=video_id,
        title=title,
        channel=channel,
        already_ingested=already,
        fetched_count=len(chunks),
        kept_count=len(kept),
        lineup_count=len(lineup_chunks),
        youtube_count=len(youtube_chunks),
        dropped_counts={k: v for k, v in dropped_counts.items() if v > 0},
        candidates=candidates,
    )


def apply_youtube_ingest_plan(
    plan: YouTubeIngestPlan,
    data_dir: Path,
    settings: Settings,
    *,
    summarize: bool = False,
) -> int:
    """Upsert a pre-analysed :class:`YouTubeIngestPlan` into ChromaDB.

    Uses the ``lineup_metadata`` already extracted during :func:`analyze_youtube_video`
    so no LLM calls are repeated.  Returns the total number of chunks stored
    (lineup + regular YouTube).
    """
    from valocoach.retrieval.embedder import embed_one
    from valocoach.retrieval.ingester import _upsert_batch
    from valocoach.retrieval.vector_store import LIVE_COLLECTION, get_collection

    if plan.skipped_reason:
        return 0

    kept = [c for c in plan.candidates if c.drop_reason is None]
    if not kept:
        return 0

    expires_at_unix = int(time.time()) + _YT_TTL_SECONDS
    total_kept = len(kept)

    texts: list[str] = []
    ids: list[str] = []
    metadatas: list[dict] = []
    lineup_count = 0

    for chunk in kept:
        embed_text = summarise_chunk(settings, chunk.text) if summarize else chunk.text

        if chunk.category == "lineups":
            # Use pre-extracted metadata — avoids a second LLM round-trip
            meta_fields = chunk.lineup_metadata or {}
            doc_id = f"lineup:{plan.video_id}:{chunk.start_seconds}"
            metadata: dict = {
                "type": "lineup",
                "video_id": plan.video_id,
                "title": plan.title,
                "channel": plan.channel,
                "start_seconds": chunk.start_seconds,
                "source": f"https://www.youtube.com/watch?v={plan.video_id}",
                "expires_at_unix": expires_at_unix,
                "expected_chunks": total_kept,
                **{k: v for k, v in meta_fields.items() if v is not None},
            }
            try:
                vec = embed_one(embed_text)
                coll = get_collection(data_dir, LIVE_COLLECTION)
                coll.upsert(
                    ids=[doc_id], documents=[embed_text], embeddings=[vec], metadatas=[metadata]
                )
                lineup_count += 1
                log.info(
                    "G1: upserted lineup chunk %s (agent=%s map=%s)",
                    doc_id,
                    meta_fields.get("agent"),
                    meta_fields.get("map"),
                )
            except Exception as exc:
                log.warning("G1: failed to upsert lineup chunk %s: %s", doc_id, exc)
            continue

        texts.append(embed_text)
        ids.append(f"youtube:{plan.video_id}:{chunk.start_seconds}")
        metadatas.append(
            {
                "type": "youtube",
                "video_id": plan.video_id,
                "title": plan.title,
                "channel": plan.channel,
                "source": f"https://www.youtube.com/watch?v={plan.video_id}",
                "start_seconds": chunk.start_seconds,
                "category": chunk.category,
                "anchor_score": round(chunk.score, 4),
                "summarized": summarize,
                "raw_text": chunk.text[:500],
                "expires_at_unix": expires_at_unix,
                "expected_chunks": total_kept,
            }
        )

    n = _upsert_batch(data_dir, texts, ids, metadatas, collection_name=LIVE_COLLECTION)
    total = n + lineup_count
    log.info(
        "Ingested %d YouTube + %d lineup chunk(s) from %r (channel=%r)",
        n,
        lineup_count,
        plan.title,
        plan.channel,
    )
    return total


# ---------------------------------------------------------------------------
# Backward-compat wrapper
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

    Backward-compatible wrapper around :func:`analyze_youtube_video` +
    :func:`apply_youtube_ingest_plan`.

    Returns:
        Number of chunks upserted (0 on skip or failure).
    """
    plan = analyze_youtube_video(data_dir, url, settings, force=force)
    if plan.skipped_reason:
        if plan.skipped_reason == "already_ingested":
            log.info(
                "YouTube %s already ingested — skipping (pass force=True to re-ingest)",
                plan.video_id,
            )
        elif plan.skipped_reason == "no_transcript":
            log.warning("No transcript chunks retrieved for %s", plan.video_id)
        elif plan.skipped_reason == "invalid_url":
            log.warning("YouTube ingest skipped — invalid URL/ID: %s", url)
        return 0

    if plan.kept_count == 0:
        log.warning("All chunks filtered for %s — nothing to ingest", plan.video_id)
        return 0

    return apply_youtube_ingest_plan(plan, data_dir, settings, summarize=summarize)


__all__ = [
    "ANCHOR_PHRASES",
    "RELEVANCE_THRESHOLD",
    "AnchorClassifier",
    "CandidateChunk",
    "YouTubeIngestPlan",
    "analyze_youtube_video",
    "apply_youtube_ingest_plan",
    "get_classifier",
    "ingest_youtube_video",
    "is_video_ingested",
    "summarise_chunk",
]

"""G1 + G2 + G4/G5 — Lineup retrieval and ingest.

Lineups live in the LIVE ChromaDB collection with the §8.2 metadata schema:

    {
        "type": "lineup",
        "agent": "Sova",
        "ability": "Recon Bolt",
        "map": "Ascent",
        "site": "A",
        "side": "attack",
        "purpose": "post-plant deny|pre-round info|site clear|retake",
        "video_id": "...",
        "title": "...",
        "channel": "...",
        "start_seconds": 234,
    }

Two entry points:
  - ``ingest_lineup_chunk()``:  called by the YouTube pipeline when a chunk is
    classified ``lineups``; runs an LLM extraction pass to fill the metadata.
  - ``ingest_seed_lineups()``:  seeds the collection from ``data/lineups_seed.json``.
  - ``search_lineups()``:       metadata-filtered + similarity-ranked retrieval
    for the ``valocoach lineup`` command.

G2 — LLM metadata extraction
------------------------------
When the YouTube pipeline classifies a chunk as ``lineups``, it calls
``extract_lineup_metadata(text, settings)`` which asks the LLM to return a
JSON object with the structured fields.  Missing fields default to None —
retrieval filters only on fields that are present.  The extraction is
best-effort: if the LLM fails or the JSON is unparseable, the chunk is still
ingested with ``type=lineup`` and all metadata fields set to None, so it
remains searchable by text similarity.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# ChromaDB LIVE collection is the home for lineup chunks
# (same TTL mechanism as YouTube transcript chunks).
_COLLECTION = "valocoach_live"

# Seed file path
_SEED_FILE = Path(__file__).parent / "data" / "lineups_seed.json"


# ---------------------------------------------------------------------------
# G2 — LLM metadata extraction
# ---------------------------------------------------------------------------

_EXTRACTION_SYSTEM = """\
You are a Valorant lineup metadata extractor.
Given a transcript excerpt describing a lineup or utility throw, extract structured metadata.
Return ONLY a valid JSON object with these fields (use null for anything not mentioned):
{
  "agent": "<agent name or null>",
  "ability": "<ability name or null>",
  "map": "<map name or null>",
  "site": "<A|B|C|Mid|null>",
  "side": "<attack|defense|null>",
  "purpose": "<post-plant deny|pre-round info|site clear|retake|null>"
}
Do not include any other text. Only output the JSON object."""


def extract_lineup_metadata(text: str, settings: Any) -> dict:
    """Ask the LLM to extract structured lineup metadata from a transcript chunk.

    Returns a dict with keys: agent, ability, map, site, side, purpose.
    All values may be None if the LLM cannot determine them.
    Never raises — returns all-None dict on any failure.
    """
    defaults: dict[str, str | None] = {
        "agent": None,
        "ability": None,
        "map": None,
        "site": None,
        "side": None,
        "purpose": None,
    }

    try:
        from valocoach.llm.provider import call_llm

        prompt = (
            f"Transcript excerpt:\n\"\"\"\n{text[:800]}\n\"\"\"\n\n"
            "Extract lineup metadata as JSON."
        )
        raw = call_llm(
            system=_EXTRACTION_SYSTEM,
            user=prompt,
            settings=settings,
            max_tokens=256,
        )
        if not raw:
            return defaults

        # Extract JSON from response (LLM sometimes wraps in markdown)
        m = re.search(r"\{[^{}]+\}", raw, re.DOTALL)
        if not m:
            return defaults

        parsed = json.loads(m.group())
        return {k: parsed.get(k) or None for k in defaults}

    except Exception as exc:
        log.debug("G2: lineup metadata extraction failed: %s", exc)
        return defaults


# ---------------------------------------------------------------------------
# G1 — Ingest a single lineup chunk (called by YouTube pipeline)
# ---------------------------------------------------------------------------


def ingest_lineup_chunk(
    data_dir: Path,
    text: str,
    *,
    video_id: str,
    title: str,
    channel: str,
    start_seconds: int,
    url: str,
    settings: Any,
) -> None:
    """Embed and upsert one lineup chunk into the LIVE collection.

    Runs the LLM metadata extraction pass (G2) then stores the chunk with
    full §8.2 metadata.  Extraction failure is non-blocking.
    """
    from valocoach.retrieval.embedder import embed_one
    from valocoach.retrieval.vector_store import get_collection

    meta_fields = extract_lineup_metadata(text, settings)

    doc_id = f"lineup:{video_id}:{start_seconds}"
    metadata: dict[str, Any] = {
        "type": "lineup",
        "video_id": video_id,
        "title": title,
        "channel": channel,
        "start_seconds": start_seconds,
        "source": url,
        **{k: v for k, v in meta_fields.items() if v is not None},
    }

    try:
        vec = embed_one(text)
        coll = get_collection(data_dir, _COLLECTION)
        coll.upsert(ids=[doc_id], documents=[text], embeddings=[vec], metadatas=[metadata])
        log.info("G1: upserted lineup chunk %s (agent=%s map=%s)", doc_id, meta_fields.get("agent"), meta_fields.get("map"))
    except Exception as exc:
        log.warning("G1: failed to ingest lineup chunk %s: %s", doc_id, exc)


# ---------------------------------------------------------------------------
# G5 — Seed ingest
# ---------------------------------------------------------------------------


def ingest_seed_lineups(data_dir: Path) -> int:
    """Upsert all entries from lineups_seed.json into the LIVE collection.

    Called by ``valocoach ingest --seed``.  Returns number of entries ingested.
    """
    if not _SEED_FILE.exists():
        log.warning("G5: lineups_seed.json not found at %s", _SEED_FILE)
        return 0

    with open(_SEED_FILE) as f:
        data = json.load(f)

    entries = data.get("lineups", [])
    if not entries:
        return 0

    from valocoach.retrieval.embedder import embed
    from valocoach.retrieval.vector_store import get_collection

    coll = get_collection(data_dir, _COLLECTION)
    texts: list[str] = []
    ids: list[str] = []
    metas: list[dict] = []

    for i, entry in enumerate(entries):
        text = entry.get("text", "")
        if not text.strip():
            continue
        doc_id = f"lineup:seed:{i}"
        meta: dict[str, Any] = {
            "type": "lineup",
            "video_id": entry.get("video_id") or "seed",
            "title": entry.get("title", "Seed lineup"),
            "channel": entry.get("channel", "seed"),
            "start_seconds": 0,
            "source": entry.get("source", "lineups_seed"),
        }
        # Add optional structured fields only if present
        for field in ("agent", "ability", "map", "site", "side", "purpose"):
            if entry.get(field):
                meta[field] = entry[field]
        texts.append(text)
        ids.append(doc_id)
        metas.append(meta)

    if not texts:
        return 0

    try:
        vecs = embed(texts)
        coll.upsert(ids=ids, documents=texts, embeddings=vecs, metadatas=metas)
        log.info("G5: seeded %d lineup entries into %s", len(texts), _COLLECTION)
        return len(texts)
    except Exception as exc:
        log.error("G5: seed ingest failed: %s", exc)
        return 0


# ---------------------------------------------------------------------------
# G4 — Lineup retrieval
# ---------------------------------------------------------------------------


def search_lineups(
    data_dir: Path,
    query: str,
    *,
    agent: str | None = None,
    map_name: str | None = None,
    site: str | None = None,
    n_results: int = 5,
) -> list[dict]:
    """Metadata-filtered + similarity-ranked lineup retrieval.

    Builds a ChromaDB ``where`` clause from the provided filters, then
    runs a similarity search limited to ``type=lineup`` documents.  Returns
    a list of hit dicts with keys: text, metadata, distance.

    Args:
        data_dir:  ChromaDB data directory.
        query:     Natural language query (e.g. "Sova bolt Ascent A site").
        agent:     Filter by agent name (case-insensitive substring match not
                   supported by ChromaDB — exact match only; callers should
                   pass the canonical agent name).
        map_name:  Filter by map name.
        site:      Filter by site letter (A, B, C).
        n_results: Maximum hits to return.

    Returns:
        List of dicts, each with:
          - ``text``: chunk text
          - ``metadata``: ChromaDB metadata dict
          - ``distance``: cosine distance (lower = more similar)
    """
    from valocoach.retrieval.embedder import embed_one
    from valocoach.retrieval.vector_store import get_collection

    # Build the where clause — ChromaDB requires $and for multiple conditions.
    conditions: list[dict] = [{"type": {"$eq": "lineup"}}]
    if agent:
        conditions.append({"agent": {"$eq": agent}})
    if map_name:
        conditions.append({"map": {"$eq": map_name}})
    if site:
        conditions.append({"site": {"$eq": site.upper()}})

    where: dict | None = None
    if len(conditions) == 1:
        where = conditions[0]
    elif len(conditions) > 1:
        where = {"$and": conditions}

    try:
        vec = embed_one(query)
        coll = get_collection(data_dir, _COLLECTION)
        results = coll.query(
            query_embeddings=[vec],
            n_results=n_results,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
    except Exception as exc:
        log.warning("G4: lineup search failed: %s", exc)
        return []

    hits: list[dict] = []
    docs = (results.get("documents") or [[]])[0]
    metas = (results.get("metadatas") or [[]])[0]
    dists = (results.get("distances") or [[]])[0]

    for text, meta, dist in zip(docs, metas, dists):
        if dist <= 0.55:  # generous threshold — lineup queries are specific
            hits.append({"text": text, "metadata": meta, "distance": dist})

    return hits


def format_lineup_results(hits: list[dict]) -> str:
    """Format lineup search results for terminal display."""
    if not hits:
        return "No lineup matches found for those filters."

    lines: list[str] = []
    for i, hit in enumerate(hits, 1):
        meta = hit["metadata"]
        agent = meta.get("agent", "?")
        ability = meta.get("ability", "ability")
        map_name = meta.get("map", "?")
        site = meta.get("site")
        purpose = meta.get("purpose")
        channel = meta.get("channel", "?")
        title = meta.get("title", "?")
        start = int(meta.get("start_seconds", 0))
        mins, secs = divmod(start, 60)

        header_parts = [f"{agent} — {ability}"]
        if map_name and map_name != "?":
            header_parts.append(f"{map_name}")
        if site:
            header_parts.append(f"{site} site")
        if purpose:
            header_parts.append(f"({purpose})")

        lines.append(f"{i}. {' · '.join(header_parts)}")
        lines.append(f"   {hit['text']}")

        if channel != "seed":
            lines.append(f"   📹 {channel} \"{title}\" @ {mins}:{secs:02d}")
        lines.append("")

    return "\n".join(lines).rstrip()

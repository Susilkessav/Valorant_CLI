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
import time
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# ChromaDB LIVE collection is the home for lineup chunks
# (same TTL mechanism as YouTube transcript chunks).
_COLLECTION = "valocoach_live"

# Seed file path
_SEED_FILE = Path(__file__).parent / "data" / "lineups_seed.json"

# Lineups live as long as YouTube chunks — 60 days.  Keeps the LIVE
# collection bounded and matches the YouTube TTL so a single purge sweep
# handles both.
_LINEUP_TTL_SECONDS = 60 * 24 * 3600

# ---------------------------------------------------------------------------
# Canonical-case normalisation
# ---------------------------------------------------------------------------
# ChromaDB $eq filters are case-sensitive; LLM extractions return arbitrary
# casing ("sova", "SOVA", "Sova").  We canonicalise both at write time and
# at query time so filtered retrieval actually finds chunks.

_CANONICAL_AGENTS: dict[str, str] = {
    a.lower(): a
    for a in (
        "Astra",
        "Breach",
        "Brimstone",
        "Chamber",
        "Clove",
        "Cypher",
        "Deadlock",
        "Fade",
        "Gekko",
        "Harbor",
        "Iso",
        "Jett",
        "KAY/O",
        "Killjoy",
        "Neon",
        "Omen",
        "Phoenix",
        "Raze",
        "Reyna",
        "Sage",
        "Skye",
        "Sova",
        "Tejo",
        "Viper",
        "Vyse",
        "Waylay",
        "Yoru",
    )
}
_CANONICAL_MAPS: dict[str, str] = {
    m.lower(): m
    for m in (
        "Ascent",
        "Bind",
        "Breeze",
        "Fracture",
        "Haven",
        "Icebox",
        "Lotus",
        "Pearl",
        "Split",
        "Sunset",
        "Abyss",
        "Corrode",
    )
}
_CANONICAL_SITES = {"a": "A", "b": "B", "c": "C", "d": "D", "mid": "Mid"}

# Primary lineup ability per agent — used as a fallback when the LLM returns null.
# These are the abilities most commonly the subject of YouTube lineup guides.
_DEFAULT_LINEUP_ABILITIES: dict[str, str] = {
    "Sova": "Recon Bolt",
    "Viper": "Snake Bite",
    "Brimstone": "Incendiary",
    "KAY/O": "ZERO/point",
    "Omen": "Dark Cover",
    "Astra": "Gravity Well",
    "Harbor": "High Tide",
    "Gekko": "Wingman",
    "Breach": "Aftershock",
    "Fade": "Seize",
    "Deadlock": "Sonic Sensor",
    "Killjoy": "Nanoswarm",
    "Cypher": "Trapwire",
    "Chamber": "Trademark",
}

# Keyword hints for purpose detection — checked in order, first match wins.
_PURPOSE_HINTS: list[tuple[list[str], str]] = [
    (["post plant", "post-plant", "spike planted", "after plant"], "post-plant deny"),
    (["retake", "take back", "reclaim"], "retake"),
    (["site clear", "clearing site", "check site", "clear the site"], "site clear"),
    (
        ["revealing", "reveal", "scanning ahead", "scan", "pre round", "pre-round", "before round"],
        "pre-round info",
    ),
]

# Site keyword hints — checked in order per word-boundary patterns.
# Map-specific entries first so they don't get eclipsed by generic ones.
_SITE_HINTS: list[tuple[list[str], str]] = [
    # Haven D-short (Haven-unique)
    (["d site", "d short", "long hall", "d main"], "D"),
    # A site variants
    (
        [
            "a site",
            "a short",
            "a main",
            "a heaven",
            "a link",
            "a lobby",
            "a bath",
            "a court",
            "onto a",
        ],
        "A",
    ),
    # B site variants
    (["b site", "b main", "b long", "b box", "b hall", "b lobby", "onto b"], "B"),
    # C site variants (Haven, Lotus, Fracture)
    (["c site", "c long", "c short", "c lobby", "c link", "onto c"], "C"),
    # Mid — last so specific sites take priority
    (["mid site", "mid-lane", " mid ", "through mid", "from mid"], "Mid"),
]


def _canon_agent(value: str | None) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    # Strip spaces + try direct lookup; if the LLM wrote "kay-o" or "kay o"
    # nothing matches and we return the title-cased value as a best effort.
    key = value.strip().lower().replace("-", "/").replace(" ", "")
    # Re-key the canonical table without the / to allow "kayo" → "KAY/O"
    flat = {k.replace("/", ""): v for k, v in _CANONICAL_AGENTS.items()}
    return flat.get(key) or _CANONICAL_AGENTS.get(value.strip().lower()) or value.strip().title()


def _canon_map(value: str | None) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    return _CANONICAL_MAPS.get(value.strip().lower()) or value.strip().title()


def _canon_site(value: str | None) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    return _CANONICAL_SITES.get(value.strip().lower()) or value.strip().upper()


# ---------------------------------------------------------------------------
# G2 — LLM metadata extraction
# ---------------------------------------------------------------------------

_EXTRACTION_SYSTEM = """\
You are a Valorant lineup metadata extractor.
Given a transcript excerpt describing a lineup or utility throw, extract structured metadata.

The video title and channel are provided as context — use them to fill in fields the
transcript does not explicitly state (e.g. if the title says "Sova Haven Defense" you
can infer agent=Sova, map=Haven, side=defense even if the speaker never says them aloud).

ABILITY NAMES — use these exact strings:
- Sova: "Recon Bolt" (dart that bounces + reveals; audio cue "scanning ahead"), "Shock Dart", "Owl Drone", "Hunter's Fury"
- Viper: "Snake Bite" (molotov), "Poison Cloud" (smoke orb), "Toxic Screen" (wall), "Viper's Pit"
- Brimstone: "Incendiary" (molotov), "Sky Smoke", "Stim Beacon", "Orbital Strike"
- Omen: "Dark Cover" (smoke), "Shrouded Step", "Paranoia", "From the Shadows"
- KAY/O: "ZERO/point" (knife), "FLASH/drive", "FRAG/ment" (molly), "NULL/cmd"
- Killjoy: "Nanoswarm" (molly), "Alarmbot", "Turret", "Lockdown"
- Cypher: "Trapwire", "Cyber Cage", "Spycam", "Neural Theft"
- Breach: "Aftershock", "Flashpoint", "Fault Line", "Rolling Thunder"
- Fade: "Seize", "Haunt", "Prowler", "Nightfall"
- Astra: "Gravity Well", "Nova Pulse", "Nebula", "Cosmic Divide"
- Harbor: "Cove", "High Tide", "Cascade", "Reckoning"
- Gekko: "Wingman", "Dizzy", "Mosh Pit", "Thrash"

SITE NAMES — use only: "A", "B", "C", "D", "Mid" (Haven has A, B, C bomb sites plus D short area)

Return ONLY a valid JSON object with these fields (use null for anything not determined):
{
  "agent": "<exact Valorant agent name or null>",
  "ability": "<exact ability name or null>",
  "map": "<exact map name or null>",
  "site": "<A|B|C|D|Mid|null>",
  "side": "<attack|defense|null>",
  "purpose": "<post-plant deny|pre-round info|site clear|retake|null>"
}
Do not include any explanation, preamble, or extra text. Output only the JSON object."""


def _infer_from_title(title: str) -> dict[str, str | None]:
    """Deterministically extract agent, map, and side from a video title.

    Searches for canonical agent and map names (case-insensitive).  Returns
    only the fields that could be matched — missing fields are omitted so the
    caller can merge this dict as a fallback without clobbering LLM results.
    """
    result: dict[str, str | None] = {}
    t = title.lower()

    for key, canonical in _CANONICAL_AGENTS.items():
        if key in t:
            result["agent"] = canonical
            break

    for key, canonical in _CANONICAL_MAPS.items():
        if key in t:
            result["map"] = canonical
            break

    if any(w in t for w in ("defense", "defend", "defending", "def ")):
        result["side"] = "defense"
    elif any(w in t for w in ("attack", "attacking", "offense", "atk ")):
        result["side"] = "attack"

    return result


def _apply_fallbacks(result: dict[str, str | None], text: str) -> None:
    """Fill in null fields using deterministic rules — modifies *result* in place.

    Called after LLM extraction so it only fires when the model couldn't determine
    a value.  Keeps extraction best-effort without silently losing known information.
    """
    # Ability: if agent is known and ability is null, use the agent's primary lineup ability
    if not result.get("ability") and result.get("agent"):
        default_ability = _DEFAULT_LINEUP_ABILITIES.get(result["agent"])
        if default_ability:
            result["ability"] = default_ability

    text_lower = text.lower()

    # Site: keyword scan — only fires when LLM returned null
    if not result.get("site"):
        for keywords, site in _SITE_HINTS:
            if any(kw in text_lower for kw in keywords):
                result["site"] = site
                break

    # Purpose: keyword scan over the transcript text
    if not result.get("purpose"):
        for keywords, purpose in _PURPOSE_HINTS:
            if any(kw in text_lower for kw in keywords):
                result["purpose"] = purpose
                break


def extract_lineup_metadata(text: str, settings: Any, *, video_title: str | None = None) -> dict:
    """Ask the LLM to extract structured lineup metadata from a transcript chunk.

    If *video_title* is provided it is used two ways:
    - Passed to the LLM as context so it can infer fields not stated in the transcript.
    - Parsed deterministically via :func:`_infer_from_title` to supply reliable
      agent/map/side defaults even when the LLM extraction fails or returns nulls.

    Returns a dict with keys: agent, ability, map, site, side, purpose.
    All values may be None if neither LLM nor title inference could determine them.
    Never raises — returns defaults on any failure.
    """
    defaults: dict[str, str | None] = {
        "agent": None,
        "ability": None,
        "map": None,
        "site": None,
        "side": None,
        "purpose": None,
    }

    # Deterministic title inference — reliable fallback regardless of LLM outcome
    title_fields = _infer_from_title(video_title) if video_title else {}

    try:
        from valocoach.llm.provider import call_llm

        context_line = f'Video: "{video_title}"\n\n' if video_title else ""
        prompt = (
            f"{context_line}"
            f'Transcript excerpt:\n"""\n{text[:1500]}\n"""\n\n'
            "Extract lineup metadata as JSON."
        )
        raw = call_llm(
            system=_EXTRACTION_SYSTEM,
            user=prompt,
            settings=settings,
            max_tokens=256,
        )
        if not raw:
            merged = dict(defaults)
            merged.update(title_fields)
            return merged

        parsed: dict | None = None
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            if m:
                try:
                    parsed = json.loads(m.group())
                except json.JSONDecodeError:
                    parsed = None

        if not isinstance(parsed, dict):
            merged = dict(defaults)
            merged.update(title_fields)
            return merged

        # Build result: title inference as base, LLM overrides any non-null field
        result: dict[str, str | None] = dict(defaults)
        result.update(title_fields)

        llm_agent = _canon_agent(parsed.get("agent"))
        llm_map = _canon_map(parsed.get("map"))
        llm_site = _canon_site(parsed.get("site"))
        if llm_agent:
            result["agent"] = llm_agent
        if llm_map:
            result["map"] = llm_map
        if llm_site:
            result["site"] = llm_site
        for key in ("ability", "side", "purpose"):
            v = parsed.get(key)
            if isinstance(v, str) and v.strip():
                result[key] = v.strip()

        _apply_fallbacks(result, text)
        return result

    except Exception as exc:
        log.debug("G2: lineup metadata extraction failed: %s", exc)
        merged = dict(defaults)
        merged.update(title_fields)
        _apply_fallbacks(merged, text)
        return merged


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
    expected_chunks: int | None = None,
    video_title: str | None = None,
) -> None:
    """Embed and upsert one lineup chunk into the LIVE collection.

    Runs the LLM metadata extraction pass (G2) then stores the chunk with
    full §8.2 metadata.  Extraction failure is non-blocking.

    ``expected_chunks`` is the total chunk count the caller expects to
    ingest for this video — stamped on every chunk so
    ``is_video_ingested`` can detect partial-ingest crashes.  Optional
    for backwards compatibility with direct callers.
    """
    from valocoach.retrieval.embedder import embed_one
    from valocoach.retrieval.vector_store import get_collection

    meta_fields = extract_lineup_metadata(text, settings, video_title=video_title or title)

    doc_id = f"lineup:{video_id}:{start_seconds}"
    metadata: dict[str, Any] = {
        "type": "lineup",
        "video_id": video_id,
        "title": title,
        "channel": channel,
        "start_seconds": start_seconds,
        "source": url,
        # TTL matches YouTube chunks so cache.purge_expired() cleans both in
        # one sweep — without this lineup chunks lived forever in Chroma.
        "expires_at_unix": int(time.time()) + _LINEUP_TTL_SECONDS,
        **{k: v for k, v in meta_fields.items() if v is not None},
    }
    if expected_chunks is not None:
        metadata["expected_chunks"] = expected_chunks

    try:
        vec = embed_one(text)
        coll = get_collection(data_dir, _COLLECTION)
        coll.upsert(ids=[doc_id], documents=[text], embeddings=[vec], metadatas=[metadata])
        log.info(
            "G1: upserted lineup chunk %s (agent=%s map=%s)",
            doc_id,
            meta_fields.get("agent"),
            meta_fields.get("map"),
        )
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
        # Add optional structured fields only if present.  Canonicalise the
        # filtered fields so retrieval $eq filters match regardless of how
        # the seed JSON was authored.
        for field in ("ability", "side", "purpose"):
            if entry.get(field):
                meta[field] = entry[field]
        if entry.get("agent"):
            meta["agent"] = _canon_agent(entry["agent"])
        if entry.get("map"):
            meta["map"] = _canon_map(entry["map"])
        if entry.get("site"):
            meta["site"] = _canon_site(entry["site"])
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
    # Note: ChromaDB only includes documents that have the filtered field present
    # in their metadata.  Lineup chunks whose LLM extraction returned None for
    # agent/map/site will not appear in filtered queries — they are still
    # reachable via unfiltered text-similarity search (omit agent/map_name/site).
    # Canonicalise filter values to match the write-side canonicalisation —
    # otherwise "sova" / "Sova" / "SOVA" all fail to match the stored "Sova".
    conditions: list[dict] = [{"type": {"$eq": "lineup"}}]
    if agent:
        conditions.append({"agent": {"$eq": _canon_agent(agent)}})
    if map_name:
        conditions.append({"map": {"$eq": _canon_map(map_name)}})
    if site:
        conditions.append({"site": {"$eq": _canon_site(site)}})

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

    for text, meta, dist in zip(docs, metas, dists, strict=False):
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
            lines.append(f'   📹 {channel} "{title}" @ {mins}:{secs:02d}')
        lines.append("")

    return "\n".join(lines).rstrip()


__all__ = [
    "_infer_from_title",
    "extract_lineup_metadata",
    "format_lineup_results",
    "ingest_lineup_chunk",
    "ingest_seed_lineups",
    "search_lineups",
]

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Context chunks ready for LLM injection."""

    static_chunks: list[str] = field(default_factory=list)
    meta_chunks: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    patch_version: str | None = None

    def to_context_string(self) -> str | None:
        """Flatten all chunks into a single GROUNDED CONTEXT block."""
        parts = self.static_chunks + self.meta_chunks
        return "\n\n".join(parts) if parts else None


def build_retrieval_queries(
    situation: str,
    map_name: str | None = None,
    agents: list[str] | None = None,
    side: str | None = None,
) -> list[str]:
    """Generate targeted retrieval queries from the coaching situation.

    Multiple focused queries outperform a single concatenated string because
    each one activates different embedding dimensions in the vector store.
    """
    queries = [situation]

    if map_name:
        queries.append(f"{map_name} callouts positions")
        if side:
            queries.append(f"{map_name} {side} strategies")

    if agents:
        for agent in agents[:3]:
            queries.append(f"{agent} abilities utility usage")

    return queries


def retrieve_static(
    situation: str,
    data_dir: Path,
    agent: str | None = None,
    map_: str | None = None,
    side: str | None = None,
    n_results: int = 5,
) -> RetrievalResult:
    """Synchronous retrieval from the static knowledge base + vector store.

    Used by the coach command (sync CLI context). Returns a RetrievalResult
    whose static_chunks are ordered: JSON facts first, then vector hits.
    """
    from valocoach.retrieval import format_agent_context, format_map_context, format_meta_context

    parts: list[str] = []
    sources: list[str] = []

    # 1. Structured JSON context — exact ability costs, callouts, meta tier.
    #    These are the most precise facts and always go first.
    if agent:
        ctx = format_agent_context(agent)
        if ctx:
            parts.append(ctx)
            sources.append(f"knowledge_base/agents/{agent}")
        else:
            log.warning("Agent '%s' not found in knowledge base.", agent)

    if map_:
        ctx = format_map_context(map_)
        if ctx:
            parts.append(ctx)
            sources.append(f"knowledge_base/maps/{map_}")
        else:
            log.warning("Map '%s' not found in knowledge base.", map_)

    meta_ctx = format_meta_context(agent=agent, map_name=map_)
    parts.append(meta_ctx)
    sources.append("knowledge_base/meta")

    # 2. Multi-query vector search — searches BOTH the static corpus and the
    #    live-scrape collection so per-query scraped meta is also surfaced.
    #    Each query targets a different semantic angle; results are deduplicated
    #    across queries and collections before being appended.
    try:
        from valocoach.retrieval.searcher import search
        from valocoach.retrieval.vector_store import LIVE_COLLECTION, STATIC_COLLECTION

        agents_list = [agent] if agent else None
        queries = build_retrieval_queries(situation, map_name=map_, agents=agents_list, side=side)

        seen: set[str] = set()
        vector_parts: list[str] = []

        for query in queries:
            for cname in (STATIC_COLLECTION, LIVE_COLLECTION):
                hits = search(
                    query,
                    data_dir,
                    n_results=3,
                    doc_types=["patch_note", "youtube", "web", "concept"],
                    max_distance=0.45,
                    collection_name=cname,
                )
                for hit in hits:
                    text = hit["text"]
                    if text not in seen:
                        seen.add(text)
                        name = hit["metadata"].get("name", "supplemental")
                        doc_type = hit["metadata"]["type"].upper()
                        vector_parts.append(f"[{doc_type}: {name}]\n{text}")
                        sources.append(hit["metadata"].get("source", "vector_store"))

        parts.extend(vector_parts[:n_results])

    except Exception:
        pass  # Ollama or ChromaDB unavailable — static JSON is the baseline

    return RetrievalResult(
        static_chunks=parts,
        meta_chunks=[],
        sources=list(dict.fromkeys(sources)),  # deduplicate, preserve order
        patch_version=None,
    )


async def retrieve_context(
    settings,
    situation: str,
    agent: str | None = None,
    map_: str | None = None,
    side: str | None = None,
    n_static: int = 5,
    n_meta: int = 3,
) -> RetrievalResult:
    """Full async retrieval pipeline: static corpus + live meta + patch version.

    Used by async orchestrators (scheduled tasks, future interactive mode).
    Falls back gracefully if live meta or patch tracking are unavailable.
    """
    from valocoach.retrieval.patch_tracker import get_current_patch

    result = retrieve_static(
        situation=situation,
        data_dir=settings.data_dir,
        agent=agent,
        map_=map_,
        side=side,
        n_results=n_static,
    )

    result.patch_version = await get_current_patch()

    try:
        meta_chunks, meta_sources = await _fetch_live_meta(
            settings=settings,
            situation=situation,
            map_=map_,
            n_results=n_meta,
        )
        result.meta_chunks = meta_chunks
        result.sources.extend(meta_sources)
    except Exception as exc:
        log.warning("Live meta retrieval failed (non-fatal): %s", exc)

    return result


async def _fetch_live_meta(
    settings,
    situation: str,
    map_: str | None,
    n_results: int,
) -> tuple[list[str], list[str]]:
    """Scrape and cache live meta content, then return top-n text chunks.

    Cache is checked first (volatile TTL). On a miss, the page is scraped,
    stored, and ingested into the vector store for future semantic queries.
    """
    from valocoach.retrieval.cache import get_cached, store_cached
    from valocoach.retrieval.ingester import ingest_text
    from valocoach.retrieval.scrapers.web import scrape_url

    urls_to_try: list[tuple[str, str]] = []
    if map_:
        urls_to_try.append(
            (
                f"https://www.metasrc.com/valorant/map/{map_.lower()}",
                "web",
            )
        )

    chunks: list[str] = []
    sources: list[str] = []

    for url, source in urls_to_try:
        text = await get_cached(url)
        if text is None:
            scraped = scrape_url(url, source=source)
            if not scraped:
                continue
            text = scraped.text
            await store_cached(url, text, source=source, ttl_tier="volatile")
            from valocoach.retrieval.vector_store import LIVE_COLLECTION

            ingest_text(
                settings.data_dir,
                text,
                doc_type="web",
                name=url,
                source=url,
                collection_name=LIVE_COLLECTION,
            )

        chunks.append(text[:2000])
        sources.append(url)

    return chunks[:n_results], sources

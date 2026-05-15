from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

log = logging.getLogger(__name__)


def _now_unix() -> int:
    """Current UTC unix timestamp (seconds).

    Used as the metadata key for live-collection TTL filtering — ChromaDB's
    ``$gt`` operator requires numeric operands and rejects strings, so we
    store the expiry as an int and compare against ``int(time.time())``.
    The SQLite cache continues to use ISO8601 strings (``MetaCache.expires_at``)
    because string comparisons via SQLAlchemy work fine there and the column
    is human-readable for debugging.
    """
    return int(datetime.now(UTC).timestamp())


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

    meta_ctx = format_meta_context(agent=agent, map_name=map_, data_dir=data_dir)
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

        # LIVE_COLLECTION docs carry ``expires_at_unix`` (int seconds) in their
        # metadata.  Filtering on ``expires_at_unix > now`` is the second line
        # of defence: even if the SQLite-backed invalidation hook didn't run
        # (process crash, patch detector skipped a cycle, ...) an expired
        # live doc cannot reach the coach prompt.  Docs ingested before this
        # change have no ``expires_at_unix`` field; ChromaDB excludes them
        # from ``$gt`` filters, which is the safe outcome — they get treated
        # as already expired and re-scraped on next demand.
        now_unix = _now_unix()

        for query in queries:
            for cname in (STATIC_COLLECTION, LIVE_COLLECTION):
                where_extra: dict | None = None
                if cname == LIVE_COLLECTION:
                    where_extra = {"expires_at_unix": {"$gt": now_unix}}
                hits = search(
                    query,
                    data_dir,
                    n_results=3,
                    doc_types=["patch_note", "youtube", "web", "concept"],
                    max_distance=0.45,
                    collection_name=cname,
                    where_extra=where_extra,
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

    except Exception as exc:
        log.warning("Vector search unavailable, falling back to static JSON: %s", exc)

    return RetrievalResult(
        static_chunks=parts,
        meta_chunks=[],
        sources=list(dict.fromkeys(sources)),  # deduplicate, preserve order
        patch_version=None,
    )



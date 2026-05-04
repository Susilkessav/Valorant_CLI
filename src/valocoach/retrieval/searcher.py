from __future__ import annotations

from pathlib import Path

from valocoach.retrieval.embedder import embed_one
from valocoach.retrieval.vector_store import (
    LIVE_COLLECTION,
    STATIC_COLLECTION,
    get_collection,
)


def search(
    query: str,
    data_dir: Path,
    n_results: int = 5,
    doc_types: list[str] | None = None,
    max_distance: float = 0.5,
    collection_name: str = STATIC_COLLECTION,
    where_extra: dict | None = None,
) -> list[dict]:
    """Semantic search over a single ChromaDB collection.

    Returns a list of ``{text, metadata, distance}`` dicts ordered by relevance,
    filtered to those with cosine distance ≤ max_distance.

    Pass ``collection_name=LIVE_COLLECTION`` to query the live-scrape bucket.
    Most callers should use ``retrieve_static`` (which searches both collections)
    rather than calling this directly.

    Args:
        where_extra: Additional metadata filter merged with ``doc_types`` via
                     ``$and``.  Used by the retriever to gate ``LIVE_COLLECTION``
                     hits on ``expires_at > now`` so an expired live cache row
                     can never reach the coach prompt — even if the cache
                     invalidation hook hasn't run yet.
    """
    collection = get_collection(data_dir, collection_name)
    count = collection.count()
    if count == 0:
        return []

    embedding = embed_one(query)

    # Merge doc_types filter with any caller-supplied where clause.  ChromaDB
    # requires a single top-level operator, so two filters need ``$and``.
    filters: list[dict] = []
    if doc_types:
        filters.append({"type": {"$in": doc_types}})
    if where_extra:
        filters.append(where_extra)
    if not filters:
        where: dict | None = None
    elif len(filters) == 1:
        where = filters[0]
    else:
        where = {"$and": filters}

    results = collection.query(
        query_embeddings=[embedding],
        n_results=min(n_results, count),
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    docs: list[dict] = []
    for i, doc in enumerate(results["documents"][0]):
        distance = results["distances"][0][i]
        if distance <= max_distance:
            docs.append(
                {
                    "text": doc,
                    "metadata": results["metadatas"][0][i],
                    "distance": distance,
                }
            )
    return docs


def collection_stats(data_dir: Path) -> dict:
    """Return total doc count and per-type breakdown across both collections.

    Keys in ``by_type`` are prefixed with the collection label so the caller
    can see which bucket each type lives in, e.g. ``"[static] concept"``.
    """
    by_type: dict[str, int] = {}
    total = 0

    for label, cname in (("static", STATIC_COLLECTION), ("live", LIVE_COLLECTION)):
        coll = get_collection(data_dir, cname)
        count = coll.count()
        total += count
        if count > 0:
            for meta in coll.get(include=["metadatas"])["metadatas"]:
                key = f"[{label}] {meta.get('type', 'unknown')}"
                by_type[key] = by_type.get(key, 0) + 1

    return {"total": total, "by_type": by_type}

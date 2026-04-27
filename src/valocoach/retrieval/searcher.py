from __future__ import annotations

from pathlib import Path

from valocoach.retrieval.embedder import embed_one
from valocoach.retrieval.vector_store import get_collection


def search(
    query: str,
    data_dir: Path,
    n_results: int = 5,
    doc_types: list[str] | None = None,
    max_distance: float = 0.5,
) -> list[dict]:
    """Semantic search over the vector store.

    Returns a list of ``{text, metadata, distance}`` dicts ordered by relevance,
    filtered to those with cosine distance ≤ max_distance.
    """
    collection = get_collection(data_dir)
    count = collection.count()
    if count == 0:
        return []

    embedding = embed_one(query)
    where = {"type": {"$in": doc_types}} if doc_types else None

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
            docs.append({
                "text": doc,
                "metadata": results["metadatas"][0][i],
                "distance": distance,
            })
    return docs


def collection_stats(data_dir: Path) -> dict:
    """Return total doc count and per-type breakdown."""
    collection = get_collection(data_dir)
    total = collection.count()
    if total == 0:
        return {"total": 0, "by_type": {}}

    all_items = collection.get(include=["metadatas"])
    by_type: dict[str, int] = {}
    for meta in all_items["metadatas"]:
        t = meta.get("type", "unknown")
        by_type[t] = by_type.get(t, 0) + 1

    return {"total": total, "by_type": by_type}

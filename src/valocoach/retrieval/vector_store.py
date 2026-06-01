from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings

log = logging.getLogger(__name__)

# Two separate collections so static corpus and live-scraped meta are
# independently manageable.  ``valocoach ingest --seed`` only touches STATIC;
# per-query web scraping only touches LIVE; ``--clear`` knows which to nuke.
STATIC_COLLECTION = "valocoach_static"
LIVE_COLLECTION = "valocoach_live"

# Backward-compat alias — prefer the explicit constants above in new code.
COLLECTION_NAME = STATIC_COLLECTION


@lru_cache(maxsize=4)
def _cached_client(chroma_dir_str: str) -> chromadb.PersistentClient:
    """Memoised PersistentClient — keyed by absolute on-disk path.

    Constructing a ``PersistentClient`` is expensive: it reopens the
    underlying SQLite, reloads the HNSW index, and re-runs migration checks.
    The coach retrieval path issues a multi-query, two-collection search,
    so without caching every coaching turn paid for ~8 fresh client builds.
    Caching by path lets tests with a ``tmp_path`` data dir get isolated
    clients while production reuses the one client per ``data_dir``.
    """
    return chromadb.PersistentClient(
        path=chroma_dir_str,
        settings=ChromaSettings(anonymized_telemetry=False),
    )


def get_client(data_dir: Path) -> chromadb.PersistentClient:
    chroma_dir = data_dir / "chroma"
    chroma_dir.mkdir(parents=True, exist_ok=True)
    return _cached_client(str(chroma_dir.resolve()))


def reset_client_cache() -> None:
    """Drop the memoised clients.

    Tests that mutate the on-disk store across runs (clear + reseed) and
    explicit ``ingest --clear`` paths can call this to ensure subsequent
    callers see a freshly-opened store.
    """
    _cached_client.cache_clear()


def get_collection(
    data_dir: Path,
    collection_name: str = STATIC_COLLECTION,
) -> chromadb.Collection:
    client = get_client(data_dir)
    return client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )


def clear_collection(
    data_dir: Path,
    collection_name: str = STATIC_COLLECTION,
) -> None:
    """Delete and recreate *one* collection (full reset for that collection)."""
    import contextlib

    client = get_client(data_dir)
    with contextlib.suppress(Exception):
        # Silently skip if the collection didn't exist yet.
        client.delete_collection(collection_name)
    client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )
    log.debug("Collection %r cleared and recreated.", collection_name)


def delete_by_metadata(
    data_dir: Path,
    where: dict,
    collection_name: str = STATIC_COLLECTION,
) -> None:
    """Delete all documents matching a metadata filter without resetting the collection."""
    collection = get_collection(data_dir, collection_name)
    collection.delete(where=where)
    log.debug("Deleted documents matching %r from %r.", where, collection_name)


def collection_count(
    data_dir: Path,
    collection_name: str = STATIC_COLLECTION,
) -> int:
    """Return the number of documents in the specified collection."""
    return get_collection(data_dir, collection_name).count()

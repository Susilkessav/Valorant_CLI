from __future__ import annotations

import logging
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings

log = logging.getLogger(__name__)

COLLECTION_NAME = "valocoach_knowledge"


def get_client(data_dir: Path) -> chromadb.PersistentClient:
    chroma_dir = data_dir / "chroma"
    chroma_dir.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(
        path=str(chroma_dir),
        settings=ChromaSettings(anonymized_telemetry=False),
    )


def get_collection(data_dir: Path) -> chromadb.Collection:
    client = get_client(data_dir)
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def clear_collection(data_dir: Path) -> None:
    """Delete and recreate the collection (full reset)."""
    client = get_client(data_dir)
    client.delete_collection(COLLECTION_NAME)
    client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    log.debug("Collection %r cleared and recreated.", COLLECTION_NAME)


def delete_by_metadata(data_dir: Path, where: dict) -> None:
    """Delete all documents matching a metadata filter without resetting the collection."""
    collection = get_collection(data_dir)
    collection.delete(where=where)
    log.debug("Deleted documents matching %r from %r.", where, COLLECTION_NAME)


def collection_count(data_dir: Path) -> int:
    """Return the total number of documents in the collection."""
    return get_collection(data_dir).count()

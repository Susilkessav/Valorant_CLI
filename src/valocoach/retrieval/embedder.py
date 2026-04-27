from __future__ import annotations

import logging

import ollama

log = logging.getLogger(__name__)

EMBED_MODEL = "nomic-embed-text"
EMBED_DIM = 768


def embed(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts using nomic-embed-text via Ollama."""
    if not texts:
        return []
    log.debug("Embedding %d text(s) with %s", len(texts), EMBED_MODEL)
    result = ollama.embed(model=EMBED_MODEL, input=texts)
    return result["embeddings"]


def embed_one(text: str) -> list[float]:
    return embed([text])[0]


def is_available() -> bool:
    """Return True if the embedding model is reachable."""
    try:
        embed_one("ping")
        return True
    except Exception:
        return False

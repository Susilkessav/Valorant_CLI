from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from valocoach.retrieval.chunker import chunk_markdown
from valocoach.retrieval.embedder import embed
from valocoach.retrieval.vector_store import get_collection

DocType = Literal["agent", "map", "meta", "concept", "patch_note", "youtube", "web"]

_AGENTS_FILE = Path(__file__).parent / "data" / "agents.json"
_MAPS_FILE = Path(__file__).parent / "data" / "maps.json"
_META_FILE = Path(__file__).parent / "data" / "meta.json"


def _upsert_batch(
    data_dir: Path,
    texts: list[str],
    ids: list[str],
    metadatas: list[dict],
    batch_size: int = 32,
) -> int:
    collection = get_collection(data_dir)
    total = 0
    for i in range(0, len(texts), batch_size):
        t = texts[i : i + batch_size]
        d = ids[i : i + batch_size]
        m = metadatas[i : i + batch_size]
        embeddings = embed(t)
        collection.upsert(ids=d, documents=t, embeddings=embeddings, metadatas=m)
        total += len(t)
    return total


def ingest_knowledge_base(data_dir: Path) -> dict[str, int]:
    """Embed and upsert all static JSON knowledge into the vector store."""
    from valocoach.retrieval.agents import format_agent_context
    from valocoach.retrieval.maps import format_map_context
    from valocoach.retrieval.meta import format_meta_context

    texts: list[str] = []
    ids: list[str] = []
    metas: list[dict] = []

    # Agents — one doc per agent
    with open(_AGENTS_FILE) as f:
        agents = json.load(f)["agents"]
    for agent in agents:
        text = format_agent_context(agent["name"]) or ""
        texts.append(text)
        ids.append(f"agent:{agent['name'].lower()}")
        metas.append({"type": "agent", "name": agent["name"], "role": agent["role"], "source": "knowledge_base"})

    n_agents = len(agents)

    # Maps — one doc per map
    with open(_MAPS_FILE) as f:
        maps = json.load(f)["maps"]
    for map_data in maps:
        text = format_map_context(map_data["name"]) or ""
        texts.append(text)
        ids.append(f"map:{map_data['name'].lower()}")
        metas.append({"type": "map", "name": map_data["name"], "source": "knowledge_base"})

    n_maps = len(maps)

    # Meta — global tier list + one doc per map-specific meta
    with open(_META_FILE) as f:
        meta_data = json.load(f)

    tier_text = format_meta_context()
    texts.append(tier_text)
    ids.append("meta:tier_list")
    metas.append({"type": "meta", "name": "tier_list", "source": "knowledge_base"})

    map_meta_entries = meta_data.get("map_meta", {})
    for map_name in map_meta_entries:
        text = format_meta_context(map_name=map_name)
        texts.append(text)
        ids.append(f"meta:map:{map_name.lower()}")
        metas.append({"type": "meta", "name": f"{map_name}_meta", "source": "knowledge_base"})

    n_meta = 1 + len(map_meta_entries)

    total = _upsert_batch(data_dir, texts, ids, metas)
    return {"agents": n_agents, "maps": n_maps, "meta": n_meta, "total": total}


def ingest_text(
    data_dir: Path,
    text: str,
    doc_type: DocType,
    name: str,
    source: str,
    max_tokens: int = 400,
    extra_metadata: dict | None = None,
) -> int:
    """Chunk a raw text block and upsert into the vector store."""
    chunks = chunk_markdown(text, source=source, max_tokens=max_tokens)
    texts = [c.text for c in chunks]
    ids = [f"{doc_type}:{source}:{c.chunk_index}" for c in chunks]
    base = {"type": doc_type, "name": name, "source": source, **(extra_metadata or {})}
    metadatas = [{**base, "chunk": c.chunk_index} for c in chunks]
    return _upsert_batch(data_dir, texts, ids, metadatas)

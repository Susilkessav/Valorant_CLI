from __future__ import annotations

import json
from pathlib import Path

_DATA_FILE = Path(__file__).parent / "data" / "meta.json"
_cache: dict | None = None


def _load() -> dict:
    global _cache
    if _cache is None:
        with open(_DATA_FILE) as f:
            _cache = json.load(f)
    return _cache


def get_meta() -> dict:
    return _load()


def format_meta_context(
    agent: str | None = None,
    map_name: str | None = None,
) -> str:
    """Return a compact LLM-injectable meta block.

    Includes the tier list header, economy thresholds, and — when provided —
    map-specific and agent-specific meta snippets.
    """
    meta = _load()
    lines: list[str] = [f"META (Patch {meta['patch']}, updated {meta['updated']})"]

    tier = meta.get("tier_list", {})
    for rank in ("S", "A", "B", "C"):
        agents = tier.get(rank, [])
        if agents:
            lines.append(f"  {rank}-Tier: {', '.join(agents)}")

    eco = meta.get("economy", {})
    lines.append(
        f"Economy thresholds — Full buy: {eco.get('full_buy', 3900)} cr  |  "
        f"Half buy: {eco.get('half_buy', 2400)} cr  |  "
        f"Eco/save: <{eco.get('eco_save', 1600)} cr"
    )

    if map_name:
        map_meta = meta.get("map_meta", {}).get(map_name.title())
        if map_meta:
            top = ", ".join(map_meta.get("top_agents", []))
            lines.append(f"\n{map_name.title()} meta — Top agents: {top}")
            if map_meta.get("notes"):
                lines.append(f"  {map_meta['notes']}")

    if agent:
        agent_meta = meta.get("agent_meta", {}).get(agent.title())
        if agent_meta:
            lines.append(
                f"\n{agent.title()} meta — Tier: {agent_meta.get('tier', '?')}  |  "
                f"Pick rate: {agent_meta.get('pick_rate', 'N/A')}  |  "
                f"Win rate: {agent_meta.get('win_rate', 'N/A')}"
            )
            if agent_meta.get("reason"):
                lines.append(f"  {agent_meta['reason']}")

    return "\n".join(lines)

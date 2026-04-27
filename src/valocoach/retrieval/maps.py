from __future__ import annotations

import json
from difflib import get_close_matches
from pathlib import Path

_DATA_FILE = Path(__file__).parent / "data" / "maps.json"
_cache: list[dict] | None = None


def _load() -> list[dict]:
    global _cache
    if _cache is None:
        with open(_DATA_FILE) as f:
            _cache = json.load(f)["maps"]
    return _cache


def get_map(name: str) -> dict | None:
    """Return map data by name — case-insensitive, fuzzy-matched."""
    maps = _load()
    canonical = {m["name"].lower(): m for m in maps}

    if name.lower() in canonical:
        return canonical[name.lower()]

    matches = get_close_matches(name.lower(), list(canonical.keys()), n=1, cutoff=0.6)
    if matches:
        return canonical[matches[0]]
    return None


def list_map_names() -> list[str]:
    return [m["name"] for m in _load()]


def format_map_context(name: str) -> str | None:
    """Return a compact LLM-injectable callout block for the given map.

    Returns None if the map name cannot be resolved.
    """
    map_data = get_map(name)
    if not map_data:
        return None

    sites_str = ", ".join(map_data["sites"])
    lines = [f"MAP: {map_data['name']}  (Sites: {sites_str})", "Callouts:"]

    for area, callouts in map_data["callouts"].items():
        lines.append(f"  {area.upper()}: {', '.join(callouts)}")

    if map_data.get("rotations"):
        lines.append(f"Rotations: {map_data['rotations']}")
    if map_data.get("notes"):
        lines.append(f"Map notes: {map_data['notes']}")

    return "\n".join(lines)

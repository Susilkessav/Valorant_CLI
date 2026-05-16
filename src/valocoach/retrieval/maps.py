from __future__ import annotations

import json
import logging
from difflib import get_close_matches
from pathlib import Path

log = logging.getLogger(__name__)

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

    # C5 — stale-map warning: warn when map data hasn't been verified against
    # the current patch so the LLM knows callouts might be outdated.
    last_verified = map_data.get("last_verified_patch")
    if last_verified:
        try:
            from valocoach.retrieval.meta import get_meta
            current_patch = get_meta().get("patch", "")
            if current_patch and last_verified != current_patch:
                lines.append(
                    f"⚠ Map data last verified for patch {last_verified} — "
                    f"callouts may be outdated for patch {current_patch}."
                )
                log.debug(
                    "C5: stale map warning for %s (verified=%s, current=%s)",
                    map_data["name"],
                    last_verified,
                    current_patch,
                )
        except Exception:
            pass  # Never block map context due to a staleness check failure

    # F3 — provenance tag so LLM can cite the source
    lines.append(f"[SOURCE: knowledge_base/maps/{map_data['name']}]")

    return "\n".join(lines)

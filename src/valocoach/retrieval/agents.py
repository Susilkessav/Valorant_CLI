from __future__ import annotations

import json
from difflib import get_close_matches
from pathlib import Path

_DATA_FILE = Path(__file__).parent / "data" / "agents.json"
_cache: list[dict] | None = None


def _load() -> list[dict]:
    global _cache
    if _cache is None:
        with open(_DATA_FILE) as f:
            _cache = json.load(f)["agents"]
    return _cache


def list_agent_names() -> list[str]:
    """Return canonical agent names from the bundled JSON, in source order."""
    return [a["name"] for a in _load()]


def get_agent(name: str) -> dict | None:
    """Return agent data by name — case-insensitive, fuzzy-matched."""
    agents = _load()
    canonical = {a["name"].lower(): a for a in agents}

    # Exact match first.
    if name.lower() in canonical:
        return canonical[name.lower()]

    # Fuzzy match on display names.
    display_names = list(canonical.keys())
    matches = get_close_matches(name.lower(), display_names, n=1, cutoff=0.6)
    if matches:
        return canonical[matches[0]]
    return None


def format_agent_context(name: str) -> str | None:
    """Return a compact LLM-injectable block for the given agent.

    Returns None if the agent name cannot be resolved.
    """
    agent = get_agent(name)
    if not agent:
        return None

    lines = [f"AGENT: {agent['name']} ({agent['role']})", "Abilities:"]
    for key, ab in agent["abilities"].items():
        if "ult_points" in ab:
            cost_str = f"{ab['ult_points']} ult pts"
        elif ab.get("cost", 0) == 0:
            cost_str = "free"
        else:
            cost_str = f"{ab['cost']} creds"

        charges = f", {ab['charges']} charge(s)" if "charges" in ab else ""
        lines.append(f"  [{key}] {ab['name']} ({cost_str}{charges}): {ab['description']}")

    if agent.get("playstyle"):
        lines.append(f"Playstyle: {agent['playstyle']}")
    if agent.get("economy_tip"):
        lines.append(f"Economy note: {agent['economy_tip']}")

    return "\n".join(lines)

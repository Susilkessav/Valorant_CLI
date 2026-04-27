#!/usr/bin/env python3
"""Fetch agent abilities and map callouts from valorant-api.com and write
structured markdown files to corpus/agents/ and corpus/maps/.

Run once after setup. Re-run whenever a new agent or map is added.

Usage:
    python scripts/build_corpus.py
    python scripts/build_corpus.py --agents-only
    python scripts/build_corpus.py --maps-only
    python scripts/build_corpus.py --dry-run      # print paths without writing
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import textwrap
from collections import defaultdict
from pathlib import Path

import httpx

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
CORPUS_AGENTS = ROOT / "corpus" / "agents"
CORPUS_MAPS = ROOT / "corpus" / "maps"
AGENTS_JSON = ROOT / "src" / "valocoach" / "retrieval" / "data" / "agents.json"
MAPS_JSON = ROOT / "src" / "valocoach" / "retrieval" / "data" / "maps.json"

API_BASE = "https://valorant-api.com/v1"
LANG = "en-US"

# ---------------------------------------------------------------------------
# Slot → keybind label
# This mapping matches Riot's internal categorisation for the majority of
# agents. A small number of agents (e.g. Gekko) have unusual slot assignments
# — ability names are always correct; treat keybind labels as indicative.
# ---------------------------------------------------------------------------
SLOT_KEY: dict[str, str] = {
    "Grenade": "C",
    "Ability1": "Q",
    "Ability2": "E",
    "Ultimate": "X",
    "Passive": "Passive",
}

# Maps to skip — range/deathmatch arenas that have no competitive sites.
_NON_COMPETITIVE = {"District", "Kasbah", "Drift", "Piazza", "Glitch"}


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def _get(endpoint: str, **params) -> dict:
    params.setdefault("language", LANG)
    resp = httpx.get(f"{API_BASE}/{endpoint}", params=params, timeout=20, follow_redirects=True)
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Local JSON cost/playstyle lookup
# ---------------------------------------------------------------------------

def _load_agent_costs() -> dict[str, dict]:
    """Return {agent_name_lower: {ability_name_lower: cost_str, playstyle, economy_tip}}."""
    if not AGENTS_JSON.exists():
        return {}
    with open(AGENTS_JSON) as f:
        raw = json.load(f)["agents"]
    out: dict[str, dict] = {}
    for a in raw:
        abilities: dict[str, str] = {}
        for ab in a["abilities"].values():
            name = ab["name"].lower()
            if "ult_points" in ab:
                abilities[name] = f"{ab['ult_points']} ult pts"
            elif ab.get("cost", 0) == 0:
                charges = f", {ab['charges']} charge(s)" if "charges" in ab else ""
                abilities[name] = f"free{charges}"
            else:
                charges = f", {ab['charges']} charge(s)" if "charges" in ab else ""
                abilities[name] = f"{ab['cost']} cr{charges}"
        out[a["name"].lower()] = {
            "costs": abilities,
            "playstyle": a.get("playstyle", ""),
            "economy_tip": a.get("economy_tip", ""),
        }
    return out


def _load_map_extras() -> dict[str, dict]:
    """Return {map_name_lower: {rotations, notes, extra_callouts}}."""
    if not MAPS_JSON.exists():
        return {}
    with open(MAPS_JSON) as f:
        raw = json.load(f)["maps"]
    out: dict[str, dict] = {}
    for m in raw:
        extra: dict[str, list[str]] = {}
        for area, names in m.get("callouts", {}).items():
            extra[area.lower()] = names
        out[m["name"].lower()] = {
            "rotations": m.get("rotations", ""),
            "notes": m.get("notes", ""),
            "extra_callouts": extra,
        }
    return out


# ---------------------------------------------------------------------------
# Agent markdown generation
# ---------------------------------------------------------------------------

def _agent_markdown(agent: dict, costs: dict[str, dict]) -> str:
    name: str = agent["displayName"]
    role: str = agent.get("role", {}).get("displayName", "Unknown")
    description: str = agent.get("description", "").strip()
    abilities: list[dict] = agent.get("abilities", [])

    local = costs.get(name.lower(), {})
    local_costs: dict[str, str] = local.get("costs", {})
    playstyle: str = local.get("playstyle", "")
    economy_tip: str = local.get("economy_tip", "")

    lines: list[str] = [f"# {name} — {role}", ""]
    if description:
        lines += [description, ""]

    lines += ["## Abilities", ""]
    for ab in abilities:
        slot = ab.get("slot", "")
        key = SLOT_KEY.get(slot, slot)
        ab_name: str = ab.get("displayName", "")
        ab_desc: str = ab.get("description", "").strip()

        # Look up cost from local JSON. Normalise both sides so "Blade Storm"
        # matches "Bladestorm" and "FRAG/ment" matches "FRAGMENT".
        def _norm(s: str) -> str:
            return re.sub(r"[\s/\-]+", "", s.lower())

        cost = local_costs.get(ab_name.lower()) or local_costs.get(_norm(ab_name), "")
        cost_str = f"  ({cost})" if cost else ""

        if key == "Passive":
            lines.append(f"### [Passive] {ab_name}")
        else:
            lines.append(f"### [{key}] {ab_name}{cost_str}")
        lines.append("")
        if ab_desc:
            # Wrap long descriptions at 100 chars for readability
            for para in ab_desc.split("\n"):
                lines.append(textwrap.fill(para.strip(), width=100) if para.strip() else "")
        lines.append("")

    if playstyle:
        lines += ["## Playstyle", "", playstyle, ""]
    if economy_tip:
        lines += ["## Economy Note", "", economy_tip, ""]

    return "\n".join(lines).rstrip() + "\n"


# ---------------------------------------------------------------------------
# Map markdown generation
# ---------------------------------------------------------------------------

def _map_markdown(map_data: dict, extras: dict[str, dict]) -> str:
    name: str = map_data["displayName"]
    tactical: str = map_data.get("tacticalDescription") or ""
    narrative: str = (map_data.get("narrativeDescription") or "").strip()
    callouts: list[dict] = map_data.get("callouts", [])

    local = extras.get(name.lower(), {})
    rotations: str = local.get("rotations", "")
    notes: str = local.get("notes", "")
    extra: dict[str, list[str]] = local.get("extra_callouts", {})

    # Group API callouts by super-region
    by_region: dict[str, list[str]] = defaultdict(list)
    for c in callouts:
        sr = c.get("superRegionName", "").strip()
        rn = c.get("regionName", "").strip()
        if sr and rn:
            by_region[sr].append(rn)

    def _strip_region_prefix(name: str, prefix: str) -> str:
        """'A Long' with prefix 'a' → 'long'. Used for deduplication."""
        lo = name.lower()
        if lo.startswith(prefix + " "):
            return lo[len(prefix) + 1:]
        return lo

    # Merge extra callouts from JSON (add names not already present from API)
    for area_key, extra_names in extra.items():
        matched = next(
            (sr for sr in by_region if sr.lower() == area_key or sr.lower().startswith(area_key)),
            None,
        )
        if matched:
            # De-dup by comparing both the full name and the prefix-stripped form.
            # This handles "Long" (API) == "A Long" (JSON) for region "A".
            prefix = matched.lower()
            existing_stripped = {_strip_region_prefix(n, prefix) for n in by_region[matched]}
            for en in extra_names:
                if _strip_region_prefix(en, prefix) not in existing_stripped:
                    by_region[matched].append(en)
                    existing_stripped.add(_strip_region_prefix(en, prefix))
        elif area_key not in ("attacker side", "defender side"):
            label = area_key.title()
            existing = {n.lower() for n in by_region.get(label, [])}
            for en in extra_names:
                if en.lower() not in existing:
                    by_region[label].append(en)

    # Derive sites list from tactical description or region names
    sites_str = tactical.replace("Sites", "").strip(" /")
    if not sites_str:
        sites_str = "/".join(
            sr for sr in sorted(by_region)
            if sr not in ("Attacker Side", "Defender Side", "Mid")
        )

    lines: list[str] = [f"# {name}", ""]
    if narrative:
        lines += [narrative, ""]
    lines += [f"**Sites:** {sites_str}", ""]

    # Canonical site order: Attacker → A → B → C → Mid → Defender
    def _region_sort_key(sr: str) -> tuple[int, str]:
        order = {"Attacker Side": 0, "A": 1, "B": 2, "C": 3, "Mid": 4, "Defender Side": 5}
        return (order.get(sr, 3), sr)

    lines.append("## Callouts")
    lines.append("")
    for region in sorted(by_region, key=_region_sort_key):
        names = sorted(set(by_region[region]))
        lines.append(f"### {region}")
        lines.append("")
        lines.append(", ".join(names))
        lines.append("")

    if rotations:
        lines += ["## Rotation Notes", "", rotations, ""]
    if notes:
        lines += ["## Tactical Notes", "", notes, ""]

    return "\n".join(lines).rstrip() + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _safe_filename(name: str) -> str:
    """Sanitise a display name for use as a filename."""
    return name.replace("/", "-").replace("\\", "-").replace(":", "-")


def build_agents(dry_run: bool, costs: dict) -> list[str]:
    print("Fetching agents from valorant-api.com …")
    data = _get("agents", isPlayableCharacter="true")
    agents = sorted(data["data"], key=lambda a: a["displayName"])

    # Deduplicate by displayName (API occasionally returns duplicates)
    seen: set[str] = set()
    unique: list[dict] = []
    for a in agents:
        if a["displayName"] not in seen:
            seen.add(a["displayName"])
            unique.append(a)

    paths: list[str] = []
    CORPUS_AGENTS.mkdir(parents=True, exist_ok=True)
    for agent in unique:
        name = agent["displayName"]
        md = _agent_markdown(agent, costs)
        path = CORPUS_AGENTS / f"{_safe_filename(name)}.md"
        if not dry_run:
            path.write_text(md, encoding="utf-8")
        paths.append(str(path.relative_to(ROOT)))

    role_summary: dict[str, int] = defaultdict(int)
    for a in unique:
        role_summary[a.get("role", {}).get("displayName", "?")] += 1

    print(f"  {len(unique)} agents written  ({dict(role_summary)})")
    return paths


def build_maps(dry_run: bool, extras: dict) -> list[str]:
    print("Fetching maps from valorant-api.com …")
    data = _get("maps")
    maps = data["data"]

    # Filter to competitive maps: must have a site-based tactical description
    # and not be a known training/deathmatch arena.
    competitive = [
        m for m in maps
        if "Site" in (m.get("tacticalDescription") or "")
        and m.get("displayName") not in _NON_COMPETITIVE
    ]
    competitive.sort(key=lambda m: m["displayName"])

    paths: list[str] = []
    CORPUS_MAPS.mkdir(parents=True, exist_ok=True)
    for map_data in competitive:
        name = map_data["displayName"]
        md = _map_markdown(map_data, extras)
        path = CORPUS_MAPS / f"{name}.md"
        if not dry_run:
            path.write_text(md, encoding="utf-8")
        paths.append(str(path.relative_to(ROOT)))

    print(f"  {len(competitive)} maps written: {[m['displayName'] for m in competitive]}")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--agents-only", action="store_true")
    parser.add_argument("--maps-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Print paths without writing files.")
    args = parser.parse_args()

    costs = _load_agent_costs()
    extras = _load_map_extras()

    all_paths: list[str] = []

    if not args.maps_only:
        all_paths += build_agents(dry_run=args.dry_run, costs=costs)

    if not args.agents_only:
        all_paths += build_maps(dry_run=args.dry_run, extras=extras)

    if args.dry_run:
        print("\nDry-run — would write:")
        for p in all_paths:
            print(f"  {p}")
    else:
        print(f"\nDone. {len(all_paths)} files in corpus/")
        print("Next: run  valocoach ingest --corpus  to embed them into the vector store.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)

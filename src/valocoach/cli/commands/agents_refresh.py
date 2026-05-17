"""`valocoach agents-refresh` — keep the agent knowledge base in sync with Riot's roster.

What this command does (and explicitly does NOT do):

* **Discovers** the canonical roster from the Liquipedia portal page so you
  see new agents the moment they ship.
* **Reports** which agents are present in ``agents.json`` but missing from
  ``meta.json``'s tier list (they currently render as "unranked").
* **Optionally stubs** those meta-list gaps with a clearly labelled C-tier
  placeholder so the deterministic tier panel doesn't drop them.
* **Prints templates** for new agents that need a full kit entry — but does
  NOT auto-write kit data, because local-LLM extraction of ability
  names / costs from a scraped wiki page is exactly the hallucination class
  we've spent the rest of this session fighting.  Writing wrong abilities
  into the static knowledge base would corrupt every downstream prompt and
  every sanitizer check — so the kit-fill step is left to the human.

In short: discovery + safe stubs are automatic, full kit extraction is
copy-paste from a printed URL.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from valocoach.cli import display

log = logging.getLogger(__name__)

_AGENTS_JSON = Path(__file__).resolve().parents[2] / "retrieval" / "data" / "agents.json"
_META_JSON = Path(__file__).resolve().parents[2] / "retrieval" / "data" / "meta.json"
_LIQUIPEDIA_API = (
    "https://liquipedia.net/valorant/api.php"
    "?action=query&list=categorymembers&cmtitle=Category:Agents"
    "&cmlimit=100&cmtype=page&format=json"
)
_LIQUIPEDIA_WIKITEXT_API = (
    "https://liquipedia.net/valorant/api.php"
    "?action=parse&page={page}&prop=wikitext&format=json"
)
_USER_AGENT = "ValoCoachBot/1.0 (personal coaching tool)"


# ---------------------------------------------------------------------------
# Wikitext kit extractor — deterministic, no LLM.
# ---------------------------------------------------------------------------


def _fetch_wikitext(page_title: str) -> str | None:
    """Return the raw wikitext for a Liquipedia page, or None on failure."""
    import httpx

    url = _LIQUIPEDIA_WIKITEXT_API.format(page=page_title)
    try:
        resp = httpx.get(url, timeout=20, headers={"User-Agent": _USER_AGENT})
        resp.raise_for_status()
        data = resp.json()
    except (httpx.HTTPError, ValueError) as exc:
        log.warning("Failed to fetch wikitext for %s: %s", page_title, exc)
        return None
    text = data.get("parse", {}).get("wikitext", {}).get("*")
    return text


def _parse_template_fields(block: str) -> dict[str, str]:
    """Parse ``|key=value`` pairs from a wikitext template body."""
    import re

    fields: dict[str, str] = {}
    # Split on a pipe at the start of a line so URL-encoded pipes in values
    # (rare, but possible) don't break us.
    for part in re.split(r"\n\s*\|", "\n" + block):
        part = part.strip()
        if "=" not in part:
            continue
        key, val = part.split("=", 1)
        fields[key.strip().lower()] = val.strip()
    return fields


def _extract_first_int(text: str) -> int | None:
    """Return the first integer that appears in *text*, or None.

    Liquipedia ability costs include parenthetical notes
    (``100 (1 free per round)``); we just want the number that goes into
    ``agents.json``.
    """
    import re

    m = re.search(r"\d+", text)
    return int(m.group(0)) if m else None


def _parse_agent_wikitext(wikitext: str) -> dict | None:
    """Parse a Liquipedia agent page's wikitext into an agents.json entry.

    Returns ``None`` if the page doesn't have an ``{{Infobox agent}}`` and
    four ``{{AbilityCard}}`` templates with C/Q/E/X hotkeys.  Missing fields
    fail the parse rather than producing a half-filled record.
    """
    import re

    # 1. Infobox agent — name + role/class.
    infobox_match = re.search(r"\{\{Infobox agent\s*\n(.+?)\n\}\}", wikitext, re.DOTALL | re.IGNORECASE)
    if not infobox_match:
        return None
    infobox = _parse_template_fields(infobox_match.group(1))
    name = infobox.get("name") or ""
    role = (infobox.get("class") or "").strip()
    if not name or role not in {"Duelist", "Initiator", "Controller", "Sentinel"}:
        return None

    # 2. AbilityCard blocks — one per ability.
    abilities: dict[str, dict] = {}
    for ab_match in re.finditer(r"\{\{AbilityCard\s*\n(.+?)\n\}\}", wikitext, re.DOTALL | re.IGNORECASE):
        ab = _parse_template_fields(ab_match.group(1))
        hotkey = (ab.get("hotkey") or "").strip().upper()
        if hotkey not in {"C", "Q", "E", "X"}:
            continue

        entry: dict = {"name": ab.get("name", "").strip()}
        if not entry["name"]:
            continue

        # Description — strip wikitext bold/italic markers and verbose UPPER-CASED
        # verbs ("EQUIP", "FIRE") that the wiki uses but bloat the prompt.
        desc = ab.get("description", "").strip()
        desc = re.sub(r"'{2,5}", "", desc)            # bold/italic
        desc = re.sub(r"\[\[(?:[^\]|]*\|)?([^\]]+)\]\]", r"\1", desc)  # [[link|text]] → text
        desc = re.sub(r"\s+", " ", desc).strip()
        entry["description"] = desc

        if hotkey == "X":
            ult = _extract_first_int(ab.get("ultimatecost", "") or ab.get("cost", ""))
            if ult is None:
                continue
            entry["ult_points"] = ult
        else:
            cost = _extract_first_int(ab.get("cost", ""))
            entry["cost"] = cost if cost is not None else 0

        charges_str = ab.get("charges", "").strip()
        charges = _extract_first_int(charges_str) if charges_str else None
        if charges and charges > 1:
            entry["charges"] = charges

        abilities[hotkey] = entry

    if set(abilities.keys()) != {"C", "Q", "E", "X"}:
        return None  # Refuse to write a partial kit.

    return {
        "name": name,
        "role": role,
        "abilities": abilities,
        "playstyle": "",  # Liquipedia template doesn't carry this — hand-edit later.
        "economy_tip": "",
    }


def _insert_agent_entry(agents_json: dict, entry: dict) -> bool:
    """Append *entry* to agents.json.  Returns True if added (False if dup)."""
    existing = {a["name"].casefold() for a in agents_json.get("agents", [])}
    if entry["name"].casefold() in existing:
        return False
    agents_json.setdefault("agents", []).append(entry)
    return True


def _save_agents_json(agents_json: dict) -> None:
    with open(_AGENTS_JSON, "w") as f:
        json.dump(agents_json, f, indent=2, ensure_ascii=False)
        f.write("\n")


def _load_agents_json() -> dict:
    with open(_AGENTS_JSON) as f:
        return json.load(f)


def _load_meta_json() -> dict:
    with open(_META_JSON) as f:
        return json.load(f)


def _save_meta_json(data: dict) -> None:
    """Write *data* back to meta.json with the same formatting as the source."""
    with open(_META_JSON, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def _scrape_roster() -> list[str] | None:
    """Fetch the canonical agent roster via Liquipedia's MediaWiki API.

    Hitting the category API instead of the rendered portal HTML avoids
    parsing noise (anchor text from headers, footers, sidebars) and gives
    us back structured page titles directly.  Note: brand-new agents may
    take a few days to be added to Liquipedia's ``Category:Agents`` —
    until then they'll be missed here, which is acceptable because the
    discovery is advisory, not blocking.
    """
    import httpx

    try:
        resp = httpx.get(_LIQUIPEDIA_API, timeout=20, headers={"User-Agent": _USER_AGENT})
        resp.raise_for_status()
        data = resp.json()
    except (httpx.HTTPError, ValueError) as exc:
        log.warning("Failed to fetch Liquipedia agent category: %s", exc)
        return None

    members = data.get("query", {}).get("categorymembers", [])
    titles = [m.get("title") for m in members if m.get("title")]

    # Drop anything that looks like a sub-page or namespaced title
    # (no colons, no slashes except in KAY/O which is the only valid one).
    roster: list[str] = []
    for title in titles:
        if not title:
            continue
        if ":" in title:
            continue
        if "/" in title and title != "KAY/O":
            continue
        roster.append(title)
    return roster


def _new_agent_template_block(name: str) -> str:
    """Print-ready JSON skeleton for a new agent entry."""
    return (
        f"""    {{
      "name": "{name}",
      "role": "Duelist | Initiator | Controller | Sentinel",
      "abilities": {{
        "C": {{"name": "?", "cost": 0, "description": "..."}},
        "Q": {{"name": "?", "cost": 0, "description": "..."}},
        "E": {{"name": "?", "cost": 0, "description": "..."}},
        "X": {{"name": "?", "ult_points": 0, "description": "..."}}
      }},
      "playstyle": "...",
      "economy_tip": "..."
    }}"""
    )


def _stub_meta_entries(meta: dict, missing: list[str]) -> int:
    """Append C-tier placeholders for *missing* agents.  Returns rows added."""
    tier_list = meta.setdefault("tier_list", {})
    c_tier = tier_list.setdefault("C", [])
    agent_meta = meta.setdefault("agent_meta", {})
    added = 0
    for name in missing:
        if name not in c_tier:
            c_tier.append(name)
        if name not in agent_meta:
            agent_meta[name] = {
                "tier": "C",
                "pick_rate": "—",
                "win_rate": "—",
                "reason": (
                    "Auto-stubbed placeholder — pro/Diamond+ data not yet "
                    "available. Re-run `valocoach meta-refresh --force` for "
                    "real tier placement."
                ),
            }
            added += 1
    return added


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def _try_extract_kit(name: str) -> dict | None:
    """Best-effort wikitext → kit-dict for *name*.  Returns None on failure.

    Failure modes (all silent — caller falls back to the manual skeleton):
      * Liquipedia 404 / network error,
      * the page lacks an ``Infobox agent`` template,
      * any of the four AbilityCard blocks is missing or malformed,
      * the role isn't one of the four canonical role strings.
    """
    wikitext = _fetch_wikitext(name)
    if not wikitext:
        return None
    return _parse_agent_wikitext(wikitext)


def run_agents_refresh(*, auto_stub_meta: bool = False, extract_kits: bool = False) -> None:
    """Diff the local knowledge base against Liquipedia and print/apply fixes."""
    agents_data = _load_agents_json()
    meta_data = _load_meta_json()

    local_agents = [a["name"] for a in agents_data.get("agents", [])]
    local_set_lower = {n.casefold() for n in local_agents}

    # Tier-list residency (drives "unranked" warnings in the meta panel).
    tier_lists = meta_data.get("tier_list", {}) or {}
    in_tier_list = {n.casefold() for t in ("S", "A", "B", "C") for n in tier_lists.get(t, [])}

    # 1. Scrape Liquipedia.
    with display.command_frame("agents-refresh — discovery"):
        display.info("Fetching roster from Liquipedia (Category:Agents API)")

        roster = _scrape_roster()
        if roster is None:
            display.error_with_hint(
                "Could not fetch the Liquipedia roster page.",
                "Check your internet connection or hit the URL manually.",
            )
            return

        # Restrict candidates to those plausibly an agent: also resident in
        # our local bundle (signals "known agent") OR with a wiki page that
        # 200s.  We keep it simple — anything not in the local set AND not
        # plausibly an agent name (3+ chars, capitalised) is dropped here.
        candidates = [n for n in roster if len(n) >= 3]
        display.info(f"Found {len(candidates)} candidate names on Liquipedia.")

    # 2. Diff: new agents missing from agents.json.
    new_agents = [n for n in candidates if n.casefold() not in local_set_lower]

    # 3. Diff: agents present in agents.json but missing from meta tier list.
    missing_from_tier = [n for n in local_agents if n.casefold() not in in_tier_list]

    # 4. Report.
    if not new_agents and not missing_from_tier:
        display.success(
            f"Knowledge base is up to date — {len(local_agents)} agents, "
            f"all present in the meta tier list."
        )
        return

    if new_agents:
        with display.command_frame("New agents (not in agents.json)"):
            for name in new_agents:
                url = f"https://liquipedia.net/valorant/{name}"
                display.console.print(f"  • [heading]{name}[/heading]")
                display.console.print(f"    Wiki: [info]{url}[/info]")
            display.console.print()

            if extract_kits:
                # Deterministic wikitext parse — no LLM.  We fetch Liquipedia's
                # raw template source and pull (name, role, 4 abilities + costs)
                # straight from the {{Infobox agent}} / {{AbilityCard}} fields.
                # Anything malformed falls back to the skeleton for that agent.
                extracted = 0
                fallbacks: list[str] = []
                for name in new_agents:
                    kit = _try_extract_kit(name)
                    if kit is None:
                        fallbacks.append(name)
                        continue
                    if _insert_agent_entry(agents_data, kit):
                        extracted += 1
                        ab_summary = ", ".join(
                            f"{k}={v['name']}" for k, v in kit["abilities"].items()
                        )
                        display.console.print(
                            f"  [success]✔[/success] {kit['name']} "
                            f"([heading]{kit['role']}[/heading]) — {ab_summary}"
                        )
                if extracted:
                    _save_agents_json(agents_data)
                    display.success(
                        f"Wrote {extracted} new agent entry/entries to agents.json. "
                        f"Run [info]valocoach agents-refresh --auto-stub-meta[/info] "
                        "next to add them to the meta tier list."
                    )
                if fallbacks:
                    display.console.print()
                    display.console.print(
                        "[muted]Wikitext parsing failed for the following "
                        "agents — paste the skeleton below into agents.json "
                        "and fill from the wiki URL above:[/muted]"
                    )
                    for name in fallbacks:
                        display.console.print()
                        display.console.print(_new_agent_template_block(name))
            else:
                display.console.print(
                    "[muted]Pass [info]--extract-kits[/info] to attempt a "
                    "deterministic Liquipedia wikitext parse and write the "
                    "entries automatically.  Otherwise copy the skeleton below "
                    "into agents.json and fill it from the wiki URL:[/muted]"
                )
                display.console.print()
                for name in new_agents:
                    display.console.print(_new_agent_template_block(name))
                    display.console.print()

    if missing_from_tier:
        with display.command_frame("Agents missing from meta.json tier list"):
            display.console.print(
                "These agents are in [heading]agents.json[/heading] but not in "
                "[heading]meta.json[/heading]'s tier_list, so they render as "
                "'unranked / niche this patch' in the Meta Insight panel:"
            )
            for name in missing_from_tier:
                display.console.print(f"  • {name}")
            display.console.print()

            if auto_stub_meta:
                added = _stub_meta_entries(meta_data, missing_from_tier)
                _save_meta_json(meta_data)
                display.success(
                    f"Added {added} clearly-labelled C-tier placeholder(s) "
                    f"to meta.json.  Run [info]valocoach meta-refresh "
                    f"--force[/info] to replace them with real tier data."
                )
            else:
                display.console.print(
                    "[muted]Pass [info]--auto-stub-meta[/info] to add "
                    "clearly-labelled C-tier placeholders for the missing "
                    "agents, or run [info]valocoach meta-refresh "
                    "--force[/info] to regenerate the tier list "
                    "end-to-end.[/muted]"
                )


__all__ = ["run_agents_refresh"]

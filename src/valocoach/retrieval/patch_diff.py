"""Patch-note diff extractor (Phase C3).

Parses scraped patch-notes text and writes a structured JSON file to
``{data_dir}/patch_changes/{version}.json``.  The result is consumed by
:func:`~valocoach.retrieval.meta.format_meta_context` (C4) to inject
*what actually changed* into the grounded context block when the coach is
answering a question about a recently buffed/nerfed agent or modified map.

Extraction strategy
-------------------
1. **Regex pass** — the primary path.  Patch notes on playvalorant.com follow
   a loose but recognisable structure:
   - Agent sections begin with the agent name as a heading (``## Jett``,
     ``### JETT``, or an all-caps line like ``JETT``).
   - Map sections have ``MAP UPDATES`` / ``MAPS`` headers.
   - Changes are expressed as bullet points (``-``, ``•``, ``*``) under each
     heading.
   - Buff keywords: decreased, reduced, lowered, buffed, improved, restored,
     reverted.
   - Nerf keywords: increased, longer, slower, nerfed, removed, higher cost.

2. **LLM fallback** — if the regex yields no findings for a given patch, a
   short LLM extraction prompt is run.  The response is a JSON object with
   the same schema.  This handles unusual patch-note formatting and
   future Riot layout changes.

3. **Unknown** — if both passes yield nothing (e.g. empty patch notes), an
   empty-changes dict is written so the caller never needs to check for file
   absence.

Output schema
-------------
``{data_dir}/patch_changes/{version}.json``::

    {
      "patch": "10.09",
      "agents": {
        "Jett": [
          {
            "ability": "Tailwind",
            "change_type": "nerf",
            "description": "Equip time increased from 0.75s to 0.85s."
          }
        ]
      },
      "maps": {
        "Bind": [
          {
            "change_type": "adjust",
            "description": "A Site plant area boundary adjusted near A Showers."
          }
        ]
      }
    }

``change_type`` values: ``"buff"``, ``"nerf"``, ``"adjust"``, ``"rework"``.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Known entity lists for section detection
# ---------------------------------------------------------------------------

_AGENT_NAMES: frozenset[str] = frozenset({
    "Astra", "Breach", "Brimstone", "Chamber", "Clove", "Cypher", "Deadlock",
    "Fade", "Gekko", "Harbor", "Iso", "Jett", "KAY/O", "Killjoy", "Miks",
    "Neon", "Omen", "Phoenix", "Raze", "Reyna", "Sage", "Skye", "Sova",
    "Tejo", "Viper", "Vyse", "Waylay", "Yoru",
})

_MAP_NAMES: frozenset[str] = frozenset({
    "Abyss", "Ascent", "Bind", "Haven", "Icebox", "Lotus", "Pearl", "Split", "Sunset",
})

_BUFF_KEYWORDS = re.compile(
    r"\b(?:decreased?|reduced?|lowered?|buff(?:ed)?|improved?|restored?|reverted?|"
    r"faster|quicker|shorter|cheaper|more|increased?\s+range|increased?\s+radius)\b",
    re.IGNORECASE,
)

_NERF_KEYWORDS = re.compile(
    r"\b(?:increased?|longer|slower|nerf(?:ed)?|removed?|higher\s+cost|"
    r"reduced?\s+range|reduced?\s+radius|now\s+costs?\s+more|increased?\s+(?:duration|time|delay))\b",
    re.IGNORECASE,
)

# A heading in the patch notes — any line that looks like a section opener.
# Matches:  "## JETT", "### Jett", "JETT", "Jett", "**Jett**"
_HEADING_RE = re.compile(
    r"^(?:#{1,4}\s*|\*{1,2})?([A-Z][A-Za-z/\-']+(?:\s[A-Z][A-Za-z/\-']+)*)(?:\*{1,2})?(?:\s+(?:UPDATES?|CHANGES?))?$",
    re.MULTILINE,
)

_BULLET_RE = re.compile(r"^\s*[-•*]\s+(.+)$", re.MULTILINE)

_ABILITY_RE = re.compile(
    r"\*{0,2}([A-Z][A-Za-z\s/]{2,25})\s*(?:\|[^*\n]+)?\*{0,2}\s*$",
    re.MULTILINE,
)


# ---------------------------------------------------------------------------
# Change-type classifier
# ---------------------------------------------------------------------------


def _classify_change(text: str) -> str:
    """Heuristically classify a change description as buff/nerf/adjust/rework."""
    nerf_score = len(_NERF_KEYWORDS.findall(text))
    buff_score = len(_BUFF_KEYWORDS.findall(text))

    if "rework" in text.lower() or "redesign" in text.lower():
        return "rework"
    if nerf_score > buff_score:
        return "nerf"
    if buff_score > nerf_score:
        return "buff"
    return "adjust"


# ---------------------------------------------------------------------------
# Regex extraction
# ---------------------------------------------------------------------------


def _regex_extract(text: str) -> tuple[dict[str, list[dict]], dict[str, list[dict]]]:
    """Extract agent and map changes using regex heuristics.

    Returns ``(agents_dict, maps_dict)`` — both may be empty if the patch
    notes format is unusual.
    """
    agents: dict[str, list[dict]] = {}
    maps: dict[str, list[dict]] = {}

    # Split into lines for section detection.
    lines = text.splitlines()

    current_entity: str | None = None
    entity_type: str | None = None  # "agent" or "map"
    current_ability: str | None = None
    buffer: list[str] = []

    def _flush():
        nonlocal buffer
        if not current_entity or not buffer:
            buffer = []
            return
        combined = " ".join(buffer).strip()
        if not combined:
            buffer = []
            return
        entry = {
            "ability": current_ability,
            "change_type": _classify_change(combined),
            "description": combined,
        }
        # Drop None ability key from map entries
        if entity_type == "map":
            entry.pop("ability", None)
        if entity_type == "agent":
            agents.setdefault(current_entity, []).append(entry)
        elif entity_type == "map":
            maps.setdefault(current_entity, []).append(entry)
        buffer = []

    for line in lines:
        stripped = line.strip()

        # Skip completely empty lines (but flush buffered bullets first)
        if not stripped:
            _flush()
            current_ability = None
            continue

        # Check for entity heading
        m = _HEADING_RE.match(stripped)
        if m:
            candidate = m.group(1).strip()

            # Check agent (case-insensitive)
            found_agent = next(
                (a for a in _AGENT_NAMES if a.upper() == candidate.upper()), None
            )
            found_map = next(
                (mp for mp in _MAP_NAMES if mp.upper() == candidate.upper()), None
            )

            if found_agent:
                _flush()
                current_entity = found_agent
                entity_type = "agent"
                current_ability = None
                continue
            if found_map:
                _flush()
                current_entity = found_map
                entity_type = "map"
                current_ability = None
                continue

            # Check for "AGENT UPDATES" / "MAP UPDATES" section headers — reset context
            upper = candidate.upper()
            if "AGENT" in upper and "UPDATE" in upper:
                _flush()
                current_entity = None
                entity_type = None
                current_ability = None
                continue
            if "MAP" in upper and "UPDATE" in upper:
                _flush()
                current_entity = None
                entity_type = None
                current_ability = None
                continue

        if current_entity is None:
            continue

        # Ability sub-heading (e.g. "**TAILWIND | E**")
        am = _ABILITY_RE.match(stripped)
        if am and entity_type == "agent":
            _flush()
            current_ability = am.group(1).strip().title()
            continue

        # Bullet point
        bm = _BULLET_RE.match(line)
        if bm:
            buffer.append(bm.group(1).strip())

    _flush()
    return agents, maps


# ---------------------------------------------------------------------------
# LLM fallback extraction
# ---------------------------------------------------------------------------

_LLM_EXTRACTION_PROMPT = """\
Extract ALL agent and map changes from the patch notes below.
Return ONLY a JSON object with this exact schema:

{
  "agents": {
    "AgentName": [
      {"ability": "AbilityName or null", "change_type": "buff|nerf|adjust|rework", "description": "..."}
    ]
  },
  "maps": {
    "MapName": [
      {"change_type": "adjust|update", "description": "..."}
    ]
  }
}

Rules:
- Include ONLY agents and maps that are explicitly mentioned as changed.
- "ability" is null for non-ability-specific changes.
- "change_type": "buff" if the change is beneficial, "nerf" if harmful,
  "adjust" if neutral, "rework" if the mechanic fundamentally changed.
- Use the agent's exact in-game name (e.g. "Jett", "KAY/O", "Viper").
- Output ONLY valid JSON — no markdown fences, no commentary.

PATCH NOTES:
"""


def _llm_extract(
    patch_notes_text: str,
    settings: object,
) -> tuple[dict[str, list[dict]], dict[str, list[dict]]]:
    """Fall back to the LLM for patch change extraction."""
    try:
        from valocoach.llm.provider import stream_completion

        user_msg = patch_notes_text[:4_000]  # keep within context budget
        tokens: list[str] = []
        for token in stream_completion(
            settings,
            system_prompt=_LLM_EXTRACTION_PROMPT,
            user_message=user_msg,
        ):
            tokens.append(token)

        raw = "".join(tokens)
        # Strip fences
        raw = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.MULTILINE)
        raw = re.sub(r"\s*```\s*$", "", raw.strip(), flags=re.MULTILINE)
        start = raw.find("{")
        if start == -1:
            return {}, {}
        depth = 0
        end = start
        for i, ch in enumerate(raw[start:], start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i
                    break
        data = json.loads(raw[start : end + 1])
        return data.get("agents", {}), data.get("maps", {})
    except Exception as exc:
        log.debug("LLM patch extraction fallback failed: %s", exc)
        return {}, {}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_patch_changes(
    patch_notes_text: str,
    patch_version: str,
    data_dir: Path,
    settings: object | None = None,
) -> dict:
    """Extract agent/map changes from patch notes and persist to disk.

    The result file lives at
    ``{data_dir}/patch_changes/{patch_version}.json``.

    Args:
        patch_notes_text: Scraped plain text of the patch notes page.
        patch_version:    Clean version string like ``"10.09"``.
        data_dir:         Root data directory (``settings.data_dir``).
        settings:         App settings; if provided and regex yields no
                          results, the LLM fallback is attempted.

    Returns:
        The extracted changes dict (always non-None, may have empty
        ``agents`` / ``maps`` dicts when no changes were found).
    """
    agents, maps = _regex_extract(patch_notes_text)

    if not agents and not maps and settings is not None:
        log.info(
            "Regex extracted no patch changes — trying LLM fallback (patch=%s)",
            patch_version,
        )
        agents, maps = _llm_extract(patch_notes_text, settings)

    result = {
        "patch": patch_version,
        "agents": agents,
        "maps": maps,
    }

    total_agent_changes = sum(len(v) for v in agents.values())
    total_map_changes = sum(len(v) for v in maps.values())
    log.info(
        "Patch %s diff: %d agent change(s) across %d agent(s), "
        "%d map change(s) across %d map(s)",
        patch_version,
        total_agent_changes,
        len(agents),
        total_map_changes,
        len(maps),
    )

    # Persist to disk — create directory if needed.
    out_dir = Path(data_dir) / "patch_changes"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{patch_version}.json"
    try:
        with open(out_file, "w") as f:
            json.dump(result, f, indent=2)
            f.write("\n")
        log.info("Patch changes saved to %s", out_file)
    except OSError as exc:
        log.warning("Could not save patch changes to %s: %s", out_file, exc)

    return result


def load_patch_changes(patch_version: str, data_dir: Path) -> dict | None:
    """Load previously extracted patch changes for *patch_version*.

    Returns ``None`` when no diff file exists for that patch (e.g. meta-refresh
    hasn't been run yet on a new install, or the version predates C3).
    """
    path = Path(data_dir) / "patch_changes" / f"{patch_version}.json"
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        log.debug("Could not load patch changes from %s: %s", path, exc)
        return None


__all__ = ["extract_patch_changes", "load_patch_changes"]

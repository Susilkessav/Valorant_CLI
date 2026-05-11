"""LLM-powered meta tier-list regeneration.

Takes patch notes + pick/win-rate stats scraped from tracker.gg and vlr.gg,
feeds them to the configured LLM, and returns a validated dict that can be
written directly to ``data/meta.json``.

The LLM is instructed to output *only* valid JSON — no markdown fences, no
commentary — so we can ``json.loads()`` the result without post-processing.
A lightweight validation step fills any gaps using the existing meta as a
fallback so a partial LLM response never corrupts the file completely.
"""

from __future__ import annotations

import json
import logging
import re

from valocoach.core.config import Settings

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT = """\
You are a Valorant meta analyst specialising in high-ELO (Diamond+) and \
professional / VCT play.

Your task is to produce an updated agent tier list in strict JSON format \
given:
  1. Official patch notes for the current patch.
  2. Current pick-rate and win-rate data from ranked and pro play.

## OUTPUT FORMAT — output ONLY this JSON, nothing else:

{
  "tier_list": {
    "S": ["AgentName", ...],
    "A": ["AgentName", ...],
    "B": ["AgentName", ...],
    "C": ["AgentName", ...]
  },
  "agent_meta": {
    "AgentName": {
      "tier": "S",
      "pick_rate": "~32%",
      "win_rate": "~51%",
      "reason": "One or two sentences on why this tier, referencing kit strength."
    }
  },
  "map_meta": {
    "MapName": {
      "top_agents": ["Agent1", "Agent2", "Agent3", "Agent4", "Agent5"],
      "notes": "One tactical sentence for this map."
    }
  }
}

## Hard rules:
- Include EVERY agent currently in the game.
- Include ALL maps: Ascent, Bind, Haven, Icebox, Lotus, Pearl, Split, \
Sunset, Abyss.
- Use "~X%" format for pick and win rates (approximate is fine).
- Keep reasons to 1-2 sentences max, focused on ability kit value.
- If stats for an agent are missing, estimate from the patch notes and \
general kit knowledge.
- Output ONLY the JSON object — no markdown fences, no commentary, no \
preamble.
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _strip_fences(text: str) -> str:
    """Remove markdown code fences that some models add despite instructions."""
    text = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.MULTILINE)
    text = re.sub(r"\s*```\s*$", "", text.strip(), flags=re.MULTILINE)
    return text.strip()


def _find_json_object(text: str) -> str:
    """Extract the first top-level ``{…}`` block from ``text``.

    Some models prepend a sentence before the JSON even when told not to.
    We locate the first ``{`` and its matching ``}`` with a simple depth
    counter rather than a regex so nested objects are handled correctly.
    """
    start = text.find("{")
    if start == -1:
        return text
    depth = 0
    for i, ch in enumerate(text[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return text[start:]


def _validate(data: dict, existing: dict) -> dict:
    """Fill gaps in the LLM output using ``existing`` meta as a fallback.

    Ensures the returned dict always has the four required top-level keys
    and that every tier (S/A/B/C) is present in ``tier_list``.
    """
    validated: dict = {
        "tier_list": data.get("tier_list") or existing.get("tier_list", {}),
        "agent_meta": data.get("agent_meta") or existing.get("agent_meta", {}),
        "map_meta": data.get("map_meta") or existing.get("map_meta", {}),
    }

    # Ensure all four tiers exist even if the LLM omitted one.
    for tier in ("S", "A", "B", "C"):
        if tier not in validated["tier_list"]:
            validated["tier_list"][tier] = (
                existing.get("tier_list", {}).get(tier, [])
            )

    return validated


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_meta_update(
    settings: Settings,
    patch_version: str,
    patch_notes_text: str,
    stats_text: str,
    existing_meta: dict,
) -> dict | None:
    """Generate updated meta JSON via the configured LLM.

    The function calls :func:`~valocoach.llm.provider.stream_completion`
    synchronously (collecting the full streamed response) because meta
    generation is a batch operation, not an interactive chat.

    Args:
        settings:          App settings (model name, host, temperature …).
        patch_version:     Clean patch number string, e.g. ``"10.09"``.
        patch_notes_text:  Scraped patch notes text (may be empty).
        stats_text:        Combined ranked + pro stats text (may be empty).
        existing_meta:     Current ``meta.json`` contents used as fallback.

    Returns:
        A validated meta dict ready to be written to ``meta.json``, or
        ``None`` if the LLM call failed or produced unparseable output.
    """
    from valocoach.llm.provider import stream_completion

    # Truncate inputs so we stay within the model's context window.
    # 6 000 chars ≈ ~1 500 tokens each; the system prompt is ~400 tokens.
    notes_snippet = (patch_notes_text or "No patch notes available.")[:6_000]
    stats_snippet = (stats_text or "No stats data available.")[:6_000]

    user_message = (
        f"CURRENT PATCH: {patch_version}\n\n"
        f"=== PATCH NOTES ===\n{notes_snippet}\n\n"
        f"=== AGENT STATS (Diamond+ Ranked + Pro/VCT) ===\n{stats_snippet}\n\n"
        "Generate the complete updated tier list JSON now."
    )

    log.info(
        "Generating meta update via LLM (model=%s, patch=%s)",
        settings.ollama_model,
        patch_version,
    )

    tokens: list[str] = []
    try:
        for token in stream_completion(
            settings,
            system_prompt=_SYSTEM_PROMPT,
            user_message=user_message,
        ):
            tokens.append(token)
    except Exception as exc:
        log.error("LLM call failed during meta generation: %s", exc)
        return None

    raw = "".join(tokens)
    log.debug("LLM raw output: %d chars", len(raw))

    # Try to extract and parse JSON.
    cleaned = _strip_fences(raw)
    json_str = _find_json_object(cleaned)

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as exc:
        log.error(
            "LLM output was not valid JSON: %s\nPreview (200 chars): %.200s",
            exc,
            json_str,
        )
        return None

    return _validate(data, existing_meta)

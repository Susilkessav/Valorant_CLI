"""LLM-powered meta tier-list regeneration.

Takes patch notes + pick/win-rate stats scraped from tracker.gg and vlr.gg,
feeds them to the configured LLM, and returns a validated dict that can be
written directly to ``data/meta.json``.

Phase C2 — Deterministic numeric tier scoring
----------------------------------------------
The LLM is no longer asked to assign S/A/B/C tiers.  Instead it outputs raw
numeric ``pick_rate_pct`` and ``win_rate_pct`` values, which are far more
stable between patches than text tier labels.  A deterministic scorer
(:func:`_tier_score`, :func:`_tier_from_score`) converts those numbers into
tier assignments — same inputs always produce the same tiers, eliminating
"flapping" across refreshes caused by LLM temperature.

The LLM is still responsible for writing the ``reason`` field (short prose
explanation) because textual reasoning is where language models add value.

Fallback behaviour
------------------
If the LLM omits numeric rates for an agent (or the old ~X% string format is
returned), the existing ``existing_meta`` tier assignment is preserved so a
partial LLM response never silently downgrades an agent.
"""

from __future__ import annotations

import json
import logging
import math
import re

from valocoach.core.config import Settings

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# C2 — Deterministic tier scoring
# ---------------------------------------------------------------------------

# Thresholds for score = win_rate_pct + log1p(pick_rate_pct) * 0.5
# Calibrated against Valorant patch history (Diamond+ ranked):
#   52% WR, 30% PR → score ≈ 53.7 → S
#   51% WR, 15% PR → score ≈ 52.4 → A
#   50% WR, 20% PR → score ≈ 51.5 → B
#   48% WR,  5% PR → score ≈ 49.8 → C
_TIER_THRESHOLDS = (
    ("S", 53.5),
    ("A", 51.5),
    ("B", 50.0),
)  # anything below B_threshold → C


def _tier_score(win_rate_pct: float, pick_rate_pct: float) -> float:
    """Deterministic composite score used for S/A/B/C bucketing.

    Win rate is the primary signal (coefficient 1.0); pick rate adds a
    logarithmic bonus so widely-picked agents get a slight lift without
    overwhelming the win-rate signal.
    """
    return win_rate_pct + math.log1p(max(pick_rate_pct, 0.0)) * 0.5


def _tier_from_score(score: float) -> str:
    """Map a composite score to S/A/B/C."""
    for tier, threshold in _TIER_THRESHOLDS:
        if score >= threshold:
            return tier
    return "C"


def _parse_rate(value: object) -> float | None:
    """Parse a rate value to a float percentage.

    Handles:
      - Plain floats / ints: ``32.0`` → ``32.0``
      - LLM "~X%" strings:  ``"~32%"`` → ``32.0``  (fallback tolerance)
      - Missing / None:      ``None``  → ``None``
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        cleaned = value.replace("~", "").replace("%", "").strip()
        try:
            return float(cleaned)
        except ValueError:
            return None
    return None


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a Valorant meta analyst specialising in high-ELO (Diamond+) and \
professional / VCT play.

Your task is to provide pick-rate and win-rate DATA for every agent, plus a \
short reason for each.  Tier assignments (S/A/B/C) will be computed \
automatically — do NOT include them in your output.

## OUTPUT FORMAT — output ONLY this JSON, nothing else:

{
  "agent_meta": {
    "AgentName": {
      "pick_rate_pct": 32.0,
      "win_rate_pct": 51.5,
      "reason": "One or two sentences on why this agent is strong or weak right now."
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
- ``pick_rate_pct`` and ``win_rate_pct`` are PLAIN FLOATS (no "~", no "%").
  Example: 32.0 not "~32%".
- Win rates cluster around 50%.  S-tier agents typically have 51-53% WR.
  Do not exaggerate — rates above 55% are extremely rare.
- Keep reasons to 1-2 sentences, focused on ability kit value and patch changes.
- If exact stats for an agent are missing, estimate conservatively from patch \
notes and kit knowledge.
- Output ONLY the JSON object ��� no markdown fences, no commentary, no \
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


def _compute_tiers(agent_meta_raw: dict, existing_agent_meta: dict) -> tuple[dict, dict]:
    """Apply deterministic tier scoring to LLM-provided rate data.

    Returns ``(tier_list, agent_meta_formatted)`` where ``tier_list`` is the
    ``{"S": [...], "A": [...], ...}`` dict and ``agent_meta_formatted`` has
    the backward-compatible ``pick_rate``, ``win_rate`` (string), and
    ``tier`` fields alongside ``reason``.

    Fallback: agents missing numeric rates keep their tier from
    ``existing_agent_meta`` so a partial LLM response never silently demotes
    an agent.
    """
    tier_list: dict[str, list[str]] = {"S": [], "A": [], "B": [], "C": []}
    formatted: dict[str, dict] = {}

    for agent_name, data in agent_meta_raw.items():
        pr = _parse_rate(data.get("pick_rate_pct") or data.get("pick_rate"))
        wr = _parse_rate(data.get("win_rate_pct") or data.get("win_rate"))
        reason = data.get("reason", "")

        if pr is not None and wr is not None:
            score = _tier_score(wr, pr)
            tier = _tier_from_score(score)
            pr_str = f"~{pr:.0f}%"
            wr_str = f"~{wr:.0f}%"
        else:
            # Fallback to existing meta's tier + rates (may be string format)
            existing = existing_agent_meta.get(agent_name, {})
            tier = existing.get("tier", "C")
            pr_str = existing.get("pick_rate", "N/A")
            wr_str = existing.get("win_rate", "N/A")
            log.debug("Agent %r missing numeric rates — keeping existing tier %r", agent_name, tier)

        tier_list.setdefault(tier, []).append(agent_name)
        formatted[agent_name] = {
            "tier": tier,
            "pick_rate": pr_str,
            "win_rate": wr_str,
            "reason": reason,
        }

    # Ensure all four tiers exist even if empty.
    for t in ("S", "A", "B", "C"):
        tier_list.setdefault(t, [])

    return tier_list, formatted


def _coerce_agent_meta(raw: object) -> dict:
    """Normalise LLM agent_meta to ``{AgentName: {...}}`` regardless of output format.

    The LLM occasionally returns a list of objects instead of a keyed dict,
    e.g. ``[{"name": "Sova", "pick_rate_pct": 32.0, ...}, ...]``.
    We detect this and re-key by the name/agent field so the rest of the
    pipeline always receives a plain dict.
    """
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, list):
        coerced: dict = {}
        for i, item in enumerate(raw):
            if not isinstance(item, dict):
                continue
            key = item.get("name") or item.get("agent") or item.get("agent_name") or str(i)
            entry = {k: v for k, v in item.items() if k not in ("name", "agent", "agent_name")}
            coerced[str(key)] = entry
        log.warning("LLM returned agent_meta as a list (%d entries) — coerced to dict", len(raw))
        return coerced
    log.warning("LLM returned agent_meta of unexpected type %s — ignoring", type(raw).__name__)
    return {}


def _coerce_map_meta(raw: object) -> dict:
    """Normalise LLM map_meta to ``{MapName: {...}}`` regardless of output format."""
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, list):
        coerced: dict = {}
        for i, item in enumerate(raw):
            if not isinstance(item, dict):
                continue
            key = item.get("map") or item.get("name") or item.get("map_name") or str(i)
            entry = {k: v for k, v in item.items() if k not in ("map", "name", "map_name")}
            coerced[str(key)] = entry
        log.warning("LLM returned map_meta as a list (%d entries) — coerced to dict", len(raw))
        return coerced
    return {}


def _validate(data: dict, existing: dict) -> dict:
    """Merge LLM output with existing meta, computing deterministic tiers (C2).

    If the LLM returned the new-format ``agent_meta`` with numeric rates,
    ``_compute_tiers()`` is called to derive ``tier_list`` deterministically.
    If the LLM returned the old format (tier strings, no numerics), we fall
    back to existing tier_list so old responses never corrupt the file.
    """
    existing_agent_meta = existing.get("agent_meta", {})
    # Normalise before touching .values() — LLM sometimes returns a list
    raw_agent_meta = _coerce_agent_meta(data.get("agent_meta", {}))

    if raw_agent_meta:
        # Detect whether ANY agent has numeric rate data.
        has_numeric = any(
            isinstance(v.get("pick_rate_pct"), (int, float))
            or isinstance(v.get("win_rate_pct"), (int, float))
            or (isinstance(v.get("pick_rate_pct"), str) and "%" not in v["pick_rate_pct"])
            for v in raw_agent_meta.values()
        )

        if has_numeric:
            tier_list, agent_meta = _compute_tiers(raw_agent_meta, existing_agent_meta)
            log.info(
                "C2: deterministic tiers computed — S:%d A:%d B:%d C:%d",
                len(tier_list["S"]),
                len(tier_list["A"]),
                len(tier_list["B"]),
                len(tier_list["C"]),
            )
        else:
            # Old format or fully missing numeric data — preserve existing tiers.
            tier_list = existing.get("tier_list", {"S": [], "A": [], "B": [], "C": []})
            agent_meta = raw_agent_meta
            log.warning(
                "C2: LLM output missing numeric pick/win rates — "
                "keeping existing tier_list as fallback."
            )
    else:
        tier_list = existing.get("tier_list", {"S": [], "A": [], "B": [], "C": []})
        agent_meta = existing_agent_meta

    # Ensure all four tiers exist even if the LLM omitted one.
    for tier in ("S", "A", "B", "C"):
        if tier not in tier_list:
            tier_list[tier] = existing.get("tier_list", {}).get(tier, [])

    raw_map_meta = _coerce_map_meta(data.get("map_meta"))
    return {
        "tier_list": tier_list,
        "agent_meta": agent_meta,
        "map_meta": raw_map_meta or existing.get("map_meta", {}),
    }


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

    # Truncate inputs to stay well within typical local-model context budgets.
    # ~3 000 chars ≈ 750 tokens each; system prompt ≈ 500 tokens; leaves
    # ~14 000 tokens of headroom for the response in a 16 384-token window.
    notes_snippet = (patch_notes_text or "No patch notes available.")[:3_000]
    stats_snippet = (stats_text or "No stats data available.")[:3_000]

    user_message = (
        f"CURRENT PATCH: {patch_version}\n\n"
        f"=== PATCH NOTES ===\n{notes_snippet}\n\n"
        f"=== AGENT STATS (Diamond+ Ranked + Pro/VCT) ===\n{stats_snippet}\n\n"
        "Generate the complete agent meta JSON now "
        "(pick_rate_pct and win_rate_pct as plain floats, no tier assignments)."
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
            # Meta generation needs a large output — the full agent+map JSON
            # for 28 agents easily exceeds the default 3 000-token cap.
            max_tokens=6_000,
            # Tell Ollama to use a 16 384-token context window so the input
            # prompt (system + notes + stats ≈ 2 000 tokens) doesn't crowd
            # out the model's output budget.
            num_ctx=16_384,
            # Disable the reasoning phase.  qwen3 (and other thinking models)
            # otherwise burn the entire 6 000-token budget on reasoning_content
            # before producing any JSON, so the provider — which only yields
            # ``content`` — collects an empty string and meta.json is never
            # updated.  This is a pure structured-extraction call, so the
            # <think> phase adds no value here anyway.
            think=False,
        ):
            tokens.append(token)
    except Exception as exc:
        log.error("LLM call failed during meta generation: %s", exc)
        return None

    raw = "".join(tokens)
    log.debug("LLM raw output: %d chars", len(raw))

    if not raw.strip():
        log.error(
            "LLM returned an empty response for meta generation "
            "(model=%s) — context window may be too small or the model "
            "timed out.  Keeping existing meta.json unchanged.",
            settings.ollama_model,
        )
        return None

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

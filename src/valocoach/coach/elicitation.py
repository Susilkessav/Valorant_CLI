"""Interactive elicitation — fills missing Situation fields before coaching.

When a user types a short or underspecified coaching query, this module
intercepts before the LLM call and asks a short sequence of targeted questions.
The enriched Situation then flows through the normal context-building pipeline.

Phase 2 — Intent-aware clarification
--------------------------------------
Elicitation now only asks about fields that are *relevant for the classified
intent*.  A "meta" question never triggers any prompts; an "economy" question
only asks for side (and optionally score); a "tactical" question asks for map,
side, and agent — but never score or phase.

At most ``MAX_QUESTIONS = 3`` questions are asked per turn.  Fields already
present in the parsed Situation or supplied via slash commands are never
re-asked.

Public API
----------
    FIELDS_BY_INTENT               dict mapping intent → relevant field names
    should_elicit(situation, raw, intent)  -> bool
    run_elicitation(parsed, agent, map_, side, *, intent) -> (Situation, agent, map_, side)
"""

from __future__ import annotations

import re
import sys
from difflib import get_close_matches

from valocoach.core.parser import Situation

# ---------------------------------------------------------------------------
# Intent → relevant fields mapping
# ---------------------------------------------------------------------------

FIELDS_BY_INTENT: dict[str, tuple[str, ...]] = {
    "tactical":      ("map", "side", "agent"),
    "clutch":        ("side", "agent"),
    "post_plant":    ("map", "agent", "side"),
    "retake":        ("map", "agent", "side"),
    "economy":       ("side", "score"),
    "agent_info":    ("agent",),
    "meta":          (),          # deterministic — never ask anything
    "stat_analysis": (),          # pulls from DB — no user input needed
    "post_game":     (),          # pre-populated by the post-game pipeline
    "general":       ("agent", "map"),
}

MAX_QUESTIONS = 3

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MAP_NAMES: tuple[str, ...] = (
    "Abyss",
    "Ascent",
    "Bind",
    "Haven",
    "Icebox",
    "Lotus",
    "Pearl",
    "Split",
    "Sunset",
)


def _agent_names() -> tuple[str, ...]:
    """Return the canonical agent roster from ``agents.json``."""
    from valocoach.retrieval.agents import list_agent_names

    return tuple(list_agent_names())


_AGENT_NAMES: tuple[str, ...] = _agent_names()

_SCORE_PATTERN = re.compile(r"^(\d{1,2})\s*[-–:]\s*(\d{1,2})$")
_CLUTCH_PATTERN = re.compile(r"^([1-5])\s*v\s*([1-5])$", re.IGNORECASE)

_SIDE_MAP: dict[str, str] = {
    "attack": "attack",
    "atk": "attack",
    "a": "attack",
    "offense": "attack",
    "defense": "defense",
    "def": "defense",
    "d": "defense",
}

_PHASE_MAP: dict[str, str] = {
    "post_plant": "post_plant",
    "post plant": "post_plant",
    "postplant": "post_plant",
    "pp": "post_plant",
    "retake": "retake",
    "ret": "retake",
    "execute": "execute",
    "exec": "execute",
    "ex": "execute",
    "default": "default",
    "def": "default",
}

_ECON_MAP: dict[str, str] = {
    "eco": "eco",
    "e": "eco",
    "half": "half_buy",
    "half_buy": "half_buy",
    "half buy": "half_buy",
    "force": "half_buy",
    "full": "full_buy",
    "full_buy": "full_buy",
    "full buy": "full_buy",
    "f": "full_buy",
}

_SITE_MAP: dict[str, str] = {"a": "A", "b": "B", "c": "C"}


# ---------------------------------------------------------------------------
# Trigger
# ---------------------------------------------------------------------------


def should_elicit(situation: Situation, raw_text: str, intent: str = "general") -> bool:
    """Return True when the situation is missing fields relevant to *intent*.

    - Never prompts when stdin is not a TTY (pipes, CI, tests).
    - Never prompts for intents with no required fields (meta, stat_analysis, post_game).
    - Only prompts when at least one intent-relevant field is still None.
    """
    if not sys.stdin.isatty():
        return False

    relevant = FIELDS_BY_INTENT.get(intent, ("agent", "map", "side"))
    if not relevant:
        return False

    field_values: dict[str, object] = {
        "map":   situation.map,
        "side":  situation.side,
        "agent": situation.primary_agent,
        "score": situation.score,
    }
    missing = [f for f in relevant if field_values.get(f) is None]
    return bool(missing)


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------


def _match_map(answer: str) -> str | None:
    """Return the canonical map name for *answer*, or None."""
    ans = answer.strip().title()
    if ans in _MAP_NAMES:
        return ans
    hits = get_close_matches(ans, _MAP_NAMES, n=1, cutoff=0.5)
    return hits[0] if hits else None


def _match_agent(answer: str) -> str | None:
    """Return the canonical agent name for *answer*, or None."""
    ans = answer.strip()
    for name in _AGENT_NAMES:
        if ans.lower() == name.lower():
            return name
    if ans.upper() in ("KAYO", "KAY/O"):
        return "KAY/O"
    hits = get_close_matches(ans.title(), _AGENT_NAMES, n=1, cutoff=0.55)
    return hits[0] if hits else None


def _match_score(answer: str) -> tuple[int, int] | None:
    m = _SCORE_PATTERN.match(answer.strip())
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        if 0 <= a <= 30 and 0 <= b <= 30:
            return (a, b)
    return None


def _match_clutch(answer: str) -> tuple[int, int] | None:
    m = _CLUTCH_PATTERN.match(answer.strip())
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        if a != b:
            return (a, b)
    return None


# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------


def _ask(prompt: str, *, optional: bool = False) -> str:
    """Print *prompt* and return stripped user input."""
    from valocoach.cli import display

    display.console.print(f"  [info]»[/info] [bold]{prompt}[/bold]", end=" ")
    try:
        answer = input().strip()
    except (EOFError, KeyboardInterrupt):
        answer = ""
    return answer


def _print_header() -> None:
    from valocoach.cli import display

    display.console.print()
    display.console.rule("[heading]Context needed[/heading]", style="dim")
    display.console.print(
        "  [muted]A quick question or two for better advice  "
        "(Enter to skip)[/muted]"
    )
    display.console.print()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_elicitation(
    parsed: Situation,
    agent: str | None,
    map_: str | None,
    side: str | None,
    *,
    intent: str = "general",
) -> tuple[Situation, str | None, str | None, str | None]:
    """Ask targeted questions for fields relevant to *intent*.

    At most ``MAX_QUESTIONS`` questions are asked.  Fields already resolved
    (from the parser or CLI flags) are never re-asked.  The original
    ``Situation.raw`` is preserved unchanged.

    Returns an enriched ``(situation, agent, map_, side)`` tuple.
    """
    from valocoach.cli import display

    # Start from what we already know
    elicited_map   = map_  or parsed.map
    elicited_side  = side  or parsed.side
    elicited_agent = agent or parsed.primary_agent
    elicited_score = parsed.score

    relevant = FIELDS_BY_INTENT.get(intent, ("agent", "map", "side"))
    if not relevant:
        return parsed, agent, map_, side

    # Determine which relevant fields are still missing
    still_missing = []
    field_values: dict[str, object] = {
        "map":   elicited_map,
        "side":  elicited_side,
        "agent": elicited_agent,
        "score": elicited_score,
    }
    for f in relevant:
        if field_values.get(f) is None:
            still_missing.append(f)

    if not still_missing:
        return parsed, agent, map_, side

    _print_header()

    questions_asked = 0
    maps_hint = "/".join(_MAP_NAMES)

    for field in still_missing:
        if questions_asked >= MAX_QUESTIONS:
            break

        if field == "map":
            ans = _ask(f"Map? [{maps_hint}]:")
            if ans:
                resolved = _match_map(ans)
                if resolved:
                    elicited_map = resolved
                else:
                    display.warn(f"Map '{ans}' not recognised — proceeding without map context.")
            questions_asked += 1

        elif field == "side":
            ans = _ask("Side? [attack/defense]:")
            elicited_side = _SIDE_MAP.get(ans.lower().strip()) if ans else None
            questions_asked += 1

        elif field == "agent":
            ans = _ask("Your agent? (Enter to skip):")
            if ans:
                resolved = _match_agent(ans)
                if resolved:
                    elicited_agent = resolved
                else:
                    display.warn(f"Agent '{ans}' not recognised — proceeding without agent context.")
            questions_asked += 1

        elif field == "score":
            ans = _ask("Score? e.g. 8-12  (Enter to skip):")
            if ans:
                elicited_score = _match_score(ans)
            questions_asked += 1

    display.console.print()

    # Build enriched agents list — prepend elicited agent if new
    agents = list(parsed.agents)
    if elicited_agent and elicited_agent not in agents:
        agents = [elicited_agent, *agents]

    enriched = parsed.model_copy(
        update={
            "agents": agents,
            "map":    elicited_map,
            "side":   elicited_side,
            "score":  elicited_score,
        }
    )

    return enriched, elicited_agent, elicited_map, elicited_side


__all__ = [
    "FIELDS_BY_INTENT",
    "MAX_QUESTIONS",
    "run_elicitation",
    "should_elicit",
]

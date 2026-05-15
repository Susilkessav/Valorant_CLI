"""Interactive elicitation — fills missing Situation fields before coaching.

When a user types a short or underspecified coaching query (e.g. "help on Haven"),
this module intercepts *before* the LLM call and asks a short sequence of
targeted questions.  The enriched Situation then flows through the normal
context-building pipeline so the coach gets map, side, agent, and optional
round details without the user needing to remember CLI flags.

Design principles
-----------------
- **Supplement, never replace** — the regex parser runs first; elicitation
  only asks about fields that are still ``None`` after parsing.
- **Fast and optional** — questions 1-3 are the critical group (map, side,
  agent); questions 4-8 are soft-optional (Enter skips them).  Total wall
  time < 1 s assuming the user types quickly.
- **No LLM, no DB** — pure CLI prompts using ``input()``.  Gracefully no-ops
  if stdin is not a TTY (non-interactive pipes).
- **Matches existing parser output** — elicited values are normalized to the
  same canonical forms the regex parser uses so downstream routing is identical.

Public API
----------
    should_elicit(situation, raw_text)      -> bool
    run_elicitation(parsed, agent, map_, side) -> tuple[Situation, str|None, str|None, str|None]
"""

from __future__ import annotations

import re
import sys
from difflib import get_close_matches

from valocoach.core.parser import Situation

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

_AGENT_NAMES: tuple[str, ...] = (
    "Astra", "Breach", "Brimstone", "Chamber", "Clove", "Cypher",
    "Deadlock", "Fade", "Gekko", "Harbor", "Iso", "Jett", "KAY/O",
    "Killjoy", "Miks", "Neon", "Omen", "Phoenix", "Raze", "Reyna",
    "Sage", "Skye", "Sova", "Tejo", "Veto", "Viper", "Vyse", "Waylay", "Yoru",
)

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


def should_elicit(situation: Situation, raw_text: str) -> bool:
    """Return True when the situation is underspecified enough to warrant questions.

    Elicitation fires when:
    - Two or more critical fields (map, side, primary_agent) are missing, OR
    - One critical field is missing AND the raw input is short (< 60 chars).

    This keeps elicitation out of long, descriptive queries where the user
    clearly gave plenty of context despite not naming every field explicitly.
    """
    if not sys.stdin.isatty():
        # Never prompt in non-interactive contexts (pipes, CI, tests).
        return False

    missing = sum([
        situation.map is None,
        situation.side is None,
        situation.primary_agent is None,
    ])
    return missing >= 2 or (missing >= 1 and len(raw_text.strip()) < 60)


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------


def _match_map(answer: str) -> str | None:
    """Return the canonical map name for *answer*, or None."""
    ans = answer.strip().title()
    if ans in _MAP_NAMES:
        return ans
    # Fuzzy match — "asc" → "Ascent", "ice" → "Icebox"
    hits = get_close_matches(ans, _MAP_NAMES, n=1, cutoff=0.5)
    return hits[0] if hits else None


def _match_agent(answer: str) -> str | None:
    """Return the canonical agent name for *answer*, or None."""
    ans = answer.strip()
    # Exact case-insensitive
    for name in _AGENT_NAMES:
        if ans.lower() == name.lower():
            return name
    # KAY/O alias
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
    """Print *prompt* and return stripped user input.

    Returns an empty string when the user just presses Enter (only valid when
    ``optional=True`` — callers treat empty as "skip").
    """
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
        "  [muted]A few quick questions for better advice  "
        "(Enter to skip optional ones)[/muted]"
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
) -> tuple[Situation, str | None, str | None, str | None]:
    """Ask targeted questions to fill missing Situation fields.

    Returns an enriched ``(situation, agent, map_, side)`` tuple with all
    elicited values merged in.  Fields already present from parsing are never
    overwritten.  The original ``Situation.raw`` is preserved unchanged.
    """
    from valocoach.cli import display

    _print_header()

    # Resolved values — start from what the parser already found.
    elicited_map = map_ or parsed.map
    elicited_side = side or parsed.side
    elicited_agent = agent or parsed.primary_agent
    elicited_score = parsed.score
    elicited_phase = parsed.phase
    elicited_site = parsed.site
    elicited_econ = parsed.econ
    elicited_clutch = parsed.clutch

    maps_hint = "/".join(_MAP_NAMES)

    # --- Question 1: Map
    if elicited_map is None:
        ans = _ask(f"Map? [{maps_hint}]:")
        if ans:
            resolved = _match_map(ans)
            if resolved:
                elicited_map = resolved
            else:
                display.warn(f"Map '{ans}' not recognised — coaching will proceed without map context.")

    # --- Question 2: Side
    if elicited_side is None:
        ans = _ask("Side? [attack/defense]:")
        elicited_side = _SIDE_MAP.get(ans.lower().strip()) if ans else None

    # --- Question 3: Agent
    if elicited_agent is None:
        ans = _ask("Your agent?:")
        if ans:
            resolved = _match_agent(ans)
            if resolved:
                elicited_agent = resolved
            else:
                display.warn(f"Agent '{ans}' not recognised — coaching will proceed without agent context.")

    # --- Questions 4-8 are soft-optional (always asked but Enter skips)

    # Question 4: Score
    if elicited_score is None:
        ans = _ask("Score? e.g. 8-12  (Enter to skip):")
        if ans:
            elicited_score = _match_score(ans)

    # Question 5: Phase
    if elicited_phase is None:
        ans = _ask("Phase? [post_plant/retake/execute/default]  (Enter to skip):")
        if ans:
            elicited_phase = _PHASE_MAP.get(ans.lower().strip())

    # Question 6: Site
    if elicited_site is None:
        ans = _ask("Site? [A/B/C]  (Enter to skip):")
        if ans:
            elicited_site = _SITE_MAP.get(ans.upper().strip())

    # Question 7: Economy
    if elicited_econ is None:
        ans = _ask("Economy? [eco/half/full]  (Enter to skip):")
        if ans:
            elicited_econ = _ECON_MAP.get(ans.lower().strip())

    # Question 8: Clutch
    if elicited_clutch is None:
        ans = _ask("Clutch? e.g. 1v3  (Enter to skip):")
        if ans:
            elicited_clutch = _match_clutch(ans)

    display.console.print()

    # Build enriched agents list — prepend elicited agent if new
    agents = list(parsed.agents)
    if elicited_agent and elicited_agent not in agents:
        agents = [elicited_agent, *agents]

    enriched = parsed.model_copy(
        update={
            "agents": agents,
            "map": elicited_map,
            "side": elicited_side,
            "score": elicited_score,
            "phase": elicited_phase,
            "site": elicited_site,
            "econ": elicited_econ,
            "clutch": elicited_clutch,
        }
    )

    return enriched, elicited_agent, elicited_map, elicited_side


__all__ = [
    "run_elicitation",
    "should_elicit",
]

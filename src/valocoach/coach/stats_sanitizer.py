"""Post-response sanity checks for numeric stat claims in LLM coaching output.

The ability sanitizer (``coach.sanitizer``) catches ability/agent/weapon
hallucinations.  This module catches the *other* class of LLM misstatement:
when the model writes a number that doesn't match the PLAYER CONTEXT block
we put in its prompt.

Example failure mode:
    PLAYER CONTEXT says:  "K/D 0.93 · HS 27.4% · ACS 188"
    Model writes:         "your K/D is 1.2 — keep it up!"

The user can't see the prompt the model received, so they have no way to
tell that "1.2" is fabricated.  This sanitizer reads the same PLAYER
CONTEXT and flags any numeric claim that contradicts the source.

Design:
* Conservative — only flags claims that are explicitly attributed
  ("your K/D is X", "K/D: X", "X% HS", ...).  Free-text mentions of
  numbers without a stat label are not touched.
* Tolerant — small rounding differences (e.g. 27 vs 27.4) are accepted.
* Symmetric to ``coach.sanitizer.validate_ability_claims`` — same
  dataclass shape, same once-per-pair dedup, same warning format.
"""

from __future__ import annotations

import contextlib
import re
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Stat label → numeric attribute on PLAYER CONTEXT
# ---------------------------------------------------------------------------

# Maps display name → regex of the LLM's likely phrasing AND the PLAYER
# CONTEXT key whose value we should compare against.
_STAT_PATTERNS: list[tuple[str, re.Pattern[str], str, str]] = [
    # (display, pattern, context_key, value_unit)
    (
        "K/D",
        re.compile(
            r"\b(?:your|the|a)\s*k(?:/|-|\s)?d(?:\s+ratio)?\s*(?:is|of|=|:)?\s*"
            r"(\d+(?:\.\d+)?)",
            re.IGNORECASE,
        ),
        "K/D",
        "",
    ),
    (
        "ACS",
        re.compile(
            r"\b(?:your|an?)?\s*acs\s*(?:is|of|=|:)?\s*(\d+(?:\.\d+)?)",
            re.IGNORECASE,
        ),
        "ACS",
        "",
    ),
    (
        "ADR",
        re.compile(
            r"\b(?:your|an?)?\s*adr\s*(?:is|of|=|:)?\s*(\d+(?:\.\d+)?)",
            re.IGNORECASE,
        ),
        "ADR",
        "",
    ),
    (
        "HS%",
        re.compile(
            r"\b(?:your|the)?\s*(?:hs|headshot)\s*(?:%|percentage|rate)?\s*"
            r"(?:is|of|=|:)?\s*(\d+(?:\.\d+)?)\s*%?",
            re.IGNORECASE,
        ),
        "HS",
        "%",
    ),
    (
        "Win rate",
        re.compile(
            r"\b(?:your|a)?\s*(?:win\s*(?:rate|%)|wr)\s*"
            r"(?:is|of|=|:)?\s*(\d+(?:\.\d+)?)\s*%?",
            re.IGNORECASE,
        ),
        "WR",
        "%",
    ),
]


# ---------------------------------------------------------------------------
# Public dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StatWarning:
    """One flagged numeric mismatch.

    Attributes
    ----------
    stat:
        Display label of the stat the model misquoted (e.g. ``"K/D"``).
    claimed:
        The numeric value the model wrote, as a string for verbatim display.
    actual:
        The numeric value present in PLAYER CONTEXT.
    snippet:
        ~60-char window of the LLM output around the claim.
    """

    stat: str
    claimed: str
    actual: str
    snippet: str

    def format(self) -> str:
        return f'{self.stat}: model said "{self.claimed}" but your real value is {self.actual}'


# ---------------------------------------------------------------------------
# PLAYER CONTEXT extractor
# ---------------------------------------------------------------------------


# Parse the canonical ``PLAYER CONTEXT — Yoursaviour02#NA …`` block this
# project's ``build_stats_context`` emits.  Looking for substrings like
# ``K/D 0.93``, ``HS 27.4%``, ``ACS 188``, ``61% WR``.
_CONTEXT_PATTERNS: dict[str, re.Pattern[str]] = {
    "K/D": re.compile(r"\bK/D\s+(\d+(?:\.\d+)?)\b"),
    "ACS": re.compile(r"\bACS\s+(\d+(?:\.\d+)?)\b"),
    "ADR": re.compile(r"\bADR\s+(\d+(?:\.\d+)?)\b"),
    "HS": re.compile(r"\bHS\s+(\d+(?:\.\d+)?)\s*%"),
    "WR": re.compile(r"\b(\d+(?:\.\d+)?)\s*%\s*WR\b", re.IGNORECASE),
}


def _extract_real_values(player_context: str) -> dict[str, float]:
    """Pull the numeric stats out of a PLAYER CONTEXT block."""
    out: dict[str, float] = {}
    if not player_context:
        return out
    for key, pat in _CONTEXT_PATTERNS.items():
        m = pat.search(player_context)
        if m:
            with contextlib.suppress(ValueError):
                out[key] = float(m.group(1))
    return out


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


# Per-stat tolerance below which a claim is considered "close enough" — we
# don't want to flag "K/D 0.9" vs "K/D 0.93" or "27%" vs "27.4%".
_TOLERANCE: dict[str, float] = {
    "K/D": 0.05,  # absolute K/D delta
    "ACS": 5.0,  # absolute ACS points
    "ADR": 5.0,  # absolute ADR points
    "HS": 2.0,  # absolute percentage points
    "WR": 3.0,  # absolute percentage points
}


def _is_close_enough(stat_key: str, claimed: float, actual: float) -> bool:
    return abs(claimed - actual) <= _TOLERANCE.get(stat_key, 0.0)


def _window(text: str, start: int, end: int, radius: int = 30) -> str:
    a = max(0, start - radius)
    b = min(len(text), end + radius)
    return text[a:b].replace("\n", " ").strip()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def validate_stat_claims(text: str, player_context: str) -> list[StatWarning]:
    """Scan *text* for numeric stat claims that contradict *player_context*.

    Returns a deduplicated list of warnings.  An empty context block (no
    PLAYER CONTEXT supplied) returns an empty list rather than flagging
    everything — the absence of ground truth isn't a mismatch.
    """
    if not text or not player_context:
        return []

    real = _extract_real_values(player_context)
    if not real:
        return []

    warnings: list[StatWarning] = []
    seen: set[tuple[str, str]] = set()

    for display, pat, ctx_key, unit in _STAT_PATTERNS:
        if ctx_key not in real:
            continue
        actual = real[ctx_key]
        for m in pat.finditer(text):
            raw = m.group(1)
            try:
                claimed = float(raw)
            except ValueError:
                continue
            if _is_close_enough(ctx_key, claimed, actual):
                continue
            key = (display, raw)
            if key in seen:
                continue
            seen.add(key)
            warnings.append(
                StatWarning(
                    stat=display,
                    claimed=f"{raw}{unit}",
                    actual=f"{actual:g}{unit}",
                    snippet=_window(text, m.start(), m.end()),
                )
            )

    return warnings


__all__ = ["StatWarning", "validate_stat_claims"]

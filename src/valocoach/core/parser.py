"""Extract structured fields from a freeform coaching situation.

The coach command originally passed raw text straight through — the LLM had
to re-extract every map name, agent, side, and score from the situation
string on each turn.  That hurt three things:

  1. **Retrieval routing.**  ``retrieve_static`` takes ``agent`` and ``map_``
     kwargs to pull exact JSON facts.  Without parsing, the user had to
     remember to pass ``--agent`` and ``--map`` flags or those slots stayed
     empty even when the situation text named them in plain English.
  2. **System prompt grounding.**  Structured metadata at the top of the
     user message (``Map: Ascent · Side: attack · Agent: Jett``) keys the
     LLM into the right frame before it reads the prose.
  3. **Multi-query construction.**  ``build_retrieval_queries`` only fires
     map-callout and per-agent ability queries when it sees those fields.
     Parsing fills them in for free.

The parser is regex-first by design.  Pure-LLM situation parsing adds 2-5 s
of latency for a local model; regex against the canonical JSON name lists
is sub-millisecond.  The full ``raw`` text is always preserved so the LLM
still sees the player's exact wording — parsing only enriches the prompt,
it never replaces it.

Vocabularies (agent names, map names) come from the bundled JSON knowledge
base, not a hardcoded list — agents added to ``agents.json`` are picked up
on the next process start without touching this file.
"""

from __future__ import annotations

import re
from functools import lru_cache

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Regex vocabulary
# ---------------------------------------------------------------------------

# "Attack" / "defense" surface in a lot of forms.  Catch the common ones,
# including the CS-style "T-side" / "CT-side" hangovers.  Word boundaries
# are tightened with non-word lookarounds so "attackers" matches but
# "matchattack" doesn't.
_SIDE_ATTACK = re.compile(
    r"(?<!\w)(attack(?:ing|ers?)?|t[-\s]?side|offense)(?!\w)",
    re.IGNORECASE,
)
_SIDE_DEFENSE = re.compile(
    r"(?<!\w)(defen(?:d(?:ing|ers?)?|[cs](?:e|ing|ders?)?)|ct[-\s]?side)(?!\w)",
    re.IGNORECASE,
)

# Score: "8-12", "8 - 12", "8:12", or with a Unicode en-dash separator.
# Two-digit cap because
# Valorant tops out at 13 in regulation and ~30 in long overtime.
_SCORE_PATTERN = re.compile(r"(?<!\d)(\d{1,2})\s*[-–:]\s*(\d{1,2})(?!\d)")

# Site: "A site", "A long", "B main", "C short".  We only capture the letter.
_SITE_PATTERN = re.compile(
    r"(?<!\w)([abc])\s*(?:site|long|short|main|heaven|hell)(?!\w)",
    re.IGNORECASE,
)

# Clutch: "1v3", "2 v 5".  Same player ranges either side (1-5).
_CLUTCH_PATTERN = re.compile(r"(?<!\d)([1-5])\s*v\s*([1-5])(?!\d)", re.IGNORECASE)

# Economy state — keyword-based because there are many phrasings.
# Order matters: "full buy" must beat the bare "buy" inside "force buy".
_ECON_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("full_buy", ("full buy", "full-buy")),
    ("half_buy", ("half buy", "half-buy", "force buy", "force-buy", "forcebuy")),
    ("eco", ("eco round", "eco buy", "save round", "saving", "pistol round", " eco ")),
)

# Round phase — what the team is *doing* this round.  Same ordering rule:
# more specific phrases first.
_PHASE_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("post_plant", ("post plant", "post-plant", "after plant", "after the plant")),
    ("retake", ("retake", "retaking")),
    ("execute", ("execute", "executing", "site hit")),
    ("default", ("default", "passive setup", "slow default")),
)


# ---------------------------------------------------------------------------
# Vocabulary loading — cached because list_*_names hits disk-backed JSON.
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _agent_patterns() -> list[tuple[str, re.Pattern[str]]]:
    """Compile (canonical_name, pattern) pairs for every agent.

    Agent names are sorted longest-first so e.g. "KAY/O" matches before any
    fragment that overlaps; ``re.escape`` handles the slash, and non-word
    lookarounds substitute for ``\\b`` (which doesn't behave around ``/``).
    """
    from valocoach.retrieval import list_agent_names

    names = sorted(list_agent_names(), key=len, reverse=True)
    return [(n, re.compile(rf"(?<!\w){re.escape(n)}(?!\w)", re.IGNORECASE)) for n in names]


@lru_cache(maxsize=1)
def _map_patterns() -> list[tuple[str, re.Pattern[str]]]:
    from valocoach.retrieval import list_map_names

    names = sorted(list_map_names(), key=len, reverse=True)
    return [(n, re.compile(rf"(?<!\w){re.escape(n)}(?!\w)", re.IGNORECASE)) for n in names]


# ---------------------------------------------------------------------------
# Public model
# ---------------------------------------------------------------------------


class Situation(BaseModel):
    """Structured fields extracted from a freeform coaching question.

    ``raw`` always carries the original input verbatim — downstream callers
    pass it to the LLM unchanged; the parsed fields only supplement.
    """

    raw: str
    agents: list[str] = Field(default_factory=list)
    map: str | None = None
    side: str | None = None  # "attack" | "defense"
    site: str | None = None  # "A" | "B" | "C"
    score: tuple[int, int] | None = None
    clutch: tuple[int, int] | None = None  # (allies, enemies); 1v3 == (1, 3)
    econ: str | None = None  # "eco" | "half_buy" | "full_buy"
    phase: str | None = None  # "post_plant" | "retake" | "execute" | "default"

    @property
    def primary_agent(self) -> str | None:
        """First agent mentioned — convention for routing single-agent retrieval."""
        return self.agents[0] if self.agents else None

    def to_metadata_block(self) -> str:
        """Render fields as a compact ``Key: value`` header for the user message.

        Empty when no fields were extracted, so the caller can safely skip
        the block on a pure-prose situation without checking each attribute.
        """
        lines: list[str] = []
        if self.map:
            lines.append(f"Map: {self.map}")
        if self.side:
            lines.append(f"Side: {self.side}")
        if self.agents:
            lines.append(f"Agent(s): {', '.join(self.agents)}")
        if self.site:
            lines.append(f"Site: {self.site}")
        if self.score:
            lines.append(f"Score: {self.score[0]}-{self.score[1]}")
        if self.clutch:
            lines.append(f"Clutch: {self.clutch[0]}v{self.clutch[1]}")
        if self.econ:
            lines.append(f"Econ: {self.econ.replace('_', ' ')}")
        if self.phase:
            lines.append(f"Phase: {self.phase.replace('_', ' ')}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def parse_situation(text: str) -> Situation:
    """Extract structured fields from freeform coaching input.

    Returns a ``Situation`` with whichever fields could be detected.  Fields
    not present in the text stay ``None`` (or empty list, for ``agents``).
    The full ``text`` is preserved on ``Situation.raw`` so the LLM still
    sees the player's exact wording.

    Examples
    --------
    >>> s = parse_situation("we keep losing 8-12 attack on Ascent as Jett")
    >>> s.map, s.side, s.score, s.agents
    ('Ascent', 'attack', (8, 12), ['Jett'])

    >>> s = parse_situation("post-plant retake B site, 1v3 with Sage")
    >>> s.site, s.phase, s.clutch, s.agents
    ('B', 'post_plant', (1, 3), ['Sage'])
    """
    # --- Map (first match wins; longest-first ordering avoids prefix collisions)
    map_name: str | None = None
    for name, pat in _map_patterns():
        if pat.search(text):
            map_name = name
            break

    # --- Agents (collect all unique matches, preserve canonical-name order)
    agents: list[str] = []
    seen: set[str] = set()
    for name, pat in _agent_patterns():
        if name not in seen and pat.search(text):
            agents.append(name)
            seen.add(name)

    # --- Side
    side: str | None = None
    if _SIDE_ATTACK.search(text):
        side = "attack"
    elif _SIDE_DEFENSE.search(text):
        side = "defense"

    # --- Score (sanity-check the magnitude — 99-99 is almost certainly noise)
    score: tuple[int, int] | None = None
    m = _SCORE_PATTERN.search(text)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        if 0 <= a <= 30 and 0 <= b <= 30:
            score = (a, b)

    # --- Site
    site: str | None = None
    m = _SITE_PATTERN.search(text)
    if m:
        site = m.group(1).upper()

    # --- Clutch
    clutch: tuple[int, int] | None = None
    m = _CLUTCH_PATTERN.search(text)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        # 5v5 isn't a clutch — it's a normal round.  Skip equal counts.
        if a != b:
            clutch = (a, b)

    # --- Econ / phase: keyword scan, padded with spaces so " eco " can require
    # a word boundary without a regex compile per call.
    haystack = f" {text.lower()} "
    econ: str | None = None
    for tier, keywords in _ECON_KEYWORDS:
        if any(kw in haystack for kw in keywords):
            econ = tier
            break

    phase: str | None = None
    for ph, keywords in _PHASE_KEYWORDS:
        if any(kw in haystack for kw in keywords):
            phase = ph
            break

    return Situation(
        raw=text,
        agents=agents,
        map=map_name,
        side=side,
        site=site,
        score=score,
        clutch=clutch,
        econ=econ,
        phase=phase,
    )

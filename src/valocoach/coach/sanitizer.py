"""Post-response sanity checks for LLM coaching output.

The model often hallucinates ability names — names that don't exist in
Valorant at all (``Riftwalk``, ``Nerve Gas``, ``Fireburst``), real abilities
attributed to the wrong agent (``Fade's Paranoia`` — Paranoia is Omen's),
weapon names recast as abilities (``Fade's Ghost`` — Ghost is a pistol),
or generic utility nouns substituting for specific kit names (``Omen's
Flash``, ``Jett's Dash``).

Prompt engineering can't fully stop this on small local models.  We
deterministically validate every claim post-stream against ``agents.json``
and emit a single warning panel listing every flagged pairing.

Detection has two stages:

* **Section-scoped scan** — walks the response, finds positions where an
  agent name appears in a heading-like context (``Omen (Controller)``,
  ``1. Omen``, ``Omen:``), then validates every Title-cased phrase inside
  that agent's section.  This catches the bullet-list pattern the model
  loves: ``Key Abilities: Ghost (stealth), Smoke (area control)`` under an
  ``Omen`` header.
* **Direct-attribution scan** — finds ``<Agent>['s]? <Phrase>``
  constructs anywhere in the text.  Catches "Use Fade's Ghost to
  reposition" type prose claims.

Each warning is emitted at most once per ``(agent, ability)`` pair so a
flooded response doesn't spam the output.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

from valocoach.retrieval.agents import _load as _load_agents

# ---------------------------------------------------------------------------
# Vocabulary sets — populated lazily from agents.json + hand-curated lists.
# ---------------------------------------------------------------------------

# Valorant weapons.  When a possessive claim ("Fade's Ghost") matches one of
# these, we flag it as "weapon, not an ability" rather than "hallucination".
_WEAPONS: frozenset[str] = frozenset(
    {
        "vandal",
        "phantom",
        "operator",
        "ghost",
        "sheriff",
        "spectre",
        "stinger",
        "guardian",
        "marshal",
        "outlaw",
        "bulldog",
        "judge",
        "frenzy",
        "shorty",
        "classic",
        "bucky",
        "ares",
        "odin",
    }
)

# Generic utility nouns the model uses as fake specific ability names.
# These are real-game concepts but NOT proper ability names — every real
# Valorant ability has a specific noun (Dark Cover, not "Smoke";
# Paranoia, not "Flash"; Cloudburst, not "Smoke").
_GENERIC_UTIL: frozenset[str] = frozenset(
    {
        "smoke",
        "smokes",
        "smoke screen",
        "smokescreen",
        "flash",
        "flashes",
        "flashbang",
        "flashbangs",
        "wall",
        "walls",
        "molly",
        "mollies",
        "dash",
        "blitz",
        "scorch",
        "blade",
        "pulse",
        "fireburst",
        "tectonic",
        "disruptor",
        "disruption",
        "annihilation",
        # NOTE: "Annihilation" IS Deadlock's ult but the model sometimes
        # writes it for other sentinels too — handled by cross-attribution
        # check (real_owner lookup) before this generic fallback.
    }
)

# Map names — appear as bare TitleCase tokens after agent headers
# (``Maps: Split, Ascent, Bind.``) and would otherwise be flagged.
_MAPS: frozenset[str] = frozenset(
    {
        "ascent",
        "bind",
        "haven",
        "split",
        "icebox",
        "breeze",
        "fracture",
        "pearl",
        "lotus",
        "sunset",
        "abyss",
    }
)

# Role / descriptor nouns that appear after agent headers and aren't
# ability claims at all.
_DESCRIPTORS: frozenset[str] = frozenset(
    {
        # Role / kit category labels — never an ability claim
        "duelist",
        "controller",
        "initiator",
        "sentinel",
        "ultimate",
        "ult",
        "abilities",
        "ability",
        "kit",
        "role",
        # Game-state / strategy nouns
        "playstyle",
        "mobility",
        "damage",
        "vision",
        "control",
        "rotation",
        "site",
        "spike",
        "plant",
        "defuse",
        "high",
        "low",
        "mid",
        "long",
        "tier",
        "rank",
        "meta",
        "burst",
        "lockdown",
        "anti-op",
        "denial",
        # Section / pronoun / connective words
        "map",
        "maps",
        "key",
        "why",
        "your",
        "his",
        "her",
        "their",
        "use",
        "uses",
        "using",
        "prioritize",
        "consider",
        "team",
        "teams",
        "teammate",
        "teammates",
        "good",
        "solid",
        "strong",
        "weak",
        "average",
        # Multi-word connectives that can leak through bigram partials
        "from",
        "the",
        "of",
        "and",
        "with",
        "for",
        # Generic English nouns that appear in qwen3:8b's "Area denial",
        # "Crowd control", "Final Tips", etc.
        "area",
        "crowd",
        "final",
        "tips",
        "tip",
        "fun",
        "easy",
        "great",
        "ideal",
        "synergy",
        "synergies",
        "split-push",
        "counter",
        "anti",
        "pro",
        "ranked",
        # Section headers the model tends to emit
        "agents",
        "summary",
        "context",
        "recommendations",
        "strategies",
        # NOTE: weapon names (Vandal, Phantom, Operator, Ghost, ...) used to
        # be duplicated here, but they belong solely to ``_WEAPONS`` so we
        # can correctly flag claims like "Key Abilities: Vandal" as
        # weapon-mis-cast-as-ability rather than silently skipping them.
    }
)


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AbilityWarning:
    """One flagged claim from the LLM response.

    ``category`` distinguishes the four failure modes:
      - ``cross_attribution``: real ability, wrong agent. ``real_owner`` set.
      - ``hallucination``:     phrase isn't a real Valorant ability at all.
      - ``weapon``:            phrase is a weapon name, not an ability.
      - ``generic``:           phrase is a generic noun (Smoke / Flash / etc.).
    """

    agent: str
    claimed_ability: str
    real_owner: str | None
    category: str
    snippet: str

    def format(self) -> str:
        ab = self.claimed_ability
        ag = self.agent
        if self.category == "cross_attribution":
            return f"\"{ag}'s {ab}\" — {ab} is {self.real_owner}'s ability"
        if self.category == "weapon":
            return f'"{ag}\'s {ab}" — {ab} is a weapon, not an ability'
        if self.category == "generic":
            return f'"{ag}\'s {ab}" — {ab} is a generic descriptor, not a specific Valorant ability'
        return f'"{ag}\'s {ab}" — {ab} is not a Valorant ability'


# ---------------------------------------------------------------------------
# Internal caches
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _index() -> tuple[
    dict[str, set[str]],  # agent_lower -> {ability_lower}
    dict[str, str],  # ability_lower -> canonical agent name
    list[str],  # canonical agent names, longest-first
    re.Pattern[str],  # agent header pattern
    re.Pattern[str],  # direct-attribution pattern
    re.Pattern[str],  # broad TitleCase-phrase pattern
]:
    agents = _load_agents()
    agent_abilities: dict[str, set[str]] = {}
    ability_owner: dict[str, str] = {}
    for a in agents:
        names = {ab["name"].lower() for ab in a["abilities"].values()}
        agent_abilities[a["name"].lower()] = names
        for ab in a["abilities"].values():
            ability_owner[ab["name"].lower()] = a["name"]

    sorted_names = sorted((a["name"] for a in agents), key=len, reverse=True)
    name_alt = "|".join(re.escape(n) for n in sorted_names)

    # Header — agent appears as a list item, with role parens, or with a colon.
    # We intentionally allow case-insensitive agent match but anchor the
    # capitalised role/punctuation that follows.  The list-marker is
    # very loose because qwen3:8b emits "1. Omen", "1 Omen", "• Omen", and
    # bare-number variants interchangeably.
    header_pattern = re.compile(
        r"(?:^|\n)\s*(?:\d+[.)]?\s+|[-•*▪]\s+)?"
        r"(?i:(" + name_alt + r"))"
        r"(?:\s*\([A-Za-z][A-Za-z\s/]*\)|\s*:)"
    )

    # Direct possessive — "<Agent>'s <Phrase>" or "<Agent> <Phrase>".
    # Phrase must START with an uppercase letter so we don't capture
    # descriptive prose ("Jett's mobility").
    direct_pattern = re.compile(
        r"(?<![A-Za-z])(?i:(" + name_alt + r"))"
        r"(?:[’']s|s)?\s+"
        r"([A-Z][\w’'/-]*"
        r"(?:\s+(?:de|the)\s+[A-Z][\w’'/-]*)?"
        r"(?:\s+[A-Z][\w’'/-]*){0,2})"
    )

    # Broad — any TitleCase 1-3 token phrase, allowing lowercase
    # connectives ("the"/"de"/"of") between cap tokens so multi-word
    # ability names ("From the Shadows", "Tour de Force") match cleanly.
    broad_pattern = re.compile(
        r"\b([A-Z][a-zA-Z'’/-]{2,}"
        r"(?:\s+(?:the|de|of)\s+[A-Z][a-zA-Z'’/-]{2,})?"
        r"(?:\s+[A-Z][a-zA-Z'’/-]{2,}){0,2})\b"
    )

    return (
        agent_abilities,
        ability_owner,
        sorted_names,
        header_pattern,
        direct_pattern,
        broad_pattern,
    )


def _agent_name_lookup(name: str, agent_names: list[str]) -> str:
    """Resolve a free-text agent token to its canonical capitalisation."""
    low = name.lower()
    for canonical in agent_names:
        if canonical.lower() == low:
            return canonical
    return name


def _classify(
    phrase: str,
    agent_lower: str,
    agent_abilities: dict[str, set[str]],
    ability_owner: dict[str, str],
    agent_names_lower: set[str],
) -> tuple[str, str | None] | None:
    """Return (category, real_owner) for a claim, or None if it's not flaggable.

    Returns ``None`` for legitimate non-ability phrases (descriptors, maps,
    other agent names, exact matches for the current agent's real
    abilities).
    """
    p = phrase.lower().strip()
    if not p:
        return None
    if p in _DESCRIPTORS or p in _MAPS or p in agent_names_lower:
        return None
    if p in agent_abilities.get(agent_lower, set()):
        return None
    if p in ability_owner:
        return ("cross_attribution", ability_owner[p])
    if p in _WEAPONS:
        return ("weapon", None)
    if p in _GENERIC_UTIL:
        return ("generic", None)
    # Pure hallucination — phrase looks ability-shaped but matches nothing.
    return ("hallucination", None)


# ---------------------------------------------------------------------------
# Section-scoped scan
# ---------------------------------------------------------------------------


_PARAGRAPH_BREAK = re.compile(r"\n\s*\n")


def _find_sections(text: str, header_pattern: re.Pattern[str]) -> list[tuple[str, int, int]]:
    """Return ``(agent_raw, section_start, section_end)`` triples.

    A section runs from the agent header until *the first of*:
      * the next agent header,
      * the next blank-line paragraph break,
      * 350 characters after the header.

    The paragraph cap is critical — qwen3:8b often emits ``Astra (Controller):
    Vision control and Disruption.\\n\\nFinal Tips:\\n• Use Fade's Ghost…``
    where the next paragraph belongs to a different topic, NOT Astra.
    """
    matches = list(header_pattern.finditer(text))
    sections: list[tuple[str, int, int]] = []
    for i, m in enumerate(matches):
        agent_raw = m.group(1)
        start = m.end()
        next_header_start = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        para = _PARAGRAPH_BREAK.search(text, start)
        next_para_start = para.start() if para else len(text)
        end = min(next_header_start, next_para_start, start + 350, len(text))
        sections.append((agent_raw, start, end))
    return sections


_ABILITY_LIST_HEADER = re.compile(
    r"(?i)(?:^|\n)\s*(?:[-•*▪]\s*)?(?:key\s+)?abilit(?:y|ies)\s*[:=]\s*(.+?)(?:\n|$)"
)
_KIT_LIST_HEADER = re.compile(r"(?i)(?:^|\n)\s*(?:[-•*▪]\s*)?kit\s*[:=]\s*(.+?)(?:\n|$)")
_LIST_ITEM = re.compile(
    r"([A-Z][\w’'/-]*(?:\s+(?:the|de|of)\s+[A-Z][\w’'/-]*)?(?:\s+[A-Z][\w’'/-]*){0,2})"
)


def _scan_section(
    agent_raw: str,
    section_text: str,
    agent_abilities: dict[str, set[str]],
    ability_owner: dict[str, str],
    agent_names_lower: set[str],
    broad_pattern: re.Pattern[str],
) -> list[AbilityWarning]:
    """Validate ability-list claims inside *section_text*.

    We deliberately restrict this scan to ``Abilities:``/``Key Abilities:``/
    ``Kit:`` labelled lists rather than the entire section.  A free-form
    section scan flags sentence-initial verbs like "Dominates", "Excels",
    "Best", etc. because they happen to be TitleCase — far too noisy.  An
    explicit ability-list label is a strong signal that what follows is
    intended as an ability roster, where false positives are rare.
    """
    agent_lower = agent_raw.lower()
    warnings: list[AbilityWarning] = []
    seen_in_section: set[str] = set()

    label_matches = list(_ABILITY_LIST_HEADER.finditer(section_text))
    label_matches.extend(_KIT_LIST_HEADER.finditer(section_text))
    if not label_matches:
        return warnings

    for label_m in label_matches:
        list_text = label_m.group(1)
        for item_m in _LIST_ITEM.finditer(list_text):
            phrase = item_m.group(1).strip()
            if phrase.lower() in seen_in_section:
                continue
            tokens = phrase.split()

            for n in range(min(3, len(tokens)), 0, -1):
                cand = " ".join(tokens[:n])
                result = _classify(
                    cand, agent_lower, agent_abilities, ability_owner, agent_names_lower
                )
                if result is None:
                    break
                category, owner = result
                # Inside an explicit ability-list label, hallucination flags
                # are safe even for short tokens — the model is explicitly
                # claiming the item IS an ability.
                warnings.append(
                    AbilityWarning(
                        agent=agent_raw,
                        claimed_ability=cand,
                        real_owner=owner,
                        category=category,
                        snippet=label_m.group(0).strip()[:100],
                    )
                )
                seen_in_section.add(cand.lower())
                break

    return warnings


# ---------------------------------------------------------------------------
# Direct-attribution scan ("<Agent>'s <Phrase>")
# ---------------------------------------------------------------------------


def _scan_direct(
    text: str,
    agent_abilities: dict[str, set[str]],
    ability_owner: dict[str, str],
    direct_pattern: re.Pattern[str],
) -> list[AbilityWarning]:
    warnings: list[AbilityWarning] = []

    for m in direct_pattern.finditer(text):
        agent_raw = m.group(1)
        agent_lower = agent_raw.lower()
        phrase = m.group(2).strip()
        tokens = phrase.split()
        # Was the match possessive ("Omen's Vandal") or bare ("Omen Vandal")?
        # Without an explicit possessive we can't reliably distinguish a
        # loadout call ("Omen Vandal full buy") from an ability claim, so we
        # only flag weapons / generic descriptors on possessive matches.
        between = text[m.start(1) + len(agent_raw) : m.start(2)]
        is_possessive = "'" in between or "’" in between

        for n in range(min(4, len(tokens)), 0, -1):
            cand = " ".join(tokens[:n])
            cl = cand.lower()

            if cl in agent_abilities.get(agent_lower, set()):
                break  # correct attribution, stop here

            if cl in ability_owner:
                warnings.append(
                    AbilityWarning(
                        agent=agent_raw,
                        claimed_ability=cand,
                        real_owner=ability_owner[cl],
                        category="cross_attribution",
                        snippet=text[max(0, m.start() - 20) : m.end() + 20].replace("\n", " "),
                    )
                )
                break
            if cl in _WEAPONS:
                if is_possessive:
                    warnings.append(
                        AbilityWarning(
                            agent=agent_raw,
                            claimed_ability=cand,
                            real_owner=None,
                            category="weapon",
                            snippet=text[max(0, m.start() - 20) : m.end() + 20].replace("\n", " "),
                        )
                    )
                break
            if cl in _GENERIC_UTIL:
                if is_possessive:
                    warnings.append(
                        AbilityWarning(
                            agent=agent_raw,
                            claimed_ability=cand,
                            real_owner=None,
                            category="generic",
                            snippet=text[max(0, m.start() - 20) : m.end() + 20].replace("\n", " "),
                        )
                    )
                break
            if cl in _DESCRIPTORS or cl in _MAPS:
                break
        else:
            # No prefix matched anything — pure hallucination candidate.
            # Require possessive ("Omen's Riftwalk") for direct-scan flags so
            # that bare prose ("Omen plays mid") doesn't trip the detector.
            if not is_possessive:
                continue
            first = tokens[0]
            if (
                len(first) >= 4
                and len(tokens) <= 3
                and first.lower() not in _DESCRIPTORS
                and first.lower() not in _MAPS
            ):
                warnings.append(
                    AbilityWarning(
                        agent=agent_raw,
                        claimed_ability=phrase,
                        real_owner=None,
                        category="hallucination",
                        snippet=text[max(0, m.start() - 20) : m.end() + 20].replace("\n", " "),
                    )
                )

    return warnings


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def validate_ability_claims(text: str) -> list[AbilityWarning]:
    """Scan *text* for ability claims that don't match ``agents.json``.

    Returns a deduplicated list of warnings; each ``(agent, ability)``
    pair appears at most once.
    """
    if not text:
        return []

    (
        agent_abilities,
        ability_owner,
        agent_names,
        header_pattern,
        direct_pattern,
        broad_pattern,
    ) = _index()
    agent_names_lower = {n.lower() for n in agent_names}

    warnings: list[AbilityWarning] = []

    # Pass 1 — agent-section-scoped broad scan.
    for agent_raw, start, end in _find_sections(text, header_pattern):
        warnings.extend(
            _scan_section(
                _agent_name_lookup(agent_raw, agent_names),
                text[start:end],
                agent_abilities,
                ability_owner,
                agent_names_lower,
                broad_pattern,
            )
        )

    # Pass 2 — direct possessive ("<Agent>'s <Phrase>") anywhere.
    warnings.extend(_scan_direct(text, agent_abilities, ability_owner, direct_pattern))

    # Dedupe by (agent_lower, ability_lower).
    seen: set[tuple[str, str]] = set()
    result: list[AbilityWarning] = []
    for w in warnings:
        key = (w.agent.lower(), w.claimed_ability.lower())
        if key in seen:
            continue
        seen.add(key)
        result.append(w)
    return result


__all__ = ["AbilityWarning", "validate_ability_claims"]

"""Intent classifier for coaching questions.

Maps a free-text coaching situation (plus its parsed ``Situation`` fields) to
one of nine ``IntentType`` values.  Downstream callers use the intent to pick
the right prompt template and panel title — so a stat-summary question gets a
table-style response instead of the tactical five-section format, an economy
round gets concise buy-decision output, and so on.

Classification is purely rule-based (regex keyword scan + field presence
checks).  Rules execute in priority order so a "1v3 post-plant" question is
always ``clutch``, not ``post_plant``.

Priority order (highest to lowest)
-----------------------------------
1. clutch          — ``situation.clutch`` is set, or explicit NvM pattern
2. post_plant      — ``situation.phase == "post_plant"``
3. retake          — ``situation.phase == "retake"`` or retake keyword
4. economy         — ``situation.econ`` is set, or buy/save keyword
5. stat_analysis   — explicit stats/performance keywords
6. agent_info      — ability/kit/guide keyword
7. meta            — meta/tier-list/patch keyword
8. tactical        — map **and** side both present (structured execute advice)
9. general         — everything else

Special case
------------
``post_game`` is also a valid ``IntentType`` but is NEVER produced by the
classifier — it's set exclusively via ``force_intent="post_game"`` from
the ``valocoach post-game`` command path, which has its own analyzer
pipeline before calling ``run_coach``.
"""

from __future__ import annotations

import re
from typing import Literal

from valocoach.core.parser import Situation

# ---------------------------------------------------------------------------
# Public type
# ---------------------------------------------------------------------------

IntentType = Literal[
    "clutch",
    "post_plant",
    "retake",
    "economy",
    "stat_analysis",
    "agent_info",
    "meta",
    "tactical",
    "general",
    "post_game",
]

# ---------------------------------------------------------------------------
# Keyword patterns (compiled once at module load)
# ---------------------------------------------------------------------------

_CLUTCH_KW = re.compile(
    r"(?<!\w)"
    r"(?:clutch|1\s*v\s*[2-5]|2\s*v\s*[3-5]|3\s*v\s*[4-5]|4\s*v\s*5"
    r"|last\s+alive|solo\s+(?:clutch|vs\b|kill)|by\s+myself|alone\s+vs)"
    r"(?!\w)",
    re.IGNORECASE,
)

_RETAKE_KW = re.compile(
    r"(?<!\w)(?:retake|retaking|take\s+back|recapture|re-take)(?!\w)",
    re.IGNORECASE,
)

_ECONOMY_KW = re.compile(
    r"(?<!\w)"
    r"(?:eco|save|force\s*buy|half\s*buy|full\s*buy|pistol|budget|credits?"
    r"|how\s+much\s+(?:to\s+)?(?:buy|spend)"
    r"|should\s+(?:we|i)\s+(?:buy|save|force))"
    r"(?!\w)",
    re.IGNORECASE,
)

_STAT_KW = re.compile(
    r"(?<!\w)"
    r"(?:my\s+stats?|my\s+kd|my\s+kda|my\s+hs|headshot\s+%|my\s+(?:win\s+)?rate"
    r"|how\s+am\s+i\s+doing|my\s+performance|my\s+average|my\s+rank"
    r"|compare\s+(?:my|me)|am\s+i\s+improving|my\s+form"
    r"|my\s+recent\s+matches?|my\s+history"
    # First-person agent/map performance — "which agent do I play better"
    r"|(?:which|what)\s+agent\s+(?:do\s+|should\s+)?i\s+play\s+(?:best|better|well|most|the\s+most)"
    r"|(?:which|what)\s+map\s+(?:do\s+|should\s+)?i\s+play\s+(?:best|better|well|most|the\s+most)"
    r"|my\s+best\s+(?:agent|map|role)"
    r"|i\s+play\s+better\s+(?:on|with|as)"
    r"|am\s+i\s+better\s+(?:on|with|as))"
    r"(?!\w)",
    re.IGNORECASE,
)

_AGENT_INFO_KW = re.compile(
    r"(?<!\w)"
    r"(?:how\s+does|what\s+does|explain|how\s+to\s+play|kit|abilities?"
    r"|ability\s+costs?|ult(?:imate)?\s+costs?|when\s+(?:to\s+)?use"
    r"|good\s+(?:at|for)|role\s+of|playstyle)"
    r"(?!\w)",
    re.IGNORECASE,
)

_META_KW = re.compile(
    r"(?<!\w)"
    r"(?:meta|tier\s*list|best\s+agent|what\s+agent"
    r"|current\s+patch|patch\s+notes?|overpowered|op\s+agent"
    r"|strong\s+agent|which\s+agent|who\s+(?:is\s+)?(?:best|strong|op)"
    r"|what(?:'?s|\s+is)\s+(?:good|strong|broken|meta))"
    r"(?!\w)",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------


def classify_intent(situation: Situation, raw: str) -> IntentType:
    """Classify the coaching intent from the parsed ``Situation`` and raw text.

    Args:
        situation: Structured fields extracted by ``parse_situation``.
        raw:       Original (un-parsed) situation string; same as
                   ``situation.raw`` but passed explicitly for clarity.

    Returns:
        One of the nine ``IntentType`` literals.

    Examples
    --------
    >>> from valocoach.core.parser import Situation
    >>> s = Situation(raw="1v3 post plant B site", clutch=(1, 3), phase="post_plant")
    >>> classify_intent(s, s.raw)
    'clutch'

    >>> s = Situation(raw="what agent is meta right now?")
    >>> classify_intent(s, s.raw)
    'meta'
    """
    # 1. Clutch — explicit NvM in parsed fields OR keyword scan
    if situation.clutch or _CLUTCH_KW.search(raw):
        return "clutch"

    # 2. Post-plant
    if situation.phase == "post_plant":
        return "post_plant"

    # 3. Retake — parsed phase *or* keyword (keyword catches "retake" without
    #    the phase parser firing first because that requires a full phase match)
    if situation.phase == "retake" or _RETAKE_KW.search(raw):
        return "retake"

    # 4. Economy — parsed econ field *or* buy/save keyword
    if situation.econ or _ECONOMY_KW.search(raw):
        return "economy"

    # 5. Stat analysis — keyword only (no parser field)
    if _STAT_KW.search(raw):
        return "stat_analysis"

    # 6. Agent info — keyword only
    if _AGENT_INFO_KW.search(raw):
        return "agent_info"

    # 7. Meta — keyword only
    if _META_KW.search(raw):
        return "meta"

    # 8. Tactical — structured execute advice when both map AND side are known
    if situation.map and situation.side:
        return "tactical"

    # 9. Fallback
    return "general"


__all__ = ["IntentType", "classify_intent"]

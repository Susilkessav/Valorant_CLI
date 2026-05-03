"""Stats-engine constants — the single source of truth.

Every formula in :mod:`valocoach.stats` references this module.
Nothing here performs computation — no functions, no derived values.
If a value must be derived from primitives here (e.g. half of regulation
rounds), do it at the call site so there is one source of truth with no
stale arithmetic to audit.

Sections:
    1. Valorant game structure   — immutable facts about the game
    2. ACS formula weights       — Riot's per-kill and multi-kill bonuses
    3. Trade detection           — timing window for trade classification
    4. Sample thresholds         — min matches/rounds before a stat is reliable
    5. Rank benchmarks           — population averages per rank tier
"""

from __future__ import annotations

from typing import Final

# ---------------------------------------------------------------------------
# 1. Valorant game structure
# ---------------------------------------------------------------------------
# Immutable facts about competitive Valorant. Any formula that reasons
# about "a full match" or side assignment starts here.

ROUNDS_PER_HALF: Final[int] = 12
"""Rounds each side plays before the half-time swap."""

ROUNDS_TO_WIN_REGULATION: Final[int] = 13
"""First team to 13 round wins takes the match in regulation."""

MAX_REGULATION_ROUNDS: Final[int] = ROUNDS_PER_HALF * 2 + 1  # 25
"""Upper bound on regulation rounds (12-12 forces one decisive round)."""

OVERTIME_ROUNDS_PER_HALF: Final[int] = 1
"""Each OT mini-half is one round; first to win both (or lead by 2) wins."""

# Round index boundaries — used by side-assignment logic elsewhere.
FIRST_HALF_END: Final[int] = ROUNDS_PER_HALF - 1  # 11  (0-indexed)
SECOND_HALF_END: Final[int] = MAX_REGULATION_ROUNDS - 1  # 24  (0-indexed)

# ---------------------------------------------------------------------------
# 2. ACS formula weights
# ---------------------------------------------------------------------------
# Riot computes ACS (Average Combat Score) per round, then averages across
# rounds. The per-round contribution has two additive bonuses on top of
# base damage dealt:
#
#   • Kill bonus   — depends on how many enemies were alive when you got
#                    the kill (more enemies alive = kill was harder = more pts)
#   • Multi-kill   — extra points for each kill beyond the first in a round
#
# These values match Riot's published formula. Change them only if Riot
# updates the scoring system.

ACS_KILL_BONUS: Final[dict[int, int]] = {
    5: 150,  # killed into a 5v1 or surviving enemy count was 5
    4: 130,
    3: 110,
    2: 90,
    1: 70,
}
"""Bonus ACS points per kill, keyed by number of enemies alive at kill time."""

ACS_MULTIKILL_BONUS: Final[dict[int, int]] = {
    2: 50,  # 2-kill round: second kill gets +50
    3: 100,  # 3-kill round: third kill gets +100
    4: 150,
    5: 200,
}
"""Cumulative bonus for Nth kill in the same round (1st kill has no bonus)."""

# ---------------------------------------------------------------------------
# 3. Trade detection
# ---------------------------------------------------------------------------

TRADE_WINDOW_MS: Final[int] = 5_000
"""A death is "traded" if a teammate kills the killer within this window (ms).

5 s matches the threshold used by tracker.gg and Blitz; community consensus
is that anything longer starts counting accidental timing as a trade."""

# ---------------------------------------------------------------------------
# 4. Sample thresholds
# ---------------------------------------------------------------------------
# Minimum matches AND rounds before a metric is considered reliable.
# Using *both* dimensions matters: a player who only plays short stomps
# (13-round matches) accumulates match-count faster than round-count, so
# match-only thresholds would mark thin data as reliable too soon.
#
# Source: BUILD_PLAN.md § "Sample-size thresholds for statistical
# reliability". Values pick the upper end of each BUILD_PLAN range so a
# ⚠ flag means "genuinely unreliable", not "a bit thin".
#
# `rounds: None` means "no round-count requirement for this metric"
# (win-rate is binary per match — rounds don't add information).

SAMPLE_THRESHOLDS: Final[dict[str, dict[str, int | None]]] = {
    "acs": {"matches": 15, "rounds": 200},
    "adr": {"matches": 15, "rounds": 200},
    "kd": {"matches": 20, "rounds": 300},
    "hs_pct": {"matches": 30, "rounds": 400},
    "kast": {"matches": 15, "rounds": 200},
    "first_blood_rate": {"matches": 30, "rounds": 400},
    "first_death_rate": {"matches": 30, "rounds": 400},
    "clutch_rate": {"matches": 30, "rounds": 600},
    "trade_efficiency": {"matches": 20, "rounds": 400},
    "econ_rating": {"matches": 15, "rounds": 300},
    "win_rate": {"matches": 30, "rounds": None},
    "win_rate_split": {"matches": 30, "rounds": None},
}
"""Per-metric reliability gates: ``{"matches": N, "rounds": N | None}``."""

# ---------------------------------------------------------------------------
# 5. Rank benchmarks
# ---------------------------------------------------------------------------
# Approximate population averages per rank tier, for coach context
# ("your ACS is below Gold average"). Source: community-tracked data
# from Blitz / tracker.gg — treat as ballpark figures, not exact.
#
# hs_pct is stored as a whole-number percentage (e.g. 22 means 22 %).
# kast is stored as a whole-number percentage (e.g. 69 means 69 %).

RANK_BENCHMARKS: Final[dict[str, dict[str, float]]] = {
    "Iron": {"acs": 130, "adr": 110, "hs_pct": 14, "kast": 62},
    "Bronze": {"acs": 155, "adr": 125, "hs_pct": 17, "kast": 65},
    "Silver": {"acs": 175, "adr": 135, "hs_pct": 20, "kast": 67},
    "Gold": {"acs": 195, "adr": 145, "hs_pct": 22, "kast": 69},
    "Platinum": {"acs": 210, "adr": 150, "hs_pct": 24, "kast": 71},
    "Diamond": {"acs": 225, "adr": 155, "hs_pct": 26, "kast": 73},
    "Ascendant": {"acs": 235, "adr": 160, "hs_pct": 28, "kast": 75},
    "Immortal": {"acs": 250, "adr": 165, "hs_pct": 30, "kast": 77},
    "Radiant": {"acs": 270, "adr": 175, "hs_pct": 33, "kast": 80},
}
"""Population-average stats per rank tier (acs, adr, hs_pct%, kast%)."""

#!/usr/bin/env python3
"""F2 — Post-game analyzer regression eval.

Runs each post-game analyzer against synthetic fixture matches (built from
``tests/eval/post_game_cases.yaml``) and checks that the expected
``Finding.root_cause_tag`` values appear (or don't appear) in the output.

All fixtures are constructed in-memory using ``types.SimpleNamespace`` — no
database session, no API calls, no Ollama.  This lets the eval run offline
and catches regressions in the deterministic analysis logic.

Usage
-----
    # Default run
    python scripts/eval_post_game.py

    # Custom cases file
    python scripts/eval_post_game.py --cases tests/eval/post_game_cases.yaml

    # Stop on first failure
    python scripts/eval_post_game.py --fail-fast

    # Save results to JSON
    python scripts/eval_post_game.py --output eval_post_game_results.json

Exit codes
----------
    0 — all cases pass
    1 — one or more cases fail
    2 — runner error (bad YAML, import failure, etc.)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

try:
    import yaml
except ImportError:
    print("PyYAML is required: pip install pyyaml", file=sys.stderr)
    sys.exit(2)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CASES = REPO_ROOT / "tests" / "eval" / "post_game_cases.yaml"

if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

# ---------------------------------------------------------------------------
# Terminal colours
# ---------------------------------------------------------------------------

RESET = "\033[0m"
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BOLD = "\033[1m"
DIM = "\033[2m"


# ---------------------------------------------------------------------------
# Minimal fixture builder
# ---------------------------------------------------------------------------
# The analyzers access Match attributes via plain attribute access (match.rounds,
# match.players, round.kills, etc.) — SimpleNamespace quacks correctly for all
# of these without needing a real SQLAlchemy session.

_PUUID = "player-puuid-0001"
_TEAM = "Blue"
_ENEMY_TEAM = "Red"


def _kill(killer_puuid: str, victim_puuid: str, t_ms: int = 5000) -> SimpleNamespace:
    k = SimpleNamespace()
    k.killer_puuid = killer_puuid
    k.victim_puuid = victim_puuid
    k.time_in_round_ms = t_ms
    k.victim_x = None
    k.victim_y = None
    k.killer_x = None
    k.killer_y = None
    k.engagement_distance = None
    k.is_traded = False
    k.weapon_name = "Vandal"
    return k


def _round_player(puuid: str, *, team: str = _TEAM, **kwargs) -> SimpleNamespace:
    rp = SimpleNamespace()
    rp.puuid = puuid
    rp.team = team
    rp.score = kwargs.get("score", 200)
    rp.kills = kwargs.get("kills", 1)
    rp.headshots = kwargs.get("headshots", 0)
    rp.bodyshots = kwargs.get("bodyshots", 5)
    rp.legshots = kwargs.get("legshots", 0)
    rp.damage_dealt = kwargs.get("damage_dealt", 150)
    rp.loadout_value = kwargs.get("loadout_value", 3900)
    rp.remaining_credits = kwargs.get("remaining_credits", 500)
    rp.survived = kwargs.get("survived", True)
    rp.was_afk = False
    rp.stayed_in_spawn = False
    rp.ability_casts_grenade = kwargs.get("ability_casts_grenade")
    rp.ability_casts_ability1 = kwargs.get("ability_casts_ability1")
    rp.ability_casts_ability2 = kwargs.get("ability_casts_ability2")
    rp.ability_casts_ultimate = kwargs.get("ability_casts_ultimate")
    return rp


def _round(
    round_number: int,
    winning_team: str,
    kills: list,
    round_players: list | None = None,
    *,
    bomb_planted: bool = False,
    planter_puuid: str | None = None,
    plant_site: str | None = None,
) -> SimpleNamespace:
    r = SimpleNamespace()
    r.round_number = round_number
    r.winning_team = winning_team
    r.result_code = "Elimination"
    r.kills = kills
    r.round_players = round_players or []
    r.bomb_planted = bomb_planted
    r.planter_puuid = planter_puuid
    r.plant_site = plant_site
    r.plant_x = None
    r.plant_y = None
    r.bomb_defused = False
    r.defuser_puuid = None
    r.defuse_x = None
    r.defuse_y = None
    return r


def _match_player(puuid: str, *, team: str = _TEAM, **kwargs) -> SimpleNamespace:
    mp = SimpleNamespace()
    mp.puuid = puuid
    mp.team = team
    mp.team_id = team  # Some analyzers use team_id
    mp.agent_name = kwargs.get("agent_name", "Jett")
    mp.won = kwargs.get("won", False)
    mp.kills = kwargs.get("kills", 10)
    mp.deaths = kwargs.get("deaths", 12)
    mp.assists = kwargs.get("assists", 2)
    mp.score = kwargs.get("score", 2500)
    mp.rounds_played = kwargs.get("rounds_played", 20)
    mp.headshots = kwargs.get("headshots", 5)
    mp.bodyshots = kwargs.get("bodyshots", 25)
    mp.legshots = kwargs.get("legshots", 2)
    mp.damage_dealt = kwargs.get("damage_dealt", 2800)
    mp.damage_received = kwargs.get("damage_received", 3000)
    mp.first_bloods = kwargs.get("first_bloods", 2)
    mp.first_deaths = kwargs.get("first_deaths", 3)
    mp.plants = kwargs.get("plants", 1)
    mp.defuses = kwargs.get("defuses", 0)
    mp.credits_spent = kwargs.get("credits_spent", 45000)
    mp.avg_loadout = kwargs.get("avg_loadout", 3800)
    mp.afk_rounds = 0
    mp.rounds_in_spawn = 0
    mp.competitive_tier = 18
    mp.started_at = "2026-05-15T12:00:00Z"
    return mp


def _build_match(
    rounds: list,
    players: list,
    *,
    map_name: str = "Ascent",
    winning_team: str = "Blue",
) -> SimpleNamespace:
    m = SimpleNamespace()
    m.match_id = "eval-match-001"
    m.map_name = map_name
    m.queue_id = "competitive"
    m.rounds = rounds
    m.players = players
    m.winning_team = winning_team
    m.red_score = 10
    m.blue_score = 13
    return m


# ---------------------------------------------------------------------------
# Scenario builders
# ---------------------------------------------------------------------------
# Each returns (match, puuid) ready to pass to the analyzers.


def _scenario_first_contact_deaths() -> tuple[SimpleNamespace, str]:
    """Player dies first in 12 of 20 rounds — triggers entry_failure."""
    enemy_puuid = "enemy-001"
    players = [
        _match_player(_PUUID, team=_TEAM, first_deaths=12, first_bloods=2),
        *[_match_player(f"teammate-{i}", team=_TEAM) for i in range(4)],
        *[_match_player(f"enemy-{i}", team=_ENEMY_TEAM) for i in range(5)],
    ]
    rounds = []
    for i in range(20):
        if i < 12:
            # Player dies first
            kills = [_kill(enemy_puuid, _PUUID, t_ms=4000)]
        else:
            # Player kills first
            kills = [_kill(_PUUID, enemy_puuid, t_ms=4000)]
        rounds.append(_round(i, _TEAM if i >= 12 else _ENEMY_TEAM, kills))
    return _build_match(rounds, players), _PUUID


def _scenario_first_contact_bloods() -> tuple[SimpleNamespace, str]:
    """Player gets first blood in 8 of 20 rounds — triggers entry_success."""
    enemy_puuid = "enemy-001"
    players = [
        _match_player(_PUUID, team=_TEAM, first_bloods=8, first_deaths=1),
        *[_match_player(f"teammate-{i}", team=_TEAM) for i in range(4)],
        *[_match_player(f"enemy-{i}", team=_ENEMY_TEAM) for i in range(5)],
    ]
    rounds = []
    for i in range(20):
        if i < 8:
            kills = [_kill(_PUUID, enemy_puuid, t_ms=3500)]
        else:
            # Some other player gets first contact
            kills = [_kill(f"teammate-{i % 4}", enemy_puuid, t_ms=3500)]
        rounds.append(_round(i, _TEAM, kills))
    return _build_match(rounds, players), _PUUID


def _scenario_side_split_gap() -> tuple[SimpleNamespace, str]:
    """ATK WR=90%, DEF WR=20% — triggers side_imbalance."""
    players = [
        _match_player(_PUUID, team=_TEAM),
        *[_match_player(f"teammate-{i}", team=_TEAM) for i in range(4)],
        *[_match_player(f"enemy-{i}", team=_ENEMY_TEAM) for i in range(5)],
    ]
    rounds = []
    # Rounds 0-11: first half — player is on attack (no plant data → round < 12 = attack)
    # Make 10/11 attack rounds won
    for i in range(11):
        won = _TEAM if i < 10 else _ENEMY_TEAM
        rp = [_round_player(_PUUID, team=_TEAM)]
        rounds.append(_round(i, won, [], round_players=rp))
    # Rounds 12-23: second half — player is on defense (round >= 12 = defense)
    # Make only 2/12 defense rounds won
    for i in range(12, 24):
        won = _TEAM if i < 14 else _ENEMY_TEAM
        rp = [_round_player(_PUUID, team=_TEAM)]
        rounds.append(_round(i, won, [], round_players=rp))
    return _build_match(rounds, players), _PUUID


def _scenario_low_utility() -> tuple[SimpleNamespace, str]:
    """Controller (Omen) casting 0.05 abilities/round — triggers low_utility."""
    players = [
        _match_player(_PUUID, team=_TEAM, agent_name="Omen"),
        *[_match_player(f"teammate-{i}", team=_TEAM) for i in range(4)],
        *[_match_player(f"enemy-{i}", team=_ENEMY_TEAM) for i in range(5)],
    ]
    rounds = []
    for i in range(20):
        # Omen baseline=1.2 casts/round; give 0 non-ult casts
        rp = [
            _round_player(
                _PUUID,
                team=_TEAM,
                ability_casts_grenade=0,
                ability_casts_ability1=0,
                ability_casts_ability2=0,
                ability_casts_ultimate=0,
            )
        ]
        rounds.append(_round(i, _TEAM, [], round_players=rp))
    return _build_match(rounds, players), _PUUID


def _scenario_average_match() -> tuple[SimpleNamespace, str]:
    """Average performance — no extreme values, should not fire critical findings."""
    enemy_puuid = "enemy-001"
    players = [
        _match_player(_PUUID, team=_TEAM, first_deaths=4, first_bloods=4),
        *[_match_player(f"teammate-{i}", team=_TEAM) for i in range(4)],
        *[_match_player(f"enemy-{i}", team=_ENEMY_TEAM) for i in range(5)],
    ]
    rounds = []
    for i in range(20):
        kills = [
            _kill(_PUUID, enemy_puuid, t_ms=8000)
            if i % 5 == 0
            else _kill(enemy_puuid, _PUUID, t_ms=8000)
            if i % 5 == 1
            else _kill(f"teammate-{i % 4}", enemy_puuid, t_ms=7000)
        ]
        won = _TEAM if i % 2 == 0 else _ENEMY_TEAM
        rounds.append(_round(i, won, kills))
    return _build_match(rounds, players), _PUUID


_SCENARIO_BUILDERS = {
    "first_contact_deaths": _scenario_first_contact_deaths,
    "first_contact_bloods": _scenario_first_contact_bloods,
    "side_split_gap": _scenario_side_split_gap,
    "low_utility": _scenario_low_utility,
    "average_match": _scenario_average_match,
}


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class CaseSpec:
    id: str
    description: str
    scenario: str
    expected_findings: list[str] = field(default_factory=list)
    must_not_find: list[str] = field(default_factory=list)


@dataclass
class CaseResult:
    case: CaseSpec
    passed: bool
    found_tags: list[str]
    violations: list[str]
    elapsed_s: float
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.case.id,
            "description": self.case.description,
            "passed": self.passed,
            "elapsed_s": round(self.elapsed_s, 3),
            "found_tags": self.found_tags,
            "violations": self.violations,
            "error": self.error,
        }


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def load_cases(path: Path) -> list[CaseSpec]:
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return [
        CaseSpec(
            id=c["id"],
            description=c.get("description", c["id"]),
            scenario=c["scenario"],
            expected_findings=c.get("expected_findings", []),
            must_not_find=c.get("must_not_find", []),
        )
        for c in data.get("cases", [])
    ]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run_case(case: CaseSpec) -> CaseResult:
    builder = _SCENARIO_BUILDERS.get(case.scenario)
    if builder is None:
        return CaseResult(
            case=case,
            passed=False,
            found_tags=[],
            violations=[f"Unknown scenario: {case.scenario!r}"],
            elapsed_s=0.0,
            error=f"Unknown scenario: {case.scenario!r}",
        )

    try:
        from valocoach.stats.post_game import (
            analyze_first_contact,
            analyze_side_split,
            analyze_utility_efficiency,
        )

        t0 = time.monotonic()
        match, puuid = builder()
        findings = []
        findings.extend(analyze_first_contact(match, puuid))
        findings.extend(analyze_side_split(match, puuid))
        findings.extend(analyze_utility_efficiency(match, puuid))
        elapsed = time.monotonic() - t0

        found_tags = [f.root_cause_tag for f in findings]

        violations: list[str] = []
        for expected in case.expected_findings:
            if expected not in found_tags:
                violations.append(f"MISSING expected finding: {expected!r}")
        for forbidden in case.must_not_find:
            if forbidden in found_tags:
                violations.append(f"PRESENT forbidden finding: {forbidden!r}")

        return CaseResult(
            case=case,
            passed=len(violations) == 0,
            found_tags=found_tags,
            violations=violations,
            elapsed_s=elapsed,
        )

    except Exception as exc:
        return CaseResult(
            case=case,
            passed=False,
            found_tags=[],
            violations=[f"RUNNER ERROR: {exc}"],
            elapsed_s=0.0,
            error=str(exc),
        )


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def _print_result(result: CaseResult) -> None:
    icon = f"{GREEN}PASS{RESET}" if result.passed else f"{RED}FAIL{RESET}"
    elapsed = f"{DIM}{result.elapsed_s:.3f}s{RESET}"
    print(f"  {icon}  [{elapsed}]  {result.case.id}  — {result.case.description}")
    if result.found_tags:
        print(f"          {DIM}findings: {', '.join(result.found_tags)}{RESET}")
    if not result.passed:
        for v in result.violations:
            print(f"          {YELLOW}{v}{RESET}")
        if result.error:
            print(f"          {RED}error: {result.error}{RESET}")


def _print_summary(results: list[CaseResult]) -> None:
    total = len(results)
    passed = sum(1 for r in results if r.passed)
    failed = total - passed
    color = GREEN if failed == 0 else RED
    print()
    print(
        f"{BOLD}Results:{RESET}  {color}{passed}/{total} passed{RESET}"
        + (f"  ({RED}{failed} failed{RESET})" if failed else "")
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run post-game analyzer eval against synthetic fixture matches."
    )
    parser.add_argument(
        "--cases",
        metavar="FILE",
        default=str(DEFAULT_CASES),
        help=f"Path to post_game_cases YAML (default: {DEFAULT_CASES.name}).",
    )
    parser.add_argument("--fail-fast", action="store_true", help="Stop after the first failure.")
    parser.add_argument("--output", metavar="FILE", help="Write full results as JSON to this file.")
    args = parser.parse_args()

    cases_path = Path(args.cases)
    if not cases_path.exists():
        print(f"Cases file not found: {cases_path}", file=sys.stderr)
        sys.exit(2)

    try:
        cases = load_cases(cases_path)
    except Exception as exc:
        print(f"Failed to load cases: {exc}", file=sys.stderr)
        sys.exit(2)

    print(f"\n{BOLD}ValoCoach Post-Game Analyzer Eval — {len(cases)} case(s){RESET}\n")

    results: list[CaseResult] = []
    for case in cases:
        result = run_case(case)
        results.append(result)
        _print_result(result)
        if args.fail_fast and not result.passed:
            print(f"\n{RED}Stopping after first failure (--fail-fast).{RESET}")
            break

    _print_summary(results)

    if args.output:
        out_path = Path(args.output)
        out_path.write_text(
            json.dumps([r.as_dict() for r in results], indent=2),
            encoding="utf-8",
        )
        print(f"Results written to {out_path}")

    sys.exit(0 if all(r.passed for r in results) else 1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""LLM evaluation harness for ValoCoach coaching output.

Loads ``tests/eval/scenarios.yaml``, runs each scenario through the real
``run_coach`` pipeline (requires Ollama running + knowledge base seeded),
then checks the response against ``must_contain`` / ``must_not_contain``
assertions.

Usage
-----
    # Full run (all scenarios, default model)
    python scripts/run_eval.py

    # Run a single scenario by ID
    python scripts/run_eval.py --scenario jett_dash_angles

    # Fail fast on first failure
    python scripts/run_eval.py --fail-fast

    # Override model
    python scripts/run_eval.py --model qwen3:14b

    # Save results to JSON
    python scripts/run_eval.py --output eval_results.json

Exit code
---------
    0 — all assertions passed
    1 — one or more assertions failed
    2 — runner error (bad YAML, Ollama unreachable, etc.)

Output
------
Each scenario prints a pass/fail summary line.  On failure, the violated
assertions are printed below.  At the end a totals row shows pass/fail/skip.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
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
SCENARIOS_PATH = REPO_ROOT / "tests" / "eval" / "scenarios.yaml"

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class Scenario:
    id: str
    description: str
    situation: str
    agent: str | None = None
    map: str | None = None
    side: str | None = None
    must_contain: list[str] = field(default_factory=list)
    must_not_contain: list[str] = field(default_factory=list)


@dataclass
class ScenarioResult:
    scenario: Scenario
    passed: bool
    response: str
    violations: list[str]
    elapsed_s: float
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.scenario.id,
            "description": self.scenario.description,
            "passed": self.passed,
            "elapsed_s": round(self.elapsed_s, 2),
            "violations": self.violations,
            "error": self.error,
            "response_snippet": self.response[:500] if self.response else None,
        }


# ---------------------------------------------------------------------------
# Scenario loader
# ---------------------------------------------------------------------------


def load_scenarios(path: Path) -> list[Scenario]:
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)

    scenarios = []
    for raw in data.get("scenarios", []):
        scenarios.append(
            Scenario(
                id=raw["id"],
                description=raw.get("description", raw["id"]),
                situation=raw["situation"],
                agent=raw.get("agent"),
                map=raw.get("map"),
                side=raw.get("side"),
                must_contain=raw.get("must_contain", []),
                must_not_contain=raw.get("must_not_contain", []),
            )
        )
    return scenarios


# ---------------------------------------------------------------------------
# Assertion evaluation
# ---------------------------------------------------------------------------


def _check_assertions(response: str, scenario: Scenario) -> list[str]:
    """Return a list of violation messages (empty = all pass)."""
    violations: list[str] = []
    lower = response.lower()

    for phrase in scenario.must_contain:
        if phrase.lower() not in lower:
            violations.append(f"MISSING  must_contain: {phrase!r}")

    for phrase in scenario.must_not_contain:
        if phrase.lower() in lower:
            violations.append(f"PRESENT  must_not_contain: {phrase!r}")

    return violations


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run_scenario(scenario: Scenario, model: str | None) -> ScenarioResult:
    """Execute one scenario through the real run_coach pipeline."""
    # Add repo src to path so run_coach is importable without installing.
    if str(REPO_ROOT / "src") not in sys.path:
        sys.path.insert(0, str(REPO_ROOT / "src"))

    try:
        from valocoach.cli.commands.coach import run_coach
        from valocoach.core.config import load_settings

        settings = load_settings()
        if model:
            object.__setattr__(settings, "ollama_model", model)

        t0 = time.monotonic()
        response = run_coach(
            situation=scenario.situation,
            agent=scenario.agent,
            map_=scenario.map,
            side=scenario.side,
            with_stats=False,  # disable stats for deterministic eval runs
        )
        elapsed = time.monotonic() - t0

        response_text = response or ""
        violations = _check_assertions(response_text, scenario)
        return ScenarioResult(
            scenario=scenario,
            passed=len(violations) == 0,
            response=response_text,
            violations=violations,
            elapsed_s=elapsed,
        )

    except Exception as exc:  # noqa: BLE001
        return ScenarioResult(
            scenario=scenario,
            passed=False,
            response="",
            violations=[f"RUNNER ERROR: {exc}"],
            elapsed_s=0.0,
            error=str(exc),
        )


# ---------------------------------------------------------------------------
# Terminal output
# ---------------------------------------------------------------------------

RESET = "\033[0m"
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BOLD = "\033[1m"
DIM = "\033[2m"


def _print_result(result: ScenarioResult) -> None:
    icon = f"{GREEN}PASS{RESET}" if result.passed else f"{RED}FAIL{RESET}"
    elapsed = f"{DIM}{result.elapsed_s:.1f}s{RESET}"
    print(f"  {icon}  [{elapsed}]  {result.scenario.id}  — {result.scenario.description}")
    if not result.passed:
        for v in result.violations:
            print(f"          {YELLOW}{v}{RESET}")
        if result.error:
            print(f"          {RED}error: {result.error}{RESET}")


def _print_summary(results: list[ScenarioResult]) -> None:
    total = len(results)
    passed = sum(1 for r in results if r.passed)
    failed = total - passed
    color = GREEN if failed == 0 else RED
    print()
    print(f"{BOLD}Results:{RESET}  {color}{passed}/{total} passed{RESET}"
          + (f"  ({RED}{failed} failed{RESET})" if failed else ""))


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run LLM eval scenarios against the ValoCoach coaching pipeline."
    )
    parser.add_argument(
        "--scenario", metavar="ID",
        help="Run a single scenario by its ID (default: all)."
    )
    parser.add_argument(
        "--fail-fast", action="store_true",
        help="Stop after the first failure."
    )
    parser.add_argument(
        "--model", metavar="NAME",
        help="Override the Ollama model for this run (e.g. qwen3:14b)."
    )
    parser.add_argument(
        "--output", metavar="FILE",
        help="Write full results as JSON to this file."
    )
    parser.add_argument(
        "--scenarios-file", metavar="PATH",
        default=str(SCENARIOS_PATH),
        help=f"Path to scenarios YAML (default: {SCENARIOS_PATH})."
    )
    args = parser.parse_args()

    scenarios_path = Path(args.scenarios_file)
    if not scenarios_path.exists():
        print(f"Scenarios file not found: {scenarios_path}", file=sys.stderr)
        sys.exit(2)

    try:
        all_scenarios = load_scenarios(scenarios_path)
    except Exception as exc:
        print(f"Failed to load scenarios: {exc}", file=sys.stderr)
        sys.exit(2)

    if args.scenario:
        filtered = [s for s in all_scenarios if s.id == args.scenario]
        if not filtered:
            known = ", ".join(s.id for s in all_scenarios)
            print(f"Unknown scenario ID {args.scenario!r}. Known: {known}", file=sys.stderr)
            sys.exit(2)
        all_scenarios = filtered

    print(f"\n{BOLD}ValoCoach LLM Eval — {len(all_scenarios)} scenario(s){RESET}\n")

    results: list[ScenarioResult] = []
    for scenario in all_scenarios:
        result = run_scenario(scenario, model=args.model)
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

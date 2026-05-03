"""Offline tests for the LLM eval harness (scripts/run_eval.py).

These tests verify the harness infrastructure — YAML loading, assertion
evaluation, result model — without running a real LLM.  The live
``run_eval.py --scenario <id>`` tests require Ollama and are excluded
from the standard pytest run.
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import pytest

# Ensure scripts/ is importable without installing as a package.
REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from run_eval import Scenario, ScenarioResult, _check_assertions, load_scenarios


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def scenarios_yaml(tmp_path: Path) -> Path:
    """Write a minimal scenarios.yaml and return its path."""
    content = textwrap.dedent("""
        scenarios:
          - id: test_agent
            description: Test agent abilities
            situation: Use Jett on Ascent A site
            agent: Jett
            map: Ascent
            must_contain:
              - Tailwind
              - A Long
            must_not_contain:
              - Paranoia
              - Bladestorm

          - id: test_map
            description: Test map callouts
            situation: Execute onto B site on Bind
            map: Bind
            must_contain:
              - Hookah
            must_not_contain:
              - B Garage

          - id: no_assertions
            description: Scenario with no assertions
            situation: General coaching question
    """)
    p = tmp_path / "scenarios.yaml"
    p.write_text(content)
    return p


@pytest.fixture()
def basic_scenario() -> Scenario:
    return Scenario(
        id="basic",
        description="Basic test",
        situation="Use Jett on Ascent",
        agent="Jett",
        map="Ascent",
        must_contain=["Tailwind", "A Long"],
        must_not_contain=["Paranoia", "Bladestorm"],
    )


# ---------------------------------------------------------------------------
# load_scenarios
# ---------------------------------------------------------------------------


class TestLoadScenarios:
    def test_loads_all_scenarios(self, scenarios_yaml: Path):
        scenarios = load_scenarios(scenarios_yaml)
        assert len(scenarios) == 3

    def test_id_and_situation_parsed(self, scenarios_yaml: Path):
        scenarios = load_scenarios(scenarios_yaml)
        first = scenarios[0]
        assert first.id == "test_agent"
        assert "Ascent" in first.situation

    def test_must_contain_list_parsed(self, scenarios_yaml: Path):
        scenarios = load_scenarios(scenarios_yaml)
        first = scenarios[0]
        assert "Tailwind" in first.must_contain
        assert "A Long" in first.must_contain

    def test_must_not_contain_list_parsed(self, scenarios_yaml: Path):
        scenarios = load_scenarios(scenarios_yaml)
        first = scenarios[0]
        assert "Paranoia" in first.must_not_contain

    def test_agent_and_map_parsed(self, scenarios_yaml: Path):
        scenarios = load_scenarios(scenarios_yaml)
        first = scenarios[0]
        assert first.agent == "Jett"
        assert first.map == "Ascent"

    def test_optional_fields_default_to_none_or_empty(self, scenarios_yaml: Path):
        scenarios = load_scenarios(scenarios_yaml)
        no_assert = scenarios[2]
        assert no_assert.agent is None
        assert no_assert.map is None
        assert no_assert.must_contain == []
        assert no_assert.must_not_contain == []

    def test_loads_real_scenarios_yaml(self):
        """The bundled scenarios.yaml must parse without error."""
        real_path = REPO_ROOT / "tests" / "eval" / "scenarios.yaml"
        assert real_path.exists(), "tests/eval/scenarios.yaml not found"
        scenarios = load_scenarios(real_path)
        assert len(scenarios) >= 20, (
            f"Expected at least 20 scenarios, got {len(scenarios)}"
        )

    def test_real_scenarios_have_ids(self):
        real_path = REPO_ROOT / "tests" / "eval" / "scenarios.yaml"
        scenarios = load_scenarios(real_path)
        ids = [s.id for s in scenarios]
        assert len(ids) == len(set(ids)), "Scenario IDs must be unique"

    def test_real_scenarios_all_have_assertions(self):
        """Every scenario should have at least one assertion."""
        real_path = REPO_ROOT / "tests" / "eval" / "scenarios.yaml"
        scenarios = load_scenarios(real_path)
        missing = [
            s.id for s in scenarios
            if not s.must_contain and not s.must_not_contain
        ]
        assert not missing, f"Scenarios missing assertions: {missing}"


# ---------------------------------------------------------------------------
# _check_assertions
# ---------------------------------------------------------------------------


class TestCheckAssertions:
    def test_passes_when_all_must_contain_present(self, basic_scenario: Scenario):
        response = "Use Tailwind to dash into A Long for the entry."
        violations = _check_assertions(response, basic_scenario)
        assert violations == []

    def test_fails_when_must_contain_missing(self, basic_scenario: Scenario):
        response = "Dash around the map somewhere."  # no "Tailwind" or "A Long"
        violations = _check_assertions(response, basic_scenario)
        assert any("Tailwind" in v for v in violations)
        assert any("A Long" in v for v in violations)

    def test_fails_when_must_not_contain_present(self, basic_scenario: Scenario):
        response = "Use Tailwind, A Long is key. Also throw a Paranoia flash."
        violations = _check_assertions(response, basic_scenario)
        # Paranoia is in must_not_contain — should fail.
        assert any("Paranoia" in v for v in violations)

    def test_case_insensitive_must_contain(self, basic_scenario: Scenario):
        """Assertions are case-insensitive."""
        response = "use tailwind to dash into a long"
        violations = _check_assertions(response, basic_scenario)
        assert violations == []

    def test_case_insensitive_must_not_contain(self, basic_scenario: Scenario):
        response = "Use Tailwind on A Long but avoid PARANOIA because wrong agent."
        violations = _check_assertions(response, basic_scenario)
        assert any("Paranoia" in v for v in violations)

    def test_no_assertions_always_passes(self):
        scenario = Scenario(
            id="empty",
            description="No assertions",
            situation="anything",
        )
        violations = _check_assertions("some response text", scenario)
        assert violations == []

    def test_violation_message_indicates_direction(self, basic_scenario: Scenario):
        """Violation messages must clearly state whether a phrase was missing or unwanted."""
        response = "Use Paranoia and forget the rest."  # missing Tailwind/A Long, has Paranoia
        violations = _check_assertions(response, basic_scenario)

        missing_tags = [v for v in violations if "MISSING" in v]
        present_tags = [v for v in violations if "PRESENT" in v]

        assert len(missing_tags) == 2  # Tailwind, A Long
        assert len(present_tags) == 1  # Paranoia

    def test_multiple_must_not_contain_violations(self):
        scenario = Scenario(
            id="multi",
            description="Multiple must_not_contain",
            situation="test",
            must_not_contain=["Paranoia", "Bladestorm", "Updraft"],
        )
        response = "Use Paranoia, then Bladestorm, and maybe Updraft."
        violations = _check_assertions(response, scenario)
        assert len(violations) == 3


# ---------------------------------------------------------------------------
# ScenarioResult
# ---------------------------------------------------------------------------


class TestScenarioResult:
    def _make_result(self, passed: bool, violations: list[str]) -> ScenarioResult:
        scenario = Scenario(id="x", description="y", situation="z")
        return ScenarioResult(
            scenario=scenario,
            passed=passed,
            response="some response text",
            violations=violations,
            elapsed_s=1.23,
        )

    def test_as_dict_includes_id(self):
        r = self._make_result(True, [])
        d = r.as_dict()
        assert d["id"] == "x"

    def test_as_dict_includes_passed_flag(self):
        r = self._make_result(False, ["MISSING x"])
        assert r.as_dict()["passed"] is False

    def test_as_dict_includes_violations(self):
        r = self._make_result(False, ["MISSING Tailwind"])
        assert "MISSING Tailwind" in r.as_dict()["violations"]

    def test_as_dict_response_snippet_truncated(self):
        scenario = Scenario(id="x", description="y", situation="z")
        long_response = "a" * 2000
        r = ScenarioResult(
            scenario=scenario,
            passed=True,
            response=long_response,
            violations=[],
            elapsed_s=0.5,
        )
        assert len(r.as_dict()["response_snippet"]) <= 500

    def test_as_dict_elapsed_rounded(self):
        r = self._make_result(True, [])
        d = r.as_dict()
        # Should be rounded to 2 decimal places.
        assert d["elapsed_s"] == round(1.23, 2)

    def test_as_dict_error_none_by_default(self):
        r = self._make_result(True, [])
        assert r.as_dict()["error"] is None

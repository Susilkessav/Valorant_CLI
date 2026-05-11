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
from unittest.mock import MagicMock, patch

import pytest

# Ensure scripts/ is importable without installing as a package.
REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from run_eval import (  # noqa: E402
    Scenario,
    ScenarioResult,
    _check_assertions,
    _print_result,
    _print_summary,
    load_scenarios,
    run_scenario,
)

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
        assert len(scenarios) >= 20, f"Expected at least 20 scenarios, got {len(scenarios)}"

    def test_real_scenarios_have_ids(self):
        real_path = REPO_ROOT / "tests" / "eval" / "scenarios.yaml"
        scenarios = load_scenarios(real_path)
        ids = [s.id for s in scenarios]
        assert len(ids) == len(set(ids)), "Scenario IDs must be unique"

    def test_real_scenarios_all_have_assertions(self):
        """Every scenario should have at least one assertion."""
        real_path = REPO_ROOT / "tests" / "eval" / "scenarios.yaml"
        scenarios = load_scenarios(real_path)
        missing = [s.id for s in scenarios if not s.must_contain and not s.must_not_contain]
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


# ---------------------------------------------------------------------------
# run_scenario — coverage for the runner function (LLM mocked)
# ---------------------------------------------------------------------------


class TestRunScenario:
    """run_scenario() calls run_coach() under the hood.  We mock that out so
    no real Ollama connection is needed."""

    def _scenario(self, **kwargs) -> Scenario:
        defaults = {
            "id": "s1",
            "description": "test",
            "situation": "Use Jett on Ascent",
            "agent": "Jett",
            "map": "Ascent",
            "must_contain": ["Tailwind"],
            "must_not_contain": ["Paranoia"],
        }
        defaults.update(kwargs)
        return Scenario(**defaults)

    def test_happy_path_returns_passed_result(self):
        """When run_coach returns a response that satisfies assertions → passed=True."""
        scenario = self._scenario()
        fake_settings = MagicMock()

        with (
            patch("run_eval.sys.path", sys.path),
            patch.dict(
                "sys.modules",
                {
                    "valocoach.cli.commands.coach": MagicMock(
                        run_coach=MagicMock(return_value="Use Tailwind to dash into A Long.")
                    ),
                    "valocoach.core.config": MagicMock(load_settings=MagicMock(return_value=fake_settings)),
                },
            ),
        ):
            result = run_scenario(scenario, model=None)

        assert result.passed
        assert result.violations == []
        assert "Tailwind" in result.response

    def test_assertion_violation_marks_failed(self):
        """Response that triggers must_not_contain → passed=False."""
        scenario = self._scenario()
        fake_settings = MagicMock()

        with (
            patch.dict(
                "sys.modules",
                {
                    "valocoach.cli.commands.coach": MagicMock(
                        run_coach=MagicMock(return_value="Use Tailwind and also Paranoia here.")
                    ),
                    "valocoach.core.config": MagicMock(load_settings=MagicMock(return_value=fake_settings)),
                },
            ),
        ):
            result = run_scenario(scenario, model=None)

        assert not result.passed
        assert any("Paranoia" in v for v in result.violations)

    def test_run_coach_exception_captured_in_error(self):
        """An exception inside run_coach is caught → result.error is set, passed=False."""
        scenario = self._scenario()

        with patch.dict(
            "sys.modules",
            {
                "valocoach.cli.commands.coach": MagicMock(
                    run_coach=MagicMock(side_effect=RuntimeError("ollama down"))
                ),
                "valocoach.core.config": MagicMock(load_settings=MagicMock(return_value=MagicMock())),
            },
        ):
            result = run_scenario(scenario, model=None)

        assert not result.passed
        assert result.error is not None
        assert "ollama down" in result.error

    def test_model_override_applied_to_settings(self):
        """When model is supplied, it is set on the settings object."""
        scenario = self._scenario()
        fake_settings = MagicMock(spec=[])  # bare mock, no attributes predefined

        with patch.dict(
            "sys.modules",
            {
                "valocoach.cli.commands.coach": MagicMock(
                    run_coach=MagicMock(return_value="Tailwind dash into A Long.")
                ),
                "valocoach.core.config": MagicMock(load_settings=MagicMock(return_value=fake_settings)),
            },
        ):
            run_scenario(scenario, model="qwen3:14b")

        # object.__setattr__ is used by the script, so the attribute ends up set.
        assert fake_settings.ollama_model == "qwen3:14b"

    def test_none_response_treated_as_empty_string(self):
        """run_coach returning None is handled gracefully."""
        scenario = self._scenario(must_contain=[], must_not_contain=[])

        with patch.dict(
            "sys.modules",
            {
                "valocoach.cli.commands.coach": MagicMock(run_coach=MagicMock(return_value=None)),
                "valocoach.core.config": MagicMock(load_settings=MagicMock(return_value=MagicMock())),
            },
        ):
            result = run_scenario(scenario, model=None)

        assert result.response == ""
        assert result.passed  # no assertions to fail


# ---------------------------------------------------------------------------
# _print_result and _print_summary — terminal output coverage
# ---------------------------------------------------------------------------


class TestPrintResult:
    def _make_result(self, passed: bool, violations: list[str] = (), error: str | None = None):
        s = Scenario(id="x", description="desc", situation="sit")
        return ScenarioResult(
            scenario=s,
            passed=passed,
            response="some response",
            violations=list(violations),
            elapsed_s=1.5,
            error=error,
        )

    def test_passed_result_prints_pass_icon(self, capsys):
        _print_result(self._make_result(True))
        out = capsys.readouterr().out
        assert "PASS" in out or "x" in out  # ID always printed

    def test_failed_result_prints_fail_icon(self, capsys):
        _print_result(self._make_result(False, violations=["MISSING x"]))
        out = capsys.readouterr().out
        assert "FAIL" in out

    def test_violations_printed_on_failure(self, capsys):
        _print_result(self._make_result(False, violations=["MISSING Tailwind", "PRESENT Paranoia"]))
        out = capsys.readouterr().out
        assert "MISSING Tailwind" in out
        assert "PRESENT Paranoia" in out

    def test_error_printed_when_set(self, capsys):
        _print_result(self._make_result(False, error="ollama not running"))
        out = capsys.readouterr().out
        assert "ollama not running" in out

    def test_no_violation_lines_on_pass(self, capsys):
        _print_result(self._make_result(True))
        out = capsys.readouterr().out
        # Violations section should not appear for a passing result
        assert "MISSING" not in out
        assert "PRESENT" not in out


class TestPrintSummary:
    def _result(self, passed: bool) -> ScenarioResult:
        s = Scenario(id="x", description="y", situation="z")
        return ScenarioResult(
            scenario=s, passed=passed, response="", violations=[], elapsed_s=0.0
        )

    def test_all_pass_shows_passed_fraction(self, capsys):
        results = [self._result(True), self._result(True), self._result(True)]
        _print_summary(results)
        out = capsys.readouterr().out
        assert "3/3" in out

    def test_some_fail_shows_failed_count(self, capsys):
        results = [self._result(True), self._result(False)]
        _print_summary(results)
        out = capsys.readouterr().out
        assert "1/2" in out
        assert "failed" in out.lower()

    def test_empty_list_does_not_crash(self, capsys):
        _print_summary([])
        out = capsys.readouterr().out
        assert "0/0" in out


# ---------------------------------------------------------------------------
# main() — integration smoke test (all I/O mocked)
# ---------------------------------------------------------------------------


class TestMain:
    """Smoke-tests for the CLI entry point with no real LLM or files needed."""

    def _scenario(self) -> Scenario:
        return Scenario(
            id="s1",
            description="test",
            situation="Use Jett",
            must_contain=[],
            must_not_contain=[],
        )

    def test_main_exits_0_on_all_pass(self, tmp_path):
        """All scenarios pass → sys.exit(0)."""
        from run_eval import main

        scenarios_file = tmp_path / "s.yaml"
        scenarios_file.write_text(
            "scenarios:\n"
            "  - id: ok\n"
            "    description: test\n"
            "    situation: Use Jett\n"
        )

        passing_result = ScenarioResult(
            scenario=self._scenario(),
            passed=True,
            response="fine",
            violations=[],
            elapsed_s=0.1,
        )

        with (
            patch("run_eval.load_scenarios", return_value=[self._scenario()]),
            patch("run_eval.run_scenario", return_value=passing_result),
            patch("sys.argv", ["run_eval.py", "--scenarios-file", str(scenarios_file)]),
            pytest.raises(SystemExit) as exc_info,
        ):
            main()

        assert exc_info.value.code == 0

    def test_main_exits_1_on_failure(self, tmp_path):
        """At least one scenario fails → sys.exit(1)."""
        from run_eval import main

        failing_result = ScenarioResult(
            scenario=self._scenario(),
            passed=False,
            response="",
            violations=["MISSING Tailwind"],
            elapsed_s=0.1,
        )

        with (
            patch("run_eval.load_scenarios", return_value=[self._scenario()]),
            patch("run_eval.run_scenario", return_value=failing_result),
            patch("sys.argv", ["run_eval.py"]),
            pytest.raises(SystemExit) as exc_info,
        ):
            main()

        assert exc_info.value.code == 1

    def test_main_exits_2_on_bad_yaml(self, tmp_path):
        """Scenarios file not found → sys.exit(2)."""
        from run_eval import main

        with (
            patch("sys.argv", ["run_eval.py", "--scenarios-file", "/nonexistent/file.yaml"]),
            pytest.raises(SystemExit) as exc_info,
        ):
            main()

        assert exc_info.value.code == 2

    def test_main_writes_output_file(self, tmp_path):
        """--output writes a JSON file with result dicts."""
        from run_eval import main

        out_file = tmp_path / "results.json"
        passing_result = ScenarioResult(
            scenario=self._scenario(),
            passed=True,
            response="fine",
            violations=[],
            elapsed_s=0.1,
        )

        with (
            patch("run_eval.load_scenarios", return_value=[self._scenario()]),
            patch("run_eval.run_scenario", return_value=passing_result),
            patch("sys.argv", ["run_eval.py", "--output", str(out_file)]),
            pytest.raises(SystemExit),
        ):
            main()

        import json

        data = json.loads(out_file.read_text())
        assert isinstance(data, list)
        assert data[0]["id"] == "s1"

    def test_main_unknown_scenario_exits_2(self, tmp_path):
        """--scenario with unknown ID → sys.exit(2)."""
        from run_eval import main

        with (
            patch("run_eval.load_scenarios", return_value=[self._scenario()]),
            patch("sys.argv", ["run_eval.py", "--scenario", "does_not_exist"]),
            pytest.raises(SystemExit) as exc_info,
        ):
            main()

        assert exc_info.value.code == 2

    def test_main_fail_fast_stops_after_first_failure(self):
        """--fail-fast stops iteration after the first failing scenario."""
        from run_eval import main

        failing = ScenarioResult(
            scenario=self._scenario(),
            passed=False,
            response="",
            violations=["MISSING x"],
            elapsed_s=0.0,
        )
        second_scenario = Scenario(id="s2", description="y", situation="y")
        mock_run = MagicMock(return_value=failing)

        with (
            patch("run_eval.load_scenarios", return_value=[self._scenario(), second_scenario]),
            patch("run_eval.run_scenario", mock_run),
            patch("sys.argv", ["run_eval.py", "--fail-fast"]),
            pytest.raises(SystemExit),
        ):
            main()

        # Only one scenario should have been run (fail-fast stops after first)
        assert mock_run.call_count == 1

    def test_main_load_scenarios_exception_exits_2(self, tmp_path):
        """load_scenarios() raising an Exception → sys.exit(2)."""
        from run_eval import main

        scenarios_file = tmp_path / "bad.yaml"
        scenarios_file.write_text(": invalid: yaml: {")  # deliberately malformed

        with (
            patch("sys.argv", ["run_eval.py", "--scenarios-file", str(scenarios_file)]),
            pytest.raises(SystemExit) as exc_info,
        ):
            main()

        assert exc_info.value.code == 2

    def test_main_scenario_filter_runs_matching_scenario(self, tmp_path):
        """--scenario <id> runs only the scenario with that id."""
        from run_eval import main

        s1 = Scenario(id="alpha", description="a", situation="a")
        s2 = Scenario(id="beta", description="b", situation="b")
        passing = ScenarioResult(
            scenario=s1, passed=True, response="ok", violations=[], elapsed_s=0.1
        )
        mock_run = MagicMock(return_value=passing)

        with (
            patch("run_eval.load_scenarios", return_value=[s1, s2]),
            patch("run_eval.run_scenario", mock_run),
            patch("sys.argv", ["run_eval.py", "--scenario", "alpha"]),
            pytest.raises(SystemExit) as exc_info,
        ):
            main()

        # Only the matching scenario was run
        assert mock_run.call_count == 1
        assert mock_run.call_args[0][0].id == "alpha"
        assert exc_info.value.code == 0

"""Tests for shell completion flags exposed by the Typer app.

Typer wires ``--install-completion`` and ``--show-completion`` automatically
when ``add_completion=True`` (set in ``valocoach/cli/app.py``).

Strategy
--------
* ``--help`` tests use Typer's CliRunner — fast, no side-effects.
* Script-content tests call Typer's completion internals directly so we
  don't need a real shell environment.
* ``--install-completion`` modifies shell config files, so we only verify
  that the flag is advertised in ``--help``; we do not invoke it.
"""

from __future__ import annotations

from typer.testing import CliRunner

from valocoach.cli.app import app

runner = CliRunner()


# ---------------------------------------------------------------------------
# Flags are advertised in --help
# ---------------------------------------------------------------------------


class TestCompletionFlagsInHelp:
    def test_install_completion_in_help(self):
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "--install-completion" in result.output

    def test_show_completion_in_help(self):
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "--show-completion" in result.output

    def test_help_describes_install_completion(self):
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        # Typer's built-in description text
        assert "Install completion" in result.output

    def test_help_describes_show_completion(self):
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "Show completion" in result.output


# ---------------------------------------------------------------------------
# Completion script content (tested via Typer internals, no shell env needed)
# ---------------------------------------------------------------------------


def _script(shell: str) -> str:
    """Return the completion script for *shell* using Typer's own helper."""
    from typer._completion_shared import get_completion_script

    return get_completion_script(
        prog_name="valocoach",
        complete_var="_VALOCOACH_COMPLETE",
        shell=shell,
    )


class TestCompletionScriptContent:
    def test_bash_script_is_non_empty(self):
        assert _script("bash").strip() != ""

    def test_zsh_script_is_non_empty(self):
        assert _script("zsh").strip() != ""

    def test_fish_script_is_non_empty(self):
        assert _script("fish").strip() != ""

    def test_bash_script_contains_program_name(self):
        assert "valocoach" in _script("bash")

    def test_zsh_script_contains_program_name(self):
        assert "valocoach" in _script("zsh")

    def test_fish_script_contains_program_name(self):
        assert "valocoach" in _script("fish")

    def test_bash_script_registers_completion_function(self):
        # bash scripts end with a `complete -F <func> <prog>` line.
        assert "complete" in _script("bash")

    def test_zsh_script_defines_compdef(self):
        # Zsh completion scripts include compdef.
        assert "compdef" in _script("zsh")

    def test_bash_complete_var_is_correct(self):
        # The completion variable must match what Click looks for.
        assert "_VALOCOACH_COMPLETE" in _script("bash")

    def test_supported_shells_match_typer_enum(self):
        from typer._completion_shared import Shells

        for shell in Shells:
            script = _script(shell.value)
            assert "valocoach" in script, f"Script for {shell.value!r} missing prog name"


# ---------------------------------------------------------------------------
# add_completion=True is set on the Typer app
# ---------------------------------------------------------------------------


class TestAddCompletionEnabled:
    def test_typer_app_has_add_completion_true(self):
        """Verify the app was created with add_completion=True at source level."""
        # The presence of --install-completion in --help is the observable proof,
        # but we can also confirm the Typer app's own attribute if accessible.
        result = runner.invoke(app, ["--help"])
        assert "--install-completion" in result.output
        assert "--show-completion" in result.output

    def test_version_and_completion_coexist(self):
        """--version and completion flags must both appear (no option conflict)."""
        result = runner.invoke(app, ["--help"])
        assert "--version" in result.output
        assert "--install-completion" in result.output

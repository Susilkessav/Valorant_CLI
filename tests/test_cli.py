from __future__ import annotations

from typer.testing import CliRunner

from valocoach.cli.app import app

runner = CliRunner()


def test_version():
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "valocoach" in result.stdout


def test_help_shows_all_commands():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for cmd in ["coach", "stats", "sync", "profile", "meta", "patch", "interactive", "config"]:
        assert cmd in result.stdout


def test_unimplemented_stub_exits_cleanly():
    # `patch` and `interactive` are still stubs — pick one to track stub→real migrations.
    result = runner.invoke(app, ["patch"])
    assert result.exit_code == 0
    assert "not implemented" in result.stdout.lower()


def test_meta_command_runs_without_args():
    result = runner.invoke(app, ["meta"])
    assert result.exit_code == 0
    assert "tier" in result.stdout.lower()


def test_meta_command_map_haven():
    result = runner.invoke(app, ["meta", "--map", "Haven"])
    assert result.exit_code == 0
    assert "Haven" in result.stdout
    assert "A Long" in result.stdout


def test_meta_command_agent_omen():
    result = runner.invoke(app, ["meta", "--agent", "Omen"])
    assert result.exit_code == 0
    assert "Omen" in result.stdout
    assert "Dark Cover" in result.stdout
    assert "ult pts" in result.stdout

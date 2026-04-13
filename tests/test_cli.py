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
    result = runner.invoke(app, ["stats"])
    assert result.exit_code == 0
    assert "not implemented" in result.stdout.lower()

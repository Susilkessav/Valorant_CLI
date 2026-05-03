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


def test_patch_command_exits_cleanly():
    # `patch` is now a real command — verify it exits 0 with mocked DB.
    from unittest.mock import AsyncMock, MagicMock, patch as mock_patch

    with (
        mock_patch("valocoach.data.database.ensure_db", new_callable=AsyncMock),
        mock_patch(
            "valocoach.retrieval.patch_tracker.get_current_patch",
            new_callable=AsyncMock,
            return_value="10.09",
        ),
        mock_patch("valocoach.core.config.load_settings", return_value=MagicMock()),
    ):
        result = runner.invoke(app, ["patch"])
    assert result.exit_code == 0
    assert "10.09" in result.stdout


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

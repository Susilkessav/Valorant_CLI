from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from unittest.mock import patch as mock_patch

from typer.testing import CliRunner

from valocoach.cli.app import app

runner = CliRunner()

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_P_ENSURE_DB = "valocoach.data.database.ensure_db"
_P_SYNC = "valocoach.data.sync.sync_player_matches"
_P_SETTINGS = "valocoach.core.config.load_settings"


def _fake_settings():
    s = MagicMock()
    s.riot_name = "T"
    s.riot_tag = "X"
    s.data_dir = MagicMock()
    return s


def _sync_result(
    *,
    matches_new: int = 1,
    matches_skipped: int = 0,
    errors: list | None = None,
    ok: bool = True,
    error: str | None = None,
):
    return SimpleNamespace(
        matches_new=matches_new,
        matches_skipped=matches_skipped,
        errors=errors or [],
        ok=ok,
        error=error,
    )


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
    from unittest.mock import AsyncMock, MagicMock
    from unittest.mock import patch as mock_patch

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


# ---------------------------------------------------------------------------
# sync command — internal branch coverage (lines 126-128, 131, 139, 142)
# ---------------------------------------------------------------------------


def test_sync_already_up_to_date():
    """matches_new=0 + no errors → 'Already up to date' message (line 131)."""
    with (
        mock_patch(_P_SETTINGS, return_value=_fake_settings()),
        mock_patch(_P_ENSURE_DB, new_callable=AsyncMock),
        mock_patch(
            _P_SYNC,
            new_callable=AsyncMock,
            return_value=_sync_result(matches_new=0, errors=[]),
        ),
    ):
        result = runner.invoke(app, ["sync"])
    assert result.exit_code == 0
    assert "Already up to date" in result.stdout


def test_sync_reports_new_matches():
    """matches_new > 0 → 'Sync complete' success message."""
    with (
        mock_patch(_P_SETTINGS, return_value=_fake_settings()),
        mock_patch(_P_ENSURE_DB, new_callable=AsyncMock),
        mock_patch(
            _P_SYNC,
            new_callable=AsyncMock,
            return_value=_sync_result(matches_new=5, matches_skipped=2),
        ),
    ):
        result = runner.invoke(app, ["sync"])
    assert result.exit_code == 0
    assert "5" in result.stdout


def test_sync_warns_on_partial_errors():
    """result.errors non-empty → each error is displayed via warn (line 139)."""
    with (
        mock_patch(_P_SETTINGS, return_value=_fake_settings()),
        mock_patch(_P_ENSURE_DB, new_callable=AsyncMock),
        mock_patch(
            _P_SYNC,
            new_callable=AsyncMock,
            return_value=_sync_result(
                matches_new=2,
                errors=["match-abc failed: timeout"],
                ok=True,
            ),
        ),
    ):
        result = runner.invoke(app, ["sync"])
    assert "match-abc failed: timeout" in result.stdout


def test_sync_warns_on_not_ok():
    """result.ok=False → 'Sync finished with errors' warning (line 142)."""
    with (
        mock_patch(_P_SETTINGS, return_value=_fake_settings()),
        mock_patch(_P_ENSURE_DB, new_callable=AsyncMock),
        mock_patch(
            _P_SYNC,
            new_callable=AsyncMock,
            return_value=_sync_result(
                matches_new=0,
                errors=[],
                ok=False,
                error="some fatal error",
            ),
        ),
    ):
        result = runner.invoke(app, ["sync"])
    assert "Sync finished with errors" in result.stdout
    assert "some fatal error" in result.stdout


def test_sync_errors_on_sync_error():
    """SyncError → display.error + exit 1 (lines 126-128)."""
    from valocoach.core.exceptions import SyncError

    with (
        mock_patch(_P_SETTINGS, return_value=_fake_settings()),
        mock_patch(_P_ENSURE_DB, new_callable=AsyncMock),
        mock_patch(_P_SYNC, new_callable=AsyncMock, side_effect=SyncError("network down")),
    ):
        result = runner.invoke(app, ["sync"])
    assert result.exit_code == 1
    assert "network down" in result.stdout


# ---------------------------------------------------------------------------
# config show — secret redaction
# ---------------------------------------------------------------------------


def test_config_show_redacts_api_key():
    """config show must never print the raw henrikdev_api_key value."""
    from unittest.mock import patch as mock_patch

    fake_settings = MagicMock()
    fake_settings.model_dump.return_value = {
        "riot_name": "TestPlayer",
        "henrikdev_api_key": "super-secret-key-abc123",
        "ollama_model": "qwen3:8b",
    }

    with mock_patch("valocoach.core.config.load_settings", return_value=fake_settings):
        result = runner.invoke(app, ["config", "show"])

    assert result.exit_code == 0
    assert "super-secret-key-abc123" not in result.stdout
    assert "redacted" in result.stdout


def test_config_show_empty_api_key_not_redacted():
    """An empty henrikdev_api_key should not be changed to 'redacted'."""
    from unittest.mock import patch as mock_patch

    fake_settings = MagicMock()
    fake_settings.model_dump.return_value = {
        "riot_name": "TestPlayer",
        "henrikdev_api_key": "",  # empty — user hasn't set it yet
        "ollama_model": "qwen3:8b",
    }

    with mock_patch("valocoach.core.config.load_settings", return_value=fake_settings):
        result = runner.invoke(app, ["config", "show"])

    assert result.exit_code == 0
    assert "redacted" not in result.stdout


def test_config_init_creates_file():
    """config init must create a default config and report its path."""
    from unittest.mock import patch as mock_patch
    from pathlib import Path

    with mock_patch(
        "valocoach.core.config.write_default_config",
        return_value=Path("/tmp/test-config.toml"),
    ):
        result = runner.invoke(app, ["config", "init"])

    assert result.exit_code == 0
    assert "test-config.toml" in result.stdout


# ---------------------------------------------------------------------------
# index command (lines 220-229)
# ---------------------------------------------------------------------------


def test_index_corpus_dir_missing(tmp_path, monkeypatch):
    """corpus/ absent → error + exit 1 (lines 222-223)."""
    monkeypatch.chdir(tmp_path)  # no corpus/ dir here
    result = runner.invoke(app, ["index"])
    assert result.exit_code == 1
    assert "corpus/" in result.stdout or "corpus" in result.stdout


def test_index_corpus_dir_present(tmp_path, monkeypatch):
    """corpus/ present → run_ingest called (line 229)."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    monkeypatch.chdir(tmp_path)

    with mock_patch("valocoach.cli.commands.ingest.run_ingest") as ingest_mock:
        result = runner.invoke(app, ["index"])

    assert result.exit_code == 0
    ingest_mock.assert_called_once()

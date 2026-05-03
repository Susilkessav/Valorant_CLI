"""Tests for ``valocoach patch`` command.

Patch targets
-------------
All dependencies are lazily imported inside ``run_patch`` / ``_check_and_refresh``,
so we patch at the *source* module (the place the symbol lives), not at the
command module.

  valocoach.data.database.ensure_db          → AsyncMock
  valocoach.retrieval.patch_tracker.get_current_patch    → AsyncMock
  valocoach.retrieval.patch_tracker.check_patch_update   → AsyncMock
  valocoach.core.config.load_settings        → MagicMock (sync)
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from typer.testing import CliRunner

from valocoach.cli.app import app

runner = CliRunner()

# Patch targets (source modules, not the command module).
_ENSURE_DB = "valocoach.data.database.ensure_db"
_GET_PATCH = "valocoach.retrieval.patch_tracker.get_current_patch"
_CHECK_PATCH = "valocoach.retrieval.patch_tracker.check_patch_update"
_LOAD_SETTINGS = "valocoach.core.config.load_settings"


def _invoke(version: str | None = "10.09"):
    """Invoke ``valocoach patch`` with mocked DB/API returning *version*."""
    with (
        patch(_ENSURE_DB, new_callable=AsyncMock),
        patch(_GET_PATCH, new_callable=AsyncMock, return_value=version),
        patch(_LOAD_SETTINGS, return_value=MagicMock()),
    ):
        return runner.invoke(app, ["patch"])


def _invoke_check(version: str = "10.09", is_new: bool = False, fail: bool = False):
    """Invoke ``valocoach patch --check`` with mocked DB/API."""
    if fail:
        check_mock = AsyncMock(side_effect=RuntimeError("network error"))
    else:
        check_mock = AsyncMock(return_value=(version, is_new))

    with (
        patch(_ENSURE_DB, new_callable=AsyncMock),
        patch(_CHECK_PATCH, check_mock),
        patch(_GET_PATCH, new_callable=AsyncMock, return_value=version),
        patch(_LOAD_SETTINGS, return_value=MagicMock()),
    ):
        return runner.invoke(app, ["patch", "--check"])


# ---------------------------------------------------------------------------
# patch command — offline (no --check)
# ---------------------------------------------------------------------------


class TestPatchCommandOffline:
    def test_exits_zero_with_patch_version(self):
        result = _invoke("10.09")
        assert result.exit_code == 0

    def test_shows_patch_version_in_output(self):
        result = _invoke("10.09")
        assert "10.09" in result.output

    def test_shows_valorant_patch_label(self):
        result = _invoke("10.09")
        output_lower = result.output.lower()
        assert "patch" in output_lower or "version" in output_lower

    def test_no_data_shows_helpful_message(self):
        result = _invoke(None)
        assert result.exit_code == 0
        out = result.output.lower()
        # Should suggest running sync or --check
        assert "sync" in out or "check" in out or "no patch" in out

    def test_help_flag(self):
        result = runner.invoke(app, ["patch", "--help"])
        assert result.exit_code == 0
        assert "--check" in result.output

    def test_help_describes_check_flag(self):
        result = runner.invoke(app, ["patch", "--help"])
        assert "Refresh" in result.output or "Henrik" in result.output


# ---------------------------------------------------------------------------
# patch command — with --check
# ---------------------------------------------------------------------------


class TestPatchCommandWithCheck:
    def test_new_patch_shows_new_message(self):
        result = _invoke_check(version="10.10", is_new=True)
        assert result.exit_code == 0
        out = result.output.lower()
        assert "new" in out or "10.10" in result.output

    def test_unchanged_patch_shows_unchanged_message(self):
        result = _invoke_check(version="10.09", is_new=False)
        assert result.exit_code == 0
        out = result.output.lower()
        assert "unchanged" in out or "10.09" in result.output

    def test_api_failure_does_not_crash(self):
        result = _invoke_check(fail=True)
        assert result.exit_code == 0

    def test_api_failure_shows_warning(self):
        result = _invoke_check(fail=True)
        out = result.output.lower()
        assert "could not" in out or "warn" in out or "error" in out or "network" in out

    def test_check_calls_check_patch_update(self):
        check_mock = AsyncMock(return_value=("10.09", False))
        with (
            patch(_ENSURE_DB, new_callable=AsyncMock),
            patch(_CHECK_PATCH, check_mock),
            patch(_GET_PATCH, new_callable=AsyncMock, return_value="10.09"),
            patch(_LOAD_SETTINGS, return_value=MagicMock()),
        ):
            runner.invoke(app, ["patch", "--check"])
        check_mock.assert_awaited_once()

    def test_no_check_does_not_call_check_patch_update(self):
        check_mock = AsyncMock(return_value=("10.09", False))
        with (
            patch(_ENSURE_DB, new_callable=AsyncMock),
            patch(_CHECK_PATCH, check_mock),
            patch(_GET_PATCH, new_callable=AsyncMock, return_value="10.09"),
            patch(_LOAD_SETTINGS, return_value=MagicMock()),
        ):
            runner.invoke(app, ["patch"])
        check_mock.assert_not_awaited()

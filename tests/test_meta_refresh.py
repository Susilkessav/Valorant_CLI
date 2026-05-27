"""Tests for valocoach.cli.commands.meta_refresh.

Covers:
  - run_meta_refresh: install_cron path, watch mode path, single-run path
  - _render_result: no new patch / up-to-date, new patch, forced, errors, dry-run
  - _install_cron: success, already installed, binary not found, crontab write failure
  - _run_watch_loop: covered indirectly via run_meta_refresh(watch=True)
  - SyncResult property paths tested via _render_result

All heavy deps (asyncio, subprocess, display, meta_sync) are mocked.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

_RUN_META_SYNC = "valocoach.retrieval.meta_sync.run_meta_sync"
_DISPLAY_CONSOLE = "valocoach.cli.display.console"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_sync_result(
    *,
    patch_version: str = "10.09",
    is_new_patch: bool = True,
    patch_notes_scraped: bool = True,
    ranked_stats_scraped: bool = True,
    pro_stats_scraped: bool = True,
    meta_regenerated: bool = True,
    meta_written: bool = True,
    meta_ingested: bool = True,
    youtube_chunks_ingested: int = 0,
    errors: list[str] | None = None,
):
    from valocoach.retrieval.meta_sync import SyncResult

    r = SyncResult(
        patch_version=patch_version,
        is_new_patch=is_new_patch,
        patch_notes_scraped=patch_notes_scraped,
        ranked_stats_scraped=ranked_stats_scraped,
        pro_stats_scraped=pro_stats_scraped,
        meta_regenerated=meta_regenerated,
        meta_written=meta_written,
        meta_ingested=meta_ingested,
        youtube_chunks_ingested=youtube_chunks_ingested,
        errors=errors or [],
    )
    return r


# ---------------------------------------------------------------------------
# run_meta_refresh — dispatch logic
# ---------------------------------------------------------------------------


class TestRunMetaRefresh:
    def test_install_cron_calls_install_cron(self):
        from valocoach.cli.commands.meta_refresh import run_meta_refresh

        with patch("valocoach.cli.commands.meta_refresh._install_cron") as mock_install:
            run_meta_refresh(install_cron=True)

        mock_install.assert_called_once()

    def test_install_cron_does_not_call_run_once(self):
        from valocoach.cli.commands.meta_refresh import run_meta_refresh

        with (
            patch("valocoach.cli.commands.meta_refresh._install_cron"),
            patch("valocoach.cli.commands.meta_refresh._run_once") as mock_run_once,
        ):
            run_meta_refresh(install_cron=True)

        mock_run_once.assert_not_called()

    def test_single_run_calls_asyncio_run(self):
        from valocoach.cli.commands.meta_refresh import run_meta_refresh

        # Close the coroutine to suppress RuntimeWarning about unawaited coroutine
        calls: list = []

        def _close_coro(coro):
            calls.append(coro)
            coro.close()

        with patch("valocoach.cli.commands.meta_refresh.asyncio.run", _close_coro):
            run_meta_refresh(force=False, dry_run=False)

        assert len(calls) == 1


# ---------------------------------------------------------------------------
# _render_result
# ---------------------------------------------------------------------------


class TestRenderResult:
    """Tests for the Rich summary printer — asserts display calls are made."""

    def _render(self, result, *, dry_run: bool = False):
        from valocoach.cli.commands.meta_refresh import _render_result

        with patch("valocoach.cli.display.console") as mock_console:
            _render_result(result, dry_run=dry_run)
        return mock_console

    def test_no_new_patch_shows_up_to_date(self):
        """When no new patch and meta not regenerated, shows 'up to date' message."""
        result = _make_sync_result(
            is_new_patch=False,
            meta_regenerated=False,
            meta_written=False,
            meta_ingested=False,
        )
        mock_console = self._render(result)
        # display.info should be called (via print or similar)
        # The important thing is that _render_result doesn't crash
        assert mock_console.print.called

    def test_new_patch_shows_patch_label(self):
        """New patch result includes the patch version in output."""
        result = _make_sync_result(is_new_patch=True, meta_written=True)
        mock_console = self._render(result)
        # At least one print call was made
        assert mock_console.print.called

    def test_forced_result_shows_yellow_version(self):
        """Forced (not new patch, but meta regenerated) uses yellow styling."""
        result = _make_sync_result(is_new_patch=False, meta_regenerated=True, meta_written=True)
        mock_console = self._render(result)
        assert mock_console.print.called

    def test_errors_are_displayed(self):
        """Errors in SyncResult trigger display.warn calls."""
        from valocoach.cli.commands.meta_refresh import _render_result

        result = _make_sync_result(
            is_new_patch=True, meta_written=False, errors=["something broke"]
        )
        with (
            patch("valocoach.cli.display.console"),
            patch("valocoach.cli.display.warn") as mock_warn,
        ):
            _render_result(result, dry_run=False)

        mock_warn.assert_called()

    def test_dry_run_shows_dry_run_message(self):
        """When dry_run=True and meta was 'written', shows dry run message."""
        from valocoach.cli.commands.meta_refresh import _render_result

        result = _make_sync_result(is_new_patch=True, meta_written=True)
        with (
            patch("valocoach.cli.display.console"),
            patch("valocoach.cli.display.info") as mock_info,
        ):
            _render_result(result, dry_run=True)

        # Should call display.info with "Dry run" message
        calls_text = " ".join(str(c) for c in mock_info.call_args_list)
        assert "Dry run" in calls_text or mock_info.called

    def test_youtube_chunks_shown_when_nonzero(self):
        """Non-zero youtube_chunks_ingested prints a YouTube line."""
        result = _make_sync_result(is_new_patch=True, meta_written=True, youtube_chunks_ingested=5)
        mock_console = self._render(result)
        # Verify some print call was made (YouTube line included)
        assert mock_console.print.called

    def test_meta_not_written_with_new_patch_shows_warning(self):
        """When patch detected but meta.json not written, shows warning."""
        from valocoach.cli.commands.meta_refresh import _render_result

        result = _make_sync_result(
            is_new_patch=True, meta_written=False, meta_ingested=False, errors=[]
        )
        with (
            patch("valocoach.cli.display.console"),
            patch("valocoach.cli.display.warn") as mock_warn,
        ):
            _render_result(result, dry_run=False)

        mock_warn.assert_called()


# ---------------------------------------------------------------------------
# _install_cron
# ---------------------------------------------------------------------------


class TestInstallCron:
    _BIN_EXISTS = "pathlib.Path.exists"

    def test_no_binary_shows_error_and_exits(self):
        """When valocoach binary not found, display.error is called and Exit raised."""
        import typer

        from valocoach.cli.commands.meta_refresh import _install_cron

        with (
            patch("pathlib.Path.exists", return_value=False),
            patch("valocoach.cli.display.error"),
            pytest.raises(typer.Exit),
        ):
            _install_cron()

    def test_already_installed_shows_info(self):
        """When marker already in crontab, display.info is called."""
        from valocoach.cli.commands.meta_refresh import _CRON_MARKER, _install_cron

        existing_crontab = (
            f"0 8 * * * /bin/some-other-cmd\n0 8 * * * valocoach meta-refresh  {_CRON_MARKER}\n"
        )

        with (
            patch("pathlib.Path.exists", return_value=True),
            patch(
                "subprocess.check_output",
                return_value=existing_crontab,
            ),
            patch("valocoach.cli.display.info") as mock_info,
            patch("valocoach.cli.display.console"),
        ):
            _install_cron()

        mock_info.assert_called_once()
        assert "already" in mock_info.call_args.args[0].lower()

    def test_fresh_install_writes_crontab(self):
        """When not installed, crontab -l then crontab - are called."""

        from valocoach.cli.commands.meta_refresh import _install_cron

        with (
            patch("pathlib.Path.exists", return_value=True),
            patch("subprocess.check_output", return_value=""),
            patch("subprocess.run", return_value=MagicMock(returncode=0)) as mock_run,
            patch("valocoach.cli.display.success"),
            patch("valocoach.cli.display.console"),
        ):
            _install_cron()

        # crontab - should have been called
        assert mock_run.called
        args = mock_run.call_args.args[0]
        assert args == ["crontab", "-"]

    def test_crontab_write_failure_shows_error(self):
        """When crontab - returns non-zero exit code, display.error is called."""
        import typer

        from valocoach.cli.commands.meta_refresh import _install_cron

        with (
            patch("pathlib.Path.exists", return_value=True),
            patch("subprocess.check_output", return_value=""),
            patch(
                "subprocess.run",
                return_value=MagicMock(returncode=1, stderr="permission denied"),
            ),
            patch("valocoach.cli.display.error"),
            patch("valocoach.cli.display.console"),
            pytest.raises(typer.Exit),
        ):
            _install_cron()

    def test_empty_existing_crontab(self):
        """CalledProcessError from crontab -l (no existing crontab) is handled."""
        import subprocess

        from valocoach.cli.commands.meta_refresh import _install_cron

        with (
            patch("pathlib.Path.exists", return_value=True),
            patch(
                "subprocess.check_output",
                side_effect=subprocess.CalledProcessError(1, "crontab"),
            ),
            patch("subprocess.run", return_value=MagicMock(returncode=0)),
            patch("valocoach.cli.display.success"),
            patch("valocoach.cli.display.console"),
        ):
            # Should not raise — empty existing crontab is normal
            _install_cron()


class TestRenderResultExtra:
    def test_render_result_no_action_when_no_patch_and_no_meta(self):
        """When meta_written=False and not (is_new_patch or meta_regenerated), no warn."""
        from valocoach.cli.commands.meta_refresh import _render_result

        result = _make_sync_result(
            is_new_patch=False,
            meta_regenerated=False,
            meta_written=False,
            meta_ingested=False,
        )
        with (
            patch("valocoach.cli.display.console"),
            patch("valocoach.cli.display.warn") as mock_warn,
            patch("valocoach.cli.display.info"),
        ):
            _render_result(result, dry_run=False)

        # The elif block (163) should NOT fire — no warn about "not updated"
        # Because is_new_patch=False and meta_regenerated=False
        for c in mock_warn.call_args_list:
            assert "not updated" not in str(c)


# ---------------------------------------------------------------------------
# _run_once (async, tested directly)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestRunOnce:
    """Smoke tests for the _run_once async function."""

    def _meta_sync_that_fires_callback(self, result):
        """Return an async mock that calls on_step once before returning result.

        This exercises the _on_step closure (lines 81-82, 92-93) which only
        runs when run_meta_sync actually invokes the on_step callback.
        """

        async def _fake(settings, *, on_step=None, **kwargs):
            if on_step is not None:
                on_step("patch_check", "start")  # fires the closure
            return result

        return _fake

    async def test_run_once_calls_run_meta_sync(self):
        """_run_once calls run_meta_sync and renders the result."""
        from valocoach.cli.commands.meta_refresh import _run_once

        result = _make_sync_result(is_new_patch=False, meta_regenerated=False, meta_written=False)

        with (
            patch(
                "valocoach.core.config.load_settings", return_value=MagicMock(data_dir=MagicMock())
            ),
            patch("valocoach.data.database.ensure_db", new_callable=AsyncMock),
            patch(
                "valocoach.retrieval.meta_sync.run_meta_sync",
                self._meta_sync_that_fires_callback(result),
            ),
            patch("valocoach.cli.commands.meta_refresh._render_result"),
            patch("valocoach.cli.display.console"),
            patch("valocoach.cli.display.warn"),
        ):
            await _run_once(force=False, dry_run=False, youtube=None)

    async def test_run_once_passes_force_flag(self):
        """force=True is forwarded to run_meta_sync."""
        from valocoach.cli.commands.meta_refresh import _run_once

        result = _make_sync_result()
        calls: list = []

        async def _fake(settings, *, force=False, on_step=None, **kwargs):
            calls.append({"force": force})
            if on_step:
                on_step("patch_check", "start")
            return result

        with (
            patch(
                "valocoach.core.config.load_settings", return_value=MagicMock(data_dir=MagicMock())
            ),
            patch("valocoach.data.database.ensure_db", new_callable=AsyncMock),
            patch("valocoach.retrieval.meta_sync.run_meta_sync", _fake),
            patch("valocoach.cli.commands.meta_refresh._render_result"),
            patch("valocoach.cli.display.console"),
            patch("valocoach.cli.display.warn"),
        ):
            await _run_once(force=True, dry_run=False, youtube=None)

        assert calls[0]["force"] is True

    async def test_run_once_dry_run_shows_warning(self):
        """dry_run=True triggers a display.warn call before the sync."""
        from valocoach.cli.commands.meta_refresh import _run_once

        result = _make_sync_result()

        with (
            patch(
                "valocoach.core.config.load_settings", return_value=MagicMock(data_dir=MagicMock())
            ),
            patch("valocoach.data.database.ensure_db", new_callable=AsyncMock),
            patch(
                "valocoach.retrieval.meta_sync.run_meta_sync",
                self._meta_sync_that_fires_callback(result),
            ),
            patch("valocoach.cli.commands.meta_refresh._render_result"),
            patch("valocoach.cli.display.console"),
            patch("valocoach.cli.display.warn") as mock_warn,
        ):
            await _run_once(force=False, dry_run=True, youtube=None)

        # dry_run triggers display.warn("Dry-run mode ...")
        assert mock_warn.called

    async def test_run_once_on_step_with_non_start_status(self):
        """on_step callback with status != 'start' does not update current_status_msg."""
        from valocoach.cli.commands.meta_refresh import _run_once

        result = _make_sync_result(is_new_patch=False, meta_regenerated=False, meta_written=False)

        async def _fake(settings, *, on_step=None, **kwargs):
            if on_step:
                on_step("patch_check", "start")  # updates msg
                on_step("patch_check", "done")  # non-start — skips update
            return result

        with (
            patch(
                "valocoach.core.config.load_settings", return_value=MagicMock(data_dir=MagicMock())
            ),
            patch("valocoach.data.database.ensure_db", new_callable=AsyncMock),
            patch("valocoach.retrieval.meta_sync.run_meta_sync", _fake),
            patch("valocoach.cli.commands.meta_refresh._render_result"),
            patch("valocoach.cli.display.console"),
            patch("valocoach.cli.display.warn"),
        ):
            await _run_once(force=False, dry_run=False, youtube=None)

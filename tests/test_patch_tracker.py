"""Unit tests for valocoach.retrieval.patch_tracker.

Covers the core logic of check_patch_update:
  - First run (no previous PatchVersion): records row, returns is_new=True,
    does NOT call invalidate_volatile (nothing to evict).
  - Patch changed (previous version differs): records new row, calls
    invalidate_volatile, returns is_new=True.
  - Patch unchanged (same version): does not record, does not invalidate,
    returns is_new=False.

All DB access and HTTP calls are mocked so no real DB or network is needed.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Patch targets (source-module level so lazy imports are captured correctly)
# ---------------------------------------------------------------------------

_HENRIK_CLIENT = "valocoach.retrieval.patch_tracker.HenrikClient"
_SESSION_SCOPE = "valocoach.retrieval.patch_tracker.session_scope"
_INVALIDATE = "valocoach.retrieval.patch_tracker.invalidate_volatile"


def _make_patch_version(game_version: str) -> MagicMock:
    pv = MagicMock()
    pv.game_version = game_version
    return pv


def _make_session_scope(existing_row):
    """Return an async context manager that yields a mock DB session.

    ``existing_row`` is what ``s.scalar(...)`` returns (None or a PatchVersion).
    ``s.add`` is kept synchronous (it is synchronous in SQLAlchemy).
    """
    mock_session = AsyncMock()
    mock_session.scalar = AsyncMock(return_value=existing_row)
    mock_session.add = MagicMock()  # sync, not async

    @asynccontextmanager
    async def _scope():
        yield mock_session

    return _scope


def _make_henrik_client(version: str):
    """Return a mock HenrikClient context manager that returns *version*."""
    client = AsyncMock()
    client.get_version = AsyncMock(return_value={"version": version})

    @asynccontextmanager
    async def _ctx(settings):
        yield client

    return _ctx


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestCheckPatchUpdate:
    async def _call(self, current_version: str, existing_row, invalidate_mock=None):
        """Helper: call check_patch_update with fully mocked deps.

        ``invalidate_mock`` defaults to an AsyncMock that returns 0 so
        tests that don't care about invalidation never hit the real cache DB.
        """
        from valocoach.retrieval.patch_tracker import check_patch_update

        settings = MagicMock()
        settings.riot_region = "na"

        if invalidate_mock is None:
            invalidate_mock = AsyncMock(return_value=0)

        with (
            patch(_HENRIK_CLIENT, _make_henrik_client(current_version)),
            patch(_SESSION_SCOPE, _make_session_scope(existing_row)),
            patch(_INVALIDATE, invalidate_mock),
        ):
            return await check_patch_update(settings)

    # ------------------------------------------------------------------
    # First run — no previous PatchVersion in DB
    # ------------------------------------------------------------------

    async def test_first_run_returns_current_version(self):
        version, _ = await self._call("10.09", existing_row=None)
        assert version == "10.09"

    async def test_first_run_returns_is_new_true(self):
        _, is_new = await self._call("10.09", existing_row=None)
        assert is_new is True

    async def test_first_run_does_not_call_invalidate_volatile(self):
        invalidate_mock = AsyncMock(return_value=0)
        await self._call("10.09", existing_row=None, invalidate_mock=invalidate_mock)
        invalidate_mock.assert_not_awaited()

    # ------------------------------------------------------------------
    # Patch changed — previous version exists but differs
    # ------------------------------------------------------------------

    async def test_patch_changed_returns_new_version(self):
        existing = _make_patch_version("10.08")
        version, _ = await self._call("10.09", existing_row=existing)
        assert version == "10.09"

    async def test_patch_changed_returns_is_new_true(self):
        existing = _make_patch_version("10.08")
        _, is_new = await self._call("10.09", existing_row=existing)
        assert is_new is True

    async def test_patch_changed_calls_invalidate_volatile(self):
        existing = _make_patch_version("10.08")
        invalidate_mock = AsyncMock(return_value=3)
        await self._call("10.09", existing_row=existing, invalidate_mock=invalidate_mock)
        invalidate_mock.assert_awaited_once()

    # ------------------------------------------------------------------
    # Patch unchanged — same version already recorded
    # ------------------------------------------------------------------

    async def test_unchanged_returns_current_version(self):
        existing = _make_patch_version("10.09")
        version, _ = await self._call("10.09", existing_row=existing)
        assert version == "10.09"

    async def test_unchanged_returns_is_new_false(self):
        existing = _make_patch_version("10.09")
        _, is_new = await self._call("10.09", existing_row=existing)
        assert is_new is False

    async def test_unchanged_does_not_call_invalidate_volatile(self):
        existing = _make_patch_version("10.09")
        invalidate_mock = AsyncMock(return_value=0)
        await self._call("10.09", existing_row=existing, invalidate_mock=invalidate_mock)
        invalidate_mock.assert_not_awaited()


# ---------------------------------------------------------------------------
# get_patch_staleness_days  (sync wrapper)
# ---------------------------------------------------------------------------


class TestGetPatchStalenessDays:
    """Tests for the sync staleness-days helper.

    The function calls asyncio.run() internally, so we mock the two async
    dependencies (ensure_db + session_scope) at source level.
    """

    _ENSURE_DB = "valocoach.data.database.ensure_db"

    def _fake_session_scope(self, detected_at: str | None):
        """Build a session_scope mock that returns a PatchVersion with detected_at."""
        from contextlib import asynccontextmanager

        if detected_at is None:
            mock_pv = None
        else:
            mock_pv = MagicMock()
            mock_pv.detected_at = detected_at

        mock_session = AsyncMock()
        mock_session.scalar = AsyncMock(return_value=mock_pv)

        @asynccontextmanager
        async def _scope():
            yield mock_session

        return _scope

    def test_returns_none_when_no_patch_recorded(self):
        from pathlib import Path

        from valocoach.retrieval.patch_tracker import get_patch_staleness_days

        with (
            patch(self._ENSURE_DB, new_callable=AsyncMock),
            patch(_SESSION_SCOPE, self._fake_session_scope(None)),
        ):
            result = get_patch_staleness_days(Path("/tmp/fake"))

        assert result is None

    def test_returns_zero_for_fresh_patch(self):
        """detected_at = now → staleness ≈ 0.0."""
        from datetime import UTC, datetime
        from pathlib import Path

        from valocoach.retrieval.patch_tracker import get_patch_staleness_days

        now_iso = datetime.now(UTC).isoformat()
        with (
            patch(self._ENSURE_DB, new_callable=AsyncMock),
            patch(_SESSION_SCOPE, self._fake_session_scope(now_iso)),
        ):
            result = get_patch_staleness_days(Path("/tmp/fake"))

        assert result is not None
        assert result < 1.0  # fresh check should be < 1 day old

    def test_returns_correct_days_for_old_patch(self):
        """detected_at = 30 days ago → result ≈ 30."""
        from datetime import UTC, datetime, timedelta
        from pathlib import Path

        from valocoach.retrieval.patch_tracker import get_patch_staleness_days

        thirty_days_ago = (datetime.now(UTC) - timedelta(days=30)).isoformat()
        with (
            patch(self._ENSURE_DB, new_callable=AsyncMock),
            patch(_SESSION_SCOPE, self._fake_session_scope(thirty_days_ago)),
        ):
            result = get_patch_staleness_days(Path("/tmp/fake"))

        assert result is not None
        assert 29.9 < result < 30.1

    def test_returns_none_on_exception(self):
        """DB errors return None — never crash the coaching turn."""
        from pathlib import Path

        from valocoach.retrieval.patch_tracker import get_patch_staleness_days

        with patch(self._ENSURE_DB, new_callable=AsyncMock, side_effect=RuntimeError("no db")):
            result = get_patch_staleness_days(Path("/tmp/fake"))

        assert result is None

    def test_handles_utc_z_suffix(self):
        """ISO timestamps with a trailing 'Z' are parsed correctly."""
        from pathlib import Path

        from valocoach.retrieval.patch_tracker import get_patch_staleness_days

        # Z-suffix ISO string (common from some storage paths)
        z_timestamp = "2026-04-01T10:00:00Z"
        with (
            patch(self._ENSURE_DB, new_callable=AsyncMock),
            patch(_SESSION_SCOPE, self._fake_session_scope(z_timestamp)),
        ):
            result = get_patch_staleness_days(Path("/tmp/fake"))

        assert result is not None
        assert result > 0  # 2026-04-01 is before today (2026-05-06)

    def test_returns_float(self):
        """Return type is always float when a row exists."""
        from datetime import UTC, datetime, timedelta
        from pathlib import Path

        from valocoach.retrieval.patch_tracker import get_patch_staleness_days

        ts = (datetime.now(UTC) - timedelta(days=5)).isoformat()
        with (
            patch(self._ENSURE_DB, new_callable=AsyncMock),
            patch(_SESSION_SCOPE, self._fake_session_scope(ts)),
        ):
            result = get_patch_staleness_days(Path("/tmp/fake"))

        assert isinstance(result, float)

    def test_handles_naive_datetime_without_tzinfo(self):
        """ISO timestamp with no timezone info is treated as UTC (line 184 branch)."""
        from pathlib import Path

        from valocoach.retrieval.patch_tracker import get_patch_staleness_days

        # Bare ISO string — no 'Z', no '+00:00' — fromisoformat returns naive datetime
        naive_ts = "2026-01-01T12:00:00"
        with (
            patch(self._ENSURE_DB, new_callable=AsyncMock),
            patch(_SESSION_SCOPE, self._fake_session_scope(naive_ts)),
        ):
            result = get_patch_staleness_days(Path("/tmp/fake"))

        # 2026-01-01 is before 2026-05-11 — result should be > 100 days
        assert result is not None
        assert result > 100


# ---------------------------------------------------------------------------
# get_current_patch
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestGetCurrentPatch:
    """Tests for the async get_current_patch helper."""

    async def test_returns_version_when_row_exists(self):
        from valocoach.retrieval.patch_tracker import get_current_patch

        pv = _make_patch_version("10.09")
        scope = _make_session_scope(pv)
        with patch(_SESSION_SCOPE, scope):
            result = await get_current_patch()

        assert result == "10.09"

    async def test_returns_none_when_no_row(self):
        from valocoach.retrieval.patch_tracker import get_current_patch

        scope = _make_session_scope(None)
        with patch(_SESSION_SCOPE, scope):
            result = await get_current_patch()

        assert result is None


# ---------------------------------------------------------------------------
# run_patch_watcher
# ---------------------------------------------------------------------------

_CHECK_PATCH = "valocoach.retrieval.patch_tracker.check_patch_update"
_ASYNCIO_SLEEP = "valocoach.retrieval.patch_tracker.asyncio.sleep"


@pytest.mark.asyncio
class TestRunPatchWatcher:
    """Tests for the continuous watcher loop."""

    async def _run_one_iteration(
        self,
        *,
        version: str = "10.09",
        is_new: bool = True,
        on_new_patch=None,
        meta_sync_mock=None,
    ):
        """Run the watcher for exactly one iteration then exit.

        The first check_patch_update call returns (version, is_new).
        asyncio.sleep is a no-op.
        The second check_patch_update call raises CancelledError — this is
        caught inside the watcher's try block and causes a clean return.
        """
        import asyncio
        from contextlib import ExitStack

        settings = MagicMock()
        # First call: real result. Second call: CancelledError (caught by watcher).
        check_mock = AsyncMock(
            side_effect=[(version, is_new), asyncio.CancelledError()]
        )
        sleep_mock = AsyncMock()  # no-op — CancelledError comes from check, not sleep

        from valocoach.retrieval.patch_tracker import run_patch_watcher

        with ExitStack() as stack:
            stack.enter_context(patch(_CHECK_PATCH, check_mock))
            stack.enter_context(patch(_ASYNCIO_SLEEP, sleep_mock))
            # Patch run_meta_sync at the source module so the lazy 'from ... import' picks it up.
            if meta_sync_mock is not None:
                stack.enter_context(
                    patch("valocoach.retrieval.meta_sync.run_meta_sync", meta_sync_mock)
                )
            await run_patch_watcher(
                settings,
                check_interval_hours=1,
                on_new_patch=on_new_patch,
            )

        return check_mock

    async def test_calls_check_patch_update(self):
        """Watcher calls check_patch_update at least once."""
        check_mock = await self._run_one_iteration(is_new=False)
        check_mock.assert_awaited()

    async def test_no_patch_skips_on_new_patch_callback(self):
        """When is_new=False, on_new_patch is never called."""
        callback = MagicMock()
        await self._run_one_iteration(is_new=False, on_new_patch=callback)
        callback.assert_not_called()

    async def test_calls_sync_on_new_patch(self):
        """When is_new=True and on_new_patch is sync, it is called with the version."""
        callback = MagicMock()
        await self._run_one_iteration(version="10.10", is_new=True, on_new_patch=callback)
        callback.assert_called_once_with("10.10")

    async def test_calls_async_on_new_patch(self):
        """When is_new=True and on_new_patch is async, it is awaited with the version."""
        callback = AsyncMock()
        await self._run_one_iteration(version="10.10", is_new=True, on_new_patch=callback)
        callback.assert_awaited_once_with("10.10")

    async def test_default_callback_runs_meta_sync_on_new_patch(self):
        """When is_new=True and no callback given, run_meta_sync is called."""
        meta_result = MagicMock()
        meta_result.ok = True
        meta_result.meta_written = True
        meta_result.meta_ingested = True
        meta_mock = AsyncMock(return_value=meta_result)

        await self._run_one_iteration(version="10.10", is_new=True, meta_sync_mock=meta_mock)

        # run_meta_sync should have been called with force=True
        meta_mock.assert_awaited_once()
        _, kwargs = meta_mock.call_args
        assert kwargs.get("force") is True

    async def test_exception_does_not_crash_watcher(self):
        """Non-CancelledError exceptions are logged and the loop retries."""
        import asyncio

        settings = MagicMock()
        # First call raises RuntimeError, second call raises CancelledError to exit
        check_mock = AsyncMock(
            side_effect=[RuntimeError("network error"), asyncio.CancelledError()]
        )
        sleep_mock = AsyncMock()

        from valocoach.retrieval.patch_tracker import run_patch_watcher

        with (
            patch(_CHECK_PATCH, check_mock),
            patch(_ASYNCIO_SLEEP, sleep_mock),
        ):
            # Should not raise — exception is caught, loop retries, then CancelledError exits
            await run_patch_watcher(settings, check_interval_hours=1)

        assert check_mock.await_count == 2

    async def test_cancelled_error_exits_cleanly(self):
        """CancelledError raised inside check_patch_update exits the loop without re-raising."""
        import asyncio

        settings = MagicMock()
        check_mock = AsyncMock(side_effect=asyncio.CancelledError())

        from valocoach.retrieval.patch_tracker import run_patch_watcher

        with patch(_CHECK_PATCH, check_mock):
            # Should return normally (not propagate CancelledError)
            await run_patch_watcher(settings, check_interval_hours=1)

    async def test_meta_sync_errors_logged_not_raised(self):
        """When run_meta_sync returns result.ok=False, errors are logged but not raised."""
        meta_result = MagicMock()
        meta_result.ok = False
        meta_result.errors = ["something went wrong"]
        meta_mock = AsyncMock(return_value=meta_result)

        # Should complete without raising even when meta sync reports errors
        await self._run_one_iteration(version="10.10", is_new=True, meta_sync_mock=meta_mock)
        meta_mock.assert_awaited_once()

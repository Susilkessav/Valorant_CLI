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

        # Mock ``get_meta`` so the "patch unchanged" branch's secondary
        # check (meta.json::patch == current?) doesn't read the real
        # bundled file — when an "unchanged" test passes
        # current_version="10.09" against a meta.json pinned to "10.08",
        # the real file would falsely surface as a meta-vs-patch drift
        # and flip is_new back to True.  We mirror current_version so
        # the secondary check reports "aligned".
        from valocoach.retrieval import meta as _meta_mod

        with (
            patch(_HENRIK_CLIENT, _make_henrik_client(current_version)),
            patch(_SESSION_SCOPE, _make_session_scope(existing_row)),
            patch(_INVALIDATE, invalidate_mock),
            patch.object(_meta_mod, "get_meta", return_value={"patch": current_version}),
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

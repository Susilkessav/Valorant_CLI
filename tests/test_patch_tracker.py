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

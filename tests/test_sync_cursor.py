"""Tests for sync resume-on-interrupt (cursor checkpoint).

Covers:
  - close_stale_syncs: returns 0 when no stale logs exist.
  - close_stale_syncs: closes stale rows, sets completed_at and error="interrupted".
  - close_stale_syncs: does NOT touch the current (excluded) sync log.
  - close_stale_syncs: closes multiple stale rows at once.
  - close_stale_syncs: ignores already-completed logs.
  - close_stale_syncs: ignores logs belonging to a different puuid.
  - SyncOrchestrator._check_resume: returns False when no stale logs.
  - SyncOrchestrator._check_resume: returns True when a stale log exists.
  - SyncOrchestrator._check_resume: passes correct exclude_id.
  - SyncOrchestrator.run: _discover called with full=True when resume triggered.
  - SyncOrchestrator.run: _discover called with original full=False when no resume.
  - SyncOrchestrator.run: resume message printed when interrupted sync detected.
  - SyncOrchestrator.run: no resume message when clean start.
  - SyncOrchestrator.run: explicit full=True preserved when no resume.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

PUUID = "aaaabbbbccccdddd"


def _now() -> str:
    return datetime.now(UTC).isoformat()


# ---------------------------------------------------------------------------
# Helpers to insert SyncLog rows during tests
# ---------------------------------------------------------------------------


async def _insert_sync_log(session, puuid: str, *, completed: bool = False) -> int:
    """Insert a SyncLog row and return its id (session must already be open)."""
    from valocoach.data.orm_models import SyncLog

    log = SyncLog(puuid=puuid)
    if completed:
        log.completed_at = _now()
    session.add(log)
    await session.flush()
    return log.id


# ---------------------------------------------------------------------------
# Repository-level tests (close_stale_syncs) — use shared db_session fixture
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestCloseStaleSync:
    async def test_no_stale_logs_returns_zero(self, db_session):
        from valocoach.data.repository import close_stale_syncs

        current_id = await _insert_sync_log(db_session, PUUID)
        count = await close_stale_syncs(db_session, PUUID, exclude_id=current_id)
        assert count == 0

    async def test_stale_log_is_closed(self, db_session):
        from valocoach.data.orm_models import SyncLog
        from valocoach.data.repository import close_stale_syncs

        stale_id = await _insert_sync_log(db_session, PUUID)
        current_id = await _insert_sync_log(db_session, PUUID)

        count = await close_stale_syncs(db_session, PUUID, exclude_id=current_id)
        assert count == 1

        stale = await db_session.get(SyncLog, stale_id)
        assert stale is not None
        assert stale.completed_at is not None
        assert stale.error == "interrupted"

    async def test_current_log_not_touched(self, db_session):
        from valocoach.data.orm_models import SyncLog
        from valocoach.data.repository import close_stale_syncs

        _stale_id = await _insert_sync_log(db_session, PUUID)
        current_id = await _insert_sync_log(db_session, PUUID)

        await close_stale_syncs(db_session, PUUID, exclude_id=current_id)

        current = await db_session.get(SyncLog, current_id)
        assert current is not None
        assert current.completed_at is None
        assert current.error is None

    async def test_multiple_stale_logs_all_closed(self, db_session):
        from valocoach.data.repository import close_stale_syncs

        await _insert_sync_log(db_session, PUUID)
        await _insert_sync_log(db_session, PUUID)
        current_id = await _insert_sync_log(db_session, PUUID)

        count = await close_stale_syncs(db_session, PUUID, exclude_id=current_id)
        assert count == 2

    async def test_already_completed_logs_not_affected(self, db_session):
        from valocoach.data.repository import close_stale_syncs

        await _insert_sync_log(db_session, PUUID, completed=True)
        current_id = await _insert_sync_log(db_session, PUUID)

        count = await close_stale_syncs(db_session, PUUID, exclude_id=current_id)
        assert count == 0

    async def test_other_puuid_not_affected(self, db_session):
        from valocoach.data.repository import close_stale_syncs

        other_puuid = "zzzzyyyy"
        await _insert_sync_log(db_session, other_puuid)
        current_id = await _insert_sync_log(db_session, PUUID)

        count = await close_stale_syncs(db_session, PUUID, exclude_id=current_id)
        assert count == 0


# ---------------------------------------------------------------------------
# SyncOrchestrator._check_resume (mocked session_scope)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestCheckResume:
    def _make_orchestrator(self):
        from valocoach.data.sync import SyncOrchestrator

        client = MagicMock()
        con = MagicMock()
        con.print = MagicMock()
        return SyncOrchestrator(client, console=con)

    async def test_no_stale_returns_false(self):
        orch = self._make_orchestrator()
        with patch(
            "valocoach.data.sync.close_stale_syncs",
            new=AsyncMock(return_value=0),
        ):
            result = await orch._check_resume(PUUID, current_log_id=42)
        assert result is False

    async def test_stale_found_returns_true(self):
        orch = self._make_orchestrator()
        with patch(
            "valocoach.data.sync.close_stale_syncs",
            new=AsyncMock(return_value=1),
        ):
            result = await orch._check_resume(PUUID, current_log_id=42)
        assert result is True

    async def test_passes_correct_exclude_id(self):
        orch = self._make_orchestrator()
        mock_close = AsyncMock(return_value=0)
        with patch("valocoach.data.sync.close_stale_syncs", new=mock_close):
            await orch._check_resume(PUUID, current_log_id=99)

        _, kwargs = mock_close.call_args
        assert kwargs.get("exclude_id") == 99


# ---------------------------------------------------------------------------
# SyncOrchestrator.run — resume wiring (all phases mocked)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestRunResumeWiring:
    def _make_orchestrator(self):
        from valocoach.data.sync import SyncOrchestrator

        client = MagicMock()
        mock_con = MagicMock()
        mock_con.print = MagicMock()
        return SyncOrchestrator(client, console=mock_con), mock_con

    def _fake_account(self):
        acc = MagicMock()
        acc.puuid = PUUID
        return acc

    async def test_discover_gets_full_true_when_interrupted(self):
        orch, _ = self._make_orchestrator()
        mock_discover = AsyncMock(return_value=([], 0, 0))

        with (
            patch.object(
                orch,
                "_resolve",
                new=AsyncMock(return_value=(self._fake_account(), MagicMock())),
            ),
            patch.object(orch, "_open_log", new=AsyncMock(return_value=7)),
            patch.object(orch, "_check_resume", new=AsyncMock(return_value=True)),
            patch.object(orch, "_discover", new=mock_discover),
            patch.object(orch, "_finalise", new=AsyncMock()),
        ):
            await orch.run("na", "Player", "NA1", full=False)

        _, kwargs = mock_discover.call_args
        assert kwargs.get("full") is True

    async def test_discover_keeps_full_false_when_no_resume(self):
        orch, _ = self._make_orchestrator()
        mock_discover = AsyncMock(return_value=([], 0, 0))

        with (
            patch.object(
                orch,
                "_resolve",
                new=AsyncMock(return_value=(self._fake_account(), MagicMock())),
            ),
            patch.object(orch, "_open_log", new=AsyncMock(return_value=7)),
            patch.object(orch, "_check_resume", new=AsyncMock(return_value=False)),
            patch.object(orch, "_discover", new=mock_discover),
            patch.object(orch, "_finalise", new=AsyncMock()),
        ):
            await orch.run("na", "Player", "NA1", full=False)

        _, kwargs = mock_discover.call_args
        assert kwargs.get("full") is False

    async def test_resume_message_printed_when_interrupted(self):
        orch, mock_con = self._make_orchestrator()

        with (
            patch.object(
                orch,
                "_resolve",
                new=AsyncMock(return_value=(self._fake_account(), MagicMock())),
            ),
            patch.object(orch, "_open_log", new=AsyncMock(return_value=7)),
            patch.object(orch, "_check_resume", new=AsyncMock(return_value=True)),
            patch.object(orch, "_discover", new=AsyncMock(return_value=([], 0, 0))),
            patch.object(orch, "_finalise", new=AsyncMock()),
        ):
            await orch.run("na", "Player", "NA1")

        all_printed = " ".join(str(c) for c in mock_con.print.call_args_list)
        assert "interrupted" in all_printed.lower() or "resuming" in all_printed.lower()

    async def test_no_resume_message_when_clean(self):
        orch, mock_con = self._make_orchestrator()

        with (
            patch.object(
                orch,
                "_resolve",
                new=AsyncMock(return_value=(self._fake_account(), MagicMock())),
            ),
            patch.object(orch, "_open_log", new=AsyncMock(return_value=7)),
            patch.object(orch, "_check_resume", new=AsyncMock(return_value=False)),
            patch.object(orch, "_discover", new=AsyncMock(return_value=([], 0, 0))),
            patch.object(orch, "_finalise", new=AsyncMock()),
        ):
            await orch.run("na", "Player", "NA1")

        all_printed = " ".join(str(c) for c in mock_con.print.call_args_list)
        assert "resuming" not in all_printed.lower()

    async def test_explicit_full_true_preserved_when_no_resume(self):
        orch, _ = self._make_orchestrator()
        mock_discover = AsyncMock(return_value=([], 0, 0))

        with (
            patch.object(
                orch,
                "_resolve",
                new=AsyncMock(return_value=(self._fake_account(), MagicMock())),
            ),
            patch.object(orch, "_open_log", new=AsyncMock(return_value=7)),
            patch.object(orch, "_check_resume", new=AsyncMock(return_value=False)),
            patch.object(orch, "_discover", new=mock_discover),
            patch.object(orch, "_finalise", new=AsyncMock()),
        ):
            await orch.run("na", "Player", "NA1", full=True)

        _, kwargs = mock_discover.call_args
        assert kwargs.get("full") is True

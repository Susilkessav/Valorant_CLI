"""Unit tests for SyncOrchestrator and sync_player_matches.

All I/O is mocked:
- HenrikClient methods are replaced with AsyncMock instances.
- session_scope() is replaced with an async context manager that yields a
  reusable AsyncMock session.
- Repository functions (upsert_player, start_sync, match_exists, …) are
  patched at the valocoach.data.sync import site.

No real network calls and no real DB writes happen in this file.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from io import StringIO
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from rich.console import Console

from valocoach.core.config import Settings
from valocoach.core.exceptions import APIError, SyncError
from valocoach.data.sync import SyncOrchestrator, SyncResult, sync_player_matches

# ---------------------------------------------------------------------------
# Patch targets — import site is valocoach.data.sync (that's where the names
# are bound after "from valocoach.data.xxx import yyy").
# ---------------------------------------------------------------------------

_P_SCOPE = "valocoach.data.sync.session_scope"
_P_UPSERT_PLAYER = "valocoach.data.sync.upsert_player"
_P_START_SYNC = "valocoach.data.sync.start_sync"
_P_MATCH_EXISTS = "valocoach.data.sync.match_exists"
_P_UPSERT_DETAILS = "valocoach.data.sync.upsert_match_details"
_P_COMPLETE_SYNC = "valocoach.data.sync.complete_sync"

# Stable fake IDs
PUUID = "20905543-1b42-5f6f-8435-ab284a0094f8"
MATCH_A = "match-id-aaaa-0001"
MATCH_B = "match-id-bbbb-0002"
SYNC_LOG_ID = 99


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _quiet_console() -> Console:
    """Rich Console that writes to StringIO — zero output during tests."""
    return Console(file=StringIO(), quiet=True)


def _stored_match(match_id: str) -> MagicMock:
    """Fake StoredMatch with the minimal .match_id property."""
    sm = MagicMock()
    sm.match_id = match_id
    return sm


def _make_client(
    account_data,
    mmr_data,
    stored: list | None = None,
    details=None,
) -> AsyncMock:
    """Build a mock HenrikClient wired with provided return values."""
    client = AsyncMock()
    client.get_account.return_value = account_data
    client.get_mmr.return_value = mmr_data
    client.get_stored_matches.return_value = stored or []
    client.get_match_details.return_value = details
    return client


def _sync_log_mock() -> MagicMock:
    """Fake SyncLog row with a stable .id."""
    log = MagicMock()
    log.id = SYNC_LOG_ID
    return log


def _make_session(
    *,
    match_exists_returns: bool = False,
    upsert_returns=None,
) -> AsyncMock:
    """Build a mock AsyncSession.

    session.get() is configured to return a fake SyncLog (used by _finalise).
    """
    session = AsyncMock()
    session.get.return_value = _sync_log_mock()
    return session


@asynccontextmanager
async def _scope(session: AsyncMock):
    """Async context manager that always yields the same mock session."""
    yield session


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def account_data():
    from valocoach.data.models import AccountData

    return AccountData(
        puuid=PUUID,
        region="na",
        account_level=240,
        name="Yoursaviour01",
        tag="SK04",
    )


@pytest.fixture
def mmr_data():
    from valocoach.data.models import CurrentRankData, HighestRank, MMRData

    return MMRData(
        name="Yoursaviour01",
        tag="SK04",
        puuid=PUUID,
        current_data=CurrentRankData(
            currenttier=12,
            currenttierpatched="Gold 1",
            ranking_in_tier=0,
            elo=900,
            mmr_change_to_last_game=-13,
        ),
        highest_rank=HighestRank(tier=14, patched_tier="Gold 3", season="e6a2"),
    )


# ---------------------------------------------------------------------------
# SyncOrchestrator.run() — integration-level tests
# ---------------------------------------------------------------------------


class TestSyncOrchestratorRun:
    """Tests for the full run() method with all phases mocked."""

    async def test_happy_path_returns_correct_result(self, account_data, mmr_data, match_details):
        """Two new matches → result counts correct, puuid set."""
        session = _make_session()
        client = _make_client(
            account_data,
            mmr_data,
            stored=[_stored_match(MATCH_A), _stored_match(MATCH_B)],
            details=match_details,
        )
        fake_match = MagicMock()

        with (
            patch(_P_SCOPE, lambda: _scope(session)),
            patch(_P_UPSERT_PLAYER, new_callable=AsyncMock),
            patch(_P_START_SYNC, new_callable=AsyncMock, return_value=_sync_log_mock()),
            patch(_P_MATCH_EXISTS, new_callable=AsyncMock, return_value=False),
            patch(_P_UPSERT_DETAILS, new_callable=AsyncMock, return_value=fake_match),
            patch(_P_COMPLETE_SYNC),
        ):
            orch = SyncOrchestrator(client, console=_quiet_console())
            result = await orch.run("na", "Yoursaviour01", "SK04", limit=20)

        assert result.ok
        assert result.puuid == PUUID
        assert result.matches_fetched == 2
        assert result.matches_new == 2
        assert result.matches_skipped == 0
        assert result.errors == []

    async def test_no_new_matches_skips_fetch_phase(self, account_data, mmr_data):
        """If _discover finds nothing new, _fetch_all is never called."""
        session = _make_session()
        client = _make_client(
            account_data,
            mmr_data,
            stored=[_stored_match(MATCH_A)],
        )

        with (
            patch(_P_SCOPE, lambda: _scope(session)),
            patch(_P_UPSERT_PLAYER, new_callable=AsyncMock),
            patch(_P_START_SYNC, new_callable=AsyncMock, return_value=_sync_log_mock()),
            patch(_P_MATCH_EXISTS, new_callable=AsyncMock, return_value=True),  # already stored
            patch(_P_UPSERT_DETAILS, new_callable=AsyncMock) as mock_upsert,
            patch(_P_COMPLETE_SYNC),
        ):
            orch = SyncOrchestrator(client, console=_quiet_console())
            result = await orch.run("na", "Yoursaviour01", "SK04")

        # upsert_match_details should never have been called
        mock_upsert.assert_not_called()
        assert result.matches_new == 0
        assert result.matches_fetched == 1
        assert result.matches_skipped == 1

    async def test_per_match_api_error_is_non_fatal(self, account_data, mmr_data, match_details):
        """APIError on one match → collected in result.errors, run continues."""
        session = _make_session()
        client = _make_client(
            account_data,
            mmr_data,
            stored=[_stored_match(MATCH_A), _stored_match(MATCH_B)],
        )
        # First call raises, second succeeds
        client.get_match_details.side_effect = [
            APIError("rate limited"),
            match_details,
        ]

        with (
            patch(_P_SCOPE, lambda: _scope(session)),
            patch(_P_UPSERT_PLAYER, new_callable=AsyncMock),
            patch(_P_START_SYNC, new_callable=AsyncMock, return_value=_sync_log_mock()),
            patch(_P_MATCH_EXISTS, new_callable=AsyncMock, return_value=False),
            patch(_P_UPSERT_DETAILS, new_callable=AsyncMock, return_value=MagicMock()),
            patch(_P_COMPLETE_SYNC),
        ):
            orch = SyncOrchestrator(client, console=_quiet_console())
            result = await orch.run("na", "Yoursaviour01", "SK04")

        # Run completes normally (ok = True) but errors list is populated
        assert result.ok
        assert len(result.errors) == 1
        assert "rate limited" in result.errors[0]
        assert result.matches_new == 1  # the second match succeeded

    async def test_resolve_api_error_raises_sync_error(self, account_data, mmr_data):
        """APIError in _resolve is promoted to SyncError and propagates out."""
        client = AsyncMock()
        client.get_account.side_effect = APIError("bad key")
        client.get_mmr.return_value = mmr_data

        session = _make_session()

        with (
            patch(_P_SCOPE, lambda: _scope(session)),
            patch(_P_UPSERT_PLAYER, new_callable=AsyncMock),
            patch(_P_START_SYNC, new_callable=AsyncMock, return_value=_sync_log_mock()),
            patch(_P_COMPLETE_SYNC),
        ):
            orch = SyncOrchestrator(client, console=_quiet_console())
            with pytest.raises(SyncError, match="Cannot resolve player"):
                await orch.run("na", "Yoursaviour01", "SK04")

    async def test_discover_api_error_raises_sync_error(self, account_data, mmr_data):
        """APIError in get_stored_matches is promoted to SyncError."""
        client = _make_client(account_data, mmr_data)
        client.get_stored_matches.side_effect = APIError("503 from henrik")

        session = _make_session()

        with (
            patch(_P_SCOPE, lambda: _scope(session)),
            patch(_P_UPSERT_PLAYER, new_callable=AsyncMock),
            patch(_P_START_SYNC, new_callable=AsyncMock, return_value=_sync_log_mock()),
            patch(_P_COMPLETE_SYNC),
        ):
            orch = SyncOrchestrator(client, console=_quiet_console())
            with pytest.raises(SyncError, match="Cannot fetch match list"):
                await orch.run("na", "Yoursaviour01", "SK04")

    async def test_finalise_always_runs_after_fatal_error(self, account_data, mmr_data):
        """_finalise (SyncLog close) must execute even when _discover raises."""
        client = _make_client(account_data, mmr_data)
        client.get_stored_matches.side_effect = APIError("fatal")

        session = _make_session()
        complete_sync_mock = MagicMock()

        with (
            patch(_P_SCOPE, lambda: _scope(session)),
            patch(_P_UPSERT_PLAYER, new_callable=AsyncMock),
            patch(_P_START_SYNC, new_callable=AsyncMock, return_value=_sync_log_mock()),
            patch(_P_COMPLETE_SYNC, complete_sync_mock),
        ):
            orch = SyncOrchestrator(client, console=_quiet_console())
            with pytest.raises(SyncError):
                await orch.run("na", "Yoursaviour01", "SK04")

        # complete_sync must have been called despite the error
        complete_sync_mock.assert_called_once()

    async def test_upsert_returns_none_increments_skipped(
        self, account_data, mmr_data, match_details
    ):
        """upsert_match_details returning None (already stored) → matches_skipped += 1."""
        session = _make_session()
        client = _make_client(
            account_data,
            mmr_data,
            stored=[_stored_match(MATCH_A)],
            details=match_details,
        )

        with (
            patch(_P_SCOPE, lambda: _scope(session)),
            patch(_P_UPSERT_PLAYER, new_callable=AsyncMock),
            patch(_P_START_SYNC, new_callable=AsyncMock, return_value=_sync_log_mock()),
            patch(_P_MATCH_EXISTS, new_callable=AsyncMock, return_value=False),
            patch(_P_UPSERT_DETAILS, new_callable=AsyncMock, return_value=None),  # already stored
            patch(_P_COMPLETE_SYNC),
        ):
            orch = SyncOrchestrator(client, console=_quiet_console())
            result = await orch.run("na", "Yoursaviour01", "SK04")

        assert result.matches_new == 0
        assert result.matches_skipped == 1


# ---------------------------------------------------------------------------
# SyncOrchestrator._discover() — phase 2 in isolation
# ---------------------------------------------------------------------------


class TestDiscover:
    """Phase 2 tests — split between new and already-stored IDs."""

    async def test_new_ids_collected(self, account_data, mmr_data):
        """IDs not yet in DB are returned in new_ids list."""
        session = _make_session()
        client = _make_client(account_data, mmr_data, stored=[_stored_match(MATCH_A)])

        with (
            patch(_P_SCOPE, lambda: _scope(session)),
            patch(_P_MATCH_EXISTS, new_callable=AsyncMock, return_value=False),
        ):
            orch = SyncOrchestrator(client, console=_quiet_console())
            new_ids, fetched, skipped = await orch._discover(
                "na", "Yoursaviour01", "SK04", limit=20, full=False, mode="competitive"
            )

        assert new_ids == [MATCH_A]
        assert fetched == 1
        assert skipped == 0

    async def test_already_stored_stops_early_in_normal_mode(self, account_data, mmr_data):
        """First stored match stops early scan when full=False."""
        stored = [_stored_match(MATCH_A), _stored_match(MATCH_B)]
        session = _make_session()
        client = _make_client(account_data, mmr_data, stored=stored)

        # MATCH_A is already in DB → should stop before examining MATCH_B
        with (
            patch(_P_SCOPE, lambda: _scope(session)),
            patch(_P_MATCH_EXISTS, new_callable=AsyncMock, return_value=True),
        ):
            orch = SyncOrchestrator(client, console=_quiet_console())
            new_ids, fetched, skipped = await orch._discover(
                "na", "Yoursaviour01", "SK04", limit=20, full=False, mode="competitive"
            )

        assert new_ids == []
        assert skipped == 1  # stopped after first duplicate — MATCH_B never checked

    async def test_full_mode_scans_past_stored_ids(self, account_data, mmr_data):
        """full=True continues scanning even when a stored match is encountered."""
        stored = [_stored_match(MATCH_A), _stored_match(MATCH_B)]
        session = _make_session()
        client = _make_client(account_data, mmr_data, stored=stored)

        # MATCH_A already stored, MATCH_B is new
        exists_side_effect = [True, False]

        with (
            patch(_P_SCOPE, lambda: _scope(session)),
            patch(
                _P_MATCH_EXISTS,
                new_callable=AsyncMock,
                side_effect=exists_side_effect,
            ),
        ):
            orch = SyncOrchestrator(client, console=_quiet_console())
            new_ids, fetched, skipped = await orch._discover(
                "na", "Yoursaviour01", "SK04", limit=20, full=True, mode="competitive"
            )

        assert new_ids == [MATCH_B]
        assert fetched == 2
        assert skipped == 1

    async def test_blank_match_id_is_skipped(self, account_data, mmr_data):
        """StoredMatch with empty match_id is silently skipped."""
        blank = MagicMock()
        blank.match_id = ""
        session = _make_session()
        client = _make_client(account_data, mmr_data, stored=[blank, _stored_match(MATCH_A)])

        with (
            patch(_P_SCOPE, lambda: _scope(session)),
            patch(_P_MATCH_EXISTS, new_callable=AsyncMock, return_value=False),
        ):
            orch = SyncOrchestrator(client, console=_quiet_console())
            new_ids, fetched, skipped = await orch._discover(
                "na", "Yoursaviour01", "SK04", limit=20, full=False, mode="competitive"
            )

        assert new_ids == [MATCH_A]  # blank skipped transparently

    async def test_discover_raises_sync_error_on_api_error(self, account_data, mmr_data):
        """APIError from get_stored_matches is wrapped in SyncError."""
        client = _make_client(account_data, mmr_data)
        client.get_stored_matches.side_effect = APIError("503")

        orch = SyncOrchestrator(client, console=_quiet_console())
        with pytest.raises(SyncError, match="Cannot fetch match list"):
            await orch._discover(
                "na", "Yoursaviour01", "SK04", limit=20, full=False, mode="competitive"
            )


# ---------------------------------------------------------------------------
# SyncOrchestrator._resolve() — phase 1 in isolation
# ---------------------------------------------------------------------------


class TestResolve:
    """Phase 1 tests — account + MMR concurrent fetch."""

    async def test_returns_account_and_mmr(self, account_data, mmr_data):
        client = _make_client(account_data, mmr_data)
        orch = SyncOrchestrator(client, console=_quiet_console())
        got_account, got_mmr = await orch._resolve("na", "Yoursaviour01", "SK04")

        assert got_account.puuid == PUUID
        assert got_mmr.current_data.currenttierpatched == "Gold 1"

    async def test_wraps_api_error_as_sync_error(self, mmr_data):
        client = AsyncMock()
        client.get_account.side_effect = APIError("401 Unauthorized")
        client.get_mmr.return_value = mmr_data

        orch = SyncOrchestrator(client, console=_quiet_console())
        with pytest.raises(SyncError, match="Cannot resolve player Yoursaviour01#SK04"):
            await orch._resolve("na", "Yoursaviour01", "SK04")

    async def test_calls_both_endpoints_concurrently(self, account_data, mmr_data):
        """Both get_account and get_mmr are called once per _resolve invocation."""
        client = _make_client(account_data, mmr_data)
        orch = SyncOrchestrator(client, console=_quiet_console())
        await orch._resolve("na", "Yoursaviour01", "SK04")

        client.get_account.assert_called_once_with("Yoursaviour01", "SK04")
        client.get_mmr.assert_called_once_with("na", "Yoursaviour01", "SK04")


# ---------------------------------------------------------------------------
# SyncOrchestrator._finalise() — phase 4 in isolation
# ---------------------------------------------------------------------------


class TestFinalise:
    async def test_complete_sync_called_with_correct_counts(self):
        """_finalise calls complete_sync with the SyncResult counts."""
        session = _make_session()
        client = AsyncMock()

        result = SyncResult(puuid=PUUID, matches_fetched=5, matches_new=3)

        with (
            patch(_P_SCOPE, lambda: _scope(session)),
            patch(_P_COMPLETE_SYNC) as mock_complete,
        ):
            orch = SyncOrchestrator(client, console=_quiet_console())
            await orch._finalise(SYNC_LOG_ID, result)

        mock_complete.assert_called_once()
        call_kwargs = mock_complete.call_args.kwargs
        assert call_kwargs["matches_fetched"] == 5
        assert call_kwargs["matches_new"] == 3
        assert call_kwargs["error"] is None

    async def test_error_summary_from_per_match_errors(self):
        """When result.errors is non-empty, error_summary mentions the count."""
        session = _make_session()
        client = AsyncMock()

        result = SyncResult(puuid=PUUID, errors=["err1", "err2"])

        with (
            patch(_P_SCOPE, lambda: _scope(session)),
            patch(_P_COMPLETE_SYNC) as mock_complete,
        ):
            orch = SyncOrchestrator(client, console=_quiet_console())
            await orch._finalise(SYNC_LOG_ID, result)

        call_kwargs = mock_complete.call_args.kwargs
        assert "2 fetch error(s)" in call_kwargs["error"]

    async def test_fatal_error_passed_through_when_no_match_errors(self):
        """result.error (fatal) is forwarded when errors list is empty."""
        session = _make_session()
        client = AsyncMock()

        result = SyncResult(puuid=PUUID, error="fatal problem")

        with (
            patch(_P_SCOPE, lambda: _scope(session)),
            patch(_P_COMPLETE_SYNC) as mock_complete,
        ):
            orch = SyncOrchestrator(client, console=_quiet_console())
            await orch._finalise(SYNC_LOG_ID, result)

        call_kwargs = mock_complete.call_args.kwargs
        assert call_kwargs["error"] == "fatal problem"


# ---------------------------------------------------------------------------
# SyncResult dataclass
# ---------------------------------------------------------------------------


class TestSyncResult:
    def test_ok_when_no_error(self):
        assert SyncResult(puuid="x").ok is True

    def test_not_ok_when_error_set(self):
        assert SyncResult(puuid="x", error="boom").ok is False

    def test_ok_with_per_match_errors_only(self):
        """Per-match errors don't affect .ok — only a fatal error does."""
        assert SyncResult(puuid="x", errors=["e1"]).ok is True


# ---------------------------------------------------------------------------
# sync_player_matches — module-level entry point
# ---------------------------------------------------------------------------


class TestSyncPlayerMatches:
    async def test_raises_sync_error_when_name_not_configured(self):
        """Missing riot_name raises SyncError before touching the network."""
        settings = Settings(
            riot_name="",
            riot_tag="",
            riot_region="na",
            henrikdev_api_key="key",
            ollama_model="qwen3:8b",
            ollama_host="http://localhost:11434",
        )
        with pytest.raises(SyncError, match="riot_name / riot_tag not configured"):
            await sync_player_matches(settings)

    async def test_raises_sync_error_when_tag_not_configured(self):
        settings = Settings(
            riot_name="Yoursaviour01",
            riot_tag="",
            riot_region="na",
            henrikdev_api_key="key",
            ollama_model="qwen3:8b",
            ollama_host="http://localhost:11434",
        )
        with pytest.raises(SyncError, match="riot_name / riot_tag not configured"):
            await sync_player_matches(settings)

    async def test_delegates_to_orchestrator(self, account_data, mmr_data, match_details):
        """sync_player_matches wires up HenrikClient and calls SyncOrchestrator.run()."""
        settings = Settings(
            riot_name="Yoursaviour01",
            riot_tag="SK04",
            riot_region="na",
            henrikdev_api_key="test-key",
            ollama_model="qwen3:8b",
            ollama_host="http://localhost:11434",
        )

        # Patch SyncOrchestrator.run to avoid needing a real HTTP client
        mock_result = SyncResult(puuid=PUUID, matches_new=3)
        with (
            patch(
                "valocoach.data.sync.SyncOrchestrator.run",
                new_callable=AsyncMock,
                return_value=mock_result,
            ) as mock_run,
            # Also patch HenrikClient context manager so no real HTTP is attempted
            patch("valocoach.data.sync.HenrikClient") as mock_henrik_cls,
        ):
            mock_henrik_cls.return_value.__aenter__.return_value = AsyncMock()
            mock_henrik_cls.return_value.__aexit__.return_value = None

            result = await sync_player_matches(settings, limit=10, mode="competitive")

        assert result.matches_new == 3
        mock_run.assert_called_once_with(
            "na", "Yoursaviour01", "SK04", limit=10, full=False, mode="competitive"
        )

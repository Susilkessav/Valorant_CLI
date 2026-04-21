"""Match sync orchestrator.

Public API
----------
sync_player_matches(settings, *, limit, full, mode) -> SyncResult
    Thin entry point used by the CLI.  Creates a HenrikClient and delegates
    to SyncOrchestrator.run().

SyncOrchestrator
    Class-based coordinator with four named phases:

      Phase 1  _resolve     — fetch account + MMR concurrently, upsert Player row,
                               open SyncLog.
      Phase 2  _discover    — fetch stored-match list (v1), filter to IDs not yet
                               in the local DB.
      Phase 3  _fetch_all   — fetch v4 match details (sequential or concurrent via
                               asyncio.Semaphore), upsert each via mapper + repository.
      Phase 4  _finalise    — close SyncLog.  Runs in a try/finally so it is
                               guaranteed to execute even if an earlier phase raises.

Design notes
------------
- Phases are separate async methods → each is independently unit-testable with a
  mocked client or session factory.
- `concurrency=1` (default) keeps fetches sequential, which is safe for the free
  tier (30 req/min).  Set concurrency=3 on a paid key to parallelise detail fetches;
  HenrikClient._throttle() is bypassed for concurrent requests (same behaviour as
  fetch_player_snapshot), so a burst of N is your responsibility.
- Per-match APIErrors are collected in SyncResult.errors but do NOT abort the run —
  other matches continue.  Fatal errors (bad key, no network) raise SyncError so the
  CLI can display them and exit 1.
- Each DB write uses its own session_scope() so one bad match cannot roll back the
  whole sync.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field

from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn

from valocoach.core.config import Settings
from valocoach.core.exceptions import APIError, SyncError
from valocoach.data.api_client import HenrikClient
from valocoach.data.database import session_scope
from valocoach.data.models import AccountData, MMRData
from valocoach.data.orm_models import SyncLog
from valocoach.data.repository import (
    complete_sync,
    match_exists,
    start_sync,
    upsert_match_details,
    upsert_player,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class SyncResult:
    """Outcome of a sync run — returned to the CLI for display."""

    puuid: str
    matches_fetched: int = 0
    matches_new: int = 0
    matches_skipped: int = 0
    error: str | None = None  # fatal error that aborted the run
    errors: list[str] = field(default_factory=list)  # per-match non-fatal errors

    @property
    def ok(self) -> bool:
        """True when no fatal error occurred (per-match errors don't count)."""
        return self.error is None


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


class SyncOrchestrator:
    """Coordinates the match sync pipeline across four named phases.

    Construct with an open HenrikClient and call run().  The console and
    concurrency limit are injectable for testing.

    Example::

        async with HenrikClient(settings) as client:
            orch = SyncOrchestrator(client)
            result = await orch.run("na", "Yoursaviour01", "SK04", limit=20)
    """

    def __init__(
        self,
        client: HenrikClient,
        *,
        console: Console | None = None,
        concurrency: int = 1,
    ) -> None:
        """
        Args:
            client:      Open HenrikClient (must be used inside ``async with``).
            console:     Rich Console for progress output.  Defaults to a fresh
                         Console(); inject a null console in tests.
            concurrency: Max concurrent v4 detail fetches.  1 = sequential (free
                         tier safe).  Increase for paid API keys.
        """
        self._client = client
        self._con = console or Console()
        self._sem = asyncio.Semaphore(concurrency)

    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------

    async def run(
        self,
        region: str,
        name: str,
        tag: str,
        *,
        limit: int = 20,
        full: bool = False,
        mode: str = "competitive",
    ) -> SyncResult:
        """Run all four phases and return a SyncResult.

        Raises:
            SyncError: If phase 1 (resolve) or phase 2 (discover) fails
                       completely.  Phase 3 (fetch) errors are non-fatal and
                       collected in SyncResult.errors.
        """
        result = SyncResult(puuid="")
        sync_log_id: int | None = None

        try:
            # ── Phase 1: resolve player identity ──────────────────────────
            account, mmr = await self._resolve(region, name, tag)
            result.puuid = account.puuid
            sync_log_id = await self._open_log(account, mmr)

            # ── Phase 2: discover which match IDs are new ─────────────────
            new_ids, result.matches_fetched, result.matches_skipped = await self._discover(
                region, name, tag, limit=limit, full=full, mode=mode
            )

            # ── Phase 3: fetch details + upsert ───────────────────────────
            if new_ids:
                await self._fetch_all(region, new_ids, result)

        except SyncError:
            raise
        except APIError as exc:
            # Unexpected API error that escaped phase handling — promote to SyncError
            result.error = str(exc)
            raise SyncError(str(exc)) from exc

        finally:
            # ── Phase 4: close SyncLog — always runs ──────────────────────
            if sync_log_id is not None:
                await self._finalise(sync_log_id, result)

        return result

    # ------------------------------------------------------------------
    # Phase 1 — resolve
    # ------------------------------------------------------------------

    async def _resolve(self, region: str, name: str, tag: str) -> tuple[AccountData, MMRData]:
        """Fetch account + MMR concurrently.  Raises SyncError on failure."""
        self._con.print(f"[cyan]Looking up {name}#{tag} ({region})…[/cyan]")
        try:
            account, mmr = await asyncio.gather(
                self._client.get_account(name, tag),
                self._client.get_mmr(region, name, tag),
            )
        except APIError as exc:
            raise SyncError(f"Cannot resolve player {name}#{tag}: {exc}") from exc

        self._con.print(
            f"  [green]✓[/green]  {account.name}#{account.tag}  "
            f"[dim]{mmr.current_data.currenttierpatched}[/dim]"
        )
        return account, mmr

    async def _open_log(self, account: AccountData, mmr: MMRData) -> int:
        """Upsert the Player row and open a SyncLog.  Returns sync_log.id."""
        async with session_scope() as session:
            await upsert_player(session, account, mmr)
            sync_log = await start_sync(session, account.puuid)
        return sync_log.id  # id populated by flush() inside start_sync

    # ------------------------------------------------------------------
    # Phase 2 — discover
    # ------------------------------------------------------------------

    async def _discover(
        self,
        region: str,
        name: str,
        tag: str,
        *,
        limit: int,
        full: bool,
        mode: str,
    ) -> tuple[list[str], int, int]:
        """Fetch stored-match list and split into new vs already-stored IDs.

        Returns:
            (new_match_ids, total_fetched, skipped_count)
        """
        self._con.print(f"[cyan]Fetching stored match list (mode={mode!r}, limit={limit})…[/cyan]")
        try:
            stored = await self._client.get_stored_matches(region, name, tag, mode=mode, size=limit)
        except APIError as exc:
            raise SyncError(f"Cannot fetch match list: {exc}") from exc

        fetched = len(stored)
        self._con.print(f"  Found [bold]{fetched}[/bold] match(es) to inspect.")

        new_ids: list[str] = []
        skipped = 0

        async with session_scope() as session:
            for sm in stored:
                mid = sm.match_id
                if not mid:
                    continue
                if await match_exists(session, mid):
                    skipped += 1
                    if not full:
                        self._con.print(
                            f"  [dim]{mid[:8]}… already stored — "
                            "stopping early (--full to scan all).[/dim]"
                        )
                        break
                else:
                    new_ids.append(mid)

        self._con.print(
            f"  [green]{len(new_ids)} new[/green]  /  " f"[dim]{skipped} already stored[/dim]"
        )
        return new_ids, fetched, skipped

    # ------------------------------------------------------------------
    # Phase 3 — fetch + store
    # ------------------------------------------------------------------

    async def _fetch_all(
        self,
        region: str,
        new_ids: list[str],
        result: SyncResult,
    ) -> None:
        """Fetch v4 match details and upsert each one.

        Uses asyncio.gather with a Semaphore(concurrency) so the caller controls
        how many detail requests run simultaneously.  With the default concurrency=1
        the gather degrades to a sequential loop — each match waits for the previous
        one's semaphore release before starting.
        """
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            console=self._con,
            transient=False,
        ) as progress:
            task = progress.add_task("Syncing…", total=len(new_ids))

            async def _one(match_id: str) -> None:
                async with self._sem:
                    progress.update(task, description=f"Syncing {match_id[:8]}…")
                    try:
                        details = await self._client.get_match_details(region, match_id)
                        async with session_scope() as session:
                            stored = await upsert_match_details(session, details)
                        if stored is not None:
                            result.matches_new += 1
                        else:
                            result.matches_skipped += 1
                    except APIError as exc:
                        msg = f"Match {match_id[:8]}…: {exc}"
                        log.warning("sync skip — %s", msg)
                        result.errors.append(msg)
                    finally:
                        progress.advance(task)

            await asyncio.gather(*(_one(mid) for mid in new_ids))

    # ------------------------------------------------------------------
    # Phase 4 — finalise (always runs)
    # ------------------------------------------------------------------

    async def _finalise(self, sync_log_id: int, result: SyncResult) -> None:
        """Close the SyncLog row with outcome counts.  Always called via finally."""
        # None when everything was fine; a fetch-error summary wins over
        # any previously-set error because aggregate counts are more useful.
        error_summary: str | None = (
            f"{len(result.errors)} fetch error(s)" if result.errors else result.error
        )

        async with session_scope() as session:
            sync_log = await session.get(SyncLog, sync_log_id)
            complete_sync(
                session,
                sync_log,
                matches_fetched=result.matches_fetched,
                matches_new=result.matches_new,
                error=error_summary,
            )


# ---------------------------------------------------------------------------
# Module-level entry point  (CLI → here → SyncOrchestrator)
# ---------------------------------------------------------------------------


async def sync_player_matches(
    settings: Settings,
    *,
    limit: int = 20,
    full: bool = False,
    mode: str = "competitive",
) -> SyncResult:
    """Create a HenrikClient and run the SyncOrchestrator.

    This is the only function the CLI imports.  It keeps the CLI free of
    HenrikClient lifecycle management and makes the orchestrator independently
    testable (inject a mock client directly into SyncOrchestrator).

    Raises:
        SyncError:   If riot_name/riot_tag are not configured, or if a fatal
                     API error occurs during the run.
        ConfigError: If the API key is missing (raised by HenrikClient.__init__).
    """
    name = settings.riot_name
    tag = settings.riot_tag
    region = settings.riot_region

    if not name or not tag:
        raise SyncError("riot_name / riot_tag not configured — run `valocoach config init`")

    async with HenrikClient(settings) as client:
        orchestrator = SyncOrchestrator(client)
        return await orchestrator.run(region, name, tag, limit=limit, full=full, mode=mode)

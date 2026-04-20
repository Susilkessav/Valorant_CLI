"""Match sync pipeline.

sync_player_matches() is the entry point for the `valocoach sync` command:

  1. Fetch account + MMR concurrently and upsert the Player row.
  2. Fetch the stored-match list (v1 — lightweight summaries with match_ids).
  3. Filter match_ids that are not yet in the local DB (match_exists check).
  4. Fetch full v4 match details for each new match and upsert to the DB.
  5. Record the sync attempt in SyncLog.

Design notes:
  - HenrikClient._throttle() paces all HTTP calls at 2 s apart; the pipeline
    gets rate-limit safety automatically.
  - full=False (default, incremental) stops at the first already-stored match,
    assuming newest-first ordering from the API.  full=True always inspects up
    to `limit` matches — useful for a first-run or gap-fill.
  - Per-match APIErrors are logged and collected but do NOT abort the sync —
    other matches continue.  Fatal errors (bad key, DB init failure) raise
    SyncError so the CLI can display them cleanly.
  - Each DB write uses its own session_scope() so a single bad match cannot
    roll back the entire sync.
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
from valocoach.data.orm_models import SyncLog
from valocoach.data.repository import (
    complete_sync,
    match_exists,
    start_sync,
    upsert_match_details,
    upsert_player,
)

log = logging.getLogger(__name__)
console = Console()


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
    error: str | None = None
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.error is None


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


async def sync_player_matches(
    settings: Settings,
    *,
    limit: int = 20,
    full: bool = False,
    mode: str = "competitive",
) -> SyncResult:
    """Fetch and store competitive match history for the configured player.

    Args:
        settings: Project settings (riot_name, riot_tag, riot_region, api_key).
        limit:    Max stored-matches to inspect per run.
        full:     False = stop early on the first already-stored match (fast
                  incremental). True = inspect all `limit` matches regardless.
        mode:     Game-mode filter passed to get_stored_matches (default
                  "competitive" so unrated/deathmatch rows are never synced).

    Returns:
        SyncResult with counts and any per-match error messages.

    Raises:
        SyncError: If account/MMR lookup or initial DB write fails completely.
    """
    name = settings.riot_name
    tag = settings.riot_tag
    region = settings.riot_region

    if not name or not tag:
        raise SyncError(
            "riot_name / riot_tag not configured — run `valocoach config init`"
        )

    result = SyncResult(puuid="")

    async with HenrikClient(settings) as client:
        # ── 1. Account + MMR (concurrent) ─────────────────────────────────────
        console.print(f"[cyan]Looking up {name}#{tag} ({region})…[/cyan]")
        try:
            account, mmr = await asyncio.gather(
                client.get_account(name, tag),
                client.get_mmr(region, name, tag),
            )
        except APIError as exc:
            raise SyncError(f"Failed to fetch account/MMR: {exc}") from exc

        result.puuid = account.puuid
        console.print(
            f"  [green]✓[/green]  {account.name}#{account.tag}  "
            f"[dim]{mmr.current_data.currenttierpatched}[/dim]"
        )

        # ── 2. Upsert Player + open SyncLog ───────────────────────────────────
        async with session_scope() as session:
            await upsert_player(session, account, mmr)
            sync_log = await start_sync(session, account.puuid)
        sync_log_id: int = sync_log.id  # id set by flush inside start_sync

        # ── 3. Stored-match list (lightweight summaries) ───────────────────────
        console.print(
            f"[cyan]Fetching stored match list (mode={mode}, limit={limit})…[/cyan]"
        )
        try:
            stored = await client.get_stored_matches(
                region, name, tag, mode=mode, size=limit
            )
        except APIError as exc:
            error_msg = f"Failed to fetch stored matches: {exc}"
            async with session_scope() as session:
                log_obj = await session.get(SyncLog, sync_log_id)
                complete_sync(session, log_obj, matches_fetched=0, matches_new=0, error=error_msg)
            raise SyncError(error_msg) from exc

        result.matches_fetched = len(stored)
        console.print(f"  Found [bold]{len(stored)}[/bold] stored matches to inspect.")

        if not stored:
            async with session_scope() as session:
                log_obj = await session.get(SyncLog, sync_log_id)
                complete_sync(session, log_obj, matches_fetched=0, matches_new=0)
            return result

        # ── 4. Filter: which match_ids are new? ───────────────────────────────
        new_match_ids: list[str] = []
        async with session_scope() as session:
            for sm in stored:
                mid = sm.match_id
                if not mid:
                    continue
                already = await match_exists(session, mid)
                if already:
                    result.matches_skipped += 1
                    if not full:
                        console.print(
                            f"  [dim]Match {mid[:8]}… already stored — "
                            "stopping (use --full to override).[/dim]"
                        )
                        break
                else:
                    new_match_ids.append(mid)

        console.print(
            f"  [green]{len(new_match_ids)} new[/green]  /  "
            f"[dim]{result.matches_skipped} already stored[/dim]"
        )

        if not new_match_ids:
            async with session_scope() as session:
                log_obj = await session.get(SyncLog, sync_log_id)
                complete_sync(
                    session,
                    log_obj,
                    matches_fetched=len(stored),
                    matches_new=0,
                )
            return result

        # ── 5. Fetch + upsert each new match ──────────────────────────────────
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            console=console,
            transient=False,
        ) as progress:
            task = progress.add_task("Syncing…", total=len(new_match_ids))

            for match_id in new_match_ids:
                progress.update(task, description=f"Syncing {match_id[:8]}…")
                try:
                    details = await client.get_match_details(region, match_id)
                    async with session_scope() as session:
                        stored_match = await upsert_match_details(session, details)
                    if stored_match is not None:
                        result.matches_new += 1
                    else:
                        result.matches_skipped += 1
                except APIError as exc:
                    msg = f"Match {match_id[:8]}…: {exc}"
                    log.warning("sync skip — %s", msg)
                    result.errors.append(msg)
                finally:
                    progress.advance(task)

        # ── 6. Complete SyncLog ────────────────────────────────────────────────
        error_summary = (
            f"{len(result.errors)} fetch error(s)" if result.errors else None
        )
        async with session_scope() as session:
            log_obj = await session.get(SyncLog, sync_log_id)
            complete_sync(
                session,
                log_obj,
                matches_fetched=len(stored),
                matches_new=result.matches_new,
                error=error_summary,
            )

    return result

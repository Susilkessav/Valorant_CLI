"""CLI implementation for ``valocoach meta-refresh``.

Runs the full automated meta sync pipeline:
  1. Detect patch via HenrikDev API
  2. Scrape official patch notes
  3. Scrape ranked + pro pick/win-rate stats
  4. Optionally ingest YouTube transcripts
  5. LLM-regenerate the tier list
  6. Write updated meta.json
  7. Re-embed everything into ChromaDB

Flags:
  --force       Run even when no new patch is detected.
  --dry-run     Execute all steps but don't write meta.json or re-ingest.
  --watch       Run in continuous mode: check daily, full sync on new patch.
  --install-cron  Write a crontab entry that runs daily patch checks automatically.
"""

from __future__ import annotations

import asyncio
import subprocess
import sys
from pathlib import Path

import typer

from valocoach.cli import display


def run_meta_refresh(
    *,
    force: bool = False,
    dry_run: bool = False,
    watch: bool = False,
    install_cron: bool = False,
    youtube: list[str] | None = None,
) -> None:
    """Entry point called by the ``meta-refresh`` CLI command."""

    if install_cron:
        _install_cron()
        return

    if watch:
        _run_watch_loop(force=force, dry_run=dry_run, youtube=youtube)
    else:
        asyncio.run(_run_once(force=force, dry_run=dry_run, youtube=youtube))


# ---------------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------------

async def _run_once(
    *,
    force: bool,
    dry_run: bool,
    youtube: list[str] | None,
) -> None:
    from valocoach.core.config import load_settings
    from valocoach.data.database import ensure_db
    from valocoach.retrieval.meta_sync import run_meta_sync

    settings = load_settings()
    await ensure_db(settings.data_dir / "valocoach.db")

    step_labels = {
        "patch_check":    "Checking patch version …",
        "patch_notes":    "Scraping patch notes …",
        "stats_scrape":   "Scraping agent stats …",
        "youtube_ingest": "Ingesting YouTube transcripts …",
        "meta_generate":  "Regenerating tier list (LLM) …",
        "meta_write":     "Writing meta.json …",
        "re_ingest":      "Re-embedding knowledge base …",
    }

    current_status_msg = ["Initialising …"]

    def on_step(name: str, status: str) -> None:
        if status == "start":
            current_status_msg[0] = step_labels.get(name, name)

    if dry_run:
        display.warn("Dry-run mode: no files will be written.")

    with display.console.status(
        "[bold cyan]Running meta refresh …[/bold cyan]", spinner="dots"
    ) as status_widget:

        def _on_step(name: str, sts: str) -> None:
            on_step(name, sts)
            status_widget.update(
                f"[bold cyan]{current_status_msg[0]}[/bold cyan]"
            )

        result = await run_meta_sync(
            settings,
            force=force,
            dry_run=dry_run,
            youtube_videos=youtube or [],
            on_step=_on_step,
        )

    _render_result(result, dry_run=dry_run)


def _render_result(result: object, *, dry_run: bool) -> None:
    """Print a Rich-formatted summary of the sync result."""
    from valocoach.retrieval.meta_sync import SyncResult

    r: SyncResult = result  # type: ignore[assignment]

    display.console.print()

    if not r.is_new_patch and not r.meta_regenerated:
        display.info(
            f"No new patch detected ([bold]{r.patch_version}[/bold]). "
            "Meta is already up to date.  Use [bold]--force[/bold] to refresh anyway."
        )
        return

    # Header
    patch_label = (
        f"[bold green]{r.patch_version}[/bold green] (new patch)"
        if r.is_new_patch
        else f"[bold yellow]{r.patch_version}[/bold yellow] (forced)"
    )
    display.console.print(f"  Patch detected: {patch_label}")

    # Step checkmarks
    rows = [
        ("Patch notes",   r.patch_notes_scraped),
        ("Ranked stats",  r.ranked_stats_scraped),
        ("Pro/VCT stats", r.pro_stats_scraped),
        ("Tier list LLM", r.meta_regenerated),
        ("meta.json",     r.meta_written),
        ("Re-ingest KB",  r.meta_ingested),
    ]
    for label, ok in rows:
        icon = "[green]✓[/green]" if ok else "[dim]–[/dim]"
        display.console.print(f"  {icon}  {label}")

    if r.youtube_chunks_ingested:
        display.console.print(
            f"  [green]✓[/green]  YouTube ({r.youtube_chunks_ingested} chunks)"
        )

    display.console.print()

    if r.errors:
        for err in r.errors:
            display.warn(err)

    if r.meta_written:
        if dry_run:
            display.info("Dry run complete — meta.json was NOT modified.")
        else:
            display.success(
                f"meta.json updated for patch [bold]{r.patch_version}[/bold]. "
                "Coaching responses will use the new tier list immediately."
            )
    elif r.is_new_patch or r.meta_regenerated:
        display.warn("Sync completed with errors — meta.json was not updated.")


# ---------------------------------------------------------------------------
# Watch mode (continuous daily checks)
# ---------------------------------------------------------------------------

def _run_watch_loop(
    *,
    force: bool,
    dry_run: bool,
    youtube: list[str] | None,
) -> None:
    """Run indefinitely: check for a new patch every 24 hours.

    When a new patch is detected the full sync pipeline fires automatically.
    The loop can be interrupted with Ctrl-C.
    """
    import time

    check_interval_hours = 24
    check_interval_secs = check_interval_hours * 3_600

    display.info(
        f"Watch mode: checking for new patches every {check_interval_hours}h. "
        "Press [bold]Ctrl-C[/bold] to stop."
    )

    while True:
        display.console.print()
        display.console.rule("[dim]patch check[/dim]")
        try:
            asyncio.run(_run_once(force=force, dry_run=dry_run, youtube=youtube))
        except Exception as exc:
            display.warn(f"Sync run failed: {exc}")

        # After the first forced run, subsequent checks are not forced.
        force = False

        display.console.print(
            f"\n[dim]Next check in {check_interval_hours} hours …  "
            "(Ctrl-C to exit)[/dim]"
        )
        try:
            time.sleep(check_interval_secs)
        except KeyboardInterrupt:
            display.info("Watch mode stopped.")
            return


# ---------------------------------------------------------------------------
# Cron installer
# ---------------------------------------------------------------------------

_CRON_MARKER = "# valocoach-meta-refresh"
_CRON_SCHEDULE = "0 8 * * *"  # 08:00 every day


def _install_cron() -> None:
    """Add a daily crontab entry that runs ``valocoach meta-refresh``.

    The entry is idempotent — running ``--install-cron`` twice does not
    add a duplicate line.
    """
    python = sys.executable
    # Resolve the valocoach entry-point script next to the Python binary.
    valocoach_bin = Path(python).parent / "valocoach"
    if not valocoach_bin.exists():
        display.error(
            f"Could not find valocoach binary at {valocoach_bin}. "
            "Make sure the package is installed in the active venv."
        )
        raise typer.Exit(1)

    cron_line = (
        f"{_CRON_SCHEDULE}  {valocoach_bin} meta-refresh  "
        f"{_CRON_MARKER}"
    )

    # Read existing crontab (empty string if none).
    try:
        existing = subprocess.check_output(
            ["crontab", "-l"], stderr=subprocess.DEVNULL, text=True
        )
    except subprocess.CalledProcessError:
        existing = ""

    if _CRON_MARKER in existing:
        display.info("Crontab entry already exists — nothing changed.")
        display.console.print(
            f"  [dim]{next(line for line in existing.splitlines() if _CRON_MARKER in line)}[/dim]"
        )
        return

    new_crontab = existing.rstrip("\n") + "\n" + cron_line + "\n"

    proc = subprocess.run(
        ["crontab", "-"],
        input=new_crontab,
        text=True,
        capture_output=True,
    )
    if proc.returncode != 0:
        display.error(f"crontab write failed: {proc.stderr.strip()}")
        raise typer.Exit(1)

    display.success("Crontab entry installed:")
    display.console.print(f"  [dim]{cron_line}[/dim]")
    display.console.print(
        "\n[dim]This will run[/dim] valocoach meta-refresh [dim]daily at 08:00. "
        "Remove it with[/dim]  crontab -e"
    )

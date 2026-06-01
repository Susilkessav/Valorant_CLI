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
  --force         Run even when no new patch is detected.
  --dry-run       Execute all steps but don't write meta.json or re-ingest.
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
    install_cron: bool = False,
    youtube: list[str] | None = None,
) -> None:
    """Entry point called by the ``meta-refresh`` CLI command."""

    if install_cron:
        _install_cron()
        return

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
        "patch_check": "Checking patch version …",
        "patch_notes": "Scraping patch notes …",
        "stats_scrape": "Scraping agent stats …",
        "youtube_ingest": "Ingesting YouTube transcripts …",
        "meta_generate": "Regenerating tier list (LLM) …",
        "meta_write": "Writing meta.json …",
        "re_ingest": "Re-embedding knowledge base …",
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
            status_widget.update(f"[bold cyan]{current_status_msg[0]}[/bold cyan]")

        result = await run_meta_sync(
            settings,
            force=force,
            dry_run=dry_run,
            youtube_videos=youtube or [],
            on_step=_on_step,
        )

    ok = _render_result(result, dry_run=dry_run)
    if not ok:
        raise typer.Exit(1)


def _render_result(result: object, *, dry_run: bool) -> bool:
    """Print a Rich-formatted summary of the sync result.

    Returns ``True`` on success and ``False`` on a hard failure (e.g. the
    patch check itself failed), so the caller can set a nonzero exit code.
    """
    from valocoach.retrieval.meta_sync import SyncResult

    r: SyncResult = result  # type: ignore[assignment]

    display.console.print()

    # Patch check failed → patch_version is left as "unknown".  This is NOT the
    # same as "up to date" — reporting it as such hides a real failure (e.g. a
    # missing API key or a network error).  Surface it honestly and signal a
    # nonzero exit.
    patch_check_failed = r.patch_version == "unknown" or any(
        "patch check failed" in e.lower() for e in r.errors
    )
    if patch_check_failed:
        display.error("Could not determine the current patch — patch check failed.")
        for err in r.errors:
            display.warn(err)
        display.console.print(
            "[muted]Verify henrikdev_api_key in ~/.valocoach/config.toml and your "
            "connection, then retry.[/muted]"
        )
        return False

    if not r.is_new_patch and not r.meta_regenerated:
        display.info(
            f"No new patch detected ([bold]{r.patch_version}[/bold]). "
            "Meta is already up to date.  Use [bold]--force[/bold] to refresh anyway."
        )
        return True

    patch_label = (
        f"[stat.good]{r.patch_version}[/stat.good] (new patch)"
        if r.is_new_patch
        else f"[warning]{r.patch_version}[/warning] (forced)"
    )
    display.console.print(f"  Patch detected: {patch_label}")

    rows = [
        ("Patch notes", r.patch_notes_scraped),
        ("Ranked stats", r.ranked_stats_scraped),
        ("Pro/VCT stats", r.pro_stats_scraped),
        ("Tier list LLM", r.meta_regenerated),
        ("meta.json", r.meta_written),
        ("Re-ingest KB", r.meta_ingested),
    ]
    for label, step_ok in rows:
        icon = "[success]✔[/success]" if step_ok else "[muted]–[/muted]"
        display.console.print(f"  {icon}  {label}")

    if r.youtube_chunks_ingested:
        display.console.print(
            f"  [success]✔[/success]  YouTube ({r.youtube_chunks_ingested} chunks)"
        )

    display.console.print()

    if r.errors:
        for err in r.errors:
            if "LLM returned no valid JSON" in err or "empty response" in err.lower():
                display.warn(err)
                display.console.print(
                    "[muted]Tip: this usually means the model ran out of context or tokens.\n"
                    "  Try: [bold]ollama pull qwen3:14b[/bold]  (larger model handles longer prompts)\n"
                    "  Or:  set a smaller model that fits in your VRAM.[/muted]"
                )
            else:
                display.warn(err)

    if r.meta_written:
        if dry_run:
            display.info("Dry run complete — meta.json was NOT modified.")
        else:
            display.success(
                f"meta.json updated for patch [bold]{r.patch_version}[/bold]. "
                "Coaching responses will use the new tier list immediately."
            )
        return True

    # A dry run intentionally skips the meta.json write, so "not written" is
    # the expected, successful outcome there.
    if dry_run and not r.errors:
        display.info("Dry run complete — no files were modified.")
        return True

    if r.is_new_patch or r.meta_regenerated:
        display.warn("Sync completed with errors — meta.json was not updated.")
        return False

    return not r.errors


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

    cron_line = f"{_CRON_SCHEDULE}  {valocoach_bin} meta-refresh  {_CRON_MARKER}"

    # Read existing crontab (empty string if none).
    try:
        existing = subprocess.check_output(["crontab", "-l"], stderr=subprocess.DEVNULL, text=True)
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

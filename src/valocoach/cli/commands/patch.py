"""`valocoach patch` — display the current Valorant patch version.

Flow
----
Default (offline)
    Read the most recently recorded ``PatchVersion`` row from the local DB.
    No API call — works without internet and without a valid Riot ID.

With ``--check``
    Call the Henrik API to fetch the current game version, record it if new,
    and invalidate stale cache entries.  Requires ``henrikdev_api_key`` and
    a network connection.

Exit codes
----------
0   Patch info printed (or "no data" message shown).
1   ``--check`` failed due to a network / API error.
"""

from __future__ import annotations

import asyncio
import logging

from valocoach.cli import display

log = logging.getLogger(__name__)


def run_patch(*, check: bool = False) -> None:
    """Entry point for ``valocoach patch``.

    Args:
        check: When True, refresh the patch version from the Henrik API
               before displaying.
    """
    from valocoach.core.config import load_settings
    from valocoach.data.database import ensure_db
    from valocoach.retrieval.patch_tracker import get_current_patch

    settings = load_settings()

    # Ensure the schema exists so the first run (before any sync) doesn't crash.
    asyncio.run(ensure_db(settings.data_dir / "valocoach.db"))

    if check:
        _check_and_refresh(settings)

    # Display whatever is stored locally (may be None if never synced / checked).
    version = asyncio.run(get_current_patch())
    _render(version, checked=check)


def _check_and_refresh(settings) -> None:
    """Call Henrik API to detect a new patch; display success / error inline."""
    from valocoach.retrieval.patch_tracker import check_patch_update

    try:
        version, is_new = asyncio.run(check_patch_update(settings))
        if is_new:
            display.success(f"New patch detected: {version}")
        else:
            display.info(f"Patch unchanged: {version}")
    except Exception as exc:
        log.debug("Patch check failed: %s", exc)
        display.warn(f"Could not check for patch update: {exc}")


def _render(version: str | None, *, checked: bool) -> None:
    """Print a compact patch card to the terminal."""
    from rich.panel import Panel
    from rich.text import Text

    from valocoach.cli.display import console

    if version is None:
        if checked:
            # We already printed an error in _check_and_refresh.
            display.warn("No patch version recorded — check above for errors.")
        else:
            display.info(
                "No patch version recorded yet.  "
                "Run  valocoach sync  or  valocoach patch --check  first."
            )
        return

    body = Text()
    body.append("Game version  ", style="dim")
    body.append(version, style="bold cyan")

    console.print(
        Panel(
            body,
            title="[bold]Valorant Patch[/bold]",
            title_align="left",
            border_style="cyan",
            padding=(0, 2),
        )
    )

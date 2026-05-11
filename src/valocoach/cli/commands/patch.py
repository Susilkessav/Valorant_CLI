"""`valocoach patch` — display the current Valorant patch version."""

from __future__ import annotations

import asyncio
import logging

from valocoach.cli import display

log = logging.getLogger(__name__)


def run_patch(*, check: bool = False) -> None:
    from valocoach.core.config import load_settings
    from valocoach.data.database import ensure_db
    from valocoach.retrieval.patch_tracker import get_current_patch

    settings = load_settings()

    asyncio.run(ensure_db(settings.data_dir / "valocoach.db"))

    if check:
        _check_and_refresh(settings)

    version = asyncio.run(get_current_patch())
    _render(version, checked=check)


def _check_and_refresh(settings) -> None:
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
    from rich.panel import Panel
    from rich.text import Text

    if version is None:
        if checked:
            display.warn("No patch version recorded — check above for errors.")
        else:
            display.error_with_hint(
                "No patch version recorded yet.",
                "Run: valocoach sync  or  valocoach patch --check",
            )
        return

    body = Text()
    body.append("Game version  ", style="stat.label")
    body.append(version, style="heading")

    with display.command_frame("Patch Status"):
        display.console.print(
            Panel(
                body,
                title="[heading]Valorant Patch[/heading]",
                title_align="left",
                border_style="border",
                padding=(0, 2),
            )
        )

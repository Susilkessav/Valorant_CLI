from __future__ import annotations

import asyncio

from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.history import FileHistory
from rich.markdown import Markdown

from valocoach.cli.display import console
from valocoach.core.config import AppSettings
from valocoach.core.models import CoachRequest
from valocoach.llm.coach import CoachService


def run_interactive_session(settings: AppSettings) -> None:
    """Run a minimal interactive coaching session."""
    settings.ensure_directories()
    session = PromptSession(
        history=FileHistory(str(settings.paths.history_file)),
        auto_suggest=AutoSuggestFromHistory(),
    )
    coach = CoachService.from_settings(settings)

    console.print(Markdown("## ValoCoach Interactive\nType `/quit` to exit."))
    while True:
        try:
            message = session.prompt("valocoach> ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print()
            break

        if not message:
            continue
        if message in {"/quit", "/exit"}:
            break
        if message == "/help":
            console.print(Markdown("`/quit` exits. Any other input is sent to the coach."))
            continue

        request = CoachRequest(message=message)
        asyncio.run(coach.run_and_render(request))

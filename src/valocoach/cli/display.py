from __future__ import annotations

from collections.abc import Iterator

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.theme import Theme

THEME = Theme(
    {
        "info": "cyan",
        "warning": "yellow",
        "error": "bold red",
        "success": "bold green",
        "coach": "green",
    }
)

console = Console(theme=THEME)


def info(msg: str) -> None:
    console.print(f"[info]ℹ[/info]  {msg}")


def warn(msg: str) -> None:
    console.print(f"[warning]⚠[/warning]  {msg}")


def error(msg: str) -> None:
    console.print(f"[error]✗[/error]  {msg}")


def success(msg: str) -> None:
    console.print(f"[success]✓[/success]  {msg}")


def coach_panel(content: str, title: str = "🎯 Coach") -> Panel:
    """Build the coaching response panel. Used inside Live for streaming."""
    return Panel(Markdown(content), title=title, border_style="coach", padding=(1, 2))


def stream_to_panel(
    token_stream: Iterator[str],
    title: str = "🎯 Coach",
    refresh_per_second: int = 8,
) -> str:
    """Render a streaming token iterator into a live markdown panel.

    Returns the full accumulated text once streaming completes.
    """
    full = ""
    with Live(
        coach_panel("", title=title),
        console=console,
        refresh_per_second=refresh_per_second,
        transient=False,
    ) as live:
        for token in token_stream:
            full += token
            live.update(coach_panel(full, title=title))
    return full

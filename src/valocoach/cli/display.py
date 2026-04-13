from __future__ import annotations

from collections.abc import AsyncIterator

from rich.console import Console, Group, RenderableType
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text

console = Console()


def coach_panel(
    renderable: RenderableType, *, title: str = "Coach", border_style: str = "green"
) -> Panel:
    """Wrap a renderable in the standard coach output panel."""
    return Panel(renderable, title=title, border_style=border_style, padding=(1, 2))


def key_value_table(title: str, rows: list[tuple[str, str]]) -> Table:
    """Create a two-column table for small config and status payloads."""
    table = Table(title=title, box=None, show_header=False, pad_edge=False)
    table.add_column("Key", style="cyan", no_wrap=True)
    table.add_column("Value", style="white")
    for key, value in rows:
        table.add_row(key, value)
    return table


def placeholder_panel(command_name: str, detail: str) -> None:
    """Render a standard placeholder panel for unimplemented commands."""
    console.print(
        coach_panel(
            Text(detail),
            title=f"{command_name} pending",
            border_style="yellow",
        )
    )


def strip_hidden_thinking(markdown_text: str) -> str:
    """Remove `<think>...</think>` blocks, including incomplete trailing blocks."""
    visible_segments: list[str] = []
    cursor = 0
    while cursor < len(markdown_text):
        open_index = markdown_text.find("<think>", cursor)
        if open_index == -1:
            visible_segments.append(markdown_text[cursor:])
            break
        visible_segments.append(markdown_text[cursor:open_index])
        close_index = markdown_text.find("</think>", open_index)
        if close_index == -1:
            break
        cursor = close_index + len("</think>")
    return "".join(visible_segments).strip()


async def stream_markdown(
    token_stream: AsyncIterator[str],
    *,
    title: str = "Coach",
    border_style: str = "green",
    refresh_per_second: int = 8,
    show_thinking: bool = False,
) -> str:
    """Stream markdown into a Rich Live panel and return the final visible text."""
    full_text = ""
    first_token_received = False

    waiting = coach_panel(
        Group(Spinner("dots", style=border_style), Text("Waiting for Ollama...", style="dim")),
        title=title,
        border_style=border_style,
    )

    with Live(
        waiting, console=console, refresh_per_second=refresh_per_second, transient=False
    ) as live:
        async for token in token_stream:
            full_text += token
            visible_text = full_text if show_thinking else strip_hidden_thinking(full_text)
            first_token_received = True
            live.update(
                coach_panel(
                    Markdown(visible_text or " "),
                    title=title,
                    border_style=border_style,
                )
            )

        if not first_token_received:
            live.update(
                coach_panel(
                    Text("No tokens were returned by the model.", style="yellow"),
                    title=title,
                    border_style="yellow",
                )
            )

    return full_text if show_thinking else strip_hidden_thinking(full_text)

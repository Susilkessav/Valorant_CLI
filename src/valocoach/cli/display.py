from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.theme import Theme

from valocoach import __version__

# ---------------------------------------------------------------------------
# Valorant-branded theme
# ---------------------------------------------------------------------------

THEME = Theme(
    {
        # Core Valorant branding
        "val.red": "#FF4655",
        "val.red.dim": "#CC3844",
        "val.blue": "#5AECE5",
        # Semantic roles
        "info": "#5AECE5",
        "warning": "#FFC857",
        "error": "bold #FF4655",
        "success": "bold #1AE670",
        "coach": "#1AE670",
        # Stat quality
        "stat.good": "#1AE670",
        "stat.bad": "#FF4655",
        "stat.neutral": "dim #ECE8E1",
        "stat.label": "dim #8B978F",
        "stat.value": "bold #ECE8E1",
        # Tier list
        "tier.s": "bold #FF4655",
        "tier.a": "bold #1AE670",
        "tier.b": "#FFC857",
        "tier.c": "dim #8B978F",
        # Rank
        "rank.up": "bold #1AE670",
        "rank.down": "bold #FF4655",
        "rank.flat": "dim",
        # Structural
        "heading": "bold #ECE8E1",
        "subheading": "#8B978F",
        "border": "#FF4655",
        "border.dim": "#5A3A3E",
        "muted": "dim #8B978F",
    }
)

console = Console(theme=THEME)

# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------

_BANNER = r"""[val.red] ╦  ╦╔═╗╦  ╔═╗╔═╗╔═╗╔═╗╔═╗╦ ╦[/val.red]
[val.red] ╚╗╔╝╠═╣║  ║ ║║  ║ ║╠═╣║  ╠═╣[/val.red]
[val.red]  ╚╝ ╩ ╩╩═╝╚═╝╚═╝╚═╝╩ ╩╚═╝╩ ╩[/val.red]  [muted]v{version}[/muted]"""


def render_banner(con: Console | None = None) -> None:
    c = con or console
    c.print(_BANNER.format(version=__version__))


# ---------------------------------------------------------------------------
# Message helpers
# ---------------------------------------------------------------------------


def info(msg: str) -> None:
    console.print(f"[info]»[/info]  {msg}")


def warn(msg: str) -> None:
    console.print(f"[warning]![/warning]  {msg}")


def error(msg: str) -> None:
    console.print(f"[error]✖[/error]  {msg}")


def error_with_hint(msg: str, hint: str) -> None:
    console.print(f"[error]✖[/error]  {msg}")
    console.print(f"    [muted]» {hint}[/muted]")


def success(msg: str) -> None:
    console.print(f"[success]✔[/success]  {msg}")


# ---------------------------------------------------------------------------
# Section header
# ---------------------------------------------------------------------------


def render_section(con: Console | None = None, title: str = "") -> None:
    c = con or console
    c.print()
    c.print(Rule(f"[heading]{title}[/heading]", style="border.dim", characters="─"))


# ---------------------------------------------------------------------------
# Command frame — unified output wrapper
# ---------------------------------------------------------------------------


@contextmanager
def command_frame(title: str, subtitle: str | None = None, con: Console | None = None):
    """Wrap a command's entire output in a branded frame.

    Usage::

        with display.command_frame("Stats Dashboard", subtitle="last 30d"):
            render_overall(con, stats)
            ...
    """
    c = con or console
    c.print()
    c.print(Rule(f"[heading]{title}[/heading]", style="border", characters="━"))
    if subtitle:
        c.print(f"  [muted]{subtitle}[/muted]")
    c.print()
    try:
        yield c
    finally:
        c.print()
        c.print(Rule(style="border.dim", characters="─"))
        c.print()


# ---------------------------------------------------------------------------
# Coach panel + streaming
# ---------------------------------------------------------------------------


def coach_panel(
    content: str,
    title: str = "Coach",
    subtitle: str | None = None,
) -> Panel:
    """Build the coaching response panel. Used inside Live for streaming."""
    return Panel(
        Markdown(content),
        title=title,
        subtitle=f"[muted]{subtitle}[/muted]" if subtitle else None,
        border_style="border",
        padding=(1, 3),
    )


def stream_to_panel(
    token_stream: Iterator[str],
    title: str = "Coach",
    subtitle: str | None = None,
    refresh_per_second: int = 8,
) -> str:
    """Render a streaming token iterator into a live markdown panel.

    Returns the full accumulated text once streaming completes.
    """
    full = ""
    with Live(
        coach_panel("", title=title, subtitle=subtitle),
        console=console,
        refresh_per_second=refresh_per_second,
        transient=False,
    ) as live:
        for token in token_stream:
            full += token
            live.update(coach_panel(full, title=title, subtitle=subtitle))
    return full

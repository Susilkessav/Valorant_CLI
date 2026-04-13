from __future__ import annotations

import asyncio
from typing import Annotated

import typer
from rich.markdown import Markdown

from valocoach import __version__
from valocoach.cli.display import console, key_value_table, placeholder_panel
from valocoach.cli.interactive import run_interactive_session
from valocoach.core.config import (
    CONFIG_FILE_ENV_VAR,
    HOME_ENV_VAR,
    clear_settings_cache,
    config_file_path,
    ensure_config_file,
    get_settings,
    set_config_value,
)
from valocoach.core.exceptions import (
    CommandNotReadyError,
    ConfigurationError,
    ProviderError,
    ValoCoachError,
)
from valocoach.core.models import CoachRequest
from valocoach.llm.coach import CoachService

app = typer.Typer(
    help="CLI-based Valorant tactical coach.",
    no_args_is_help=True,
    add_completion=False,
)
config_app = typer.Typer(help="Manage ValoCoach configuration.")
app.add_typer(config_app, name="config")


def _render_error(error: Exception) -> None:
    """Render a user-facing error without a traceback."""
    console.print(f"[bold red]error:[/bold red] {error}")


def _run_placeholder(command_name: str) -> None:
    raise CommandNotReadyError(
        f"`{command_name}` is stubbed in week 1. The CLI surface exists, but the backing data "
        "pipeline lands in later milestones."
    )


@app.callback()
def main_callback(
    version: Annotated[
        bool, typer.Option("--version", help="Show the application version.")
    ] = False,
) -> None:
    """Top-level CLI callback."""
    if version:
        console.print(__version__)
        raise typer.Exit()


@app.command()
def coach(
    message: Annotated[str, typer.Argument(help="Natural-language match situation to coach.")],
    agent: Annotated[str | None, typer.Option("--agent", help="Optional agent context.")] = None,
    map_name: Annotated[str | None, typer.Option("--map", help="Optional map context.")] = None,
    side: Annotated[
        str | None,
        typer.Option("--side", help="Optional side context: attack or defense."),
    ] = None,
) -> None:
    """Stream a coaching response from Ollama."""
    if not message:
        raise typer.BadParameter("A match situation must be provided.", param_hint="MESSAGE")
    try:
        settings = get_settings()
        request = CoachRequest(message=message, agent=agent, map_name=map_name, side=side)
        asyncio.run(CoachService.from_settings(settings).run_and_render(request))
    except (ConfigurationError, ProviderError, ValoCoachError) as error:
        _render_error(error)
        raise typer.Exit(code=1) from error


@app.command()
def stats(
    agent: Annotated[str | None, typer.Option("--agent", help="Filter stats to an agent.")] = None,
    map_name: Annotated[str | None, typer.Option("--map", help="Filter stats to a map.")] = None,
    period: Annotated[
        str, typer.Option("--period", help="Lookback window: 7d, 30d, or 90d.")
    ] = "30d",
) -> None:
    """Show player performance stats."""
    try:
        _ = (agent, map_name, period)
        _run_placeholder("stats")
    except CommandNotReadyError as error:
        placeholder_panel("stats", str(error))


@app.command()
def sync(
    full: Annotated[
        bool, typer.Option("--full", help="Fetch a full sync rather than incremental.")
    ] = False,
    limit: Annotated[
        int, typer.Option("--limit", help="Limit the number of matches to fetch.")
    ] = 20,
) -> None:
    """Sync matches from the API."""
    try:
        _ = (full, limit)
        _run_placeholder("sync")
    except CommandNotReadyError as error:
        placeholder_panel("sync", str(error))


@app.command()
def profile() -> None:
    """Show the player profile summary."""
    try:
        _run_placeholder("profile")
    except CommandNotReadyError as error:
        placeholder_panel("profile", str(error))


@app.command()
def meta(
    agent: Annotated[
        str | None, typer.Option("--agent", help="Filter meta info to an agent.")
    ] = None,
    map_name: Annotated[
        str | None, typer.Option("--map", help="Filter meta info to a map.")
    ] = None,
) -> None:
    """Show the current meta snapshot."""
    try:
        _ = (agent, map_name)
        _run_placeholder("meta")
    except CommandNotReadyError as error:
        placeholder_panel("meta", str(error))


@app.command()
def interactive() -> None:
    """Open the interactive coaching REPL."""
    try:
        run_interactive_session(get_settings())
    except (ConfigurationError, ProviderError, ValoCoachError) as error:
        _render_error(error)
        raise typer.Exit(code=1) from error


@app.command()
def patch() -> None:
    """Show the current patch information."""
    try:
        _run_placeholder("patch")
    except CommandNotReadyError as error:
        placeholder_panel("patch", str(error))


@config_app.command("init")
def config_init(
    force: Annotated[
        bool, typer.Option("--force", help="Overwrite an existing config file.")
    ] = False,
) -> None:
    """Create the default config file."""
    try:
        path = ensure_config_file(force=force)
        clear_settings_cache()
        console.print(f"Created config at [bold]{path}[/bold]")
    except ConfigurationError as error:
        _render_error(error)
        raise typer.Exit(code=1) from error


@config_app.command("show")
def config_show() -> None:
    """Show the resolved configuration."""
    settings = get_settings()
    console.print(
        key_value_table(
            "ValoCoach configuration",
            [
                ("config_file", str(config_file_path())),
                ("paths.home", str(settings.paths.home)),
                ("paths.data_dir", str(settings.paths.data_dir)),
                ("paths.sessions_dir", str(settings.paths.sessions_dir)),
                ("ollama.host", settings.ollama.host),
                ("ollama.model", settings.ollama.model),
                ("coach.temperature", f"{settings.coach.temperature:.2f}"),
                ("ui.show_thinking", str(settings.ui.show_thinking)),
            ],
        )
    )


@config_app.command("path")
def config_path() -> None:
    """Show the active config file path."""
    console.print(str(config_file_path()))


@config_app.command("set")
def config_set(
    key: Annotated[str, typer.Argument(help="Dotted config key to update, e.g. ollama.model.")],
    value: Annotated[
        str, typer.Argument(help="Replacement value serialized as TOML-friendly text.")
    ],
) -> None:
    """Update a single config value."""
    try:
        path = set_config_value(key, value)
        clear_settings_cache()
        console.print(f"Updated [bold]{key}[/bold] in [bold]{path}[/bold]")
    except ConfigurationError as error:
        _render_error(error)
        raise typer.Exit(code=1) from error


@config_app.command("env")
def config_env() -> None:
    """Show the environment variables that override the config file."""
    console.print(
        Markdown(
            "\n".join(
                [
                    "## Environment overrides",
                    f"- `{CONFIG_FILE_ENV_VAR}` selects a custom config file path.",
                    f"- `{HOME_ENV_VAR}` relocates the ValoCoach home directory.",
                    "- Nested settings can be overridden with `VALOCOACH_...` variables.",
                ]
            )
        )
    )


def main() -> None:
    """Entrypoint for the console script."""
    try:
        app()
    except ValoCoachError as error:
        _render_error(error)
        raise typer.Exit(code=1) from error

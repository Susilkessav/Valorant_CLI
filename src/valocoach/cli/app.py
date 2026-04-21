from __future__ import annotations

import typer

from valocoach import __version__
from valocoach.cli import display

app = typer.Typer(
    name="valocoach",
    help="Valorant tactical coaching CLI",
    no_args_is_help=True,
    add_completion=True,
)


def _version_callback(value: bool) -> None:
    if value:
        display.console.print(f"valocoach v{__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        "-v",
        callback=_version_callback,
        is_eager=True,
        help="Show version and exit.",
    ),
) -> None:
    """Valorant tactical coaching CLI."""


@app.command()
def coach(
    situation: str = typer.Argument(..., help="Describe the match situation"),
    agent: str | None = typer.Option(None, "--agent", "-a"),
    map_: str | None = typer.Option(None, "--map", "-m", help="Map name"),
    side: str | None = typer.Option(None, "--side", "-s", help="attack or defense"),
    with_stats: bool = typer.Option(
        True,
        "--with-stats/--no-stats",
        help="Include your recent performance stats in the prompt (default on).",
    ),
) -> None:
    """Get tactical coaching for a match situation."""
    from valocoach.cli.commands.coach import run_coach

    run_coach(
        situation=situation,
        agent=agent,
        map_=map_,
        side=side,
        with_stats=with_stats,
    )


@app.command()
def stats(
    agent: str | None = typer.Option(None, "--agent", "-a", help="Filter to a single agent."),
    map_: str | None = typer.Option(None, "--map", "-m", help="Filter to a single map."),
    period: str = typer.Option(
        "30d",
        "--period",
        "-p",
        help="Time window: 'Nd' for last N days (e.g. 7d, 30d) or 'all'.",
    ),
) -> None:
    """Show your performance stats (overall + per-agent + per-map)."""
    from valocoach.cli.commands.stats import run_stats

    run_stats(agent=agent, map_=map_, period=period)


@app.command()
def sync(
    full: bool = typer.Option(
        False,
        "--full",
        help="Inspect all --limit matches instead of stopping at the first already-stored one.",
    ),
    limit: int = typer.Option(
        20,
        "--limit",
        help="Maximum stored-matches to inspect per run.",
    ),
    mode: str = typer.Option(
        "competitive",
        "--mode",
        help="Game-mode filter (competitive / unrated / …).",
    ),
) -> None:
    """Sync match history from the HenrikDev API into the local database."""
    import asyncio

    from valocoach.core.config import load_settings
    from valocoach.core.exceptions import ConfigError, SyncError
    from valocoach.data.database import ensure_db
    from valocoach.data.sync import sync_player_matches

    settings = load_settings()

    async def _run() -> None:
        await ensure_db(settings.data_dir / "valocoach.db")

        try:
            result = await sync_player_matches(settings, limit=limit, full=full, mode=mode)
        except (SyncError, ConfigError) as exc:
            display.error(str(exc))
            raise typer.Exit(1) from exc

        if result.matches_new == 0 and not result.errors:
            display.info("Already up to date — no new matches to store.")
        else:
            display.success(
                f"Sync complete: [bold]{result.matches_new}[/bold] new match(es) stored  "
                f"({result.matches_skipped} already in DB)"
            )

        for err in result.errors:
            display.warn(err)

        if not result.ok:
            display.warn(f"Sync finished with errors: {result.error}")

    asyncio.run(_run())


@app.command()
def profile(
    name: str | None = typer.Option(
        None,
        "--name",
        "-n",
        help="Riot username. Defaults to your configured riot_name.",
    ),
    tag: str | None = typer.Option(
        None,
        "--tag",
        "-t",
        help="Riot tag. Must be given together with --name.",
    ),
    limit: int = typer.Option(
        20,
        "--limit",
        "-l",
        help="Number of recent matches to summarise.",
    ),
) -> None:
    """Show player identity + compact recent-performance card."""
    from valocoach.cli.commands.profile import run_profile

    run_profile(name=name, tag=tag, limit=limit)


@app.command()
def meta(
    agent: str | None = typer.Option(None, "--agent", "-a"),
    map_: str | None = typer.Option(None, "--map", "-m"),
) -> None:
    """Show current Valorant meta. (stub — week 4)"""
    display.warn("meta: not implemented yet (week 4)")


@app.command()
def patch() -> None:
    """Show current patch info. (stub — week 4)"""
    display.warn("patch: not implemented yet (week 4)")


@app.command()
def interactive() -> None:
    """Start interactive coaching REPL. (stub — week 5)"""
    display.warn("interactive: not implemented yet (week 5)")


config_app = typer.Typer(help="Manage configuration")
app.add_typer(config_app, name="config")


@config_app.command("init")
def config_init() -> None:
    """Create a default config file at ~/.valocoach/config.toml."""
    from valocoach.core.config import write_default_config

    path = write_default_config()
    display.success(f"Config written to {path}")


@config_app.command("show")
def config_show() -> None:
    """Display current effective settings."""
    from valocoach.core.config import load_settings

    s = load_settings()
    display.console.print(s.model_dump())


if __name__ == "__main__":
    app()

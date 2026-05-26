from __future__ import annotations

import logging

import typer

from valocoach import __version__
from valocoach.cli import display

log = logging.getLogger(__name__)

app = typer.Typer(
    name="valocoach",
    help="Valorant tactical coaching CLI",
    no_args_is_help=False,
    add_completion=True,
)


def _version_callback(value: bool) -> None:
    if value:
        display.render_banner()
        raise typer.Exit()


def _require_llm() -> None:
    """Check Ollama is reachable and exit with an actionable error if not."""
    from valocoach.core.config import load_settings
    from valocoach.core.preflight import check_ollama

    result = check_ollama(load_settings())
    if not result.ok:
        display.error_with_hint(
            result.message,
            result.hint or "Start Ollama with: ollama serve",
        )
        raise typer.Exit(1)


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
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
    import sys

    if ctx.invoked_subcommand is None:
        if sys.stdin.isatty():
            from valocoach.cli.hub import show_hub
            show_hub()
        else:
            # Non-interactive context (pipe/CI) — emit plain help text
            display.render_banner()
            display.console.print()
            display.console.print("[muted]Usage: valocoach <command> [args][/muted]")
            display.console.print("[muted]Run valocoach --help for all commands.[/muted]")


@app.command(rich_help_panel="Coaching")
def coach(
    situation: str | None = typer.Argument(
        None,
        help="Describe the match situation.  Omit to open the interactive REPL.",
    ),
    agent: str | None = typer.Option(None, "--agent", "-a"),
    map_: str | None = typer.Option(None, "--map", "-m", help="Map name"),
    side: str | None = typer.Option(None, "--side", "-s", help="attack or defense"),
    with_stats: bool = typer.Option(
        True,
        "--with-stats/--no-stats",
        help="Include your recent performance stats in the prompt (default on).",
    ),
    no_elicit: bool = typer.Option(
        False,
        "--no-elicit",
        help="Skip the context questions and go straight to coaching.",
    ),
) -> None:
    """Get tactical coaching.

    With a situation argument:  one-shot advice for a specific scenario.
    Without arguments (on a terminal):  opens the interactive coaching REPL.
    """
    import sys

    _require_llm()

    if situation is None:
        # No situation provided — launch the REPL on TTY, show help on pipes.
        if sys.stdin.isatty():
            from valocoach.cli.commands.interactive import run_interactive
            run_interactive()
        else:
            display.console.print(
                "[muted]Usage: valocoach coach \"<situation>\"[/muted]\n"
                "[muted]Example: valocoach coach \"1v2 post-plant B site Haven attack\"[/muted]"
            )
        return

    from valocoach.cli.commands.coach import run_coach

    run_coach(
        situation=situation,
        agent=agent,
        map_=map_,
        side=side,
        with_stats=with_stats,
        no_elicit=no_elicit,
    )


@app.command(rich_help_panel="Performance")
def stats(
    agent: str | None = typer.Option(None, "--agent", "-a", help="Filter to a single agent."),
    map_: str | None = typer.Option(None, "--map", "-m", help="Filter to a single map."),
    period: str = typer.Option(
        "90d",
        "--period",
        "-p",
        help="Time window: 'Nd' for last N days (e.g. 7d, 30d) or 'all'.",
    ),
    result: str | None = typer.Option(
        None,
        "--result",
        "-r",
        help="Filter by match outcome: 'win' or 'loss'. Omit for both.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Emit raw JSON instead of the Rich-rendered tables. Useful for scripting.",
    ),
) -> None:
    """Show your performance stats (overall + win/loss split + per-agent + per-map)."""
    from valocoach.cli.commands.stats import run_stats

    run_stats(agent=agent, map_=map_, period=period, result=result, json_output=json_output)


@app.command(rich_help_panel="Data")
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

    async def _run():
        await ensure_db(settings.data_dir / "valocoach.db")

        # Housekeeping: every sync also sweeps expired retrieval cache rows
        # (SQLite meta_cache + ChromaDB live docs).  Without this the cache
        # grows unbounded — purge_expired() was previously only called by
        # tests.  Failures are logged at debug, never block the sync.
        try:
            from valocoach.retrieval.cache import purge_expired

            purged = await purge_expired(settings.data_dir)
            if purged:
                log.debug("Purged %d expired cache entries before sync", purged)
        except Exception:
            log.debug("cache purge during sync failed", exc_info=True)

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

        return result

    with display.command_frame("Match Sync"):
        display.info("Syncing match history…")
        result = asyncio.run(_run())

    if result is not None and result.matches_new > 0:
        try:
            from valocoach.coach.session_manager import get_mmr_trend

            history = get_mmr_trend(settings, result.puuid, limit=2)
            if len(history) >= 2:
                elo_delta = history[0].elo - history[1].elo
                delta_str = f"+{elo_delta}" if elo_delta >= 0 else str(elo_delta)
                arrow = "↑" if elo_delta > 0 else ("↓" if elo_delta < 0 else "→")
                display.info(
                    f"Rank snapshot: {history[0].tier_patched} ({history[0].rr} RR)  "
                    f"[muted]{arrow} {delta_str} elo since last sync[/muted]"
                )
        except Exception:
            log.debug("MMR snapshot rendering failed", exc_info=True)


@app.command(rich_help_panel="Performance")
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
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Emit raw profile JSON instead of the Rich-rendered panel.",
    ),
) -> None:
    """Show player identity + compact recent-performance card."""
    from valocoach.cli.commands.profile import run_profile

    run_profile(name=name, tag=tag, limit=limit, json_output=json_output)


@app.command(rich_help_panel="Game Info")
def meta(
    agent: str | None = typer.Option(
        None, "--agent", "-a", help="Agent name for ability and meta info."
    ),
    map_: str | None = typer.Option(
        None, "--map", "-m", help="Map name for callouts and meta info."
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Emit the raw meta.json contents instead of the Rich-rendered panel.",
    ),
) -> None:
    """Show current meta: tier list, agent abilities, or map callouts."""
    from valocoach.cli.commands.meta import run_meta

    run_meta(agent=agent, map_=map_, json_output=json_output)


@app.command(rich_help_panel="Data")
def ingest(
    url: str | None = typer.Option(
        None, "--url", "-u", help="Scrape and ingest a URL (patch notes, blog post, etc.)."
    ),
    youtube: list[str] = typer.Option(  # noqa: B008
        [],
        "--youtube",
        "-y",
        help="YouTube URL or video ID to ingest. Repeatable: --youtube url1 --youtube url2.",
    ),
    youtube_list: str | None = typer.Option(
        None,
        "--youtube-list",
        help="Path to a text file with one YouTube URL per line (batch ingest).",
    ),
    corpus: bool = typer.Option(
        False,
        "--corpus",
        "-c",
        help="Embed markdown files from corpus/ (built by scripts/build_corpus.py).",
    ),
    seed: bool = typer.Option(False, "--seed", help="Re-embed the built-in JSON knowledge base."),
    clear: bool = typer.Option(False, "--clear", help="Wipe the vector store before ingesting."),
    show_stats: bool = typer.Option(
        False, "--stats", help="Show what's currently in the vector store."
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Re-ingest a YouTube video even if it is already stored (bypasses dedup check).",
    ),
    preview: bool = typer.Option(
        False,
        "--preview",
        help="Analyse a YouTube video and show what would be ingested, without writing anything.",
    ),
    add_lineup: bool = typer.Option(
        False,
        "--add-lineup",
        help="Interactively add a hand-curated lineup entry to the database.",
    ),
) -> None:
    """Populate the vector store (run once, then augmented by --url / --youtube / --corpus).

    \b
    Batch YouTube ingest:
      valocoach ingest --youtube url1 --youtube url2
      valocoach ingest --youtube-list urls.txt

    \b
    Hand-curate a lineup:
      valocoach ingest --add-lineup
    """
    from valocoach.cli.commands.ingest import run_ingest

    run_ingest(
        url=url,
        youtube=youtube or [],
        youtube_list=youtube_list,
        corpus=corpus,
        seed=seed,
        clear=clear,
        show_stats=show_stats,
        force=force,
        preview=preview,
        add_lineup=add_lineup,
    )


@app.command(rich_help_panel="Game Info")
def patch(
    check: bool = typer.Option(
        False,
        "--check",
        help="Refresh patch version from the Henrik API before displaying.",
    ),
) -> None:
    """Show the current Valorant patch / game version."""
    from valocoach.cli.commands.patch import run_patch

    run_patch(check=check)


@app.command("agents-refresh", rich_help_panel="Game Info")
def agents_refresh(
    auto_stub_meta: bool = typer.Option(
        False,
        "--auto-stub-meta",
        help="Append C-tier placeholders to meta.json for agents that exist in "
        "agents.json but are missing from the tier list.",
    ),
    extract_kits: bool = typer.Option(
        False,
        "--extract-kits",
        help="Deterministically parse new agents' kit data from Liquipedia's "
        "wikitext templates and write to agents.json. No LLM is used.",
    ),
) -> None:
    """Sync the agent knowledge base with Riot's current roster.

    Compares ``agents.json`` against the Liquipedia agents portal. New
    agents can be auto-imported by deterministically parsing Liquipedia's
    ``{{Infobox agent}}`` and ``{{AbilityCard}}`` wikitext templates — no
    LLM involved, so no hallucination risk. ``meta.json`` tier-list gaps
    can be auto-stubbed as clearly-labelled C-tier placeholders.

    \b
    Examples:
      valocoach agents-refresh                                  # discovery only
      valocoach agents-refresh --extract-kits                   # auto-import new agents
      valocoach agents-refresh --auto-stub-meta                 # patch tier-list gaps
      valocoach agents-refresh --extract-kits --auto-stub-meta  # both in one pass

    If Liquipedia's templates are malformed (or a brand-new agent isn't
    in their category index yet), the command falls back to the printed
    JSON skeleton + wiki URL for manual entry.
    """
    from valocoach.cli.commands.agents_refresh import run_agents_refresh

    run_agents_refresh(auto_stub_meta=auto_stub_meta, extract_kits=extract_kits)


@app.command("meta-refresh", rich_help_panel="Game Info")
def meta_refresh(
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Run the full sync even when no new patch is detected.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Simulate all steps but do not write meta.json or re-ingest.",
    ),
    install_cron: bool = typer.Option(
        False,
        "--install-cron",
        help="Add a daily crontab entry that runs meta-refresh automatically.",
    ),
    youtube: list[str] = typer.Option(  # noqa: B008
        [],
        "--youtube",
        "-y",
        help="YouTube video ID or URL to ingest as supplemental meta context. Repeatable.",
    ),
) -> None:
    """Auto-update meta.json when a new patch drops.

    Detects the current patch via the HenrikDev API, scrapes official patch
    notes + Diamond+/pro pick-rate stats, regenerates the tier list with the
    LLM, writes meta.json, and re-embeds everything into the vector store.

    \b
    Examples:
      valocoach meta-refresh               # run once (only fires on new patch)
      valocoach meta-refresh --force       # force update regardless of patch
      valocoach meta-refresh --dry-run     # preview without writing anything
      valocoach meta-refresh --install-cron  # register daily OS-level cron job
    """
    from valocoach.cli.commands.meta_refresh import run_meta_refresh

    run_meta_refresh(
        force=force,
        dry_run=dry_run,
        install_cron=install_cron,
        youtube=youtube or None,
    )


@app.command("lineup", rich_help_panel="Coaching")
def lineup(
    agent: str | None = typer.Argument(None, help="Agent name (e.g. Sova, Viper, KAY/O)."),
    map_: str | None = typer.Option(None, "--map", "-m", help="Map name (e.g. Ascent, Bind)."),
    site: str | None = typer.Option(None, "--site", "-s", help="Site letter: A, B, or C."),
    query: str | None = typer.Option(
        None, "--query", "-q", help="Override the search query (natural language)."
    ),
    n_results: int = typer.Option(5, "--top", "-n", help="Number of lineups to show (default 5)."),
) -> None:
    """Find ability lineups for an agent, map, and site.

    Searches the lineup database (seeded from YouTube guides + built-in entries).
    Run  valocoach ingest --seed  to load the bundled lineups.

    Examples:

      valocoach lineup Sova --map Ascent --site A

      valocoach lineup Viper --map Bind

      valocoach lineup --query "Sova post-plant shock bolt Haven C"
    """
    from valocoach.cli.commands.lineup import run_lineup

    run_lineup(agent=agent, map_name=map_, site=site, query=query, n_results=n_results)


@app.command("post-game", rich_help_panel="Coaching")
def post_game(
    match_id: str | None = typer.Argument(
        None,
        help="Match ID to analyse (8-char prefix or full ID).  Defaults to the most recent match.",
    ),
    no_notes: bool = typer.Option(
        False,
        "--no-notes",
        help="Skip auto-creating coaching notes for critical findings.",
    ),
    no_repl: bool = typer.Option(
        False,
        "--no-repl",
        help="Skip the interactive REPL handoff offer at the end.",
    ),
) -> None:
    """Debrief your last match: run post-game analyzers then get LLM coaching.

    Loads the most recent competitive match from the local DB (or a specific
    match by ID), runs deterministic analyzers across all rounds, surfaces the
    top 3 findings, auto-saves critical ones as coaching notes, and asks the
    LLM to write a structured post-game debrief.

    \b
    Examples:
      valocoach post-game                    # debrief your latest match
      valocoach post-game abc12345           # debrief a specific match
      valocoach post-game --no-notes         # skip auto-note creation
      valocoach post-game --no-repl          # skip interactive handoff
    """
    from valocoach.cli.commands.post_game import run_post_game

    run_post_game(match_id=match_id, no_notes=no_notes, no_repl=no_repl)


notes_app = typer.Typer(help="Manage coaching notes (list, add, resolve).")
app.add_typer(notes_app, name="notes", rich_help_panel="Coaching")


@notes_app.callback(invoke_without_command=True)
def notes_default(ctx: typer.Context) -> None:
    """Show open coaching notes (default when no sub-command is given)."""
    if ctx.invoked_subcommand is None:
        from valocoach.cli.commands.notes import run_notes_list

        run_notes_list()


@notes_app.command("list")
def notes_list() -> None:
    """List all open (unresolved) coaching notes."""
    from valocoach.cli.commands.notes import run_notes_list

    run_notes_list()


@notes_app.command("add")
def notes_add(
    text: str = typer.Argument(..., help="Note text, e.g. 'work on crossfire at A long'."),
    priority: int = typer.Option(
        2,
        "--priority",
        "-p",
        help="Note priority: 1 (high), 2 (medium, default), 3 (low).",
    ),
) -> None:
    """Add a coaching note.  Category is auto-inferred from the note text."""
    from valocoach.cli.commands.notes import run_notes_add

    run_notes_add(text, priority=priority)


@notes_app.command("resolve")
def notes_resolve(
    note_id: int = typer.Argument(..., help="ID of the note to resolve."),
) -> None:
    """Mark a coaching note as resolved."""
    from valocoach.cli.commands.notes import run_notes_resolve

    run_notes_resolve(note_id)


sessions_app = typer.Typer(help="Manage coaching sessions (list, close).")
app.add_typer(sessions_app, name="sessions", rich_help_panel="Coaching")


@sessions_app.callback(invoke_without_command=True)
def sessions_default(ctx: typer.Context) -> None:
    """Show recent coaching sessions (default when no sub-command is given)."""
    if ctx.invoked_subcommand is None:
        from valocoach.cli.commands.sessions import run_sessions_list

        run_sessions_list()


@sessions_app.command("list")
def sessions_list(
    limit: int = typer.Option(
        20,
        "--limit",
        "-l",
        help="Number of recent sessions to show (newest first).",
    ),
) -> None:
    """List recent coaching sessions."""
    from valocoach.cli.commands.sessions import run_sessions_list

    run_sessions_list(limit=limit)


@sessions_app.command("close")
def sessions_close(
    session_id: int = typer.Argument(..., help="ID of the session to close."),
) -> None:
    """Close (end) an open coaching session."""
    from valocoach.cli.commands.sessions import run_sessions_close

    run_sessions_close(session_id)


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
    """Display current effective settings (API keys are redacted)."""
    from rich.table import Table

    from valocoach.core.config import load_settings

    s = load_settings()
    data = s.model_dump()
    if data.get("henrikdev_api_key"):
        data["henrikdev_api_key"] = "***redacted***"

    table = Table(show_header=True, header_style="bold", box=None, pad_edge=False)
    table.add_column("Setting", style="heading")
    table.add_column("Value")
    for key in sorted(data.keys()):
        value = data[key]
        # Render Path / PosixPath etc. as plain strings so the user
        # doesn't see ``PosixPath(...)`` leaking out of the CLI.
        if value is None:
            shown = "[muted]<unset>[/muted]"
        elif isinstance(value, bool):
            shown = "true" if value else "false"
        else:
            shown = str(value)
        table.add_row(key, shown)

    with display.command_frame("Configuration"):
        display.console.print(table)


if __name__ == "__main__":
    app()

"""`valocoach post-game` — post-match debrief with LLM coaching.

Loads the player's most recent competitive match (or a specific match by ID),
runs the deterministic post-game analyzers, and asks the LLM to build a
coaching narrative from the top findings.

Phase B features implemented here
----------------------------------
B2/B3 — Finding dataclass + 9 analyzers live in ``valocoach.stats.post_game``.
B4    — This command: loads the match, runs analyzers, calls the LLM.
B5    — Auto-creates coaching notes for every **critical** finding.
B6    — After the LLM debrief, offers to launch the interactive REPL with
         the match's agent/map/score pre-loaded into ``SessionMatchContext``.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import typer

from valocoach.cli import display
from valocoach.coach.match_context import SessionMatchContext
from valocoach.stats.post_game import (
    Finding,
    analyze_mmr_trend,
    format_findings_block,
    run_analyzers,
    select_top_findings,
)


# ---------------------------------------------------------------------------
# DB loading helper
# ---------------------------------------------------------------------------


def _db_path(settings) -> Path:
    return Path(settings.data_dir) / "valocoach.db"


def _load_match_sync(settings, puuid: str, match_id: str | None):
    """Load a fully-hydrated Match tree (players + rounds + kills + round_players).

    When *match_id* is ``None``, loads the player's most recent competitive match.
    Returns ``None`` when nothing is found or on any DB error.

    The returned ORM object is safe to use after the session closes because:
      - All relationships are loaded via ``selectinload`` (no lazy triggers).
      - The engine is configured with ``expire_on_commit=False``.
    """

    async def _run():
        from valocoach.data.database import ensure_db, session_scope
        from valocoach.data.repository import (
            get_player_by_name,
            get_post_game_match,
            get_recent_matches,
        )

        await ensure_db(_db_path(settings))
        async with session_scope() as db:
            if match_id:
                return await get_post_game_match(db, match_id)
            player = await get_player_by_name(db, settings.riot_name, settings.riot_tag)
            if player is None:
                return None
            rows = await get_recent_matches(db, player.puuid, limit=1)
            if not rows:
                return None
            return await get_post_game_match(db, rows[0].match_id)

    try:
        return asyncio.run(_run())
    except Exception as exc:
        display.warn(f"DB error loading match: {exc}")
        return None


def _load_mmr_sync(settings, puuid: str, limit: int = 10) -> list:
    """Load up to *limit* MMRHistory rows for *puuid*, most-recent first.

    Returns an empty list on any DB error or when no rows exist.
    """

    async def _run():
        from valocoach.data.database import ensure_db, session_scope
        from valocoach.data.repository import get_mmr_history

        await ensure_db(_db_path(settings))
        async with session_scope() as db:
            return await get_mmr_history(db, puuid, limit=limit)

    try:
        return asyncio.run(_run())
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Findings panel rendering
# ---------------------------------------------------------------------------

_SEV_COLOUR = {
    "critical": "[val.red]",
    "warning":  "[warning]",
    "neutral":  "[muted]",
    "positive": "[success]",
}
_SEV_ICON = {
    "critical": "🔴",
    "warning":  "🟡",
    "neutral":  "⚪",
    "positive": "🟢",
}


def _render_findings_panel(findings: list[Finding], match) -> None:
    """Print a styled Rich panel summarising the top findings."""
    won = _match_won(match)
    result_str = "[success]WIN[/success]" if won else "[val.red]LOSS[/val.red]"
    # Map → strip extra whitespace; fallback to match_id prefix
    map_name = (match.map_name or "Unknown").strip()
    score = f"{match.red_score}-{match.blue_score}"

    with display.command_frame("Post-Game Analysis"):
        display.console.print(
            f"[heading]{map_name}[/heading]  {result_str}  "
            f"[muted]{score}[/muted]  [muted]{match.match_id[:8]}…[/muted]"
        )
        display.console.print()

        if not findings:
            display.console.print("[muted]No significant patterns detected in this match.[/muted]")
            return

        display.console.print(f"[heading]Top {len(findings)} finding(s):[/heading]")
        display.console.print()
        for i, f in enumerate(findings, 1):
            colour = _SEV_COLOUR.get(f.severity, "[muted]")
            end_colour = colour.replace("[", "[/")
            icon = _SEV_ICON.get(f.severity, "•")
            display.console.print(
                f"  {icon}  {colour}{f.severity.upper()}[/{end_colour.strip('[/')}]  "
                f"[heading]{f.headline}[/heading]"
            )
            display.console.print(f"     [muted]{f.category} · {f.root_cause_tag}[/muted]")
            display.console.print()


def _match_won(match) -> bool | None:
    """Return True/False/None for the match result (None = unknown)."""
    # winning_team is "Red" or "Blue"; we need to check if it matches any player's team
    wt = match.winning_team
    if not wt or not match.players:
        return None
    return bool(match.players)  # fallback — will be overridden by player check


def _match_won_for_player(match, puuid: str) -> bool | None:
    for mp in match.players:
        if mp.puuid == puuid:
            return mp.won
    return None


# ---------------------------------------------------------------------------
# B5 — Auto-note criticals
# ---------------------------------------------------------------------------

_FINDING_CATEGORY_MAP: dict[str, str] = {
    "duels": "aim",
    "positioning": "positioning",
    "economy": "economy",
    "utility": "agent_usage",
    "clutch": "tactical",
}


def _auto_note_criticals(
    settings,
    puuid: str,
    findings: list[Finding],
    match_id: str,
) -> int:
    """Create coaching notes for critical findings.  Returns the count saved."""
    from valocoach.coach.session_manager import add_coaching_note, get_or_open_coaching_session

    criticals = [f for f in findings if f.severity == "critical"]
    if not criticals:
        return 0

    session_id = get_or_open_coaching_session(settings, puuid)
    if session_id is None:
        display.warn("Couldn't open coaching session — critical findings not saved as notes.")
        return 0

    saved = 0
    for f in criticals:
        cat = _FINDING_CATEGORY_MAP.get(f.category, "general")
        # Compose a concise note body from headline + first sentence of detail
        first_sentence = f.detail.split(".")[0].strip() + "."
        note_body = f"{f.headline}. {first_sentence}"
        note_id = add_coaching_note(
            settings,
            session_id,
            puuid,
            note_body,
            category=cat,
            priority=1,  # critical → high priority
        )
        if note_id is not None:
            saved += 1
            display.success(
                f"Note #{note_id} saved  [muted](critical · {cat})[/muted]"
            )

    return saved


# ---------------------------------------------------------------------------
# B6 — Build SessionMatchContext from match data
# ---------------------------------------------------------------------------


def _build_match_context(match, puuid: str) -> SessionMatchContext:
    """Populate a ``SessionMatchContext`` from the loaded match row."""
    ctx = SessionMatchContext()
    ctx.map = match.map_name.strip() if match.map_name else None

    # Player's agent + result
    for mp in match.players:
        if mp.puuid == puuid:
            ctx.agent = mp.agent_name
            ctx.result = "won" if mp.won else "lost"
            # Score — own/opp
            if match.winning_team:
                own_score = match.red_score if mp.team == "Red" else match.blue_score
                opp_score = match.blue_score if mp.team == "Red" else match.red_score
                ctx.score = (own_score, opp_score)
            break

    # Enemy agents — the other team's agents
    for mp in match.players:
        if mp.puuid != puuid:
            my_team = ctx.agent and next(
                (p.team for p in match.players if p.puuid == puuid), None
            )
            if my_team and mp.team != my_team:
                ctx.add_enemy(mp.agent_name)

    return ctx


# ---------------------------------------------------------------------------
# B6 — REPL handoff offer
# ---------------------------------------------------------------------------


def _offer_repl_handoff(match_ctx: SessionMatchContext) -> None:
    """Ask the player if they want to continue in the interactive REPL."""
    if not sys.stdin.isatty():
        return  # non-interactive context (pipe, CI) — skip

    display.console.print()
    display.console.print(
        "[muted]Continue coaching this match in interactive mode?[/muted]"
    )
    try:
        answer = input("Launch REPL? [y/N] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        display.console.print()
        return

    if answer in ("y", "yes"):
        from valocoach.cli.commands.interactive import run_interactive

        run_interactive(initial_match_context=match_ctx)


# ---------------------------------------------------------------------------
# G6 — Lineup suggestion for low-utility agents
# ---------------------------------------------------------------------------

# Agents where lineup/ability placement matters enough to surface suggestions
_UTIL_HEAVY_AGENTS = frozenset({
    "Sova", "Viper", "KAY/O", "Fade", "Brimstone", "Skye", "Breach",
    "Astra", "Harbor", "Gekko", "Tejo",
})


def _suggest_lineups_for_low_util(
    findings: list[Finding],
    agent: str | None,
    map_name: str | None,
    settings,
) -> str | None:
    """Return a lineup suggestion block if low_utility fired for a util-heavy agent.

    Searches the lineup database and returns the top 2 results formatted as a
    coaching context block.  Returns None if no suggestions are available.
    """
    if not agent or not map_name:
        return None

    agent_clean = agent.strip()
    if agent_clean not in _UTIL_HEAVY_AGENTS:
        return None

    has_low_util = any(f.root_cause_tag == "low_utility" for f in findings)
    if not has_low_util:
        return None

    try:
        from valocoach.retrieval.lineups import format_lineup_results, search_lineups

        hits = search_lineups(
            settings.data_dir,
            query=f"{agent_clean} ability lineup {map_name}",
            agent=agent_clean,
            map_name=map_name.strip(),
            n_results=2,
        )
        if not hits:
            return None

        formatted = format_lineup_results(hits)
        return (
            f"LINEUP SUGGESTIONS for {agent_clean} on {map_name.strip()} "
            "(player used fewer abilities than expected):\n"
            + formatted
        )
    except Exception as exc:
        import logging
        logging.getLogger(__name__).debug("G6: lineup suggestion failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_post_game(
    match_id: str | None = None,
    *,
    no_notes: bool = False,
    no_repl: bool = False,
) -> None:
    """Run the post-game analysis pipeline for *match_id* (or most recent match).

    Args:
        match_id:  Specific match to analyse.  When ``None``, uses the most
                   recent competitive match in the local DB.
        no_notes:  Skip auto-creating coaching notes for critical findings.
        no_repl:   Skip the interactive REPL handoff offer at the end.
    """
    from valocoach.core.config import load_settings
    from valocoach.core.preflight import check_ollama

    settings = load_settings()

    # ── Ollama preflight ──────────────────────────────────────────────────
    ollama_result = check_ollama(settings)
    if not ollama_result.ok:
        display.error_with_hint(
            ollama_result.message,
            ollama_result.hint or "Start Ollama with: ollama serve",
        )
        raise typer.Exit(1)

    # ── Player identity ───────────────────────────────────────────────────
    from valocoach.coach.session_manager import get_player_puuid

    puuid = get_player_puuid(settings)
    if not puuid:
        display.error_with_hint(
            "No player profile found.",
            "Run:  valocoach sync",
        )
        raise typer.Exit(1)

    # ── Load match ────────────────────────────────────────────────────────
    match_label = match_id[:8] + "…" if match_id else "most recent match"
    display.info(f"Loading {match_label}…")
    match = _load_match_sync(settings, puuid, match_id)

    if match is None:
        if match_id:
            display.error_with_hint(
                f"Match {match_id!r} not found in local DB.",
                "Run  valocoach sync  then retry.",
            )
        else:
            display.error_with_hint(
                "No competitive matches found in local DB.",
                "Run  valocoach sync  to fetch your match history.",
            )
        raise typer.Exit(1)

    # ── Run analyzers ─────────────────────────────────────────────────────
    findings = run_analyzers(match, puuid)

    # E4: prepend MMR trend findings (loaded separately, not from match data)
    mmr_rows = _load_mmr_sync(settings, puuid, limit=10)
    mmr_findings = analyze_mmr_trend(mmr_rows)
    findings = mmr_findings + findings

    top = select_top_findings(findings, n=3)

    # ── Render findings panel ─────────────────────────────────────────────
    _render_findings_panel(top, match)

    # ── B5: Auto-note critical findings ───────────────────────────────────
    if not no_notes:
        n_saved = _auto_note_criticals(settings, puuid, top, match.match_id)
        if n_saved:
            display.console.print(
                f"[muted]{n_saved} critical finding(s) saved as coaching note(s).[/muted]"
            )

    if not top:
        display.info("No notable patterns found — good match!")
        return

    # ── LLM debrief ───────────────────────────────────────────────────────
    from valocoach.cli.commands.coach import run_coach

    # Get player's agent name for grounded context lookup
    player_mp = next((p for p in match.players if p.puuid == puuid), None)
    player_agent = player_mp.agent_name if player_mp else None

    findings_block = format_findings_block(top)

    # G6 — If a utility-heavy agent had low casts, surface relevant lineup suggestions.
    lineup_block = _suggest_lineups_for_low_util(top, player_agent, match.map_name, settings)

    situation = (
        f"{findings_block}\n\n{lineup_block}\n\nDebrief this match."
        if lineup_block
        else f"{findings_block}\n\nDebrief this match."
    )

    run_coach(
        situation=situation,
        agent=player_agent,
        map_=match.map_name.strip() if match.map_name else None,
        with_stats=False,   # analyzers already have the stats; avoid double-loading
        no_elicit=True,
        force_intent="post_game",
    )

    # ── B6: REPL handoff ──────────────────────────────────────────────────
    if not no_repl:
        match_ctx = _build_match_context(match, puuid)
        _offer_repl_handoff(match_ctx)

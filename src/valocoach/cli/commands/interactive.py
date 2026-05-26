"""Interactive coaching REPL.

Wraps ``run_coach`` in a persistent prompt_toolkit session so the player
can have a multi-turn conversation without re-typing the command each time.

Match context (/agent, /map, /side, /score, etc.) is persisted across turns
via ``SessionMatchContext`` so the player only needs to set each field once
per session.  The context is injected into every ``run_coach`` call as a
structured header, ensuring the LLM always has the current match state.
"""

from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger(__name__)

from valocoach.cli import display
from valocoach.coach.match_context import SessionMatchContext
from valocoach.coach.session_manager import (
    REPLCoachState,
    add_coaching_note,
    close_coaching_session,
    get_player_puuid,
    list_open_notes,
    open_coaching_session,
    resolve_coaching_note,
)
from valocoach.core.memory import ConversationMemory
from valocoach.core.session_store import (
    latest_session,
    list_sessions,
    load_session,
    save_session,
    session_summary,
)

_SLASH_HELP: dict[str, str] = {
    # ── Match context ──────────────────────────────────────────────────
    "/agent":   "Set your agent:  /agent Jett",
    "/map":     "Set the map:     /map Ascent",
    "/side":    "Set your side:   /side attack  or  /side defense",
    "/score":   "Set the score:   /score 9-11",
    "/won":     "Mark last match as won.",
    "/lost":    "Mark last match as lost.",
    "/eco":     "Set economy:     /eco eco  |  /eco half  |  /eco full",
    "/enemy":   "Add an enemy agent:  /enemy Cypher",
    "/half":    "Toggle side at half-time (attack ↔ defense).",
    "/context": "Show current match context.",
    "/reset":   "Clear all match context for this session.",
    # ── Session management ─────────────────────────────────────────────
    "/help":    "Show this help message.",
    "/clear":   "Clear conversation memory — start a fresh session.",
    "/memory":  "Show turn count and token usage in the current window.",
    "/save":    "Save the current session to disk immediately.",
    "/sessions":"List previously saved sessions.",
    "/stats":   "Display your recent stats card.",
    "/note":    "Add a coaching note:  /note <text>",
    "/notes":   "List your open (unresolved) coaching notes.",
    "/resolve": "Resolve a coaching note by id:  /resolve <id>",
    "/quit":    "Exit the REPL (also: Ctrl-D, Ctrl-C).",
}

_CONTEXT_CMDS = frozenset({
    "/agent", "/map", "/side", "/score", "/won", "/lost",
    "/eco", "/enemy", "/half", "/context", "/reset",
})

_WELCOME_PARTS = [
    "",
    "[val.red]━━━ Interactive Coaching Mode ━━━[/val.red]",
    "",
    "[heading]Ask anything about your gameplay.[/heading]",
    '[muted]Set context first: /agent Jett  /map Ascent  /side attack[/muted]',
    '[muted]Then ask: "how do I hold A site better?"[/muted]',
    "",
    "[muted]Commands: /help  ·  Ctrl-D to exit[/muted]",
]


def _print_help() -> None:
    display.console.print()
    display.console.print("[heading]Match context commands:[/heading]")
    for cmd in _CONTEXT_CMDS:
        if cmd in _SLASH_HELP:
            display.console.print(f"  [info]{cmd:<10}[/info] {_SLASH_HELP[cmd]}")
    display.console.print()
    display.console.print("[heading]Session commands:[/heading]")
    for cmd, desc in _SLASH_HELP.items():
        if cmd not in _CONTEXT_CMDS:
            display.console.print(f"  [info]{cmd:<10}[/info] {desc}")
    display.console.print()


def _handle_context_slash(
    raw_input: str,
    cmd: str,
    match_ctx: SessionMatchContext,
) -> None:
    """Handle match-context slash commands that write to ``match_ctx``."""
    from valocoach.coach.elicitation import _match_agent, _match_map, _match_score, _SIDE_MAP, _ECON_MAP

    arg = raw_input.strip()[len(cmd):].strip()

    if cmd == "/agent":
        if not arg:
            display.warn("Usage: /agent <name>   e.g. /agent Jett")
            return
        resolved = _match_agent(arg)
        if resolved:
            match_ctx.agent = resolved
            display.success(f"Agent set: {resolved}")
        else:
            display.warn(f"Agent '{arg}' not recognised.  Check spelling or try the full name.")

    elif cmd == "/map":
        if not arg:
            display.warn("Usage: /map <name>   e.g. /map Ascent")
            return
        resolved = _match_map(arg)
        if resolved:
            match_ctx.map = resolved
            display.success(f"Map set: {resolved}")
        else:
            display.warn(f"Map '{arg}' not recognised.")

    elif cmd == "/side":
        if not arg:
            display.warn("Usage: /side attack  or  /side defense")
            return
        normalized = _SIDE_MAP.get(arg.lower())
        if normalized:
            match_ctx.side = normalized
            display.success(f"Side set: {normalized}")
        else:
            display.warn(f"Side '{arg}' not recognised.  Use 'attack' or 'defense'.")

    elif cmd == "/score":
        if not arg:
            display.warn("Usage: /score 9-11")
            return
        parsed_score = _match_score(arg)
        if parsed_score:
            match_ctx.score = parsed_score
            display.success(f"Score set: {parsed_score[0]}-{parsed_score[1]}")
        else:
            display.warn(f"Score '{arg}' not recognised.  Use format: 9-11")

    elif cmd == "/won":
        match_ctx.result = "won"
        display.success("Result set: won")

    elif cmd == "/lost":
        match_ctx.result = "lost"
        display.success("Result set: lost")

    elif cmd == "/eco":
        if not arg:
            display.warn("Usage: /eco eco  |  /eco half  |  /eco full")
            return
        normalized = _ECON_MAP.get(arg.lower())
        if normalized:
            match_ctx.econ = normalized
            display.success(f"Economy set: {normalized.replace('_', ' ')}")
        else:
            display.warn(f"Economy '{arg}' not recognised.  Use: eco / half / full")

    elif cmd == "/enemy":
        if not arg:
            display.warn("Usage: /enemy <agent>   e.g. /enemy Cypher")
            return
        resolved = _match_agent(arg)
        if resolved:
            added = match_ctx.add_enemy(resolved)
            if added:
                display.success(f"Enemy agent added: {resolved}")
            else:
                display.info(f"{resolved} is already in enemy list.")
        else:
            display.warn(f"Agent '{arg}' not recognised.")

    elif cmd == "/half":
        old_side = match_ctx.side
        match_ctx.flip_side()
        if match_ctx.side != old_side and match_ctx.side:
            display.success(f"Side flipped: {old_side or '?'} → {match_ctx.side}  (half-time)")
        elif match_ctx.side is None:
            display.warn("No side set yet — use /side attack or /side defense first.")
        else:
            display.info(f"Side unchanged: {match_ctx.side}")

    elif cmd == "/context":
        display.console.print()
        display.console.print("[heading]Current match context:[/heading]")
        display.console.print(f"  {match_ctx.summary_line()}")
        if not match_ctx.is_empty:
            ctx_block = match_ctx.to_context_block()
            if ctx_block:
                display.console.print()
                for line in ctx_block.splitlines():
                    display.console.print(f"  [muted]{line}[/muted]")
        display.console.print()

    elif cmd == "/reset":
        match_ctx.reset()
        display.success("Match context cleared.")


def _handle_slash(
    raw_input: str,
    memory: ConversationMemory,
    state: REPLCoachState | None = None,
    *,
    match_ctx: SessionMatchContext | None = None,
) -> None:
    cmd = raw_input.strip().lower().split()[0]

    # Context commands are handled by a dedicated function
    if cmd in _CONTEXT_CMDS:
        if match_ctx is None:
            match_ctx = SessionMatchContext()
        _handle_context_slash(raw_input, cmd, match_ctx)
        return

    if cmd == "/help":
        _print_help()

    elif cmd == "/clear":
        memory.clear()
        display.success("Conversation memory cleared.")

    elif cmd == "/memory":
        turns = len(memory)
        tokens = memory.token_count
        display.info(f"Memory: {turns} turn(s) · {tokens} tokens")

    elif cmd == "/save":
        path = save_session(memory.messages)
        if path:
            display.success(f"Session saved to {path.name}")
        else:
            display.warn("Nothing to save — start a coaching conversation first.")

    elif cmd == "/sessions":
        sessions = list_sessions()
        if not sessions:
            display.info("No saved sessions found.")
        else:
            display.console.print()
            display.console.print("[heading]Saved sessions:[/heading]")
            for i, p in enumerate(sessions[:10], 1):
                display.console.print(
                    f"  [info]{i:>2}.[/info] {p.name}  [muted]{session_summary(p)}[/muted]"
                )
            display.console.print()

    elif cmd == "/stats":
        try:
            from valocoach.cli.commands.stats import run_stats

            run_stats()
        except Exception as exc:
            display.warn(f"Couldn't load stats: {exc}")

    elif cmd == "/note":
        body = raw_input.strip()[len("/note"):].strip()
        if not body:
            display.warn("Usage: /note <text>   e.g. /note Work on crossfire at A long")
            return
        if state is None or not state.active:
            display.warn(
                "No active coaching session — run `valocoach sync` first so "
                "ValoCoach can identify your player profile."
            )
            return
        note_id = add_coaching_note(state.settings, state.coaching_session_id, state.puuid, body)
        if note_id is not None:
            display.success(f"Note #{note_id} saved.")
        else:
            display.warn("Couldn't save note — check logs.")

    elif cmd == "/notes":
        if state is None or state.puuid is None:
            display.warn("No player profile found — run `valocoach sync` first.")
            return
        notes = list_open_notes(state.settings, state.puuid)
        if not notes:
            display.info("No open coaching notes.")
            return
        display.console.print(f"\n[heading]Open coaching notes ({len(notes)}):[/heading]")
        for n in notes:
            pri_icon = {1: "[val.red]●[/val.red]", 2: "[warning]●[/warning]", 3: "[muted]●[/muted]"}.get(
                n.priority, "●"
            )
            display.console.print(
                f"  [muted]{n.id:>4}.[/muted] {pri_icon} [{n.category}] {n.body}"
            )
        display.console.print()

    elif cmd == "/resolve":
        id_str = raw_input.strip()[len("/resolve"):].strip()
        if not id_str:
            display.warn("Usage: /resolve <note-id>   e.g. /resolve 12")
            return
        if not id_str.isdigit():
            display.warn(f"Note id must be a number; got: {id_str!r}")
            return
        if state is None or state.puuid is None:
            display.warn("No player profile found — run `valocoach sync` first.")
            return
        ok = resolve_coaching_note(state.settings, int(id_str))
        if ok:
            display.success(f"Note #{id_str} marked as resolved.")
        else:
            display.warn(f"Note #{id_str} not found or already resolved.")

    elif cmd == "/quit":
        raise SystemExit(0)

    else:
        display.warn(f"Unknown command: {cmd!r}  (try /help)")


def _build_completer():
    try:
        from prompt_toolkit.completion import WordCompleter

        from valocoach.retrieval import list_agent_names, list_map_names

        words = list(list_agent_names()) + list(list_map_names()) + list(_SLASH_HELP.keys())
        return WordCompleter(words, ignore_case=True, sentence=True)
    except Exception:
        # prompt_toolkit / agent list optional — falling back to no completer
        # is the right behavior; just leave a debug breadcrumb.
        log.debug("autocompleter unavailable", exc_info=True)
        return None


_OLLAMA_RECONNECT_KEYWORDS = ("connection", "refused", "timeout", "unreachable", "connect")


def run_interactive(
    initial_match_context: SessionMatchContext | None = None,
) -> None:
    """Start the interactive coaching REPL.

    Args:
        initial_match_context: Pre-populated :class:`SessionMatchContext` to
            carry into the REPL.  When provided the REPL skips the blank
            context and the welcome hint updates to show what is already set.
            Typical caller: ``post-game`` command (B6 handoff).
    """
    from valocoach.core.config import load_settings
    from valocoach.core.preflight import check_ollama

    settings = load_settings()
    ollama_result = check_ollama(settings)
    if not ollama_result.ok:
        display.error_with_hint(
            ollama_result.message,
            ollama_result.hint or "Start Ollama with: ollama serve",
        )
        return

    try:
        from prompt_toolkit import PromptSession
        from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
        from prompt_toolkit.history import FileHistory
    except ImportError:
        display.error_with_hint(
            "prompt_toolkit is required for interactive mode.",
            "Install it with: pip install prompt-toolkit",
        )
        return

    memory = ConversationMemory(max_turns=20, max_tokens=3_000)
    match_ctx = initial_match_context or SessionMatchContext()

    coach_state = REPLCoachState()
    coach_state.settings = settings
    puuid = get_player_puuid(settings)
    if puuid:
        coach_state.puuid = puuid
        # Reap any orphan ``open`` rows from earlier crashes before opening
        # a new one — keeps ``sessions list`` honest after a kill -9 / OOM.
        try:
            from valocoach.coach.session_manager import reap_stale_sessions

            reaped = reap_stale_sessions(settings, puuid)
            if reaped:
                log.debug("Reaped %d stale open coaching session(s)", reaped)
        except Exception:
            log.debug("Stale-session reaper failed (non-fatal)", exc_info=True)

        session_id = open_coaching_session(settings, puuid)
        if session_id is not None:
            coach_state.coaching_session_id = session_id
            display.info(f"Coaching session #{session_id} started.  Use /note to save takeaways.")

    last = latest_session()
    if last:
        summary = session_summary(last)
        display.console.print(f"\n[muted]Previous session found:[/muted] [info]{summary}[/info]")
        try:
            answer = input("Resume? [y/N] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            answer = "n"
        if answer in ("y", "yes"):
            turns = load_session(last)
            if turns:
                for t in turns:
                    memory.add(t["role"], t["content"])
                display.success(f"Resumed {len(turns)} turn(s) from {last.name}")
            else:
                display.warn("Could not load session — starting fresh.")
        display.console.print()

    history_path = Path.home() / ".valocoach" / "history"
    history_path.parent.mkdir(parents=True, exist_ok=True)

    session: PromptSession = PromptSession(
        history=FileHistory(str(history_path)),
        auto_suggest=AutoSuggestFromHistory(),
        completer=_build_completer(),
        complete_while_typing=False,
    )

    welcome = "\n".join(_WELCOME_PARTS)
    display.console.print(welcome)
    if initial_match_context is not None and not initial_match_context.is_empty:
        display.console.print(
            f"[muted]Match context pre-loaded: {initial_match_context.summary_line()}[/muted]"
        )
        display.console.print("[muted]Use /context to review · /reset to clear.[/muted]")
    display.console.print()

    from valocoach.cli.commands.coach import run_coach

    try:
        while True:
            try:
                raw = session.prompt("vc > ")
            except KeyboardInterrupt:
                display.console.print("\n[muted]Interrupted — type /quit or Ctrl-D to exit.[/muted]")
                continue
            except EOFError:
                display.console.print("\n[muted]Bye.[/muted]")
                break

            user_input = raw.strip()
            if not user_input:
                continue

            if user_input.startswith("/"):
                try:
                    _handle_slash(user_input, memory, coach_state, match_ctx=match_ctx)
                except SystemExit:
                    display.console.print("[muted]Bye.[/muted]")
                    break
                continue

            prior_history = memory.messages if not memory.is_empty else None

            try:
                response = run_coach(
                    situation=user_input,
                    conversation_history=prior_history,
                    match_context=match_ctx,
                )
            except Exception as exc:
                err_lower = str(exc).lower()
                if any(kw in err_lower for kw in _OLLAMA_RECONNECT_KEYWORDS):
                    display.error_with_hint(
                        "LLM call failed — Ollama may have stopped.",
                        "Check with: ollama list  |  Restart with: ollama serve",
                    )
                else:
                    display.error(f"Coaching failed: {exc}")
                continue

            if response:
                memory.add("user", user_input)
                memory.add("assistant", response)
    finally:
        if coach_state.coaching_session_id is not None:
            close_coaching_session(settings, coach_state.coaching_session_id)

        saved = save_session(memory.messages)
        if saved:
            display.console.print(f"[muted]Session saved → {saved.name}[/muted]")

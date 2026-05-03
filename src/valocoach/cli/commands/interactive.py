"""Interactive coaching REPL.

Wraps ``run_coach`` in a persistent prompt_toolkit session so the player
can have a multi-turn conversation without re-typing the command each time.
The session maintains sliding-window conversation memory across turns so
the LLM sees previous exchanges and can build on earlier advice.

Slash commands available inside the REPL::

    /help     — print this command list
    /clear    — wipe conversation memory (start fresh)
    /memory   — show how many turns are in memory and their token count
    /stats    — show your recent stats card (same output as ``valocoach stats``)
    /quit     — exit the REPL (Ctrl-D or Ctrl-C also exit cleanly)
"""

from __future__ import annotations

from pathlib import Path

from valocoach.cli import display
from valocoach.core.memory import ConversationMemory

# Slash command descriptions used in /help.  Single source of truth — the
# WordCompleter pulls keys from this dict so adding a command here is enough
# to make it tab-completable.
_SLASH_HELP: dict[str, str] = {
    "/help":   "Show this help message.",
    "/clear":  "Clear conversation memory — start a fresh session.",
    "/memory": "Show turn count and token usage in the current window.",
    "/stats":  "Display your recent stats card.",
    "/quit":   "Exit the REPL (also: Ctrl-D, Ctrl-C).",
}

_WELCOME = """
[bold green]ValoCoach interactive mode[/bold green]
Type a coaching question, e.g. [italic]"losing on Ascent attack as Jett 8-12"[/italic].
Slash commands: [cyan]{cmds}[/cyan]  ·  [dim]Ctrl-D / Ctrl-C to quit[/dim]
""".strip()


def _print_help() -> None:
    display.console.print("\n[bold]Slash commands:[/bold]")
    for cmd, desc in _SLASH_HELP.items():
        display.console.print(f"  [cyan]{cmd:<10}[/cyan] {desc}")
    display.console.print()


def _handle_slash(cmd: str, memory: ConversationMemory) -> None:
    """Dispatch a slash command entered in the REPL.

    Raises ``SystemExit`` for ``/quit`` so the caller can break the REPL
    loop cleanly without an extra return-value protocol.
    """
    cmd = cmd.strip().lower().split()[0]  # normalize, ignore trailing args

    if cmd == "/help":
        _print_help()

    elif cmd == "/clear":
        memory.clear()
        display.success("Conversation memory cleared.")

    elif cmd == "/memory":
        turns = len(memory)
        tokens = memory.token_count
        display.info(f"Memory: {turns} turn(s) · {tokens} tokens")

    elif cmd == "/stats":
        try:
            from valocoach.cli.commands.stats import run_stats
            run_stats()
        except Exception as exc:
            display.warn(f"Couldn't load stats: {exc}")

    elif cmd == "/quit":
        raise SystemExit(0)

    else:
        display.warn(f"Unknown command: {cmd!r}  (try /help)")


def _build_completer():
    """Build a WordCompleter with agent names, map names, and slash commands.

    Returns None if prompt_toolkit / the JSON loaders are unavailable so the
    REPL still works (no autocomplete) on a stripped-down install.
    """
    try:
        from prompt_toolkit.completion import WordCompleter

        from valocoach.retrieval import list_agent_names, list_map_names

        words = (
            list(list_agent_names())
            + list(list_map_names())
            + list(_SLASH_HELP.keys())
        )
        return WordCompleter(words, ignore_case=True, sentence=True)
    except Exception:
        return None


_OLLAMA_RECONNECT_KEYWORDS = ("connection", "refused", "timeout", "unreachable", "connect")


def run_interactive() -> None:
    """Start the interactive coaching REPL.

    Exits cleanly on Ctrl-D (EOF), ``/quit``, or repeated Ctrl-C.
    """
    # Pre-flight: verify Ollama is reachable before bothering with prompt_toolkit.
    # Checked first so users get the most actionable error even when prompt_toolkit
    # is also missing — "start ollama serve" is always the first thing to fix.
    from valocoach.core.config import load_settings
    from valocoach.core.preflight import check_ollama

    settings = load_settings()
    ollama_result = check_ollama(settings)
    if not ollama_result.ok:
        display.error(ollama_result.message)
        if ollama_result.hint:
            display.warn(ollama_result.hint)
        return

    try:
        from prompt_toolkit import PromptSession
        from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
        from prompt_toolkit.history import FileHistory
    except ImportError:
        display.error(
            "prompt_toolkit is required for interactive mode. "
            "Install it with: pip install prompt-toolkit"
        )
        return

    # 10 exchanges = 20 individual turns (user + assistant each count as 1).
    memory = ConversationMemory(max_turns=20, max_tokens=3_000)

    # History file — persist ↑↓ command recall across sessions.
    history_path = Path.home() / ".valocoach" / "history"
    history_path.parent.mkdir(parents=True, exist_ok=True)

    session: PromptSession = PromptSession(
        history=FileHistory(str(history_path)),
        auto_suggest=AutoSuggestFromHistory(),
        completer=_build_completer(),
        complete_while_typing=False,  # Tab-only, not intrusive mid-sentence
    )

    cmd_list = "  ".join(_SLASH_HELP.keys())
    display.console.print(_WELCOME.format(cmds=cmd_list))
    display.console.print()

    # Lazy-import run_coach so prompt_toolkit ImportError above can return
    # before we pull the heavy LLM/retrieval graph into memory.
    from valocoach.cli.commands.coach import run_coach

    while True:
        try:
            raw = session.prompt("valocoach> ")
        except KeyboardInterrupt:
            display.console.print("\n[dim]Interrupted — type /quit or Ctrl-D to exit.[/dim]")
            continue
        except EOFError:
            display.console.print("\n[dim]Bye.[/dim]")
            break

        user_input = raw.strip()
        if not user_input:
            continue

        # --- Slash commands ---
        if user_input.startswith("/"):
            try:
                _handle_slash(user_input, memory)
            except SystemExit:
                display.console.print("[dim]Bye.[/dim]")
                break
            continue

        # --- Coaching turn ---
        # Snapshot the prior history *before* adding the new user turn — the
        # LLM should see [prev_user, prev_assistant, …] and then the current
        # user message that ``run_coach`` builds itself, not the question twice.
        prior_history = memory.messages if not memory.is_empty else None

        try:
            response = run_coach(
                situation=user_input,
                conversation_history=prior_history,
            )
        except Exception as exc:
            # Surface a targeted reconnect hint when the error looks like an
            # Ollama connectivity failure (e.g. the server was killed mid-session).
            err_lower = str(exc).lower()
            if any(kw in err_lower for kw in _OLLAMA_RECONNECT_KEYWORDS):
                display.error("LLM call failed — Ollama may have stopped.")
                display.warn(
                    "Check with:  ollama list"
                    "  |  Restart with:  ollama serve"
                )
            else:
                display.error(f"Coaching failed: {exc}")
            continue

        # Store both sides only when we got a response — partial / failed
        # streams should not pollute memory with one-sided turns.
        if response:
            memory.add("user", user_input)
            memory.add("assistant", response)

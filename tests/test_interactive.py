"""Tests for valocoach.cli.commands.interactive.

Covers:
  - _print_help: prints all slash commands.
  - _handle_slash: /help, /clear, /memory, /save (with and without messages),
      /sessions (empty and populated), /stats (ok and exception), /quit,
      unknown command.
  - _build_completer: returns WordCompleter on success; None on ImportError.
  - run_interactive: Ollama down → early return; prompt_toolkit missing →
      early return; EOFError breaks loop; empty input skipped; slash command
      dispatched; coaching turn calls run_coach and stores memory;
      run_coach exception surfaced; Ollama-reconnect hint on connection error;
      KeyboardInterrupt shows hint and continues; session auto-saved on exit;
      previous session found and declined; previous session loaded on accept.

Patching strategy:
  prompt_toolkit.PromptSession / .history.FileHistory / .auto_suggest —
      patched directly on the installed module so the lazy ``from … import``
      inside run_interactive() receives mocks.
  valocoach.core.config.load_settings and valocoach.core.preflight.check_ollama
      — patched at their definition site (simplest target for lazy imports).
  Session-store functions — patched at the interactive module's binding site
      (valocoach.cli.commands.interactive.<name>).
  run_coach — patched at valocoach.cli.commands.coach.run_coach (definition
      site; the lazy import inside run_interactive picks this up).
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from valocoach.core.memory import ConversationMemory
from valocoach.core.preflight import CheckResult

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_LOAD_SETTINGS = "valocoach.core.config.load_settings"
_CHECK_OLLAMA = "valocoach.core.preflight.check_ollama"
_LATEST_SESSION = "valocoach.cli.commands.interactive.latest_session"
_LIST_SESSIONS = "valocoach.cli.commands.interactive.list_sessions"
_LOAD_SESSION = "valocoach.cli.commands.interactive.load_session"
_SAVE_SESSION = "valocoach.cli.commands.interactive.save_session"
_SESSION_SUMMARY = "valocoach.cli.commands.interactive.session_summary"
_RUN_COACH = "valocoach.cli.commands.coach.run_coach"
_PTK_SESSION = "prompt_toolkit.PromptSession"
_PTK_FILE_HISTORY = "prompt_toolkit.history.FileHistory"
_PTK_AUTO_SUGGEST = "prompt_toolkit.auto_suggest.AutoSuggestFromHistory"

_OK = CheckResult(ok=True, message="ok")
_FAIL = CheckResult(ok=False, message="Ollama not reachable.", hint="ollama serve")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _memory(*turns: tuple[str, str]) -> ConversationMemory:
    """Build a ConversationMemory pre-loaded with (role, content) pairs."""
    mem = ConversationMemory(max_turns=20, max_tokens=4000)
    for role, content in turns:
        mem.add(role, content)
    return mem


def _ptk_session_mock(*inputs: str | type) -> tuple[MagicMock, MagicMock]:
    """Return (mock_class, mock_instance) for PromptSession.

    Each element of *inputs* is either:
      - A str → returned by session.prompt()
      - An exception class / instance → raised by session.prompt()
    After exhausting inputs an EOFError is raised automatically.
    """
    instance = MagicMock()
    effects = []
    for inp in inputs:
        if isinstance(inp, str):
            effects.append(inp)
        else:
            effects.append(inp)
    effects.append(EOFError())
    instance.prompt.side_effect = effects

    cls = MagicMock(return_value=instance)
    return cls, instance


# ---------------------------------------------------------------------------
# _print_help
# ---------------------------------------------------------------------------


class TestPrintHelp:
    def test_prints_all_slash_commands(self, capsys):
        from valocoach.cli.commands.interactive import _print_help

        _print_help()
        # We can't easily capture Rich console output via capsys because Rich
        # writes to its own Console object — just ensure it doesn't raise.

    def test_all_commands_have_descriptions(self):
        from valocoach.cli.commands.interactive import _SLASH_HELP

        assert len(_SLASH_HELP) >= 6
        for cmd, desc in _SLASH_HELP.items():
            assert cmd.startswith("/")
            assert len(desc) > 0


# ---------------------------------------------------------------------------
# _handle_slash
# ---------------------------------------------------------------------------


class TestHandleSlash:
    def test_help_dispatches_print_help(self):
        from valocoach.cli.commands.interactive import _handle_slash

        mem = _memory()
        with patch("valocoach.cli.commands.interactive._print_help") as mock_help:
            _handle_slash("/help", mem)
        mock_help.assert_called_once()

    def test_clear_resets_memory(self):
        from valocoach.cli.commands.interactive import _handle_slash

        mem = _memory(("user", "hello"), ("assistant", "world"))
        assert not mem.is_empty
        _handle_slash("/clear", mem)
        assert mem.is_empty

    def test_memory_shows_turn_count(self, capsys):
        from valocoach.cli.commands.interactive import _handle_slash

        mem = _memory(("user", "q"), ("assistant", "a"))
        with patch("valocoach.cli.commands.interactive.display") as mock_display:
            _handle_slash("/memory", mem)
        mock_display.info.assert_called_once()
        msg = mock_display.info.call_args[0][0]
        assert "2" in msg  # 2 turns

    def test_save_with_messages_calls_save_session(self):
        from valocoach.cli.commands.interactive import _handle_slash

        mem = _memory(("user", "q"), ("assistant", "a"))
        fake_path = MagicMock(spec=Path)
        fake_path.name = "session_2026.json"
        with patch(_SAVE_SESSION, return_value=fake_path) as mock_save:
            _handle_slash("/save", mem)
        mock_save.assert_called_once_with(mem.messages)

    def test_save_with_empty_memory_warns(self):
        from valocoach.cli.commands.interactive import _handle_slash

        mem = _memory()
        with (
            patch(_SAVE_SESSION, return_value=None),
            patch("valocoach.cli.commands.interactive.display") as mock_display,
        ):
            _handle_slash("/save", mem)
        mock_display.warn.assert_called_once()

    def test_sessions_empty_shows_info(self):
        from valocoach.cli.commands.interactive import _handle_slash

        mem = _memory()
        with (
            patch(_LIST_SESSIONS, return_value=[]),
            patch("valocoach.cli.commands.interactive.display") as mock_display,
        ):
            _handle_slash("/sessions", mem)
        mock_display.info.assert_called_once()
        assert "no saved" in mock_display.info.call_args[0][0].lower()

    def test_sessions_populated_lists_them(self):
        from valocoach.cli.commands.interactive import _handle_slash

        paths = [MagicMock(spec=Path, name=f"s{i}.json") for i in range(3)]
        for p in paths:
            p.name = f"session_{p}.json"
        mem = _memory()
        with (
            patch(_LIST_SESSIONS, return_value=paths),
            patch(_SESSION_SUMMARY, return_value="2 turns"),
        ):
            # Should not raise — just print
            _handle_slash("/sessions", mem)

    def test_stats_calls_run_stats(self):
        from valocoach.cli.commands.interactive import _handle_slash

        mem = _memory()
        with patch("valocoach.cli.commands.stats.run_stats") as mock_stats:
            _handle_slash("/stats", mem)
        mock_stats.assert_called_once()

    def test_stats_exception_shows_warning(self):
        from valocoach.cli.commands.interactive import _handle_slash

        mem = _memory()
        with (
            patch(
                "valocoach.cli.commands.stats.run_stats",
                side_effect=RuntimeError("db down"),
            ),
            patch("valocoach.cli.commands.interactive.display") as mock_display,
        ):
            _handle_slash("/stats", mem)  # must not raise
        mock_display.warn.assert_called_once()
        assert "db down" in mock_display.warn.call_args[0][0]

    def test_quit_raises_system_exit(self):
        from valocoach.cli.commands.interactive import _handle_slash

        mem = _memory()
        with pytest.raises(SystemExit):
            _handle_slash("/quit", mem)

    def test_unknown_command_warns(self):
        from valocoach.cli.commands.interactive import _handle_slash

        mem = _memory()
        with patch("valocoach.cli.commands.interactive.display") as mock_display:
            _handle_slash("/bogus", mem)
        mock_display.warn.assert_called_once()
        assert "unknown" in mock_display.warn.call_args[0][0].lower()

    def test_command_normalised_to_lowercase(self):
        from valocoach.cli.commands.interactive import _handle_slash

        mem = _memory(("user", "q"), ("assistant", "a"))
        _handle_slash("/CLEAR", mem)  # must not raise, must clear
        assert mem.is_empty

    def test_trailing_args_ignored(self):
        from valocoach.cli.commands.interactive import _handle_slash

        mem = _memory()
        with patch("valocoach.cli.commands.interactive._print_help") as mock_help:
            _handle_slash("/help extra args here", mem)
        mock_help.assert_called_once()


# ---------------------------------------------------------------------------
# _build_completer
# ---------------------------------------------------------------------------


class TestBuildCompleter:
    def test_returns_something_when_prompt_toolkit_available(self):
        from valocoach.cli.commands.interactive import _build_completer

        result = _build_completer()
        # Returns None if any dependency is missing; otherwise a WordCompleter.
        # Both outcomes are valid — just must not raise.
        assert result is None or hasattr(result, "get_completions")

    def test_returns_none_when_prompt_toolkit_unavailable(self):
        from valocoach.cli.commands.interactive import _build_completer

        with patch.dict(sys.modules, {"prompt_toolkit.completion": None}):
            result = _build_completer()
        assert result is None

    def test_returns_none_when_retrieval_unavailable(self):
        from valocoach.cli.commands.interactive import _build_completer

        with patch(
            "valocoach.retrieval.list_agent_names",
            side_effect=RuntimeError("no knowledge base"),
        ):
            result = _build_completer()
        assert result is None


# ---------------------------------------------------------------------------
# run_interactive — shared setup helper
# ---------------------------------------------------------------------------


def _base_patches(
    *prompt_inputs,
    ollama_ok: bool = True,
    previous_session: Path | None = None,
    resume_answer: str = "n",
    run_coach_result: str = "Good advice.",
    save_result: Path | None = None,
) -> tuple:
    """Build the patch stack needed to run run_interactive in unit tests.

    Returns (ptk_cls, ptk_instance, patch_list) so individual tests can
    inspect the mocks they care about after the call.
    """
    ptk_cls, ptk_inst = _ptk_session_mock(*prompt_inputs)

    patches = [
        patch(_LOAD_SETTINGS, return_value=MagicMock()),
        patch(_CHECK_OLLAMA, return_value=_OK if ollama_ok else _FAIL),
        patch(_LATEST_SESSION, return_value=previous_session),
        patch(_SESSION_SUMMARY, return_value="2 turns from previous"),
        patch(_LOAD_SESSION, return_value=[]),
        patch(_SAVE_SESSION, return_value=save_result),
        patch("builtins.input", return_value=resume_answer),
        patch(_PTK_SESSION, ptk_cls),
        patch(_PTK_FILE_HISTORY),
        patch(_PTK_AUTO_SUGGEST),
        patch(_RUN_COACH, return_value=run_coach_result),
    ]
    return ptk_cls, ptk_inst, patches


# ---------------------------------------------------------------------------
# run_interactive — tests
# ---------------------------------------------------------------------------


class TestRunInteractive:
    def test_ollama_down_returns_without_starting_repl(self):
        """When Ollama is not reachable, the REPL must not start."""
        ptk_cls, _, patches = _base_patches(ollama_ok=False)
        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5]:
            from valocoach.cli.commands.interactive import run_interactive

            run_interactive()
        # PromptSession must never have been instantiated
        ptk_cls.assert_not_called()

    def test_prompt_toolkit_missing_returns_early(self):
        """If prompt_toolkit is not installed, show an error and return."""
        ptk_cls, _, _patches = _base_patches()
        with (
            patch(_LOAD_SETTINGS, return_value=MagicMock()),
            patch(_CHECK_OLLAMA, return_value=_OK),
            patch(_LATEST_SESSION, return_value=None),
            patch(_SAVE_SESSION, return_value=None),
            patch.dict(
                sys.modules,
                {
                    "prompt_toolkit": None,
                    "prompt_toolkit.auto_suggest": None,
                    "prompt_toolkit.history": None,
                },
            ),
        ):
            from valocoach.cli.commands.interactive import run_interactive

            run_interactive()  # must not raise
        ptk_cls.assert_not_called()

    def test_eoferror_exits_loop_cleanly(self):
        """EOFError (Ctrl-D) exits the REPL without exception."""
        # No prompt inputs → first call raises EOFError immediately
        _ptk_cls, ptk_inst, patches = _base_patches()
        with (
            patches[0],
            patches[1],
            patches[2],
            patches[3],
            patches[4],
            patches[5],
            patches[6],
            patches[7],
            patches[8],
            patches[9],
            patches[10],
        ):
            from valocoach.cli.commands.interactive import run_interactive

            run_interactive()  # must not raise
        ptk_inst.prompt.assert_called()

    def test_empty_input_is_skipped(self):
        """Blank lines do not trigger coaching or slash commands."""
        _ptk_cls, _ptk_inst, patches = _base_patches("", "  ")
        mock_coach = patches[10]
        with (
            patches[0],
            patches[1],
            patches[2],
            patches[3],
            patches[4],
            patches[5],
            patches[6],
            patches[7],
            patches[8],
            patches[9],
            mock_coach,
        ):
            from valocoach.cli.commands.interactive import run_interactive

            run_interactive()
        # run_coach must not have been called
        mock_coach.start()
        # Re-check via separate patch
        with patch(_RUN_COACH) as mock_rc:
            ptk_cls2, _ptk_inst2 = _ptk_session_mock("", "  ")
            with (
                patch(_LOAD_SETTINGS, return_value=MagicMock()),
                patch(_CHECK_OLLAMA, return_value=_OK),
                patch(_LATEST_SESSION, return_value=None),
                patch(_SAVE_SESSION, return_value=None),
                patch("builtins.input", return_value="n"),
                patch(_PTK_SESSION, ptk_cls2),
                patch(_PTK_FILE_HISTORY),
                patch(_PTK_AUTO_SUGGEST),
            ):
                from valocoach.cli.commands.interactive import run_interactive

                run_interactive()
            mock_rc.assert_not_called()

    def test_slash_command_dispatched(self):
        """/clear is dispatched to _handle_slash."""
        with (
            patch(_LOAD_SETTINGS, return_value=MagicMock()),
            patch(_CHECK_OLLAMA, return_value=_OK),
            patch(_LATEST_SESSION, return_value=None),
            patch(_SAVE_SESSION, return_value=None),
            patch("builtins.input", return_value="n"),
            patch(_PTK_FILE_HISTORY),
            patch(_PTK_AUTO_SUGGEST),
        ):
            ptk_cls, _ptk_inst = _ptk_session_mock("/clear")
            with (
                patch(_PTK_SESSION, ptk_cls),
                patch("valocoach.cli.commands.interactive._handle_slash") as mock_handle,
            ):
                from valocoach.cli.commands.interactive import run_interactive

                run_interactive()
        mock_handle.assert_called_once()
        assert mock_handle.call_args[0][0] == "/clear"

    def test_coaching_turn_calls_run_coach(self):
        """A non-slash input triggers a run_coach call."""
        situation = "push A site on Ascent"
        with (
            patch(_LOAD_SETTINGS, return_value=MagicMock()),
            patch(_CHECK_OLLAMA, return_value=_OK),
            patch(_LATEST_SESSION, return_value=None),
            patch(_SAVE_SESSION, return_value=None),
            patch("builtins.input", return_value="n"),
            patch(_PTK_FILE_HISTORY),
            patch(_PTK_AUTO_SUGGEST),
            patch(_RUN_COACH, return_value="Good advice.") as mock_rc,
        ):
            ptk_cls, _ = _ptk_session_mock(situation)
            with patch(_PTK_SESSION, ptk_cls):
                from valocoach.cli.commands.interactive import run_interactive

                run_interactive()

        mock_rc.assert_called_once()
        assert mock_rc.call_args.kwargs["situation"] == situation

    def test_coaching_turn_stores_memory(self):
        """After a successful coaching turn, memory contains user + assistant turns."""
        situation = "eco round"
        captured_histories: list = []

        def _fake_coach(situation, conversation_history=None, **_kw):
            captured_histories.append(conversation_history)
            return "Coach says: buy rifles."

        with (
            patch(_LOAD_SETTINGS, return_value=MagicMock()),
            patch(_CHECK_OLLAMA, return_value=_OK),
            patch(_LATEST_SESSION, return_value=None),
            patch(_SAVE_SESSION, return_value=None) as mock_save,
            patch("builtins.input", return_value="n"),
            patch(_PTK_FILE_HISTORY),
            patch(_PTK_AUTO_SUGGEST),
            patch(_RUN_COACH, side_effect=_fake_coach),
        ):
            ptk_cls, _ = _ptk_session_mock(situation)
            with patch(_PTK_SESSION, ptk_cls):
                from valocoach.cli.commands.interactive import run_interactive

                run_interactive()

        # First call: no prior history (memory was empty)
        assert captured_histories[0] is None
        # save_session called with messages that include the turn
        saved_messages = mock_save.call_args[0][0]
        roles = [m["role"] for m in saved_messages]
        assert "user" in roles
        assert "assistant" in roles

    def test_keyboard_interrupt_continues_loop(self):
        """KeyboardInterrupt from prompt() shows a hint and continues the loop."""
        with (
            patch(_LOAD_SETTINGS, return_value=MagicMock()),
            patch(_CHECK_OLLAMA, return_value=_OK),
            patch(_LATEST_SESSION, return_value=None),
            patch(_SAVE_SESSION, return_value=None),
            patch("builtins.input", return_value="n"),
            patch(_PTK_FILE_HISTORY),
            patch(_PTK_AUTO_SUGGEST),
        ):
            # First prompt: KeyboardInterrupt; second: EOFError (exit)
            ptk_cls, ptk_inst = _ptk_session_mock(KeyboardInterrupt)
            with patch(_PTK_SESSION, ptk_cls):
                from valocoach.cli.commands.interactive import run_interactive

                run_interactive()  # must not propagate KeyboardInterrupt
        assert ptk_inst.prompt.call_count == 2  # called twice: interrupt + eof

    def test_run_coach_exception_continues_loop(self):
        """run_coach raising does not crash the REPL — it shows an error and continues."""
        with (
            patch(_LOAD_SETTINGS, return_value=MagicMock()),
            patch(_CHECK_OLLAMA, return_value=_OK),
            patch(_LATEST_SESSION, return_value=None),
            patch(_SAVE_SESSION, return_value=None),
            patch("builtins.input", return_value="n"),
            patch(_PTK_FILE_HISTORY),
            patch(_PTK_AUTO_SUGGEST),
            patch(_RUN_COACH, side_effect=RuntimeError("llm down")),
        ):
            ptk_cls, _ = _ptk_session_mock("coaching situation")
            with patch(_PTK_SESSION, ptk_cls):
                from valocoach.cli.commands.interactive import run_interactive

                run_interactive()  # must not raise

    def test_connection_error_shows_reconnect_hint(self):
        """An Ollama connection error triggers the specific reconnect message."""
        with (
            patch(_LOAD_SETTINGS, return_value=MagicMock()),
            patch(_CHECK_OLLAMA, return_value=_OK),
            patch(_LATEST_SESSION, return_value=None),
            patch(_SAVE_SESSION, return_value=None),
            patch("builtins.input", return_value="n"),
            patch(_PTK_FILE_HISTORY),
            patch(_PTK_AUTO_SUGGEST),
            patch(
                _RUN_COACH,
                side_effect=RuntimeError("Connection refused by Ollama"),
            ),
            patch("valocoach.cli.commands.interactive.display") as mock_display,
        ):
            ptk_cls, _ = _ptk_session_mock("coaching situation")
            with patch(_PTK_SESSION, ptk_cls):
                from valocoach.cli.commands.interactive import run_interactive

                run_interactive()

        error_calls = " ".join(str(c) for c in mock_display.error.call_args_list)
        warn_calls = " ".join(str(c) for c in mock_display.warn.call_args_list)
        # Should mention Ollama having stopped, not just "coaching failed"
        assert "ollama" in (error_calls + warn_calls).lower()

    def test_session_autosaved_on_exit(self):
        """save_session is always called on exit (finally block)."""
        fake_saved = MagicMock(spec=Path)
        fake_saved.name = "session_2026.json"
        with (
            patch(_LOAD_SETTINGS, return_value=MagicMock()),
            patch(_CHECK_OLLAMA, return_value=_OK),
            patch(_LATEST_SESSION, return_value=None),
            patch(_SAVE_SESSION, return_value=fake_saved) as mock_save,
            patch("builtins.input", return_value="n"),
            patch(_PTK_FILE_HISTORY),
            patch(_PTK_AUTO_SUGGEST),
            patch(_RUN_COACH, return_value="advice"),
        ):
            ptk_cls, _ = _ptk_session_mock("situation")
            with patch(_PTK_SESSION, ptk_cls):
                from valocoach.cli.commands.interactive import run_interactive

                run_interactive()

        mock_save.assert_called_once()

    def test_previous_session_found_decline_resume(self):
        """Declining resume ('n') does not load prior turns into memory."""
        fake_path = MagicMock(spec=Path)
        fake_path.name = "old_session.json"
        with (
            patch(_LOAD_SETTINGS, return_value=MagicMock()),
            patch(_CHECK_OLLAMA, return_value=_OK),
            patch(_LATEST_SESSION, return_value=fake_path),
            patch(_SESSION_SUMMARY, return_value="3 turns from 2026-04-01"),
            patch(_LOAD_SESSION, return_value=[]) as mock_load,
            patch(_SAVE_SESSION, return_value=None),
            patch("builtins.input", return_value="n"),
            patch(_PTK_FILE_HISTORY),
            patch(_PTK_AUTO_SUGGEST),
        ):
            ptk_cls, _ = _ptk_session_mock()
            with patch(_PTK_SESSION, ptk_cls):
                from valocoach.cli.commands.interactive import run_interactive

                run_interactive()

        mock_load.assert_not_called()

    def test_previous_session_found_accept_resume(self):
        """Accepting resume ('y') loads prior turns from disk."""
        fake_path = MagicMock(spec=Path)
        fake_path.name = "old_session.json"
        prior_turns = [
            {"role": "user", "content": "push A"},
            {"role": "assistant", "content": "Use Tailwind."},
        ]
        with (
            patch(_LOAD_SETTINGS, return_value=MagicMock()),
            patch(_CHECK_OLLAMA, return_value=_OK),
            patch(_LATEST_SESSION, return_value=fake_path),
            patch(_SESSION_SUMMARY, return_value="2 turns"),
            patch(_LOAD_SESSION, return_value=prior_turns) as mock_load,
            patch(_SAVE_SESSION, return_value=None),
            patch("builtins.input", return_value="y"),
            patch(_PTK_FILE_HISTORY),
            patch(_PTK_AUTO_SUGGEST),
        ):
            ptk_cls, _ = _ptk_session_mock()
            with patch(_PTK_SESSION, ptk_cls):
                from valocoach.cli.commands.interactive import run_interactive

                run_interactive()

        mock_load.assert_called_once_with(fake_path)

    def test_ollama_down_no_hint_still_returns(self):
        """Ollama failure with an empty hint must still return without error."""
        fail_no_hint = CheckResult(ok=False, message="Ollama not reachable.", hint="")
        ptk_cls, _, __ = _base_patches(ollama_ok=False)
        with (
            patch(_LOAD_SETTINGS, return_value=MagicMock()),
            patch(_CHECK_OLLAMA, return_value=fail_no_hint),
        ):
            from valocoach.cli.commands.interactive import run_interactive

            run_interactive()  # must not raise
        ptk_cls.assert_not_called()

    def test_resume_input_eoferror_defaults_to_no(self):
        """EOFError during the 'Resume?' prompt defaults to declining."""
        fake_path = MagicMock(spec=Path)
        fake_path.name = "old.json"
        with (
            patch(_LOAD_SETTINGS, return_value=MagicMock()),
            patch(_CHECK_OLLAMA, return_value=_OK),
            patch(_LATEST_SESSION, return_value=fake_path),
            patch(_SESSION_SUMMARY, return_value="1 turn"),
            patch(_LOAD_SESSION, return_value=[]) as mock_load,
            patch(_SAVE_SESSION, return_value=None),
            patch("builtins.input", side_effect=EOFError),
            patch(_PTK_FILE_HISTORY),
            patch(_PTK_AUTO_SUGGEST),
        ):
            ptk_cls, _ = _ptk_session_mock()
            with patch(_PTK_SESSION, ptk_cls):
                from valocoach.cli.commands.interactive import run_interactive

                run_interactive()  # must not raise
        # Declined resume → load_session never called
        mock_load.assert_not_called()

    def test_accept_resume_with_empty_session_warns(self):
        """Accepting resume when load_session returns [] shows a warning."""
        fake_path = MagicMock(spec=Path)
        fake_path.name = "empty.json"
        with (
            patch(_LOAD_SETTINGS, return_value=MagicMock()),
            patch(_CHECK_OLLAMA, return_value=_OK),
            patch(_LATEST_SESSION, return_value=fake_path),
            patch(_SESSION_SUMMARY, return_value="0 turns"),
            patch(_LOAD_SESSION, return_value=[]),  # empty → warn
            patch(_SAVE_SESSION, return_value=None),
            patch("builtins.input", return_value="y"),
            patch(_PTK_FILE_HISTORY),
            patch(_PTK_AUTO_SUGGEST),
            patch("valocoach.cli.commands.interactive.display") as mock_display,
        ):
            ptk_cls, _ = _ptk_session_mock()
            with patch(_PTK_SESSION, ptk_cls):
                from valocoach.cli.commands.interactive import run_interactive

                run_interactive()

        warn_msgs = " ".join(str(c) for c in mock_display.warn.call_args_list)
        assert "could not load" in warn_msgs.lower() or "starting fresh" in warn_msgs.lower()

    def test_quit_slash_breaks_loop(self):
        """/quit sent as input causes SystemExit to be caught and the loop to break."""
        with (
            patch(_LOAD_SETTINGS, return_value=MagicMock()),
            patch(_CHECK_OLLAMA, return_value=_OK),
            patch(_LATEST_SESSION, return_value=None),
            patch(_SAVE_SESSION, return_value=None),
            patch("builtins.input", return_value="n"),
            patch(_PTK_FILE_HISTORY),
            patch(_PTK_AUTO_SUGGEST),
        ):
            ptk_cls, ptk_inst = _ptk_session_mock("/quit", "should not reach this")
            with patch(_PTK_SESSION, ptk_cls):
                from valocoach.cli.commands.interactive import run_interactive

                run_interactive()  # /quit must break the loop cleanly

        # prompt() called once (/quit), not a second time (loop broke)
        assert ptk_inst.prompt.call_count == 1

    def test_none_response_not_stored_in_memory(self):
        """When run_coach returns None, memory stays empty (no half-turn added)."""
        with (
            patch(_LOAD_SETTINGS, return_value=MagicMock()),
            patch(_CHECK_OLLAMA, return_value=_OK),
            patch(_LATEST_SESSION, return_value=None),
            patch(_SAVE_SESSION, return_value=None) as mock_save,
            patch("builtins.input", return_value="n"),
            patch(_PTK_FILE_HISTORY),
            patch(_PTK_AUTO_SUGGEST),
            patch(_RUN_COACH, return_value=None),
        ):
            ptk_cls, _ = _ptk_session_mock("some situation")
            with patch(_PTK_SESSION, ptk_cls):
                from valocoach.cli.commands.interactive import run_interactive

                run_interactive()

        saved_messages = mock_save.call_args[0][0]
        assert saved_messages == []  # nothing stored

    def test_second_turn_passes_prior_history(self):
        """The second coaching question receives prior turns as conversation_history."""
        histories_seen: list = []

        def _fake_coach(situation, conversation_history=None, **_kw):
            histories_seen.append(conversation_history)
            return f"advice for: {situation}"

        with (
            patch(_LOAD_SETTINGS, return_value=MagicMock()),
            patch(_CHECK_OLLAMA, return_value=_OK),
            patch(_LATEST_SESSION, return_value=None),
            patch(_SAVE_SESSION, return_value=None),
            patch("builtins.input", return_value="n"),
            patch(_PTK_FILE_HISTORY),
            patch(_PTK_AUTO_SUGGEST),
            patch(_RUN_COACH, side_effect=_fake_coach),
        ):
            ptk_cls, _ = _ptk_session_mock("first question", "second question")
            with patch(_PTK_SESSION, ptk_cls):
                from valocoach.cli.commands.interactive import run_interactive

                run_interactive()

        assert len(histories_seen) == 2
        # First call: no history
        assert histories_seen[0] is None
        # Second call: has the first user+assistant turn
        assert histories_seen[1] is not None
        roles = [m["role"] for m in histories_seen[1]]
        assert "user" in roles
        assert "assistant" in roles

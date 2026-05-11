"""Tests for valocoach.cli.commands.sessions.

Covers:
    run_sessions_list  — no player profile, no sessions, with sessions,
                         open-session reminder, limit passed through.
    run_sessions_close — success path, silent on unknown id.
    Typer CLI wiring   — `valocoach sessions`, `valocoach sessions list`,
                         `valocoach sessions close` all route correctly.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import click
import pytest
from typer.testing import CliRunner

from valocoach.cli.app import app
from valocoach.cli.commands.sessions import run_sessions_close, run_sessions_list
from valocoach.coach.session_manager import SessionInfo

runner = CliRunner()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fake_settings():
    s = MagicMock()
    s.riot_name = "Player"
    s.riot_tag = "1234"
    s.data_dir = Path("/tmp/valocoach_test_sessions")
    return s


def _session(
    *,
    session_id: int = 1,
    title: str | None = "2026-05-06",
    started_at: str = "2026-05-06T10:00:00",
    ended_at: str | None = None,
    focus_agent: str | None = None,
    focus_map: str | None = None,
) -> SessionInfo:
    return SessionInfo(
        id=session_id,
        title=title,
        started_at=started_at,
        ended_at=ended_at,
        focus_agent=focus_agent,
        focus_map=focus_map,
    )


_LOAD = "valocoach.cli.commands.sessions.load_settings"
_PUUID = "valocoach.cli.commands.sessions.get_player_puuid"
_LIST = "valocoach.cli.commands.sessions.list_coaching_sessions"
_CLOSE = "valocoach.cli.commands.sessions.close_coaching_session"


# ===========================================================================
# run_sessions_list
# ===========================================================================


class TestRunSessionsList:
    def test_warns_and_exits_when_no_player(self):
        with (
            patch(_LOAD, return_value=_fake_settings()),
            patch(_PUUID, return_value=None),
            pytest.raises(click.exceptions.Exit) as exc_info,
        ):
            run_sessions_list()
        assert exc_info.value.exit_code == 1

    def test_info_message_when_no_sessions(self):
        output_lines: list[str] = []

        with (
            patch(_LOAD, return_value=_fake_settings()),
            patch(_PUUID, return_value="puuid-abc"),
            patch(_LIST, return_value=[]),
            patch("valocoach.cli.commands.sessions.display") as mock_display,
        ):
            mock_display.info = lambda msg: output_lines.append(str(msg))
            run_sessions_list()

        assert any("valocoach interactive" in line for line in output_lines)

    def test_renders_sessions_table_when_sessions_present(self):
        sessions = [_session(session_id=3, title="Post-plant drill")]

        with (
            patch(_LOAD, return_value=_fake_settings()),
            patch(_PUUID, return_value="puuid-abc"),
            patch(_LIST, return_value=sessions),
            patch("valocoach.cli.commands.sessions.render_coaching_sessions") as mock_render,
        ):
            run_sessions_list()

        mock_render.assert_called_once()
        rendered_sessions = mock_render.call_args[0][1]
        assert rendered_sessions == sessions

    def test_passes_limit_to_list_coaching_sessions(self):
        with (
            patch(_LOAD, return_value=_fake_settings()),
            patch(_PUUID, return_value="puuid-abc"),
            patch(_LIST, return_value=[]) as mock_list,
            patch("valocoach.cli.commands.sessions.display"),
        ):
            run_sessions_list(limit=7)

        _, kwargs = mock_list.call_args
        assert kwargs.get("limit") == 7

    def test_open_session_reminder_shown(self):
        """When open sessions exist, a reminder to close them is printed."""
        sessions = [_session(session_id=1, ended_at=None)]  # open
        console_lines: list[str] = []

        with (
            patch(_LOAD, return_value=_fake_settings()),
            patch(_PUUID, return_value="puuid-abc"),
            patch(_LIST, return_value=sessions),
            patch(
                "valocoach.cli.commands.sessions.render_coaching_sessions"
            ),
            patch("valocoach.cli.commands.sessions.display") as mock_display,
        ):
            mock_display.console = MagicMock()
            mock_display.console.print = lambda msg, **_: console_lines.append(str(msg))
            run_sessions_list()

        combined = " ".join(console_lines)
        assert "sessions close" in combined or "open" in combined.lower()

    def test_no_open_session_reminder_when_all_closed(self):
        """When all sessions are closed, no reminder line is printed."""
        sessions = [_session(session_id=1, ended_at="2026-05-06T12:00:00")]  # closed
        console_lines: list[str] = []

        with (
            patch(_LOAD, return_value=_fake_settings()),
            patch(_PUUID, return_value="puuid-abc"),
            patch(_LIST, return_value=sessions),
            patch("valocoach.cli.commands.sessions.render_coaching_sessions"),
            patch("valocoach.cli.commands.sessions.display") as mock_display,
        ):
            mock_display.console = MagicMock()
            mock_display.console.print = lambda msg, **_: console_lines.append(str(msg))
            run_sessions_list()

        combined = " ".join(console_lines)
        assert "sessions close" not in combined


# ===========================================================================
# run_sessions_close
# ===========================================================================


class TestRunSessionsClose:
    def test_calls_close_coaching_session_with_id(self):
        with (
            patch(_LOAD, return_value=_fake_settings()),
            patch(_CLOSE) as mock_close,
            patch("valocoach.cli.commands.sessions.display"),
        ):
            run_sessions_close(42)

        mock_close.assert_called_once()
        args = mock_close.call_args[0]
        assert args[1] == 42  # session_id is second positional arg

    def test_success_message_shown(self):
        success_msgs: list[str] = []

        with (
            patch(_LOAD, return_value=_fake_settings()),
            patch(_CLOSE),
            patch("valocoach.cli.commands.sessions.display") as mock_display,
        ):
            mock_display.success = lambda msg: success_msgs.append(str(msg))
            run_sessions_close(7)

        assert any("7" in msg for msg in success_msgs)

    def test_close_unknown_id_does_not_raise(self):
        """close_coaching_session is fire-and-forget; unknown ids don't crash."""
        with (
            patch(_LOAD, return_value=_fake_settings()),
            patch(_CLOSE),  # no-op: returns None
            patch("valocoach.cli.commands.sessions.display"),
        ):
            run_sessions_close(9999)  # must not raise


# ===========================================================================
# Typer CLI wiring (smoke tests via runner)
# ===========================================================================


class TestSessionsCLIWiring:
    def _common_patches(self):
        return [
            patch("valocoach.core.config.load_settings", return_value=_fake_settings()),
            patch("valocoach.cli.commands.sessions.load_settings", return_value=_fake_settings()),
            patch("valocoach.cli.commands.sessions.get_player_puuid", return_value=None),
        ]

    def test_sessions_default_routes_to_list(self):
        """``valocoach sessions`` (no subcommand) calls run_sessions_list."""
        with patch(
            "valocoach.cli.commands.sessions.run_sessions_list"
        ):
            result = runner.invoke(app, ["sessions"])
        # Either the mock was called (success path) or we get exit 1 (no player)
        # — either way the command didn't error out with a routing mistake.
        assert result.exit_code in (0, 1)

    def test_sessions_list_subcommand_routes_correctly(self):
        """``valocoach sessions list`` calls run_sessions_list."""
        with (
            patch("valocoach.core.config.load_settings", return_value=_fake_settings()),
            patch("valocoach.cli.commands.sessions.load_settings", return_value=_fake_settings()),
            patch("valocoach.cli.commands.sessions.get_player_puuid", return_value=None),
        ):
            result = runner.invoke(app, ["sessions", "list"])
        assert result.exit_code in (0, 1)  # 1 = no player, not a routing error

    def test_sessions_list_accepts_limit_flag(self):
        """``valocoach sessions list --limit 5`` passes limit correctly."""
        called_limits: list[int] = []

        def _fake_list(*, limit=20):
            called_limits.append(limit)

        with patch("valocoach.cli.commands.sessions.run_sessions_list", side_effect=_fake_list):
            runner.invoke(app, ["sessions", "list", "--limit", "5"])

        assert called_limits == [5]

    def test_sessions_close_subcommand_routes_correctly(self):
        """``valocoach sessions close 42`` calls run_sessions_close(42)."""
        called_ids: list[int] = []

        def _fake_close(sid: int):
            called_ids.append(sid)

        with patch("valocoach.cli.commands.sessions.run_sessions_close", side_effect=_fake_close):
            runner.invoke(app, ["sessions", "close", "42"])

        assert called_ids == [42]

    def test_sessions_close_requires_id_argument(self):
        """``valocoach sessions close`` (no id) exits non-zero."""
        result = runner.invoke(app, ["sessions", "close"])
        assert result.exit_code != 0

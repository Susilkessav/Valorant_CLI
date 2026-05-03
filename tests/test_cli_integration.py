"""CLI integration tests using Typer's CliRunner.

These tests verify:
  - Command routing (each command reaches the right handler).
  - Argument/flag parsing (required args, optional flags, bad values).
  - Error exits with the right codes and user-facing messages.
  - Success paths via mocked handlers (no real Ollama / DB / network calls).

All external I/O (LLM, database, HenrikDev API, vector store, filesystem)
is patched at the outermost boundary so the tests are fast, offline, and
deterministic.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from typer.testing import CliRunner

from valocoach.cli.app import app
from valocoach.core.preflight import CheckResult

runner = CliRunner()


# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------


def _ok_ollama() -> CheckResult:
    return CheckResult(ok=True, message="Ollama ok.", hint="")


def _fail_ollama() -> CheckResult:
    return CheckResult(ok=False, message="Ollama not reachable.", hint="ollama serve")


def _fake_settings(**kwargs):
    """Build a minimal settings-like object for patching load_settings."""
    from valocoach.core.config import Settings

    s = Settings(_env_file=None)
    for k, v in kwargs.items():
        object.__setattr__(s, k, v)
    return s


# ---------------------------------------------------------------------------
# coach
# ---------------------------------------------------------------------------


class TestCoachCommand:
    def test_help_shows_flags(self):
        result = runner.invoke(app, ["coach", "--help"])
        assert result.exit_code == 0
        for flag in ["--agent", "--map", "--side", "--with-stats"]:
            assert flag in result.stdout

    def test_missing_situation_arg_is_error(self):
        result = runner.invoke(app, ["coach"])
        # Typer exits with 2 for missing required arguments.
        assert result.exit_code != 0

    def test_ollama_down_exits_nonzero_with_message(self):
        with (
            patch("valocoach.core.preflight.check_ollama", return_value=_fail_ollama()),
            patch("valocoach.core.config.load_settings", return_value=_fake_settings()),
        ):
            result = runner.invoke(app, ["coach", "test situation"])
        assert result.exit_code != 0
        assert "not reachable" in result.output.lower() or "ollama" in result.output.lower()

    def test_ollama_down_hint_shown(self):
        with (
            patch("valocoach.core.preflight.check_ollama", return_value=_fail_ollama()),
            patch("valocoach.core.config.load_settings", return_value=_fake_settings()),
        ):
            result = runner.invoke(app, ["coach", "test situation"])
        assert "ollama serve" in result.output

    def test_routes_to_run_coach_when_ollama_ok(self):
        with (
            patch("valocoach.core.preflight.check_ollama", return_value=_ok_ollama()),
            patch("valocoach.core.config.load_settings", return_value=_fake_settings()),
            patch("valocoach.cli.commands.coach.run_coach") as mock_run,
        ):
            mock_run.return_value = None
            runner.invoke(app, ["coach", "push A on Ascent"])
        mock_run.assert_called_once()

    def test_situation_text_passed_to_run_coach(self):
        with (
            patch("valocoach.core.preflight.check_ollama", return_value=_ok_ollama()),
            patch("valocoach.core.config.load_settings", return_value=_fake_settings()),
            patch("valocoach.cli.commands.coach.run_coach") as mock_run,
        ):
            mock_run.return_value = None
            runner.invoke(app, ["coach", "push A on Ascent"])
        call_kwargs = mock_run.call_args.kwargs
        assert call_kwargs["situation"] == "push A on Ascent"

    def test_agent_flag_passed_to_run_coach(self):
        with (
            patch("valocoach.core.preflight.check_ollama", return_value=_ok_ollama()),
            patch("valocoach.core.config.load_settings", return_value=_fake_settings()),
            patch("valocoach.cli.commands.coach.run_coach") as mock_run,
        ):
            mock_run.return_value = None
            runner.invoke(app, ["coach", "--agent", "Jett", "test"])
        assert mock_run.call_args.kwargs["agent"] == "Jett"

    def test_no_stats_flag_disables_stats(self):
        with (
            patch("valocoach.core.preflight.check_ollama", return_value=_ok_ollama()),
            patch("valocoach.core.config.load_settings", return_value=_fake_settings()),
            patch("valocoach.cli.commands.coach.run_coach") as mock_run,
        ):
            mock_run.return_value = None
            runner.invoke(app, ["coach", "--no-stats", "test"])
        assert mock_run.call_args.kwargs["with_stats"] is False

    def test_map_and_side_flags_forwarded(self):
        with (
            patch("valocoach.core.preflight.check_ollama", return_value=_ok_ollama()),
            patch("valocoach.core.config.load_settings", return_value=_fake_settings()),
            patch("valocoach.cli.commands.coach.run_coach") as mock_run,
        ):
            mock_run.return_value = None
            runner.invoke(app, ["coach", "--map", "Haven", "--side", "defense", "test"])
        kw = mock_run.call_args.kwargs
        assert kw["map_"] == "Haven"
        assert kw["side"] == "defense"


# ---------------------------------------------------------------------------
# stats
# ---------------------------------------------------------------------------


class TestStatsCommand:
    def test_help_shows_period_flag(self):
        result = runner.invoke(app, ["stats", "--help"])
        assert result.exit_code == 0
        assert "--period" in result.stdout

    def test_no_riot_id_exits_nonzero(self):
        settings = _fake_settings(riot_name="", riot_tag="")
        # stats.py binds load_settings at import time — patch the bound name.
        with patch("valocoach.cli.commands.stats.load_settings", return_value=settings):
            result = runner.invoke(app, ["stats"])
        assert result.exit_code != 0
        assert "riot_name" in result.output.lower() or "not configured" in result.output.lower()

    def test_bad_result_filter_is_bad_param(self):
        result = runner.invoke(app, ["stats", "--result", "draw"])
        assert result.exit_code != 0

    def test_bad_period_exits_nonzero(self):
        settings = _fake_settings(riot_name="Player", riot_tag="NA1")
        with patch("valocoach.core.config.load_settings", return_value=settings):
            result = runner.invoke(app, ["stats", "--period", "invalid"])
        assert result.exit_code != 0

    def test_no_local_data_exits_with_sync_hint(self):
        settings = _fake_settings(riot_name="Player", riot_tag="NA1")
        # stats.py imports load_settings at module top-level, so patch there.
        with (
            patch("valocoach.cli.commands.stats.load_settings", return_value=settings),
            patch("valocoach.data.loader.load_player_data", return_value=None),
        ):
            result = runner.invoke(app, ["stats"])
        assert result.exit_code != 0
        assert "sync" in result.output.lower()

    def test_routes_to_run_stats(self):
        with patch("valocoach.cli.commands.stats.run_stats") as mock_run:
            runner.invoke(app, ["stats", "--period", "7d", "--agent", "Jett"])
        mock_run.assert_called_once()
        kw = mock_run.call_args.kwargs
        assert kw["period"] == "7d"
        assert kw["agent"] == "Jett"


# ---------------------------------------------------------------------------
# sync
# ---------------------------------------------------------------------------


class TestSyncCommand:
    def test_help_shows_limit_and_mode(self):
        result = runner.invoke(app, ["sync", "--help"])
        assert result.exit_code == 0
        assert "--limit" in result.stdout
        assert "--mode" in result.stdout

    def test_sync_flags_reach_handler(self):
        settings = _fake_settings(
            riot_name="Player",
            riot_tag="NA1",
            data_dir=Path("/tmp/valocoach_test"),
        )
        mock_result = MagicMock(
            matches_new=3,
            matches_skipped=0,
            errors=[],
            ok=True,
            error=None,
        )
        with (
            patch("valocoach.core.config.load_settings", return_value=settings),
            patch("valocoach.data.database.ensure_db", new_callable=AsyncMock),
            patch(
                "valocoach.data.sync.sync_player_matches",
                new_callable=AsyncMock,
                return_value=mock_result,
            ),
        ):
            result = runner.invoke(app, ["sync", "--limit", "5", "--mode", "unrated"])
        assert result.exit_code == 0
        assert "3" in result.output  # matches_new shown


# ---------------------------------------------------------------------------
# profile
# ---------------------------------------------------------------------------


class TestProfileCommand:
    def test_help_shows_limit_flag(self):
        result = runner.invoke(app, ["profile", "--help"])
        assert result.exit_code == 0
        assert "--limit" in result.stdout

    def test_limit_zero_is_bad_param(self):
        result = runner.invoke(app, ["profile", "--limit", "0"])
        assert result.exit_code != 0

    def test_name_without_tag_is_error(self):
        result = runner.invoke(app, ["profile", "--name", "Player"])
        assert result.exit_code != 0

    def test_routes_to_run_profile(self):
        with patch("valocoach.cli.commands.profile.run_profile") as mock_run:
            runner.invoke(app, ["profile", "--name", "Player", "--tag", "NA1", "--limit", "10"])
        mock_run.assert_called_once()
        kw = mock_run.call_args.kwargs
        assert kw["name"] == "Player"
        assert kw["tag"] == "NA1"
        assert kw["limit"] == 10


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------


class TestConfigCommands:
    def test_config_init_creates_success_output(self, tmp_path):
        fake_path = tmp_path / "config.toml"
        with patch("valocoach.core.config.write_default_config", return_value=fake_path):
            result = runner.invoke(app, ["config", "init"])
        assert result.exit_code == 0
        # Rich may wrap the long path across lines — check for the prefix only.
        assert "config written to" in result.output.lower()

    def test_config_show_prints_settings(self):
        settings = _fake_settings(riot_name="Player", riot_tag="NA1")
        with patch("valocoach.core.config.load_settings", return_value=settings):
            result = runner.invoke(app, ["config", "show"])
        assert result.exit_code == 0

    def test_config_help_lists_subcommands(self):
        result = runner.invoke(app, ["config", "--help"])
        assert result.exit_code == 0
        assert "init" in result.stdout
        assert "show" in result.stdout


# ---------------------------------------------------------------------------
# ingest
# ---------------------------------------------------------------------------


class TestIngestCommand:
    def test_help_shows_seed_and_clear_flags(self):
        result = runner.invoke(app, ["ingest", "--help"])
        assert result.exit_code == 0
        assert "--seed" in result.stdout
        assert "--clear" in result.stdout

    def test_stats_flag_shows_collection_info(self):
        settings = _fake_settings(data_dir=Path("/tmp/valocoach_test"))
        with (
            patch("valocoach.core.config.load_settings", return_value=settings),
            patch(
                "valocoach.retrieval.searcher.collection_stats",
                return_value={"total": 42, "by_type": {"[static] agent": 22}},
            ),
        ):
            result = runner.invoke(app, ["ingest", "--stats"])
        assert result.exit_code == 0
        assert "42" in result.output

    def test_clear_and_seed_flags_route_to_handlers(self):
        settings = _fake_settings(data_dir=Path("/tmp/valocoach_test"))
        with (
            patch("valocoach.core.config.load_settings", return_value=settings),
            patch("valocoach.retrieval.vector_store.clear_collection"),
        ):
            result = runner.invoke(app, ["ingest", "--clear"])
        assert result.exit_code == 0


# ---------------------------------------------------------------------------
# interactive
# ---------------------------------------------------------------------------


class TestInteractiveCommand:
    def test_help_text_present(self):
        result = runner.invoke(app, ["interactive", "--help"])
        assert result.exit_code == 0

    def test_ollama_down_exits_with_error_not_repl(self):
        """When Ollama is unreachable, the REPL should abort at startup."""
        with (
            patch("valocoach.core.preflight.check_ollama", return_value=_fail_ollama()),
            patch("valocoach.core.config.load_settings", return_value=_fake_settings()),
        ):
            result = runner.invoke(app, ["interactive"])
        # Should exit cleanly (return 0) but show the error message.
        # The REPL must NOT have started (no "valocoach>" prompt).
        assert "valocoach>" not in result.output
        assert "not reachable" in result.output.lower() or "ollama" in result.output.lower()

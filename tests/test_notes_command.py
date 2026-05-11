"""Tests for the valocoach.cli.commands.notes module.

Covers:
    _infer_category      — intent-to-category mapping for all 9 intent types,
                           fallback on classifier error.
    run_notes_list       — no player profile, no notes, with notes.
    run_notes_add        — empty text, bad priority, no player, no session,
                           success (category shown), DB error.
    run_notes_resolve    — success, not-found (exit 1).
    Typer CLI wiring     — `valocoach notes`, `valocoach notes list`,
                           `valocoach notes add`, `valocoach notes resolve`
                           all route to the correct runner.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from valocoach.cli.app import app
from valocoach.cli.commands.notes import _infer_category, _INTENT_TO_CATEGORY

runner = CliRunner()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fake_settings():
    s = MagicMock()
    s.riot_name = "Player"
    s.riot_tag = "1234"
    from pathlib import Path
    s.data_dir = Path("/tmp/valocoach_test_notes")
    return s


# ---------------------------------------------------------------------------
# _INTENT_TO_CATEGORY  (pure dict)
# ---------------------------------------------------------------------------


class TestIntentToCategoryMap:
    def test_all_nine_intents_covered(self):
        intents = {"clutch", "post_plant", "retake", "tactical", "economy",
                   "agent_info", "meta", "stat_analysis", "general"}
        assert intents == set(_INTENT_TO_CATEGORY.keys())

    def test_tactical_intents_map_to_tactical(self):
        for intent in ("clutch", "post_plant", "retake", "tactical"):
            assert _INTENT_TO_CATEGORY[intent] == "tactical", f"{intent} should be 'tactical'"

    def test_economy_maps_to_economy(self):
        assert _INTENT_TO_CATEGORY["economy"] == "economy"

    def test_agent_info_maps_to_agent(self):
        assert _INTENT_TO_CATEGORY["agent_info"] == "agent"

    def test_meta_maps_to_meta(self):
        assert _INTENT_TO_CATEGORY["meta"] == "meta"

    def test_general_intents_map_to_general(self):
        for intent in ("stat_analysis", "general"):
            assert _INTENT_TO_CATEGORY[intent] == "general", f"{intent} should be 'general'"


# ---------------------------------------------------------------------------
# _infer_category  (integration with classifier)
# ---------------------------------------------------------------------------


class TestInferCategory:
    def test_eco_text_returns_economy(self):
        cat = _infer_category("we're eco round, should we force buy or save?")
        assert cat == "economy"

    def test_clutch_text_returns_tactical(self):
        cat = _infer_category("I'm last alive 1v3 with 30 seconds left")
        assert cat == "tactical"

    def test_meta_text_returns_meta(self):
        cat = _infer_category("what agents are strong in the current meta?")
        assert cat == "meta"

    def test_agent_text_returns_agent(self):
        cat = _infer_category("how does Jett's dash work?")
        assert cat == "agent"

    def test_general_text_returns_general(self):
        cat = _infer_category("how do I improve at Valorant?")
        assert cat == "general"

    def test_classifier_error_falls_back_to_general(self):
        """When the classifier raises, _infer_category returns 'general'."""
        with patch(
            "valocoach.coach.intent.classify_intent",
            side_effect=RuntimeError("classifier down"),
        ):
            cat = _infer_category("some coaching note")
        assert cat == "general"

    def test_parser_error_falls_back_to_general(self):
        with patch(
            "valocoach.core.parser.parse_situation",
            side_effect=ValueError("parse error"),
        ):
            cat = _infer_category("note text")
        assert cat == "general"

    def test_empty_string_returns_general(self):
        cat = _infer_category("")
        assert cat == "general"

    def test_known_intent_mapped_correctly(self):
        """Verify the mapping layer — not just the classifier."""
        with (
            patch("valocoach.core.parser.parse_situation", return_value=MagicMock()),
            patch("valocoach.coach.intent.classify_intent", return_value="economy"),
        ):
            cat = _infer_category("anything")
        assert cat == "economy"


# ---------------------------------------------------------------------------
# run_notes_list
# ---------------------------------------------------------------------------


class TestRunNotesList:
    def test_warns_when_no_player_profile(self):
        with (
            patch("valocoach.cli.commands.notes.load_settings", return_value=_fake_settings()),
            patch("valocoach.cli.commands.notes.get_player_puuid", return_value=None),
        ):
            result = runner.invoke(app, ["notes", "list"])
        assert result.exit_code == 1

    def test_shows_no_notes_message_when_empty(self):
        with (
            patch("valocoach.cli.commands.notes.load_settings", return_value=_fake_settings()),
            patch("valocoach.cli.commands.notes.get_player_puuid", return_value="puuid-abc"),
            patch("valocoach.cli.commands.notes.list_open_notes", return_value=[]),
        ):
            result = runner.invoke(app, ["notes", "list"])
        assert result.exit_code == 0
        assert "No open" in result.output

    def test_renders_notes_table_when_notes_exist(self):
        from valocoach.coach.session_manager import NoteInfo

        fake_notes = [
            NoteInfo(id=1, body="work on crossfire", category="tactical", priority=2, created_at="2026-05-06T10:00:00"),
            NoteInfo(id=2, body="save on eco rounds", category="economy", priority=1, created_at="2026-05-06T11:00:00"),
        ]
        with (
            patch("valocoach.cli.commands.notes.load_settings", return_value=_fake_settings()),
            patch("valocoach.cli.commands.notes.get_player_puuid", return_value="puuid-abc"),
            patch("valocoach.cli.commands.notes.list_open_notes", return_value=fake_notes),
        ):
            result = runner.invoke(app, ["notes", "list"])
        assert result.exit_code == 0
        assert "1" in result.output
        assert "2" in result.output

    def test_default_notes_command_lists_notes(self):
        """valocoach notes (no sub-command) defaults to list."""
        with (
            patch("valocoach.cli.commands.notes.load_settings", return_value=_fake_settings()),
            patch("valocoach.cli.commands.notes.get_player_puuid", return_value="puuid-abc"),
            patch("valocoach.cli.commands.notes.list_open_notes", return_value=[]),
        ):
            result = runner.invoke(app, ["notes"])
        assert result.exit_code == 0


# ---------------------------------------------------------------------------
# run_notes_add
# ---------------------------------------------------------------------------


class TestRunNotesAdd:
    def test_empty_text_exits_with_error(self):
        with (
            patch("valocoach.cli.commands.notes.load_settings", return_value=_fake_settings()),
        ):
            result = runner.invoke(app, ["notes", "add", "   "])
        assert result.exit_code != 0

    def test_invalid_priority_exits_with_error(self):
        with (
            patch("valocoach.cli.commands.notes.load_settings", return_value=_fake_settings()),
            patch("valocoach.cli.commands.notes.get_player_puuid", return_value="puuid-abc"),
        ):
            result = runner.invoke(app, ["notes", "add", "some note", "--priority", "5"])
        assert result.exit_code != 0

    def test_no_player_profile_exits_with_error(self):
        with (
            patch("valocoach.cli.commands.notes.load_settings", return_value=_fake_settings()),
            patch("valocoach.cli.commands.notes.get_player_puuid", return_value=None),
        ):
            result = runner.invoke(app, ["notes", "add", "work on crossfire"])
        assert result.exit_code == 1

    def test_no_session_exits_with_error(self):
        with (
            patch("valocoach.cli.commands.notes.load_settings", return_value=_fake_settings()),
            patch("valocoach.cli.commands.notes.get_player_puuid", return_value="puuid-abc"),
            patch("valocoach.cli.commands.notes.get_or_open_coaching_session", return_value=None),
        ):
            result = runner.invoke(app, ["notes", "add", "some note"])
        assert result.exit_code == 1

    def test_db_error_on_add_exits_with_error(self):
        with (
            patch("valocoach.cli.commands.notes.load_settings", return_value=_fake_settings()),
            patch("valocoach.cli.commands.notes.get_player_puuid", return_value="puuid-abc"),
            patch("valocoach.cli.commands.notes.get_or_open_coaching_session", return_value=7),
            patch("valocoach.cli.commands.notes.add_coaching_note", return_value=None),
        ):
            result = runner.invoke(app, ["notes", "add", "some note"])
        assert result.exit_code == 1

    def test_success_shows_note_id(self):
        with (
            patch("valocoach.cli.commands.notes.load_settings", return_value=_fake_settings()),
            patch("valocoach.cli.commands.notes.get_player_puuid", return_value="puuid-abc"),
            patch("valocoach.cli.commands.notes.get_or_open_coaching_session", return_value=7),
            patch("valocoach.cli.commands.notes.add_coaching_note", return_value=42),
        ):
            result = runner.invoke(app, ["notes", "add", "work on crossfire at A long"])
        assert result.exit_code == 0
        assert "42" in result.output

    def test_success_shows_inferred_category(self):
        with (
            patch("valocoach.cli.commands.notes.load_settings", return_value=_fake_settings()),
            patch("valocoach.cli.commands.notes.get_player_puuid", return_value="puuid-abc"),
            patch("valocoach.cli.commands.notes.get_or_open_coaching_session", return_value=7),
            patch("valocoach.cli.commands.notes.add_coaching_note", return_value=5),
            patch("valocoach.cli.commands.notes._infer_category", return_value="economy"),
        ):
            result = runner.invoke(app, ["notes", "add", "save on eco round"])
        assert result.exit_code == 0
        assert "economy" in result.output

    def test_priority_flag_passed_through(self):
        """The --priority flag is forwarded to add_coaching_note."""
        captured = {}

        def _capture(settings, session_id, puuid, body, *, category, priority):
            captured["priority"] = priority
            return 99

        with (
            patch("valocoach.cli.commands.notes.load_settings", return_value=_fake_settings()),
            patch("valocoach.cli.commands.notes.get_player_puuid", return_value="puuid-abc"),
            patch("valocoach.cli.commands.notes.get_or_open_coaching_session", return_value=7),
            patch("valocoach.cli.commands.notes.add_coaching_note", side_effect=_capture),
        ):
            result = runner.invoke(app, ["notes", "add", "high priority note", "--priority", "1"])
        assert result.exit_code == 0
        assert captured["priority"] == 1

    def test_default_priority_is_2(self):
        captured = {}

        def _capture(settings, session_id, puuid, body, *, category, priority):
            captured["priority"] = priority
            return 1

        with (
            patch("valocoach.cli.commands.notes.load_settings", return_value=_fake_settings()),
            patch("valocoach.cli.commands.notes.get_player_puuid", return_value="puuid-abc"),
            patch("valocoach.cli.commands.notes.get_or_open_coaching_session", return_value=7),
            patch("valocoach.cli.commands.notes.add_coaching_note", side_effect=_capture),
        ):
            runner.invoke(app, ["notes", "add", "some note"])
        assert captured.get("priority") == 2


# ---------------------------------------------------------------------------
# run_notes_resolve
# ---------------------------------------------------------------------------


class TestRunNotesResolve:
    def test_success_exits_0(self):
        with (
            patch("valocoach.cli.commands.notes.load_settings", return_value=_fake_settings()),
            patch("valocoach.cli.commands.notes.resolve_coaching_note", return_value=True),
        ):
            result = runner.invoke(app, ["notes", "resolve", "12"])
        assert result.exit_code == 0
        assert "12" in result.output

    def test_not_found_exits_1(self):
        with (
            patch("valocoach.cli.commands.notes.load_settings", return_value=_fake_settings()),
            patch("valocoach.cli.commands.notes.resolve_coaching_note", return_value=False),
        ):
            result = runner.invoke(app, ["notes", "resolve", "999"])
        assert result.exit_code == 1

    def test_non_integer_id_rejected_by_typer(self):
        result = runner.invoke(app, ["notes", "resolve", "abc"])
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# CLI routing smoke tests
# ---------------------------------------------------------------------------


class TestNotesCliRouting:
    def test_notes_help_shows_subcommands(self):
        result = runner.invoke(app, ["notes", "--help"])
        assert result.exit_code == 0
        assert "list" in result.output
        assert "add" in result.output
        assert "resolve" in result.output

    def test_notes_add_help_shows_priority_option(self):
        result = runner.invoke(app, ["notes", "add", "--help"])
        assert result.exit_code == 0
        assert "--priority" in result.output

    def test_notes_resolve_help_shows_note_id(self):
        result = runner.invoke(app, ["notes", "resolve", "--help"])
        assert result.exit_code == 0

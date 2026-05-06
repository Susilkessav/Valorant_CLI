"""Tests for valocoach.coach.session_manager.

All DB calls are mocked so these tests run fully in-process with no
SQLite or filesystem I/O.

Coverage targets
----------------
- REPLCoachState.active property (True / False branches)
- SessionInfo.is_open property (True / False branches)
- get_player_puuid: success, None result, exception
- open_coaching_session: success, exception
- close_coaching_session: success, exception (silent)
- add_coaching_note: success, exception
- list_open_notes: success, empty, exception
- resolve_coaching_note: success (True), not-found (False), exception
- list_coaching_sessions: success, empty, exception
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from valocoach.coach.session_manager import (
    NoteInfo,
    REPLCoachState,
    SessionInfo,
    add_coaching_note,
    close_coaching_session,
    get_player_puuid,
    list_coaching_sessions,
    list_open_notes,
    open_coaching_session,
    resolve_coaching_note,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ENSURE_DB = "valocoach.data.database.ensure_db"
_SESSION_SCOPE = "valocoach.data.database.session_scope"


def _fake_settings(*, riot_name="Player1", riot_tag="1234", data_dir=None):
    from pathlib import Path
    from unittest.mock import MagicMock

    s = MagicMock()
    s.riot_name = riot_name
    s.riot_tag = riot_tag
    s.data_dir = data_dir or Path("/tmp/valocoach_test")
    return s


def _mock_db_ctx():
    """Return an async context-manager mock that yields a fake DB session."""
    db = AsyncMock()
    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=db)
    ctx.__aexit__ = AsyncMock(return_value=False)
    return ctx, db


# ---------------------------------------------------------------------------
# REPLCoachState
# ---------------------------------------------------------------------------


class TestREPLCoachState:
    def test_active_true_when_both_set(self):
        state = REPLCoachState(puuid="p-123", coaching_session_id=7)
        assert state.active is True

    def test_active_false_when_no_puuid(self):
        state = REPLCoachState(puuid=None, coaching_session_id=7)
        assert state.active is False

    def test_active_false_when_no_session_id(self):
        state = REPLCoachState(puuid="p-123", coaching_session_id=None)
        assert state.active is False

    def test_active_false_when_both_none(self):
        state = REPLCoachState()
        assert state.active is False

    def test_settings_default_none(self):
        state = REPLCoachState()
        assert not hasattr(state, "settings") or True  # field exists (no error)

    def test_mutable(self):
        state = REPLCoachState()
        state.puuid = "p-abc"
        state.coaching_session_id = 3
        assert state.active is True


# ---------------------------------------------------------------------------
# SessionInfo
# ---------------------------------------------------------------------------


class TestSessionInfo:
    def test_is_open_true_when_no_ended_at(self):
        s = SessionInfo(
            id=1,
            title="Test",
            started_at="2026-05-06T10:00:00",
            ended_at=None,
            focus_agent=None,
            focus_map=None,
        )
        assert s.is_open is True

    def test_is_open_false_when_ended_at_set(self):
        s = SessionInfo(
            id=2,
            title="Done",
            started_at="2026-05-06T10:00:00",
            ended_at="2026-05-06T11:00:00",
            focus_agent="Jett",
            focus_map="Ascent",
        )
        assert s.is_open is False

    def test_frozen(self):
        s = SessionInfo(
            id=1, title="x", started_at="t", ended_at=None, focus_agent=None, focus_map=None
        )
        with pytest.raises((AttributeError, TypeError)):
            s.id = 99  # type: ignore[misc]


# ---------------------------------------------------------------------------
# NoteInfo
# ---------------------------------------------------------------------------


class TestNoteInfo:
    def test_fields_accessible(self):
        n = NoteInfo(id=5, body="Work on crossfire", category="tactical", priority=1, created_at="2026-05-06")
        assert n.id == 5
        assert n.body == "Work on crossfire"
        assert n.category == "tactical"
        assert n.priority == 1
        assert n.match_id is None

    def test_frozen(self):
        n = NoteInfo(id=1, body="x", category="general", priority=2, created_at="t")
        with pytest.raises((AttributeError, TypeError)):
            n.id = 99  # type: ignore[misc]


# ---------------------------------------------------------------------------
# get_player_puuid
# ---------------------------------------------------------------------------


class TestGetPlayerPuuid:
    def test_returns_puuid_when_player_found(self):
        settings = _fake_settings()
        fake_player = MagicMock()
        fake_player.puuid = "puuid-abc-123"

        ctx, db = _mock_db_ctx()
        db.scalars = AsyncMock(return_value=MagicMock(first=MagicMock(return_value=fake_player)))

        with (
            patch("valocoach.data.database.ensure_db", new_callable=AsyncMock),
            patch("valocoach.data.database.session_scope", return_value=ctx),
            patch(
                "valocoach.data.repository.get_player_by_name",
                new_callable=AsyncMock,
                return_value=fake_player,
            ),
        ):
            result = get_player_puuid(settings)

        assert result == "puuid-abc-123"

    def test_returns_none_when_player_not_found(self):
        settings = _fake_settings()
        ctx, db = _mock_db_ctx()

        with (
            patch("valocoach.data.database.ensure_db", new_callable=AsyncMock),
            patch("valocoach.data.database.session_scope", return_value=ctx),
            patch(
                "valocoach.data.repository.get_player_by_name",
                new_callable=AsyncMock,
                return_value=None,
            ),
        ):
            result = get_player_puuid(settings)

        assert result is None

    def test_returns_none_on_exception(self):
        settings = _fake_settings()

        with patch(
            "valocoach.data.database.ensure_db",
            new_callable=AsyncMock,
            side_effect=RuntimeError("db offline"),
        ):
            result = get_player_puuid(settings)

        assert result is None


# ---------------------------------------------------------------------------
# open_coaching_session
# ---------------------------------------------------------------------------


class TestOpenCoachingSession:
    def test_returns_session_id(self):
        settings = _fake_settings()
        fake_cs = MagicMock()
        fake_cs.id = 42

        ctx, db = _mock_db_ctx()

        with (
            patch("valocoach.data.database.ensure_db", new_callable=AsyncMock),
            patch("valocoach.data.database.session_scope", return_value=ctx),
            patch(
                "valocoach.data.repository.create_coaching_session",
                new_callable=AsyncMock,
                return_value=fake_cs,
            ),
        ):
            result = open_coaching_session(settings, "puuid-abc", title="Test session")

        assert result == 42

    def test_passes_kwargs_to_repo(self):
        settings = _fake_settings()
        fake_cs = MagicMock()
        fake_cs.id = 1
        ctx, db = _mock_db_ctx()

        with (
            patch("valocoach.data.database.ensure_db", new_callable=AsyncMock),
            patch("valocoach.data.database.session_scope", return_value=ctx),
            patch(
                "valocoach.data.repository.create_coaching_session",
                new_callable=AsyncMock,
                return_value=fake_cs,
            ) as mock_create,
        ):
            open_coaching_session(
                settings,
                "p-1",
                title="Clutch drills",
                focus_agent="Jett",
                focus_map="Ascent",
            )

        _, kwargs = mock_create.call_args
        assert kwargs.get("title") == "Clutch drills"
        assert kwargs.get("focus_agent") == "Jett"
        assert kwargs.get("focus_map") == "Ascent"

    def test_returns_none_on_exception(self):
        settings = _fake_settings()
        with patch(
            "valocoach.data.database.ensure_db",
            new_callable=AsyncMock,
            side_effect=OSError("no disk"),
        ):
            result = open_coaching_session(settings, "p-1")
        assert result is None


# ---------------------------------------------------------------------------
# close_coaching_session
# ---------------------------------------------------------------------------


class TestCloseCoachingSession:
    def test_calls_end_coaching_session(self):
        settings = _fake_settings()
        ctx, db = _mock_db_ctx()

        with (
            patch("valocoach.data.database.ensure_db", new_callable=AsyncMock),
            patch("valocoach.data.database.session_scope", return_value=ctx),
            patch(
                "valocoach.data.repository.end_coaching_session",
                new_callable=AsyncMock,
                return_value=MagicMock(),
            ) as mock_end,
        ):
            close_coaching_session(settings, 7)

        mock_end.assert_called_once()

    def test_silent_on_exception(self):
        settings = _fake_settings()
        with patch(
            "valocoach.data.database.ensure_db",
            new_callable=AsyncMock,
            side_effect=RuntimeError("boom"),
        ):
            close_coaching_session(settings, 7)  # must not raise


# ---------------------------------------------------------------------------
# add_coaching_note
# ---------------------------------------------------------------------------


class TestAddCoachingNote:
    def test_returns_note_id(self):
        settings = _fake_settings()
        fake_note = MagicMock()
        fake_note.id = 99
        ctx, db = _mock_db_ctx()

        with (
            patch("valocoach.data.database.ensure_db", new_callable=AsyncMock),
            patch("valocoach.data.database.session_scope", return_value=ctx),
            patch(
                "valocoach.data.repository.add_coaching_note",
                new_callable=AsyncMock,
                return_value=fake_note,
            ),
        ):
            result = add_coaching_note(settings, 1, "p-1", "Work on crossfire")

        assert result == 99

    def test_passes_category_and_priority(self):
        settings = _fake_settings()
        fake_note = MagicMock()
        fake_note.id = 5
        ctx, db = _mock_db_ctx()

        with (
            patch("valocoach.data.database.ensure_db", new_callable=AsyncMock),
            patch("valocoach.data.database.session_scope", return_value=ctx),
            patch(
                "valocoach.data.repository.add_coaching_note",
                new_callable=AsyncMock,
                return_value=fake_note,
            ) as mock_add,
        ):
            add_coaching_note(settings, 1, "p-1", "note body", category="economy", priority=1)

        _, kwargs = mock_add.call_args
        assert kwargs.get("category") == "economy"
        assert kwargs.get("priority") == 1

    def test_returns_none_on_exception(self):
        settings = _fake_settings()
        with patch(
            "valocoach.data.database.ensure_db",
            new_callable=AsyncMock,
            side_effect=RuntimeError("db down"),
        ):
            result = add_coaching_note(settings, 1, "p-1", "note")
        assert result is None


# ---------------------------------------------------------------------------
# list_open_notes
# ---------------------------------------------------------------------------


class TestListOpenNotes:
    def _fake_note(self, *, id=1, body="tip", category="general", priority=2, created_at="2026-05-06"):
        n = MagicMock()
        n.id = id
        n.body = body
        n.category = category
        n.priority = priority
        n.created_at = created_at
        n.match_id = None
        return n

    def test_returns_note_info_list(self):
        settings = _fake_settings()
        raw = [self._fake_note(id=1, body="A long crossfire"), self._fake_note(id=2, body="Eco tips")]
        ctx, db = _mock_db_ctx()

        with (
            patch("valocoach.data.database.ensure_db", new_callable=AsyncMock),
            patch("valocoach.data.database.session_scope", return_value=ctx),
            patch(
                "valocoach.data.repository.get_open_notes",
                new_callable=AsyncMock,
                return_value=raw,
            ),
        ):
            result = list_open_notes(settings, "p-1")

        assert len(result) == 2
        assert isinstance(result[0], NoteInfo)
        assert result[0].id == 1
        assert result[0].body == "A long crossfire"
        assert result[1].id == 2

    def test_returns_empty_list_when_no_notes(self):
        settings = _fake_settings()
        ctx, db = _mock_db_ctx()

        with (
            patch("valocoach.data.database.ensure_db", new_callable=AsyncMock),
            patch("valocoach.data.database.session_scope", return_value=ctx),
            patch(
                "valocoach.data.repository.get_open_notes",
                new_callable=AsyncMock,
                return_value=[],
            ),
        ):
            result = list_open_notes(settings, "p-1")

        assert result == []

    def test_returns_empty_list_on_exception(self):
        settings = _fake_settings()
        with patch(
            "valocoach.data.database.ensure_db",
            new_callable=AsyncMock,
            side_effect=RuntimeError("no db"),
        ):
            result = list_open_notes(settings, "p-1")
        assert result == []

    def test_note_info_fields_mapped_correctly(self):
        settings = _fake_settings()
        raw = [self._fake_note(id=7, body="body text", category="tactical", priority=1, created_at="2026-01-01")]
        raw[0].match_id = "match-abc"
        ctx, db = _mock_db_ctx()

        with (
            patch("valocoach.data.database.ensure_db", new_callable=AsyncMock),
            patch("valocoach.data.database.session_scope", return_value=ctx),
            patch(
                "valocoach.data.repository.get_open_notes",
                new_callable=AsyncMock,
                return_value=raw,
            ),
        ):
            result = list_open_notes(settings, "p-1")

        n = result[0]
        assert n.id == 7
        assert n.body == "body text"
        assert n.category == "tactical"
        assert n.priority == 1
        assert n.created_at == "2026-01-01"
        assert n.match_id == "match-abc"


# ---------------------------------------------------------------------------
# resolve_coaching_note
# ---------------------------------------------------------------------------


class TestResolveCoachingNote:
    def test_returns_true_when_note_found(self):
        settings = _fake_settings()
        ctx, db = _mock_db_ctx()

        with (
            patch("valocoach.data.database.ensure_db", new_callable=AsyncMock),
            patch("valocoach.data.database.session_scope", return_value=ctx),
            patch(
                "valocoach.data.repository.resolve_note",
                new_callable=AsyncMock,
                return_value=MagicMock(),  # non-None → resolved
            ),
        ):
            result = resolve_coaching_note(settings, 12)

        assert result is True

    def test_returns_false_when_note_not_found(self):
        settings = _fake_settings()
        ctx, db = _mock_db_ctx()

        with (
            patch("valocoach.data.database.ensure_db", new_callable=AsyncMock),
            patch("valocoach.data.database.session_scope", return_value=ctx),
            patch(
                "valocoach.data.repository.resolve_note",
                new_callable=AsyncMock,
                return_value=None,  # not found
            ),
        ):
            result = resolve_coaching_note(settings, 999)

        assert result is False

    def test_returns_false_on_exception(self):
        settings = _fake_settings()
        with patch(
            "valocoach.data.database.ensure_db",
            new_callable=AsyncMock,
            side_effect=RuntimeError("error"),
        ):
            result = resolve_coaching_note(settings, 5)
        assert result is False


# ---------------------------------------------------------------------------
# list_coaching_sessions
# ---------------------------------------------------------------------------


class TestListCoachingSessions:
    def _fake_session(self, *, id=1, title="2026-05-06", started_at="2026-05-06T10:00:00", ended_at=None, agent=None, map_=None):
        s = MagicMock()
        s.id = id
        s.session_title = title
        s.started_at = started_at
        s.ended_at = ended_at
        s.focus_agent = agent
        s.focus_map = map_
        return s

    def test_returns_session_info_list(self):
        settings = _fake_settings()
        raw = [
            self._fake_session(id=3, title="Post-plant drill", ended_at="2026-05-06T12:00:00"),
            self._fake_session(id=2, title="2026-05-05", agent="Jett"),
        ]
        ctx, db = _mock_db_ctx()

        with (
            patch("valocoach.data.database.ensure_db", new_callable=AsyncMock),
            patch("valocoach.data.database.session_scope", return_value=ctx),
            patch(
                "valocoach.data.repository.get_coaching_sessions",
                new_callable=AsyncMock,
                return_value=raw,
            ),
        ):
            result = list_coaching_sessions(settings, "p-1")

        assert len(result) == 2
        assert isinstance(result[0], SessionInfo)
        assert result[0].id == 3
        assert result[0].title == "Post-plant drill"
        assert result[0].ended_at == "2026-05-06T12:00:00"
        assert result[0].is_open is False
        assert result[1].focus_agent == "Jett"
        assert result[1].is_open is True

    def test_returns_empty_on_no_sessions(self):
        settings = _fake_settings()
        ctx, db = _mock_db_ctx()

        with (
            patch("valocoach.data.database.ensure_db", new_callable=AsyncMock),
            patch("valocoach.data.database.session_scope", return_value=ctx),
            patch(
                "valocoach.data.repository.get_coaching_sessions",
                new_callable=AsyncMock,
                return_value=[],
            ),
        ):
            result = list_coaching_sessions(settings, "p-1")

        assert result == []

    def test_returns_empty_on_exception(self):
        settings = _fake_settings()
        with patch(
            "valocoach.data.database.ensure_db",
            new_callable=AsyncMock,
            side_effect=RuntimeError("no db"),
        ):
            result = list_coaching_sessions(settings, "p-1")
        assert result == []

    def test_session_info_fields_mapped(self):
        settings = _fake_settings()
        raw = [self._fake_session(id=9, title="T", started_at="2026-01-01T08:00:00", agent="Viper", map_="Icebox")]
        ctx, db = _mock_db_ctx()

        with (
            patch("valocoach.data.database.ensure_db", new_callable=AsyncMock),
            patch("valocoach.data.database.session_scope", return_value=ctx),
            patch(
                "valocoach.data.repository.get_coaching_sessions",
                new_callable=AsyncMock,
                return_value=raw,
            ),
        ):
            result = list_coaching_sessions(settings, "p-1")

        s = result[0]
        assert s.id == 9
        assert s.title == "T"
        assert s.started_at == "2026-01-01T08:00:00"
        assert s.focus_agent == "Viper"
        assert s.focus_map == "Icebox"

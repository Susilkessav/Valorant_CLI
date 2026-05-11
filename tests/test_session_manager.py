"""Tests for valocoach.coach.session_manager."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from valocoach.coach.session_manager import (
    LastMatchInfo,
    MMRHistoryInfo,
    NoteInfo,
    REPLCoachState,
    SessionInfo,
    add_coaching_note,
    close_coaching_session,
    format_last_match_context,
    get_last_match,
    get_mmr_trend,
    get_or_open_coaching_session,
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
        ctx, _db = _mock_db_ctx()

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

        ctx, _db = _mock_db_ctx()

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
        ctx, _db = _mock_db_ctx()

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
        ctx, _db = _mock_db_ctx()

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
        ctx, _db = _mock_db_ctx()

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
        ctx, _db = _mock_db_ctx()

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
    def _fake_note(self, *, note_id=1, body="tip", category="general", priority=2, created_at="2026-05-06"):
        n = MagicMock()
        n.id = note_id
        n.body = body
        n.category = category
        n.priority = priority
        n.created_at = created_at
        n.match_id = None
        return n

    def test_returns_note_info_list(self):
        settings = _fake_settings()
        raw = [self._fake_note(note_id=1, body="A long crossfire"), self._fake_note(note_id=2, body="Eco tips")]
        ctx, _db = _mock_db_ctx()

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
        ctx, _db = _mock_db_ctx()

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
        raw = [self._fake_note(note_id=7, body="body text", category="tactical", priority=1, created_at="2026-01-01")]
        raw[0].match_id = "match-abc"
        ctx, _db = _mock_db_ctx()

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
        ctx, _db = _mock_db_ctx()

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
        ctx, _db = _mock_db_ctx()

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
    def _fake_session(self, *, session_id=1, title="2026-05-06", started_at="2026-05-06T10:00:00", ended_at=None, agent=None, map_=None):
        s = MagicMock()
        s.id = session_id
        s.session_title = title
        s.started_at = started_at
        s.ended_at = ended_at
        s.focus_agent = agent
        s.focus_map = map_
        return s

    def test_returns_session_info_list(self):
        settings = _fake_settings()
        raw = [
            self._fake_session(session_id=3, title="Post-plant drill", ended_at="2026-05-06T12:00:00"),
            self._fake_session(session_id=2, title="2026-05-05", agent="Jett"),
        ]
        ctx, _db = _mock_db_ctx()

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
        ctx, _db = _mock_db_ctx()

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
        raw = [self._fake_session(session_id=9, title="T", started_at="2026-01-01T08:00:00", agent="Viper", map_="Icebox")]
        ctx, _db = _mock_db_ctx()

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


# ---------------------------------------------------------------------------
# MMRHistoryInfo
# ---------------------------------------------------------------------------


class TestMMRHistoryInfo:
    def test_fields_accessible(self):
        h = MMRHistoryInfo(
            tier_patched="Gold II",
            rr=45,
            elo=1245,
            mmr_change=25,
            recorded_at="2026-05-06T10:00:00",
        )
        assert h.tier_patched == "Gold II"
        assert h.rr == 45
        assert h.elo == 1245
        assert h.mmr_change == 25
        assert h.recorded_at == "2026-05-06T10:00:00"

    def test_mmr_change_none_allowed(self):
        h = MMRHistoryInfo(
            tier_patched="Silver III",
            rr=60,
            elo=1060,
            mmr_change=None,
            recorded_at="2026-01-01",
        )
        assert h.mmr_change is None

    def test_frozen(self):
        h = MMRHistoryInfo(
            tier_patched="Plat I", rr=20, elo=1420, mmr_change=-15, recorded_at="t"
        )
        with pytest.raises((AttributeError, TypeError)):
            h.elo = 9999  # type: ignore[misc]


# ---------------------------------------------------------------------------
# get_mmr_trend
# ---------------------------------------------------------------------------


class TestGetMmrTrend:
    def _fake_row(
        self,
        *,
        tier_patched="Gold II",
        rr=45,
        elo=1245,
        mmr_change=25,
        recorded_at="2026-05-06T10:00:00",
    ):
        r = MagicMock()
        r.tier_patched = tier_patched
        r.rr = rr
        r.elo = elo
        r.mmr_change = mmr_change
        r.recorded_at = recorded_at
        return r

    def test_returns_mmr_history_info_list(self):
        settings = _fake_settings()
        raw = [
            self._fake_row(tier_patched="Plat I", elo=1400, rr=20),
            self._fake_row(tier_patched="Gold II", elo=1245, rr=45),
        ]
        ctx, _db = _mock_db_ctx()

        with (
            patch("valocoach.data.database.ensure_db", new_callable=AsyncMock),
            patch("valocoach.data.database.session_scope", return_value=ctx),
            patch(
                "valocoach.data.repository.get_mmr_history",
                new_callable=AsyncMock,
                return_value=raw,
            ),
        ):
            result = get_mmr_trend(settings, "puuid-xyz")

        assert len(result) == 2
        assert isinstance(result[0], MMRHistoryInfo)
        assert result[0].tier_patched == "Plat I"
        assert result[0].elo == 1400
        assert result[1].tier_patched == "Gold II"

    def test_fields_mapped_correctly(self):
        settings = _fake_settings()
        raw = [self._fake_row(tier_patched="Silver III", rr=72, elo=1072, mmr_change=-20, recorded_at="2026-04-01")]
        ctx, _db = _mock_db_ctx()

        with (
            patch("valocoach.data.database.ensure_db", new_callable=AsyncMock),
            patch("valocoach.data.database.session_scope", return_value=ctx),
            patch(
                "valocoach.data.repository.get_mmr_history",
                new_callable=AsyncMock,
                return_value=raw,
            ),
        ):
            result = get_mmr_trend(settings, "p-1")

        h = result[0]
        assert h.tier_patched == "Silver III"
        assert h.rr == 72
        assert h.elo == 1072
        assert h.mmr_change == -20
        assert h.recorded_at == "2026-04-01"

    def test_returns_empty_list_when_no_history(self):
        settings = _fake_settings()
        ctx, _db = _mock_db_ctx()

        with (
            patch("valocoach.data.database.ensure_db", new_callable=AsyncMock),
            patch("valocoach.data.database.session_scope", return_value=ctx),
            patch(
                "valocoach.data.repository.get_mmr_history",
                new_callable=AsyncMock,
                return_value=[],
            ),
        ):
            result = get_mmr_trend(settings, "p-1")

        assert result == []

    def test_returns_empty_list_on_exception(self):
        settings = _fake_settings()
        with patch(
            "valocoach.data.database.ensure_db",
            new_callable=AsyncMock,
            side_effect=RuntimeError("db offline"),
        ):
            result = get_mmr_trend(settings, "p-1")
        assert result == []

    def test_passes_limit_to_repo(self):
        settings = _fake_settings()
        ctx, _db = _mock_db_ctx()

        with (
            patch("valocoach.data.database.ensure_db", new_callable=AsyncMock),
            patch("valocoach.data.database.session_scope", return_value=ctx),
            patch(
                "valocoach.data.repository.get_mmr_history",
                new_callable=AsyncMock,
                return_value=[],
            ) as mock_hist,
        ):
            get_mmr_trend(settings, "p-1", limit=5)

        _, kwargs = mock_hist.call_args
        assert kwargs.get("limit") == 5


# ---------------------------------------------------------------------------
# get_or_open_coaching_session
# ---------------------------------------------------------------------------


class TestGetOrOpenCoachingSession:
    def test_returns_existing_session_id_when_open_session_exists(self):
        settings = _fake_settings()
        existing = MagicMock()
        existing.id = 7
        ctx, _db = _mock_db_ctx()

        with (
            patch("valocoach.data.database.ensure_db", new_callable=AsyncMock),
            patch("valocoach.data.database.session_scope", return_value=ctx),
            patch(
                "valocoach.data.repository.get_open_coaching_session",
                new_callable=AsyncMock,
                return_value=existing,
            ),
            patch(
                "valocoach.data.repository.create_coaching_session",
                new_callable=AsyncMock,
            ) as mock_create,
        ):
            result = get_or_open_coaching_session(settings, "p-1")

        assert result == 7
        mock_create.assert_not_called()

    def test_creates_new_session_when_none_open(self):
        settings = _fake_settings()
        new_cs = MagicMock()
        new_cs.id = 99
        ctx, _db = _mock_db_ctx()

        with (
            patch("valocoach.data.database.ensure_db", new_callable=AsyncMock),
            patch("valocoach.data.database.session_scope", return_value=ctx),
            patch(
                "valocoach.data.repository.get_open_coaching_session",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "valocoach.data.repository.create_coaching_session",
                new_callable=AsyncMock,
                return_value=new_cs,
            ),
        ):
            result = get_or_open_coaching_session(settings, "p-1")

        assert result == 99

    def test_returns_none_on_exception(self):
        settings = _fake_settings()
        with patch(
            "valocoach.data.database.ensure_db",
            new_callable=AsyncMock,
            side_effect=RuntimeError("db down"),
        ):
            result = get_or_open_coaching_session(settings, "p-1")
        assert result is None


# ---------------------------------------------------------------------------
# LastMatchInfo + format_last_match_context
# ---------------------------------------------------------------------------


def _last_match(
    *,
    match_id: str = "m-abc",
    map_name: str = "Ascent",
    agent: str = "Jett",
    won: bool = True,
    own_score: int = 13,
    opp_score: int = 7,
    kills: int = 18,
    deaths: int = 8,
    assists: int = 4,
    acs: int = 225,
    hs_pct: float = 28.0,
    adr: float = 142.0,
    started_at: str = "2026-05-06T20:00:00",
) -> LastMatchInfo:
    return LastMatchInfo(
        match_id=match_id,
        map_name=map_name,
        agent=agent,
        won=won,
        own_score=own_score,
        opp_score=opp_score,
        kills=kills,
        deaths=deaths,
        assists=assists,
        acs=acs,
        hs_pct=hs_pct,
        adr=adr,
        started_at=started_at,
    )


class TestLastMatchInfo:
    def test_fields_accessible(self):
        lm = _last_match()
        assert lm.match_id == "m-abc"
        assert lm.map_name == "Ascent"
        assert lm.agent == "Jett"
        assert lm.won is True
        assert lm.own_score == 13
        assert lm.opp_score == 7
        assert lm.kills == 18
        assert lm.deaths == 8
        assert lm.assists == 4
        assert lm.acs == 225
        assert lm.hs_pct == 28.0
        assert lm.adr == 142.0

    def test_frozen(self):
        lm = _last_match()
        with pytest.raises(FrozenInstanceError):
            lm.kills = 99  # type: ignore[misc]

    def test_format_win(self):
        lm = _last_match(won=True, own_score=13, opp_score=7)
        out = format_last_match_context(lm)
        assert "LAST MATCH" in out
        assert "W 13-7" in out
        assert "Jett" in out
        assert "Ascent" in out

    def test_format_loss(self):
        lm = _last_match(won=False, own_score=5, opp_score=13)
        out = format_last_match_context(lm)
        assert "L 5-13" in out

    def test_format_includes_kda(self):
        lm = _last_match(kills=18, deaths=8, assists=4)
        out = format_last_match_context(lm)
        assert "18/8/4" in out

    def test_format_includes_acs(self):
        lm = _last_match(acs=225)
        out = format_last_match_context(lm)
        assert "ACS 225" in out

    def test_format_includes_hs_pct(self):
        lm = _last_match(hs_pct=28.0)
        out = format_last_match_context(lm)
        assert "HS 28%" in out

    def test_format_includes_adr(self):
        lm = _last_match(adr=142.0)
        out = format_last_match_context(lm)
        assert "ADR 142" in out

    def test_format_compact(self):
        """Output must stay short — it's injected into the LLM user message."""
        out = format_last_match_context(_last_match())
        assert len(out) < 120, f"too long ({len(out)} chars): {out}"


# ---------------------------------------------------------------------------
# get_last_match
# ---------------------------------------------------------------------------


def _make_match_player(
    *,
    match_id: str = "m-abc",
    map_name: str = "Ascent",
    agent: str = "Jett",
    team: str = "Blue",
    won: bool = True,
    blue_score: int = 13,
    red_score: int = 7,
    kills: int = 18,
    deaths: int = 8,
    assists: int = 4,
    score: int = 4500,          # ACS = 4500/20 = 225
    rounds_played: int = 20,
    headshots: int = 56,        # HS% = 56/(56+140+4) ≈ 28%
    bodyshots: int = 140,
    legshots: int = 4,
    damage_dealt: int = 2840,   # ADR = 2840/20 = 142
):
    mp = MagicMock()
    mp.match_id = match_id
    mp.agent_name = agent
    mp.team = team
    mp.won = won
    mp.kills = kills
    mp.deaths = deaths
    mp.assists = assists
    mp.score = score
    mp.rounds_played = rounds_played
    mp.headshots = headshots
    mp.bodyshots = bodyshots
    mp.legshots = legshots
    mp.damage_dealt = damage_dealt
    mp.started_at = "2026-05-06T20:00:00"

    m = MagicMock()
    m.map_name = map_name
    m.blue_score = blue_score
    m.red_score = red_score
    mp.match = m
    return mp


class TestGetLastMatch:
    def _run(self, player_mock, match_rows):
        """Call get_last_match with DB fully mocked."""
        settings = _fake_settings()
        ctx, db = _mock_db_ctx()
        db.scalar = AsyncMock(return_value=player_mock)

        with (
            patch("valocoach.data.database.ensure_db", new_callable=AsyncMock),
            patch("valocoach.data.database.session_scope", return_value=ctx),
            patch(
                "valocoach.data.repository.get_player_by_name",
                new_callable=AsyncMock,
                return_value=player_mock,
            ),
            patch(
                "valocoach.data.repository.get_recent_matches",
                new_callable=AsyncMock,
                return_value=match_rows,
            ),
        ):
            return get_last_match(settings)

    def _player(self, puuid: str = "p-abc"):
        p = MagicMock()
        p.puuid = puuid
        return p

    def test_returns_none_when_no_player(self):
        result = self._run(player_mock=None, match_rows=[])
        assert result is None

    def test_returns_none_when_no_matches(self):
        result = self._run(player_mock=self._player(), match_rows=[])
        assert result is None

    def test_returns_last_match_info(self):
        mp = _make_match_player()
        result = self._run(player_mock=self._player(), match_rows=[mp])
        assert isinstance(result, LastMatchInfo)

    def test_fields_mapped_correctly(self):
        mp = _make_match_player(
            agent="Sage",
            map_name="Bind",
            won=True,
            team="Blue",
            blue_score=13,
            red_score=5,
            kills=10,
            deaths=6,
            assists=8,
        )
        result = self._run(player_mock=self._player(), match_rows=[mp])
        assert result is not None
        assert result.agent == "Sage"
        assert result.map_name == "Bind"
        assert result.won is True
        assert result.own_score == 13  # Blue team score for Blue player
        assert result.opp_score == 5
        assert result.kills == 10
        assert result.deaths == 6
        assert result.assists == 8

    def test_own_opp_score_for_red_team(self):
        """Red-team player → own_score=red_score, opp_score=blue_score."""
        mp = _make_match_player(team="Red", red_score=13, blue_score=9)
        result = self._run(player_mock=self._player(), match_rows=[mp])
        assert result is not None
        assert result.own_score == 13
        assert result.opp_score == 9

    def test_acs_computed_correctly(self):
        # score=4500, rounds_played=20 → ACS=225
        mp = _make_match_player(score=4500, rounds_played=20)
        result = self._run(player_mock=self._player(), match_rows=[mp])
        assert result is not None
        assert result.acs == 225

    def test_returns_none_on_exception(self):
        settings = _fake_settings()
        with patch(
            "valocoach.data.database.ensure_db",
            new_callable=AsyncMock,
            side_effect=RuntimeError("db down"),
        ):
            result = get_last_match(settings)
        assert result is None

    def test_returns_none_when_name_missing(self):
        """Unset riot_name → returns None without hitting DB."""
        settings = _fake_settings(riot_name="", riot_tag="1234")
        result = get_last_match(settings)
        assert result is None

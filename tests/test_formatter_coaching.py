"""Tests for the coaching-specific formatter functions added in Phase B.

Covers:
    render_coaching_sessions — empty list (no output), one session, multiple
        sessions, open vs. closed status, focus agent/map rendering,
        title truncation, missing title shows "—".
    render_open_notes       — empty list (no output), one note, multiple notes,
        priority icons (1/2/3/unknown), category + body truncation.
    _PRIORITY_ICON          — all three defined keys present.
"""

from __future__ import annotations

from io import StringIO

from rich.console import Console

from valocoach.cli.formatter import (
    _PRIORITY_ICON,
    render_coaching_sessions,
    render_open_notes,
    render_rank_trend,
)
from valocoach.coach.session_manager import MMRHistoryInfo, NoteInfo, SessionInfo

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _console() -> tuple[Console, StringIO]:
    """Return a Console that writes to a StringIO buffer for assertion."""
    from valocoach.cli.display import THEME

    buf = StringIO()
    con = Console(file=buf, highlight=False, no_color=True, width=120, theme=THEME)
    return con, buf


def _mmr(
    *,
    tier_patched: str = "Gold II",
    rr: int = 45,
    elo: int = 1245,
    mmr_change: int | None = 25,
    recorded_at: str = "2026-05-06T10:00:00",
) -> MMRHistoryInfo:
    return MMRHistoryInfo(
        tier_patched=tier_patched,
        rr=rr,
        elo=elo,
        mmr_change=mmr_change,
        recorded_at=recorded_at,
    )


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


def _note(
    *,
    note_id: int = 1,
    body: str = "Work on crossfire at A long",
    category: str = "tactical",
    priority: int = 2,
    created_at: str = "2026-05-06T10:00:00",
    match_id: str | None = None,
) -> NoteInfo:
    return NoteInfo(
        id=note_id,
        body=body,
        category=category,
        priority=priority,
        created_at=created_at,
        match_id=match_id,
    )


# ===========================================================================
# render_coaching_sessions
# ===========================================================================


class TestRenderCoachingSessions:
    def test_empty_list_produces_no_output(self):
        con, buf = _console()
        render_coaching_sessions(con, [])
        assert buf.getvalue() == ""

    def test_single_session_shows_id(self):
        con, buf = _console()
        render_coaching_sessions(con, [_session(session_id=7)])
        out = buf.getvalue()
        assert "7" in out

    def test_title_rendered(self):
        con, buf = _console()
        render_coaching_sessions(con, [_session(title="Post-plant drill")])
        assert "Post-plant drill" in buf.getvalue()

    def test_none_title_shows_dash(self):
        con, buf = _console()
        render_coaching_sessions(con, [_session(title=None)])
        assert "—" in buf.getvalue()

    def test_open_session_shows_open_status(self):
        con, buf = _console()
        render_coaching_sessions(con, [_session(ended_at=None)])
        assert "open" in buf.getvalue()

    def test_closed_session_shows_closed_status(self):
        con, buf = _console()
        render_coaching_sessions(con, [_session(ended_at="2026-05-06T12:00:00")])
        assert "closed" in buf.getvalue()

    def test_focus_agent_rendered(self):
        con, buf = _console()
        render_coaching_sessions(con, [_session(focus_agent="Jett")])
        assert "Jett" in buf.getvalue()

    def test_focus_map_rendered(self):
        con, buf = _console()
        render_coaching_sessions(con, [_session(focus_map="Ascent")])
        assert "Ascent" in buf.getvalue()

    def test_no_focus_shows_dash(self):
        con, buf = _console()
        render_coaching_sessions(con, [_session(focus_agent=None, focus_map=None)])
        assert "—" in buf.getvalue()

    def test_started_date_extracted(self):
        con, buf = _console()
        render_coaching_sessions(con, [_session(started_at="2026-05-06T10:00:00")])
        assert "2026-05-06" in buf.getvalue()

    def test_count_in_table_title(self):
        con, buf = _console()
        sessions = [_session(session_id=i) for i in range(3)]
        render_coaching_sessions(con, sessions)
        assert "3" in buf.getvalue()

    def test_multiple_sessions_all_ids_present(self):
        con, buf = _console()
        sessions = [_session(session_id=1), _session(session_id=2), _session(session_id=3)]
        render_coaching_sessions(con, sessions)
        out = buf.getvalue()
        for i in (1, 2, 3):
            assert str(i) in out

    def test_title_truncated_at_26_chars(self):
        """Titles longer than 26 chars are silently truncated."""
        long_title = "A" * 30
        con, buf = _console()
        render_coaching_sessions(con, [_session(title=long_title)])
        out = buf.getvalue()
        # Truncated to 26 — the remaining 4 'A's must not appear
        assert long_title not in out
        assert "A" * 26 in out

    def test_focus_agent_and_map_joined(self):
        con, buf = _console()
        render_coaching_sessions(con, [_session(focus_agent="Viper", focus_map="Bind")])
        out = buf.getvalue()
        assert "Viper" in out
        assert "Bind" in out

    def test_empty_started_at_shows_dash(self):
        """Handles sessions with empty started_at (defensive)."""
        s = SessionInfo(
            id=1, title="t", started_at="", ended_at=None, focus_agent=None, focus_map=None
        )
        con, buf = _console()
        render_coaching_sessions(con, [s])
        assert "—" in buf.getvalue()


# ===========================================================================
# render_open_notes
# ===========================================================================


class TestRenderOpenNotes:
    def test_empty_list_produces_no_output(self):
        con, buf = _console()
        render_open_notes(con, [])
        assert buf.getvalue() == ""

    def test_note_id_rendered(self):
        con, buf = _console()
        render_open_notes(con, [_note(note_id=12)])
        assert "12" in buf.getvalue()

    def test_body_rendered(self):
        con, buf = _console()
        render_open_notes(con, [_note(body="Work on crossfire")])
        assert "Work on crossfire" in buf.getvalue()

    def test_category_rendered(self):
        con, buf = _console()
        render_open_notes(con, [_note(category="economy")])
        assert "economy" in buf.getvalue()

    def test_count_in_table_title(self):
        con, buf = _console()
        notes = [_note(note_id=i) for i in range(4)]
        render_open_notes(con, notes)
        assert "4" in buf.getvalue()

    def test_multiple_notes_all_ids_present(self):
        con, buf = _console()
        notes = [_note(note_id=1), _note(note_id=2), _note(note_id=3)]
        render_open_notes(con, notes)
        out = buf.getvalue()
        for i in (1, 2, 3):
            assert str(i) in out

    def test_long_body_truncated(self):
        """Bodies longer than 72 chars are truncated with an ellipsis."""
        long_body = "B" * 80
        con, buf = _console()
        render_open_notes(con, [_note(body=long_body)])
        out = buf.getvalue()
        assert long_body not in out  # not shown in full
        assert "…" in out

    def test_exact_72_char_body_not_truncated(self):
        """72-char body is exactly at the limit — no ellipsis."""
        body = "C" * 72
        con, buf = _console()
        render_open_notes(con, [_note(body=body)])
        out = buf.getvalue()
        assert "…" not in out

    def test_category_truncated_at_8_chars(self):
        """Category is truncated to 8 chars in the table."""
        long_cat = "verylongcategory"
        con, buf = _console()
        render_open_notes(con, [_note(category=long_cat)])
        out = buf.getvalue()
        # Must show truncated prefix, not full word
        assert long_cat not in out
        assert long_cat[:8] in out

    def test_priority_icon_keys_defined(self):
        """All three priority levels have an icon entry."""
        assert 1 in _PRIORITY_ICON
        assert 2 in _PRIORITY_ICON
        assert 3 in _PRIORITY_ICON


# ===========================================================================
# _PRIORITY_ICON
# ===========================================================================


class TestPriorityIcon:
    def test_three_entries(self):
        assert len(_PRIORITY_ICON) == 3

    def test_all_values_are_strings(self):
        for k, v in _PRIORITY_ICON.items():
            assert isinstance(v, str), f"Expected str for priority {k}, got {type(v)}"

    def test_icons_are_distinct(self):
        values = list(_PRIORITY_ICON.values())
        assert len(values) == len(set(values)), "Priority icons should all be distinct"


# ===========================================================================
# render_rank_trend
# ===========================================================================


class TestRenderRankTrend:
    def test_empty_list_produces_no_output(self):
        con, buf = _console()
        render_rank_trend(con, [])
        assert buf.getvalue() == ""

    def test_single_snapshot_produces_no_output(self):
        """Need at least 2 snapshots for a trend."""
        con, buf = _console()
        render_rank_trend(con, [_mmr()])
        assert buf.getvalue() == ""

    def test_two_snapshots_produces_output(self):
        con, buf = _console()
        render_rank_trend(
            con, [_mmr(elo=1300, tier_patched="Plat I"), _mmr(elo=1200, tier_patched="Gold II")]
        )
        assert buf.getvalue() != ""

    def test_shows_tier_range_when_different(self):
        """oldest → newest tier range is rendered."""
        con, buf = _console()
        render_rank_trend(
            con,
            [
                _mmr(tier_patched="Plat I", elo=1400),  # newest
                _mmr(tier_patched="Gold II", elo=1200),  # oldest
            ],
        )
        out = buf.getvalue()
        assert "Gold II" in out
        assert "Plat I" in out

    def test_shows_same_tier_once_when_no_change(self):
        """When oldest and newest tiers are identical, no arrow between them."""
        con, buf = _console()
        render_rank_trend(
            con,
            [
                _mmr(tier_patched="Gold II", elo=1260),
                _mmr(tier_patched="Gold II", elo=1245),
            ],
        )
        out = buf.getvalue()
        assert "Gold II" in out
        # Should not show "Gold II → Gold II" (i.e. tier name appears once)
        assert out.count("Gold II") == 1

    def test_positive_elo_delta_shown(self):
        con, buf = _console()
        render_rank_trend(
            con,
            [
                _mmr(elo=1300),  # newest
                _mmr(elo=1200),  # oldest
            ],
        )
        out = buf.getvalue()
        assert "+100" in out

    def test_negative_elo_delta_shown(self):
        con, buf = _console()
        render_rank_trend(
            con,
            [
                _mmr(elo=1100),  # newest
                _mmr(elo=1300),  # oldest
            ],
        )
        out = buf.getvalue()
        assert "-200" in out

    def test_snapshot_count_shown(self):
        con, buf = _console()
        history = [_mmr(elo=1200 + i * 10) for i in range(5)]
        render_rank_trend(con, history)
        assert "5" in buf.getvalue()

    def test_sparkline_contains_up_arrow_for_gain(self):
        """Positive mmr_change should produce a ▲ in the sparkline."""
        con, buf = _console()
        render_rank_trend(
            con,
            [
                _mmr(elo=1300, mmr_change=20),  # newest
                _mmr(elo=1200, mmr_change=15),  # oldest
            ],
        )
        assert "▲" in buf.getvalue()

    def test_sparkline_contains_down_arrow_for_loss(self):
        """Negative mmr_change should produce a ▼ in the sparkline."""
        con, buf = _console()
        render_rank_trend(
            con,
            [
                _mmr(elo=1100, mmr_change=-25),  # newest
                _mmr(elo=1200, mmr_change=-15),  # oldest
            ],
        )
        assert "▼" in buf.getvalue()

    def test_sparkline_handles_none_mmr_change(self):
        """None mmr_change should not raise — renders as dot."""
        con, buf = _console()
        render_rank_trend(
            con,
            [
                _mmr(elo=1300, mmr_change=None),
                _mmr(elo=1200, mmr_change=None),
            ],
        )
        out = buf.getvalue()
        assert "·" in out  # dot placeholder

    def test_zero_elo_delta_shown(self):
        """Same ELO in both snapshots shows zero delta."""
        con, buf = _console()
        render_rank_trend(
            con,
            [_mmr(elo=1200), _mmr(elo=1200)],
        )
        out = buf.getvalue()
        assert "0" in out

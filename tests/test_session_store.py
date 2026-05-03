"""Tests for valocoach.core.session_store — REPL session persistence."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from valocoach.core.session_store import (
    MAX_SESSIONS,
    _prune,
    latest_session,
    list_sessions,
    load_session,
    save_session,
    session_summary,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _turns(n_pairs: int = 1) -> list[dict[str, str]]:
    """Build a list of alternating user+assistant turns."""
    turns = []
    for i in range(n_pairs):
        turns.append({"role": "user", "content": f"question {i}"})
        turns.append({"role": "assistant", "content": f"answer {i}"})
    return turns


# ---------------------------------------------------------------------------
# save_session
# ---------------------------------------------------------------------------


class TestSaveSession:
    def test_saves_file_to_sessions_dir(self, tmp_path: Path):
        path = save_session(_turns(1), sessions_dir=tmp_path)
        assert path is not None
        assert path.parent == tmp_path
        assert path.suffix == ".json"

    def test_filename_uses_utc_timestamp_format(self, tmp_path: Path):
        path = save_session(_turns(1), sessions_dir=tmp_path)
        assert path is not None
        # Format: YYYY-MM-DDTHH-MM-SS.json
        name = path.stem
        assert len(name) == 19
        assert name[4] == "-" and name[7] == "-" and name[10] == "T"

    def test_saved_file_contains_turns(self, tmp_path: Path):
        turns = _turns(2)
        path = save_session(turns, sessions_dir=tmp_path)
        assert path is not None
        data = json.loads(path.read_text())
        assert data["turns"] == turns

    def test_saved_file_contains_saved_at(self, tmp_path: Path):
        path = save_session(_turns(1), sessions_dir=tmp_path)
        assert path is not None
        data = json.loads(path.read_text())
        assert "saved_at" in data
        assert "T" in data["saved_at"]  # ISO-8601 format

    def test_returns_none_for_empty_turns(self, tmp_path: Path):
        result = save_session([], sessions_dir=tmp_path)
        assert result is None

    def test_returns_none_for_single_turn(self, tmp_path: Path):
        # One user message only (no assistant reply) — not worth saving.
        result = save_session([{"role": "user", "content": "hi"}], sessions_dir=tmp_path)
        assert result is None

    def test_creates_sessions_dir_if_missing(self, tmp_path: Path):
        nested = tmp_path / "deep" / "sessions"
        path = save_session(_turns(1), sessions_dir=nested)
        assert path is not None
        assert nested.is_dir()

    def test_returns_none_on_permission_error(self, tmp_path: Path):
        # Make the directory read-only so the write fails.
        tmp_path.chmod(0o555)
        result = save_session(_turns(1), sessions_dir=tmp_path / "sub")
        assert result is None
        tmp_path.chmod(0o755)  # restore

    def test_prunes_old_sessions_after_save(self, tmp_path: Path):
        # Create MAX_SESSIONS + 5 sessions; after one more save, total should
        # be capped at MAX_SESSIONS.
        for _ in range(MAX_SESSIONS + 5):
            save_session(_turns(1), sessions_dir=tmp_path)
        files = list(tmp_path.glob("*.json"))
        assert len(files) <= MAX_SESSIONS


# ---------------------------------------------------------------------------
# load_session
# ---------------------------------------------------------------------------


class TestLoadSession:
    def test_round_trip(self, tmp_path: Path):
        turns = _turns(3)
        path = save_session(turns, sessions_dir=tmp_path)
        assert path is not None
        loaded = load_session(path)
        assert loaded == turns

    def test_returns_none_for_missing_file(self, tmp_path: Path):
        result = load_session(tmp_path / "nonexistent.json")
        assert result is None

    def test_returns_none_for_malformed_json(self, tmp_path: Path):
        bad = tmp_path / "bad.json"
        bad.write_text("not valid json")
        assert load_session(bad) is None

    def test_returns_none_when_turns_missing_role(self, tmp_path: Path):
        bad = tmp_path / "bad.json"
        bad.write_text(json.dumps({"turns": [{"content": "hi"}]}))
        assert load_session(bad) is None

    def test_returns_none_when_turns_missing_content(self, tmp_path: Path):
        bad = tmp_path / "bad.json"
        bad.write_text(json.dumps({"turns": [{"role": "user"}]}))
        assert load_session(bad) is None

    def test_returns_empty_list_for_empty_turns_key(self, tmp_path: Path):
        f = tmp_path / "empty.json"
        f.write_text(json.dumps({"saved_at": "2026-01-01T00:00:00+00:00", "turns": []}))
        result = load_session(f)
        assert result == []


# ---------------------------------------------------------------------------
# list_sessions / latest_session
# ---------------------------------------------------------------------------


class TestListSessions:
    def test_returns_empty_list_when_dir_missing(self, tmp_path: Path):
        assert list_sessions(tmp_path / "no_such_dir") == []

    def test_returns_newest_first(self, tmp_path: Path):
        # Create files with distinct timestamps so ordering is deterministic
        # even when the test runs in under a second.
        names = [
            "2026-01-01T00-00-01.json",
            "2026-01-01T00-00-02.json",
            "2026-01-01T00-00-03.json",
        ]
        paths = []
        for name in names:
            p = tmp_path / name
            p.write_text(json.dumps({"saved_at": "2026-01-01T00:00:00+00:00", "turns": _turns(1)}))
            paths.append(p)
        listed = list_sessions(tmp_path)
        assert listed == sorted(paths, reverse=True)

    def test_latest_session_returns_most_recent(self, tmp_path: Path):
        save_session(_turns(1), sessions_dir=tmp_path)
        save_session(_turns(2), sessions_dir=tmp_path)
        latest = latest_session(tmp_path)
        all_sessions = list_sessions(tmp_path)
        assert latest == all_sessions[0]

    def test_latest_session_returns_none_when_empty(self, tmp_path: Path):
        assert latest_session(tmp_path) is None


# ---------------------------------------------------------------------------
# session_summary
# ---------------------------------------------------------------------------


class TestSessionSummary:
    def test_includes_turn_count(self, tmp_path: Path):
        path = save_session(_turns(3), sessions_dir=tmp_path)
        assert path is not None
        summary = session_summary(path)
        assert "6" in summary  # 3 pairs = 6 turns

    def test_includes_relative_time(self, tmp_path: Path):
        path = save_session(_turns(1), sessions_dir=tmp_path)
        assert path is not None
        summary = session_summary(path)
        # Freshly saved should say "just now" or "0 min ago"
        assert "just now" in summary or "min ago" in summary or "ago" in summary

    def test_returns_filename_on_error(self, tmp_path: Path):
        bad = tmp_path / "corrupt.json"
        bad.write_text("not json")
        summary = session_summary(bad)
        assert "corrupt" in summary

    # ------------------------------------------------------------------
    # Relative-time branches (lines 145-152): min ago / h ago / day(s) ago
    # and the inner except that falls back to path.stem.
    # ------------------------------------------------------------------

    def _write_with_saved_at(self, tmp_path: Path, saved_at: str) -> Path:
        """Write a minimal but valid session file with a custom saved_at."""
        p = tmp_path / "session.json"
        p.write_text(
            json.dumps(
                {
                    "saved_at": saved_at,
                    "turns": [
                        {"role": "user", "content": "q"},
                        {"role": "assistant", "content": "a"},
                    ],
                }
            )
        )
        return p

    def test_min_ago_branch(self, tmp_path: Path):
        """70 seconds old → 'min ago'."""
        saved_at = (datetime.now(tz=UTC) - timedelta(seconds=70)).isoformat()
        path = self._write_with_saved_at(tmp_path, saved_at)
        assert "min ago" in session_summary(path)

    def test_h_ago_branch(self, tmp_path: Path):
        """2 hours old → 'h ago'."""
        saved_at = (datetime.now(tz=UTC) - timedelta(hours=2)).isoformat()
        path = self._write_with_saved_at(tmp_path, saved_at)
        assert "h ago" in session_summary(path)

    def test_days_ago_branch(self, tmp_path: Path):
        """3 days old → 'day(s) ago'."""
        saved_at = (datetime.now(tz=UTC) - timedelta(days=3)).isoformat()
        path = self._write_with_saved_at(tmp_path, saved_at)
        assert "day" in session_summary(path)

    def test_invalid_saved_at_falls_back_to_stem(self, tmp_path: Path):
        """An unparseable saved_at triggers the inner except → stem used as fallback."""
        path = self._write_with_saved_at(tmp_path, "not-a-valid-datetime-at-all")
        summary = session_summary(path)
        assert path.stem in summary


# ---------------------------------------------------------------------------
# _prune (internal)
# ---------------------------------------------------------------------------


class TestPrune:
    def test_prune_removes_oldest_files(self, tmp_path: Path):
        # Create MAX_SESSIONS + 3 files manually.
        for i in range(MAX_SESSIONS + 3):
            f = tmp_path / f"2026-01-{i + 1:02d}T00-00-00.json"
            f.write_text("{}")
        _prune(tmp_path)
        remaining = list(tmp_path.glob("*.json"))
        assert len(remaining) == MAX_SESSIONS

    def test_prune_keeps_newest(self, tmp_path: Path):
        names = [f"2026-01-{i + 1:02d}T00-00-00.json" for i in range(MAX_SESSIONS + 3)]
        for name in names:
            (tmp_path / name).write_text("{}")
        _prune(tmp_path)
        remaining = {f.name for f in tmp_path.glob("*.json")}
        # Newest are the ones with the highest sort order (highest date numbers).
        expected_kept = set(sorted(names, reverse=True)[:MAX_SESSIONS])
        assert remaining == expected_kept

    def test_prune_is_noop_when_under_limit(self, tmp_path: Path):
        for i in range(5):
            (tmp_path / f"2026-01-0{i + 1}T00-00-00.json").write_text("{}")
        _prune(tmp_path)
        assert len(list(tmp_path.glob("*.json"))) == 5

    def test_prune_exception_is_swallowed(self, tmp_path: Path):
        """PermissionError on unlink must not propagate — covers lines 171-172."""
        # Create MAX_SESSIONS + 2 files so the pruning loop will attempt unlink.
        for i in range(MAX_SESSIONS + 2):
            f = tmp_path / f"2026-01-{i + 1:02d}T00-00-00.json"
            f.write_text("{}")
        # Make the directory read-only so unlink raises PermissionError.
        tmp_path.chmod(0o555)
        try:
            _prune(tmp_path)  # must not raise
        finally:
            tmp_path.chmod(0o755)  # restore so pytest can clean up tmp_path

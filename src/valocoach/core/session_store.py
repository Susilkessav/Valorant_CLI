"""Persistent REPL session storage.

Serialises ``ConversationMemory`` turns to a JSON file under
``~/.valocoach/sessions/`` so a coaching session can be resumed
after the REPL exits.

File format
-----------
Each session is a single UTF-8 JSON file named by a UTC timestamp::

    ~/.valocoach/sessions/2026-05-03T14-22-07.json

The file contains a JSON object with two keys:

    {
        "saved_at": "<ISO-8601 UTC timestamp>",
        "turns": [
            {"role": "user", "content": "..."},
            {"role": "assistant", "content": "..."},
            ...
        ]
    }

Only sessions with at least one complete user+assistant exchange are saved —
empty or half-finished sessions are discarded silently.

Design notes
------------
- Sessions are stored as separate files (not a single DB) so they are trivially
  inspectable with a text editor and easy to delete individually.
- The ``list_sessions`` helper returns sessions newest-first so callers can
  present "resume most recent session?" without sorting.
- ``MAX_SESSIONS`` caps the number of stored files; the oldest are pruned after
  each save so the directory never grows unbounded.
- Failures (disk full, permission error) are non-raising — the REPL must not
  crash on a persistence error.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

log = logging.getLogger(__name__)

SESSIONS_DIR = Path.home() / ".valocoach" / "sessions"
MAX_SESSIONS = 20  # prune oldest files beyond this count


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def save_session(turns: list[dict[str, str]], sessions_dir: Path = SESSIONS_DIR) -> Path | None:
    """Serialise *turns* to a timestamped JSON file.

    Returns the path of the saved file, or ``None`` if saving failed or
    there were no turns worth keeping (fewer than one complete exchange).

    Args:
        turns:        The list of ``{"role": ..., "content": ...}`` dicts
                      returned by ``ConversationMemory.messages``.
        sessions_dir: Override for tests; defaults to ``~/.valocoach/sessions/``.
    """
    # At least one user + one assistant turn needed.
    if len(turns) < 2:
        return None

    try:
        sessions_dir.mkdir(parents=True, exist_ok=True)
        now = datetime.now(tz=UTC)
        # Replace colons in the time part so the filename is cross-platform safe.
        filename = now.strftime("%Y-%m-%dT%H-%M-%S") + ".json"
        path = sessions_dir / filename

        payload = {
            "saved_at": now.isoformat(),
            "turns": turns,
        }
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

        _prune(sessions_dir)
        return path

    except Exception as exc:
        log.debug("Failed to save session: %s", exc)
        return None


def load_session(path: Path) -> list[dict[str, str]] | None:
    """Load turns from a previously saved session file.

    Returns the list of turn dicts, or ``None`` if the file is missing or
    malformed.
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        turns = data.get("turns", [])
        # Validate structure — every item must have role and content.
        if all(isinstance(t, dict) and "role" in t and "content" in t for t in turns):
            return turns
        return None
    except Exception as exc:
        log.debug("Failed to load session from %s: %s", path, exc)
        return None


def list_sessions(sessions_dir: Path = SESSIONS_DIR) -> list[Path]:
    """Return session files sorted newest-first.

    Returns an empty list when the directory does not exist yet.
    """
    if not sessions_dir.exists():
        return []
    return sorted(sessions_dir.glob("*.json"), reverse=True)


def latest_session(sessions_dir: Path = SESSIONS_DIR) -> Path | None:
    """Return the most recently saved session file, or ``None``."""
    sessions = list_sessions(sessions_dir)
    return sessions[0] if sessions else None


def session_summary(path: Path) -> str:
    """Return a one-line human-readable description of a session file.

    Format: ``"<N> turns · saved <relative-time>"``
    Used in the REPL's resume prompt.
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        turns = data.get("turns", [])
        saved_at = data.get("saved_at", "")

        # Build a short relative timestamp like "2 h ago" or "yesterday".
        try:
            saved = datetime.fromisoformat(saved_at)
            delta = datetime.now(tz=UTC) - saved
            secs = int(delta.total_seconds())
            if secs < 60:
                rel = "just now"
            elif secs < 3600:
                rel = f"{secs // 60} min ago"
            elif secs < 86400:
                rel = f"{secs // 3600} h ago"
            else:
                rel = f"{secs // 86400} day(s) ago"
        except Exception:
            rel = path.stem  # fall back to filename

        n = len(turns)
        return f"{n} turn(s) · saved {rel}"
    except Exception:
        return path.name


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _prune(sessions_dir: Path) -> None:
    """Remove oldest session files beyond ``MAX_SESSIONS``."""
    try:
        files = sorted(sessions_dir.glob("*.json"), reverse=True)
        for old in files[MAX_SESSIONS:]:
            old.unlink(missing_ok=True)
    except Exception as exc:
        log.debug("Session prune failed: %s", exc)

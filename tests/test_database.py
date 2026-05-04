"""Tests for valocoach.data.database — covering the remaining uncovered branches.

Gaps being covered:
  38-40  _set_sqlite_pragma — PRAGMA foreign_keys + synchronous + cursor.close()
  47-49  get_engine()       — both None-raise and success paths
  64     _find_alembic_dir  — return None when no alembic/ found in parents
  84     _stamp_if_needed   — early return when _find_alembic_dir() returns None
  95     _stamp_if_needed   — early return when head is None
  124    session_scope      — RuntimeError when _SessionLocal is None
  129-131 session_scope     — rollback path on exception inside the context
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _real_alembic_dir() -> Path:
    """Locate the project's alembic/ directory (used for mocking _find)."""
    return (Path(__file__).resolve().parents[1] / "alembic").resolve()


# ---------------------------------------------------------------------------
# Lines 38-40: _set_sqlite_pragma fires and applies all three PRAGMAs
#
# aiosqlite executes the connect listener in a thread-pool worker, which is
# invisible to coverage.py's default main-thread tracer.  The fix: extract
# the registered listener from the pool's dispatch and call it directly in
# the test thread with a real sqlite3 connection so coverage can track it.
# ---------------------------------------------------------------------------


def test_pragma_listener_covers_all_lines_directly(tmp_path: Path) -> None:
    """Call _set_sqlite_pragma in the test thread so coverage tracks lines 38-40.

    We extract the listener from the pool's connect dispatch — it is the last
    function in the list (registered last by init_engine).
    """
    import sqlite3

    from valocoach.data.database import init_engine

    engine = init_engine(tmp_path / "pragma_direct.db")
    pool = engine.sync_engine.pool

    # The last listener in pool.dispatch.connect is _set_sqlite_pragma.
    listeners = list(pool.dispatch.connect)
    pragma_fn = listeners[-1]
    assert pragma_fn.__name__ == "_set_sqlite_pragma"

    # Call it in-thread with a real sqlite3 connection.
    conn = sqlite3.connect(str(tmp_path / "pragma_direct.db"))
    try:
        pragma_fn(conn, None)  # covers lines 36-40

        # Verify the PRAGMAs were actually applied.
        cur = conn.cursor()
        fk = cur.execute("PRAGMA foreign_keys").fetchone()[0]
        jm = cur.execute("PRAGMA journal_mode").fetchone()[0]
        cur.close()
        assert fk == 1, "PRAGMA foreign_keys=ON not applied"
        assert jm.lower() == "wal", "PRAGMA journal_mode=WAL not applied"
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Lines 47-49: get_engine()
# ---------------------------------------------------------------------------


def test_get_engine_raises_when_engine_is_none(monkeypatch) -> None:
    """get_engine() must raise RuntimeError when _engine is None (lines 47-48)."""
    import valocoach.data.database as db_mod

    monkeypatch.setattr(db_mod, "_engine", None)
    with pytest.raises(RuntimeError, match="Engine not initialised"):
        db_mod.get_engine()


def test_get_engine_returns_engine_when_set(tmp_path: Path) -> None:
    """get_engine() returns the module-level engine when it is set (line 49)."""
    from valocoach.data.database import get_engine, init_engine

    engine = init_engine(tmp_path / "ge_test.db")
    result = get_engine()
    assert result is engine


# ---------------------------------------------------------------------------
# Line 64: _find_alembic_dir returns None when not found
# ---------------------------------------------------------------------------


def test_find_alembic_dir_returns_none_when_not_found(tmp_path: Path, monkeypatch) -> None:
    """When database.__file__ is inside a tree with no alembic/env.py, return None."""
    import valocoach.data.database as db_mod

    # Point __file__ at a temp tree that has no alembic/ anywhere.
    fake_file = tmp_path / "src" / "mymodule" / "database.py"
    fake_file.parent.mkdir(parents=True)
    monkeypatch.setattr(db_mod, "__file__", str(fake_file))

    result = db_mod._find_alembic_dir()
    assert result is None


# ---------------------------------------------------------------------------
# Line 84: _stamp_if_needed returns early when _find_alembic_dir() returns None
# ---------------------------------------------------------------------------


async def test_stamp_if_needed_skips_when_no_alembic_dir(tmp_path: Path, monkeypatch) -> None:
    """_stamp_if_needed must return silently when there is no alembic/ dir (line 84)."""
    import valocoach.data.database as db_mod

    monkeypatch.setattr(db_mod, "_find_alembic_dir", lambda: None)

    # Create a real engine so we can get a real sync conn for the function.
    engine = db_mod.init_engine(tmp_path / "nostamp.db")
    try:
        async with engine.begin() as conn:
            # Must not raise; stamp call is skipped entirely.
            await conn.run_sync(db_mod._stamp_if_needed)
    finally:
        await engine.dispose()


# ---------------------------------------------------------------------------
# Line 95: _stamp_if_needed returns early when head is None
# ---------------------------------------------------------------------------


async def test_stamp_if_needed_skips_when_head_is_none(tmp_path: Path, monkeypatch) -> None:
    """When ScriptDirectory has no head revision, _stamp_if_needed returns early (line 95)."""
    import valocoach.data.database as db_mod

    # Point to the real alembic dir so _find_alembic_dir succeeds.
    monkeypatch.setattr(db_mod, "_find_alembic_dir", lambda: _real_alembic_dir())

    # Mock ScriptDirectory.from_config to return a script with no head.
    mock_script = MagicMock()
    mock_script.get_current_head.return_value = None

    engine = db_mod.init_engine(tmp_path / "nohead.db")
    try:
        async with engine.begin() as conn:
            # Use a mock MigrationContext so get_current_revision() → None
            # (meaning "not yet stamped", so we proceed past line 88).
            mock_ctx = MagicMock()
            mock_ctx.get_current_revision.return_value = None

            with (
                patch(
                    "alembic.runtime.migration.MigrationContext.configure", return_value=mock_ctx
                ),
                patch("alembic.script.ScriptDirectory.from_config", return_value=mock_script),
            ):
                await conn.run_sync(db_mod._stamp_if_needed)

            # ctx.stamp must NOT have been called (we hit the head-is-None early return).
            mock_ctx.stamp.assert_not_called()
    finally:
        await engine.dispose()


# ---------------------------------------------------------------------------
# Line 124: session_scope raises when _SessionLocal is None
# ---------------------------------------------------------------------------


async def test_session_scope_raises_when_not_initialised(monkeypatch) -> None:
    """session_scope must raise RuntimeError when _SessionLocal is None (line 124)."""
    import valocoach.data.database as db_mod

    monkeypatch.setattr(db_mod, "_SessionLocal", None)

    with pytest.raises(RuntimeError, match="Engine not initialised"):
        async with db_mod.session_scope():
            pass  # should never reach here


# ---------------------------------------------------------------------------
# Lines 129-131: session_scope rolls back and re-raises on exception
# ---------------------------------------------------------------------------


async def test_session_scope_rolls_back_on_exception(tmp_path: Path) -> None:
    """An exception inside session_scope triggers rollback and is re-raised (lines 129-131)."""
    from valocoach.data.database import init_engine, session_scope

    init_engine(tmp_path / "rollback_test.db")

    sentinel = ValueError("deliberate test error")
    with pytest.raises(ValueError, match="deliberate test error"):
        async with session_scope() as _session:
            raise sentinel

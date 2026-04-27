"""Tests for ``ensure_db`` — the CLI's DB bootstrap path.

The main thing we guard against here is silent regressions in the
alembic-stamp step. If ``ensure_db`` goes back to a plain ``create_all``
(as it was before this change), a later ``alembic upgrade head`` against
a CLI-created DB will try to re-create existing tables and explode.
Pinning the stamp behaviour keeps that escape hatch alive.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from valocoach.data.database import ensure_db


def _alembic_head() -> str:
    """Parse the head revision from alembic/versions/ directly.

    We deliberately don't import from alembic here — the assertion needs
    to fail if someone adds a migration but forgets to update this test,
    which only happens when the two sources drift apart.
    """
    import re

    versions = Path(__file__).resolve().parents[1] / "alembic" / "versions"
    revs: dict[str, str | None] = {}
    for py in versions.glob("*.py"):
        if py.name == "__init__.py":
            continue
        rev: str | None = None
        down: str | None = None
        for line in py.read_text().splitlines():
            if line.startswith("revision: "):
                # handles both "abc123" and 'abc123'
                m = re.search(r'["\']([a-f0-9]+)["\']', line)
                if m:
                    rev = m.group(1)
            elif line.startswith("down_revision: "):
                m = re.search(r'["\']([a-f0-9]+)["\']', line)
                down = m.group(1) if m else None
        if rev:
            revs[rev] = down

    # Head = the revision nothing points at as down_revision.
    downs = set(revs.values())
    heads = [r for r in revs if r not in downs]
    assert len(heads) == 1, f"expected exactly one head, got {heads}"
    return heads[0]


async def test_ensure_db_creates_schema_and_stamps_head(tmp_path: Path) -> None:
    """First-run: both app tables and alembic_version land in one shot."""
    db = tmp_path / "valocoach.db"
    await ensure_db(db)

    con = sqlite3.connect(db)
    try:
        tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        rows = con.execute("SELECT version_num FROM alembic_version").fetchall()
    finally:
        con.close()

    # App schema exists (canary: one of the core tables from orm_models).
    assert "matches" in tables
    assert "match_players" in tables
    # And the stamp is written to the real head.
    assert "alembic_version" in tables
    assert rows == [(_alembic_head(),)]


async def test_ensure_db_is_idempotent(tmp_path: Path) -> None:
    """Calling ensure_db twice must not duplicate or reset alembic_version."""
    db = tmp_path / "valocoach.db"
    await ensure_db(db)
    await ensure_db(db)

    con = sqlite3.connect(db)
    try:
        rows = con.execute("SELECT version_num FROM alembic_version").fetchall()
    finally:
        con.close()

    # Exactly one row, still at head.
    assert rows == [(_alembic_head(),)]


async def test_ensure_db_does_not_overwrite_existing_stamp(tmp_path: Path) -> None:
    """If a prior alembic run already wrote alembic_version (possibly to
    an older revision mid-upgrade), ensure_db must NOT clobber it — alembic
    is the source of truth once it's been invoked."""
    db = tmp_path / "valocoach.db"
    await ensure_db(db)  # baseline: stamped at head

    # Simulate someone mid-migration by pinning to a fake older rev.
    con = sqlite3.connect(db)
    try:
        con.execute("UPDATE alembic_version SET version_num = 'older_rev_xyz'")
        con.commit()
    finally:
        con.close()

    await ensure_db(db)  # should leave the stamp alone

    con = sqlite3.connect(db)
    try:
        rows = con.execute("SELECT version_num FROM alembic_version").fetchall()
    finally:
        con.close()
    assert rows == [("older_rev_xyz",)], (
        "ensure_db overwrote an existing alembic_version row — alembic "
        "must remain authoritative once it has stamped the DB."
    )

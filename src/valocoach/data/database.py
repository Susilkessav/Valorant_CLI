from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from sqlalchemy import event
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.pool import NullPool


class Base(DeclarativeBase):
    """SQLAlchemy 2.0 declarative base."""


_engine: AsyncEngine | None = None
_SessionLocal: async_sessionmaker[AsyncSession] | None = None
_engine_db_path: str | None = None
_ensured_db_paths: set[str] = set()


def init_engine(db_path: Path) -> AsyncEngine:
    """Initialise the async SQLite engine.  Idempotent for the same path.

    Uses ``NullPool`` so each ``asyncio.run()`` boundary doesn't leak
    aiosqlite worker threads bound to a now-closed event loop — which would
    otherwise spam ``RuntimeError: Event loop is closed`` between turns in
    interactive sessions.

    The first call for a given ``db_path`` builds the engine; subsequent
    calls return the cached one.  Without this, every ``asyncio.run`` +
    ``ensure_db`` boundary in the coach path tore down and rebuilt the
    engine, re-registering pragma listeners each time.
    """
    global _engine, _SessionLocal, _engine_db_path

    resolved = str(db_path.resolve())
    if _engine is not None and _engine_db_path == resolved:
        return _engine

    db_path.parent.mkdir(parents=True, exist_ok=True)
    url = f"sqlite+aiosqlite:///{db_path}"
    _engine = create_async_engine(url, echo=False, future=True, poolclass=NullPool)
    _engine_db_path = resolved

    # Enable WAL + foreign keys on every connection
    @event.listens_for(_engine.sync_engine, "connect")
    def _set_sqlite_pragma(dbapi_connection, _):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.close()

    _SessionLocal = async_sessionmaker(bind=_engine, expire_on_commit=False)
    return _engine


def get_engine() -> AsyncEngine:
    if _engine is None:
        raise RuntimeError("Engine not initialised. Call init_engine() first.")
    return _engine


def _find_alembic_dir() -> Path | None:
    """Locate the repo's ``alembic/`` directory by walking up from this module.

    Returns ``None`` when running from an installed wheel — alembic sources
    aren't packaged (see ``[tool.hatch.build.targets.wheel]`` in pyproject),
    and without them we can't stamp. That's fine: an installed-wheel user
    can't run ``alembic upgrade`` either, so skipping the stamp is safe.
    """
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "alembic"
        if (candidate / "env.py").exists():
            return candidate
    return None


def _stamp_if_needed(sync_conn) -> None:
    """Mark the DB as being at the current Alembic head if it isn't already.

    Problem this solves: ``Base.metadata.create_all`` writes the schema but
    does not populate ``alembic_version``. A later ``alembic upgrade head``
    then tries to re-create existing tables and fails. Stamping here keeps
    CLI-created DBs forward-compatible with migrations.

    Idempotent: if ``alembic_version`` already has a row (because an alembic
    command ran previously), we leave it alone — alembic is in charge now.
    """
    from alembic.config import Config
    from alembic.runtime.migration import MigrationContext
    from alembic.script import ScriptDirectory

    alembic_dir = _find_alembic_dir()
    if alembic_dir is None:
        return

    ctx = MigrationContext.configure(sync_conn)
    if ctx.get_current_revision() is not None:
        return  # already stamped — leave it to alembic

    cfg = Config()
    cfg.set_main_option("script_location", str(alembic_dir))
    script = ScriptDirectory.from_config(cfg)
    head = script.get_current_head()
    if head is None:
        return

    # stamp() writes ``alembic_version`` within the current transaction,
    # which is what we want — keeps the schema + version write atomic.
    ctx.stamp(script, head)


async def ensure_db(db_path: Path) -> AsyncEngine:
    """Initialise the engine, create all tables, and stamp the Alembic head.

    Idempotent for the same ``db_path`` in the same process — the first call
    builds the engine, runs ``CREATE TABLE IF NOT EXISTS``, and stamps the
    Alembic head; subsequent calls return the cached engine and skip the
    schema + stamp work.  This matters because the coach path issues ~5
    independent ``asyncio.run(ensure_db(...))`` boundaries per turn (stats
    context, last match, open notes, top-played agents, …) — without the
    guard, each one re-imports Alembic and reruns ``ScriptDirectory``
    discovery to no effect.

    The guard is keyed on the resolved path string, so tests pointing at a
    different ``tmp_path`` each pytest run still get a clean ensure cycle.
    """
    engine = init_engine(db_path)

    resolved = str(db_path.resolve())
    if resolved in _ensured_db_paths:
        return engine

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await conn.run_sync(_stamp_if_needed)

    _ensured_db_paths.add(resolved)
    return engine


def reset_db_cache() -> None:
    """Drop the cached engine + ensure-marker.

    Tests that point at a fresh ``tmp_path`` per case and any code that
    moves the database file (rare — currently only the test suite) should
    call this between database swaps so the next ``ensure_db`` rebuilds
    the engine and re-runs the schema/stamp step against the new path.
    """
    global _engine, _SessionLocal, _engine_db_path
    _engine = None
    _SessionLocal = None
    _engine_db_path = None
    _ensured_db_paths.clear()


@asynccontextmanager
async def session_scope() -> AsyncIterator[AsyncSession]:
    """Transactional scope around a series of operations."""
    if _SessionLocal is None:
        raise RuntimeError("Engine not initialised.")
    session = _SessionLocal()
    try:
        yield session
        await session.commit()
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.close()

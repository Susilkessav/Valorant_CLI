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


class Base(DeclarativeBase):
    """SQLAlchemy 2.0 declarative base."""


_engine: AsyncEngine | None = None
_SessionLocal: async_sessionmaker[AsyncSession] | None = None


def init_engine(db_path: Path) -> AsyncEngine:
    """Initialise the async SQLite engine. Call once at startup."""
    global _engine, _SessionLocal

    db_path.parent.mkdir(parents=True, exist_ok=True)
    url = f"sqlite+aiosqlite:///{db_path}"
    _engine = create_async_engine(url, echo=False, future=True)

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

    Callers that need the DB ready (sync, stats, profile) should start with:

        engine = await ensure_db(settings.data_dir / "valocoach.db")

    Safe to call on every command invocation — ``CREATE TABLE IF NOT EXISTS``
    is a no-op when tables already exist, and the stamp step is guarded so
    it only writes ``alembic_version`` on first init.
    """
    engine = init_engine(db_path)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await conn.run_sync(_stamp_if_needed)
    return engine


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

"""add coaching_sessions and coaching_notes tables

Revision ID: e5f6a7b8c9d0
Revises: d4e5f6a7b8c9
Create Date: 2026-05-05 00:00:00.000000

Adds two tables for persisting coaching conversations:

``coaching_sessions``
    One row per coaching conversation or focus block.  Tied to a player via
    puuid FK (CASCADE delete).  Stores optional focus agent/map and open/closed
    timestamps so the CLI can detect an in-progress session on restart.

``coaching_notes``
    Individual coaching takeaways / action items produced during a session.
    FK to coaching_sessions (CASCADE delete).  The puuid column is denormalised
    to allow efficient "all open notes for player X" queries without a join.
    match_id is a plain string (no FK) so notes can reference matches not yet
    synced or future matches.

Indexes:
  idx_cs_puuid        — covers "sessions for player X ordered by time"
  idx_cn_session      — covers "notes in session X" (used by session detail view)
  idx_cn_puuid_resolved — covers "unresolved notes for player X"
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "e5f6a7b8c9d0"
down_revision: str | Sequence[str] | None = "d4e5f6a7b8c9"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "coaching_sessions",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("puuid", sa.Text(), nullable=False),
        sa.Column("started_at", sa.Text(), nullable=False),
        sa.Column("ended_at", sa.Text(), nullable=True),
        sa.Column("session_title", sa.Text(), nullable=True),
        sa.Column("focus_agent", sa.Text(), nullable=True),
        sa.Column("focus_map", sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(["puuid"], ["players.puuid"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("idx_cs_puuid", "coaching_sessions", ["puuid", "started_at"])

    op.create_table(
        "coaching_notes",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("session_id", sa.Integer(), nullable=False),
        sa.Column("puuid", sa.Text(), nullable=False),
        sa.Column("body", sa.Text(), nullable=False),
        sa.Column("category", sa.Text(), server_default="general", nullable=False),
        sa.Column("priority", sa.Integer(), server_default="2", nullable=False),
        sa.Column("resolved", sa.Boolean(), server_default="0", nullable=False),
        sa.Column("match_id", sa.Text(), nullable=True),
        sa.Column("created_at", sa.Text(), nullable=False),
        sa.Column("resolved_at", sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(["session_id"], ["coaching_sessions.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("idx_cn_session", "coaching_notes", ["session_id"])
    op.create_index("idx_cn_puuid_resolved", "coaching_notes", ["puuid", "resolved"])


def downgrade() -> None:
    op.drop_index("idx_cn_puuid_resolved", table_name="coaching_notes")
    op.drop_index("idx_cn_session", table_name="coaching_notes")
    op.drop_table("coaching_notes")
    op.drop_index("idx_cs_puuid", table_name="coaching_sessions")
    op.drop_table("coaching_sessions")

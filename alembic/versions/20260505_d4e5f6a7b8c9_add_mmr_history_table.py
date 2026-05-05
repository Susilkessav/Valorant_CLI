"""add mmr_history table

Revision ID: d4e5f6a7b8c9
Revises: c1d2e3f4a5b6
Create Date: 2026-05-05 00:00:00.000000

Adds ``mmr_history`` — one rank snapshot per sync per player — enabling
rank-progression charts and session-level RR-delta reporting.

A snapshot is inserted when the player's ELO differs from their previous
snapshot; identical re-syncs (e.g. player opened the app twice without
playing) produce no new row.  This keeps the table sparse and query-friendly.

Index on ``(puuid, recorded_at)`` covers the three primary access patterns:
  - "all snapshots for player X" (coaching context)
  - "last N snapshots for player X" (sparkline / progression card)
  - "snapshots between date A and date B" (season comparison)
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "d4e5f6a7b8c9"
down_revision: str | Sequence[str] | None = "c1d2e3f4a5b6"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "mmr_history",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("puuid", sa.Text(), nullable=False),
        sa.Column("recorded_at", sa.Text(), nullable=False),
        sa.Column("tier", sa.Integer(), server_default="0", nullable=False),
        sa.Column("tier_patched", sa.Text(), server_default="Unranked", nullable=False),
        sa.Column("rr", sa.Integer(), server_default="0", nullable=False),
        sa.Column("elo", sa.Integer(), server_default="0", nullable=False),
        sa.Column("mmr_change", sa.Integer(), nullable=True),
        sa.ForeignKeyConstraint(["puuid"], ["players.puuid"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("idx_mmr_history_puuid", "mmr_history", ["puuid", "recorded_at"])


def downgrade() -> None:
    op.drop_index("idx_mmr_history_puuid", table_name="mmr_history")
    op.drop_table("mmr_history")

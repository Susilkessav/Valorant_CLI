"""add idx_kills_round and team CHECK constraints

Revision ID: b7c4d9e2a3f1
Revises: 3d2d3a1122f4
Create Date: 2026-05-04 00:00:00.000000

Two correctness fixes from the May 2026 review:

1. ``idx_kills_round`` — Round-level analysis (KAST, clutch, trade) loads
   kills via ``selectinload(Round.kills)`` which fires one query per round
   filtered by ``kills.round_id``.  Without this index, each query falls
   back to a full kills-table scan.  A 30-match recent-form analysis with
   ~750 rounds was doing 750 scans.

2. CHECK constraints on ``team`` columns — ``match_players.team`` and
   ``rounds.winning_team`` were declared TEXT NOT NULL but accepted any
   string.  A mapper typo (e.g. "RED" vs "Red") would silently corrupt
   every side-split calculation downstream because the side-assignment
   helper compares string-equal to "Red"/"Blue".  The DB now enforces the
   enum directly.
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "b7c4d9e2a3f1"
down_revision: str | Sequence[str] | None = "3d2d3a1122f4"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # idx_kills_round on the kills table.
    with op.batch_alter_table("kills", schema=None) as batch_op:
        batch_op.create_index("idx_kills_round", ["round_id"], unique=False)

    # CHECK constraints — SQLite needs batch_alter_table for this since it
    # rebuilds the table to add the constraint cleanly.
    with op.batch_alter_table("match_players", schema=None) as batch_op:
        batch_op.create_check_constraint(
            "ck_match_player_team",
            sa.text("team IN ('Red', 'Blue')"),
        )

    with op.batch_alter_table("rounds", schema=None) as batch_op:
        batch_op.create_check_constraint(
            "ck_round_winning_team",
            sa.text("winning_team IN ('Red', 'Blue')"),
        )


def downgrade() -> None:
    with op.batch_alter_table("rounds", schema=None) as batch_op:
        batch_op.drop_constraint("ck_round_winning_team", type_="check")

    with op.batch_alter_table("match_players", schema=None) as batch_op:
        batch_op.drop_constraint("ck_match_player_team", type_="check")

    with op.batch_alter_table("kills", schema=None) as batch_op:
        batch_op.drop_index("idx_kills_round")

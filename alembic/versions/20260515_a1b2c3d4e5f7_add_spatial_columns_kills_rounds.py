"""add spatial columns to kills and rounds (Phase A1+A2)

Revision ID: a1b2c3d4e5f7
Revises: f1e2d3c4b5a6
Create Date: 2026-05-15 00:00:00.000000

Phase A1 — Kill coordinates
---------------------------
Adds killer/victim map-coordinate columns and engagement_distance to the
``kills`` table.  All columns are NULL-able so existing rows from before
this migration remain valid — spatial data will only be present for matches
synced after this migration runs.

  killer_x, killer_y        — map coordinates of the killing shot origin
  victim_x, victim_y        — map coordinates of the victim at death
  engagement_distance       — Euclidean distance (TEXT, cast to float on read)

Phase A2 — Plant/defuse coordinates
-------------------------------------
Adds plant and defuse map coordinates to the ``rounds`` table.  Same
NULL-able treatment — existing rows unaffected.

  plant_x, plant_y          — bomb plant position
  defuse_x, defuse_y        — bomb defuse position

All coordinates are in HenrikDev's API coordinate system (raw integers).
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "a1b2c3d4e5f7"
down_revision: str | Sequence[str] | None = "f1e2d3c4b5a6"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # ── Phase A1: kills spatial columns ───────────────────────────────────
    with op.batch_alter_table("kills") as batch_op:
        batch_op.add_column(sa.Column("killer_x", sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column("killer_y", sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column("victim_x", sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column("victim_y", sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column("engagement_distance", sa.Text(), nullable=True))

    # ── Phase A2: rounds plant/defuse coordinates ──────────────────────────
    with op.batch_alter_table("rounds") as batch_op:
        batch_op.add_column(sa.Column("plant_x", sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column("plant_y", sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column("defuse_x", sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column("defuse_y", sa.Integer(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("rounds") as batch_op:
        batch_op.drop_column("defuse_y")
        batch_op.drop_column("defuse_x")
        batch_op.drop_column("plant_y")
        batch_op.drop_column("plant_x")

    with op.batch_alter_table("kills") as batch_op:
        batch_op.drop_column("engagement_distance")
        batch_op.drop_column("victim_y")
        batch_op.drop_column("victim_x")
        batch_op.drop_column("killer_y")
        batch_op.drop_column("killer_x")

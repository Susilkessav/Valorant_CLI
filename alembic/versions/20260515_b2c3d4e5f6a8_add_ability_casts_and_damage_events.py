"""add ability casts and damage events to round_players (Phase A3+A4)

Revision ID: b2c3d4e5f6a8
Revises: a1b2c3d4e5f7
Create Date: 2026-05-15 00:00:00.000000

Phase A3 — Ability casts per round per player
----------------------------------------------
Adds four per-round ability cast count columns to ``round_players``.  The
HenrikDev v4 API provides these in ``rounds[*].stats[*].ability_casts`` but
they were previously dropped at the mapper layer.

  ability_casts_grenade  INTEGER NULL  — C/grenade slot casts
  ability_casts_ability1 INTEGER NULL  — Q/ability-1 casts
  ability_casts_ability2 INTEGER NULL  — E/ability-2 casts
  ability_casts_ultimate INTEGER NULL  — X/ultimate casts

Phase A4 — Damage events summary per round per player
------------------------------------------------------
Adds a compact JSON summary of damage dealt per target in a round.  Raw
damage_events are lists of ``{receiver, damage, headshots, bodyshots,
legshots}`` — we strip receiver team info and store a compact array.

  damage_events_json  TEXT NULL  — JSON array of compact damage objects

All columns are NULL-able; existing rows unaffected.
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "b2c3d4e5f6a8"
down_revision: str | Sequence[str] | None = "a1b2c3d4e5f7"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("round_players") as batch_op:
        # Phase A3: ability casts
        batch_op.add_column(sa.Column("ability_casts_grenade", sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column("ability_casts_ability1", sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column("ability_casts_ability2", sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column("ability_casts_ultimate", sa.Integer(), nullable=True))
        # Phase A4: damage events
        batch_op.add_column(sa.Column("damage_events_json", sa.Text(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("round_players") as batch_op:
        batch_op.drop_column("damage_events_json")
        batch_op.drop_column("ability_casts_ultimate")
        batch_op.drop_column("ability_casts_ability2")
        batch_op.drop_column("ability_casts_ability1")
        batch_op.drop_column("ability_casts_grenade")

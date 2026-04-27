"""add planter/defuser puuid to rounds and economy cols to match_players

Revision ID: a1b2c3d4e5f6
Revises: 9edf9a0a8f87
Create Date: 2026-04-21 12:00:00.000000

Adds:
    rounds.planter_puuid   — puuid of the player who planted the spike
    rounds.defuser_puuid   — puuid of the player who defused the spike
    match_players.credits_spent — total credits spent across the match
    match_players.avg_loadout   — average loadout value per round

All four columns are nullable so existing rows remain valid without a
full re-sync.  New syncs (v4) populate them; old rows leave them NULL.
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "a1b2c3d4e5f6"
down_revision: str | None = "9edf9a0a8f87"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("rounds", schema=None) as batch_op:
        batch_op.add_column(sa.Column("planter_puuid", sa.String(), nullable=True))
        batch_op.add_column(sa.Column("defuser_puuid", sa.String(), nullable=True))

    with op.batch_alter_table("match_players", schema=None) as batch_op:
        batch_op.add_column(sa.Column("credits_spent", sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column("avg_loadout", sa.Integer(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("match_players", schema=None) as batch_op:
        batch_op.drop_column("avg_loadout")
        batch_op.drop_column("credits_spent")

    with op.batch_alter_table("rounds", schema=None) as batch_op:
        batch_op.drop_column("defuser_puuid")
        batch_op.drop_column("planter_puuid")

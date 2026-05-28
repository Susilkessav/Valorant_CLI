"""add match_context_json to coaching_sessions

Revision ID: f1e2d3c4b5a6
Revises: e5f6a7b8c9d0
Create Date: 2026-05-15 00:00:00.000000

Adds a ``match_context_json`` column to ``coaching_sessions`` so that the
``SessionMatchContext`` state (agent, map, side, score, enemies etc.) can be
persisted when a REPL session ends and restored on the next startup.

Column is NULL-able — existing rows and sessions that never used the context
slash commands simply leave it NULL.  The JSON schema mirrors
``SessionMatchContext.to_json()`` / ``from_json()``.
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "f1e2d3c4b5a6"
down_revision: str | Sequence[str] | None = "e5f6a7b8c9d0"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("coaching_sessions") as batch_op:
        batch_op.add_column(sa.Column("match_context_json", sa.Text(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("coaching_sessions") as batch_op:
        batch_op.drop_column("match_context_json")

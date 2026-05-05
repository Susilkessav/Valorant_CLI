"""add round_players table

Revision ID: c1d2e3f4a5b6
Revises: b7c4d9e2a3f1
Create Date: 2026-05-05 00:00:00.000000

Adds the ``round_players`` table — one row per (round_id, puuid) — which
stores the per-player, per-round stats available in the HenrikDev v4
``rounds[n].stats[*]`` payload.

Motivation
----------
The existing ``match_players`` table records aggregate stats per match.
``round_players`` fills the gaps that aggregates can't represent:

* **Econ rating** — ``loadout_value`` and ``remaining_credits`` per round
  enable a credits-based damage efficiency metric.  The match-level
  ``credits_spent`` approximation (a single sum) loses round-by-round
  context (e.g. buy round vs eco round).

* **Survival flag** — ``survived=True/False`` answers "did the player
  die this round?" without scanning the kills table for every KAST query.

* **Round-score variance** — ``score`` per round lets the stats engine
  detect rounds where the player underperformed vs. their own average
  even when the match aggregate looks fine.

* **Damage per round** — ``damage_dealt`` (summed from damage_events) is
  the numerator for a per-round ADR computation.

Schema notes
------------
* ``survived`` is derived at map-time from the kills list and stored so
  the KAST S-dimension is a simple column scan rather than a join.
* ``damage_dealt`` is the sum of all ``damage_events[*].damage`` values
  for the player in that round (includes friendly fire when present, but
  this mirrors how the API reports it at the match level too).
* The index on ``(match_id, puuid)`` covers the most common query:
  "fetch all round_players for matches [ids] for player [puuid]."
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "c1d2e3f4a5b6"
down_revision: str | Sequence[str] | None = "b7c4d9e2a3f1"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "round_players",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("round_id", sa.Integer(), nullable=False),
        sa.Column("match_id", sa.Text(), nullable=False),
        sa.Column("puuid", sa.Text(), nullable=False),
        sa.Column("team", sa.Text(), nullable=False),
        sa.Column("score", sa.Integer(), server_default="0", nullable=False),
        sa.Column("kills", sa.Integer(), server_default="0", nullable=False),
        sa.Column("headshots", sa.Integer(), server_default="0", nullable=False),
        sa.Column("bodyshots", sa.Integer(), server_default="0", nullable=False),
        sa.Column("legshots", sa.Integer(), server_default="0", nullable=False),
        sa.Column("damage_dealt", sa.Integer(), server_default="0", nullable=False),
        sa.Column("loadout_value", sa.Integer(), nullable=True),
        sa.Column("remaining_credits", sa.Integer(), nullable=True),
        sa.Column("survived", sa.Boolean(), server_default="1", nullable=False),
        sa.Column("was_afk", sa.Boolean(), server_default="0", nullable=False),
        sa.Column("stayed_in_spawn", sa.Boolean(), server_default="0", nullable=False),
        sa.ForeignKeyConstraint(["round_id"], ["rounds.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("round_id", "puuid", name="uq_round_player"),
    )
    op.create_index("idx_rp_match_puuid", "round_players", ["match_id", "puuid"])


def downgrade() -> None:
    op.drop_index("idx_rp_match_puuid", table_name="round_players")
    op.drop_table("round_players")

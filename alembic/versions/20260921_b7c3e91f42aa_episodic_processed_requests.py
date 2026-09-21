"""episodic add processed_requests

Revision ID: b7c3e91f42aa
Revises: a1b2c3d4e5f6
Create Date: 2026-09-21

Registre d'idempotence des écritures. Le contrat AML rejoue la même écriture
logique — même request_id, même charge — jusqu'à 32 fois sur erreur réseau ou
5xx. Sans ce registre, un rejeu consécutif à une écriture réussie mais mal
acquittée dupliquerait les souvenirs.
"""
from __future__ import annotations

from alembic import op
from sqlalchemy import inspect

revision = "b7c3e91f42aa"
down_revision = "a1b2c3d4e5f6"
branch_labels = None
depends_on = None

TARGET_DB = "episodic"


def _has_table(table: str) -> bool:
    return table in set(inspect(op.get_bind()).get_table_names())


def upgrade(db_name: str = "") -> None:
    if db_name != TARGET_DB:
        return
    if not _has_table("processed_requests"):
        op.execute(
            """
            CREATE TABLE processed_requests (
              tenant          TEXT NOT NULL,
              request_id      TEXT NOT NULL,
              created_at      INTEGER NOT NULL,
              episode_count   INTEGER NOT NULL,
              PRIMARY KEY (tenant, request_id)
            ) STRICT
            """
        )


def downgrade(db_name: str = "") -> None:
    if db_name != TARGET_DB:
        return
    if _has_table("processed_requests"):
        op.execute("DROP TABLE processed_requests")

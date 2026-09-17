"""episodic add extraction_attempted_at

Revision ID: a1b2c3d4e5f6
Revises: f4a92c81e3d7
Create Date: 2026-09-17

Ajoute la colonne `extraction_attempted_at` (INTEGER) à `episodes` pour éviter
la ré-extraction infinie des épisodes sans faits consolidables.
"""
from __future__ import annotations

from alembic import op
from sqlalchemy import inspect

revision = "a1b2c3d4e5f6"
down_revision = "f4a92c81e3d7"
branch_labels = None
depends_on = None

TARGET_DB = "episodic"


def _has_column(table: str, column: str) -> bool:
    insp = inspect(op.get_bind())
    return column in {c["name"] for c in insp.get_columns(table)}


def upgrade(db_name: str = "") -> None:
    if db_name != TARGET_DB:
        return
    if not _has_column("episodes", "extraction_attempted_at"):
        op.execute("ALTER TABLE episodes ADD COLUMN extraction_attempted_at INTEGER")


def downgrade(db_name: str = "") -> None:
    if db_name != TARGET_DB:
        return
    if _has_column("episodes", "extraction_attempted_at"):
        op.execute("ALTER TABLE episodes DROP COLUMN extraction_attempted_at")

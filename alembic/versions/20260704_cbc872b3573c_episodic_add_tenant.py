"""episodic add tenant (multi-tenant, Lot 1 / P1)

Revision ID: cbc872b3573c
Revises: 30beed46db04
Create Date: 2026-07-04

Ajoute la colonne `tenant` (TEXT NOT NULL DEFAULT 'user') à `episodes` et
l'index `idx_episodes_tenant`. Idempotente (guard sur la présence de la
colonne/index) et réversible.

Backfill : `ADD COLUMN ... NOT NULL DEFAULT 'user'` remplit l'existant avec
le tenant personnel — les épisodes déjà en base restent la mémoire de `user`.
Rollback : DROP de l'index + de la colonne (SQLite ≥ 3.35 supporte DROP
COLUMN). Note complète dans MNEMOS_API.md / la note de migration.
"""
from __future__ import annotations

from alembic import op
from sqlalchemy import inspect

from mnemos.tenancy import DEFAULT_TENANT

revision = "cbc872b3573c"
down_revision = "30beed46db04"
branch_labels = None
depends_on = None

TARGET_DB = "episodic"


def _has_column(table: str, column: str) -> bool:
    insp = inspect(op.get_bind())
    return column in {c["name"] for c in insp.get_columns(table)}


def _has_index(table: str, index: str) -> bool:
    insp = inspect(op.get_bind())
    return index in {i["name"] for i in insp.get_indexes(table)}


def upgrade(db_name: str = "") -> None:
    if db_name != TARGET_DB:
        return
    if not _has_column("episodes", "tenant"):
        op.execute(
            f"ALTER TABLE episodes ADD COLUMN tenant TEXT NOT NULL "
            f"DEFAULT '{DEFAULT_TENANT}'"
        )
    if not _has_index("episodes", "idx_episodes_tenant"):
        op.execute("CREATE INDEX idx_episodes_tenant ON episodes(tenant, created_at DESC)")


def downgrade(db_name: str = "") -> None:
    if db_name != TARGET_DB:
        return
    if _has_index("episodes", "idx_episodes_tenant"):
        op.execute("DROP INDEX idx_episodes_tenant")
    if _has_column("episodes", "tenant"):
        op.execute("ALTER TABLE episodes DROP COLUMN tenant")

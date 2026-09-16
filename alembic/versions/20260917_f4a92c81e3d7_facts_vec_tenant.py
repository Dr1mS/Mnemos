"""semantic facts_vec add tenant (partitionnement natif vec0 & purge invalidés)

Revision ID: f4a92c81e3d7
Revises: e3f8194b1c23
Create Date: 2026-09-17

Recrée la table virtuelle facts_vec (vec0) avec la colonne de partition `tenant`
pour éliminer le masquage KNN cross-tenant. Seuls les faits ACTIFS (valid_until IS NULL)
sont migrés, éliminant ainsi les vecteurs fantômes obsolètes qui masquaient
les faits courants lors des recherches vectorielles.
"""
from __future__ import annotations

from alembic import op
from sqlalchemy import inspect

revision = "f4a92c81e3d7"
down_revision = "e3f8194b1c23"
branch_labels = None
depends_on = None

TARGET_DB = "semantic"


def _has_column(table: str, column: str) -> bool:
    insp = inspect(op.get_bind())
    return column in {c["name"] for c in insp.get_columns(table)}


def upgrade(db_name: str = "") -> None:
    if db_name != TARGET_DB:
        return
    if _has_column("facts_vec", "tenant"):
        return

    # 1. Sauvegarde temporaire des vecteurs des faits ACTIFS (valid_until IS NULL) avec leur tenant
    op.execute(
        """
        CREATE TABLE _temp_facts_vec_mig AS
        SELECT v.fact_id, f.tenant, v.embedding
        FROM facts_vec v
        JOIN facts f ON f.id = v.fact_id
        WHERE f.valid_until IS NULL
        """
    )
    # 2. Recréation de facts_vec avec tenant
    op.execute("DROP TABLE facts_vec")
    op.execute(
        """
        CREATE VIRTUAL TABLE facts_vec USING vec0(
          fact_id          TEXT PRIMARY KEY,
          tenant           TEXT,
          embedding        FLOAT[1024] distance_metric=cosine
        )
        """
    )
    # 3. Restauration des vecteurs des faits actifs
    op.execute(
        """
        INSERT INTO facts_vec(fact_id, tenant, embedding)
        SELECT fact_id, tenant, embedding FROM _temp_facts_vec_mig
        """
    )
    # 4. Nettoyage
    op.execute("DROP TABLE _temp_facts_vec_mig")


def downgrade(db_name: str = "") -> None:
    if db_name != TARGET_DB:
        return
    if not _has_column("facts_vec", "tenant"):
        return

    op.execute(
        """
        CREATE TABLE _temp_facts_vec_downgrade AS
        SELECT fact_id, embedding FROM facts_vec
        """
    )
    op.execute("DROP TABLE facts_vec")
    op.execute(
        """
        CREATE VIRTUAL TABLE facts_vec USING vec0(
          fact_id          TEXT PRIMARY KEY,
          embedding        FLOAT[1024] distance_metric=cosine
        )
        """
    )
    op.execute(
        """
        INSERT INTO facts_vec(fact_id, embedding)
        SELECT fact_id, embedding FROM _temp_facts_vec_downgrade
        """
    )
    op.execute("DROP TABLE _temp_facts_vec_downgrade")

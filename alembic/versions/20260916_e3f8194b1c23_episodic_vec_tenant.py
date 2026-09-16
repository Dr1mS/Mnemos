"""episodic vec add tenant (partitionnement natif vec0)

Revision ID: e3f8194b1c23
Revises: df716235da4c
Create Date: 2026-09-16

Recrée la table virtuelle episodes_vec (vec0) avec la colonne de partition `tenant`
pour éliminer le masquage KNN multi-tenant. Les embeddings existants sont préservés
par table temporaire et jointure sur episodes.tenant.
"""
from __future__ import annotations

from alembic import op
from sqlalchemy import inspect

revision = "e3f8194b1c23"
down_revision = "df716235da4c"
branch_labels = None
depends_on = None

TARGET_DB = "episodic"


def _has_column(table: str, column: str) -> bool:
    insp = inspect(op.get_bind())
    return column in {c["name"] for c in insp.get_columns(table)}


def upgrade(db_name: str = "") -> None:
    if db_name != TARGET_DB:
        return
    if _has_column("episodes_vec", "tenant"):
        return

    # 1. Sauvegarde temporaire des vecteurs existants avec leur tenant
    op.execute(
        """
        CREATE TABLE _temp_vec_mig AS
        SELECT v.episode_id, e.tenant, v.embedding
        FROM episodes_vec v
        JOIN episodes e ON e.id = v.episode_id
        """
    )
    # 2. Recréation de episodes_vec avec tenant
    op.execute("DROP TABLE episodes_vec")
    op.execute(
        """
        CREATE VIRTUAL TABLE episodes_vec USING vec0(
          episode_id      TEXT PRIMARY KEY,
          tenant          TEXT,
          embedding       FLOAT[1024] distance_metric=cosine
        )
        """
    )
    # 3. Restauration des vecteurs
    op.execute(
        """
        INSERT INTO episodes_vec(episode_id, tenant, embedding)
        SELECT episode_id, tenant, embedding FROM _temp_vec_mig
        """
    )
    # 4. Nettoyage
    op.execute("DROP TABLE _temp_vec_mig")


def downgrade(db_name: str = "") -> None:
    if db_name != TARGET_DB:
        return
    if not _has_column("episodes_vec", "tenant"):
        return

    op.execute(
        """
        CREATE TABLE _temp_vec_downgrade AS
        SELECT episode_id, embedding FROM episodes_vec
        """
    )
    op.execute("DROP TABLE episodes_vec")
    op.execute(
        """
        CREATE VIRTUAL TABLE episodes_vec USING vec0(
          episode_id      TEXT PRIMARY KEY,
          embedding       FLOAT[1024] distance_metric=cosine
        )
        """
    )
    op.execute(
        """
        INSERT INTO episodes_vec(episode_id, embedding)
        SELECT episode_id, embedding FROM _temp_vec_downgrade
        """
    )
    op.execute("DROP TABLE _temp_vec_downgrade")

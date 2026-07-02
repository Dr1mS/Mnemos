"""semantic initial schema

Revision ID: 30beed46db04
Revises: accd3092e2d2
Create Date: 2026-07-02 11:48:48

DDL brut depuis mnemos.models.semantic (source de vérité — cf. §18).
"""
from __future__ import annotations

from alembic import op

from mnemos.models.semantic import SEMANTIC_SCHEMA_DROP_SQL, SEMANTIC_SCHEMA_SQL

revision = "30beed46db04"
down_revision = "accd3092e2d2"
branch_labels = None
depends_on = None

# 'episodic' ou 'semantic' — la migration ne s'applique qu'à cette base.
TARGET_DB = "semantic"


def upgrade(db_name: str = "") -> None:
    if db_name != TARGET_DB:
        return
    for stmt in SEMANTIC_SCHEMA_SQL:
        op.execute(stmt)


def downgrade(db_name: str = "") -> None:
    if db_name != TARGET_DB:
        return
    for stmt in SEMANTIC_SCHEMA_DROP_SQL:
        op.execute(stmt)

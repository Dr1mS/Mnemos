"""episodic initial schema

Revision ID: accd3092e2d2
Revises:
Create Date: 2026-07-02 11:25:14

DDL brut depuis mnemos.models.episodic (source de vérité — tables STRICT
et vec0 inexprimables via l'autogenerate, cf. §18).
"""
from __future__ import annotations

from alembic import op

from mnemos.models.episodic import EPISODIC_SCHEMA_DROP_SQL, EPISODIC_SCHEMA_SQL

revision = "accd3092e2d2"
down_revision = None
branch_labels = None
depends_on = None

# 'episodic' ou 'semantic' — la migration ne s'applique qu'à cette base.
TARGET_DB = "episodic"


def upgrade(db_name: str = "") -> None:
    if db_name != TARGET_DB:
        return
    for stmt in EPISODIC_SCHEMA_SQL:
        op.execute(stmt)


def downgrade(db_name: str = "") -> None:
    if db_name != TARGET_DB:
        return
    for stmt in EPISODIC_SCHEMA_DROP_SQL:
        op.execute(stmt)

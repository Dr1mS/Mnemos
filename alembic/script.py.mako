"""${message}

Revision ID: ${up_revision}
Revises: ${down_revision | comma,n}
Create Date: ${create_date}

Multi-DB : chaque migration déclare sa base cible via TARGET_DB et
no-op sur l'autre (env.py passe db_name à run_migrations).
"""
from __future__ import annotations

${imports if imports else ""}
from alembic import op
import sqlalchemy as sa

revision = ${repr(up_revision)}
down_revision = ${repr(down_revision)}
branch_labels = ${repr(branch_labels)}
depends_on = ${repr(depends_on)}

# 'episodic' ou 'semantic' — la migration ne s'applique qu'à cette base.
TARGET_DB = "episodic"


def upgrade(db_name: str = "") -> None:
    if db_name != TARGET_DB:
        return
    ${upgrades if upgrades else "pass"}


def downgrade(db_name: str = "") -> None:
    if db_name != TARGET_DB:
        return
    ${downgrades if downgrades else "pass"}

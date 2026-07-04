"""semantic add tenant (multi-tenant, Lot 1 / P1)

Revision ID: df716235da4c
Revises: cbc872b3573c
Create Date: 2026-07-04

Ajoute `tenant` à `facts` (+ réindexation tenant-aware) et recrée `entities`
avec une PK COMPOSITE (tenant, canonical_name) — SQLite n'autorise pas
d'ALTER sur une clé primaire, d'où la recréation copy → drop → rename.

Idempotente (guards) et réversible. Backfill : tout l'existant retombe sur
le tenant `user` (mémoire personnelle). Les 47 faits / 36 entités déjà en
base restent inchangés à ceci près qu'ils portent désormais tenant='user'.

Rollback : facts → DROP COLUMN tenant + réindexation d'origine ; entities →
recréation avec PK (canonical_name) seule (perd la dimension tenant, à ne
faire que si un seul tenant existe, sinon collision de PK).
"""
from __future__ import annotations

from alembic import op
from sqlalchemy import inspect

from mnemos.tenancy import DEFAULT_TENANT

revision = "df716235da4c"
down_revision = "cbc872b3573c"
branch_labels = None
depends_on = None

TARGET_DB = "semantic"


def _has_column(table: str, column: str) -> bool:
    insp = inspect(op.get_bind())
    return column in {c["name"] for c in insp.get_columns(table)}


def _pk_columns(table: str) -> list[str]:
    insp = inspect(op.get_bind())
    return list(insp.get_pk_constraint(table)["constrained_columns"])


def upgrade(db_name: str = "") -> None:
    if db_name != TARGET_DB:
        return

    # ── facts : ADD COLUMN tenant + réindexation tenant-aware ────────────────
    if not _has_column("facts", "tenant"):
        op.execute(
            f"ALTER TABLE facts ADD COLUMN tenant TEXT NOT NULL "
            f"DEFAULT '{DEFAULT_TENANT}'"
        )
    op.execute("DROP INDEX IF EXISTS idx_facts_subject")
    op.execute("DROP INDEX IF EXISTS idx_facts_current")
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_facts_subject "
        "ON facts(tenant, subject, predicate, valid_until)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_facts_current "
        "ON facts(tenant, valid_until) WHERE valid_until IS NULL"
    )

    # ── entities : recréation avec PK composite (tenant, canonical_name) ─────
    if _pk_columns("entities") != ["tenant", "canonical_name"]:
        op.execute("DROP INDEX IF EXISTS idx_entities_last_seen")
        op.execute(
            f"""
            CREATE TABLE entities_new (
              tenant           TEXT NOT NULL DEFAULT '{DEFAULT_TENANT}',
              canonical_name   TEXT NOT NULL,
              aliases          TEXT NOT NULL DEFAULT '[]',
              entity_type      TEXT,
              first_seen       INTEGER NOT NULL,
              last_seen        INTEGER NOT NULL,
              episode_count    INTEGER NOT NULL DEFAULT 0,
              PRIMARY KEY (tenant, canonical_name)
            ) STRICT
            """
        )
        op.execute(
            f"""
            INSERT INTO entities_new
              (tenant, canonical_name, aliases, entity_type,
               first_seen, last_seen, episode_count)
            SELECT '{DEFAULT_TENANT}', canonical_name, aliases, entity_type,
               first_seen, last_seen, episode_count
            FROM entities
            """
        )
        op.execute("DROP TABLE entities")
        op.execute("ALTER TABLE entities_new RENAME TO entities")
        op.execute("CREATE INDEX idx_entities_last_seen ON entities(tenant, last_seen DESC)")


def downgrade(db_name: str = "") -> None:
    if db_name != TARGET_DB:
        return

    # ── entities : retour à la PK (canonical_name) seule ────────────────────
    if _pk_columns("entities") == ["tenant", "canonical_name"]:
        op.execute("DROP INDEX IF EXISTS idx_entities_last_seen")
        op.execute(
            """
            CREATE TABLE entities_old (
              canonical_name   TEXT PRIMARY KEY,
              aliases          TEXT NOT NULL DEFAULT '[]',
              entity_type      TEXT,
              first_seen       INTEGER NOT NULL,
              last_seen        INTEGER NOT NULL,
              episode_count    INTEGER NOT NULL DEFAULT 0
            ) STRICT
            """
        )
        # Un seul tenant attendu au rollback : on aplatit sur canonical_name.
        op.execute(
            """
            INSERT OR IGNORE INTO entities_old
              (canonical_name, aliases, entity_type, first_seen, last_seen, episode_count)
            SELECT canonical_name, aliases, entity_type, first_seen, last_seen, episode_count
            FROM entities
            """
        )
        op.execute("DROP TABLE entities")
        op.execute("ALTER TABLE entities_old RENAME TO entities")
        op.execute("CREATE INDEX idx_entities_last_seen ON entities(last_seen DESC)")

    # ── facts : réindexation d'origine + DROP COLUMN tenant ─────────────────
    op.execute("DROP INDEX IF EXISTS idx_facts_subject")
    op.execute("DROP INDEX IF EXISTS idx_facts_current")
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_facts_subject "
        "ON facts(subject, predicate, valid_until)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_facts_current "
        "ON facts(valid_until) WHERE valid_until IS NULL"
    )
    if _has_column("facts", "tenant"):
        op.execute("ALTER TABLE facts DROP COLUMN tenant")

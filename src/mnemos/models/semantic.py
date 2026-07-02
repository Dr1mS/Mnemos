"""Schéma semantic (§5.2) — ORM + DDL source de vérité.

Faits versionnés à la Memstate : subject + predicate + valid_until=NULL =
"fait courant". Même convention que models/episodic.py : le DDL brut est
consommé par la migration ET les fixtures de test.
"""

from __future__ import annotations

from sqlalchemy.orm import Mapped, mapped_column

from mnemos.models.base import Base

SEMANTIC_SCHEMA_SQL: list[str] = [
    """
    CREATE TABLE facts (
      id               TEXT PRIMARY KEY,            -- ULID
      subject          TEXT NOT NULL,
      predicate        TEXT NOT NULL,
      object           TEXT NOT NULL,
      -- Temporal validity
      valid_from       INTEGER NOT NULL,
      valid_until      INTEGER,                     -- NULL = encore valide
      -- Provenance
      confidence       REAL NOT NULL DEFAULT 1.0,
      source_episodes  TEXT NOT NULL DEFAULT '[]',  -- JSON array of episode IDs
      -- Versioning
      superseded_by    TEXT,                        -- FK -> facts.id
      created_at       INTEGER NOT NULL,
      FOREIGN KEY (superseded_by) REFERENCES facts(id)
    ) STRICT
    """,
    "CREATE INDEX idx_facts_subject ON facts(subject, predicate, valid_until)",
    "CREATE INDEX idx_facts_current ON facts(valid_until) WHERE valid_until IS NULL",
    "CREATE INDEX idx_facts_superseded ON facts(superseded_by)",
    """
    CREATE TABLE entities (
      canonical_name   TEXT PRIMARY KEY,
      aliases          TEXT NOT NULL DEFAULT '[]',  -- JSON array
      entity_type      TEXT,                        -- person|org|place|concept|product|null
      first_seen       INTEGER NOT NULL,
      last_seen        INTEGER NOT NULL,
      episode_count    INTEGER NOT NULL DEFAULT 0
    ) STRICT
    """,
    "CREATE INDEX idx_entities_last_seen ON entities(last_seen DESC)",
    """
    CREATE VIRTUAL TABLE facts_vec USING vec0(
      fact_id          TEXT PRIMARY KEY,
      embedding        FLOAT[1024] distance_metric=cosine
    )
    """,
]

SEMANTIC_SCHEMA_DROP_SQL: list[str] = [
    "DROP TABLE IF EXISTS facts_vec",
    "DROP TABLE IF EXISTS entities",
    "DROP TABLE IF EXISTS facts",
]


class Fact(Base):
    __tablename__ = "facts"

    id: Mapped[str] = mapped_column(primary_key=True)
    subject: Mapped[str]
    predicate: Mapped[str]
    object: Mapped[str]
    valid_from: Mapped[int]
    valid_until: Mapped[int | None]
    confidence: Mapped[float] = mapped_column(default=1.0)
    source_episodes: Mapped[str] = mapped_column(default="[]")
    superseded_by: Mapped[str | None]
    created_at: Mapped[int]


class Entity(Base):
    __tablename__ = "entities"

    canonical_name: Mapped[str] = mapped_column(primary_key=True)
    aliases: Mapped[str] = mapped_column(default="[]")
    entity_type: Mapped[str | None]
    first_seen: Mapped[int]
    last_seen: Mapped[int]
    episode_count: Mapped[int] = mapped_column(default=0)

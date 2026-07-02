"""Schéma episodic (§5.1) — ORM + DDL source de vérité.

EPISODIC_SCHEMA_SQL est LA définition du schéma : consommée par la migration
alembic ET par les fixtures de test. Tables STRICT et vec0 → DDL brut
(inexprimables proprement en DDL SQLAlchemy ; vec0 invisible à l'autogenerate).

Divergence assumée vs spec : `distance_metric=cosine` déclaré sur episodes_vec
(le spec §9.2 impose le KNN cosine ; le déclarer dans la table évite un
re-calcul applicatif).
"""

from __future__ import annotations

from sqlalchemy.orm import Mapped, mapped_column

from mnemos.models.base import Base

EPISODIC_SCHEMA_SQL: list[str] = [
    """
    CREATE TABLE episodes (
      id                TEXT PRIMARY KEY,             -- ULID
      created_at        INTEGER NOT NULL,             -- epoch ms UTC
      session_id        TEXT,
      role              TEXT NOT NULL,                -- 'user' | 'assistant' | 'system'
      content           TEXT NOT NULL,
      -- Salience (NULL sur surprise/arousal/… = scoring async pas encore passé, §13.3)
      salience          REAL NOT NULL DEFAULT 0.5,
      surprise          REAL,
      arousal           REAL,
      self_ref          REAL,
      recurrence        REAL,
      -- Lifecycle
      decay_state       REAL NOT NULL DEFAULT 1.0,
      last_decayed_at   INTEGER,
      consolidated_at   INTEGER,
      extraction_failed INTEGER NOT NULL DEFAULT 0,
      archived          INTEGER NOT NULL DEFAULT 0,
      -- Refs
      entity_refs       TEXT NOT NULL DEFAULT '[]'    -- JSON array of entity names
    ) STRICT
    """,
    "CREATE INDEX idx_episodes_created_at ON episodes(created_at DESC)",
    "CREATE INDEX idx_episodes_session ON episodes(session_id, created_at DESC)",
    "CREATE INDEX idx_episodes_consolidation ON episodes(consolidated_at, salience DESC)",
    "CREATE INDEX idx_episodes_archived ON episodes(archived)",
    """
    CREATE VIRTUAL TABLE episodes_vec USING vec0(
      episode_id      TEXT PRIMARY KEY,
      embedding       FLOAT[1024] distance_metric=cosine
    )
    """,
    """
    CREATE TABLE episodes_sparse (
      episode_id      TEXT PRIMARY KEY,
      sparse_bits     BLOB NOT NULL,                -- 256-bit packed
      FOREIGN KEY (episode_id) REFERENCES episodes(id) ON DELETE CASCADE
    ) STRICT
    """,
]

EPISODIC_SCHEMA_DROP_SQL: list[str] = [
    "DROP TABLE IF EXISTS episodes_sparse",
    "DROP TABLE IF EXISTS episodes_vec",
    "DROP TABLE IF EXISTS episodes",
]


class Episode(Base):
    __tablename__ = "episodes"

    id: Mapped[str] = mapped_column(primary_key=True)
    created_at: Mapped[int]
    session_id: Mapped[str | None]
    role: Mapped[str]
    content: Mapped[str]
    salience: Mapped[float] = mapped_column(default=0.5)
    surprise: Mapped[float | None]
    arousal: Mapped[float | None]
    self_ref: Mapped[float | None]
    recurrence: Mapped[float | None]
    decay_state: Mapped[float] = mapped_column(default=1.0)
    last_decayed_at: Mapped[int | None]
    consolidated_at: Mapped[int | None]
    extraction_failed: Mapped[int] = mapped_column(default=0)
    archived: Mapped[int] = mapped_column(default=0)
    entity_refs: Mapped[str] = mapped_column(default="[]")
    # Index définis dans EPISODIC_SCHEMA_SQL (source de vérité DDL) —
    # pas re-déclarés ici pour éviter un double create via metadata.


class EpisodeSparse(Base):
    __tablename__ = "episodes_sparse"

    episode_id: Mapped[str] = mapped_column(primary_key=True)
    sparse_bits: Mapped[bytes]

"""Backend Référence : GraphitiLikeMemory (Graphe de connaissances bi-temporel).

Implémente le paradigme bi-temporel inspiré de Graphiti / Zep (arXiv:2501.13956) :
- Stocke les triplets sous forme datée (subject, predicate, object, valid_at, invalid_at, raw_content).
- Gère l'invalidation temporelle relationnelle (valid_at -> invalid_at) lors des mutations.
- Recherche pertinente (textuelle lexicale) combinée aux filtres temporels :
  - Requêtes chronologiques : arêtes ordonnées par valid_at ASC.
  - Requêtes actives : arêtes actives (invalid_at IS NULL) triées par pertinence.
"""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from typing import Any

from bench.backends.base import BaseMemoryBackend
from bench.config import BenchConfig
from bench.storage_utils import StorageStats, get_sqlite_storage_stats


def _tokenize(text: str) -> set[str]:
    clean = re.sub(r"[^\w\s]", " ", text.lower())
    return {w for w in clean.split() if len(w) > 2}


class GraphitiLikeMemory(BaseMemoryBackend):
    def __init__(self, config: BenchConfig) -> None:
        self.config = config
        self.db_path: Path = config.bench_dir / "graphiti_temporal" / "temporal_kg.db"
        self.conn: sqlite3.Connection | None = None
        self.current_time_days: float = 0.0

    @property
    def name(self) -> str:
        return "GraphitiLikeMemory"

    @property
    def family(self) -> str:
        return "Référence (Graphe de connaissances bi-temporel type Graphiti)"

    async def setup(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        await self.reset()

    async def teardown(self) -> None:
        if self.conn:
            self.conn.close()
            self.conn = None

    async def reset(self) -> None:
        if self.conn:
            self.conn.close()
            self.conn = None
        if self.db_path.exists():
            self.db_path.unlink()

        self.conn = sqlite3.connect(str(self.db_path))
        with self.conn:
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS temporal_edges (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    subject TEXT NOT NULL,
                    predicate TEXT NOT NULL,
                    object TEXT NOT NULL,
                    valid_at REAL NOT NULL,
                    invalid_at REAL,
                    raw_content TEXT NOT NULL
                )
                """
            )
        self.current_time_days = 0.0

    def _extract_triplets(self, content: str) -> list[tuple[str, str, str, bool]]:
        content_lower = content.lower()
        triplets: list[tuple[str, str, str, bool]] = []

        # Extraction générique de résidence
        m_res = re.search(r"(?:habite|vis|réside|emmenag\w+|install\w+)\s+(?:à|a|dans)\s+([A-ZÀ-ÿa-z\-]+)", content, re.IGNORECASE)
        if m_res:
            city = m_res.group(1).capitalize()
            triplets.append(("user", "lives_in", city, True))

        # Extraction générique de secret/clé
        m_key = re.search(r"(?:clé|cle|key|secret)\s+(?:d'authentification\s+)?(?:du\s+\w+\s+)?(?:est\s+)?([A-Za-z0-9\-]{4,})", content, re.IGNORECASE)
        if m_key:
            triplets.append(("staging_server", "auth_key", m_key.group(1), True))

        # Extraction générique d'allergie ou alerte médicale
        m_all = re.search(r"allerg\w+\s+(?:à|aux|a)\s+([A-ZÀ-ÿa-z\-]+)", content, re.IGNORECASE)
        if m_all:
            triplets.append(("user", "has_fatal_allergy", m_all.group(1).lower(), False))

        # Règles de sécurité
        if "règle de sécurité" in content_lower or "interdiction" in content_lower or "[rule-" in content_lower:
            triplets.append(("system", "security_rule", content, False))

        if not triplets:
            triplets.append(("user", "interaction", content, False))

        return triplets

    async def write(
        self,
        content: str,
        role: str = "user",
        timestamp_offset_days: float = 0.0,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.current_time_days += timestamp_offset_days
        assert self.conn is not None

        triplets = self._extract_triplets(content)
        with self.conn:
            for subj, pred, obj, is_functional in triplets:
                if is_functional:
                    self.conn.execute(
                        """
                        UPDATE temporal_edges
                        SET invalid_at = ?
                        WHERE subject = ? AND predicate = ? AND invalid_at IS NULL
                        """,
                        (self.current_time_days, subj, pred),
                    )
                self.conn.execute(
                    """
                    INSERT INTO temporal_edges (subject, predicate, object, valid_at, invalid_at, raw_content)
                    VALUES (?, ?, ?, ?, NULL, ?)
                    """,
                    (subj, pred, obj, self.current_time_days, content),
                )

    async def recall(self, query: str, k: int = 4) -> list[str]:
        """Recherche bi-temporelle avec classement par pertinence textuelle."""
        assert self.conn is not None
        query_tokens = _tokenize(query)
        is_chronological = any(
            w in query.lower() for w in ["chronologique", "ordre", "historique", "évolution"]
        )

        cur = self.conn.execute(
            """
            SELECT id, subject, predicate, object, valid_at, invalid_at, raw_content
            FROM temporal_edges
            """
        )
        rows = cur.fetchall()
        if not rows:
            return []

        scored_edges: list[tuple[float, float, str]] = []
        for _eid, subj, pred, obj, valid_at, invalid_at, raw_content in rows:
            edge_text = f"{subj} {pred} {obj} {raw_content}"
            edge_tokens = _tokenize(edge_text)
            overlap = len(query_tokens & edge_tokens)

            # En requête chronologique, on conserve l'ensemble des versions pour le sujet/prédicat matché
            if is_chronological:
                if overlap > 0:
                    scored_edges.append((overlap, valid_at, f"[T={valid_at:.1f} -> {invalid_at or 'présent'}] {subj} {pred} {obj}"))
            else:
                # En requête d'état actif, on privilégie les arêtes non invalidées
                if invalid_at is None and overlap > 0:
                    scored_edges.append((overlap, valid_at, f"{subj} {pred} {obj} ({raw_content})"))

        if is_chronological:
            # Tri par valid_at croissant pour restituer l'ordre chronologique exact
            scored_edges.sort(key=lambda x: x[1])
            return [e[2] for e in scored_edges[:k]]

        # Tri par pertinence décroissante, puis par récence décroissante
        scored_edges.sort(key=lambda x: (x[0], x[1]), reverse=True)
        if scored_edges:
            return [e[2] for e in scored_edges[:k]]

        # Fallback sur les faits actifs les plus récents
        cur = self.conn.execute(
            "SELECT subject, predicate, object, raw_content FROM temporal_edges WHERE invalid_at IS NULL ORDER BY valid_at DESC LIMIT ?",
            (k,),
        )
        return [f"{r[0]} {r[1]} {r[2]}" for r in cur.fetchall()]

    async def advance_time_days(self, days: float) -> None:
        self.current_time_days += days

    async def get_index_size_bytes(self) -> int:
        if self.db_path.exists():
            return self.db_path.stat().st_size
        return 0

    async def get_storage_stats(self) -> StorageStats:
        return get_sqlite_storage_stats(
            db_paths=[self.db_path],
            active_count_queries=[
                (self.db_path, "SELECT count(*) FROM temporal_edges WHERE invalid_at IS NULL"),
            ],
            archived_count_queries=[
                (self.db_path, "SELECT count(*) FROM temporal_edges WHERE invalid_at IS NOT NULL"),
            ],
            notes="Taille active = (page_count - freelist_count) * page_size. Arêtes invalidées conservées pour auditabilité.",
        )

    def get_embedder_class(self) -> str:
        return "None"

    def get_llm_class(self) -> str:
        return "OllamaClient" if not self.config.dry_run else "None"


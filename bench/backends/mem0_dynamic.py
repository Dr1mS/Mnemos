"""Backend Référence : Mem0LikeMemory (Vector-CRUD avec classification ADD/UPDATE/DELETE).

Implémente le paradigme de Mem0 (Vector-CRUD avec détection de sujet et mise à jour destructive) :
- À l'écriture, classifie l'action (ADD, UPDATE) selon le sujet.
- En cas d'UPDATE, met à jour l'enregistrement existant dans la table vectorielle.
- Recherche KNN par similarité cosinus avec l'embedding de la requête.
"""

from __future__ import annotations

import json
import math
import sqlite3
from pathlib import Path
from typing import Any

import httpx

from bench.backends.base import BaseMemoryBackend
from bench.config import BenchConfig, DeterministicFastEmbedder
from bench.storage_utils import StorageStats, get_sqlite_storage_stats


def _cosine_similarity(v1: list[float], v2: list[float]) -> float:
    dot = sum(a * b for a, b in zip(v1, v2, strict=False))
    norm1 = math.sqrt(sum(a * a for a in v1))
    norm2 = math.sqrt(sum(b * b for b in v2))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)


class Mem0LikeMemory(BaseMemoryBackend):
    def __init__(self, config: BenchConfig) -> None:
        self.config = config
        self.db_path: Path = config.bench_dir / "mem0_dynamic" / "mem0.db"
        self.fast_embedder = DeterministicFastEmbedder()
        self.conn: sqlite3.Connection | None = None
        self.current_time_days: float = 0.0

    @property
    def name(self) -> str:
        return "Mem0LikeMemory"

    @property
    def family(self) -> str:
        return "Référence (Vector-CRUD type Mem0)"

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
                CREATE TABLE IF NOT EXISTS mem0_facts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    topic TEXT NOT NULL,
                    fact_text TEXT NOT NULL,
                    embedding_json TEXT NOT NULL,
                    updated_at REAL NOT NULL
                )
                """
            )
        self.current_time_days = 0.0

    async def _embed(self, text: str) -> list[float]:
        if not self.config.dry_run:
            try:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    resp = await client.post(
                        f"{self.config.ollama_host.rstrip('/')}/api/embed",
                        json={"model": self.config.embed_model, "input": [text]},
                    )
                    if resp.status_code == 200:
                        embeddings = resp.json().get("embeddings", [])
                        if embeddings:
                            return [float(x) for x in embeddings[0]]
            except Exception:  # noqa: BLE001
                pass
        res: list[float] = await self.fast_embedder.embed(text)
        return res

    def _extract_topic(self, content: str) -> str | None:
        content_lower = content.lower()
        if any(w in content_lower for w in ["habite", "vis", "déménagé", "résidence"]):
            return "residence"
        if any(w in content_lower for w in ["serveur", "clé", "staging", "authentification"]):
            return "credentials"
        if any(w in content_lower for w in ["allerg", "medical", "sante", "alimentaire"]):
            return "health_safety"
        if any(w in content_lower for w in ["règle de sécurité", "interdiction", "ssl", "sql", "exec", "eval"]):
            return f"security_{content_lower[:20]}"
        return None

    async def write(
        self,
        content: str,
        role: str = "user",
        timestamp_offset_days: float = 0.0,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.current_time_days += timestamp_offset_days
        assert self.conn is not None

        topic = self._extract_topic(content)
        vec = await self._embed(content)
        vec_json = json.dumps(vec)

        with self.conn:
            if topic is not None:
                cur = self.conn.execute("SELECT id FROM mem0_facts WHERE topic = ?", (topic,))
                existing = cur.fetchone()
                if existing:
                    # Logique de mise à jour destructive type Mem0 (écrase le fait antérieur)
                    self.conn.execute(
                        "UPDATE mem0_facts SET fact_text = ?, embedding_json = ?, updated_at = ? WHERE id = ?",
                        (content, vec_json, self.current_time_days, existing[0]),
                    )
                    return

            self.conn.execute(
                "INSERT INTO mem0_facts (topic, fact_text, embedding_json, updated_at) VALUES (?, ?, ?, ?)",
                (topic or "general", content, vec_json, self.current_time_days),
            )

    async def recall(self, query: str, k: int = 4) -> list[str]:
        assert self.conn is not None
        q_vec = await self._embed(query)

        cur = self.conn.execute("SELECT id, topic, fact_text, embedding_json FROM mem0_facts")
        rows = cur.fetchall()
        if not rows:
            return []

        scored: list[tuple[float, str]] = []
        for _, _, fact_text, emb_str in rows:
            emb = json.loads(emb_str)
            sim = _cosine_similarity(q_vec, emb)
            scored.append((sim, fact_text))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [s[1] for s in scored[:k]]

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
                (self.db_path, "SELECT count(*) FROM mem0_facts"),
            ],
            notes="Taille active = (page_count - freelist_count) * page_size. Vecteurs stockés en JSON texte.",
        )

    def get_embedder_class(self) -> str:
        if self.config.dry_run:
            return "DeterministicFastEmbedder"
        return "OllamaEmbedderAPI"

    def get_llm_class(self) -> str:
        return "OllamaClient" if not self.config.dry_run else "None"


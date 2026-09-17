"""Backend Référence : GenerativeAgentsLikeMemory (Stanford Memory Stream).

Implémente le flux de mémoire inspiré des agents génératifs de Stanford (Park et al.) :
- Score de récupération combiné : Score = α·Récence + β·Importance + γ·Pertinence (Cosinus).
- Récence modélisée par une décroissance exponentielle e^(-lambda * delta_t).
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


class GenerativeAgentsLikeMemory(BaseMemoryBackend):
    def __init__(
        self,
        config: BenchConfig,
        w_recency: float = 0.2,
        w_importance: float = 0.3,
        w_relevance: float = 0.5,
        decay_lambda: float = 0.02,
    ) -> None:
        self.config = config
        self.w_recency = w_recency
        self.w_importance = w_importance
        self.w_relevance = w_relevance
        self.decay_lambda = decay_lambda
        self.db_path: Path = config.bench_dir / "generative_agents" / "stream.db"
        self.fast_embedder = DeterministicFastEmbedder()
        self.conn: sqlite3.Connection | None = None
        self.current_time_days: float = 0.0

    @property
    def name(self) -> str:
        return "GenerativeAgentsLikeMemory"

    @property
    def family(self) -> str:
        return "Référence (Flux de mémoire type Stanford Generative Agents)"

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
                CREATE TABLE IF NOT EXISTS memory_stream (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    content TEXT NOT NULL,
                    role TEXT NOT NULL,
                    importance REAL NOT NULL,
                    created_at_days REAL NOT NULL,
                    embedding_json TEXT NOT NULL
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

    def _rate_importance(self, content: str) -> float:
        content_lower = content.lower()
        if any(w in content_lower for w in ["clé", "sec-", "allerg", "mortel", "secret", "règle de sécurité", "interdiction"]):
            return 0.95
        if any(w in content_lower for w in ["habite", "déménagé", "vis", "adresse"]):
            return 0.85
        if any(w in content_lower for w in ["café", "soleil", "bureau", "souris", "ventilateur", "météo"]):
            return 0.15
        return 0.50

    async def write(
        self,
        content: str,
        role: str = "user",
        timestamp_offset_days: float = 0.0,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.current_time_days += timestamp_offset_days
        importance = self._rate_importance(content)
        vec = await self._embed(content)
        assert self.conn is not None

        with self.conn:
            self.conn.execute(
                """
                INSERT INTO memory_stream (content, role, importance, created_at_days, embedding_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                (content, role, importance, self.current_time_days, json.dumps(vec)),
            )

    async def recall(self, query: str, k: int = 4) -> list[str]:
        assert self.conn is not None
        q_vec = await self._embed(query)

        cur = self.conn.execute(
            "SELECT id, content, role, importance, created_at_days, embedding_json FROM memory_stream"
        )
        rows = cur.fetchall()
        if not rows:
            return []

        scored: list[tuple[float, str]] = []
        for _, content, role, importance, created_at_days, emb_str in rows:
            emb = json.loads(emb_str)
            relevance = _cosine_similarity(q_vec, emb)
            age_days = max(0.0, self.current_time_days - created_at_days)
            recency = math.exp(-self.decay_lambda * age_days)

            score = (
                self.w_recency * recency
                + self.w_importance * importance
                + self.w_relevance * relevance
            )
            scored.append((score, f"{role}: {content}"))

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
                (self.db_path, "SELECT count(*) FROM memory_stream"),
            ],
            notes="Taille active = (page_count - freelist_count) * page_size. Flux additif sans purge.",
        )

    def get_embedder_class(self) -> str:
        if self.config.dry_run:
            return "DeterministicFastEmbedder"
        return "OllamaEmbedderAPI"

    def get_llm_class(self) -> str:
        return "None"


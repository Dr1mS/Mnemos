"""Backend Baseline R3 : NaiveVectorMemory (Standard Vector RAG / ChromaDB / Mem0 baseline).

Stocke chaque tour brut dans une base SQLite vectorielle avec embedding dense (bge-m3).
Recherche par similarité cosinus KNN brute (top-k = 4).
Ne possède aucun mécanisme d'invalidation temporelle, de versioning ni de filtrage de saillance.
Illustre parfaitement l'effet fantôme (Ghost Vector Effect).
"""

from __future__ import annotations

import math
import sqlite3
from pathlib import Path
from typing import Any

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


class NaiveVectorMemory(BaseMemoryBackend):
    def __init__(self, config: BenchConfig) -> None:
        self.config = config
        self.db_path: Path = config.bench_dir / "naive_vector" / "naive_vector.db"
        self.fast_embedder = DeterministicFastEmbedder()
        self.conn: sqlite3.Connection | None = None
        self._current_time_days: float = 0.0

    @property
    def name(self) -> str:
        return "NaiveVectorMemory"

    @property
    def family(self) -> str:
        return "Baseline R3 (RAG Vectoriel Dense Standard)"

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
                CREATE TABLE IF NOT EXISTS naive_episodes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    time_days REAL NOT NULL,
                    embedding_json TEXT NOT NULL
                )
                """
            )
        self._current_time_days = 0.0

    async def _embed(self, text: str) -> list[float]:
        if self.config.dry_run:
            return await self.fast_embedder.embed(text)
        try:
            import httpx

            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(
                    f"{self.config.ollama_host.rstrip('/')}/api/embed",
                    json={"model": self.config.embed_model, "input": [text]},
                )
                if resp.status_code == 200:
                    data = resp.json()
                    embeddings: list[list[float]] = data.get("embeddings", [])
                    if embeddings:
                        return embeddings[0]
        except Exception:  # noqa: BLE001
            pass
        return await self.fast_embedder.embed(text)

    async def write(
        self,
        content: str,
        role: str = "user",
        timestamp_offset_days: float = 0.0,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._current_time_days += timestamp_offset_days
        vec = await self._embed(content)
        import json

        assert self.conn is not None
        with self.conn:
            self.conn.execute(
                "INSERT INTO naive_episodes (role, content, time_days, embedding_json) VALUES (?, ?, ?, ?)",
                (role, content, self._current_time_days, json.dumps(vec)),
            )

    async def recall(self, query: str, k: int = 4) -> list[str]:
        q_vec = await self._embed(query)
        assert self.conn is not None
        import json

        cursor = self.conn.execute("SELECT id, role, content, embedding_json FROM naive_episodes")
        rows = cursor.fetchall()
        if not rows:
            return []

        scored: list[tuple[float, str]] = []
        for _, role, content, emb_str in rows:
            emb = json.loads(emb_str)
            sim = _cosine_similarity(q_vec, emb)
            scored.append((sim, f"{role}: {content}"))

        # Tri décroissant selon similarité cosinus (KNN brut)
        scored.sort(key=lambda x: x[0], reverse=True)
        return [s[1] for s in scored[:k]]

    async def advance_time_days(self, days: float) -> None:
        self._current_time_days += days

    async def get_index_size_bytes(self) -> int:
        if self.db_path.exists():
            return self.db_path.stat().st_size
        return 0

    async def get_storage_stats(self) -> StorageStats:
        return get_sqlite_storage_stats(
            db_paths=[self.db_path],
            active_count_queries=[
                (self.db_path, "SELECT count(*) FROM naive_episodes"),
            ],
            notes="Taille active = (page_count - freelist_count) * page_size. Vecteurs stockés en JSON texte sans purge.",
        )

    def get_embedder_class(self) -> str:
        if self.config.dry_run:
            return "DeterministicFastEmbedder"
        return "OllamaEmbedderAPI"

    def get_llm_class(self) -> str:
        return "None"


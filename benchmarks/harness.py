"""Harnais d'exécution des benchmarks pour Mnemos.

Fournit l'isolation complète des données (dossier data/bench/), la gestion des
moteurs mock et réels (Ollama), le suivi précis des métriques de latence
et l'analyse des tailles disque SQLite.
"""

from __future__ import annotations

import json
import math
import shutil
from dataclasses import dataclass
from hashlib import blake2b
from pathlib import Path
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from benchmarks.config import BenchConfig
from mnemos.clock import Clock
from mnemos.config import Settings
from mnemos.embeddings.dense import DenseEmbedder
from mnemos.llm.model_manager import ModelManager
from mnemos.llm.ollama_client import OllamaClient
from mnemos.models.base import make_async_engine
from mnemos.models.episodic import EPISODIC_SCHEMA_SQL
from mnemos.models.semantic import SEMANTIC_SCHEMA_SQL
from mnemos.stores.episodic import EpisodicStore
from mnemos.stores.semantic import SemanticStore


@dataclass
class LatencyStats:
    count: int = 0
    total_time: float = 0.0
    throughput_ops_sec: float = 0.0
    mean_ms: float = 0.0
    min_ms: float = 0.0
    p50_ms: float = 0.0
    p90_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0
    max_ms: float = 0.0


class LatencyTracker:
    def __init__(self, name: str = "metric") -> None:
        self.name = name
        self.durations: list[float] = []

    def record(self, duration_sec: float) -> None:
        self.durations.append(duration_sec)

    def compute(self, total_wall_time: float | None = None) -> LatencyStats:
        if not self.durations:
            return LatencyStats()
        sorted_d = sorted(self.durations)
        count = len(sorted_d)
        total_time = sum(sorted_d)
        wall_time = total_wall_time if total_wall_time is not None else total_time
        throughput = count / wall_time if wall_time > 0 else 0.0

        def percentile(p: float) -> float:
            idx = int(math.ceil(p * count)) - 1
            return max(0.0, sorted_d[min(max(idx, 0), count - 1)] * 1000.0)

        return LatencyStats(
            count=count,
            total_time=total_time,
            throughput_ops_sec=throughput,
            mean_ms=(total_time / count) * 1000.0,
            min_ms=sorted_d[0] * 1000.0,
            p50_ms=percentile(0.50),
            p90_ms=percentile(0.90),
            p95_ms=percentile(0.95),
            p99_ms=percentile(0.99),
            max_ms=sorted_d[-1] * 1000.0,
        )


class DeterministicFastEmbedder:
    """Embedder synthétique ultra-rapide 1024-dim normalisé (L2 = 1.0).

    Dérive un vecteur déterministe via blake2b + harmoniques pseudo-aléatoires.
    Permet de tester des milliers d'insertions et de requêtes sqlite-vec sans
    dépendre du débit GPU/Ollama.
    """

    def __init__(self, dim: int = 1024) -> None:
        self.dim = dim

    async def embed(self, text_input: str) -> list[float]:
        # Hash 32 bytes
        h = blake2b(text_input.encode("utf-8"), digest_size=32).digest()
        # Déploiement en 1024 floats via fonction sinusoïdale déterministe
        raw: list[float] = []
        for i in range(self.dim):
            byte_val = h[i % 32]
            val = math.sin((byte_val + 1) * (i + 1) * 0.1)
            raw.append(val)
        # Normalisation L2
        norm = math.sqrt(sum(x * x for x in raw))
        if norm == 0:
            return [1.0 / math.sqrt(self.dim)] * self.dim
        return [x / norm for x in raw]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [await self.embed(t) for t in texts]


class MockFastLLMManager:
    """LLM Manager instantané simulant la réponse de qwen3:4b."""

    async def generate(self, prompt: str, model: str, **kwargs: Any) -> str:
        # Si c'est un prompt de saillance
        if "surprise" in prompt or "arousal" in prompt or "salience" in prompt.lower():
            # Simule une note de saillance basée sur des mots-clés
            p_lower = prompt.lower()
            if any(w in p_lower for w in ["important", "critique", "urgence", "secret", "mort", "déménage"]):
                return json.dumps({"surprise": 0.85, "arousal": 0.8, "self_ref": 0.9, "recurrence": 0.1})
            return json.dumps({"surprise": 0.2, "arousal": 0.1, "self_ref": 0.3, "recurrence": 0.0})

        # Si c'est un prompt d'extraction de faits
        if "facts" in prompt or "entities" in prompt or "triplets" in prompt:
            return json.dumps({
                "entities": [{"name": "User", "entity_type": "person", "aliases": ["moi"]}],
                "facts": [
                    {
                        "subject": "User",
                        "predicate": "lives_in",
                        "object": "Annecy",
                        "confidence": 0.95,
                    }
                ],
            })

        return "{}"

    async def health_check(self) -> bool:
        return True

    async def embed_probe(self) -> str | None:
        return None


@dataclass
class DbDiskSizes:
    episodic_bytes: int = 0
    episodic_wal_bytes: int = 0
    semantic_bytes: int = 0
    semantic_wal_bytes: int = 0
    total_bytes: int = 0
    episodes_count: int = 0
    facts_count: int = 0


class BenchmarkHarness:
    def __init__(self, config: BenchConfig) -> None:
        self.config = config
        self.epi_engine: AsyncEngine | None = None
        self.sem_engine: AsyncEngine | None = None
        self.episodic_store: EpisodicStore | None = None
        self.semantic_store: SemanticStore | None = None
        self.settings: Settings | None = None
        self.clock: Clock = Clock()
        self.embedder: Any = None
        self.llm_manager: Any = None

    async def setup(self, reset: bool = True, custom_clock: Clock | None = None) -> None:
        """Initialise l'environnement de benchmark isolé."""
        if reset and self.config.bench_dir.exists():
            shutil.rmtree(self.config.bench_dir)
        self.config.bench_dir.mkdir(parents=True, exist_ok=True)
        self.config.procedural_dir.mkdir(parents=True, exist_ok=True)

        self.clock = custom_clock or Clock()
        self.settings = Settings(
            _env_file=None,
            DATA_DIR=self.config.bench_dir,
            EPISODIC_DB=self.config.episodic_db,
            SEMANTIC_DB=self.config.semantic_db,
            PROCEDURAL_DIR=self.config.procedural_dir,
            OLLAMA_HOST=self.config.ollama_host,
            EMBED_MODEL=self.config.embed_model,
            SALIENCE_MODEL=self.config.llm_model,
            EXTRACTION_MODEL=self.config.llm_model,
            SALIENCE_QUEUE_MAXSIZE=1000,
            SALIENCE_QUEUE_WORKERS=2,
        )

        self.epi_engine = make_async_engine(self.config.episodic_db)
        self.sem_engine = make_async_engine(self.config.semantic_db)

        # Création des schémas DDL
        async with self.epi_engine.begin() as conn:
            for stmt in EPISODIC_SCHEMA_SQL:
                await conn.execute(text(stmt))

        async with self.sem_engine.begin() as conn:
            for stmt in SEMANTIC_SCHEMA_SQL:
                await conn.execute(text(stmt))

        # Initialisation de l'embedder et LLM selon le mode
        if self.config.mode == "real":
            client = OllamaClient(self.settings)
            self.llm_manager = ModelManager(self.settings, client)
            self.embedder = DenseEmbedder(self.llm_manager, self.settings)
        else:
            self.llm_manager = MockFastLLMManager()
            self.embedder = DeterministicFastEmbedder()

        self.episodic_store = EpisodicStore(
            self.epi_engine, self.embedder, self.clock, self.settings
        )
        self.semantic_store = SemanticStore(
            self.sem_engine, self.embedder, self.clock, self.settings
        )

    async def get_db_stats(self) -> DbDiskSizes:
        """Mesure précise de l'empreinte disque et du nombre de lignes."""
        sizes = DbDiskSizes()
        if self.config.episodic_db.exists():
            sizes.episodic_bytes = self.config.episodic_db.stat().st_size
        epi_wal = Path(f"{self.config.episodic_db}-wal")
        if epi_wal.exists():
            sizes.episodic_wal_bytes = epi_wal.stat().st_size

        if self.config.semantic_db.exists():
            sizes.semantic_bytes = self.config.semantic_db.stat().st_size
        sem_wal = Path(f"{self.config.semantic_db}-wal")
        if sem_wal.exists():
            sizes.semantic_wal_bytes = sem_wal.stat().st_size

        sizes.total_bytes = (
            sizes.episodic_bytes
            + sizes.episodic_wal_bytes
            + sizes.semantic_bytes
            + sizes.semantic_wal_bytes
        )

        if self.epi_engine:
            async with self.epi_engine.connect() as conn:
                res = await conn.execute(text("SELECT COUNT(*) FROM episodes"))
                sizes.episodes_count = int(res.scalar() or 0)

        if self.sem_engine:
            async with self.sem_engine.connect() as conn:
                res = await conn.execute(text("SELECT COUNT(*) FROM facts"))
                sizes.facts_count = int(res.scalar() or 0)

        return sizes

    async def teardown(self, clean_files: bool = False) -> None:
        """Ferme proprement les connexions SQLAlchemy."""
        if self.epi_engine:
            await self.epi_engine.dispose()
        if self.sem_engine:
            await self.sem_engine.dispose()
        if clean_files and self.config.bench_dir.exists():
            shutil.rmtree(self.config.bench_dir)

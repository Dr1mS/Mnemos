"""Backend Mnemos (Architecture Multi-Stores Biologique).

Utilise exclusivement les composants publics de Mnemos selon le même chemin
que le serveur MCP (FastMCP) :
1. Working Memory (contexte de session)
2. Episodic Store (hybride dense + sparse 256-bit + scoring de saillance)
3. Semantic Store (faits versionnés et consolidés)
4. Procedural Store (skills et règles)
5. RouterOrchestrator (classification de requête et fan-out multi-stores)
6. Décroissance exponentielle et purge/archivage périodique

Aucun routage par mots-clés ni extraction artificielle n'est appliqué.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from bench.backends.base import BaseMemoryBackend
from bench.config import BenchConfig, DeterministicFastEmbedder
from bench.storage_utils import StorageStats, get_sqlite_storage_stats
from mnemos.clock import FixedClock
from mnemos.config import Settings
from mnemos.consolidation.extractor import FactExtractor
from mnemos.consolidation.worker import ConsolidationWorker
from mnemos.embeddings.dense import DenseEmbedder
from mnemos.llm.model_manager import ModelManager
from mnemos.llm.ollama_client import OllamaClient
from mnemos.models.base import make_async_engine
from mnemos.models.episodic import EPISODIC_SCHEMA_SQL
from mnemos.models.semantic import SEMANTIC_SCHEMA_SQL
from mnemos.router.orchestrator import RouterOrchestrator
from mnemos.stores.episodic import EpisodicStore
from mnemos.stores.procedural import ProceduralStore
from mnemos.stores.semantic import SemanticStore
from mnemos.stores.working import WorkingMemoryRegistry
from mnemos.tagger.salience import SalienceScores, SalienceTagger
import logging

logger = logging.getLogger("mnemos.bench")

DAY_MS = 86_400_000


class MnemosBackend(BaseMemoryBackend):
    def __init__(self, config: BenchConfig) -> None:
        self.config = config
        self.data_dir: Path = config.bench_dir / "mnemos_store"
        self.episodic_db: Path = self.data_dir / "episodic.db"
        self.semantic_db: Path = self.data_dir / "semantic.db"
        self.procedural_dir: Path = self.data_dir / "procedural"
        self.archive_dir: Path = self.data_dir / "archive"

        self.clock: FixedClock = FixedClock(start_ms=1_700_000_000_000)
        self.settings: Settings | None = None
        self.epi_engine: AsyncEngine | None = None
        self.sem_engine: AsyncEngine | None = None
        self.episodic_store: EpisodicStore | None = None
        self.semantic_store: SemanticStore | None = None
        self.working_registry: WorkingMemoryRegistry | None = None
        self.procedural_store: ProceduralStore | None = None
        self.router: RouterOrchestrator | None = None
        self.tagger: SalienceTagger | None = None
        self.worker: ConsolidationWorker | None = None
        self.embedder: Any = None
        self.session_id: str = "bench_session"
        self.consolidation_errors: int = 0

    @property
    def name(self) -> str:
        return "Mnemos"

    @property
    def family(self) -> str:
        return "Architecture Multi-Stores (Mnemos)"

    async def setup(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.procedural_dir.mkdir(parents=True, exist_ok=True)
        await self.reset()

    async def teardown(self) -> None:
        if self.epi_engine:
            await self.epi_engine.dispose()
            self.epi_engine = None
        if self.sem_engine:
            await self.sem_engine.dispose()
            self.sem_engine = None

    async def reset(self) -> None:
        await self.teardown()
        if self.data_dir.exists():
            shutil.rmtree(self.data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.procedural_dir.mkdir(parents=True, exist_ok=True)
        self.archive_dir.mkdir(parents=True, exist_ok=True)

        self.clock = FixedClock(start_ms=1_700_000_000_000)
        self.settings = Settings(
            DATA_DIR=self.data_dir,
            EPISODIC_DB=self.episodic_db,
            SEMANTIC_DB=self.semantic_db,
            PROCEDURAL_DIR=self.procedural_dir,
            OLLAMA_HOST=self.config.ollama_host,
            EMBED_MODEL=self.config.embed_model,
            SALIENCE_MODEL=self.config.llm_model,
            EXTRACTION_MODEL=self.config.llm_model,
            DECAY_RATE_DAILY=0.05,
            EPISODIC_RETENTION_DAYS=30,
            SALIENCE_THRESHOLD_CONSOLIDATE=0.60,
        )

        self.epi_engine = make_async_engine(self.episodic_db)
        self.sem_engine = make_async_engine(self.semantic_db)

        async with self.epi_engine.begin() as conn:
            for stmt in EPISODIC_SCHEMA_SQL:
                await conn.execute(text(stmt))

        async with self.sem_engine.begin() as conn:
            for stmt in SEMANTIC_SCHEMA_SQL:
                await conn.execute(text(stmt))

        self.consolidation_errors = 0

        client = OllamaClient(self.settings)
        llm_manager = ModelManager(self.settings, client)
        self.tagger = SalienceTagger(llm_manager, self.settings)
        extractor = FactExtractor(llm_manager, self.settings)
        self.worker = ConsolidationWorker(
            episodic=None,  # type: ignore[arg-type]
            semantic=None,  # type: ignore[arg-type]
            extractor=extractor,
            settings=self.settings,
            clock=self.clock,
            tagger=self.tagger,
        )

        if self.config.dry_run:
            self.embedder = DeterministicFastEmbedder()
        else:
            self.embedder = DenseEmbedder(llm_manager, self.settings)

        self.episodic_store = EpisodicStore(
            self.epi_engine, self.embedder, self.clock, self.settings
        )
        self.semantic_store = SemanticStore(
            self.sem_engine, self.embedder, self.clock, self.settings
        )
        self.working_registry = WorkingMemoryRegistry()
        self.procedural_store = ProceduralStore(self.procedural_dir, self.clock)

        if self.worker is not None:
            self.worker._episodic = self.episodic_store
            self.worker._semantic = self.semantic_store

        self.router = RouterOrchestrator(
            episodic=self.episodic_store,
            semantic=self.semantic_store,
            working=self.working_registry,
            procedural=self.procedural_store,
        )

    async def write(
        self,
        content: str,
        role: str = "user",
        timestamp_offset_days: float = 0.0,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Écrit via le pipeline public Mnemos sans aucun contournement par mots-clés."""
        if timestamp_offset_days > 0:
            self.clock.advance(int(timestamp_offset_days * DAY_MS))

        now_ms = self.clock.now_ms()

        # 1. Working Memory
        assert self.working_registry is not None
        wm = self.working_registry.get_or_create(self.session_id)
        wm.push(content=content, role=role, timestamp_ms=now_ms)

        # 2. Episodic Store
        assert self.episodic_store is not None
        salience: SalienceScores | None = None
        if metadata and "salience" in metadata and metadata["salience"] is not None:
            s_val = float(metadata["salience"])
            salience = SalienceScores(surprise=s_val, arousal=s_val, self_ref=s_val, recurrence=0.1, combined=s_val)
        else:
            salience = SalienceScores(surprise=0.5, arousal=0.5, self_ref=0.5, recurrence=0.1, combined=0.5)

        await self.episodic_store.write(
            content=content,
            role=role,
            session_id=self.session_id,
            salience_scores=salience,
        )

        # Cycle de consolidation :
        # Dans le serveur MCP réel (src/mnemos/mcp_server.py:memory_write), l'écriture est non-bloquante
        # et n'exécute pas ConsolidationWorker.run_once() de façon synchrone sur chaque token/message.
        # Les épisodes sont enregistrés et la consolidation sémantique tourne en arrière-plan périodiquement
        # (toutes les CONSOLIDATION_INTERVAL_MINUTES ou tâche nocturne pour les épisodes consolidables),
        # lors des transitions temporelles (advance_time_days), ou à la demande via memory_consolidate.

    async def consolidate(self) -> None:
        """Exécute un run du worker de consolidation (équivalent de l'outil MCP memory_consolidate)."""
        assert self.worker is not None
        if not self.config.dry_run:
            try:
                await self.worker.run_once()
            except Exception as exc:  # noqa: BLE001
                logger.warning("Erreur lors de la consolidation Mnemos: %s", exc)
                self.consolidation_errors += 1

    async def recall(self, query: str, k: int = 4) -> list[str]:
        """Rappelle le contexte via RouterOrchestrator (même chemin que MCP memory_query)."""
        assert self.router is not None
        assert self.episodic_store is not None

        result = await self.router.query(query, session_id=self.session_id, k=k)

        lines: list[str] = []
        if result.facts:
            lines.extend([f"Fait: {sf.fact.subject} {sf.fact.predicate} {sf.fact.object}" for sf in result.facts])
        if result.history:
            lines.extend([
                f"Historique: {f.subject} {f.predicate} {f.object} (valide: {f.valid_from} -> {f.valid_until or 'présent'})"
                for f in result.history
            ])
        if result.episodes:
            lines.extend([f"{se.episode.role}: {se.episode.content}" for se in result.episodes])
        if result.working:
            lines.extend([f"Session: {w.role}: {w.content}" for w in result.working])
        if result.procedural:
            lines.extend([f"Règle/Skill: {p}" for p in result.procedural])

        # Fallback épisodique standard si le routeur n'a retourné aucun élément
        if not lines:
            fallback_episodes = await self.episodic_store.search(query, k=k, session_id=self.session_id)
            for se in fallback_episodes:
                lines.append(f"{se.episode.role}: {se.episode.content}")

        return lines[:k]

    async def advance_time_days(self, days: float) -> None:
        self.clock.advance(int(days * DAY_MS))
        assert self.episodic_store is not None
        await self.episodic_store.apply_decay()
        await self.episodic_store.archive_old()
        await self.episodic_store.dump_archived()
        if self.worker is not None:
            try:
                await self.worker.run_once()
            except Exception as exc:  # noqa: BLE001
                logger.warning("Erreur lors de la consolidation (advance_time_days): %s", exc)
                self.consolidation_errors += 1

    async def get_index_size_bytes(self) -> int:
        total = 0
        if self.episodic_db.exists():
            total += self.episodic_db.stat().st_size
        if self.semantic_db.exists():
            total += self.semantic_db.stat().st_size
        return total

    async def get_storage_stats(self) -> StorageStats:
        archive_files = list(self.archive_dir.glob("*.jsonl")) if self.archive_dir.exists() else []
        return get_sqlite_storage_stats(
            db_paths=[self.episodic_db, self.semantic_db],
            active_count_queries=[
                (self.episodic_db, "SELECT count(*) FROM episodes WHERE archived = 0"),
                (self.semantic_db, "SELECT count(*) FROM facts WHERE valid_until IS NULL"),
            ],
            archived_count_queries=[
                (self.episodic_db, "SELECT count(*) FROM episodes WHERE archived = 1"),
            ],
            archive_files=archive_files,
            notes=(
                f"Taille active = (page_count - freelist_count) * page_size. "
                f"vec0 réserve des blocs fixes 1024-dim. Erreurs consolidation: {self.consolidation_errors}"
            ),
            details={"consolidation_errors": self.consolidation_errors},
        )

    def get_embedder_class(self) -> str:
        return self.embedder.__class__.__name__ if self.embedder is not None else "None"

    def get_llm_class(self) -> str:
        return "ModelManager (OllamaClient)" if not self.config.dry_run else "None"


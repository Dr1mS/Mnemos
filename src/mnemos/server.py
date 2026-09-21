"""FastAPI app (§16) — assemblage des composants + lifespan.

Les composants (store, tagger, queue…) sont construits au lifespan sauf
s'ils sont déjà posés sur app.state (injection de doubles par les tests).
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress

from fastapi import FastAPI

from mnemos import __version__
from mnemos.api.aml_routes import aml_router
from mnemos.api.routes import router, viz_router
from mnemos.clock import Clock
from mnemos.config import Settings, get_settings
from mnemos.consolidation.extractor import FactExtractor
from mnemos.consolidation.worker import ConsolidationWorker
from mnemos.embeddings.dense import DenseEmbedder
from mnemos.llm.model_manager import ModelManager
from mnemos.llm.ollama_client import OllamaClient
from mnemos.logging import configure_logging, get_logger
from mnemos.models.base import make_async_engine
from mnemos.router.orchestrator import RouterOrchestrator
from mnemos.stores.episodic import EpisodicStore
from mnemos.stores.procedural import ProceduralStore
from mnemos.stores.semantic import SemanticStore
from mnemos.stores.working import WorkingMemoryRegistry
from mnemos.tagger.salience import SalienceTagger, ScoringQueue

logger = get_logger(__name__)


async def _consolidation_loop(
    worker: ConsolidationWorker,
    interval_s: float,
) -> None:
    """Tâche d'arrière-plan exécutant périodiquement la consolidation cognitive."""
    logger.info("consolidation_loop_started", interval_s=interval_s)
    while True:
        try:
            await asyncio.sleep(interval_s)
            report = await worker.run_once()
            if report.candidates > 0:
                logger.info(
                    "consolidation_cycle_complete",
                    candidates=report.candidates,
                    facts_inserted=report.facts_inserted,
                    entities=report.entities_upserted,
                )
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.warning("consolidation_loop_error", error=str(e))
    logger.info("consolidation_loop_stopped")


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    state = app.state
    settings: Settings = state.settings
    owns_engine = not hasattr(state, "store")
    if not hasattr(state, "clock"):
        state.clock = Clock()
    if owns_engine:
        client = OllamaClient(settings)
        state.client = client
        state.manager = ModelManager(settings, client)
        embedder = DenseEmbedder(state.manager, settings)
        clock = state.clock
        state.engine = make_async_engine(settings.EPISODIC_DB)
        state.semantic_engine = make_async_engine(settings.SEMANTIC_DB)
        state.store = EpisodicStore(state.engine, embedder, clock, settings)
        state.semantic = SemanticStore(state.semantic_engine, embedder, clock, settings)
    if not hasattr(state, "wm"):
        state.wm = WorkingMemoryRegistry()
    if not hasattr(state, "procedural"):
        state.procedural = ProceduralStore(settings.PROCEDURAL_DIR, Clock())
    if not hasattr(state, "orchestrator"):
        state.orchestrator = RouterOrchestrator(
            state.store, state.semantic, state.wm, state.procedural
        )
    if not hasattr(state, "worker"):
        state.worker = ConsolidationWorker(
            state.store,
            state.semantic,
            FactExtractor(state.manager, settings),
            settings,
            Clock(),
            tagger=SalienceTagger(state.manager, settings),
        )
    if not hasattr(state, "queue"):
        tagger = SalienceTagger(state.manager, settings)
        state.queue = ScoringQueue(
            tagger,
            state.store,
            maxsize=settings.SALIENCE_QUEUE_MAXSIZE,
            workers=settings.SALIENCE_QUEUE_WORKERS,
        )
    consolidation_task: asyncio.Task[None] | None = None
    if settings.CONSOLIDATION_AUTO:
        consolidation_task = asyncio.create_task(
            _consolidation_loop(state.worker, settings.CONSOLIDATION_INTERVAL_SECONDS),
            name="consolidation-worker-loop",
        )
    await state.queue.start()
    logger.info("server_started", host=settings.API_HOST, port=settings.API_PORT)
    yield
    if consolidation_task is not None:
        consolidation_task.cancel()
        with suppress(asyncio.CancelledError):
            await consolidation_task
    await state.queue.stop()
    if owns_engine:
        await state.engine.dispose()
        await state.semantic_engine.dispose()
        await state.manager.aclose()  # client llama.cpp, s'il en possède un
        await state.client.aclose()
    logger.info("server_stopped")


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.LOG_LEVEL)
    app = FastAPI(title="mnemos", version=__version__, lifespan=_lifespan)
    app.state.settings = settings
    app.include_router(router)
    app.include_router(viz_router)
    app.include_router(aml_router)
    return app

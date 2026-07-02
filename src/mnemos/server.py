"""FastAPI app (§16) — assemblage des composants + lifespan.

Les composants (store, tagger, queue…) sont construits au lifespan sauf
s'ils sont déjà posés sur app.state (injection de doubles par les tests).
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from mnemos import __version__
from mnemos.api.routes import router
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


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    state = app.state
    settings: Settings = state.settings
    owns_engine = not hasattr(state, "store")
    if owns_engine:
        client = OllamaClient(settings)
        state.client = client
        state.manager = ModelManager(settings, client)
        embedder = DenseEmbedder(state.manager, settings)
        clock = Clock()
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
        state.queue = ScoringQueue(tagger, state.store)
    await state.queue.start()
    logger.info("server_started", host=settings.API_HOST, port=settings.API_PORT)
    yield
    await state.queue.stop()
    if owns_engine:
        await state.engine.dispose()
        await state.semantic_engine.dispose()
        await state.client.aclose()
    logger.info("server_stopped")


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.LOG_LEVEL)
    app = FastAPI(title="mnemos", version=__version__, lifespan=_lifespan)
    app.state.settings = settings
    app.include_router(router)
    return app

"""Fixtures communes.

Marqueur @pytest.mark.requires_ollama (§19.4) : skip auto si Ollama down.
"""

from __future__ import annotations

import json
from hashlib import blake2b
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from mnemos.clock import Clock, FixedClock
from mnemos.config import Settings

OLLAMA_HOST = "http://localhost:11434"


class StubEmbedder:
    """Vecteur 1024-dim déterministe dérivé du hash du texte — pas d'Ollama."""

    async def embed(self, content: str) -> list[float]:
        seed = blake2b(content.encode(), digest_size=8).digest()
        return ([(b / 255.0) - 0.5 for b in seed] * 128)[:1024]


class StubLLMManager:
    """Salience canned + health ok — pas d'Ollama."""

    async def generate(self, prompt: str, model: str, **kwargs: object) -> str:
        return json.dumps(
            {"surprise": 0.2, "arousal": 0.1, "self_ref": 0.8, "recurrence": 0.0}
        )

    async def health_check(self) -> bool:
        return True


async def make_stub_app(
    tmp_path: Path, settings: Settings | None = None, clock: Clock | None = None
) -> tuple[FastAPI, list[AsyncEngine]]:
    """App complète sur DB tmp avec stubs (embedder + LLM). Les engines
    retournés sont à disposer par l'appelant après le lifespan."""
    from mnemos.models.base import make_async_engine
    from mnemos.models.episodic import EPISODIC_SCHEMA_SQL
    from mnemos.models.semantic import SEMANTIC_SCHEMA_SQL
    from mnemos.server import create_app
    from mnemos.stores.episodic import EpisodicStore
    from mnemos.stores.semantic import SemanticStore
    from mnemos.tagger.salience import SalienceTagger, ScoringQueue

    settings = settings or Settings(
        _env_file=None,
        DATA_DIR=tmp_path,
        EPISODIC_DB=tmp_path / "episodic.db",
        SEMANTIC_DB=tmp_path / "semantic.db",
        PROCEDURAL_DIR=tmp_path / "procedural",
    )
    clock = clock or Clock()
    epi_engine = make_async_engine(settings.EPISODIC_DB)
    sem_engine = make_async_engine(settings.SEMANTIC_DB)
    async with epi_engine.begin() as conn:
        for stmt in EPISODIC_SCHEMA_SQL:
            await conn.execute(text(stmt))
    async with sem_engine.begin() as conn:
        for stmt in SEMANTIC_SCHEMA_SQL:
            await conn.execute(text(stmt))

    app = create_app(settings)
    embedder = StubEmbedder()
    manager = StubLLMManager()
    store = EpisodicStore(epi_engine, embedder, clock, settings)  # type: ignore[arg-type]
    app.state.engine = epi_engine
    app.state.semantic_engine = sem_engine
    app.state.manager = manager
    app.state.store = store
    app.state.semantic = SemanticStore(sem_engine, embedder, clock, settings)  # type: ignore[arg-type]
    app.state.queue = ScoringQueue(
        SalienceTagger(manager, settings), store  # type: ignore[arg-type]
    )
    return app, [epi_engine, sem_engine]


def _ollama_up() -> bool:
    try:
        return httpx.get(f"{OLLAMA_HOST}/api/version", timeout=2).status_code == 200
    except httpx.HTTPError:
        return False


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if _ollama_up():
        return
    skip = pytest.mark.skip(reason="Ollama down — test requires_ollama skippé")
    for item in items:
        if "requires_ollama" in item.keywords:
            item.add_marker(skip)


@pytest.fixture
def fixed_clock() -> FixedClock:
    # 2026-07-02T10:00:00Z
    return FixedClock(start_ms=1_782_727_200_000)

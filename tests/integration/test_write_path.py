"""Write path API (§18 Phase 3, §19.2).

Deux niveaux :
- sans Ollama : app stub (conftest.make_stub_app) — logique complète
  (201, salience null puis mise à jour, 422, 401, 404) ;
- avec Ollama (requires_ollama) : latence POST /v1/episodes p50 < 500 ms
  (bge-m3 réel) et salience réelle mise à jour après coup.
"""

from __future__ import annotations

import asyncio
import statistics
import time
from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import pytest
from sqlalchemy import text
from tests.conftest import make_stub_app

from mnemos.clock import Clock
from mnemos.config import Settings
from mnemos.embeddings.dense import DenseEmbedder
from mnemos.llm.model_manager import ModelManager
from mnemos.llm.ollama_client import OllamaClient
from mnemos.models.base import make_async_engine
from mnemos.models.episodic import EPISODIC_SCHEMA_SQL
from mnemos.models.semantic import SEMANTIC_SCHEMA_SQL
from mnemos.server import create_app
from mnemos.stores.episodic import EpisodicStore
from mnemos.stores.semantic import SemanticStore
from mnemos.tagger.salience import SalienceTagger, ScoringQueue


@pytest.fixture
async def stub_client(tmp_path: Path) -> AsyncIterator[httpx.AsyncClient]:
    app, engines = await make_stub_app(tmp_path)
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
            yield client
    for engine in engines:
        await engine.dispose()


async def test_post_episode_201_salience_null_puis_scoree(
    stub_client: httpx.AsyncClient,
) -> None:
    resp = await stub_client.post(
        "/v1/episodes", json={"content": "je suis data engineer", "role": "user"}
    )
    assert resp.status_code == 201
    body = resp.json()
    assert body["salience"] is None  # §16.1 : pas encore scoré
    assert body["id"]

    # le scoring async passe (stub instantané) → salience renseignée
    for _ in range(50):
        got = await stub_client.get(f"/v1/episodes/{body['id']}")
        if got.json()["salience"] is not None:
            break
        await asyncio.sleep(0.05)
    assert got.json()["salience"] == 0.8  # boost-floor self_ref du stub


async def test_episode_cherchable_immediatement(stub_client: httpx.AsyncClient) -> None:
    """§13.3 : l'épisode est cherchable avant le scoring."""
    await stub_client.post(
        "/v1/episodes", json={"content": "le risotto aux cèpes", "role": "user"}
    )
    resp = await stub_client.get("/v1/episodes/search", params={"q": "le risotto aux cèpes"})
    assert resp.status_code == 200
    assert any("risotto" in r["episode"]["content"] for r in resp.json())


async def test_validation_422(stub_client: httpx.AsyncClient) -> None:
    assert (
        await stub_client.post("/v1/episodes", json={"content": "", "role": "user"})
    ).status_code == 422
    assert (
        await stub_client.post("/v1/episodes", json={"content": "x", "role": "robot"})
    ).status_code == 422
    assert (
        await stub_client.post(
            "/v1/episodes", json={"content": "x", "role": "user", "inconnu": 1}
        )
    ).status_code == 422


async def test_get_404(stub_client: httpx.AsyncClient) -> None:
    assert (await stub_client.get("/v1/episodes/inexistant")).status_code == 404


async def test_health(stub_client: httpx.AsyncClient) -> None:
    resp = await stub_client.get("/v1/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["ollama"] is True
    assert body["dbs"]["episodic"] is True


async def test_auth_api_key(tmp_path: Path) -> None:
    settings = Settings(
        _env_file=None,
        EPISODIC_DB=tmp_path / "episodic.db",
        SEMANTIC_DB=tmp_path / "semantic.db",
        API_KEY="secret",
    )
    app, engines = await make_stub_app(tmp_path, settings=settings)
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
            payload = {"content": "x", "role": "user"}
            assert (await client.post("/v1/episodes", json=payload)).status_code == 401
            ok = await client.post(
                "/v1/episodes", json=payload, headers={"X-API-Key": "secret"}
            )
            assert ok.status_code == 201
    for engine in engines:
        await engine.dispose()


# ── Avec Ollama réel : latence + salience réelle (§18 Phase 3) ────────────────


@pytest.mark.requires_ollama
async def test_write_path_reel_sous_500ms_et_salience_apres_coup(tmp_path: Path) -> None:
    settings = Settings(
        _env_file=None,
        EPISODIC_DB=tmp_path / "episodic.db",
        SEMANTIC_DB=tmp_path / "semantic.db",
    )
    epi_engine = make_async_engine(settings.EPISODIC_DB)
    sem_engine = make_async_engine(settings.SEMANTIC_DB)
    async with epi_engine.begin() as conn:
        for stmt in EPISODIC_SCHEMA_SQL:
            await conn.execute(text(stmt))
    async with sem_engine.begin() as conn:
        for stmt in SEMANTIC_SCHEMA_SQL:
            await conn.execute(text(stmt))
    app = create_app(settings)
    client_ollama = OllamaClient(settings)
    manager = ModelManager(settings, client_ollama)
    embedder = DenseEmbedder(manager, settings)
    clock = Clock()
    store = EpisodicStore(epi_engine, embedder, clock, settings)
    app.state.engine = epi_engine
    app.state.semantic_engine = sem_engine
    app.state.manager = manager
    app.state.store = store
    app.state.semantic = SemanticStore(sem_engine, embedder, clock, settings)
    app.state.queue = ScoringQueue(SalienceTagger(manager, settings), store)
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
            # Warm-up : charge bge-m3 (non compté)
            await client.post(
                "/v1/episodes", json={"content": "warmup embedding", "role": "user"}
            )
            latencies = []
            first_id: str | None = None
            for i in range(5):
                t0 = time.perf_counter()
                resp = await client.post(
                    "/v1/episodes",
                    json={
                        "content": f"Je bosse chez Datalyse comme data engineer ({i}).",
                        "role": "user",
                        "session_id": "latence",
                    },
                )
                latencies.append(time.perf_counter() - t0)
                assert resp.status_code == 201
                assert resp.json()["salience"] is None
                first_id = first_id or resp.json()["id"]

            p50_ms = statistics.median(latencies) * 1000
            assert p50_ms < 500, f"write path p50 = {p50_ms:.0f} ms (cible < 500)"

            # La salience réelle (qwen3:4b) arrive après coup — self_ref attendu haut
            assert first_id is not None
            salience: float | None = None
            for _ in range(120):  # jusqu'à 2 min sur CPU (5 jobs en file)
                got = await client.get(f"/v1/episodes/{first_id}")
                salience = got.json()["salience"]
                if salience is not None:
                    break
                await asyncio.sleep(1)
            assert salience is not None, "scoring async jamais passé"
            assert salience >= 0.5  # révélation perso → boost-floor self_ref
    await epi_engine.dispose()
    await sem_engine.dispose()
    await client_ollama.aclose()

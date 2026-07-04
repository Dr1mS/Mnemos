"""GET /v1/health (§Santé) — sonde DB + embedding, panne nommée précisément.

Le bug qui motive cet endpoint : une panne de /api/embed a rendu query ET
write inutilisables avec un message générique. /health doit distinguer une
panne embedding d'une panne DB d'une panne /api/version, et la nommer.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import pytest
from tests.conftest import make_stub_app


@pytest.fixture
async def app_and_client(
    tmp_path: Path,
) -> AsyncIterator[tuple[object, httpx.AsyncClient]]:
    app, engines = await make_stub_app(tmp_path)
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
            yield app, client
    for engine in engines:
        await engine.dispose()


async def test_health_tout_vert(
    app_and_client: tuple[object, httpx.AsyncClient],
) -> None:
    _, client = app_and_client
    body = (await client.get("/v1/health")).json()
    assert body["ok"] is True
    assert body["ollama"] is True
    assert body["embedding"] is True
    assert body["dbs"] == {"episodic": True, "semantic": True}
    assert body["failures"] == {}


async def test_health_panne_embedding_nommee(
    app_and_client: tuple[object, httpx.AsyncClient],
) -> None:
    """Ollama up (/api/version) mais /api/embed KO → ok=False, embedding=False,
    la panne est nommée. C'est exactement le cas qui cassait query+write."""
    app, client = app_and_client

    class EmbedDownManager:
        async def health_check(self) -> bool:
            return True  # /api/version répond

        async def embed_probe(self) -> str | None:
            return "/api/embed HTTP 404 (modèle bge-m3 ?)"

    app.state.manager = EmbedDownManager()  # type: ignore[attr-defined]
    body = (await client.get("/v1/health")).json()
    assert body["ok"] is False
    assert body["ollama"] is True  # version OK…
    assert body["embedding"] is False  # …mais embed KO
    assert "embedding" in body["failures"]
    assert "/api/embed" in body["failures"]["embedding"]


@pytest.mark.requires_ollama
async def test_embed_probe_reel_ollama_up() -> None:
    """Sonde /api/embed réelle contre Ollama : le modèle bge-m3 doit répondre
    (warm-up d'abord pour ne pas tomber sur le cold-start > 2 s)."""
    from mnemos.config import Settings
    from mnemos.llm.model_manager import ModelManager
    from mnemos.llm.ollama_client import OllamaClient

    settings = Settings(_env_file=None)
    client = OllamaClient(settings)
    manager = ModelManager(settings, client)
    try:
        # 1er appel : charge le modèle (peut dépasser 2 s → toléré ici).
        await client.embed_probe(settings.EMBED_MODEL)
        # 2e appel (warm) : doit répondre None (OK) sous le timeout court.
        assert await manager.embed_probe() is None
        # Modèle inexistant → panne nommée, jamais None.
        missing = await client.embed_probe("does-not-exist:99b")
        assert missing is not None and "/api/embed" in missing
    finally:
        await client.aclose()


async def test_health_panne_db_nommee(tmp_path: Path) -> None:
    """DB inaccessible → dbs.episodic=False, ok=False, panne nommée."""
    app, engines = await make_stub_app(tmp_path)
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
            # On casse l'engine épisodique en le disposant.
            await app.state.engine.dispose()  # type: ignore[attr-defined]
            # Rendre l'engine inutilisable : on remplace le sessionmaker par un
            # store dont la DB pointe sur un chemin invalide.
            from tests.conftest import StubEmbedder

            from mnemos.models.base import make_async_engine
            from mnemos.stores.episodic import EpisodicStore

            broken = make_async_engine(tmp_path / "nonexistent_dir" / "x.db")
            app.state.store = EpisodicStore(  # type: ignore[attr-defined]
                broken, StubEmbedder(), app.state.clock, app.state.settings  # type: ignore[arg-type,attr-defined]
            )
            body = (await client.get("/v1/health")).json()
            assert body["ok"] is False
            assert body["dbs"]["episodic"] is False
            assert "episodic_db" in body["failures"]
            await broken.dispose()
    for engine in engines:
        await engine.dispose()

"""Endpoints admin (§16.1) + query PROCEDURAL — app stub, pas d'Ollama."""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import pytest
from tests.conftest import make_stub_app

from mnemos.stores.procedural import SkillMeta


@pytest.fixture
async def client(tmp_path: Path) -> AsyncIterator[httpx.AsyncClient]:
    app, engines = await make_stub_app(tmp_path)
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
            c.app_state = app.state  # type: ignore[attr-defined]
            yield c
    for engine in engines:
        await engine.dispose()


async def test_admin_decay(client: httpx.AsyncClient) -> None:
    await client.post("/v1/episodes", json={"content": "un souvenir", "role": "user"})
    resp = await client.post("/v1/admin/decay")
    assert resp.status_code == 200
    body = resp.json()
    assert body["scanned"] == 1
    assert body["dry_run"] is False


async def test_admin_consolidate_plumbing(client: httpx.AsyncClient) -> None:
    """Le stub LLM ne produit pas de faits — on teste la plomberie : le run
    passe, rapporte, écrit le marker, et health l'expose."""
    resp = await client.post("/v1/admin/consolidate")
    assert resp.status_code == 200
    body = resp.json()
    assert body["candidates"] == 0  # rien de saillant+mûr dans la DB tmp
    health = (await client.get("/v1/health")).json()
    assert health["worker_last_run"] is not None


async def test_query_procedural_via_router(client: httpx.AsyncClient) -> None:
    procedural = client.app_state.procedural  # type: ignore[attr-defined]
    procedural.register_skill(
        "send_email",
        "def send(): ...",
        SkillMeta(name="send_email", desc="envoyer un mail avec pièce jointe",
                  signature="send() -> bool"),
    )
    resp = await client.post("/v1/query", json={"q": "comment envoyer un mail ?"})
    body = resp.json()
    assert body["type"] == "procedural"
    assert body["procedural"] == ["send_email"]

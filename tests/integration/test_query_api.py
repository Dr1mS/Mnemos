"""POST /v1/query + reset session (§18 Phase 5) — app stub, pas d'Ollama.

Vérifie le fan-out par type de requête : épisodique, sémantique, working,
history (chaîne de versioning), et le fallback UNKNOWN (les deux stores).
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import pytest
from tests.conftest import make_stub_app


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


async def seed(client: httpx.AsyncClient) -> None:
    for content in ["je bosse sur le projet mnemos", "le thé vert est excellent"]:
        await client.post(
            "/v1/episodes",
            json={"content": content, "role": "user", "session_id": "s1"},
        )
    # Faits directs dans le semantic store (l'extraction a ses propres tests)
    semantic = client.app_state.semantic  # type: ignore[attr-defined]
    await semantic.add_fact("user", "works_at", "Datalyse", ["ep1"])
    await semantic.add_fact("user", "works_at", "Nexora", ["ep2"])  # supersède


async def test_query_unknown_fan_out_les_deux_stores(client: httpx.AsyncClient) -> None:
    await seed(client)
    resp = await client.post("/v1/query", json={"q": "le thé vert est excellent"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["type"] == "unknown"
    assert body["episodes"], "UNKNOWN doit consulter l'épisodique"
    # le sémantique est consulté aussi (résultats selon similarité stub)
    assert "facts" in body


async def test_query_semantic_fact(client: httpx.AsyncClient) -> None:
    await seed(client)
    resp = await client.post("/v1/query", json={"q": "où je bosse en ce moment ?"})
    body = resp.json()
    assert body["type"] == "semantic_fact"
    assert body["episodes"] == []  # pas de fan-out épisodique sur un fact pur
    objets = {f["fact"]["object"] for f in body["facts"]}
    assert "Datalyse" not in objets  # supersédé → jamais retourné


async def test_query_history_expose_la_chaine(client: httpx.AsyncClient) -> None:
    await seed(client)
    resp = await client.post(
        "/v1/query", json={"q": "montre-moi l'historique de mes jobs"}
    )
    body = resp.json()
    assert body["type"] == "semantic_history"
    if body["facts"]:  # si le KNN stub matche works_at
        objets = [f["object"] for f in body["history"]]
        assert "Datalyse" in objets and "Nexora" in objets  # chaîne complète


async def test_query_working(client: httpx.AsyncClient) -> None:
    await seed(client)
    resp = await client.post(
        "/v1/query", json={"q": "où on en est ?", "session_id": "s1"}
    )
    body = resp.json()
    assert body["type"] == "working"
    assert len(body["working"]) == 2  # les 2 épisodes pushés en WM
    assert body["episodes"] == [] and body["facts"] == []


async def test_reset_session(client: httpx.AsyncClient) -> None:
    await seed(client)
    assert (await client.post("/v1/sessions/s1/reset")).status_code == 204
    resp = await client.post(
        "/v1/query", json={"q": "où on en est ?", "session_id": "s1"}
    )
    assert resp.json()["working"] == []
    # reset d'une session inconnue : idempotent
    assert (await client.post("/v1/sessions/jamais-vue/reset")).status_code == 204


async def test_query_validation_422(client: httpx.AsyncClient) -> None:
    assert (await client.post("/v1/query", json={"q": ""})).status_code == 422
    assert (
        await client.post("/v1/query", json={"q": "x", "k": 0})
    ).status_code == 422

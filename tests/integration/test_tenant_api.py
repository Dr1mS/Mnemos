"""Isolation multi-tenant au niveau API HTTP (Lot 1 / P1).

Vérifie l'étanchéité end-to-end via l'app FastAPI (stub embedder/LLM) :
- POST /v1/episodes avec tenant → l'épisode n'est cherchable que dans son tenant ;
- GET /v1/facts et /v1/facts/history filtrés par tenant ;
- le défaut (pas de tenant) = tenant personnel `user` (non-régression) ;
- écritures croisées, étanchéité dans les deux sens.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import pytest
from tests.conftest import make_stub_app

TENANT_B = "atelios"


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


async def test_episode_tenant_dans_reponse_et_recherche(
    app_and_client: tuple[object, httpx.AsyncClient],
) -> None:
    _, client = app_and_client
    r = await client.post(
        "/v1/episodes",
        json={"content": "sprint review atelios", "role": "user", "tenant": TENANT_B},
    )
    assert r.status_code == 201
    assert r.json()["tenant"] == TENANT_B

    # Cherchable dans son tenant…
    hit = await client.get(
        "/v1/episodes/search", params={"q": "sprint review atelios", "tenant": TENANT_B}
    )
    assert any("atelios" in e["episode"]["content"] for e in hit.json())
    # …invisible dans le tenant personnel (défaut).
    miss = await client.get("/v1/episodes/search", params={"q": "sprint review atelios"})
    assert all("atelios" not in e["episode"]["content"] for e in miss.json())


async def test_defaut_est_user(app_and_client: tuple[object, httpx.AsyncClient]) -> None:
    _, client = app_and_client
    r = await client.post("/v1/episodes", json={"content": "sans tenant", "role": "user"})
    assert r.json()["tenant"] == "user"


async def test_facts_endpoint_isole_par_tenant(
    app_and_client: tuple[object, httpx.AsyncClient],
) -> None:
    app, client = app_and_client
    # On écrit directement via le store sémantique (les faits normaux passent
    # par la consolidation LLM, hors scope de ce test stub).
    semantic = app.state.semantic  # type: ignore[attr-defined]
    await semantic.add_fact("user", "lives_in", "Annecy", ["e"], tenant="user")
    await semantic.add_fact("atelios", "lives_in", "Paris", ["e"], tenant=TENANT_B)

    a = await client.get("/v1/facts")  # défaut user
    b = await client.get("/v1/facts", params={"tenant": TENANT_B})
    assert a.status_code == 200 and b.status_code == 200
    assert {f["object"] for f in a.json()} == {"Annecy"}
    assert {f["object"] for f in b.json()} == {"Paris"}
    assert all(f["tenant"] == "user" for f in a.json())
    assert all(f["tenant"] == TENANT_B for f in b.json())


async def test_facts_history_endpoint(
    app_and_client: tuple[object, httpx.AsyncClient],
) -> None:
    app, client = app_and_client
    semantic = app.state.semantic  # type: ignore[attr-defined]
    await semantic.add_fact("user", "works_at", "A", ["e"], tenant="user")
    await semantic.add_fact("user", "works_at", "B", ["e"], tenant="user")  # supersède
    r = await client.get(
        "/v1/facts/history", params={"subject": "user", "predicate": "works_at"}
    )
    assert r.status_code == 200
    objs = [f["object"] for f in r.json()]
    assert objs == ["A", "B"]  # chaîne de versioning, ancien → récent
    # history exige subject ET predicate.
    assert (await client.get("/v1/facts/history", params={"subject": "user"})).status_code == 422


async def test_query_isole_par_tenant(
    app_and_client: tuple[object, httpx.AsyncClient],
) -> None:
    _, client = app_and_client
    await client.post(
        "/v1/episodes",
        json={"content": "note confidentielle atelios", "role": "user", "tenant": TENANT_B},
    )
    q_b = await client.post(
        "/v1/query", json={"q": "note confidentielle atelios", "tenant": TENANT_B}
    )
    q_user = await client.post("/v1/query", json={"q": "note confidentielle atelios"})
    assert any("atelios" in e["episode"]["content"] for e in q_b.json()["episodes"])
    assert all(
        "atelios" not in e["episode"]["content"] for e in q_user.json()["episodes"]
    )

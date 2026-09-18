"""Tests d'intégration et de conformité contractuelle pour l'adaptateur AML.

Vérifie :
1. Les sondes de santé GET /health et GET /aml/health (non authentifiées, 200 OK)
2. Le flux POST /add synchrone et persistant
3. Le flux POST /search immédiat (recherche unifiée)
4. L'isolation stricte par user_id (tenant)
5. La fusion ordonnée des faits sémantiques et des épisodes bruts
6. Le respect du plafond top_k (<= 100)
7. Les trois modes d'authentification requis par AML (Bearer, Token, X-API-Key)
8. La validation stricte Pydantic (422 sur requête invalide)
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import pytest
from tests.conftest import make_stub_app

from mnemos.config import Settings


@pytest.fixture
async def client(tmp_path: Path) -> AsyncIterator[httpx.AsyncClient]:
    app, engines = await make_stub_app(tmp_path)
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            c.app_state = app.state  # type: ignore[attr-defined]
            yield c
    for engine in engines:
        await engine.dispose()


@pytest.fixture
async def authed_client(tmp_path: Path) -> AsyncIterator[httpx.AsyncClient]:
    settings = Settings(
        _env_file=None,
        DATA_DIR=tmp_path,
        EPISODIC_DB=tmp_path / "episodic_auth.db",
        SEMANTIC_DB=tmp_path / "semantic_auth.db",
        PROCEDURAL_DIR=tmp_path / "procedural",
        API_KEY="aml-secret-key-42",
    )
    app, engines = await make_stub_app(tmp_path, settings=settings)
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            c.app_state = app.state  # type: ignore[attr-defined]
            yield c
    for engine in engines:
        await engine.dispose()


async def test_aml_health_endpoints_unauthenticated(client: httpx.AsyncClient) -> None:
    """GET /health et GET /aml/health doivent renvoyer 200 OK sans authentification."""
    for path in ("/health", "/aml/health"):
        resp = await client.get(path)
        assert resp.status_code == 200
        body = resp.json()
        assert body.get("status") == "healthy"
        assert "version" in body


async def test_aml_add_and_search_synchronous(client: httpx.AsyncClient) -> None:
    """POST /add garantit la persistance synchrone et l'immédiateté de POST /search."""
    add_payload = {
        "request_id": "eval:run_01:locomo:chunk_0",
        "messages": [
            {
                "role": "user",
                "timestamp": 1704067200000,
                "content": "J'ai emménagé à Annecy près du lac en 2026.",
            }
        ],
        "user_id": "eval:run_01:user_alpha",
        "session_id": "eval:run_01:session_10",
    }

    # Test sur /add
    resp_add = await client.post("/add", json=add_payload)
    assert resp_add.status_code == 200
    body_add = resp_add.json()
    assert body_add["success"] is True
    assert body_add["request_id"] == add_payload["request_id"]
    assert body_add["user_id"] == add_payload["user_id"]
    assert body_add["session_id"] == add_payload["session_id"]

    # Recherche immédiate sur /search
    search_payload = {
        "query": "Où est-ce que j'habite ?",
        "options": ["A. Paris", "B. Annecy"],
        "user_id": "eval:run_01:user_alpha",
        "top_k": 50,
    }
    resp_search = await client.post("/search", json=search_payload)
    assert resp_search.status_code == 200
    body_search = resp_search.json()
    assert "data" in body_search
    assert len(body_search["data"]) >= 1

    first_item = body_search["data"][0]
    assert "id" in first_item
    assert "Annecy" in first_item["content"]
    assert first_item["score"] is not None
    assert first_item["created_at"] is not None

    # Vérification de l'alias /aml/search
    resp_aml_search = await client.post("/aml/search", json=search_payload)
    assert resp_aml_search.status_code == 200
    assert len(resp_aml_search.json()["data"]) >= 1


async def test_aml_user_isolation(client: httpx.AsyncClient) -> None:
    """Deux user_id différents doivent être strictement isolés."""
    # Écriture pour Alice
    await client.post(
        "/add",
        json={
            "request_id": "req_alice",
            "messages": [{"role": "user", "content": "Mon code secret est 9482."}],
            "user_id": "user_alice",
            "session_id": "sess_alice",
        },
    )

    # Bob cherche le code secret
    resp_bob = await client.post(
        "/search",
        json={
            "query": "Quel est mon code secret ?",
            "user_id": "user_bob",
            "top_k": 10,
        },
    )
    assert resp_bob.status_code == 200
    assert resp_bob.json()["data"] == [], "Bob ne doit voir aucun souvenir d'Alice"

    # Alice cherche son code secret
    resp_alice = await client.post(
        "/search",
        json={
            "query": "Quel est mon code secret ?",
            "user_id": "user_alice",
            "top_k": 10,
        },
    )
    assert resp_alice.status_code == 200
    assert len(resp_alice.json()["data"]) >= 1
    assert "9482" in resp_alice.json()["data"][0]["content"]


async def test_aml_semantic_and_episodic_fusion(client: httpx.AsyncClient) -> None:
    """La recherche retourne à la fois les faits consolidés et les épisodes bruts."""
    user_id = "user_charlie"

    # 1. Épisode brut
    await client.post(
        "/add",
        json={
            "request_id": "req_c1",
            "messages": [{"role": "user", "content": "Je préfère le café au thé."}],
            "user_id": user_id,
            "session_id": "sess_c",
        },
    )

    # 2. Fait sémantique direct dans SemanticStore
    semantic = client.app_state.semantic  # type: ignore[attr-defined]
    await semantic.add_fact(
        subject="Charlie",
        predicate="works_at",
        object_="Atelios",
        source_episode_ids=["ep_c1"],
        tenant=user_id,
    )

    # 3. Recherche
    resp = await client.post(
        "/search",
        json={
            "query": "préférence et travail de Charlie",
            "user_id": user_id,
            "top_k": 10,
        },
    )
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert len(data) >= 2

    ids = [item["id"] for item in data]
    assert any(i.startswith("fact_") for i in ids), "Doit contenir le fait sémantique"
    assert any(i.startswith("ep_") for i in ids), "Doit contenir l'épisode brut"


async def test_aml_top_k_clamping(client: httpx.AsyncClient) -> None:
    """top_k doit être respecté et borné à 100."""
    user_id = "user_topk"
    for i in range(5):
        await client.post(
            "/add",
            json={
                "request_id": f"req_{i}",
                "messages": [{"role": "user", "content": f"Note numéro {i} sur le projet."}],
                "user_id": user_id,
                "session_id": "sess_topk",
            },
        )

    # Demande top_k = 2
    resp_2 = await client.post(
        "/search",
        json={"query": "Note projet", "user_id": user_id, "top_k": 2},
    )
    assert len(resp_2.json()["data"]) <= 2

    # Demande top_k = 200 -> plafonné à 100
    resp_200 = await client.post(
        "/search",
        json={"query": "Note projet", "user_id": user_id, "top_k": 200},
    )
    assert len(resp_200.json()["data"]) <= 100

    # Vérification que top_k > 50 (ex: 60) fonctionne sans le plafond de 50
    user_large = "user_large_topk"
    messages_large = [
        {"role": "user", "content": f"Élément {i} pour grand top_k"} for i in range(60)
    ]
    await client.post(
        "/add",
        json={
            "request_id": "req_large",
            "messages": messages_large,
            "user_id": user_large,
            "session_id": "sess_large",
        },
    )
    resp_60 = await client.post(
        "/search",
        json={"query": "Élément grand top_k", "user_id": user_large, "top_k": 60},
    )
    assert len(resp_60.json()["data"]) == 60


async def test_aml_auth_schemes(authed_client: httpx.AsyncClient) -> None:
    """L'authentification supporte Bearer, Token et X-API-Key."""
    payload_add = {
        "request_id": "req_auth",
        "messages": [{"role": "user", "content": "Test auth"}],
        "user_id": "user_auth",
        "session_id": "sess_auth",
    }

    # Sans clé -> 401
    resp_no_key = await authed_client.post("/add", json=payload_add)
    assert resp_no_key.status_code == 401

    # Clé erronée -> 401
    resp_bad_key = await authed_client.post(
        "/add", json=payload_add, headers={"Authorization": "Bearer bad-key"}
    )
    assert resp_bad_key.status_code == 401

    # Header Authorization: Bearer <key> -> 200
    resp_bearer = await authed_client.post(
        "/add",
        json=payload_add,
        headers={"Authorization": "Bearer aml-secret-key-42"},
    )
    assert resp_bearer.status_code == 200

    # Header Authorization: Token <key> -> 200
    resp_token = await authed_client.post(
        "/add",
        json=payload_add,
        headers={"Authorization": "Token aml-secret-key-42"},
    )
    assert resp_token.status_code == 200

    # Header X-API-Key: <key> -> 200
    resp_x_api = await authed_client.post(
        "/add",
        json=payload_add,
        headers={"X-API-Key": "aml-secret-key-42"},
    )
    assert resp_x_api.status_code == 200

    # Health check reste 200 sans auth même avec API_KEY configurée
    resp_health = await authed_client.get("/health")
    assert resp_health.status_code == 200


async def test_aml_validation_errors(client: httpx.AsyncClient) -> None:
    """Les requêtes malformées renvoient HTTP 422."""
    # Add sans messages
    resp_add = await client.post(
        "/add",
        json={"request_id": "req_1", "user_id": "u1", "session_id": "s1"},
    )
    assert resp_add.status_code == 422

    # Search sans query -> 422
    resp_search = await client.post(
        "/search",
        json={"user_id": "u1", "top_k": 10},
    )
    assert resp_search.status_code == 422

    # Search sans top_k (requis par AML) -> 422
    resp_search_no_topk = await client.post(
        "/search",
        json={"query": "Test query", "user_id": "u1"},
    )
    assert resp_search_no_topk.status_code == 422

    # Search avec top_k <= 0 -> 422
    resp_search_zero = await client.post(
        "/search",
        json={"query": "Test query", "user_id": "u1", "top_k": 0},
    )
    assert resp_search_zero.status_code == 422


async def test_aml_add_multi_message_batch(client: httpx.AsyncClient) -> None:
    """POST /add avec un lot de messages (chunking standard AML) persiste tous les messages."""
    n_messages = 10
    messages = [
        {
            "role": "user" if i % 2 == 0 else "assistant",
            "content": f"Tour de parole numéro {i} dans la conversation.",
            "timestamp": 1704067200000 + (i * 10000),
        }
        for i in range(n_messages)
    ]
    payload = {
        "request_id": "eval:run_batch:chunk_01",
        "user_id": "eval:run_batch:user_multi",
        "session_id": "eval:run_batch:session_01",
        "messages": messages,
    }

    resp = await client.post("/add", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert data["request_id"] == payload["request_id"]
    assert data["user_id"] == payload["user_id"]
    assert data["session_id"] == payload["session_id"]

    # Vérification que chaque message est immédiatement cherchable
    search_resp = await client.post(
        "/search",
        json={
            "query": "Tour de parole numéro 7",
            "user_id": "eval:run_batch:user_multi",
            "top_k": 10,
        },
    )
    assert search_resp.status_code == 200
    search_data = search_resp.json()
    assert len(search_data["data"]) >= 1
    contents = [item["content"] for item in search_data["data"]]
    assert any("numéro 7" in c for c in contents)


async def test_aml_multimodal_content_parts_support(client: httpx.AsyncClient) -> None:
    """Vérifie la robustesse face aux tableaux de ContentPart (multimodal) pour Add et Search."""
    user_id = "user_mm_support"
    add_payload = {
        "request_id": "req_mm_01",
        "user_id": user_id,
        "session_id": "sess_mm_01",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Photo du chat Yuzu à Annecy"},
                    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,mock"}},
                ],
            }
        ],
    }
    resp_add = await client.post("/add", json=add_payload)
    assert resp_add.status_code == 200

    # Search avec query sous forme de ContentPart[] et options conditionnelles
    search_payload = {
        "query": [
            {"type": "text", "text": "Comment s'appelle le chat ?"},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,mock"}},
        ],
        "options": ["A. Yuzu", "B. Félix"],
        "user_id": user_id,
        "top_k": 5,
    }
    resp_search = await client.post("/search", json=search_payload)
    assert resp_search.status_code == 200
    assert len(resp_search.json()["data"]) >= 1
    assert any("Yuzu" in item["content"] for item in resp_search.json()["data"])

    # Search sans options (question ouverte) reste valide
    search_open = {
        "query": "Chat Yuzu",
        "user_id": user_id,
        "top_k": 5,
    }
    resp_open = await client.post("/search", json=search_open)
    assert resp_open.status_code == 200

    # user_id manquant -> 422
    resp_no_uid = await client.post("/search", json={"query": "Chat Yuzu", "top_k": 5})
    assert resp_no_uid.status_code == 422

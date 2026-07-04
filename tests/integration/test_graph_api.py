"""GET /v1/graph + /viz — contrat du visualiseur {entities, facts, memories}."""

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


async def test_graph_contrat_complet(client: httpx.AsyncClient) -> None:
    semantic = client.app_state.semantic  # type: ignore[attr-defined]
    store = client.app_state.store  # type: ignore[attr-defined]

    # Deux entités connues + une supersession Datalyse → Nexora + provenance
    await semantic.upsert_entity("Datalyse", entity_type="org")
    await semantic.upsert_entity("Nexora", entity_type="org")
    episode = await store.write("je bosse chez Nexora", role="user")
    await store.set_entity_refs(episode.id, ["Nexora"])
    await semantic.add_fact("user", "works_at", "Datalyse", ["ep0"])
    await semantic.add_fact("user", "works_at", "Nexora", [episode.id])
    # Un fait dont l'object n'est pas une entité → entité-concept synthétique
    await semantic.add_fact("user", "has_goal", "apprendre Rust", ["ep1"])

    body = (await client.get("/v1/graph")).json()

    entities = {e["id"]: e for e in body["entities"]}
    facts = {f["predicate"]: f for f in body["facts"]}
    assert entities["Nexora"]["type"] == "organisation"
    assert "expired" not in entities["Nexora"]

    # Le fait courant pointe Nexora, libellé FR, palette emploi, chaîne d'historique
    work = facts["travaille chez"]
    assert work["object"] == "Nexora"
    assert work["type"] == "emploi"
    assert work["history"][0]["value"] == "Datalyse"
    assert work["history"][0]["entity"] == "Datalyse"

    # Datalyse : fantôme rattaché à Nexora
    assert entities["Datalyse"]["expired"] is True
    assert entities["Datalyse"]["tether"] == "Nexora"

    # Object hors entités → concept synthétique
    goal = facts["a pour objectif"]
    assert goal["object"] == "c:apprendre rust"
    assert entities["c:apprendre rust"]["label"] == "apprendre Rust"

    # Memories : freshness/importance/anchor
    memory = next(m for m in body["memories"] if "Nexora" in m["label"])
    assert 0 <= memory["freshness"] <= 1
    assert 0 <= memory["importance"] <= 1
    assert memory["anchor"] == "Nexora"


async def test_graph_isole_par_tenant(client: httpx.AsyncClient) -> None:
    """Le graphe d'un tenant ne montre jamais les faits/entités/épisodes d'un
    autre. Le filtre subject utilise le canonical_subject du tenant."""
    semantic = client.app_state.semantic  # type: ignore[attr-defined]
    store = client.app_state.store  # type: ignore[attr-defined]

    # Tenant personnel : subject 'user'
    await semantic.add_fact("user", "lives_in", "Annecy", ["e"], tenant="user")
    await store.write("note perso", role="user", tenant="user")
    # Tenant applicatif atelios : subject 'atelios' (canonical_subject fallback)
    await semantic.add_fact("atelios", "lives_in", "Paris", ["e"], tenant="atelios")
    await store.write("note atelios", role="user", tenant="atelios")

    user_graph = (await client.get("/v1/graph")).json()
    atelios_graph = (await client.get("/v1/graph", params={"tenant": "atelios"})).json()

    # Objects rendus en entités-concepts synthétiques (Annecy/Paris ne sont pas
    # des entités enregistrées) → on vérifie via les labels des entités du graphe.
    def entity_labels(graph: dict) -> set[str]:
        return {e["label"] for e in graph["entities"]}

    assert "Annecy" in entity_labels(user_graph)
    assert "Paris" not in entity_labels(user_graph)  # tenant atelios n'a pas fuité
    assert "Paris" in entity_labels(atelios_graph)
    assert "Annecy" not in entity_labels(atelios_graph)  # tenant user n'a pas fuité
    assert atelios_graph["tenant"] == "atelios"
    assert user_graph["tenant"] == "user"
    assert all("atelios" not in m["label"] for m in user_graph["memories"])
    assert all("perso" not in m["label"] for m in atelios_graph["memories"])


async def test_graph_vide_et_page_viz(client: httpx.AsyncClient) -> None:
    body = (await client.get("/v1/graph")).json()
    assert body["entities"] == [] and body["facts"] == [] and body["memories"] == []

    page = await client.get("/viz")
    assert page.status_code == 200
    assert "Memory Constellation" in page.text
    assert "/v1/graph" in page.text  # la page se branche bien sur l'API
    assert "__resources" not in page.text  # plus de ressources embarquées
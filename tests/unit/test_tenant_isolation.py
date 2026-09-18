"""Isolation multi-tenant au niveau store (Lot 1 / P1).

Deux tenants, écritures croisées, vérification de l'étanchéité DANS LES DEUX
SENS : query épisodique, faits, résolution d'entités, historique, rétraction,
comptage de doublons. Aucun chemin de code ne doit franchir la cloison.

Stub embedder — pas d'Ollama.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from hashlib import blake2b
from pathlib import Path

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from mnemos.clock import FixedClock
from mnemos.config import Settings
from mnemos.models.base import make_async_engine
from mnemos.models.episodic import EPISODIC_SCHEMA_SQL
from mnemos.models.semantic import SEMANTIC_SCHEMA_SQL
from mnemos.stores.episodic import EpisodicStore
from mnemos.stores.semantic import SemanticStore

TENANT_A = "user"  # tenant personnel (défaut)
TENANT_B = "atelios"  # tenant applicatif


class StubEmbedder:
    async def embed(self, content: str) -> list[float]:
        seed = blake2b(content.encode(), digest_size=8).digest()
        return ([(b / 255.0) - 0.5 for b in seed] * 128)[:1024]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [await self.embed(t) for t in texts]


@pytest.fixture
async def engines(tmp_path: Path) -> AsyncIterator[tuple[AsyncEngine, AsyncEngine]]:
    epi = make_async_engine(tmp_path / "episodic.db")
    sem = make_async_engine(tmp_path / "semantic.db")
    async with epi.begin() as conn:
        for stmt in EPISODIC_SCHEMA_SQL:
            await conn.execute(text(stmt))
    async with sem.begin() as conn:
        for stmt in SEMANTIC_SCHEMA_SQL:
            await conn.execute(text(stmt))
    yield epi, sem
    await epi.dispose()
    await sem.dispose()


@pytest.fixture
async def stores(
    engines: tuple[AsyncEngine, AsyncEngine], fixed_clock: FixedClock
) -> tuple[EpisodicStore, SemanticStore]:
    epi, sem = engines
    settings = Settings(_env_file=None)
    episodic = EpisodicStore(epi, StubEmbedder(), fixed_clock, settings)  # type: ignore[arg-type]
    semantic = SemanticStore(sem, StubEmbedder(), fixed_clock, settings)  # type: ignore[arg-type]
    return episodic, semantic


# ── Faits : écritures croisées, étanchéité bidirectionnelle ──────────────────


async def test_facts_isoles_entre_tenants(stores: tuple[EpisodicStore, SemanticStore]) -> None:
    _, semantic = stores
    await semantic.add_fact("user", "lives_in", "Annecy", ["ep1"], tenant=TENANT_A)
    await semantic.add_fact("atelios", "lives_in", "Paris", ["ep2"], tenant=TENANT_B)

    a_facts = await semantic.get_current_facts(tenant=TENANT_A)
    b_facts = await semantic.get_current_facts(tenant=TENANT_B)
    assert {f.object for f in a_facts} == {"Annecy"}
    assert {f.object for f in b_facts} == {"Paris"}
    # Sens inverse : aucun fait de B ne fuit dans A et réciproquement.
    assert all(f.tenant == TENANT_A for f in a_facts)
    assert all(f.tenant == TENANT_B for f in b_facts)


async def test_meme_fait_courant_dans_deux_tenants_nest_pas_doublon(
    stores: tuple[EpisodicStore, SemanticStore],
) -> None:
    """Un functional identique (subject/predicate/object) dans deux tenants
    coexiste : ce n'est pas un doublon, et aucun ne supersède l'autre."""
    _, semantic = stores
    r1 = await semantic.add_fact("user", "works_at", "Datalyse", ["ep1"], tenant=TENANT_A)
    r2 = await semantic.add_fact("user", "works_at", "Datalyse", ["ep2"], tenant=TENANT_B)
    assert r1.action == "inserted"
    assert r2.action == "inserted"  # PAS duplicate : autre tenant
    assert await semantic.count_duplicate_current() == 0


async def test_supersession_ne_traverse_pas_le_tenant(
    stores: tuple[EpisodicStore, SemanticStore], fixed_clock: FixedClock
) -> None:
    """Superseder un functional dans A ne touche pas le fait de B."""
    _, semantic = stores
    await semantic.add_fact("user", "works_at", "Datalyse", ["ep1"], tenant=TENANT_A)
    await semantic.add_fact("user", "works_at", "OldCorp", ["ep2"], tenant=TENANT_B)
    fixed_clock.advance(1_000)
    r = await semantic.add_fact("user", "works_at", "Nexora", ["ep3"], tenant=TENANT_A)
    assert r.action == "superseded"
    # B intact : son fait courant reste OldCorp.
    b = await semantic.get_current_facts("user", "works_at", tenant=TENANT_B)
    assert [f.object for f in b] == ["OldCorp"]
    a = await semantic.get_current_facts("user", "works_at", tenant=TENANT_A)
    assert [f.object for f in a] == ["Nexora"]


async def test_history_isole_par_tenant(
    stores: tuple[EpisodicStore, SemanticStore], fixed_clock: FixedClock
) -> None:
    _, semantic = stores
    await semantic.add_fact("user", "works_at", "A1", ["e"], tenant=TENANT_A)
    fixed_clock.advance(1_000)
    await semantic.add_fact("user", "works_at", "A2", ["e"], tenant=TENANT_A)
    await semantic.add_fact("user", "works_at", "B1", ["e"], tenant=TENANT_B)
    hist_a = await semantic.get_history("user", "works_at", tenant=TENANT_A)
    hist_b = await semantic.get_history("user", "works_at", tenant=TENANT_B)
    assert [f.object for f in hist_a] == ["A1", "A2"]
    assert [f.object for f in hist_b] == ["B1"]


async def test_retract_ne_touche_que_son_tenant(
    stores: tuple[EpisodicStore, SemanticStore],
) -> None:
    _, semantic = stores
    await semantic.add_fact("user", "prefers", "café", ["e"], tenant=TENANT_A)
    await semantic.add_fact("user", "prefers", "café", ["e"], tenant=TENANT_B)
    retracted = await semantic.retract_fact("user", "prefers", "café", tenant=TENANT_A)
    assert retracted is not None
    assert not await semantic.get_current_facts("user", "prefers", tenant=TENANT_A)
    # B toujours là.
    assert {f.object for f in await semantic.get_current_facts(
        "user", "prefers", tenant=TENANT_B
    )} == {"café"}


# ── Entités : homonymes isolés, résolution d'alias non cross-tenant ──────────


async def test_entites_homonymes_isolees(
    stores: tuple[EpisodicStore, SemanticStore],
) -> None:
    """Une entité "Tom" chez A et une chez B coexistent (PK composite),
    et resolve_name ne résout jamais vers l'autre tenant."""
    _, semantic = stores
    await semantic.upsert_entity("Tom Dupont", "person", ["Tom"], tenant=TENANT_A)
    await semantic.upsert_entity("Tom Martin", "person", ["Tom"], tenant=TENANT_B)
    assert await semantic.resolve_name("Tom", TENANT_A) == "Tom Dupont"
    assert await semantic.resolve_name("Tom", TENANT_B) == "Tom Martin"
    # Un alias connu de A n'est pas résolu dans B si B ne le connaît pas.
    await semantic.upsert_entity("Google Inc", "org", ["GOOG"], tenant=TENANT_A)
    assert await semantic.resolve_name("GOOG", TENANT_A) == "Google Inc"
    assert await semantic.resolve_name("GOOG", TENANT_B) == "GOOG"  # inconnu dans B


# ── Épisodes : recherche isolée ──────────────────────────────────────────────


async def test_episodes_recherche_isolee(
    stores: tuple[EpisodicStore, SemanticStore],
) -> None:
    episodic, _ = stores
    await episodic.write("réunion projet Atelios budget", "user", tenant=TENANT_A)
    await episodic.write("réunion projet Atelios budget", "user", tenant=TENANT_B)
    res_a = await episodic.search("réunion projet Atelios budget", k=10, tenant=TENANT_A)
    res_b = await episodic.search("réunion projet Atelios budget", k=10, tenant=TENANT_B)
    assert len(res_a) == 1
    assert len(res_b) == 1
    assert res_a[0].episode.tenant == TENANT_A
    assert res_b[0].episode.tenant == TENANT_B


async def test_list_recent_isole(stores: tuple[EpisodicStore, SemanticStore]) -> None:
    episodic, _ = stores
    await episodic.write("perso", "user", tenant=TENANT_A)
    await episodic.write("appli", "user", tenant=TENANT_B)
    a = await episodic.list_recent(tenant=TENANT_A)
    b = await episodic.list_recent(tenant=TENANT_B)
    assert [e.content for e in a] == ["perso"]
    assert [e.content for e in b] == ["appli"]


async def test_pending_counts_par_tenant(
    stores: tuple[EpisodicStore, SemanticStore],
) -> None:
    episodic, _ = stores
    await episodic.write("a1", "user", tenant=TENANT_A)
    await episodic.write("a2", "user", tenant=TENANT_A)
    await episodic.write("b1", "user", tenant=TENANT_B)
    counts_a = await episodic.pending_counts(tenant=TENANT_A)
    counts_b = await episodic.pending_counts(tenant=TENANT_B)
    counts_all = await episodic.pending_counts()  # tous tenants
    assert counts_a["unscored"] == 2
    assert counts_b["unscored"] == 1
    assert counts_all["unscored"] == 3


# ── Défaut = tenant personnel (non-régression) ───────────────────────────────


async def test_defaut_est_tenant_personnel(
    stores: tuple[EpisodicStore, SemanticStore],
) -> None:
    """Sans tenant explicite, tout retombe sur `user` — les clients existants
    (MCP, Claude Code, CLI) restent fonctionnels sans changement."""
    episodic, semantic = stores
    await semantic.add_fact("user", "lives_in", "Annecy", ["e"])  # pas de tenant
    ep = await episodic.write("bonjour", "user")  # pas de tenant
    assert ep.tenant == "user"
    facts = await semantic.get_current_facts()  # pas de tenant
    assert {f.object for f in facts} == {"Annecy"}
    assert all(f.tenant == "user" for f in facts)


async def test_recherche_episodique_non_masquee_par_autre_tenant_plus_proche(
    stores: tuple[EpisodicStore, SemanticStore],
) -> None:
    """Vérifie que sqlite-vec partitionne nativement par tenant dans episodes_vec.

    Même si tenant A contient plus de KNN_CANDIDATES (50+) épisodes très proches
    de la requête, la recherche dans tenant B ne doit JAMAIS être masquée.
    """
    episodic, _ = stores
    # 60 épisodes dans TENANT_A (tous sur le même sujet que la requête)
    for i in range(60):
        await episodic.write(f"projet secret robotique numéro {i}", "user", tenant=TENANT_A)

    # 1 seul épisode dans TENANT_B
    target = await episodic.write("projet secret robotique chez atelios", "user", tenant=TENANT_B)

    # Recherche dans TENANT_B : doit retourner l'épisode de TENANT_B sans être masqué
    results = await episodic.search("projet secret robotique", k=10, tenant=TENANT_B)
    assert len(results) == 1
    assert results[0].episode.id == target.id
    assert results[0].episode.tenant == TENANT_B


async def test_recherche_facts_non_masquee_par_autre_tenant_plus_proche(
    stores: tuple[EpisodicStore, SemanticStore],
) -> None:
    """Vérifie que sqlite-vec partitionne nativement par tenant dans facts_vec.

    Même si tenant A contient plus de 60 faits très proches de la requête,
    la recherche sémantique dans tenant B ne doit JAMAIS être masquée.
    """
    _, semantic = stores
    for i in range(60):
        await semantic.add_fact(
            f"sujet_{i}", "knows_about", f"robotique_{i}", ["e"], tenant=TENANT_A
        )

    target_res = await semantic.add_fact(
        "sujet_cible", "knows_about", "robotique_atelios", ["e"], tenant=TENANT_B
    )
    target_id = target_res.fact.id

    results = await semantic.search_facts("knows about robotique", k=10, tenant=TENANT_B)
    assert len(results) == 1
    assert results[0].fact.id == target_id
    assert results[0].fact.tenant == TENANT_B


async def test_facts_superseded_ne_masquent_pas_fait_courant_dans_facts_vec(
    stores: tuple[EpisodicStore, SemanticStore], fixed_clock: FixedClock
) -> None:
    """Vérifie que facts_vec ne contient QUE les faits actifs (valid_until IS NULL).

    Après N supersessions sur un prédicat fonctionnel, les anciens faits
    sont purgés de facts_vec pour éviter le bug Mem0 où les faits périmés
    saturent le top-K et masquent le fait courant.
    """
    _, semantic = stores
    # 30 supersessions successives
    for i in range(30):
        fixed_clock.advance(1_000)
        await semantic.add_fact("user", "works_at", f"Company_{i}", ["e"], tenant=TENANT_A)

    # Le fait courant est Company_29
    current_facts = await semantic.get_current_facts("user", "works_at", tenant=TENANT_A)
    assert len(current_facts) == 1
    assert current_facts[0].object == "Company_29"

    # Recherche vectorielle : doit trouver Company_29 avec k=5
    results = await semantic.search_facts("works at Company", k=5, tenant=TENANT_A)
    assert len(results) == 1
    assert results[0].fact.object == "Company_29"

    # Vérification directe dans la table virtuelle facts_vec
    async with semantic._sessions() as session:
        count = (
            await session.execute(
                text("SELECT count(*) FROM facts_vec WHERE tenant = :t"),
                {"t": TENANT_A},
            )
        ).scalar_one()
        assert count == 1


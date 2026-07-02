"""Tests conflict resolver (§10.2, §19.1) — les 5 cas + normalisation d'alias.

Cas : insert / duplicate / supersede (functional) / insert additionnel
(multi) / history. Stub embedder — pas d'Ollama.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from hashlib import blake2b
from pathlib import Path

import pytest
from sqlalchemy import text

from mnemos.clock import FixedClock
from mnemos.config import Settings
from mnemos.models.base import make_async_engine
from mnemos.models.semantic import SEMANTIC_SCHEMA_SQL
from mnemos.stores.semantic import SemanticStore


class StubEmbedder:
    async def embed(self, content: str) -> list[float]:
        seed = blake2b(content.encode(), digest_size=8).digest()
        return ([(b / 255.0) - 0.5 for b in seed] * 128)[:1024]


@pytest.fixture
async def store(tmp_path: Path, fixed_clock: FixedClock) -> AsyncIterator[SemanticStore]:
    engine = make_async_engine(tmp_path / "semantic.db")
    async with engine.begin() as conn:
        for stmt in SEMANTIC_SCHEMA_SQL:
            await conn.execute(text(stmt))
    yield SemanticStore(engine, StubEmbedder(), fixed_clock, Settings(_env_file=None))  # type: ignore[arg-type]
    await engine.dispose()


# ── Cas 1 : insert ────────────────────────────────────────────────────────────


async def test_insert_nouveau_fait(store: SemanticStore) -> None:
    result = await store.add_fact("user", "works_at", "Datalyse", ["ep1"])
    assert result.action == "inserted"
    facts = await store.get_current_facts("user", "works_at")
    assert len(facts) == 1
    assert facts[0].object == "Datalyse"
    assert facts[0].valid_until is None


# ── Cas 2 : duplicate ─────────────────────────────────────────────────────────


async def test_duplicate_merge_confidence_et_sources(store: SemanticStore) -> None:
    await store.add_fact("user", "prefers", "thé", ["ep1"], confidence=0.7)
    result = await store.add_fact("user", "prefers", "Thé", ["ep2"], confidence=0.9)
    assert result.action == "duplicate"  # casse-insensible
    facts = await store.get_current_facts("user", "prefers")
    assert len(facts) == 1
    assert facts[0].confidence == 0.9  # max
    assert json.loads(facts[0].source_episodes) == ["ep1", "ep2"]


# ── Cas 3 : supersede (functional) ────────────────────────────────────────────


async def test_supersede_functional(store: SemanticStore, fixed_clock: FixedClock) -> None:
    r1 = await store.add_fact("user", "works_at", "Datalyse", ["ep1"])
    fixed_clock.advance(86_400_000)
    r2 = await store.add_fact("user", "works_at", "Nexora", ["ep2"])
    assert r2.action == "superseded"
    assert r2.superseded is not None and r2.superseded.id == r1.fact.id

    current = await store.get_current_facts("user", "works_at")
    assert len(current) == 1  # jamais deux faits courants sur un functional
    assert current[0].object == "Nexora"

    old = r2.superseded
    assert old.valid_until == fixed_clock.now_ms()
    assert old.superseded_by == r2.fact.id


# ── Cas 4 : insert additionnel (multi) ────────────────────────────────────────


async def test_multi_valeurs_coexistent(store: SemanticStore) -> None:
    """"Je préfère le thé" n'invalide pas "je préfère le café" (§10.2)."""
    await store.add_fact("user", "prefers", "café", ["ep1"])
    result = await store.add_fact("user", "prefers", "thé", ["ep2"])
    assert result.action == "inserted"  # JAMAIS superseded sur un multi
    objets = {f.object for f in await store.get_current_facts("user", "prefers")}
    assert objets == {"café", "thé"}


# ── Cas 5 : history ───────────────────────────────────────────────────────────


async def test_history_chaine_de_versioning(
    store: SemanticStore, fixed_clock: FixedClock
) -> None:
    await store.add_fact("user", "works_at", "A", ["ep1"])
    fixed_clock.advance(1_000)
    await store.add_fact("user", "works_at", "B", ["ep2"])
    fixed_clock.advance(1_000)
    await store.add_fact("user", "works_at", "C", ["ep3"])

    history = await store.get_history("user", "works_at")
    assert [f.object for f in history] == ["A", "B", "C"]
    assert history[0].superseded_by == history[1].id
    assert history[1].superseded_by == history[2].id
    assert history[2].valid_until is None  # seul le dernier est courant
    assert await store.count_duplicate_current() == 0


# ── Normalisation d'alias (§10.2 étape 0) ─────────────────────────────────────


async def test_alias_normalisation_dedupe(store: SemanticStore) -> None:
    await store.upsert_entity("Google Inc", entity_type="org", aliases=["Google", "GOOG"])
    r1 = await store.add_fact("user", "works_at", "Google Inc", ["ep1"])
    # "google" (alias, casse différente) → résolu vers "Google Inc" → duplicate
    r2 = await store.add_fact("user", "works_at", "google", ["ep2"])
    assert r1.action == "inserted"
    assert r2.action == "duplicate"
    facts = await store.get_current_facts("user", "works_at")
    assert len(facts) == 1
    assert facts[0].object == "Google Inc"


async def test_alias_normalisation_du_subject(store: SemanticStore) -> None:
    await store.upsert_entity("Tom Dupont", entity_type="person", aliases=["Tom"])
    await store.add_fact("Tom", "works_at", "Airbus", ["ep1"])
    facts = await store.get_current_facts("Tom Dupont", "works_at")
    assert len(facts) == 1
    assert facts[0].subject == "Tom Dupont"


# ── Entités ───────────────────────────────────────────────────────────────────


async def test_upsert_entity_incremente_et_merge_alias(
    store: SemanticStore, fixed_clock: FixedClock
) -> None:
    e1 = await store.upsert_entity("Datalyse", entity_type="org")
    fixed_clock.advance(1_000)
    e2 = await store.upsert_entity("datalyse", aliases=["Datalyse SAS"])
    assert e2.canonical_name == e1.canonical_name
    assert e2.episode_count == 2
    assert e2.last_seen > e1.first_seen
    assert "Datalyse SAS" in json.loads(e2.aliases)


async def test_retract_fact_invalide_sans_supersession(
    store: SemanticStore, fixed_clock: FixedClock
) -> None:
    """Rétractation : valid_until posé, PAS de superseded_by (fin de validité,
    pas remplacement). Le fait reste en historique."""
    await store.add_fact("user", "prefers", "café", ["ep1"])
    await store.add_fact("user", "prefers", "thé", ["ep2"])
    retracted = await store.retract_fact("user", "prefers", "Café")  # casse-insensible
    assert retracted is not None
    assert retracted.valid_until == fixed_clock.now_ms()
    assert retracted.superseded_by is None

    current = {f.object for f in await store.get_current_facts("user", "prefers")}
    assert current == {"thé"}  # café retiré, thé intact
    history = await store.get_history("user", "prefers")
    assert any(f.object == "café" for f in history)  # audit préservé


async def test_retract_fact_inconnu_retourne_none(store: SemanticStore) -> None:
    await store.add_fact("user", "prefers", "thé", ["ep1"])
    assert await store.retract_fact("user", "prefers", "inexistant") is None
    assert await store.retract_fact("user", "works_at", "thé") is None
    assert len(await store.get_current_facts("user", "prefers")) == 1  # rien touché


async def test_retract_puis_reaffirmation(store: SemanticStore, fixed_clock: FixedClock) -> None:
    """Rétracter puis ré-affirmer → nouveau fait courant, chaîne complète."""
    await store.add_fact("user", "prefers", "café", ["ep1"])
    await store.retract_fact("user", "prefers", "café")
    fixed_clock.advance(1_000)
    result = await store.add_fact("user", "prefers", "café", ["ep2"])
    assert result.action == "inserted"  # pas duplicate : l'ancien est invalidé
    assert len(await store.get_history("user", "prefers")) == 2


async def test_search_facts_exclut_les_invalides(store: SemanticStore) -> None:
    """Anti-pattern 6 : les faits supersédés ne polluent pas la recherche."""
    await store.add_fact("user", "works_at", "Datalyse", ["ep1"])
    await store.add_fact("user", "works_at", "Nexora", ["ep2"])  # supersède
    results = await store.search_facts("user works_at Datalyse", k=10)
    objets = {s.fact.object for s in results}
    assert "Nexora" in objets
    assert "Datalyse" not in objets  # invalidé → filtré malgré le match dense parfait

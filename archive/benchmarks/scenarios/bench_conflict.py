"""Scénario 3 : Épreuve de Vérité Cognitive, Résolution de Conflits & Temporalité.

Mesure :
1. Résolution de conflits sous "Chaos Temporel" (5 mises à jour de situation de vie consécutives).
2. Vérification de l'intégrité de la chaîne de versioning (valid_from, valid_until, superseded_by).
3. Respect strict de la cardinalité (ONE vs MANY) selon l'ontologie.
4. Isolation cognitive stricte entre tenants (deux vérités courantes contradictoires).
5. Résolution d'alias et normalisation d'entités (évitement de doublons de faits).
"""

from __future__ import annotations

from dataclasses import dataclass

from benchmarks.config import BenchConfig
from benchmarks.harness import BenchmarkHarness


@dataclass
class TemporalChainCheck:
    predicate: str
    total_facts: int
    current_facts_count: int
    chain_continuous: bool
    final_value: str


@dataclass
class CognitiveReport:
    temporal_chain: TemporalChainCheck | None = None
    cardinality_respected: bool = False
    cross_tenant_conflict_isolated: bool = False
    alias_deduplication_clean: bool = False
    duplicate_current_count: int = 0


async def run_conflict_benchmark(config: BenchConfig) -> CognitiveReport:
    harness = BenchmarkHarness(config)
    await harness.setup(reset=True)
    assert harness.semantic_store is not None

    report = CognitiveReport()
    sem = harness.semantic_store

    # ── Test 1 : Chaos Temporel & Chaîne de Versioning ───────────────────────
    locations = ["Paris", "Tokyo", "Lyon", "Annecy"]
    inserted_ids: list[str] = []

    for loc in locations:
        res = await sem.add_fact(
            subject="user",
            predicate="lives_in",
            object_=loc,
            source_episode_ids=["ep_bench"],
            confidence=0.95,
            tenant="user",
        )
        inserted_ids.append(res.fact.id)

    history = await sem.get_history(subject="user", predicate="lives_in", tenant="user")
    current_facts = [f for f in history if f.valid_until is None]

    continuous = True
    for i in range(len(history) - 1):
        prev_f = history[i]
        next_f = history[i + 1]
        if prev_f.superseded_by != next_f.id:
            continuous = False
        if prev_f.valid_until != next_f.valid_from:
            continuous = False

    final_val = current_facts[0].object if current_facts else ""
    report.temporal_chain = TemporalChainCheck(
        predicate="lives_in",
        total_facts=len(history),
        current_facts_count=len(current_facts),
        chain_continuous=continuous,
        final_value=final_val,
    )

    # ── Test 2 : Cardinalité FUNCTIONAL (ONE) vs MULTI (MANY) ────────────────
    await sem.add_fact("user", "prefers", "pizza", ["ep1"], tenant="user")
    await sem.add_fact("user", "prefers", "sushi", ["ep2"], tenant="user")
    prefers_history = await sem.get_history("user", "prefers", tenant="user")
    prefers_current = [f for f in prefers_history if f.valid_until is None]

    report.cardinality_respected = (len(prefers_current) == 2 and len(current_facts) == 1)

    # ── Test 3 : Isolation Cognitive Cross-Tenant ────────────────────────────
    await sem.add_fact("user", "lives_in", "Berlin", ["epA"], tenant="tenant_a")
    await sem.add_fact("user", "lives_in", "Madrid", ["epB"], tenant="tenant_b")

    facts_a = await sem.get_current_facts(tenant="tenant_a")
    facts_b = await sem.get_current_facts(tenant="tenant_b")

    loc_a = next((f.object for f in facts_a if f.predicate == "lives_in"), None)
    loc_b = next((f.object for f in facts_b if f.predicate == "lives_in"), None)

    report.cross_tenant_conflict_isolated = (loc_a == "Berlin" and loc_b == "Madrid")

    # ── Test 4 : Normalisation d'alias & Déduplication ───────────────────────
    await sem.upsert_entity(
        name="Google",
        entity_type="org",
        aliases=["google", "Google Inc", "Alphabet"],
        tenant="user",
    )
    # Ajout d'un fait ciblant un alias
    r1 = await sem.add_fact("Google Inc", "owns", "Android", ["ep1"], tenant="user")
    # Ajout d'un fait ciblant le nom canonique
    r2 = await sem.add_fact("google", "owns", "Android", ["ep2"], tenant="user")

    report.alias_deduplication_clean = (r1.action == "inserted" and r2.action == "duplicate")
    report.duplicate_current_count = await sem.count_duplicate_current()

    await harness.teardown()
    return report

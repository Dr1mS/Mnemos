"""Consolidation loop end-to-end (§19.2) — Ollama réel (qwen3:4b).

Scénario Alice : elle bosse chez Datalyse, préfère le thé, PUIS change pour
Nexora → après consolidation, le sémantique doit refléter le versioning
(works_at supersédé, préférence conservée) et les entités être peuplées.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from sqlalchemy import text

from mnemos.clock import FixedClock
from mnemos.config import Settings
from mnemos.consolidation.extractor import FactExtractor
from mnemos.consolidation.worker import ConsolidationWorker
from mnemos.embeddings.dense import DenseEmbedder
from mnemos.llm.model_manager import ModelManager
from mnemos.llm.ollama_client import OllamaClient
from mnemos.models.base import make_async_engine
from mnemos.models.episodic import EPISODIC_SCHEMA_SQL
from mnemos.models.semantic import SEMANTIC_SCHEMA_SQL
from mnemos.stores.episodic import EpisodicStore
from mnemos.stores.semantic import SemanticStore
from mnemos.tagger.salience import SalienceScores

pytestmark = pytest.mark.requires_ollama

HIGH = SalienceScores(surprise=0.5, arousal=0.5, self_ref=0.9, recurrence=0.1, combined=0.9)
LOW = SalienceScores(surprise=0.1, arousal=0.1, self_ref=0.1, recurrence=0.1, combined=0.2)

HOUR_MS = 3_600_000


@pytest.fixture
async def env(
    tmp_path: Path, fixed_clock: FixedClock
) -> AsyncIterator[tuple[ConsolidationWorker, EpisodicStore, SemanticStore, FixedClock]]:
    settings = Settings(
        _env_file=None,
        DATA_DIR=tmp_path,
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
    client = OllamaClient(settings)
    manager = ModelManager(settings, client)
    embedder = DenseEmbedder(manager, settings)
    episodic = EpisodicStore(epi_engine, embedder, fixed_clock, settings)
    semantic = SemanticStore(sem_engine, embedder, fixed_clock, settings)
    worker = ConsolidationWorker(
        episodic, semantic, FactExtractor(manager, settings), settings, fixed_clock
    )
    yield worker, episodic, semantic, fixed_clock
    await epi_engine.dispose()
    await sem_engine.dispose()
    await client.aclose()


async def test_consolidation_loop_versioning(
    env: tuple[ConsolidationWorker, EpisodicStore, SemanticStore, FixedClock],
) -> None:
    worker, episodic, semantic, clock = env

    # Épisodes saillants (scores posés directement — le scoring LLM a ses
    # propres tests ; ici on teste la consolidation).
    await episodic.write(
        "Je bosse chez Datalyse comme data engineer.", "user", salience_scores=HIGH
    )
    await episodic.write(
        "Je préfère le thé au café, surtout le thé vert.", "user", salience_scores=HIGH
    )
    noise = await episodic.write("ok merci", "user", salience_scores=LOW)
    clock.advance(2 * HOUR_MS)  # dépasse CONSOLIDATION_DELAY_HOURS=1

    report1 = await worker.run_once()
    assert report1.candidates == 2  # le bruit (0.2 < 0.6) n'est pas candidat
    assert report1.consolidated == 2
    assert report1.extraction_failures == 0

    facts = await semantic.get_current_facts("user", "works_at")
    assert len(facts) == 1
    assert "datalyse" in facts[0].object.lower()

    # Changement de job → supersession attendue au prochain run
    await episodic.write(
        "Grosse nouvelle : je quitte Datalyse, je rejoins Nexora !", "user",
        salience_scores=HIGH,
    )
    clock.advance(2 * HOUR_MS)
    report2 = await worker.run_once()
    assert report2.consolidated == 1
    assert report2.facts_superseded >= 1  # works_at Datalyse → Nexora

    current = await semantic.get_current_facts("user", "works_at")
    assert len(current) == 1, "jamais deux works_at courants (§10.2)"
    assert "nexora" in current[0].object.lower()

    history = await semantic.get_history("user", "works_at")
    assert len(history) >= 2  # la chaîne Datalyse → Nexora est préservée
    assert any("datalyse" in f.object.lower() and f.valid_until is not None for f in history)

    # La préférence thé (multi) n'a pas été touchée par le changement de job
    prefs = await semantic.get_current_facts("user", "prefers")
    assert any("thé" in f.object.lower() for f in prefs)

    # État épisodique : consolidés marqués, le bruit non ; quality gate §21
    assert (await episodic.get_by_id(noise.id)).consolidated_at is None  # type: ignore[union-attr]
    assert await semantic.count_duplicate_current() == 0

    # Entités peuplées par le flux (§15) + marker worker écrit
    async with semantic._sessions() as session:
        n_entities = (await session.execute(text("SELECT COUNT(*) FROM entities"))).scalar_one()
    assert n_entities >= 1
    marker = worker._settings.DATA_DIR / "worker_last_run"
    assert marker.exists()


async def test_consolidation_idempotente(
    env: tuple[ConsolidationWorker, EpisodicStore, SemanticStore, FixedClock],
) -> None:
    """Re-run immédiat : plus aucun candidat, pas de double-écriture."""
    worker, episodic, semantic, clock = env
    await episodic.write("J'habite à Lyon.", "user", salience_scores=HIGH)
    clock.advance(2 * HOUR_MS)
    await worker.run_once()
    report = await worker.run_once()
    assert report.candidates == 0
    assert await semantic.count_duplicate_current() == 0

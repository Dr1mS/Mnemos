"""Tests unitaires dédiés au bug de consolidation et d'archivage :
1. Un épisode dont l'extraction produit 0 fait ne doit pas être marqué consolidé.
2. Un épisode à haute saillance ne doit pas être archivé par archive_old(),
   même si son decay_state tombe sous 0.1.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from sqlalchemy import text

from mnemos.clock import FixedClock
from mnemos.config import Settings
from mnemos.consolidation.extractor import FactExtractor
from mnemos.consolidation.worker import ConsolidationWorker
from mnemos.models.base import make_async_engine
from mnemos.models.episodic import EPISODIC_SCHEMA_SQL, Episode
from mnemos.models.semantic import SEMANTIC_SCHEMA_SQL
from mnemos.stores.episodic import EpisodicStore
from mnemos.stores.semantic import SemanticStore
from mnemos.tagger.salience import SalienceScores


class StubEmbedder:
    async def embed(self, content: str) -> list[float]:
        return [0.0] * 1024


class StubManager:
    def __init__(self, response_dict: dict) -> None:
        self.response = json.dumps(response_dict)

    async def generate(self, prompt: str, model: str, **kwargs: object) -> str:
        return self.response


def high_salience_scores() -> SalienceScores:
    return SalienceScores(
        surprise=0.8,
        arousal=0.8,
        self_ref=0.8,
        recurrence=0.8,
        combined=0.85,
    )


@pytest.mark.asyncio
async def test_zero_facts_extraction_leaves_episode_unconsolidated(tmp_path: Path) -> None:
    clock = FixedClock(1_782_727_200_000)
    settings = Settings(_env_file=None, DATA_DIR=tmp_path)
    epi_engine = make_async_engine(tmp_path / "epi.db")
    sem_engine = make_async_engine(tmp_path / "sem.db")

    async with epi_engine.begin() as conn:
        for stmt in EPISODIC_SCHEMA_SQL:
            await conn.execute(text(stmt))
    async with sem_engine.begin() as conn:
        for stmt in SEMANTIC_SCHEMA_SQL:
            await conn.execute(text(stmt))

    episodic = EpisodicStore(epi_engine, StubEmbedder(), clock, settings)  # type: ignore[arg-type]
    semantic = SemanticStore(sem_engine, StubEmbedder(), clock, settings)  # type: ignore[arg-type]

    # Extractor qui renvoie 0 fait
    stub_extractor = FactExtractor(StubManager({"facts": [], "entities": []}), settings)  # type: ignore[arg-type]
    worker = ConsolidationWorker(episodic, semantic, stub_extractor, settings, clock)

    ep = await episodic.write(
        "Clé de staging confidentielle : SEC-9482",
        role="user",
        salience_scores=high_salience_scores(),
    )
    # Vérifier l'état initial
    initial_ep = await episodic.get_by_id(ep.id)
    assert initial_ep is not None
    assert initial_ep.consolidated_at is None
    assert initial_ep.salience >= settings.SALIENCE_THRESHOLD_CONSOLIDATE

    # Avancer l'horloge pour être éligible à la consolidation
    clock.advance(2 * 3_600_000)
    report = await worker.run_once()

    assert report.candidates == 1
    assert report.consolidated == 0, "Un épisode sans faits ne doit PAS être compté consolidé"
    assert report.facts_inserted == 0

    after_ep = await episodic.get_by_id(ep.id)
    assert after_ep is not None
    assert after_ep.consolidated_at is None, "consolidated_at doit rester NULL si 0 fait extrait"

    await epi_engine.dispose()
    await sem_engine.dispose()


@pytest.mark.asyncio
async def test_archive_old_protects_high_salience_episodes(tmp_path: Path) -> None:
    clock = FixedClock(1_782_727_200_000)
    settings = Settings(_env_file=None, DATA_DIR=tmp_path)
    epi_engine = make_async_engine(tmp_path / "epi.db")

    async with epi_engine.begin() as conn:
        for stmt in EPISODIC_SCHEMA_SQL:
            await conn.execute(text(stmt))

    episodic = EpisodicStore(epi_engine, StubEmbedder(), clock, settings)  # type: ignore[arg-type]

    # 1. Épisode haute saillance (0.85 >= 0.6)
    ep_high = await episodic.write(
        "Information critique mais decayée",
        role="user",
        salience_scores=high_salience_scores(),
    )
    # 2. Épisode basse saillance (0.3 < 0.6)
    ep_low = await episodic.write(
        "Bruit quelconque",
        role="user",
        salience_scores=SalienceScores(
            surprise=0.1, arousal=0.1, self_ref=0.1, recurrence=0.1, combined=0.3
        ),
    )

    # Forcer decay_state < 0.1 et consolidated_at IS NOT NULL sur les deux
    async with episodic._sessions() as session, session.begin():
        for ep_id in [ep_high.id, ep_low.id]:
            await session.execute(
                text("UPDATE episodes SET decay_state = 0.05, consolidated_at = :now WHERE id = :id"),
                {"now": clock.now_ms(), "id": ep_id},
            )

    report = await episodic.archive_old()

    # Seul l'épisode à basse saillance doit être archivé par la règle 1
    assert report.decayed_consolidated == 1

    stored_high = await episodic.get_by_id(ep_high.id)
    assert stored_high is not None
    assert stored_high.archived == 0, "L'épisode à haute saillance ne doit JAMAIS être archivé"

    stored_low = await episodic.get_by_id(ep_low.id)
    assert stored_low is not None
    assert stored_low.archived == 1, "L'épisode à basse saillance consolidé et decayé doit être archivé"

    await epi_engine.dispose()

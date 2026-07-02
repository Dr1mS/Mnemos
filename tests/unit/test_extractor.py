"""Tests FactExtractor (§15.2) — validation post-parse, mapping des prédicats.
LLM mocké."""

from __future__ import annotations

import json

import pytest

from mnemos.config import Settings
from mnemos.consolidation.extractor import FactExtractor, map_predicate
from mnemos.ontology import FALLBACK_PREDICATE


class StubManager:
    def __init__(self, response: str) -> None:
        self.response = response

    async def generate(self, prompt: str, model: str, **kwargs: object) -> str:
        return self.response


def make_extractor(payload: object) -> FactExtractor:
    stub = StubManager(json.dumps(payload))
    return FactExtractor(stub, Settings(_env_file=None))  # type: ignore[arg-type]


async def test_extraction_nominale() -> None:
    extractor = make_extractor(
        {
            "facts": [
                {"subject": "user", "predicate": "works_at", "object": "Nexora",
                 "confidence": 0.9}
            ],
            "entities": [{"name": "Nexora", "entity_type": "org", "aliases": []}],
        }
    )
    result = await extractor.extract("je bosse chez Nexora", "user", 1_782_727_200_000)
    assert len(result.facts) == 1
    assert result.facts[0].predicate == "works_at"
    assert result.entities[0].entity_type == "org"


async def test_fait_invalide_skippe_pas_le_reste() -> None:
    extractor = make_extractor(
        {
            "facts": [
                {"subject": "", "predicate": "works_at", "object": "X"},  # vide → skip
                {"subject": "user", "predicate": "owns", "object": "vélo",
                 "confidence": 1.5},  # hors bornes → skip
                {"subject": "user", "predicate": "prefers", "object": "thé",
                 "confidence": 0.8},  # valide
            ],
            "entities": [],
        }
    )
    result = await extractor.extract("...", "user", 0)
    assert len(result.facts) == 1
    assert result.facts[0].object == "thé"


async def test_entity_type_hors_vocab_devient_null() -> None:
    extractor = make_extractor(
        {"facts": [], "entities": [{"name": "Yuzu", "entity_type": "animal"}]}
    )
    result = await extractor.extract("...", "user", 0)
    assert result.entities[0].entity_type is None


async def test_json_invalide_leve() -> None:
    stub = StubManager("pas du json")
    extractor = FactExtractor(stub, Settings(_env_file=None))  # type: ignore[arg-type]
    with pytest.raises(json.JSONDecodeError):
        await extractor.extract("...", "user", 0)  # le worker gère le retry


def test_map_predicate_exact() -> None:
    assert map_predicate("works_at", "X") == ("works_at", "X")
    assert map_predicate("  Works_At ", "X") == ("works_at", "X")


def test_map_predicate_fuzzy() -> None:
    # "work_at" (typo LLM) → works_at par similarité
    predicate, obj = map_predicate("work_at", "Nexora")
    assert predicate == "works_at"
    assert obj == "Nexora"


def test_map_predicate_fallback_preserve_le_brut() -> None:
    """Anti-pattern 3 : jamais de predicate créé — fallback has_attribute
    avec le predicate brut dans l'object."""
    predicate, obj = map_predicate("plays_instrument", "guitare")
    assert predicate == FALLBACK_PREDICATE
    assert obj == "plays_instrument: guitare"


# ── Rescan des non-scorés (worker) ────────────────────────────────────────────


async def test_worker_rescore_les_episodes_non_scores(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Jobs de salience perdus (process mort) → rattrapés au run_once suivant."""
    import json as _json

    from sqlalchemy import text as _sql
    from tests.conftest import StubEmbedder

    from mnemos.clock import FixedClock
    from mnemos.consolidation.worker import ConsolidationWorker
    from mnemos.models.base import make_async_engine
    from mnemos.models.episodic import EPISODIC_SCHEMA_SQL
    from mnemos.models.semantic import SEMANTIC_SCHEMA_SQL
    from mnemos.stores.episodic import EpisodicStore
    from mnemos.stores.semantic import SemanticStore
    from mnemos.tagger.salience import SalienceTagger

    class StubGen:
        async def generate(self, prompt: str, model: str, **kw: object) -> str:
            if "salience" in prompt:
                return _json.dumps(
                    {"surprise": 0.2, "arousal": 0.2, "self_ref": 0.9, "recurrence": 0.0}
                )
            return _json.dumps({"facts": [], "entities": []})

    clock = FixedClock(start_ms=1_782_727_200_000)
    settings = Settings(_env_file=None, DATA_DIR=tmp_path)
    epi_engine = make_async_engine(tmp_path / "e.db")
    sem_engine = make_async_engine(tmp_path / "s.db")
    async with epi_engine.begin() as conn:
        for stmt in EPISODIC_SCHEMA_SQL:
            await conn.execute(_sql(stmt))
    async with sem_engine.begin() as conn:
        for stmt in SEMANTIC_SCHEMA_SQL:
            await conn.execute(_sql(stmt))
    stub = StubGen()
    episodic = EpisodicStore(epi_engine, StubEmbedder(), clock, settings)  # type: ignore[arg-type]
    worker = ConsolidationWorker(
        episodic,
        SemanticStore(sem_engine, StubEmbedder(), clock, settings),  # type: ignore[arg-type]
        FactExtractor(stub, settings),  # type: ignore[arg-type]
        settings,
        clock,
        tagger=SalienceTagger(stub, settings),  # type: ignore[arg-type]
    )
    ep = await episodic.write("je suis dev", role="user")  # jamais scoré → 0.5
    clock.advance(2 * 3_600_000)
    report = await worker.run_once()
    assert report.rescored == 1
    got = await episodic.get_by_id(ep.id)
    assert got is not None
    assert got.salience == 0.9  # boost-floor self_ref appliqué au rattrapage
    assert report.candidates == 1  # devenu candidat et consolidé dans le même run
    await epi_engine.dispose()
    await sem_engine.dispose()

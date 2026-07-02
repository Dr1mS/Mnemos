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

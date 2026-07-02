"""Extracteur de faits + entités (§15.2) — prompt v4 validé par le POC.

Les exemples de bascule sont la partie du prompt qui porte : un 4B
sur-supprime avec des règles abstraites seules (passé composé, has_goal).
Ne pas les retirer pour "raccourcir le prompt" (cf. poc/RESULTS.md).

Validation post-parsing : predicate ∈ vocabulaire (sinon mapping fuzzy ou
FALLBACK_PREDICATE — anti-pattern 3 : jamais de predicate à la volée),
subject/object non vides, confidence ∈ [0,1], entity_type ∈ vocabulaire ou
null. Toute extraction invalide est skippée et loggée.
"""

from __future__ import annotations

import difflib
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime

from mnemos.config import Settings
from mnemos.llm.model_manager import ModelManager
from mnemos.logging import get_logger
from mnemos.ontology import ENTITY_TYPES, FALLBACK_PREDICATE, PREDICATES

logger = get_logger(__name__)

EXTRACTION_PROMPT = """Extract structured facts and named entities from this conversation episode.
Output JSON object: {{"facts": [...], "entities": [...]}}.
Each fact: {{subject, predicate, object, confidence}}.
Each entity: {{name, entity_type, aliases}} where entity_type is one of
             person|org|place|concept|product, and aliases lists other
             surface forms used in the episode (may be empty).

Allowed predicates: works_at, lives_in, prefers, dislikes, owns,
                    is_a, has_attribute, knows_about, has_goal, has_skill

Rules:
- subject is EXACTLY "user" when the fact is about the user speaking; when
  the sentence explicitly names another person/entity as the actor, that
  entity is the subject
- if the actor is a pronoun whose referent is NOT named in this episode,
  extract NOTHING about it
- extract facts that are CURRENTLY true. A past event that established a
  current state IS a current fact. A state explicitly ended is NOT.
- personal goals and desires to learn/do something ARE facts (has_goal)
- IGNORE questions, unrealistic hypotheticals/conditionals, jokes, sarcasm,
  and statements the speaker is unsure about
- use canonical English for predicates, but keep the object in the original
  language of the episode (do not translate it)
- confidence is a number between 0.0 and 1.0
- entities: only entities actually mentioned; use the most complete surface
  form as name
- if nothing extractable, return {{"facts": [], "entities": []}}

Examples:
- "Avant je bossais chez TechCorp." → facts: []  (state ended, no longer true)
- "J'ai adopté un chat, Yuzu." → {{"subject": "user", "predicate": "owns", "object": "Yuzu", "confidence": 0.9}}  (past event, current state)
- "J'aimerais apprendre Rust." → {{"subject": "user", "predicate": "has_goal", "object": "Rust", "confidence": 0.9}}
- "Mon frère Tom travaille chez Airbus." → {{"subject": "Tom", "predicate": "works_at", "object": "Airbus", "confidence": 0.9}}
- "Je ne bois plus de thé, je suis passée au maté." → {{"subject": "user", "predicate": "prefers", "object": "maté", "confidence": 0.9}}  (only the NEW preference)
- "Si je gagnais au loto, j'achèterais une villa." → facts: []  (hypothetical)

Episode (role={role}, timestamp={ts}):
{content}

Output ONLY JSON."""

FUZZY_CUTOFF = 0.75


@dataclass(frozen=True)
class ExtractedFact:
    subject: str
    predicate: str
    object: str
    confidence: float


@dataclass(frozen=True)
class ExtractedEntity:
    name: str
    entity_type: str | None
    aliases: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Extraction:
    facts: list[ExtractedFact]
    entities: list[ExtractedEntity]


def map_predicate(raw: str, object_: str) -> tuple[str, str]:
    """Predicate hors vocabulaire → plus proche (fuzzy) ou FALLBACK avec le
    predicate brut préservé dans l'object (§10.2). Jamais de création."""
    predicate = raw.strip().lower()
    if predicate in PREDICATES:
        return predicate, object_
    close = difflib.get_close_matches(predicate, list(PREDICATES), n=1, cutoff=FUZZY_CUTOFF)
    if close:
        logger.info("predicate_fuzzy_mapped", raw=raw, mapped=close[0])
        return close[0], object_
    logger.info("predicate_fallback", raw=raw)
    return FALLBACK_PREDICATE, f"{predicate}: {object_}"


class FactExtractor:
    def __init__(self, manager: ModelManager, settings: Settings) -> None:
        self._manager = manager
        self._model = settings.EXTRACTION_MODEL

    async def extract(self, content: str, role: str, timestamp_ms: int) -> Extraction:
        """Lève en cas d'échec LLM/parse — le worker gère le retry (§15.1)."""
        ts = datetime.fromtimestamp(timestamp_ms / 1000, tz=UTC).isoformat()
        raw = await self._manager.generate(
            EXTRACTION_PROMPT.format(role=role, ts=ts, content=content),
            self._model,
            format="json",
            options={"temperature": 0.0, "num_predict": 768},
        )
        data = json.loads(raw)
        return Extraction(
            facts=self._validate_facts(data.get("facts", [])),
            entities=self._validate_entities(data.get("entities", [])),
        )

    def _validate_facts(self, raw_facts: object) -> list[ExtractedFact]:
        facts: list[ExtractedFact] = []
        if not isinstance(raw_facts, list):
            logger.warning("extraction_facts_not_a_list")
            return facts
        for item in raw_facts:
            try:
                subject = str(item["subject"]).strip()
                object_ = str(item["object"]).strip()
                if not subject or not object_:
                    raise ValueError("subject/object vide")
                confidence = float(item.get("confidence", 1.0))
                if not 0.0 <= confidence <= 1.0:
                    raise ValueError(f"confidence hors bornes : {confidence}")
                predicate, object_ = map_predicate(str(item["predicate"]), object_)
                facts.append(ExtractedFact(subject, predicate, object_, confidence))
            except (KeyError, TypeError, ValueError) as exc:
                logger.warning("extraction_fact_skipped", error=str(exc))
        return facts

    def _validate_entities(self, raw_entities: object) -> list[ExtractedEntity]:
        entities: list[ExtractedEntity] = []
        if not isinstance(raw_entities, list):
            logger.warning("extraction_entities_not_a_list")
            return entities
        for item in raw_entities:
            try:
                name = str(item["name"]).strip()
                if not name:
                    raise ValueError("name vide")
                entity_type = item.get("entity_type")
                if entity_type is not None:
                    entity_type = str(entity_type).strip().lower()
                    if entity_type not in ENTITY_TYPES:
                        entity_type = None
                raw_aliases = item.get("aliases", [])
                aliases = (
                    [str(a).strip() for a in raw_aliases if str(a).strip()]
                    if isinstance(raw_aliases, list)
                    else []
                )
                entities.append(ExtractedEntity(name, entity_type, aliases))
            except (KeyError, TypeError, ValueError) as exc:
                logger.warning("extraction_entity_skipped", error=str(exc))
        return entities

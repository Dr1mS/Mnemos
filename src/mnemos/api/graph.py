"""Construction du graphe mémoire pour le visualiseur (GET /v1/graph).

Contrat défini par la page « Memory Constellation » (static/constellation.html) :
{ entities, facts, memories }
- entities : {id, type ∈ {personne, lieu, projet, organisation}, label,
  mentions, expired?, tether?} — expired+tether = fantôme rattaché au
  successeur (l'ancien job flotte derrière le nouveau)
- facts    : {id, predicate (libellé FR), object (id d'entité), type ∈
  {localisation, emploi, relation, projet, loisir} (palette), since,
  detail?, history: [{value, entity?, from, to}]}
- memories : {id, label, date (affichage FR), freshness (=decay_state),
  importance (=salience), anchor (id d'entité, optionnel)}

Lecture seule, aucun appel LLM. Les objects de faits sans entité connue
deviennent des entités-concepts synthétiques (id préfixé "c:").
Les faits rétractés (valid_until posé sans superseded_by) ne sont pas
affichés — interrogeables via l'API.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select

from mnemos.models.episodic import Episode
from mnemos.models.semantic import Entity, Fact
from mnemos.stores.episodic import EpisodicStore
from mnemos.stores.semantic import SemanticStore
from mnemos.tenancy import DEFAULT_TENANT, canonical_subject

MAX_EPISODES = 300  # les plus récents — au-delà le rendu devient une bouillie
LABEL_MAX = 90
OBJECT_LABEL_MAX = 48

ENTITY_TYPE_FR = {"person": "personne", "org": "organisation", "place": "lieu"}

PREDICATE_FR = {
    "works_at": "travaille chez",
    "lives_in": "habite à",
    "prefers": "préfère",
    "dislikes": "n'aime pas",
    "owns": "possède",
    "is_a": "est",
    "has_attribute": "a pour trait",
    "knows_about": "connaît",
    "has_goal": "a pour objectif",
    "has_skill": "maîtrise",
}

# Palette de liens de la page : localisation/emploi/relation/projet/loisir
FACT_TYPE = {
    "lives_in": "localisation",
    "works_at": "emploi",
    "is_a": "relation",
    "has_attribute": "relation",
    "prefers": "loisir",
    "dislikes": "loisir",
    "owns": "loisir",
    "knows_about": "loisir",
    "has_goal": "projet",
    "has_skill": "projet",
}

_MONTHS_FR = ["janv.", "févr.", "mars", "avr.", "mai", "juin",
              "juil.", "août", "sept.", "oct.", "nov.", "déc."]


def _fmt_month(ms: int) -> str:
    dt = datetime.fromtimestamp(ms / 1000, tz=UTC)
    return f"{_MONTHS_FR[dt.month - 1]} {dt.year}"


def _fmt_day(ms: int) -> str:
    dt = datetime.fromtimestamp(ms / 1000, tz=UTC)
    return f"{dt.day} {_MONTHS_FR[dt.month - 1]} {dt.year}"


def _truncate(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


async def build_graph(
    episodic: EpisodicStore, semantic: SemanticStore, tenant: str = DEFAULT_TENANT
) -> dict[str, Any]:
    # Tout est borné au tenant : entités, faits et épisodes du tenant demandé
    # uniquement — le visualiseur d'un tenant ne montre jamais un autre.
    subject_root = canonical_subject(tenant)
    async with semantic._sessions() as session:  # noqa: SLF001 — lecture seule
        db_entities = list(
            (
                await session.execute(select(Entity).where(Entity.tenant == tenant))
            ).scalars()
        )
        db_facts = list(
            (await session.execute(select(Fact).where(Fact.tenant == tenant))).scalars()
        )
    async with episodic._sessions() as session:  # noqa: SLF001
        db_episodes = list(
            (
                await session.execute(
                    select(Episode)
                    .where(Episode.tenant == tenant, Episode.archived == 0)
                    .order_by(Episode.created_at.desc())
                    .limit(MAX_EPISODES)
                )
            ).scalars()
        )

    entity_id_by_lower = {e.canonical_name.lower(): e.canonical_name for e in db_entities}
    entities: dict[str, dict[str, Any]] = {
        e.canonical_name: {
            "id": e.canonical_name,
            "type": ENTITY_TYPE_FR.get(e.entity_type or "", "projet"),
            "label": e.canonical_name,
            "mentions": max(1, e.episode_count),
        }
        for e in db_entities
    }

    def resolve_object(fact: Fact) -> str:
        """Id d'entité pour l'object du fait — synthétise une entité-concept
        si l'object n'est pas une entité connue."""
        known = entity_id_by_lower.get(fact.object.lower())
        if known is not None:
            return known
        concept_id = f"c:{fact.object.lower()}"
        if concept_id not in entities:
            family = FACT_TYPE.get(fact.predicate, "projet")
            entities[concept_id] = {
                "id": concept_id,
                "type": {"localisation": "lieu", "emploi": "organisation"}.get(
                    family, "projet"
                ),
                "label": _truncate(fact.object, OBJECT_LABEL_MAX),
                "mentions": 1,
            }
        return concept_id

    # Chaînes de supersession : ancêtres de chaque fait courant
    by_successor: dict[str, Fact] = {
        f.superseded_by: f for f in db_facts if f.superseded_by is not None
    }

    facts_out: list[dict[str, Any]] = []
    for fact in db_facts:
        if fact.valid_until is not None:
            continue  # invalidés : rendus via history/fantômes
        if fact.subject.lower() != subject_root.lower():
            continue  # la page trace <sujet canonique> → object ; faits tiers hors scope v1
        object_id = resolve_object(fact)
        history: list[dict[str, Any]] = []
        ancestor = by_successor.get(fact.id)
        while ancestor is not None:
            ghost_entity = entity_id_by_lower.get(ancestor.object.lower())
            history.append(
                {
                    "value": _truncate(ancestor.object, OBJECT_LABEL_MAX),
                    "entity": ghost_entity,
                    "from": _fmt_month(ancestor.valid_from),
                    "to": _fmt_month(ancestor.valid_until or ancestor.valid_from),
                }
            )
            if ghost_entity is not None:
                entity = entities[ghost_entity]
                # fantôme rattaché au successeur — sauf s'il est encore
                # l'object d'un autre fait courant
                still_current = any(
                    f.valid_until is None and f.object.lower() == ghost_entity.lower()
                    for f in db_facts
                )
                if not still_current:
                    entity["expired"] = True
                    entity["tether"] = object_id
            ancestor = by_successor.get(ancestor.id)
        facts_out.append(
            {
                "id": f"f:{fact.id}",
                "predicate": PREDICATE_FR.get(fact.predicate, fact.predicate),
                "object": object_id,
                "type": FACT_TYPE.get(fact.predicate, "projet"),
                "since": _fmt_month(fact.valid_from),
                "detail": f"confiance {fact.confidence:.0%}",
                "history": history,
            }
        )

    memories: list[dict[str, Any]] = []
    for episode in db_episodes:
        anchor = next(
            (
                entity_id_by_lower[ref.lower()]
                for ref in json.loads(episode.entity_refs)
                if ref.lower() in entity_id_by_lower
            ),
            None,
        )
        memories.append(
            {
                "id": f"m:{episode.id}",
                "label": _truncate(episode.content, LABEL_MAX),
                "date": _fmt_day(episode.created_at),
                "freshness": round(episode.decay_state, 3),
                "importance": round(episode.salience, 3),
                **({"anchor": anchor} if anchor else {}),
            }
        )

    return {"entities": list(entities.values()), "facts": facts_out, "memories": memories}

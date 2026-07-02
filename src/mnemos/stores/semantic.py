"""Semantic store (§10) — faits versionnés + entités + résolution de conflit.

Anti-pattern 2 : AUCUN write au sémantique sans passer par add_fact — la
résolution de conflit doit toujours s'exécuter.

Anti-pattern 6 : jamais de KNN sur facts_vec sans re-filtrer
`valid_until IS NULL` (le bug Mem0) — d'où le sur-fetch 4*k (§10.1).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Literal

import sqlite_vec  # type: ignore[import-untyped]
from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker
from ulid import ULID

from mnemos.clock import Clock
from mnemos.config import Settings
from mnemos.embeddings.dense import DenseEmbedder
from mnemos.logging import get_logger
from mnemos.models.semantic import Entity, Fact
from mnemos.ontology import PREDICATES, Cardinality

logger = get_logger(__name__)

SEARCH_OVERFETCH = 4  # KNN top-4*k : les faits invalidés consomment des slots (§10.1)

WriteAction = Literal["inserted", "superseded", "duplicate"]


@dataclass(frozen=True)
class FactWriteResult:
    action: WriteAction
    fact: Fact
    superseded: Fact | None = None


@dataclass(frozen=True)
class ScoredFact:
    fact: Fact
    score: float


class SemanticStore:
    def __init__(
        self,
        engine: AsyncEngine,
        embedder: DenseEmbedder,
        clock: Clock,
        settings: Settings,
    ) -> None:
        self._sessions = async_sessionmaker(engine, expire_on_commit=False)
        self._embedder = embedder
        self._clock = clock
        self._settings = settings

    # ── Normalisation entités (§10.2 étape 0) ────────────────────────────────

    async def resolve_name(self, name: str) -> str:
        """Nom → canonical_name si l'entité (ou un alias) existe, sinon le nom
        d'origine. Lookup casse-insensible. Sans ça, "Google"/"google"/"Google
        Inc" créent des faits parallèles jamais dédupliqués."""
        needle = name.strip().lower()
        if not needle:
            return name
        async with self._sessions() as session:
            row = (
                await session.execute(
                    select(Entity.canonical_name).where(
                        func.lower(Entity.canonical_name) == needle
                    )
                )
            ).scalar_one_or_none()
            if row is not None:
                return row
            row = (
                await session.execute(
                    text(
                        "SELECT e.canonical_name FROM entities e, json_each(e.aliases) a "
                        "WHERE lower(a.value) = :needle LIMIT 1"
                    ),
                    {"needle": needle},
                )
            ).scalar_one_or_none()
            return row if row is not None else name

    async def upsert_entity(
        self,
        name: str,
        entity_type: str | None = None,
        aliases: list[str] | None = None,
    ) -> Entity:
        now = self._clock.now_ms()
        canonical = await self.resolve_name(name)
        async with self._sessions() as session, session.begin():
            entity = await session.get(Entity, canonical)
            if entity is None:
                entity = Entity(
                    canonical_name=name.strip(),
                    aliases=json.dumps(aliases or [], ensure_ascii=False),
                    entity_type=entity_type,
                    first_seen=now,
                    last_seen=now,
                    episode_count=1,
                )
                session.add(entity)
                return entity
            entity.last_seen = now
            entity.episode_count += 1
            if entity.entity_type is None and entity_type is not None:
                entity.entity_type = entity_type
            if aliases:
                known = {a.lower() for a in json.loads(entity.aliases)}
                known.add(entity.canonical_name.lower())
                fresh = [a for a in aliases if a.lower() not in known]
                if fresh:
                    entity.aliases = json.dumps(
                        json.loads(entity.aliases) + fresh, ensure_ascii=False
                    )
            session.add(entity)
            return entity

    # ── Écriture de faits (§10.2) ─────────────────────────────────────────────

    async def add_fact(
        self,
        subject: str,
        predicate: str,
        object_: str,
        source_episode_ids: list[str],
        confidence: float = 1.0,
    ) -> FactWriteResult:
        now = self._clock.now_ms()
        # Étape 0 : normalisation par alias AVANT toute comparaison.
        subject = await self.resolve_name(subject)
        object_ = await self.resolve_name(object_)
        cardinality = PREDICATES[predicate]  # KeyError = bug appelant (extractor valide)

        async with self._sessions() as session, session.begin():
            current = list(
                (
                    await session.execute(
                        select(Fact).where(
                            func.lower(Fact.subject) == subject.lower(),
                            Fact.predicate == predicate,
                            Fact.valid_until.is_(None),
                        )
                    )
                ).scalars()
            )

            # Cas 2 : même object (après normalisation) → duplicate.
            for fact in current:
                if fact.object.lower() == object_.lower():
                    fact.confidence = max(fact.confidence, confidence)
                    sources = json.loads(fact.source_episodes)
                    merged = sources + [e for e in source_episode_ids if e not in sources]
                    fact.source_episodes = json.dumps(merged)
                    session.add(fact)
                    logger.info("fact_duplicate", fact_id=fact.id, predicate=predicate)
                    return FactWriteResult(action="duplicate", fact=fact)

            new_fact = Fact(
                id=str(ULID()),
                subject=subject,
                predicate=predicate,
                object=object_,
                valid_from=now,
                valid_until=None,
                confidence=confidence,
                source_episodes=json.dumps(source_episode_ids),
                created_at=now,
            )

            superseded: Fact | None = None
            if current and cardinality is Cardinality.FUNCTIONAL:
                # Cas 3 : object différent, predicate functional → supersession.
                # (Un seul fait courant possible par construction — on prend le
                # premier et on logge si l'invariant est violé.)
                if len(current) > 1:
                    logger.error(
                        "functional_invariant_violated",
                        subject=subject,
                        predicate=predicate,
                        count=len(current),
                    )
                superseded = current[0]
                superseded.valid_until = now
                superseded.superseded_by = new_fact.id
                session.add(superseded)
            # Cas 4 (multi) : insert additionnel, JAMAIS de supersession.

            session.add(new_fact)
            embedding = await self._embedder.embed(
                f"{new_fact.subject} {new_fact.predicate} {new_fact.object}"
            )
            await session.execute(
                text("INSERT INTO facts_vec(fact_id, embedding) VALUES (:id, :emb)"),
                {"id": new_fact.id, "emb": sqlite_vec.serialize_float32(embedding)},
            )
            action: WriteAction = "superseded" if superseded is not None else "inserted"
            logger.info("fact_written", fact_id=new_fact.id, action=action, predicate=predicate)
            return FactWriteResult(action=action, fact=new_fact, superseded=superseded)

    # ── Lecture ───────────────────────────────────────────────────────────────

    async def get_current_facts(
        self, subject: str | None = None, predicate: str | None = None
    ) -> list[Fact]:
        stmt = select(Fact).where(Fact.valid_until.is_(None)).order_by(Fact.created_at)
        if subject is not None:
            resolved = await self.resolve_name(subject)
            stmt = stmt.where(func.lower(Fact.subject) == resolved.lower())
        if predicate is not None:
            stmt = stmt.where(Fact.predicate == predicate)
        async with self._sessions() as session:
            return list((await session.execute(stmt)).scalars())

    async def get_history(self, subject: str, predicate: str) -> list[Fact]:
        """Tous les faits (incl. invalidés) pour cette paire, du plus ancien
        au plus récent — la chaîne de versioning."""
        resolved = await self.resolve_name(subject)
        async with self._sessions() as session:
            return list(
                (
                    await session.execute(
                        select(Fact)
                        .where(
                            func.lower(Fact.subject) == resolved.lower(),
                            Fact.predicate == predicate,
                        )
                        .order_by(Fact.valid_from, Fact.created_at)
                    )
                ).scalars()
            )

    async def search_facts(self, query: str, k: int = 10) -> list[ScoredFact]:
        embedding = await self._embedder.embed(query)
        async with self._sessions() as session:
            knn = await session.execute(
                text(
                    "SELECT fact_id, distance FROM facts_vec "
                    "WHERE embedding MATCH :emb AND k = :k"
                ),
                {"emb": sqlite_vec.serialize_float32(embedding), "k": SEARCH_OVERFETCH * k},
            )
            distances = {row[0]: float(row[1]) for row in knn}
            if not distances:
                return []
            # JOIN + filtre valid_until IS NULL en SQL (anti-pattern 6)
            facts = list(
                (
                    await session.execute(
                        select(Fact).where(
                            Fact.id.in_(distances), Fact.valid_until.is_(None)
                        )
                    )
                ).scalars()
            )
        scored = [ScoredFact(fact=f, score=1.0 - distances[f.id]) for f in facts]
        scored.sort(key=lambda s: s.score, reverse=True)
        return scored[:k]

    async def count_duplicate_current(self) -> int:
        """Nombre de paires (subject, predicate, object) courantes en doublon —
        doit rester 0 (quality gate §21)."""
        async with self._sessions() as session:
            row = await session.execute(
                text(
                    "SELECT COUNT(*) FROM ("
                    "  SELECT lower(subject), predicate, lower(object) FROM facts"
                    "  WHERE valid_until IS NULL"
                    "  GROUP BY lower(subject), predicate, lower(object)"
                    "  HAVING COUNT(*) > 1"
                    ")"
                )
            )
            return int(row.scalar_one())

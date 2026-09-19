"""Orchestration des lectures (§14.2) — fan-out parallèle vers les stores.

UNKNOWN consulte épisodique + sémantique : c'est le fallback safe.
SEMANTIC_FACT consulte aussi l'épisodique : le KNN des faits n'a pas de seuil et
renvoie toujours le fait le plus proche, même hors sujet, donc « aucun fait »
n'est jamais une condition de repli atteignable. Les épisodes qui n'ont produit
que des faits périmés sont écartés pour ne pas contredire la vérité active.
SEMANTIC_HISTORY inclut l'historique complet des faits matchés (chaîne de
versioning) en plus des faits courants.
Procedural : best-effort, vide tant que le ProceduralStore n'existe pas
(Phase 6).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from mnemos.models.semantic import Fact
from mnemos.router.classifier import QueryType, classify
from mnemos.stores.episodic import EpisodicStore, ScoredEpisode
from mnemos.stores.procedural import ProceduralStore
from mnemos.stores.semantic import ScoredFact, SemanticStore
from mnemos.stores.working import WMItem, WorkingMemoryRegistry
from mnemos.tenancy import DEFAULT_TENANT

_EPISODIC_TYPES = frozenset(
    {
        QueryType.EPISODIC_TEMPORAL,
        QueryType.EPISODIC_FUZZY,
        QueryType.SEMANTIC_FACT,
        QueryType.UNKNOWN,
    }
)
_SEMANTIC_TYPES = frozenset(
    {QueryType.SEMANTIC_FACT, QueryType.SEMANTIC_HISTORY, QueryType.UNKNOWN}
)


@dataclass(frozen=True)
class QueryResult:
    type: QueryType
    episodes: list[ScoredEpisode] = field(default_factory=list)
    facts: list[ScoredFact] = field(default_factory=list)
    history: list[Fact] = field(default_factory=list)  # SEMANTIC_HISTORY uniquement
    working: list[WMItem] = field(default_factory=list)
    procedural: list[str] = field(default_factory=list)  # noms de skills (Phase 6)


class RouterOrchestrator:
    def __init__(
        self,
        episodic: EpisodicStore,
        semantic: SemanticStore,
        working: WorkingMemoryRegistry,
        procedural: ProceduralStore | None = None,
    ) -> None:
        self._episodic = episodic
        self._semantic = semantic
        self._working = working
        self._procedural = procedural

    async def query(
        self,
        q: str,
        session_id: str | None = None,
        k: int = 10,
        tenant: str = DEFAULT_TENANT,
    ) -> QueryResult:
        qtype = classify(q)
        drop_stale = qtype is QueryType.SEMANTIC_FACT
        # Marge pour compenser les épisodes périmés écartés après la recherche.
        episodes_k = k * 2 if drop_stale else k

        episodes_task = (
            asyncio.create_task(
                self._episodic.search(q, k=episodes_k, session_id=session_id, tenant=tenant)
            )
            if qtype in _EPISODIC_TYPES
            else None
        )
        facts_task = (
            asyncio.create_task(self._semantic.search_facts(q, k=k, tenant=tenant))
            if qtype in _SEMANTIC_TYPES
            else None
        )

        episodes: list[ScoredEpisode] = await episodes_task if episodes_task else []
        facts: list[ScoredFact] = await facts_task if facts_task else []
        if drop_stale:
            stale = await self._semantic.stale_source_episode_ids(tenant)
            episodes = [e for e in episodes if e.episode.id not in stale][:k]

        history: list[Fact] = []
        if qtype is QueryType.SEMANTIC_HISTORY:
            # Chaîne de versioning des paires (subject, predicate) matchées.
            seen: set[tuple[str, str]] = set()
            for scored in facts:
                pair = (scored.fact.subject, scored.fact.predicate)
                if pair not in seen:
                    seen.add(pair)
                    history.extend(
                        await self._semantic.get_history(*pair, tenant=tenant)
                    )

        working: list[WMItem] = []
        if qtype is QueryType.WORKING and session_id is not None:
            wm = self._working.peek(session_id, tenant=tenant)
            if wm is not None:
                working = wm.get_context()

        # Procedural : toujours best-effort (§14.2) — jamais bloquant.
        procedural: list[str] = []
        if self._procedural is not None and qtype is QueryType.PROCEDURAL:
            procedural = [meta.name for meta in self._procedural.search(q, k=5)]

        return QueryResult(
            type=qtype,
            episodes=episodes,
            facts=facts,
            history=history,
            working=working,
            procedural=procedural,
        )

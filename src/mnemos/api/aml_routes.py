"""Routes API pour l'adaptateur officiel Agent Memory Leaderboard (AML / Challenge Cycle 2).

Fournit les endpoints conformes au contrat de la compétition :
- POST /add (et /aml/add) : écriture synchrone, immédiateté de recherche
- POST /search (et /aml/search) : recherche unifiée (sémantique + épisodique)
- GET /health (et /aml/health) : sonde de santé sans authentification
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated

from fastapi import APIRouter, Depends

from mnemos import __version__
from mnemos.api.aml_schemas import (
    AMLAddRequest,
    AMLAddResponse,
    AMLMemoryItem,
    AMLSearchRequest,
    AMLSearchResponse,
)
from mnemos.api.deps import (
    get_queue,
    get_semantic,
    get_store,
    get_wm,
    require_api_key,
)
from mnemos.stores.episodic import BatchEpisodeItem, EpisodicStore
from mnemos.stores.semantic import SemanticStore
from mnemos.stores.working import WorkingMemoryRegistry
from mnemos.tagger.salience import ScoringJob, ScoringQueue

aml_router = APIRouter(tags=["aml"])

StoreDep = Annotated[EpisodicStore, Depends(get_store)]
SemanticDep = Annotated[SemanticStore, Depends(get_semantic)]
WMDep = Annotated[WorkingMemoryRegistry, Depends(get_wm)]
QueueDep = Annotated[ScoringQueue, Depends(get_queue)]


def _ts_to_iso(ts_ms: int) -> str:
    """Convertit un timestamp en millisecondes en chaîne ISO 8601 UTC."""
    return datetime.fromtimestamp(ts_ms / 1000.0, tz=UTC).isoformat()


@aml_router.post(
    "/add", response_model=AMLAddResponse, dependencies=[Depends(require_api_key)]
)
@aml_router.post(
    "/aml/add", response_model=AMLAddResponse, dependencies=[Depends(require_api_key)]
)
async def aml_add(
    payload: AMLAddRequest,
    store: StoreDep,
    wm: WMDep,
    queue: QueueDep,
) -> AMLAddResponse:
    """Ingestion synchrone de mémoires selon le contrat AML.

    L'écriture est persistée en SQLite (WAL + sqlite-vec) avant de renvoyer
    HTTP 200 pour garantir que le souvenir est immédiatement cherchable.
    Le périmètre user_id AML correspond directement au tenant Mnemos.
    Optimisé via batch embedding vectoriel et transaction atomique unique.
    """
    tenant = payload.user_id
    session_id = payload.session_id

    # Récupération initiale de l'historique récent avant ce lot
    recent_episodes = await store.list_recent(session_id, n=5, tenant=tenant)
    history: list[str] = [e.content for e in recent_episodes]

    # Préparation du lot d'épisodes
    items = [
        BatchEpisodeItem(
            content=msg.content,
            role=msg.role,
            session_id=session_id,
            tenant=tenant,
            created_at=msg.timestamp,
        )
        for msg in payload.messages
    ]

    # Écriture synchrone groupée (batch embedding Ollama + transaction atomique SQLite)
    episodes = await store.write_batch(items)

    # Post-traitement : mémoire de travail et file de saillance asynchrone
    wm_session = wm.get_or_create(session_id, tenant=tenant)
    for episode in episodes:
        queue.enqueue(
            ScoringJob(
                episode_id=episode.id,
                content=episode.content,
                recent_history=list(history[-5:]),
                tenant=tenant,
            )
        )
        history.append(episode.content)
        wm_session.push(
            episode.content, episode.role, episode.created_at
        )

    return AMLAddResponse(
        success=True,
        request_id=payload.request_id,
        user_id=payload.user_id,
        session_id=payload.session_id,
    )


@aml_router.post(
    "/search", response_model=AMLSearchResponse, dependencies=[Depends(require_api_key)]
)
@aml_router.post(
    "/aml/search", response_model=AMLSearchResponse, dependencies=[Depends(require_api_key)]
)
async def aml_search(
    payload: AMLSearchRequest,
    store: StoreDep,
    semantic: SemanticDep,
) -> AMLSearchResponse:
    """Recherche unifiée conforme au contrat AML.

    Interroge à la fois la mémoire sémantique (faits consolidés non périmés)
    et la mémoire épisodique (souvenirs récents, y compris non encore consolidés)
    en isolant strictement par user_id (tenant).
    Retourne au maximum top_k (plafonné à 100) mémoires ordonnées par pertinence.
    """
    tenant = payload.user_id
    top_k = min(payload.top_k, 100)

    # 1. Recherche sémantique des faits actifs
    facts = await semantic.search_facts(payload.query, k=top_k, tenant=tenant)

    # 2. Recherche épisodique (hybride dense + sparse + récence)
    episodes = await store.search(payload.query, k=top_k, tenant=tenant)

    items: list[AMLMemoryItem] = []
    seen_contents: set[str] = set()

    # Intégration des faits sémantiques (preuve concise et consolidée)
    for f in facts:
        content = f"{f.fact.subject} {f.fact.predicate} {f.fact.object}"
        if content.lower() in seen_contents:
            continue
        seen_contents.add(content.lower())
        items.append(
            AMLMemoryItem(
                id=f"fact_{f.fact.id}",
                content=content,
                score=round(float(f.score), 4),
                created_at=_ts_to_iso(f.fact.created_at),
            )
        )

    # Intégration des épisodes bruts
    for e in episodes:
        content = e.episode.content
        if content.lower() in seen_contents:
            continue
        seen_contents.add(content.lower())
        items.append(
            AMLMemoryItem(
                id=f"ep_{e.episode.id}",
                content=content,
                score=round(float(e.score), 4),
                created_at=_ts_to_iso(e.episode.created_at),
            )
        )

    # Tri par score de pertinence décroissant
    items.sort(key=lambda x: x.score if x.score is not None else 0.0, reverse=True)

    return AMLSearchResponse(data=items[:top_k])


@aml_router.get("/health")
@aml_router.get("/aml/health")
async def aml_health() -> dict[str, str]:
    """Sonde de santé non-authentifiée conforme au contrat AML (tout 2xx valide)."""
    return {"status": "healthy", "version": __version__}

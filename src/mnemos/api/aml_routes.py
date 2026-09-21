"""Routes API pour l'adaptateur officiel Agent Memory Leaderboard (AML / Challenge Cycle 2).

Fournit les endpoints conformes au contrat de la compétition :
- POST /add (et /aml/add) : écriture synchrone, immédiateté de recherche
- POST /search (et /aml/search) : recherche unifiée (sémantique + épisodique)
- GET /health (et /aml/health) : sonde de santé sans authentification
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Request, Response

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
from mnemos.llm.ollama_client import is_probe_transient
from mnemos.logging import get_logger
from mnemos.stores.episodic import BatchEpisodeItem, EpisodicStore
from mnemos.stores.semantic import SemanticStore
from mnemos.stores.working import WorkingMemoryRegistry
from mnemos.tagger.salience import ScoringJob, ScoringQueue

logger = get_logger(__name__)

aml_router = APIRouter(tags=["aml"])

# La sonde d'embedding sollicite le GPU : un résultat sert 30 s, quel que soit
# le rythme d'appel de la plateforme ou des moniteurs externes.
HEALTH_CACHE_TTL_S = 30.0
# En deçà de cet âge, le dernier embedding réussi tient lieu de sonde (§Santé).
HEALTH_TRAFFIC_WINDOW_S = 15.0

StoreDep = Annotated[EpisodicStore, Depends(get_store)]
SemanticDep = Annotated[SemanticStore, Depends(get_semantic)]
WMDep = Annotated[WorkingMemoryRegistry, Depends(get_wm)]
QueueDep = Annotated[ScoringQueue, Depends(get_queue)]


def _ts_to_iso(ts_ms: int) -> str:
    """Convertit un timestamp en millisecondes en chaîne ISO 8601 UTC avec suffixe Z."""
    return datetime.fromtimestamp(ts_ms / 1000.0, tz=UTC).isoformat().replace("+00:00", "Z")


def _extract_text(val: str | list[dict[str, Any]]) -> str:
    """Extrait le texte brut d'une chaîne ou d'une liste ordonnée de ContentPart (multimodal)."""
    if isinstance(val, str):
        return val
    parts = [
        str(p.get("text", ""))
        for p in val
        if isinstance(p, dict) and p.get("type") == "text"
    ]
    text_content = " ".join(filter(None, parts))
    return text_content if text_content else str(val)


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
            content=_extract_text(msg.content),
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
    query_str = _extract_text(payload.query)

    # 1. Recherche sémantique des faits actifs
    facts = await semantic.search_facts(query_str, k=top_k, tenant=tenant)

    # 2. Recherche épisodique (hybride dense + sparse + récence)
    episodes = await store.search(query_str, k=top_k, tenant=tenant)

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
        # Contrat : un item sans content non vide fait échouer toute l'étape Search.
        if not content.strip() or content.lower() in seen_contents:
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


@dataclass
class _HealthSnapshot:
    status: str  # healthy | degraded | unhealthy
    failures: list[str] = field(default_factory=list)
    at: float = field(default_factory=time.monotonic)


async def _probe_health(request: Request) -> _HealthSnapshot:
    """Mêmes sondes que /v1/health : Ollama, embedding réel et les deux bases.

    Un embedding trop lent (timeout de la sonde) ou un runner Ollama en cours de
    démarrage signale de la charge ou un cold start, pas une panne : l'état reste
    2xx (« degraded ») pour ne pas faire croire à la plateforme que le service est
    tombé pendant un run chargé. La sonde est hors sémaphore (§7.2) : sous charge
    elle est donc précisément l'appel qui tombe sur un runner pas encore prêt."""
    state = request.app.state
    failures: list[str] = []
    details: dict[str, str] = {}
    busy = False

    version_error = await state.manager.version_probe()
    if version_error is not None:
        details["ollama"] = version_error
        if is_probe_transient(version_error):
            busy = True
        else:
            failures.append("ollama")

    # Preuve de vie par le trafic : un embedding réel réussi il y a moins de
    # HEALTH_TRAFFIC_WINDOW_S prouve qu'Ollama sert. Sonder en plus ajouterait
    # un /api/embed concurrent — et c'est cette concurrence-là qui fait démarrer
    # un runner à Ollama, donc qui provoque les 400 qu'on cherche à éviter.
    if state.manager.last_embed_ok_age_s <= HEALTH_TRAFFIC_WINDOW_S:
        embed_error = None
    else:
        embed_error = await state.manager.embed_probe()
    if embed_error is not None:
        details["embedding"] = embed_error
        if is_probe_transient(embed_error):
            busy = True
        else:
            failures.append("embedding")
    for name, store in (("episodic_db", state.store), ("semantic_db", state.semantic)):
        db_error = await store.ping()
        if db_error is not None:
            failures.append(name)
            details[name] = db_error
    if failures or busy:
        # Détails dans les logs uniquement : l'endpoint est public.
        logger.warning("aml_health_degraded", failures=failures, **details)
    status = "unhealthy" if failures else "degraded" if busy else "healthy"
    return _HealthSnapshot(status=status, failures=failures)


async def _health_snapshot(request: Request) -> _HealthSnapshot:
    state = request.app.state
    if not hasattr(state, "aml_health_lock"):
        state.aml_health_lock = asyncio.Lock()
    async with state.aml_health_lock:
        cached: _HealthSnapshot | None = getattr(state, "aml_health_cache", None)
        if cached is None or time.monotonic() - cached.at > HEALTH_CACHE_TTL_S:
            cached = await _probe_health(request)
            state.aml_health_cache = cached
        return cached


@aml_router.get("/health")
@aml_router.get("/aml/health")
async def aml_health(request: Request, response: Response) -> dict[str, Any]:
    """Sonde de santé non authentifiée (contrat AML : tout 2xx = sain).

    503 si Ollama, l'embedding ou une base est en panne : un moniteur externe
    sans clé voit alors aussi les pannes d'Ollama. Résultat mis en cache 30 s."""
    snapshot = await _health_snapshot(request)
    if snapshot.status == "unhealthy":
        response.status_code = 503
    body: dict[str, Any] = {"status": snapshot.status, "version": __version__}
    if snapshot.failures:
        body["failures"] = snapshot.failures
    return body

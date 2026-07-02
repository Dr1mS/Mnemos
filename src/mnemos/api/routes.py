"""Routes API v1 (§16) — parsing/validation/délégation uniquement.

POST /v1/episodes : write + embedding SYNCHRONES, scoring salience
ASYNCHRONE (§13.3). La réponse part avec salience=null ; l'épisode est
déjà cherchable.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from mnemos.api.deps import get_orchestrator, get_queue, get_store, get_wm, require_api_key
from mnemos.api.schemas import (
    EpisodeCreate,
    EpisodeOut,
    HealthOut,
    QueryIn,
    QueryResultOut,
    ScoredEpisodeOut,
)
from mnemos.router.orchestrator import RouterOrchestrator
from mnemos.stores.episodic import EpisodicStore
from mnemos.stores.working import WorkingMemoryRegistry
from mnemos.tagger.salience import ScoringJob, ScoringQueue

router = APIRouter(prefix="/v1", dependencies=[Depends(require_api_key)])

StoreDep = Annotated[EpisodicStore, Depends(get_store)]
QueueDep = Annotated[ScoringQueue, Depends(get_queue)]
WMDep = Annotated[WorkingMemoryRegistry, Depends(get_wm)]
OrchestratorDep = Annotated[RouterOrchestrator, Depends(get_orchestrator)]


@router.post("/episodes", status_code=201)
async def create_episode(
    payload: EpisodeCreate, store: StoreDep, queue: QueueDep, wm: WMDep
) -> EpisodeOut:
    # Historique AVANT le write : le nouvel épisode ne doit pas être
    # son propre contexte de scoring (§13.2).
    history = [e.content for e in await store.list_recent(payload.session_id, n=5)]
    episode = await store.write(
        content=payload.content, role=payload.role, session_id=payload.session_id
    )
    queue.enqueue(
        ScoringJob(episode_id=episode.id, content=episode.content, recent_history=history)
    )
    if payload.session_id is not None:
        wm.get_or_create(payload.session_id).push(
            episode.content, episode.role, episode.created_at
        )
    return EpisodeOut.from_episode(episode)


@router.post("/query")
async def query(payload: QueryIn, orchestrator: OrchestratorDep) -> QueryResultOut:
    result = await orchestrator.query(payload.q, session_id=payload.session_id, k=payload.k)
    return QueryResultOut.from_result(result)


@router.post("/sessions/{session_id}/reset", status_code=204)
async def reset_session(session_id: str, wm: WMDep) -> None:
    wm.reset(session_id)  # idempotent : session inconnue = no-op (§16.1)


@router.get("/episodes/search")
async def search_episodes(
    store: StoreDep,
    q: Annotated[str, Query(min_length=1)],
    k: Annotated[int, Query(ge=1, le=100)] = 10,
    session_id: str | None = None,
    min_salience: Annotated[float, Query(ge=0.0, le=1.0)] = 0.0,
) -> list[ScoredEpisodeOut]:
    results = await store.search(q, k=k, session_id=session_id, min_salience=min_salience)
    return [ScoredEpisodeOut.from_scored(s) for s in results]


@router.get("/episodes/{episode_id}")
async def get_episode(episode_id: str, store: StoreDep, request: Request) -> EpisodeOut:
    episode = await store.get_by_id(episode_id)
    if episode is None:
        raise HTTPException(status_code=404, detail="épisode inconnu")
    return EpisodeOut.from_episode(episode)


@router.get("/health")
async def health(request: Request, queue: QueueDep) -> HealthOut:
    settings = request.app.state.settings
    ollama_ok: bool = await request.app.state.manager.health_check()
    dbs = {
        "episodic": settings.EPISODIC_DB.exists(),
        "semantic": settings.SEMANTIC_DB.exists(),
    }
    return HealthOut(
        ok=ollama_ok and all(dbs.values()),
        ollama=ollama_ok,
        dbs=dbs,
        salience_queue_depth=queue.depth,
    )

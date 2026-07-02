"""Routes API v1 (§16) — parsing/validation/délégation uniquement.

POST /v1/episodes : write + embedding SYNCHRONES, scoring salience
ASYNCHRONE (§13.3). La réponse part avec salience=null ; l'épisode est
déjà cherchable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import FileResponse

from mnemos.api.deps import get_orchestrator, get_queue, get_store, get_wm, require_api_key
from mnemos.api.schemas import (
    ConsolidationReportOut,
    DecayReportOut,
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
async def health(request: Request, queue: QueueDep, store: StoreDep) -> HealthOut:
    import json as json_

    settings = request.app.state.settings
    ollama_ok: bool = await request.app.state.manager.health_check()
    dbs = {
        "episodic": settings.EPISODIC_DB.exists(),
        "semantic": settings.SEMANTIC_DB.exists(),
    }
    marker = settings.DATA_DIR / "worker_last_run"
    status_file = settings.DATA_DIR / "worker_status.json"
    return HealthOut(
        ok=ollama_ok and all(dbs.values()),
        ollama=ollama_ok,
        dbs=dbs,
        salience_queue_depth=queue.depth,
        worker_last_run=marker.read_text().strip() if marker.exists() else None,
        worker=json_.loads(status_file.read_text()) if status_file.exists() else None,
        pending=await store.pending_counts(),
    )


@router.get("/graph")
async def graph(request: Request, store: StoreDep) -> dict[str, object]:
    """Graphe mémoire pour le visualiseur — contrat {entities, facts, memories}."""
    from mnemos.api.graph import build_graph

    payload = await build_graph(store, request.app.state.semantic)
    payload["generated_at"] = request.app.state.clock.now_ms()
    return payload


# Hors /v1 : la page du visualiseur (fetch /v1/graph elle-même)
viz_router = APIRouter()


@viz_router.get("/viz", include_in_schema=False)
async def constellation() -> FileResponse:
    return FileResponse(Path(__file__).parent / "static" / "constellation.html")


# ── Admin (§16.1) — auth requise via require_api_key global ──────────────────


@router.post("/admin/consolidate")
async def admin_consolidate(request: Request) -> ConsolidationReportOut:
    report = await request.app.state.worker.run_once()
    return ConsolidationReportOut(
        candidates=report.candidates,
        consolidated=report.consolidated,
        extraction_failures=report.extraction_failures,
        facts_inserted=report.facts_inserted,
        facts_superseded=report.facts_superseded,
        facts_duplicate=report.facts_duplicate,
        entities_upserted=report.entities_upserted,
    )


@router.post("/admin/decay")
async def admin_decay(store: StoreDep, dry_run: bool = False) -> DecayReportOut:
    report = await store.apply_decay(dry_run=dry_run)
    return DecayReportOut(scanned=report.scanned, dry_run=report.dry_run, now_ms=report.now_ms)

"""Dépendances FastAPI (§16) — accès à l'état applicatif + auth optionnelle.

Anti-pattern 4 : aucune logique métier ici ni dans les routes —
uniquement du câblage.
"""

from __future__ import annotations

from fastapi import Header, HTTPException, Request

from mnemos.config import Settings
from mnemos.router.orchestrator import RouterOrchestrator
from mnemos.stores.episodic import EpisodicStore
from mnemos.stores.semantic import SemanticStore
from mnemos.stores.working import WorkingMemoryRegistry
from mnemos.tagger.salience import ScoringQueue


def get_app_settings(request: Request) -> Settings:
    settings: Settings = request.app.state.settings
    return settings


def get_store(request: Request) -> EpisodicStore:
    store: EpisodicStore = request.app.state.store
    return store


def get_semantic(request: Request) -> SemanticStore:
    semantic: SemanticStore = request.app.state.semantic
    return semantic


def get_queue(request: Request) -> ScoringQueue:
    queue: ScoringQueue = request.app.state.queue
    return queue


def get_wm(request: Request) -> WorkingMemoryRegistry:
    wm: WorkingMemoryRegistry = request.app.state.wm
    return wm


def get_orchestrator(request: Request) -> RouterOrchestrator:
    orchestrator: RouterOrchestrator = request.app.state.orchestrator
    return orchestrator


async def require_api_key(
    request: Request,
    x_api_key: str | None = Header(default=None),
    authorization: str | None = Header(default=None),
) -> None:
    """Auth optionnelle (§16) : si API_KEY est défini dans la config,
    le header X-API-Key ou Authorization (Bearer / Token) doit correspondre.
    Sinon ouvert (localhost). Compatible avec le contrat AML."""
    expected = get_app_settings(request).API_KEY
    if expected is None:
        return

    if x_api_key == expected:
        return

    if authorization is not None:
        parts = authorization.split(maxsplit=1)
        if len(parts) == 2 and parts[0].lower() in ("bearer", "token") and parts[1] == expected:
            return

    raise HTTPException(status_code=401, detail="API Key invalide ou manquante")

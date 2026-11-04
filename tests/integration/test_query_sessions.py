"""La recherche d'épisodes ne filtre jamais par session (README, §14.2).

Régression : `memory_query` passait toujours `session_id="claude"` et
l'orchestrateur le transmettait à `episodic.search`, ce qui écartait tout
épisode écrit sous une autre session (ex. `archiviste-j1`). La session reste
utilisée pour le contexte de travail (route WORKING).
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from types import SimpleNamespace

import pytest
from sqlalchemy import text
from tests.conftest import StubEmbedder

from mnemos.clock import Clock
from mnemos.config import Settings
from mnemos.mcp_server import DEFAULT_SESSION, AppContext, memory_query
from mnemos.models.base import make_async_engine
from mnemos.models.episodic import EPISODIC_SCHEMA_SQL
from mnemos.models.semantic import SEMANTIC_SCHEMA_SQL
from mnemos.router.orchestrator import RouterOrchestrator
from mnemos.stores.episodic import EpisodicStore
from mnemos.stores.semantic import SemanticStore
from mnemos.stores.working import WorkingMemoryRegistry

TENANT = "zygos-test"


@pytest.fixture
async def app_ctx(tmp_path: Path) -> AsyncIterator[AppContext]:
    settings = Settings(
        _env_file=None,
        DATA_DIR=tmp_path,
        EPISODIC_DB=tmp_path / "episodic.db",
        SEMANTIC_DB=tmp_path / "semantic.db",
        PROCEDURAL_DIR=tmp_path / "procedural",
        TENANT=TENANT,
    )
    clock = Clock()
    embedder = StubEmbedder()
    epi_engine = make_async_engine(settings.EPISODIC_DB)
    sem_engine = make_async_engine(settings.SEMANTIC_DB)
    async with epi_engine.begin() as conn:
        for stmt in EPISODIC_SCHEMA_SQL:
            await conn.execute(text(stmt))
    async with sem_engine.begin() as conn:
        for stmt in SEMANTIC_SCHEMA_SQL:
            await conn.execute(text(stmt))
    episodic = EpisodicStore(epi_engine, embedder, clock, settings)  # type: ignore[arg-type]
    semantic = SemanticStore(sem_engine, embedder, clock, settings)  # type: ignore[arg-type]
    working = WorkingMemoryRegistry()
    orchestrator = RouterOrchestrator(episodic, semantic, working)
    yield AppContext(
        settings=settings, episodic=episodic, semantic=semantic, working=working,
        orchestrator=orchestrator, worker=None, queue=None, client=None,  # type: ignore[arg-type]
        engines=[epi_engine, sem_engine],
    )
    for engine in (epi_engine, sem_engine):
        await engine.dispose()


def _ctx(app: AppContext) -> SimpleNamespace:
    return SimpleNamespace(request_context=SimpleNamespace(lifespan_context=app))


async def test_memory_query_voit_un_episode_d_une_autre_session(app_ctx: AppContext) -> None:
    content = "le jalon deux de zygos-app est validé par Adrien"
    await app_ctx.episodic.write(content, "user", session_id="autre", tenant=TENANT)

    out = await memory_query(content, _ctx(app_ctx))  # type: ignore[arg-type]

    assert content in out, out


async def test_orchestrator_ne_filtre_pas_les_episodes_par_session(
    app_ctx: AppContext,
) -> None:
    for session in ("archiviste-j1", "claude"):
        await app_ctx.episodic.write(
            f"écrit par {session}", "user", session_id=session, tenant=TENANT
        )

    result = await app_ctx.orchestrator.query(
        "écrit par archiviste-j1", session_id="claude", k=10, tenant=TENANT
    )

    sessions = {s.episode.session_id for s in result.episodes}
    assert sessions == {"archiviste-j1", "claude"}


async def test_route_working_garde_la_session(app_ctx: AppContext) -> None:
    app_ctx.working.get_or_create(DEFAULT_SESSION, tenant=TENANT).push(
        "contexte de la session claude", "user", 1
    )
    app_ctx.working.get_or_create("autre", tenant=TENANT).push(
        "contexte d'une autre session", "user", 1
    )

    out = await memory_query("où on en est ?", _ctx(app_ctx), tenant=TENANT)  # type: ignore[arg-type]

    assert "[route : working]" in out, out
    assert "contexte de la session claude" in out
    assert "contexte d'une autre session" not in out

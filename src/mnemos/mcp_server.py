"""Serveur MCP — expose Mnemos comme mémoire persistante pour Claude.

Transport stdio (stdout = protocole, logs structlog sur stderr). Les
composants Mnemos sont embarqués directement (pas besoin de `mnemos serve`).
La config vient des variables d'environnement (pydantic-settings) — le
`.mcp.json` du projet pointe DATA_DIR/EPISODIC_DB/… vers la mémoire cible.

Tools :
- memory_write       : écrit un épisode (salience scorée en arrière-plan)
- memory_query       : question routée multi-store (FR/EN)
- memory_facts       : faits courants, ou historique d'une paire
- memory_consolidate : force un run du worker (lent sur CPU — extraction LLM)
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime

from mcp.server.fastmcp import Context, FastMCP

from mnemos.clock import Clock
from mnemos.config import Settings, get_settings
from mnemos.consolidation.extractor import FactExtractor
from mnemos.consolidation.worker import ConsolidationWorker
from mnemos.embeddings.dense import DenseEmbedder
from mnemos.llm.model_manager import ModelManager
from mnemos.llm.ollama_client import OllamaClient
from mnemos.logging import configure_logging, get_logger
from mnemos.models.base import make_async_engine
from mnemos.models.semantic import Fact
from mnemos.router.orchestrator import RouterOrchestrator
from mnemos.stores.episodic import EpisodicStore
from mnemos.stores.procedural import ProceduralStore
from mnemos.stores.semantic import SemanticStore
from mnemos.stores.working import WorkingMemoryRegistry
from mnemos.tagger.salience import SalienceTagger, ScoringJob, ScoringQueue

logger = get_logger(__name__)

DEFAULT_SESSION = "claude"


@dataclass
class AppContext:
    settings: Settings
    episodic: EpisodicStore
    semantic: SemanticStore
    orchestrator: RouterOrchestrator
    worker: ConsolidationWorker
    queue: ScoringQueue
    client: OllamaClient
    engines: list[object]


@asynccontextmanager
async def app_lifespan(_server: FastMCP) -> AsyncIterator[AppContext]:
    settings = get_settings()
    configure_logging(settings.LOG_LEVEL)
    clock = Clock()
    client = OllamaClient(settings)
    manager = ModelManager(settings, client)
    embedder = DenseEmbedder(manager, settings)
    epi_engine = make_async_engine(settings.EPISODIC_DB)
    sem_engine = make_async_engine(settings.SEMANTIC_DB)
    episodic = EpisodicStore(epi_engine, embedder, clock, settings)
    semantic = SemanticStore(sem_engine, embedder, clock, settings)
    orchestrator = RouterOrchestrator(
        episodic, semantic, WorkingMemoryRegistry(),
        ProceduralStore(settings.PROCEDURAL_DIR, clock),
    )
    tagger = SalienceTagger(manager, settings)
    worker = ConsolidationWorker(
        episodic, semantic, FactExtractor(manager, settings), settings, clock,
        tagger=tagger,
    )
    queue = ScoringQueue(tagger, episodic)
    await queue.start()
    logger.info("mcp_server_started", episodic_db=str(settings.EPISODIC_DB))
    try:
        yield AppContext(
            settings=settings, episodic=episodic, semantic=semantic,
            orchestrator=orchestrator, worker=worker, queue=queue,
            client=client, engines=[epi_engine, sem_engine],
        )
    finally:
        await queue.stop(drain_timeout_s=5)
        for engine in [epi_engine, sem_engine]:
            await engine.dispose()
        await client.aclose()
        logger.info("mcp_server_stopped")


mcp = FastMCP(
    "mnemos-memory",
    lifespan=app_lifespan,
    instructions=(
        "Persistent local memory for this user (Mnemos). Use memory_query at "
        "the START of a conversation or when the user references past context, "
        "preferences, or their own life/projects. Use memory_write to store "
        "NEW lasting information the user reveals (identity, preferences, "
        "life/project facts) — write the user's statement, not your reply. "
        "When the user CORRECTS a stored fact ('ce n'est plus vrai', 'c'était "
        "pas moi'), use memory_forget — never write instruction-like episodes "
        "such as 'il faut oublier X'. Do not write secrets or throwaway chatter."
    ),
)


def _app(ctx: Context) -> AppContext:  # type: ignore[type-arg]
    app_ctx: AppContext = ctx.request_context.lifespan_context
    return app_ctx


def _fmt_date(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=UTC).strftime("%Y-%m-%d")


def _fmt_fact(fact: Fact) -> str:
    status = "" if fact.valid_until is None else " [invalidé]"
    return f"{fact.subject} — {fact.predicate} — {fact.object}{status}"


@mcp.tool()
async def memory_write(
    content: str, ctx: Context, session_id: str = DEFAULT_SESSION  # type: ignore[type-arg]
) -> str:
    """Store a new episode in the user's persistent memory.

    Write meaningful statements from the user (identity, preferences, life or
    project facts, decisions). The salience scoring and fact extraction happen
    asynchronously — the episode is searchable immediately.
    """
    app = _app(ctx)
    history = [e.content for e in await app.episodic.list_recent(session_id, n=5)]
    episode = await app.episodic.write(content, role="user", session_id=session_id)
    app.queue.enqueue(ScoringJob(episode.id, episode.content, history))
    return f"mémorisé ({episode.id}) — salience et consolidation en arrière-plan"


@mcp.tool()
async def memory_query(
    q: str, ctx: Context, k: int = 8  # type: ignore[type-arg]
) -> str:
    """Query the user's memory (French or English natural language).

    Routes automatically: current facts, fact history, past episodes, or
    recent working context. Use this to recall who the user is, their
    preferences, projects, or what was said before.
    """
    app = _app(ctx)
    result = await app.orchestrator.query(q, session_id=DEFAULT_SESSION, k=k)
    lines = [f"[route : {result.type.value}]"]
    if result.facts:
        lines.append("Faits :")
        lines += [f"  • {_fmt_fact(s.fact)} (score {s.score:.2f})" for s in result.facts]
    if result.history:
        lines.append("Historique du fait :")
        lines += [f"  • {_fmt_fact(f)} ({_fmt_date(f.valid_from)})" for f in result.history]
    if result.episodes:
        lines.append("Épisodes :")
        lines += [
            f"  • [{_fmt_date(s.episode.created_at)}] {s.episode.content}"
            for s in result.episodes
        ]
    if result.working:
        lines.append("Contexte de session :")
        lines += [f"  • {w.content}" for w in result.working]
    if result.procedural:
        lines.append("Skills : " + ", ".join(result.procedural))
    if len(lines) == 1:
        lines.append("(aucun souvenir pertinent)")
    return "\n".join(lines)


@mcp.tool()
async def memory_facts(
    ctx: Context,  # type: ignore[type-arg]
    subject: str | None = None,
    predicate: str | None = None,
    history: bool = False,
) -> str:
    """List current facts about a subject (default: everything known).

    Set history=true with both subject and predicate to see the full
    versioned chain of a fact (e.g. how works_at changed over time).
    """
    app = _app(ctx)
    if history:
        if not subject or not predicate:
            return "erreur : history=true exige subject ET predicate"
        facts = await app.semantic.get_history(subject, predicate)
    else:
        facts = await app.semantic.get_current_facts(subject, predicate)
    if not facts:
        return "(aucun fait)"
    return "\n".join(f"• {_fmt_fact(f)}" for f in facts)


@mcp.tool()
async def memory_forget(
    ctx: Context,  # type: ignore[type-arg]
    predicate: str,
    object: str,
    subject: str = "user",
) -> str:
    """Retract a fact that is no longer true or was wrong.

    Use when the user corrects the memory ("je n'aime plus X", "c'était pas
    moi", "ce n'est plus vrai"). The fact is invalidated (kept in history),
    not deleted. The object must match an existing current fact exactly —
    if unsure, the tool lists the candidates so you can retry.
    """
    app = _app(ctx)
    retracted = await app.semantic.retract_fact(subject, predicate, object)
    if retracted is not None:
        return f"rétracté : {_fmt_fact(retracted)}"
    candidates = await app.semantic.get_current_facts(subject, predicate)
    if not candidates:
        return f"aucun fait courant {subject}/{predicate}"
    listing = "\n".join(f"  • {f.object}" for f in candidates)
    return (
        f"object non trouvé. Faits courants {subject}/{predicate} :\n{listing}\n"
        f"Réessaie avec l'object exact."
    )


@mcp.tool()
async def memory_consolidate(ctx: Context) -> str:  # type: ignore[type-arg]
    """Force a consolidation run: extract facts from recent salient episodes.

    SLOW on CPU (~15s per pending episode, LLM extraction) — normally runs on
    a schedule; only call when the user explicitly asks to consolidate now.
    """
    app = _app(ctx)
    report = await app.worker.run_once()
    return (
        f"candidats={report.candidates} consolidés={report.consolidated} "
        f"échecs={report.extraction_failures} faits: +{report.facts_inserted} "
        f"~{report.facts_superseded} ={report.facts_duplicate} "
        f"entités={report.entities_upserted}"
    )


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()

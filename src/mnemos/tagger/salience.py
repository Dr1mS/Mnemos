"""Salience tagger (§13) — scoring LLM + queue asynchrone hors write path.

Un seul appel LLM (SALIENCE_MODEL, JSON mode) par épisode. Le scoring est
asynchrone (§13.3) : jamais dans le chemin critique de POST /v1/episodes.
Si la queue est pleine, on DROP le scoring (l'épisode garde salience=0.5) —
jamais de backpressure sur le write.

Robustesse : parse JSON échoué → scores neutres 0.5 + log, jamais d'exception
qui remonte au write path.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Protocol, TypedDict

from mnemos.config import Settings
from mnemos.llm.model_manager import ModelManager
from mnemos.logging import get_logger

logger = get_logger(__name__)


class SalienceScores(TypedDict):
    surprise: float  # [0..1]
    arousal: float  # [0..1] — intensité émotionnelle, positive OU négative.
    #                 Nommé "arousal" et pas "valence" : une valence serait
    #                 signée ; ici le signe est volontairement perdu.
    self_ref: float  # [0..1]
    recurrence: float  # [0..1]
    combined: float  # [0..1]


SALIENCE_PROMPT = """You score the salience of a single message for memory consolidation.
Return JSON with four floats in [0,1]:

- surprise: how unexpected/novel is this content vs typical conversation
- arousal: emotional intensity (positive or negative, both score high)
- self_ref: how much the user reveals about themselves (preferences, identity, life facts)
- recurrence: 0 if this topic is new in the recent history, higher if it repeats

Recent history (last 5 turns):
{recent_history}

Current message:
{content}

Output ONLY JSON: {{"surprise": 0.X, "arousal": 0.X, "self_ref": 0.X, "recurrence": 0.X}}"""

# Historique vide : instruction explicite, PAS un placeholder type "(empty)" —
# constaté sur qwen3:4b : "(empty)" fait rendre 0.0 partout (y compris self_ref
# sur une révélation perso évidente), là où une instruction explicite scorre
# normalement. Cas premier-message-de-session.
EMPTY_HISTORY_PLACEHOLDER = (
    "(no messages yet — this is the first message: set recurrence to 0 "
    "and score the other dimensions normally)"
)

NEUTRAL_SCORES = SalienceScores(
    surprise=0.5, arousal=0.5, self_ref=0.5, recurrence=0.5, combined=0.5
)


def combine(surprise: float, arousal: float, self_ref: float, recurrence: float) -> float:
    """Formule §13.2 — self_ref en boost-floor : un fait sur le user est
    toujours intéressant à consolider, même sans surprise."""
    weighted = 0.4 * surprise + 0.3 * self_ref + 0.2 * arousal + 0.1 * recurrence
    return max(weighted, self_ref)


def _clamp(value: float) -> float:
    return min(1.0, max(0.0, value))


class SalienceTagger:
    def __init__(self, manager: ModelManager, settings: Settings) -> None:
        self._manager = manager
        self._model = settings.SALIENCE_MODEL

    async def score(self, content: str, recent_history: list[str]) -> SalienceScores:
        prompt = SALIENCE_PROMPT.format(
            recent_history="\n".join(recent_history[-5:]) or EMPTY_HISTORY_PLACEHOLDER,
            content=content,
        )
        try:
            raw = await self._manager.generate(
                prompt,
                self._model,
                format="json",
                options={"temperature": 0.0, "num_predict": 256},
            )
            data = json.loads(raw)
            surprise = _clamp(float(data["surprise"]))
            arousal = _clamp(float(data["arousal"]))
            self_ref = _clamp(float(data["self_ref"]))
            recurrence = _clamp(float(data["recurrence"]))
        except Exception as exc:  # noqa: BLE001 — jamais bloquer le write path (§13.2)
            logger.error("salience_parse_failed", error=str(exc))
            return NEUTRAL_SCORES
        return SalienceScores(
            surprise=surprise,
            arousal=arousal,
            self_ref=self_ref,
            recurrence=recurrence,
            combined=combine(surprise, arousal, self_ref, recurrence),
        )


# ── Queue de scoring asynchrone (§13.3) ──────────────────────────────────────


@dataclass(frozen=True)
class ScoringJob:
    episode_id: str
    content: str
    recent_history: list[str] = field(default_factory=list)


class SalienceStoreProtocol(Protocol):
    """Contrat structurel minimal du store côté queue (pas d'import stores →
    pas de cycle tagger ↔ stores)."""

    async def update_salience(self, episode_id: str, scores: SalienceScores) -> None: ...


class ScoringQueue:
    """Queue bornée + workers en tâche de fond. Drop si pleine (§13.3)."""

    def __init__(
        self,
        tagger: SalienceTagger,
        store: SalienceStoreProtocol,
        maxsize: int = 100,
        workers: int = 1,
    ) -> None:
        self._tagger = tagger
        self._store = store
        self._queue: asyncio.Queue[ScoringJob] = asyncio.Queue(maxsize=maxsize)
        self._n_workers = workers
        self._tasks: list[asyncio.Task[None]] = []

    def enqueue(self, job: ScoringJob) -> bool:
        try:
            self._queue.put_nowait(job)
            return True
        except asyncio.QueueFull:
            # L'épisode garde salience=0.5 — jamais de backpressure sur le write.
            logger.warning("salience_queue_full_drop", episode_id=job.episode_id)
            return False

    @property
    def depth(self) -> int:
        return self._queue.qsize()

    async def start(self) -> None:
        self._tasks = [
            asyncio.create_task(self._worker(), name=f"salience-worker-{i}")
            for i in range(self._n_workers)
        ]

    async def stop(self, drain_timeout_s: float = 30.0) -> None:
        try:
            await asyncio.wait_for(self._queue.join(), timeout=drain_timeout_s)
        except TimeoutError:
            logger.warning("salience_queue_stop_timeout", pending=self._queue.qsize())
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks = []

    async def join(self) -> None:
        """Attend que tous les jobs enqueued soient traités (tests)."""
        await self._queue.join()

    async def _worker(self) -> None:
        while True:
            job = await self._queue.get()
            try:
                scores = await self._tagger.score(job.content, job.recent_history)
                await self._store.update_salience(job.episode_id, scores)
                logger.info(
                    "salience_scored",
                    episode_id=job.episode_id,
                    combined=scores["combined"],
                )
            except Exception as exc:  # noqa: BLE001 — un job raté ne tue pas le worker
                logger.error("salience_job_failed", episode_id=job.episode_id, error=str(exc))
            finally:
                self._queue.task_done()

"""Tests SalienceTagger (§19.1) — LLM mocké : formule combined, boost-floor
self_ref, fallback sur parse error, clamping, queue (drop si pleine)."""

from __future__ import annotations

import asyncio
import json

import pytest

from mnemos.config import Settings
from mnemos.tagger.salience import (
    SalienceScores,
    SalienceTagger,
    ScoringJob,
    ScoringQueue,
    combine,
)


class StubManager:
    def __init__(self, response: str) -> None:
        self.response = response
        self.prompts: list[str] = []

    async def generate(self, prompt: str, model: str, **kwargs: object) -> str:
        self.prompts.append(prompt)
        return self.response


def make_tagger(response: str) -> tuple[SalienceTagger, StubManager]:
    stub = StubManager(response)
    return SalienceTagger(stub, Settings(_env_file=None)), stub  # type: ignore[arg-type]


def test_combine_formule_ponderee() -> None:
    # 0.4*0.5 + 0.3*0.2 + 0.2*0.5 + 0.1*1.0 = 0.46 > self_ref=0.2
    assert combine(surprise=0.5, arousal=0.5, self_ref=0.2, recurrence=1.0) == pytest.approx(0.46)


def test_combine_boost_floor_self_ref() -> None:
    """§13.2 : self_ref seul suffit à passer le seuil — test obligatoire Phase 3."""
    # pondéré = 0.3*0.9 = 0.27, mais combined = max(0.27, 0.9) = 0.9
    assert combine(surprise=0.0, arousal=0.0, self_ref=0.9, recurrence=0.0) == 0.9


async def test_score_nominal() -> None:
    tagger, stub = make_tagger(
        json.dumps({"surprise": 0.8, "arousal": 0.4, "self_ref": 0.6, "recurrence": 0.1})
    )
    scores = await tagger.score("je déménage à Paris", ["salut", "ça va ?"])
    assert scores["surprise"] == 0.8
    # 0.4*0.8 + 0.3*0.6 + 0.2*0.4 + 0.1*0.1 = 0.59 < self_ref ? non : max(0.59, 0.6)=0.6
    assert scores["combined"] == pytest.approx(0.6)
    assert "je déménage à Paris" in stub.prompts[0]
    assert "salut" in stub.prompts[0]  # l'historique est dans le prompt


async def test_score_parse_error_fallback_neutre() -> None:
    """JSON invalide → combined=0.5, pas d'exception (§13.2)."""
    tagger, _ = make_tagger("pas du json {{{")
    scores = await tagger.score("contenu", [])
    assert scores["combined"] == 0.5


async def test_score_cle_manquante_fallback() -> None:
    tagger, _ = make_tagger(json.dumps({"surprise": 0.9}))
    scores = await tagger.score("contenu", [])
    assert scores["combined"] == 0.5


async def test_score_clamp_hors_bornes() -> None:
    tagger, _ = make_tagger(
        json.dumps({"surprise": 3.0, "arousal": -1.0, "self_ref": 0.5, "recurrence": 0.0})
    )
    scores = await tagger.score("contenu", [])
    assert scores["surprise"] == 1.0
    assert scores["arousal"] == 0.0


async def test_score_exception_manager_fallback() -> None:
    class BoomManager:
        async def generate(self, *a: object, **k: object) -> str:
            raise RuntimeError("ollama down")

    tagger = SalienceTagger(BoomManager(), Settings(_env_file=None))  # type: ignore[arg-type]
    scores = await tagger.score("contenu", [])
    assert scores["combined"] == 0.5


# ── ScoringQueue ──────────────────────────────────────────────────────────────


class RecordingStore:
    def __init__(self) -> None:
        self.updates: dict[str, SalienceScores] = {}

    async def update_salience(self, episode_id: str, scores: SalienceScores) -> None:
        self.updates[episode_id] = scores


async def test_queue_score_puis_update() -> None:
    tagger, _ = make_tagger(
        json.dumps({"surprise": 0.2, "arousal": 0.2, "self_ref": 0.9, "recurrence": 0.0})
    )
    store = RecordingStore()
    queue = ScoringQueue(tagger, store)
    await queue.start()
    assert queue.enqueue(ScoringJob("ep1", "je suis dev", []))
    await asyncio.wait_for(queue.join(), timeout=5)
    await queue.stop()
    assert store.updates["ep1"]["combined"] == 0.9  # boost-floor appliqué


async def test_queue_pleine_drop_sans_bloquer() -> None:
    """§13.3 : queue pleine → drop + log, jamais de backpressure."""
    tagger, _ = make_tagger("{}")
    queue = ScoringQueue(tagger, RecordingStore(), maxsize=2)
    # workers PAS démarrés : la queue se remplit
    assert queue.enqueue(ScoringJob("a", "x", []))
    assert queue.enqueue(ScoringJob("b", "x", []))
    assert not queue.enqueue(ScoringJob("c", "x", []))  # droppé
    assert queue.depth == 2


async def test_queue_job_rate_ne_tue_pas_le_worker() -> None:
    class FlakyStore:
        def __init__(self) -> None:
            self.updates: dict[str, SalienceScores] = {}
            self.calls = 0

        async def update_salience(self, episode_id: str, scores: SalienceScores) -> None:
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("db lock")
            self.updates[episode_id] = scores

    tagger, _ = make_tagger(
        json.dumps({"surprise": 0.5, "arousal": 0.5, "self_ref": 0.5, "recurrence": 0.5})
    )
    store = FlakyStore()
    queue = ScoringQueue(tagger, store)
    await queue.start()
    queue.enqueue(ScoringJob("ko", "x", []))
    queue.enqueue(ScoringJob("ok", "x", []))
    await asyncio.wait_for(queue.join(), timeout=5)
    await queue.stop()
    assert "ok" in store.updates  # le worker a survécu au job raté

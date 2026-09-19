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


async def test_score_sortie_llm_avec_think_et_markdown() -> None:
    """Sortie avec balises <think> et bloc ```json parsée avec succès."""
    raw = """<think>
Message à forte composante personnelle.
</think>
```json
{"surprise": 0.3, "arousal": 0.2, "self_ref": 0.9, "recurrence": 0.0}
```"""
    tagger, _ = make_tagger(raw)
    scores = await tagger.score("je suis marié depuis 10 ans", [])
    assert scores["self_ref"] == 0.9
    assert scores["combined"] == 0.9



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


async def test_queue_sans_worker_desactive_la_saillance() -> None:
    """SALIENCE_QUEUE_WORKERS=0 (mode épisodique) : rien n'est mis en file. Un job
    jamais consommé saturerait la file (un avertissement par message) et bloquerait
    join()."""
    tagger, _ = make_tagger("{}")
    queue = ScoringQueue(tagger, RecordingStore(), maxsize=2, workers=0)
    for i in range(5):
        assert not queue.enqueue(ScoringJob(f"ep{i}", "x", []))
    assert queue.depth == 0
    await asyncio.wait_for(queue.join(), timeout=1)  # ne bloque pas


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


async def test_queue_auto_drain_unscored_from_store() -> None:
    """Vérifie que ScoringQueue dépile les épisodes unscored de la DB quand la queue est idle."""
    from dataclasses import dataclass

    @dataclass
    class FakeEpisode:
        id: str
        content: str
        session_id: str | None = None
        tenant: str = "user"

    class DrainableStore:
        def __init__(self) -> None:
            self.unscored: list[FakeEpisode] = [
                FakeEpisode(id="ep_dropped_1", content="je vis à Annecy"),
                FakeEpisode(id="ep_dropped_2", content="mon chat s'appelle Miso"),
            ]
            self.updates: dict[str, SalienceScores] = {}

        async def list_unscored(self, limit: int = 5) -> list[FakeEpisode]:
            batch = self.unscored[:limit]
            self.unscored = self.unscored[limit:]
            return batch

        async def list_recent(
            self, session_id: str | None = None, n: int = 5, tenant: str = "user"
        ) -> list[FakeEpisode]:
            return []

        async def update_salience(self, episode_id: str, scores: SalienceScores) -> None:
            self.updates[episode_id] = scores

    tagger, _ = make_tagger(
        json.dumps({"surprise": 0.8, "arousal": 0.7, "self_ref": 0.9, "recurrence": 0.0})
    )
    store = DrainableStore()
    queue = ScoringQueue(tagger, store, maxsize=10, workers=1)
    await queue.start()

    # On n'enqueue rien en mémoire : le worker doit constater que la queue est vide
    # et auto-drainer les épisodes depuis la base.
    for _ in range(30):
        if len(store.updates) == 2:
            break
        await asyncio.sleep(0.1)

    await queue.stop()
    assert "ep_dropped_1" in store.updates
    assert "ep_dropped_2" in store.updates
    assert store.updates["ep_dropped_1"]["combined"] == 0.9


async def test_score_tenant_aware() -> None:
    """Le prompt de saillance doit injecter le sujet canonique du tenant."""
    tagger, stub = make_tagger(
        json.dumps({"surprise": 0.5, "arousal": 0.5, "self_ref": 0.7, "recurrence": 0.0})
    )
    scores = await tagger.score(
        "Sortie du moteur de simulation v2 en septembre",
        ["mise au point terminée"],
        tenant="atelios",
    )
    assert scores["combined"] == 0.7
    assert "Target subject of this memory stream: atelios" in stub.prompts[0]
    assert 'reveals about "atelios"' in stub.prompts[0]


"""Scénario 2 : Concurrence, Deadlock, Contention Sémaphores & Salience Burst.

Mesure :
1. Scalabilité en débit sous concurrence multi-workers (1, 2, 4, 8, 16 workers simultanés).
2. Vérification de non-deadlock et respect strict des tiers du ModelManager.
3. Stress-test de la file de saillance (ScoringQueue) : burst de 200 écritures subites,
   mesure des jobs en mémoire vs différés, et validation de l'auto-drain complet.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

from benchmarks.config import BenchConfig
from benchmarks.datasets import generate_synthetic_episodes
from benchmarks.harness import BenchmarkHarness, LatencyTracker
from mnemos.config import Settings
from mnemos.llm.model_manager import ModelManager, Tier
from mnemos.tagger.salience import SalienceTagger, ScoringQueue


@dataclass
class ConcurrencyWorkerResult:
    concurrency_level: int
    throughput_ops_sec: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    errors_count: int


@dataclass
class SalienceBurstResult:
    burst_size: int
    accepted_memory: int
    deferred_db: int
    drain_time_sec: float
    unscored_remaining: int


@dataclass
class ConcurrencyReport:
    workers_results: list[ConcurrencyWorkerResult] = field(default_factory=list)
    tier_deadlock_free: bool = False
    salience_burst: SalienceBurstResult | None = None


async def run_concurrency_benchmark(config: BenchConfig) -> ConcurrencyReport:
    harness = BenchmarkHarness(config)
    await harness.setup(reset=True)
    assert harness.episodic_store is not None

    report = ConcurrencyReport()

    # ── Test 1 : Courbe de débit multi-workers (1, 2, 4, 8, 16) ─────────────
    concurrency_levels = [1, 2, 4, 8, 16]
    ops_per_worker = 15 if config.mode == "real" else 40
    episodes = generate_synthetic_episodes(1000, tenant="user")

    for c_level in concurrency_levels:
        tracker = LatencyTracker(f"concurrency_{c_level}")
        errors = 0

        async def worker_task(worker_id: int, trk: LatencyTracker = tracker) -> None:
            nonlocal errors
            for i in range(ops_per_worker):
                op_idx = (worker_id * ops_per_worker + i) % len(episodes)
                ep = episodes[op_idx]
                t0 = time.monotonic()
                try:
                    # 70% écritures, 30% recherches
                    if i % 10 < 7:
                        await harness.episodic_store.write(
                            ep["content"],
                            role=ep["role"],
                            session_id=f"sess_{worker_id}",
                            tenant="user",
                        )
                    else:
                        await harness.episodic_store.search(
                            ep["content"][:30], k=5, tenant="user"
                        )
                    trk.record(time.monotonic() - t0)
                except Exception:
                    errors += 1

        t_start = time.monotonic()
        tasks = [asyncio.create_task(worker_task(w)) for w in range(c_level)]
        await asyncio.gather(*tasks)
        total_wall = time.monotonic() - t_start

        stats = tracker.compute(total_wall_time=total_wall)
        report.workers_results.append(
            ConcurrencyWorkerResult(
                concurrency_level=c_level,
                throughput_ops_sec=stats.throughput_ops_sec,
                p50_ms=stats.p50_ms,
                p95_ms=stats.p95_ms,
                p99_ms=stats.p99_ms,
                errors_count=errors,
            )
        )

    # ── Test 2 : Contention des Tiers ModelManager (Sémaphores) ──────────────
    class DummyClient:
        pass

    settings = Settings(
        LLM_TIER_SMALL_CONCURRENCY=2,
        LLM_TIER_MEDIUM_CONCURRENCY=1,
    )
    mm = ModelManager(settings, DummyClient())  # type: ignore[arg-type]

    active_executions: list[str] = []
    deadlock_detected = False

    async def run_small() -> None:
        async with mm.use(Tier.SMALL):
            active_executions.append("small")
            # Invariant : jamais de MEDIUM actif pendant SMALL
            if "medium" in active_executions:
                nonlocal deadlock_detected
                deadlock_detected = True
            await asyncio.sleep(0.01)
            active_executions.remove("small")

    async def run_medium() -> None:
        async with mm.use(Tier.MEDIUM):
            active_executions.append("medium")
            # Invariant : jamais de SMALL actif pendant MEDIUM
            if "small" in active_executions:
                nonlocal deadlock_detected
                deadlock_detected = True
            await asyncio.sleep(0.02)
            active_executions.remove("medium")

    try:
        contention_tasks = []
        for i in range(20):
            if i % 3 == 0:
                contention_tasks.append(run_medium())
            else:
                contention_tasks.append(run_small())
        await asyncio.wait_for(asyncio.gather(*contention_tasks), timeout=5.0)
        report.tier_deadlock_free = not deadlock_detected
    except TimeoutError:
        report.tier_deadlock_free = False

    # ── Test 3 : Salience Queue Burst & Auto-Drain ───────────────────────────
    from sqlalchemy import text

    from mnemos.tagger.salience import ScoringJob

    # Marque les écritures du Test 1 comme scorées pour tester le burst sans interférence de backlog
    async with harness.epi_engine.begin() as conn:
        await conn.execute(
            text("UPDATE episodes SET surprise = 0.2 WHERE surprise IS NULL")
        )

    burst_count = 150
    tagger = SalienceTagger(harness.llm_manager, harness.settings)  # type: ignore[arg-type]
    queue = ScoringQueue(tagger, harness.episodic_store, maxsize=50, workers=2)
    await queue.start()

    burst_episodes = [
        await harness.episodic_store.write(f"Burst message #{i} urgent et saillant", "user")
        for i in range(burst_count)
    ]

    accepted = 0
    deferred = 0
    for ep in burst_episodes:
        job = ScoringJob(episode_id=ep.id, content=ep.content, recent_history=[])
        enqueued = queue.enqueue(job)
        if enqueued:
            accepted += 1
        else:
            deferred += 1

    t0_drain = time.monotonic()
    for _ in range(200):
        unscored = await harness.episodic_store.list_unscored(limit=10)
        if len(unscored) == 0 and queue.depth == 0:
            break
        await asyncio.sleep(0.1)

    drain_duration = time.monotonic() - t0_drain
    await queue.stop()

    unscored_final = await harness.episodic_store.list_unscored(limit=100)
    report.salience_burst = SalienceBurstResult(
        burst_size=burst_count,
        accepted_memory=accepted,
        deferred_db=deferred,
        drain_time_sec=drain_duration,
        unscored_remaining=len(unscored_final),
    )

    await harness.teardown()
    return report

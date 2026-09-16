"""Scénario 1 : Stress-Test de Charge & Volume (Throughput, Scalabilité Vectorielle & KNN).

Mesure :
1. Débit et latence d'ingestion à volume croissant (500 -> 5 000+ épisodes).
2. Évolution de l'empreinte disque SQLite (base principale + journal WAL + vec0).
3. Latence de recherche KNN en fonction du volume total indexé.
4. Robustesse du partitionnement natif vec0 sous asymétrie (95% tenant A, 5% tenant B).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from benchmarks.config import BenchConfig
from benchmarks.datasets import generate_synthetic_episodes
from benchmarks.harness import BenchmarkHarness, LatencyStats, LatencyTracker


@dataclass
class VolumeStepResult:
    target_count: int
    write_stats: LatencyStats
    search_stats: LatencyStats
    db_size_kb: float
    wal_size_kb: float
    bytes_per_episode: float


@dataclass
class VolumeKnnReport:
    mode: str
    steps: list[VolumeStepResult] = field(default_factory=list)
    tenant_skew_recall: float = 0.0
    tenant_skew_search_ms: float = 0.0


async def run_volume_knn_benchmark(config: BenchConfig) -> VolumeKnnReport:
    harness = BenchmarkHarness(config)
    await harness.setup(reset=True)
    assert harness.episodic_store is not None

    report = VolumeKnnReport(mode=config.mode)
    episodes_pool = generate_synthetic_episodes(max(config.volume_steps) + 1000, tenant="user")

    current_count = 0
    pool_idx = 0

    try:
        for step_target in config.volume_steps:
            to_insert = step_target - current_count
            if to_insert <= 0:
                continue

            write_tracker = LatencyTracker("write_episode")
            t_start_step = time.monotonic()

            for _ in range(to_insert):
                ep = episodes_pool[pool_idx]
                pool_idx += 1
                t0 = time.monotonic()
                await harness.episodic_store.write(
                    content=ep["content"],
                    role=ep["role"],
                    session_id=ep["session_id"],
                    tenant=ep["tenant"],
                )
                write_tracker.record(time.monotonic() - t0)

            total_step_wall = time.monotonic() - t_start_step
            current_count = step_target
            write_stats = write_tracker.compute(total_wall_time=total_step_wall)

            search_tracker = LatencyTracker("knn_search")
            test_queries = [
                "base de données PostgreSQL",
                "chat Miso",
                "latence réseau HTTP/2",
                "risotto aux champignons",
                "linting Ruff et mypy",
                "contrat de maintenance annuel",
                "allergie arachides et noix",
                "reverse proxy Caddy",
            ]

            t_start_search = time.monotonic()
            num_searches = 20 if config.mode == "real" else 40
            for i in range(num_searches):
                q = test_queries[i % len(test_queries)]
                t0 = time.monotonic()
                results = await harness.episodic_store.search(q, k=10, tenant="user")
                search_tracker.record(time.monotonic() - t0)
                assert len(results) > 0

            search_wall = time.monotonic() - t_start_search
            search_stats = search_tracker.compute(total_wall_time=search_wall)

            db_stats = await harness.get_db_stats()
            bytes_per_ep = (
                (db_stats.episodic_bytes + db_stats.episodic_wal_bytes) / current_count
                if current_count > 0
                else 0.0
            )

            report.steps.append(
                VolumeStepResult(
                    target_count=current_count,
                    write_stats=write_stats,
                    search_stats=search_stats,
                    db_size_kb=db_stats.episodic_bytes / 1024.0,
                    wal_size_kb=db_stats.episodic_wal_bytes / 1024.0,
                    bytes_per_episode=bytes_per_ep,
                )
            )

        # Test d'isolation sous asymétrie extrême (95% tenant A / 5% tenant B)
        tenant_b_pool = [
            f"Projet confidentiel Alpha chez le client Acronis, étape {i}"
            for i in range(20)
        ]
        for c in tenant_b_pool:
            await harness.episodic_store.write(c, role="user", tenant="tenant_b")

        t0 = time.monotonic()
        b_results = await harness.episodic_store.search(
            "Projet confidentiel Alpha", k=10, tenant="tenant_b"
        )
        report.tenant_skew_search_ms = (time.monotonic() - t0) * 1000.0
        b_count = sum(1 for r in b_results if r.episode.tenant == "tenant_b")
        total_ret = min(len(b_results), 10)
        report.tenant_skew_recall = (b_count / total_ret) * 100.0 if total_ret > 0 else 0.0

    finally:
        await harness.teardown()

    return report

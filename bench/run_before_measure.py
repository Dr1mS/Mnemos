"""Mesure 'Avant' : Évalue Mnemos seul sur le commit précédant 4f9fc9e (5b428bc) sur TOUS les tests du jeu Dev."""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import asdict
from pathlib import Path

from bench.backends.mnemos_backend import MnemosBackend
from bench.config import BenchConfig
from bench.datasets import (
    DEV_BLAND_INSTANCES,
    DEV_CONVENTION_PROMPTS,
    DEV_EXTERNAL_INSTANCES,
    DEV_GHOST_INSTANCES,
)
from bench.scenarios.test1_ghost_vector import run_test1_ghost_vector
from bench.scenarios.test1b_external_ontology import run_test1b_external_ontology
from bench.scenarios.test2_bland_noise import run_test2_bland_noise
from bench.scenarios.test3_compliance import run_test3_compliance


async def main() -> None:
    bench_dir = Path("data/bench_before")
    results_dir = Path("bench/results")
    config = BenchConfig(
        bench_dir=bench_dir,
        dry_run=False,
        llm_model="qwen2.5:3b",
        embed_model="bge-m3:latest",
        temperature=0.0,
        seed=42,
        results_dir=results_dir,
    )

    print("=" * 70)
    print("Mesure 'Avant' : Mnemos seul (commit 5b428bc) sur TOUS les tests Dev")
    print(f"Modèle : {config.llm_model} | Embeddings : {config.embed_model}")
    print("=" * 70)

    backend = MnemosBackend(config)
    await backend.setup()

    # 1. Épreuve 1
    t0 = time.time()
    print("\n[1/4] Exécution Épreuve 1 : Mutation Temporelle (14 instances)...")
    r1 = await run_test1_ghost_vector(backend, config, DEV_GHOST_INSTANCES)
    print(f"  -> Sonde 1.1: {r1.passed_1_1_count}/14 ({r1.probe_1_1_score*100:.1f}%) | Sonde 1.2: {r1.passed_1_2_count}/14 ({r1.probe_1_2_score*100:.1f}%) [{time.time()-t0:.1f}s]")

    # 2. Épreuve 1b
    t0 = time.time()
    print("\n[2/4] Exécution Épreuve 1b : Mutation Hors Ontologie (14 instances)...")
    r1b = await run_test1b_external_ontology(backend, config, DEV_EXTERNAL_INSTANCES)
    print(f"  -> Sonde 1b.1: {r1b.passed_1b_1_count}/14 ({r1b.probe_1b_1_score*100:.1f}%) | Sonde 1b.2: {r1b.passed_1b_2_count}/14 ({r1b.probe_1b_2_score*100:.1f}%) [{time.time()-t0:.1f}s]")

    # 3. Épreuve 2
    t0 = time.time()
    print("\n[3/4] Exécution Épreuve 2 : Bruit Bland & Rétention 90j (14 instances)...")
    r2 = await run_test2_bland_noise(backend, config, DEV_BLAND_INSTANCES)
    print(f"  -> Rétention exacte: {r2.critical_facts_answer_success_rate*100:.1f}% ({r2.passed_2_1_count+r2.passed_2_2_count}/28) | Bruit: {r2.noise_ratio_in_recall*100:.1f}% [{time.time()-t0:.1f}s]")

    # 4. Épreuve 3
    t0 = time.time()
    print("\n[4/4] Exécution Épreuve 3 : Conventions de Projet Arbitraires (14 requêtes)...")
    r3 = await run_test3_compliance(backend, config, DEV_CONVENTION_PROMPTS)
    print(f"  -> Conventions respectées: {r3.passed_count}/14 ({r3.compliance_score*100:.1f}%) [{time.time()-t0:.1f}s]")

    await backend.teardown()

    summary = {
        "commit": "5b428bc",
        "t1_1": r1.probe_1_1_score,
        "t1_2": r1.probe_1_2_score,
        "t1b_1": r1b.probe_1b_1_score,
        "t1b_2": r1b.probe_1b_2_score,
        "t2": r2.critical_facts_answer_success_rate,
        "t3": r3.compliance_score,
        "t1": asdict(r1),
        "t1b": asdict(r1b),
        "t2_details": asdict(r2),
        "t3_details": asdict(r3),
    }

    out_file = results_dir / "mnemos_before_metrics.json"
    out_file.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nMétriques 'Avant' enregistrées dans : {out_file}")


if __name__ == "__main__":
    asyncio.run(main())

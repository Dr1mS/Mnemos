"""Point d'entrée principal pour lancer les benchmarks de Mnemos.

Exemples :
    # Tout lancer en mode rapide synthétique (stress volume 5 000+, concurrence 16 workers)
    uv run python -m benchmarks.run --scenario all --mode fast

    # Lancer le stress volume à échelle heavy (10 000 vecteurs)
    uv run python -m benchmarks.run --scenario volume --scale heavy

    # Lancer avec la vraie inférence locale Ollama (bge-m3 + qwen3:4b)
    uv run python -m benchmarks.run --scenario all --mode real --scale small
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from dataclasses import asdict
from pathlib import Path

from benchmarks.config import BenchConfig
from benchmarks.report import format_markdown_report, save_reports
from benchmarks.scenarios.bench_concurrency import run_concurrency_benchmark
from benchmarks.scenarios.bench_conflict import run_conflict_benchmark
from benchmarks.scenarios.bench_decay import run_decay_benchmark
from benchmarks.scenarios.bench_volume_knn import run_volume_knn_benchmark


async def main_async() -> int:
    parser = argparse.ArgumentParser(description="Suite de benchmarks et stress-test de Mnemos.")
    parser.add_argument(
        "-s",
        "--scenario",
        choices=["all", "volume", "concurrency", "conflict", "decay"],
        default="all",
        help="Scénario à exécuter (défaut: all)",
    )
    parser.add_argument(
        "-m",
        "--mode",
        choices=["fast", "real"],
        default="fast",
        help="Mode: fast (vecteurs synthétiques instantanés) ou real (Ollama local)",
    )
    parser.add_argument(
        "--scale",
        choices=["small", "medium", "heavy", "extreme"],
        default="medium",
        help="Échelle de volume: small, medium, heavy, extreme (défaut: medium)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("benchmarks/results"),
        help="Répertoire de sortie des rapports",
    )

    args = parser.parse_args()

    config = BenchConfig(mode=args.mode, scale=args.scale)
    config.apply_scale()
    import os
    os.environ["LOG_LEVEL"] = "WARNING"
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("=" * 70)
    print(">>> DEMARRAGE DU BENCHMARK MNEMOS")
    print(f"   Scenario : {args.scenario} | Mode : {args.mode} | Echelle : {args.scale}")
    print(f"   Repertoire isole de test : {config.bench_dir.resolve()}")
    print("=" * 70)

    volume_res = None
    concurrency_res = None
    conflict_res = None
    decay_res = None

    t0_global = time.perf_counter()

    # 1. Volume & KNN
    if args.scenario in ("all", "volume"):
        print("\n[*] [1/4] Execution du benchmark Volume & KNN Scalability...")
        t0 = time.perf_counter()
        volume_res = await run_volume_knn_benchmark(config)
        print(f"  [OK] Volume & KNN termine en {time.perf_counter() - t0:.2f} s")

    # 2. Concurrence & Deadlock
    if args.scenario in ("all", "concurrency"):
        print("\n[*] [2/4] Execution du benchmark Concurrence & Deadlock Stress...")
        t0 = time.perf_counter()
        concurrency_res = await run_concurrency_benchmark(config)
        print(f"  [OK] Concurrence terminee en {time.perf_counter() - t0:.2f} s")

    # 3. Conflit & Vérité Cognitive
    if args.scenario in ("all", "conflict"):
        print("\n[*] [3/4] Execution du benchmark Verite Cognitive & Resolution de Conflits...")
        t0 = time.perf_counter()
        conflict_res = await run_conflict_benchmark(config)
        print(f"  [OK] Conflits termines en {time.perf_counter() - t0:.2f} s")

    # 4. Simulation Decay 365 Jours
    if args.scenario in ("all", "decay"):
        print("\n[*] [4/4] Execution de la simulation Longue Duree (365j Decay Biologique)...")
        t0 = time.perf_counter()
        decay_res = await run_decay_benchmark(config)
        print(f"  [OK] Decay 365j termine en {time.perf_counter() - t0:.2f} s")

    total_duration = time.perf_counter() - t0_global

    # Génération et sauvegarde des rapports
    md_content = format_markdown_report(
        volume_res,
        concurrency_res,
        conflict_res,
        decay_res,
        total_duration,
    )

    raw_data = {
        "scenario": args.scenario,
        "mode": args.mode,
        "scale": args.scale,
        "total_duration_sec": total_duration,
        "volume": asdict(volume_res) if volume_res else None,
        "concurrency": asdict(concurrency_res) if concurrency_res else None,
        "conflict": asdict(conflict_res) if conflict_res else None,
        "decay": asdict(decay_res) if decay_res else None,
    }

    md_file, json_file = save_reports(args.output, md_content, raw_data)

    print("\n" + "=" * 70)
    print(f"[SUCCESS] BENCHMARKS TERMINES AVEC SUCCES en {total_duration:.2f} s")
    print(f"Rapport Markdown : {md_file.resolve()}")
    print(f"Donnees JSON     : {json_file.resolve()}")
    print("=" * 70)
    print("\n" + md_content)

    return 0


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(asyncio.run(main_async()))


if __name__ == "__main__":
    main()

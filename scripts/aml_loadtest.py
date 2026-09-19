"""Test de charge d'un déploiement Mnemos via le contrat AML (Add puis Search).

Mesure débit, latences et erreurs par niveau de concurrence, pour déclarer à la
plateforme une capacité réaliste. Rejoue des historiques PersonaMem locaux
(bench/data/personamem, cf. bench/bench_personamem.py) découpés comme la
plateforme (20 messages / 2 000 mots). Chaque requête est distincte pour ne
jamais profiter du cache d'embeddings. Les écritures vont dans des user_id
« loadtest:<run>:… » isolés.

Usage :
    .venv\\Scripts\\python.exe scripts\\aml_loadtest.py `
        --url https://mnemos.dr1ms.fr --key <API_KEY>
"""

from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import sys
import time
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx
from bench.bench_personamem import (
    DATA_DIR,
    chunk_messages,
    load_chat,
    load_personas,
    parse_query,
)

TOP_K = 100


def _percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(pct / 100 * (len(ordered) - 1)))] if ordered else 0.0


async def run_level(
    client: httpx.AsyncClient,
    path: str,
    payloads: list[dict[str, Any]],
    concurrency: int,
    headers: dict[str, str],
) -> dict[str, Any]:
    """Envoie les payloads avec au plus `concurrency` requêtes en vol."""
    sem = asyncio.Semaphore(concurrency)
    latencies: list[float] = []
    statuses: Counter[str] = Counter()

    async def one(payload: dict[str, Any]) -> None:
        async with sem:
            t0 = time.perf_counter()
            try:
                resp = await client.post(path, json=payload, headers=headers)
                statuses[str(resp.status_code)] += 1
            except httpx.HTTPError as exc:
                statuses[type(exc).__name__] += 1
            latencies.append((time.perf_counter() - t0) * 1000)

    t_start = time.perf_counter()
    await asyncio.gather(*(one(p) for p in payloads))
    wall_s = time.perf_counter() - t_start
    return {
        "concurrency": concurrency,
        "requests": len(payloads),
        "wall_s": wall_s,
        "req_s": len(payloads) / wall_s,
        "p50_ms": _percentile(latencies, 50),
        "p95_ms": _percentile(latencies, 95),
        "max_ms": max(latencies, default=0.0),
        "errors": sum(n for s, n in statuses.items() if s != "200"),
        "statuses": dict(statuses),
    }


async def main_async(args: argparse.Namespace) -> dict[str, Any]:
    run_id = uuid.uuid4().hex[:8]
    headers = {"Authorization": f"Bearer {args.key}"}
    personas = load_personas(DATA_DIR / "val.csv")
    chats = {
        pid: DATA_DIR / "chats" / Path(rows[0]["chat_history_32k_link"]).name
        for pid, rows in personas.items()
    }
    available = [pid for pid, path in chats.items() if path.exists()]

    # Add : chaque niveau reçoit des lots neufs (personas distinctes)
    add_levels = [int(c) for c in args.add_levels.split(",")]
    chunk_pool: list[dict[str, Any]] = []
    for pid in available:
        user_id = f"loadtest:{run_id}:{pid}"
        for i, chunk in enumerate(chunk_messages(load_chat(chats[pid]))):
            chunk_pool.append({
                "request_id": f"{user_id}:chunk-{i}",
                "user_id": user_id,
                "session_id": f"{user_id}:chat",
                "messages": chunk,
            })
    needed = args.adds_per_level * len(add_levels)
    if len(chunk_pool) < needed:
        raise SystemExit(f"{len(chunk_pool)} lots disponibles, {needed} requis")

    # Search : questions des personas ingérées, rendues uniques par niveau
    ingested = {p["user_id"] for p in chunk_pool[:needed]}
    questions = [
        (f"loadtest:{run_id}:{pid}", parse_query(row["user_query"]))
        for pid, rows in personas.items()
        if f"loadtest:{run_id}:{pid}" in ingested
        for row in rows
    ]
    search_levels = [int(c) for c in args.search_levels.split(",")]

    report: dict[str, Any] = {"url": args.url, "run_id": run_id, "add": [], "search": []}
    async with httpx.AsyncClient(base_url=args.url.rstrip("/"), timeout=args.timeout) as client:
        print(f"Cible : {args.url} — run {run_id}\n")
        print("ADD (lots de ≤ 20 messages)")
        for n, c in enumerate(add_levels):
            batch = chunk_pool[n * args.adds_per_level : (n + 1) * args.adds_per_level]
            res = await run_level(client, "/add", batch, c, headers)
            res["msg_s"] = sum(len(p["messages"]) for p in batch) / res["wall_s"]
            report["add"].append(res)
            print(
                f"  c={c:>2} : {res['msg_s']:6.1f} msg/s | p50 {res['p50_ms']:6.0f} ms"
                f" | p95 {res['p95_ms']:6.0f} ms | erreurs {res['errors']} {res['statuses']}"
            )

        print(f"\nSEARCH (top_k={TOP_K})")
        for c in search_levels:
            batch = [
                {"query": f"{q} ({c}/{i})", "user_id": uid, "top_k": TOP_K}
                for i, (uid, q) in enumerate(
                    itertools.islice(itertools.cycle(questions), args.searches_per_level)
                )
            ]
            res = await run_level(client, "/search", batch, c, headers)
            report["search"].append(res)
            print(
                f"  c={c:>2} : {res['req_s']:6.1f} req/s | p50 {res['p50_ms']:6.0f} ms"
                f" | p95 {res['p95_ms']:6.0f} ms | erreurs {res['errors']} {res['statuses']}"
            )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Test de charge AML d'un déploiement Mnemos")
    parser.add_argument("--url", required=True)
    parser.add_argument("--key", required=True, help="API_KEY du serveur")
    parser.add_argument("--add-levels", default="1,2,4,8")
    parser.add_argument("--adds-per-level", type=int, default=24)
    parser.add_argument("--search-levels", default="1,4,8,16")
    parser.add_argument("--searches-per-level", type=int, default=48)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--output", type=Path, default=None, help="Rapport JSON")
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # console Windows cp1252
    report = asyncio.run(main_async(args))
    if args.output:
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

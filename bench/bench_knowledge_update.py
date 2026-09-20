"""Mise à jour de connaissance : les faits consolidés servent-ils à répondre ?

C'est le scénario pour lequel la couche LLM de Mnemos a été conçue (§10.2, POC) :
une valeur change plusieurs fois, et la question porte sur la valeur **actuelle**.
Les benchs existants ne le testaient pas : test1_ghost_vector n'exécute jamais la
consolidation (aucun fait n'existe) et sa sonde « où j'habite actuellement ? » est
classée `working`, donc répondue par la mémoire de session — une fenêtre FIFO y
réussit aussi bien.

Ici, pour chaque instance (ville_0 → ville_1 → ville_2, séparées par des tours
neutres) : ingestion via POST /add, consolidation complète, puis deux contextes
issus du chemin noté POST /search, qui ne consulte jamais la mémoire de travail :
- A : les épisodes seuls (configuration de compétition) ;
- B : les épisodes plus les faits consolidés.
Un modèle local répond à partir de chaque contexte. La réponse est jugée par le
contrôleur strict du banc (ville active exigée, ville périmée présentée comme
actuelle = échec), pas par un LLM.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Comme en production AML : pas d'oubli temporel.
os.environ.setdefault("EPISODIC_RETENTION_DAYS", "36500")
os.environ.setdefault("DECAY_RATE_DAILY", "0.0")

import httpx

from bench.bench_locomo import drain_consolidation, setup_bench_app
from bench.datasets import DEV_GHOST_INSTANCES, GhostVectorInstance
from bench.eval_utils import check_active_residence_answer, normalize_text
from mnemos.config import Settings
from mnemos.llm.ollama_client import OllamaClient

PROBE = "Dans quelle ville est-ce que je vis ?"
ANSWER_PROMPT = """Using only the memories below, answer the question.
The memories are listed most relevant first, with their date.

Memories:
{context}

Question: {question}
Answer with the city name only."""

DAY_MS = 86_400_000


def build_messages(inst: GhostVectorInstance, base_ms: int) -> list[dict[str, Any]]:
    """Même déroulé que test1_ghost_vector : ville_0, tours neutres, ville_1,
    tours neutres, ville_2 ; horodatages croissants."""
    msgs: list[dict[str, Any]] = []
    t = base_ms

    def add(content: str, role: str, delta_days: float) -> None:
        nonlocal t
        t += int(delta_days * DAY_MS)
        msgs.append({"role": role, "content": content, "timestamp": t})

    add(f"J'habite à {inst.city_0}.", "user", 0)
    add(f"Bien noté, vous résidez à {inst.city_0}.", "assistant", 0.001)
    for user_q, asst_a in inst.neutral_turns_1:
        add(user_q, "user", 0.05)
        add(asst_a, "assistant", 0.001)
    add(f"J'ai déménagé, j'habite maintenant à {inst.city_1}.", "user", 1.0)
    add(f"Félicitations pour votre emménagement à {inst.city_1} !", "assistant", 0.001)
    for user_q, asst_a in inst.neutral_turns_2:
        add(user_q, "user", 0.05)
        add(asst_a, "assistant", 0.001)
    add(f"Finalement je vis à {inst.city_2}.", "user", 1.0)
    add(f"C'est noté, vous êtes maintenant installé à {inst.city_2}.", "assistant", 0.001)
    return msgs


def _iso_day(ts_ms: int) -> str:
    return datetime.fromtimestamp(ts_ms / 1000, tz=UTC).date().isoformat()


def _context(items: list[tuple[str, str]]) -> str:
    return "\n".join(f"{i}. [{date}] {content}" for i, (date, content) in enumerate(items, 1))


async def run_instance(
    inst: GhostVectorInstance,
    llm: OllamaClient,
    answer_model: str,
    top_k: int,
    user_id: str,
    llm_model: str,
) -> dict[str, Any]:
    tmp = Path(tempfile.mkdtemp(prefix="mnemos_ku_"))
    try:
        app, _, _ = await setup_bench_app(
            tmp, "ollama", llm_model=llm_model, salience_workers=2
        )
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport, base_url="http://t", timeout=600
            ) as client:
                messages = build_messages(inst, 1_690_000_000_000)
                for i in range(0, len(messages), 20):
                    resp = await client.post("/add", json={
                        "request_id": f"{inst.id}:chunk-{i // 20}",
                        "user_id": user_id,
                        "session_id": f"{inst.id}:session",
                        "messages": messages[i:i + 20],
                    })
                    resp.raise_for_status()
                await app.state.queue.join()
                cons = await drain_consolidation(app)

                episodes = await app.state.store.search(PROBE, k=top_k, tenant=user_id)
                # Même format de date dans les deux contextes : sinon la différence
                # de présentation pourrait expliquer un écart de réponse.
                ctx_a = [
                    (_iso_day(e.episode.created_at), e.episode.content) for e in episodes
                ]
                items = (await client.post("/search", json={
                    "query": PROBE, "user_id": user_id, "top_k": top_k,
                })).json()["data"]
                ctx_b = [(str(it.get("created_at", ""))[:10], it["content"]) for it in items]
                facts = [it["content"] for it in items if it["id"].startswith("fact_")]

        row: dict[str, Any] = {
            "instance": inst.id,
            "cities": [inst.city_0, inst.city_1, inst.city_2],
            "facts_inserted": cons["facts_inserted"],
            "facts_in_context": facts,
            "active_city_in_facts": any(normalize_text(inst.city_2) in normalize_text(f) for f in facts),
            "stale_city_in_facts": any(
                normalize_text(c) in normalize_text(f) for f in facts for c in (inst.city_0, inst.city_1)
            ),
        }
        for label, ctx in (("A", ctx_a), ("B", ctx_b)):
            raw = await llm.generate(
                ANSWER_PROMPT.format(context=_context(ctx), question=PROBE),
                answer_model,
                options={"temperature": 0.0, "num_ctx": 8192, "num_predict": 32},
            )
            answer = raw.strip()
            ok, reason = check_active_residence_answer(
                answer, inst.city_2, [inst.city_0, inst.city_1]
            )
            row[f"answer_{label}"] = answer
            row[f"correct_{label}"] = ok
            row[f"reason_{label}"] = reason
        return row
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


async def main_async(args: argparse.Namespace) -> dict[str, Any]:
    instances = DEV_GHOST_INSTANCES[: args.instances]
    llm = OllamaClient(Settings(_env_file=None))  # type: ignore[call-arg]
    records: list[dict[str, Any]] = []
    try:
        for n, inst in enumerate(instances, 1):
            row = await run_instance(
                inst, llm, args.answer_model, args.top_k, args.user_id, args.llm_model
            )
            records.append(row)
            print(
                f"  [{n}/{len(instances)}] {row['cities'][0]}→{row['cities'][1]}→{row['cities'][2]}"
                f" | A={'✓' if row['correct_A'] else '✗'} ({row['answer_A'][:22]})"
                f" | B={'✓' if row['correct_B'] else '✗'} ({row['answer_B'][:22]})"
                f" | faits: {len(row['facts_in_context'])}"
            )
    finally:
        await llm.aclose()

    n = len(records)
    a = sum(r["correct_A"] for r in records)
    b = sum(r["correct_B"] for r in records)
    result = {
        "config": {
            "instances": n, "top_k": args.top_k, "user_id": args.user_id,
            "llm_model": args.llm_model,
            "answer_model": args.answer_model, "probe": PROBE,
        },
        "A_episodes_only": a, "B_with_facts": b,
        "only_A_correct": sum(r["correct_A"] and not r["correct_B"] for r in records),
        "only_B_correct": sum(r["correct_B"] and not r["correct_A"] for r in records),
        "instances_with_active_fact": sum(r["active_city_in_facts"] for r in records),
        "instances_with_stale_fact": sum(r["stale_city_in_facts"] for r in records),
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nA (épisodes seuls) : {a}/{n}   |   B (avec faits) : {b}/{n}")
    print(f"seul A juste : {result['only_A_correct']} | seul B juste : {result['only_B_correct']}")
    print(f"fait portant la ville active présent : {result['instances_with_active_fact']}/{n}"
          f" | fait portant une ville périmée : {result['instances_with_stale_fact']}/{n}")
    print(f"Rapport : {args.output}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Mise à jour de connaissance : avec et sans faits")
    parser.add_argument("--instances", type=int, default=14)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--answer-model", default="qwen3.5:9b")
    parser.add_argument("--user-id", default="user", help="sert de sujet aux faits extraits")
    parser.add_argument("--llm-model", default="qwen2.5:3b", help="saillance + extraction")
    parser.add_argument("--output", type=Path, default=Path("bench/results/gpu/knowledge_update.json"))
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # console Windows cp1252
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()

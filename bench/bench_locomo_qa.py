"""Évaluation question → réponse → jugement sur LoCoMo, avec et sans faits consolidés.

Mesure ce que les benchs de récupération ne peuvent pas voir : les faits extraits
par le LLM aident-ils à *répondre* ? Même ingestion et même consolidation pour
les deux contextes, seule la présence des faits change :
- A (épisodique) : les top_k épisodes de la recherche hybride ;
- B (complet) : la réponse de POST /search, faits consolidés compris.
Un modèle local répond à partir de chaque contexte, puis juge chaque réponse
contre la référence LoCoMo. Les questions adversariales (catégorie 5, sans
réponse de référence) sont exclues. L'oubli temporel est neutralisé comme dans
le déploiement AML (les messages portent des dates de 2023).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import shutil
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Avant tout import de Settings : pas d'archivage des messages datés de 2023.
os.environ.setdefault("EPISODIC_RETENTION_DAYS", "36500")
os.environ.setdefault("DECAY_RATE_DAILY", "0.0")

import httpx

from bench.bench_locomo import drain_consolidation, parse_locomo_datetime, setup_bench_app
from mnemos.config import Settings
from mnemos.llm.json_cleaner import parse_llm_json
from mnemos.llm.ollama_client import OllamaClient

ANSWER_PROMPT = """You answer questions about a long conversation between two people,
using only the memories retrieved below (most relevant first, with their date).

Memories:
{context}

Question: {question}
Answer in a few words. If the memories do not contain the answer, say "unknown"."""

JUDGE_PROMPT = """Grade a candidate answer against the gold answer to a question.
The candidate is correct if it conveys the same information as the gold answer
(paraphrases, partial dates matching the gold date, and extra detail are fine).
It is wrong if it contradicts the gold answer, misses its key point, or says unknown.

Question: {question}
Gold answer: {gold}
Candidate answer: {candidate}

Reply with JSON only: {{"correct": true}} or {{"correct": false}}"""


def _format_context(items: list[tuple[str, str]]) -> str:
    return "\n".join(f"{i}. [{date}] {content}" for i, (date, content) in enumerate(items, 1))


def _iso_day(ts_ms: int) -> str:
    return datetime.fromtimestamp(ts_ms / 1000, tz=UTC).date().isoformat()


async def run_qa_bench(
    n_questions: int,
    seed: int,
    top_k: int,
    answer_model: str,
    output: Path,
    user_id: str = "locomo_qa",
) -> dict[str, Any]:
    data = json.loads(Path("bench/data/locomo_sample.json").read_text(encoding="utf-8"))
    conv = data["conversation"]
    pool = [q for q in data["qa"] if q.get("category") != 5 and q.get("answer") is not None]
    questions = random.Random(seed).sample(pool, min(n_questions, len(pool)))

    llm = OllamaClient(Settings(_env_file=None))  # type: ignore[call-arg]
    llm_opts = {"temperature": 0.0, "num_ctx": 8192, "num_predict": 64}

    async def ask(prompt: str, fmt: str | None = None) -> str:
        return await llm.generate(prompt, answer_model, format=fmt, options=llm_opts)  # type: ignore[arg-type]

    tmp = Path(tempfile.mkdtemp(prefix="mnemos_locomo_qa_"))
    try:
        app, _, _ = await setup_bench_app(tmp, "ollama", salience_workers=2)
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://t", timeout=600) as client:
                # 1. Ingestion (même format que bench_locomo) puis consolidation complète
                sessions = sorted(
                    (k for k in conv if k.startswith("session_") and not k.endswith("_date_time")),
                    key=lambda k: int(k.split("_")[1]),
                )
                for s_key in sessions:
                    base_ms = parse_locomo_datetime(conv.get(f"{s_key}_date_time"))
                    messages = [
                        {
                            "role": "user" if t.get("speaker") == conv.get("speaker_a") else "assistant",
                            "content": f"[{t.get('dia_id')}] {t.get('speaker')}: {t.get('text', '')}",
                            "timestamp": base_ms + i * 15000,
                        }
                        for i, t in enumerate(conv[s_key])
                    ]
                    resp = await client.post("/add", json={
                        "request_id": s_key, "user_id": user_id, "session_id": s_key, "messages": messages,
                    })
                    resp.raise_for_status()
                t0 = time.perf_counter()
                await app.state.queue.join()
                cons = await drain_consolidation(app)
                print(f"Consolidation : {cons['facts_inserted']} faits en {time.perf_counter() - t0:.0f} s")

                # 2. Deux contextes par question, réponse puis jugement
                records: list[dict[str, Any]] = []
                for n, q in enumerate(questions, 1):
                    episodes = await app.state.store.search(q["question"], k=top_k, tenant=user_id)
                    ctx_a = [(_iso_day(e.episode.created_at), e.episode.content) for e in episodes]
                    resp = await client.post("/search", json={
                        "query": q["question"], "user_id": user_id, "top_k": top_k,
                    })
                    items = resp.json()["data"]
                    ctx_b = [(str(it.get("created_at", ""))[:10], it["content"]) for it in items]
                    fact_ranks = [i for i, it in enumerate(items, 1) if it["id"].startswith("fact_")]

                    row: dict[str, Any] = {
                        "question": q["question"], "gold": str(q["answer"]), "category": q["category"],
                        "facts_in_context": len(fact_ranks),
                        "first_fact_rank": fact_ranks[0] if fact_ranks else None,
                    }
                    for label, ctx in (("A", ctx_a), ("B", ctx_b)):
                        answer = (await ask(ANSWER_PROMPT.format(
                            context=_format_context(ctx), question=q["question"],
                        ))).strip()
                        verdict = parse_llm_json(await ask(
                            JUDGE_PROMPT.format(question=q["question"], gold=q["answer"], candidate=answer),
                            fmt="json",
                        ))
                        row[f"answer_{label}"] = answer
                        row[f"correct_{label}"] = bool(isinstance(verdict, dict) and verdict.get("correct"))
                    records.append(row)
                    print(f"  [{n}/{len(questions)}] cat {q['category']} | A={'✓' if row['correct_A'] else '✗'} "
                          f"B={'✓' if row['correct_B'] else '✗'} | faits dans B : {row['facts_in_context']}")
    finally:
        await llm.aclose()
        shutil.rmtree(tmp, ignore_errors=True)

    n = len(records)
    acc = {lab: sum(r[f"correct_{lab}"] for r in records) / n for lab in ("A", "B")}
    only_a = sum(r["correct_A"] and not r["correct_B"] for r in records)
    only_b = sum(r["correct_B"] and not r["correct_A"] for r in records)
    by_cat: dict[int, dict[str, float]] = {}
    for c in sorted({r["category"] for r in records}):
        rows = [r for r in records if r["category"] == c]
        by_cat[c] = {
            "count": len(rows),
            "A": sum(r["correct_A"] for r in rows) / len(rows),
            "B": sum(r["correct_B"] for r in rows) / len(rows),
        }
    result = {
        "config": {"questions": n, "seed": seed, "top_k": top_k, "answer_and_judge_model": answer_model,
                   "user_id": user_id,
                   "extraction_model": "qwen2.5:3b", "facts_inserted": cons["facts_inserted"]},
        "accuracy": acc, "only_A_correct": only_a, "only_B_correct": only_b,
        "by_category": by_cat, "records": records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nExactitude A (épisodique) : {acc['A'] * 100:.1f} %  |  B (avec faits) : {acc['B'] * 100:.1f} %")
    print(f"Désaccords : seul A juste {only_a} | seul B juste {only_b}  (sur {n} questions)")
    for c, m in by_cat.items():
        print(f"  catégorie {c} (N={m['count']}) : A {m['A'] * 100:.0f} % | B {m['B'] * 100:.0f} %")
    print(f"Rapport : {output}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="LoCoMo : exactitude des réponses avec et sans faits")
    parser.add_argument("--questions", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--answer-model", default="qwen3.5:9b")
    parser.add_argument("--output", type=Path, default=Path("bench/results/gpu/locomo_qa.json"))
    # Le user_id sert de sujet des faits extraits (canonical_subject) : un identifiant
    # opaque comme ceux d'AML ("eval:run:…") pollue les faits et leur embedding.
    parser.add_argument("--user-id", default="locomo_qa")
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # console Windows cp1252
    asyncio.run(run_qa_bench(
        args.questions, args.seed, args.top_k, args.answer_model, args.output, args.user_id
    ))


if __name__ == "__main__":
    main()

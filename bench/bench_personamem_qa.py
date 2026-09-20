"""PersonaMem-v2 en choix multiples : la mémoire permet-elle de répondre juste ?

Format de la plateforme AML : la question arrive avec ses `options` au niveau
racine, et le système de mémoire ne fournit que le contexte. Ici, le contexte
vient de POST /search (top_k=100), un modèle local choisit une option, et la
notation est déterministe (bonne lettre ou non) — pas de juge LLM.

Couvre les catégories AML non mesurées jusqu'ici :
- D (gouvernance) via `ask_to_forget` : une préférence explicitement oubliée ne
  doit plus être utilisée ;
- E (personnalisation, santé) via les autres `pref_type`.

Deux configurations comparables sur les mêmes personas et les mêmes questions :
--llm-model vide (défaut) = épisodique seul, configuration de compétition ;
--llm-model qwen3.5:9b = saillance et extraction activées avec ce modèle.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Comme en production AML : pas d'oubli temporel.
os.environ.setdefault("EPISODIC_RETENTION_DAYS", "36500")
os.environ.setdefault("DECAY_RATE_DAILY", "0.0")

import httpx

from bench.bench_locomo import drain_consolidation, setup_bench_app
from bench.bench_personamem import DATA_DIR, chunk_messages, load_chat, load_personas, parse_query
from mnemos.config import Settings
from mnemos.llm.ollama_client import OllamaClient

LETTERS = "ABCD"
ANSWER_PROMPT = """You are answering on behalf of an assistant that remembers this user.
Use only the memories below (most relevant first, with their date).

Memories:
{context}

Question: {question}

Options:
{options}

Reply with the letter of the best option only."""


def build_options(row: dict[str, str], rng: random.Random) -> tuple[list[str], str]:
    """Options mélangées de façon déterministe ; renvoie (options, lettre correcte)."""
    correct = row["correct_answer"].strip()
    try:
        wrong = [str(a).strip() for a in json.loads(row["incorrect_answers"])]
    except json.JSONDecodeError:
        wrong = []
    options = [correct, *wrong[: len(LETTERS) - 1]]
    rng.shuffle(options)
    return options, LETTERS[options.index(correct)]


def parse_letter(raw: str) -> str | None:
    """Lettre isolée uniquement : « The answer is D. » → D, « aucune » → None
    (un simple balayage caractère par caractère renverrait le « a » de « answer »)."""
    text = raw.strip().upper()
    m = re.match(rf"^\W*([{LETTERS}])\b", text) or re.search(rf"\b([{LETTERS}])\b", text)
    return m.group(1) if m else None


async def run(args: argparse.Namespace) -> dict[str, Any]:
    persona_rows = load_personas(DATA_DIR / "val.csv")
    selected = [
        pid for pid in list(persona_rows)
        if (DATA_DIR / "chats" / Path(persona_rows[pid][0]["chat_history_32k_link"]).name).exists()
    ][: args.personas]
    rng = random.Random(args.seed)
    llm = OllamaClient(Settings(_env_file=None))  # type: ignore[call-arg]
    llm_opts = {"temperature": 0.0, "num_ctx": args.num_ctx, "num_predict": 8}
    # Ollama tronque le DÉBUT du prompt au-delà de num_ctx : ce sont les souvenirs
    # les mieux classés qui sauteraient. On mesure donc la taille de chaque prompt.
    budget_chars = args.num_ctx * 4
    consolidated = bool(args.llm_model)

    tmp = Path(tempfile.mkdtemp(prefix="mnemos_pm_qa_"))
    records: list[dict[str, Any]] = []
    cons: dict[str, int] = {}
    try:
        app, _, _ = await setup_bench_app(
            tmp, "ollama",
            llm_model=args.llm_model or "qwen2.5:3b",
            salience_workers=2 if consolidated else 0,
        )
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://t", timeout=900) as client:
                t0 = time.perf_counter()
                n_msg = 0
                for pid in selected:
                    chat = DATA_DIR / "chats" / Path(persona_rows[pid][0]["chat_history_32k_link"]).name
                    messages = load_chat(chat)
                    n_msg += len(messages)
                    for i, chunk in enumerate(chunk_messages(messages)):
                        resp = await client.post("/add", json={
                            "request_id": f"pm:{pid}:chunk-{i}", "user_id": f"pm:{pid}",
                            "session_id": f"pm:{pid}:chat", "messages": chunk,
                        })
                        resp.raise_for_status()
                print(f"Ingestion : {n_msg} messages en {time.perf_counter() - t0:.0f} s")
                if consolidated:
                    t1 = time.perf_counter()
                    await app.state.queue.join()
                    cons = await drain_consolidation(app)
                    print(f"Consolidation ({args.llm_model}) : {cons['facts_inserted']} faits "
                          f"en {time.perf_counter() - t1:.0f} s")

                for pid in selected:
                    for row in persona_rows[pid]:
                        if not row.get("correct_answer") or not row.get("incorrect_answers"):
                            continue
                        options, gold = build_options(row, rng)
                        if len(options) < 2:
                            continue
                        question = parse_query(row["user_query"])
                        items = (await client.post("/search", json={
                            "query": question, "user_id": f"pm:{pid}", "top_k": args.top_k,
                        })).json()["data"]
                        ctx = "\n".join(
                            f"{i}. [{str(it.get('created_at', ''))[:10]}] {it['content']}"
                            for i, it in enumerate(items, 1)
                        )
                        prompt = ANSWER_PROMPT.format(
                            context=ctx, question=question,
                            options="\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(options)),
                        )
                        raw = await llm.generate(prompt, args.answer_model, options=llm_opts)
                        letter = parse_letter(raw)
                        records.append({
                            "persona": pid, "pref_type": row.get("pref_type", ""),
                            "who": row.get("who", ""), "updated": row.get("updated", ""),
                            "gold": gold, "answer": letter, "correct": letter == gold,
                            "facts_in_context": sum(it["id"].startswith("fact_") for it in items),
                            "prompt_chars": len(prompt), "truncated": len(prompt) > budget_chars,
                        })
                        if len(records) % 10 == 0:
                            acc = sum(r["correct"] for r in records) / len(records)
                            print(f"  [{len(records)}] exactitude courante {acc * 100:.1f} %")
    finally:
        await llm.aclose()
        shutil.rmtree(tmp, ignore_errors=True)

    n = len(records)
    by_type: dict[str, dict[str, float]] = {}
    for t in sorted({r["pref_type"] for r in records}):
        rows = [r for r in records if r["pref_type"] == t]
        by_type[t] = {"count": len(rows), "accuracy": sum(r["correct"] for r in rows) / len(rows)}
    result = {
        "config": {
            "personas": len(selected), "questions": n, "top_k": args.top_k, "seed": args.seed,
            "mode": f"consolidation ({args.llm_model})" if consolidated else "épisodique seul",
            "answer_model": args.answer_model, "facts_inserted": cons.get("facts_inserted", 0),
        },
        "accuracy": sum(r["correct"] for r in records) / n if n else 0.0,
        "prompts_truncated": sum(r["truncated"] for r in records),
        "prompt_chars_max": max((r["prompt_chars"] for r in records), default=0),
        "facts_in_context_avg": sum(r["facts_in_context"] for r in records) / n if n else 0.0,
        "by_pref_type": by_type,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nMode : {result['config']['mode']} | {n} questions")
    print(f"Exactitude globale : {result['accuracy'] * 100:.1f} % (hasard ≈ 25 %)")
    for t, m in by_type.items():
        print(f"  {t:<32} N={int(m['count']):>3}  {m['accuracy'] * 100:>5.1f} %")
    print(f"Faits dans le contexte (moyenne) : {result['facts_in_context_avg']:.1f}")
    print(f"Prompts tronqués : {result['prompts_truncated']}/{n} "
          f"(le plus long : {result['prompt_chars_max']:,} car., budget {args.num_ctx * 4:,})")
    print(f"Rapport : {args.output}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="PersonaMem-v2 en choix multiples")
    parser.add_argument("--personas", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--answer-model", default="qwen3.5:9b")
    parser.add_argument("--num-ctx", type=int, default=24576, help="fenêtre du modèle de réponse")
    parser.add_argument("--llm-model", default="", help="vide = épisodique seul ; sinon saillance + extraction")
    parser.add_argument("--output", type=Path, default=Path("bench/results/gpu/personamem_qa.json"))
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # console Windows cp1252
    asyncio.run(run(args))


if __name__ == "__main__":
    main()

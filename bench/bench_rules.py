"""Catégorie G (règles et procédures) : une convention énoncée remonte-t-elle à temps ?

Des conventions de projet arbitraires sont énoncées au début d'un historique, puis
noyées sous des tours neutres. Pour chaque tâche de code, on mesure :
1. la **récupération** — notre responsabilité : la règle cible est-elle dans les
   résultats de POST /search, et à quel rang ?
2. la **conformité** — vérifiée de façon déterministe (jetons requis présents,
   jetons interdits absents), avec le contexte et **sans mémoire du tout**, pour
   isoler ce que la mémoire apporte réellement.

Les règles sont ingérées comme des messages `user` : le contrat AML n'accepte que
les rôles user et assistant.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ.setdefault("EPISODIC_RETENTION_DAYS", "36500")
os.environ.setdefault("DECAY_RATE_DAILY", "0.0")

import httpx

from bench.bench_locomo import setup_bench_app
from bench.datasets import (
    CONVENTION_BASELINE_RULES,
    DEV_CONVENTION_PROMPTS,
    DEV_GHOST_INSTANCES,
)
from bench.scenarios.test3_compliance import evaluate_convention_answer
from mnemos.config import Settings
from mnemos.llm.ollama_client import OllamaClient

DAY_MS = 86_400_000
PROMPT = """Tu es un ingénieur Python. Respecte scrupuleusement les conventions du
projet si elles figurent dans la mémoire.

Mémoire :
\"\"\"
{context}
\"\"\"

Requête : {task}
Code Python uniquement."""


def build_history(n_noise_blocks: int) -> list[dict[str, Any]]:
    """Les 5 conventions, puis des tours neutres qui les éloignent dans l'historique."""
    msgs: list[dict[str, Any]] = []
    t = 1_690_000_000_000

    def add(content: str, role: str) -> None:
        nonlocal t
        t += int(0.01 * DAY_MS)
        msgs.append({"role": role, "content": content, "timestamp": t})

    for item in CONVENTION_BASELINE_RULES:
        add(item["rule"], "user")
        add("Bien noté, je respecterai cette convention.", "assistant")
    for inst in DEV_GHOST_INSTANCES[:n_noise_blocks]:
        for user_q, asst_a in inst.neutral_turns_1 + inst.neutral_turns_2:
            add(user_q, "user")
            add(asst_a, "assistant")
    return msgs


async def main_async(args: argparse.Namespace) -> dict[str, Any]:
    messages = build_history(args.noise_blocks)
    llm = OllamaClient(Settings(_env_file=None))  # type: ignore[call-arg]
    gen_opts = {"temperature": 0.0, "num_ctx": 24576, "num_predict": 400}
    user_id = "rules"
    tmp = Path(tempfile.mkdtemp(prefix="mnemos_rules_"))
    records: list[dict[str, Any]] = []
    try:
        app, _, _ = await setup_bench_app(tmp, "ollama", salience_workers=0)
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://t", timeout=900) as c:
                for i in range(0, len(messages), 20):
                    resp = await c.post("/add", json={
                        "request_id": f"rules:chunk-{i // 20}", "user_id": user_id,
                        "session_id": "rules:session", "messages": messages[i:i + 20],
                    })
                    resp.raise_for_status()
                print(f"Historique : {len(messages)} messages dont {len(CONVENTION_BASELINE_RULES)} conventions")

                for item in DEV_CONVENTION_PROMPTS:
                    items = (await c.post("/search", json={
                        "query": item.prompt, "user_id": user_id, "top_k": args.top_k,
                    })).json()["data"]
                    contents = [str(it["content"]) for it in items]
                    rank = next((i for i, t in enumerate(contents, 1) if item.marker in t), None)
                    context = "\n".join(f"- {t}" for t in contents)

                    answers: dict[str, Any] = {}
                    for label, ctx in (("memoire", context), ("sans_memoire", "(aucun souvenir)")):
                        raw = await llm.generate(
                            PROMPT.format(context=ctx, task=item.prompt),
                            args.answer_model, options=gen_opts,
                        )
                        ok, reason = evaluate_convention_answer(raw, item)
                        answers[label] = {"ok": ok, "reason": reason}
                    records.append({
                        "id": item.id, "marker": item.marker, "rule_rank": rank,
                        "rules_in_context": sum(1 for t in contents if "[RULE-" in t),
                        "compliant_with_memory": answers["memoire"]["ok"],
                        "compliant_without_memory": answers["sans_memoire"]["ok"],
                        "reason_with_memory": answers["memoire"]["reason"],
                    })
                    print(f"  {item.id:<14} {item.marker:<16} rang de la règle : {str(rank or 'ABSENTE'):>7}"
                          f" | conforme avec mémoire : {'oui' if answers['memoire']['ok'] else 'non':<3}"
                          f" | sans mémoire : {'oui' if answers['sans_memoire']['ok'] else 'non'}")
    finally:
        await llm.aclose()
        shutil.rmtree(tmp, ignore_errors=True)

    n = len(records)
    ranks = [r["rule_rank"] for r in records if r["rule_rank"]]
    result = {
        "config": {
            "tasks": n, "top_k": args.top_k, "history_messages": len(messages),
            "answer_model": args.answer_model, "mode": "épisodique seul",
        },
        "rule_retrieved": len(ranks),
        "rule_in_top_10": sum(1 for r in ranks if r <= 10),
        "rule_rank_median": sorted(ranks)[len(ranks) // 2] if ranks else None,
        "compliant_with_memory": sum(r["compliant_with_memory"] for r in records),
        "compliant_without_memory": sum(r["compliant_without_memory"] for r in records),
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nRègle cible retrouvée : {result['rule_retrieved']}/{n} "
          f"(dans le top 10 : {result['rule_in_top_10']}/{n}, rang médian {result['rule_rank_median']})")
    print(f"Conformité avec mémoire : {result['compliant_with_memory']}/{n}"
          f"   |   sans mémoire : {result['compliant_without_memory']}/{n}")
    print(f"Rapport : {args.output}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Catégorie G : récupération et respect des règles")
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--noise-blocks", type=int, default=5, help="blocs de tours neutres")
    parser.add_argument("--answer-model", default="qwen3.5:9b")
    parser.add_argument("--output", type=Path, default=Path("bench/results/gpu/rules_g.json"))
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # console Windows cp1252
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()

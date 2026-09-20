"""Catégorie H : abstention et divulgation minimale.

Deux mesures, toutes deux déterministes :

1. **Abstention.** On pose à chaque persona des questions dont la réponse n'est pas
   dans son historique (elles viennent d'une autre persona : bien formées, plausibles,
   mais sans réponse ici). La bonne conduite est de répondre UNKNOWN. On compare
   **avec** le contexte de /search et **sans mémoire du tout** : si l'abstention
   chute avec la mémoire, c'est que nos résultats poussent à inventer. Des questions
   répondables servent de contrôle, pour ne pas confondre prudence et mutisme.

2. **Divulgation minimale.** Les lignes `sensitive_info` de PersonaMem désignent des
   passages sensibles. On mesure combien de fois ils remontent dans le top 10 alors
   que la question posée ne les concerne pas.
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
from bench.bench_personamem import (
    DATA_DIR,
    _normalize,
    chunk_messages,
    load_chat,
    load_personas,
    parse_evidence,
    parse_query,
)
from mnemos.config import Settings
from mnemos.llm.ollama_client import OllamaClient

PROMPT = """Answer the question using only the memories below.
If the memories do not contain the answer, reply exactly: UNKNOWN

Memories:
{context}

Question: {question}
Answer briefly."""

NO_MEMORY = "(no memories available)"


ABSTENTION_MARKERS = (
    "unknown", "i don't know", "i do not know", "not mentioned", "no information",
    "cannot determine", "can't determine", "not available", "je ne sais pas",
    "aucune information", "pas d'information",
)


def abstained(answer: str) -> bool:
    """Le prompt demande « UNKNOWN », mais on accepte les formulations équivalentes :
    les compter comme des réponses inventées surestimerait les hallucinations."""
    head = answer.strip().lower()[:80]
    return any(m in head for m in ABSTENTION_MARKERS)


async def main_async(args: argparse.Namespace) -> dict[str, Any]:
    rows = load_personas(DATA_DIR / "val.csv")
    avail = [
        p for p in rows
        if (DATA_DIR / "chats" / Path(rows[p][0]["chat_history_32k_link"]).name).exists()
    ][: args.personas]
    llm = OllamaClient(Settings(_env_file=None))  # type: ignore[call-arg]
    opts = {"temperature": 0.0, "num_ctx": 24576, "num_predict": 48}
    records: list[dict[str, Any]] = []
    leaks = {"queries": 0, "sensitive_in_top10": 0}

    tmp = Path(tempfile.mkdtemp(prefix="mnemos_abst_"))
    try:
        app, _, _ = await setup_bench_app(tmp, "ollama", salience_workers=0)
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://t", timeout=900) as c:
                chats = {}
                for pid in avail:
                    msgs = load_chat(DATA_DIR / "chats" / Path(rows[pid][0]["chat_history_32k_link"]).name)
                    chats[pid] = {_normalize(m["content"]) for m in msgs}
                    for i, chunk in enumerate(chunk_messages(msgs)):
                        resp = await c.post("/add", json={
                            "request_id": f"abst:{pid}:{i}", "user_id": f"abst:{pid}",
                            "session_id": f"abst:{pid}", "messages": chunk,
                        })
                        resp.raise_for_status()
                print(f"Ingestion : {len(avail)} personas")

                for n, pid in enumerate(avail):
                    other = avail[(n + 1) % len(avail)]
                    # Passages sensibles de CETTE persona (pour la divulgation minimale)
                    sensitive = {
                        s for r in rows[pid] if r.get("pref_type") == "sensitive_info"
                        for s in parse_evidence(r.get("related_conversation_snippet", ""))
                    }
                    cases = [("répondable", r) for r in rows[pid][: args.per_persona]]
                    # Questions d'une autre persona : sans réponse ici (preuve absente)
                    cases += [
                        ("sans_réponse", r) for r in rows[other]
                        if not (parse_evidence(r.get("related_conversation_snippet", "")) & chats[pid])
                    ][: args.per_persona]

                    for kind, row in cases:
                        question = parse_query(row["user_query"])
                        items = (await c.post("/search", json={
                            "query": question, "user_id": f"abst:{pid}", "top_k": args.top_k,
                        })).json()["data"]
                        ctx = "\n".join(f"{i}. {it['content']}" for i, it in enumerate(items, 1))
                        if sensitive and kind == "sans_réponse":
                            leaks["queries"] += 1
                            top10 = {_normalize(str(it["content"])) for it in items[:10]}
                            leaks["sensitive_in_top10"] += bool(top10 & sensitive)

                        answers = {}
                        for label, context in (("mémoire", ctx), ("sans_mémoire", NO_MEMORY)):
                            raw = await llm.generate(
                                PROMPT.format(context=context, question=question),
                                args.answer_model, options=opts,
                            )
                            answers[label] = raw.strip()
                        records.append({
                            "persona": pid, "kind": kind, "question": question[:90],
                            "abstained_with_memory": abstained(answers["mémoire"]),
                            "abstained_without_memory": abstained(answers["sans_mémoire"]),
                            "answer_with_memory": answers["mémoire"][:120],
                        })
                    print(f"  [{n + 1}/{len(avail)}] persona {pid} : {len(cases)} questions")
    finally:
        await llm.aclose()
        shutil.rmtree(tmp, ignore_errors=True)

    def rate(kind: str, key: str) -> tuple[int, int]:
        rows_ = [r for r in records if r["kind"] == kind]
        return sum(r[key] for r in rows_), len(rows_)

    na_mem, na_tot = rate("sans_réponse", "abstained_with_memory")
    na_nomem, _ = rate("sans_réponse", "abstained_without_memory")
    ok_mem, ok_tot = rate("répondable", "abstained_with_memory")
    ok_nomem, _ = rate("répondable", "abstained_without_memory")
    result = {
        "config": {"personas": len(avail), "top_k": args.top_k, "answer_model": args.answer_model,
                   "questions": len(records)},
        "unanswerable": {"total": na_tot, "abstained_with_memory": na_mem,
                         "abstained_without_memory": na_nomem},
        "answerable": {"total": ok_tot, "abstained_with_memory": ok_mem,
                       "abstained_without_memory": ok_nomem},
        "minimal_disclosure": leaks,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nQuestions SANS réponse en mémoire ({na_tot}) — bonne conduite : s'abstenir")
    print(f"   abstention avec mémoire  : {na_mem}/{na_tot}")
    print(f"   abstention sans mémoire  : {na_nomem}/{na_tot}  (référence du modèle seul)")
    print(f"Questions répondables ({ok_tot}) — bonne conduite : répondre")
    print(f"   abstention à tort avec mémoire : {ok_mem}/{ok_tot} | sans mémoire : {ok_nomem}/{ok_tot}")
    if leaks["queries"]:
        print(f"Divulgation : passage sensible dans le top 10 sur {leaks['sensitive_in_top10']}/"
              f"{leaks['queries']} questions sans rapport")
    print(f"Rapport : {args.output}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Catégorie H : abstention et divulgation minimale")
    parser.add_argument("--personas", type=int, default=10)
    parser.add_argument("--per-persona", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--answer-model", default="qwen3.5:9b")
    parser.add_argument("--output", type=Path, default=Path("bench/results/gpu/abstention_h.json"))
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # console Windows cp1252
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()

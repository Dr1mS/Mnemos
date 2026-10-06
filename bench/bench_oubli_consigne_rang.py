"""Demandes d'oubli : la consigne de l'utilisateur est-elle dans le top 100 ?

Carte des faiblesses du 06/10/2026 (`bench_carte_personamem.py`) : sur les questions
`ask_to_forget` de PersonaMem-v2, la réponse avec le contexte Mnemos (21 %) est sous le
plafond obtenu avec tout l'historique (34 %), sur 47 questions. La préférence oubliée,
elle, est toujours retrouvée. Hypothèse à vérifier avant tout correctif : le répondeur
voit la préférence mais pas la CONSIGNE qui l'annule, parce que celle-ci n'entre pas
dans les 100 souvenirs rendus par `/search`.

Sans modèle de langage : mêmes personas que la carte (graine 7), ingestion par la route
AML, puis rang de la consigne d'oubli (message utilisateur détecté par
`router/forget.py` et portant sur la préférence oubliée) dans la réponse de `/search`.
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
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ.setdefault("EPISODIC_RETENTION_DAYS", "36500")
os.environ.setdefault("DECAY_RATE_DAILY", "0.0")

import httpx

from bench.bench_forget_targeting import _mots, porte
from bench.bench_locomo import setup_bench_app
from bench.bench_personamem import DATA_DIR, chunk_messages, load_chat, load_personas, parse_query
from bench.bench_temporal_recall import garder_la_boucle
from mnemos.router.forget import detect_forget


async def run(args: argparse.Namespace) -> None:
    garder_la_boucle(asyncio.get_running_loop())
    lignes = load_personas(DATA_DIR / "val.csv")
    dispo = sorted(p for p, rs in lignes.items()
                   if (DATA_DIR / "chats" / Path(rs[0]["chat_history_32k_link"]).name).exists())
    personas = random.Random(args.graine).sample(dispo, min(args.personas, len(dispo)))
    resultats: list[dict[str, Any]] = []
    tmp = Path(tempfile.mkdtemp(prefix="mnemos_oubli_rang_"))
    try:
        app, _, _ = await setup_bench_app(tmp, "ollama", salience_workers=0)
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://t", timeout=600) as client:
                for pid in personas:
                    rows = [r for r in lignes[pid] if r.get("pref_type") == "ask_to_forget"
                            and r.get("correct_answer") and r.get("incorrect_answers")]
                    if not rows:
                        continue
                    tenant = f"pm:{pid}"
                    chat = load_chat(DATA_DIR / "chats" / Path(rows[0]["chat_history_32k_link"]).name)
                    for c, chunk in enumerate(chunk_messages(chat)):
                        (await client.post("/add", json={"request_id": f"{tenant}:{c}", "user_id": tenant,
                                                         "session_id": tenant, "messages": chunk})
                         ).raise_for_status()
                    for row in rows:
                        mots = _mots(row.get("prev_pref", ""))
                        consignes = [m["content"] for m in chat if m["role"] == "user"
                                     and detect_forget(m["content"], "user") is not None
                                     and porte(m["content"], mots)]
                        items = (await client.post("/search", json={
                            "query": parse_query(row["user_query"]), "user_id": tenant,
                            "top_k": 100})).json()["data"]
                        contenus = [it["content"] for it in items]
                        rangs = [next((i for i, c in enumerate(contenus, 1) if c.startswith(k[:200])), None)
                                 for k in consignes]
                        trouves = [r for r in rangs if r is not None]
                        resultats.append({"persona": pid, "question": parse_query(row["user_query"])[:120],
                                          "consignes": len(consignes),
                                          "rang_consigne": min(trouves) if trouves else None})
                    print(f"  persona {pid} : {len(resultats)} questions", flush=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    avec = [r for r in resultats if r["consignes"]]
    dedans = [r for r in avec if r["rang_consigne"] is not None]
    rangs: list[int] = sorted(int(r["rang_consigne"]) for r in dedans)
    res: dict[str, Any] = {"questions": len(resultats), "consigne_detectee": len(avec), "consigne_au_top100": len(dedans),
           "rang_median": rangs[len(rangs) // 2] if rangs else None,
           "consigne_au_top10": sum(r <= 10 for r in rangs), "details": resultats}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nquestions ask_to_forget : {res['questions']} | consigne retrouvée dans l'historique :"
          f" {res['consigne_detectee']} | dans le top 100 : {res['consigne_au_top100']}"
          f" | top 10 : {res['consigne_au_top10']} | rang médian : {res['rang_median']}")
    print(f"Rapport : {args.output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Demandes d'oubli : rang de la consigne")
    parser.add_argument("--personas", type=int, default=70)
    parser.add_argument("--graine", type=int, default=7)
    parser.add_argument("--output", type=Path, default=Path("bench/results/oubli/consigne_rang.json"))
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(run(args))


if __name__ == "__main__":
    main()

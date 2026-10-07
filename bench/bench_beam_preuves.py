"""BEAM (100K) : `/search` rend-il les messages qui contiennent la réponse ?

Carte des faiblesses, dernier jeu public. BEAM 100K : 20 conversations d'environ 63 000 mots
(~190 messages chacune), 400 questions sur 10 types. Chaque question (sauf l'abstention)
désigne ses messages-preuves (`source_chat_ids`) : on mesure sans modèle de langage si le
top 100 de `/search` les contient tous (preuve@100), et à quel rang.

Ingestion : une conversation = un tenant, les messages dans l'ordre, un lot `/add` par
session de la conversation. Le découpage et les horodatages de la plateforme ne sont pas
publiés ; les `time_anchor` du jeu sont rares (quelques messages) et ne sont pas envoyés.
Données : github.com/mohammadtavakoli78/BEAM, `chats/100K` (CC BY-SA 4.0), hors de git.
Embeddings par llama-server, la pile de production.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ.setdefault("EPISODIC_RETENTION_DAYS", "36500")
os.environ.setdefault("DECAY_RATE_DAILY", "0.0")

import httpx

from bench.bench_locomo import setup_bench_app
from bench.bench_temporal_recall import garder_la_boucle

RACINE = Path("bench/data/beam/100K")
TOP_K = 100
LOT_MESSAGES = 20


def charger(n: int) -> tuple[list[list[dict[str, Any]]], dict[str, list[dict[str, Any]]]]:
    """Les sessions de la conversation `n` (messages dans l'ordre) et ses questions par type."""
    chat = json.loads((RACINE / str(n) / "chat.json").read_text(encoding="utf-8"))
    sessions = [[m for tour in lot["turns"] for m in tour] for lot in chat]
    questions = json.loads((RACINE / str(n) / "probing_questions.json").read_text(encoding="utf-8"))
    return sessions, questions


def ids_preuves(valeur: Any) -> list[int]:
    """`source_chat_ids` aplati : liste d'entiers, listes imbriquées (ordre des événements) ou
    dictionnaire (énoncé d'origine / mis à jour, premier / second événement…)."""
    if isinstance(valeur, dict):
        return [i for v in valeur.values() for i in ids_preuves(v)]
    if isinstance(valeur, list):
        return [i for v in valeur for i in ids_preuves(v)]
    if isinstance(valeur, int) or (isinstance(valeur, str) and valeur.isdigit()):
        return [int(valeur)]
    return []


def present(contenu: str, rendus: list[str]) -> int | None:
    """Rang (1-based) du message dans la réponse de `/search`, ou None."""
    tete = contenu.strip()[:200]
    return next((i for i, c in enumerate(rendus, 1) if c.strip().startswith(tete)), None)


async def run(args: argparse.Namespace) -> None:
    garder_la_boucle(asyncio.get_running_loop())
    details: list[dict[str, Any]] = []
    tmp = Path(tempfile.mkdtemp(prefix="mnemos_beam_"))
    try:
        app, _, _ = await setup_bench_app(tmp, "ollama", salience_workers=0, embed_backend="llamacpp")
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://t", timeout=900) as client:
                for n in range(1, args.conversations + 1):
                    sessions, questions = charger(n)
                    tenant = f"beam:{n}"
                    par_id: dict[int, str] = {}
                    for s, msgs in enumerate(sessions):
                        for m in msgs:
                            par_id[int(m["id"])] = str(m["content"])
                        for i in range(0, len(msgs), LOT_MESSAGES):
                            lot = [{"role": m["role"], "content": str(m["content"])} for m in msgs[i:i + LOT_MESSAGES]
                                   if str(m["content"]).strip()]
                            (await client.post("/add", json={
                                "request_id": f"{tenant}:{s}:{i}", "user_id": tenant,
                                "session_id": f"{tenant}:{s}", "messages": lot})).raise_for_status()
                    for type_q, qs in questions.items():
                        for k, q in enumerate(qs):
                            rendus = [str(it["content"]) for it in (await client.post("/search", json={
                                "query": str(q["question"]), "user_id": tenant, "top_k": TOP_K})).json()["data"]]
                            preuves = [x for x in dict.fromkeys(ids_preuves(q.get("source_chat_ids"))) if x in par_id]
                            rangs = [present(par_id[p], rendus) for p in preuves]
                            details.append({"conv": n, "type": type_q, "k": k, "preuves": len(preuves),
                                            "rangs": rangs, "messages": len(par_id)})
                    print(f"  conversation {n} : {len(par_id)} messages, {len(details)} questions", flush=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    rapport: dict[str, Any] = {"conversations": args.conversations, "par_type": {}}
    par_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for d in details:
        if d["preuves"]:
            par_type[d["type"]].append(d)
    print(f"\n{'type':28s} {'n':>4s} {'toutes@100':>11s} {'preuves@100':>12s} {'rang médian':>12s}")
    tous: list[dict[str, Any]] = []
    for t in sorted(par_type):
        xs = par_type[t]
        tous += xs
        rangs = [r for x in xs for r in x["rangs"]]
        res: dict[str, Any] = {"n": len(xs), "toutes_au_top100": sum(all(r is not None for r in x["rangs"]) for x in xs),
               "preuves": len(rangs), "preuves_au_top100": sum(r is not None for r in rangs),
               "rang_median": median([r for r in rangs if r is not None]) if any(r is not None for r in rangs) else None}
        rapport["par_type"][t] = res
        print(f"{t:28s} {res['n']:>4d} {res['toutes_au_top100']:>5d} ({100 * res['toutes_au_top100'] / res['n']:3.0f} %)"
              f" {res['preuves_au_top100']:>5d}/{res['preuves']:<5d} {res['rang_median']!s:>12s}")
    rangs = [r for x in tous for r in x["rangs"]]
    rapport["ensemble"] = {"n": len(tous), "toutes_au_top100": sum(all(r is not None for r in x["rangs"]) for x in tous),
                           "preuves": len(rangs), "preuves_au_top100": sum(r is not None for r in rangs)}
    print(f"ensemble : {rapport['ensemble']}")
    rapport["messages_par_conversation"] = median(d["messages"] for d in details)
    rapport["details"] = details
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rapport, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Rapport : {args.output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="BEAM 100K : preuves dans le top 100 de /search")
    parser.add_argument("--conversations", type=int, default=20)
    parser.add_argument("--output", type=Path, default=Path("bench/results/carte/beam_preuves.json"))
    args = parser.parse_args()
    if not (RACINE / "1" / "chat.json").exists():
        raise SystemExit(f"{RACINE} absent : télécharger chats/100K depuis github.com/mohammadtavakoli78/BEAM")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(run(args))


if __name__ == "__main__":
    main()

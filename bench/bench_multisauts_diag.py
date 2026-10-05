"""Multi-sauts : pourquoi la recherche rate-t-elle une preuve sur deux ?

Mesuré le 05/10/2026 (`bench_temporal_recall.py`) : seules 50,7 % des questions
multi-sauts de LoCoMo (catégorie 1) ont TOUTES leurs preuves dans le top 100 de
`/search`. Ce diagnostic, sans aucun modèle de langage, dit pour chaque preuve
ratée ce qui la sépare du top 100 — chaque réponse désigne un levier différent :

- **rang complet** (classement sur toute la conversation) : juste derrière le top
  100, c'est un problème de classement ; très loin, un problème de sens ;
- **hors du vivier dense** : la recherche ne reclasse que les 2 × k plus proches
  voisins par embedding ; une preuve hors de ce vivier ne peut pas être rattrapée
  par le score lexical ;
- **mots communs** avec la question, hors mots-outils ;
- **voisine d'un souvenir retrouvé** : réplique adjacente (±1, ±2) d'un épisode du
  top 10, ou session déjà représentée dans le top 100 ;
- **locuteur** : la personne nommée dans la question est-elle celle qui parle ?

LoCoMo est public. Contenu ingéré réaliste : « Locuteur: texte ».
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ.setdefault("EPISODIC_RETENTION_DAYS", "36500")
os.environ.setdefault("DECAY_RATE_DAILY", "0.0")

import httpx

from bench.bench_locomo import parse_locomo_datetime, setup_bench_app
from bench.bench_temporal_recall import PAS_MS, _ecrire, _iso, _preuves_locomo, garder_la_boucle
from mnemos.stores import episodic as _episodic

LOCOMO = Path("bench/data/locomo10.json")
MOTS_OUTILS = frozenset({
    "a", "an", "the", "and", "or", "but", "if", "of", "to", "in", "on", "at", "by", "for",
    "with", "from", "as", "is", "are", "was", "were", "be", "been", "being", "do", "does",
    "did", "done", "have", "has", "had", "having", "what", "which", "who", "whom", "whose",
    "when", "where", "why", "how", "that", "this", "these", "those", "it", "its", "he",
    "she", "they", "them", "his", "her", "their", "i", "you", "we", "me", "my", "your",
    "our", "us", "about", "into", "over", "after", "before", "during", "since", "until",
    "any", "some", "all", "each", "every", "more", "most", "other", "such", "only", "own",
    "same", "so", "than", "too", "very", "can", "could", "will", "would", "should", "may",
    "might", "must", "shall", "not", "no", "nor", "just", "also", "there", "here", "then",
    "ever",
})


def mots_pleins(texte: str) -> set[str]:
    return {m for m in re.findall(r"[a-z0-9']+", texte.lower())
            if m not in MOTS_OUTILS and len(m) > 2}


async def diagnostiquer(args: argparse.Namespace) -> dict[str, Any]:
    garder_la_boucle(asyncio.get_running_loop())
    data = json.loads(LOCOMO.read_text(encoding="utf-8"))
    tmp = Path(tempfile.mkdtemp(prefix="mnemos_multisauts_"))
    lignes: list[dict[str, Any]] = []
    try:
        app, _, _ = await setup_bench_app(tmp, "ollama", salience_workers=0)
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://t", timeout=600) as client:
                for c in data:
                    conv, sid = c["conversation"], c["sample_id"]
                    user_id = f"locomo:{sid}"
                    tours: dict[str, dict[str, Any]] = {}
                    cle_vers_dia: dict[str, str] = {}
                    sessions = sorted((k for k in conv if k.startswith("session_")
                                       and not k.endswith("_date_time")),
                                      key=lambda k: int(k.split("_")[1]))
                    for s_key in sessions:
                        base = parse_locomo_datetime(conv.get(f"{s_key}_date_time"))
                        msgs = []
                        for i, t in enumerate(conv[s_key]):
                            ts = base + i * PAS_MS
                            cle_vers_dia[_iso(ts)] = t["dia_id"]
                            tours[t["dia_id"]] = {"session": s_key, "pos": i, "locuteur": t.get("speaker"),
                                                  "texte": t.get("text", "")}
                            msgs.append({"role": "user" if t.get("speaker") == conv["speaker_a"] else "assistant",
                                         "content": f"{t.get('speaker')}: {t.get('text', '')}",
                                         "timestamp": ts})
                        await _ecrire(client, user_id, s_key, msgs)
                    noms = {conv["speaker_a"], conv["speaker_b"]}
                    for q in c["qa"]:
                        if q.get("category") != args.categorie:
                            continue
                        preuves = [p for p in _preuves_locomo(q) if p in tours]
                        if not preuves:
                            continue
                        items = (await client.post("/search", json={
                            "query": q["question"], "user_id": user_id, "top_k": 100})).json()["data"]
                        top = [cle_vers_dia.get(str(it.get("created_at", ""))) for it in items]
                        # Classement complet : toute la conversation, vivier dense compris.
                        complet = await app.state.store.search(q["question"], k=args.k_complet, tenant=user_id)
                        rang_complet = {cle_vers_dia.get(_iso(e.episode.created_at)): r
                                        for r, e in enumerate(complet, 1)}
                        top10 = {d for d in top[:10] if d}
                        sessions_top = {tours[d]["session"] for d in top if d}
                        nommes = {n for n in noms if n.lower() in q["question"].lower()}
                        mq = mots_pleins(q["question"]) - {n.lower() for n in noms}
                        for p in preuves:
                            t = tours[p]
                            voisins = {f"D{p[1:].split(':')[0]}:{int(p.split(':')[1]) + d}" for d in (-2, -1, 1, 2)}
                            lignes.append({
                                "conv": sid, "question": q["question"], "preuve": p, "n_preuves": len(preuves),
                                "dans_top100": p in top, "rang_top100": top.index(p) + 1 if p in top else None,
                                "rang_complet": rang_complet.get(p),
                                "mots_communs": sorted(mq & mots_pleins(t["texte"])),
                                "voisine_top10": bool(voisins & top10),
                                "session_dans_top100": t["session"] in sessions_top,
                                "locuteur_nomme": (t["locuteur"] in nommes) if nommes else None,
                                "texte": f"{t['locuteur']}: {t['texte']}"[:160],
                            })
                    print(f"  {sid} : {sum(1 for x in lignes if x['conv'] == sid)} preuves", flush=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return {"config": {"categorie": args.categorie, "k_complet": args.k_complet,
                       "knn_candidats_top100": max(200, _episodic.KNN_CANDIDATES)},
            "preuves": lignes}


def resumer(r: dict[str, Any]) -> None:
    lignes = r["preuves"]
    ratees = [x for x in lignes if not x["dans_top100"]]
    vivier = r["config"]["knn_candidats_top100"]
    print(f"\nPreuves : {len(lignes)} | ratées (hors top 100) : {len(ratees)} ({len(ratees) / len(lignes):.0%})")
    tranches: Counter[str] = Counter()
    for x in ratees:
        rg = x["rang_complet"]
        tranches["absente du classement complet" if rg is None else
                 "101-150" if rg <= 150 else "151-200" if rg <= 200 else "201-400" if rg <= 400 else "> 400"] += 1
    print("  rang complet des ratées :", dict(tranches))
    print(f"  ratées sans aucun mot plein commun avec la question : "
          f"{sum(not x['mots_communs'] for x in ratees)}/{len(ratees)}"
          f" (retrouvées : {sum(not x['mots_communs'] for x in lignes if x['dans_top100'])}"
          f"/{len(lignes) - len(ratees)})")
    print(f"  ratées voisines (±2 répliques) d'un épisode du top 10 : {sum(x['voisine_top10'] for x in ratees)}")
    print(f"  ratées dont la session est déjà dans le top 100 : {sum(x['session_dans_top100'] for x in ratees)}")
    loc = [x for x in ratees if x["locuteur_nomme"] is not None]
    print(f"  ratées où la question nomme un locuteur : {len(loc)}, dont prononcées par lui : "
          f"{sum(x['locuteur_nomme'] for x in loc)}")
    print(f"  (vivier dense de /search à top_k=100 : {vivier} candidats)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-sauts : diagnostic des preuves ratées")
    parser.add_argument("--categorie", type=int, default=1, help="1 = multi-sauts dans LoCoMo")
    parser.add_argument("--k-complet", type=int, default=1500, help="classement complet (toute la conversation)")
    parser.add_argument("--output", type=Path, default=Path("bench/results/multisauts/multisauts_diag.json"))
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    r = asyncio.run(diagnostiquer(args))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(r, indent=2, ensure_ascii=False), encoding="utf-8")
    resumer(r)
    print(f"Rapport : {args.output}")


if __name__ == "__main__":
    main()

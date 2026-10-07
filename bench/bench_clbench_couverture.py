"""CL-bench (catégorie G) : Mnemos perd-il de l'information avant le répondeur ?

Carte des faiblesses, dernier jeu public non mesuré. Une tâche CL-bench = un message
système, un message utilisateur qui porte un document de référence (médiane 1 300 mots,
jusqu'à 49 000), puis parfois des échanges (33 % des tâches ont plusieurs tours, 55 % en
« Empirical Discovery », notre G4 à 18,43). La consigne officielle (`clbench/pipeline.py`)
donne le message système au répondeur à part et dit que « la question inclut un document de
référence ». La façon dont la plateforme découpe une tâche entre `/add` et la question n'est
pas publiée : on mesure sous trois hypothèses, sans modèle de langage.

  * `tache`   — un tenant par tâche, chaque message antérieur ingéré tel quel, question =
                dernier message utilisateur ;
  * `contexte` — un tenant par document (`context_id`) : les tâches sœurs (jusqu'à 12)
                 s'accumulent dans la même mémoire ;
  * `passages` — un tenant par tâche, le document découpé en passages d'environ 300 mots.

Mesure : part des messages (ou passages) antérieurs présents dans le top 100 de `/search`.
Plus un contrôle de robustesse : le plus gros document au `/add`, une requête de la taille
d'une question à un tour au `/search` (statut, latence, troncature des embeddings).
Embeddings par llama-server, la pile de production.
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

DONNEES = Path("bench/data/clbench/CL-bench.jsonl")
TOP_K = 100
MOTS_PASSAGE = 300
LOT_MESSAGES = 20


def passages(texte: str, n: int = MOTS_PASSAGE) -> list[str]:
    mots = texte.split()
    return [" ".join(mots[i:i + n]) for i in range(0, len(mots), n)] or [texte]


def present(cible: str, contenus: list[str]) -> bool:
    tete = cible.strip()[:200]
    return any(c.strip().startswith(tete) for c in contenus)


async def ajouter(client: httpx.AsyncClient, tenant: str, messages: list[dict[str, str]], prefixe: str) -> float:
    debut = time.perf_counter()
    for i in range(0, len(messages), LOT_MESSAGES):
        r = await client.post("/add", json={"request_id": f"{prefixe}:{i}", "user_id": tenant,
                                            "session_id": tenant, "messages": messages[i:i + LOT_MESSAGES]})
        r.raise_for_status()
    return time.perf_counter() - debut


async def chercher(client: httpx.AsyncClient, tenant: str, requete: str) -> tuple[list[str], float, int]:
    debut = time.perf_counter()
    r = await client.post("/search", json={"query": requete, "user_id": tenant, "top_k": TOP_K})
    duree = time.perf_counter() - debut
    contenus = [str(it["content"]) for it in r.json().get("data", [])] if r.status_code == 200 else []
    return contenus, duree, r.status_code


def anterieurs(tache: dict[str, Any]) -> list[dict[str, str]]:
    """Messages avant la dernière question, sans le message système (donné à part au répondeur)."""
    return [{"role": m["role"], "content": str(m["content"])} for m in tache["messages"][1:-1]]


async def run(args: argparse.Namespace) -> None:
    garder_la_boucle(asyncio.get_running_loop())
    # Découpe sur les seuls sauts de ligne : les documents contiennent des U+2028, que
    # splitlines() couperait au milieu d'une chaîne JSON.
    taches = [json.loads(x) for x in DONNEES.read_text(encoding="utf-8").split("\n") if x.strip()]
    par_contexte: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for t in taches:
        par_contexte[t["metadata"]["context_id"]].append(t)
    # Contextes ayant au moins une tâche à plusieurs tours, `--par-categorie` par catégorie
    rng = random.Random(args.graine)
    choisis: list[str] = []
    for cat in sorted({t["metadata"]["context_category"] for t in taches}):
        ids = sorted(c for c, ts in par_contexte.items()
                     if ts[0]["metadata"]["context_category"] == cat and any(len(t["messages"]) > 2 for t in ts))
        choisis += rng.sample(ids, min(args.par_categorie, len(ids)))
    rapport: dict[str, Any] = {"contextes": len(choisis), "robustesse": {}, "modes": {}}
    tmp = Path(tempfile.mkdtemp(prefix="mnemos_clbench_"))
    try:
        app, _, _ = await setup_bench_app(tmp, "ollama", salience_workers=0, embed_backend="llamacpp")
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://t", timeout=900) as client:
                # 1. Robustesse : le plus gros document, et une requête géante
                geant = max(taches, key=lambda t: len(str(t["messages"][1]["content"]).split()))
                doc = str(geant["messages"][1]["content"])
                d_add = await ajouter(client, "rob", [{"role": "user", "content": doc}], "rob")
                _, d_norm, s_norm = await chercher(client, "rob", "What are the rules for scoring?")
                q_geante = " ".join(doc.split()[:30000])
                contenus, d_geant, s_geant = await chercher(client, "rob", q_geante)
                rapport["robustesse"] = {
                    "mots_document": len(doc.split()), "add_s": round(d_add, 2),
                    "search_normale": {"statut": s_norm, "s": round(d_norm, 2)},
                    "search_requete_30k_mots": {"statut": s_geant, "s": round(d_geant, 2),
                                                "document_rendu": present(doc, contenus)}}
                print(f"robustesse : {rapport['robustesse']}", flush=True)

                # 2. Couverture sous les trois hypothèses d'ingestion
                for mode in ("tache", "contexte", "passages"):
                    couv: list[float] = []
                    par_cat: dict[str, list[float]] = defaultdict(list)
                    for cid in choisis:
                        soeurs = par_contexte[cid]
                        cat = soeurs[0]["metadata"]["context_category"]
                        if mode == "contexte":
                            tenant = f"ctx:{cid}"
                            for j, t in enumerate(soeurs):
                                await ajouter(client, tenant, anterieurs(t), f"{tenant}:{j}")
                        for j, t in enumerate(soeurs):
                            if len(t["messages"]) <= 2:
                                continue  # un seul tour : rien d'antérieur à retrouver
                            msgs = anterieurs(t)
                            if mode == "tache":
                                tenant = f"t:{cid}:{j}"
                                await ajouter(client, tenant, msgs, tenant)
                            elif mode == "passages":
                                tenant = f"p:{cid}:{j}"
                                msgs = [{"role": m["role"], "content": p} for m in msgs for p in passages(m["content"])]
                                await ajouter(client, tenant, msgs, tenant)
                            contenus, _, statut = await chercher(client, tenant, str(t["messages"][-1]["content"]))
                            if statut != 200:
                                print(f"  {mode} {cid}#{j} : /search {statut}", flush=True)
                            c = sum(present(m["content"], contenus) for m in msgs) / len(msgs)
                            couv.append(c)
                            par_cat[cat].append(c)
                    rapport["modes"][mode] = {
                        "taches": len(couv), "couverture_moyenne": round(sum(couv) / len(couv), 4),
                        "couverture_complete": sum(c == 1.0 for c in couv),
                        "par_categorie": {k: {"n": len(v), "moyenne": round(sum(v) / len(v), 4),
                                              "minimum": round(min(v), 4), "mediane": median(v)}
                                          for k, v in par_cat.items()}}
                    print(f"{mode} : {rapport['modes'][mode]}", flush=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rapport, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Rapport : {args.output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="CL-bench : couverture de l'historique par /search")
    parser.add_argument("--par-categorie", type=int, default=10)
    parser.add_argument("--graine", type=int, default=7)
    parser.add_argument("--output", type=Path, default=Path("bench/results/carte/clbench_couverture.json"))
    args = parser.parse_args()
    if not DONNEES.exists():
        raise SystemExit(f"{DONNEES} absent : télécharger CL-bench.jsonl depuis huggingface.co/datasets/tencent/CL-bench")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(run(args))


if __name__ == "__main__":
    main()

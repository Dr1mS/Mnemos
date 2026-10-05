"""Carte des faiblesses, PersonaMem-v2 : où perd-on, et est-ce à nous ?

Trois nombres par question (méthode fixée le 05/10/2026, AML_COMPETITION_PREP.md §29) :
- **preuve@100** : l'extrait de conversation qui porte la préférence est-il dans les
  100 souvenirs de `/search` ? Recherche seule, sans modèle de langage ;
- **A** : la consigne QCM officielle (`data/personamem/pipeline_v2.py`, dépôt public
  des organisateurs) nourrie du contexte Mnemos, lettre extraite par leur fonction ;
- **O** : la même consigne nourrie de tout l'historique 32k — le plafond du répondeur.

Un écart O − A avec une preuve absente est à nous ; un O bas ne l'est pas.

Deux pistes gardées en réserve sont mesurées au passage, sans rien déployer :
- **options** (piste 1) : la preuve entre-t-elle dans le top 100 si l'on fusionne la
  recherche de la question avec une recherche par option ? (le texte rendu resterait
  celui des souvenirs ; la décision de principe revient à l'utilisateur) ;
- **messages longs** (piste 2) : longueur en mots de l'extrait-preuve, pour voir si
  les preuves manquées sont surtout longues.

PersonaMem-v2 est public. La consigne et l'extracteur viennent du dépôt des
organisateurs, lus depuis une copie locale non suivie par git (pas de licence).
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
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ.setdefault("EPISODIC_RETENTION_DAYS", "36500")
os.environ.setdefault("DECAY_RATE_DAILY", "0.0")

import httpx

from bench import aml_officiel
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
from bench.bench_temporal_recall import garder_la_boucle
from bench.remote_llm import RemoteChat
from mnemos.config import Settings
from mnemos.llm.ollama_client import OllamaClient

TOP_K = 100


def fusion(listes: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """Tourniquet entre la recherche de la question et celles des options, sans
    doublon, coupé à TOP_K : ce que renverrait une recherche guidée par les options."""
    vus: set[str] = set()
    out: list[dict[str, Any]] = []
    for rang in range(max((len(x) for x in listes), default=0)):
        for x in listes:
            if rang < len(x) and x[rang]["id"] not in vus:
                vus.add(x[rang]["id"])
                out.append(x[rang])
    return out[:TOP_K]


def present(items: list[dict[str, Any]], preuves: set[str]) -> bool:
    return any(_normalize(it["content"]) in preuves for it in items)


def consigne(module: Any, contexte: str, requete: str, options: list[str]) -> str:
    """La consigne QCM officielle, aplatie en un seul message : nos clients
    (RemoteChat, Ollama generate) n'envoient qu'un message utilisateur."""
    lettres = "\n".join(f"{chr(65 + i)}. {o}" for i, o in enumerate(options))
    return (f"{contexte}\n\nUser: {requete}{module.RECALL_SUFFIX}\n\n"
            + str(module.MCQ_PROMPT_TEMPLATE).format(options=lettres))


async def run(args: argparse.Namespace) -> None:
    garder_la_boucle(asyncio.get_running_loop())
    officiel = aml_officiel.charger("personamem", "pipeline_v2.py")
    lignes = load_personas(DATA_DIR / "val.csv")
    dispo = sorted(p for p, rs in lignes.items()
                   if (DATA_DIR / "chats" / Path(rs[0]["chat_history_32k_link"]).name).exists())
    personas = random.Random(args.graine).sample(dispo, min(args.personas, len(dispo)))
    repondeur: Any = (OllamaClient(Settings(_env_file=None))  # type: ignore[call-arg]
                      if args.answer_backend == "ollama" else RemoteChat(args.answer_backend))
    opts = {"temperature": 0.0, "num_ctx": 40960, "num_predict": 768}
    faits: dict[str, dict[str, Any]] = {}
    if args.cache.exists():
        for ligne in args.cache.read_text(encoding="utf-8").splitlines():
            if ligne.strip():
                x = json.loads(ligne)
                faits[x["cle"]] = x
    if faits:
        print(f"Reprise : {len(faits)} questions déjà dans {args.cache}", flush=True)
    tmp = Path(tempfile.mkdtemp(prefix="mnemos_carte_pm_"))
    try:
        app, _, _ = await setup_bench_app(tmp, "ollama", salience_workers=0)
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://t", timeout=600) as client:

                async def chercher(q: str, tenant: str) -> list[dict[str, Any]]:
                    r = await client.post("/search", json={"query": q, "user_id": tenant, "top_k": TOP_K})
                    return list(r.json()["data"])

                for n, pid in enumerate(personas, 1):
                    rows = [r for r in lignes[pid] if r.get("correct_answer") and r.get("incorrect_answers")]
                    a_faire = [(k, r) for k, r in enumerate(rows) if f"{pid}#{k}" not in faits]
                    if not a_faire:
                        continue
                    tenant = f"pm:{pid}"
                    chat = load_chat(DATA_DIR / "chats" / Path(rows[0]["chat_history_32k_link"]).name)
                    for c, chunk in enumerate(chunk_messages(chat)):
                        (await client.post("/add", json={"request_id": f"{tenant}:{c}", "user_id": tenant,
                                                         "session_id": tenant, "messages": chunk})
                         ).raise_for_status()
                    historique = "\n".join(f"{m['role']}: {m['content']}" for m in chat)
                    for k, row in a_faire:
                        requete = parse_query(row["user_query"])
                        preuves = parse_evidence(row.get("related_conversation_snippet", ""))
                        incorrectes = json.loads(row["incorrect_answers"])
                        options = [str(row["correct_answer"]), *(str(x) for x in incorrectes)]
                        random.Random(f"{pid}:{k}:{args.graine}").shuffle(options)
                        bonne = chr(65 + options.index(str(row["correct_answer"])))
                        items = await chercher(requete, tenant)
                        par_option = [await chercher(o, tenant) for o in options]
                        memoires = "Memories from the conversation history:\n" + "\n".join(
                            f"[{it.get('created_at', '')}] {it['content']}" for it in items)
                        reponses = {}
                        try:
                            for nom, contexte in (("A", memoires), ("O", "Conversation history:\n" + historique)):
                                sortie = await repondeur.generate(
                                    consigne(officiel, contexte, requete, options), args.answer_model, options=opts)
                                reponses[nom] = officiel.extract_final_letter(sortie)
                        except Exception as exc:  # noqa: BLE001 — Nvidia (429/503) ou Ollama (500)
                            print(f"  sautée, à refaire : {pid}#{k} ({exc})", flush=True)
                            continue
                        try:
                            extrait = json.loads(row.get("related_conversation_snippet") or "[]")
                        except json.JSONDecodeError:
                            extrait = []
                        longueurs = [len(str(m.get("content", "")).split())
                                     for m in extrait if isinstance(m, dict)]
                        x = {"cle": f"{pid}#{k}", "persona": pid, "type": row.get("pref_type", ""),
                             "preuve_100": present(items, preuves) if preuves else None,
                             "preuve_100_options": (present(fusion([items, *par_option]), preuves)
                                                    if preuves else None),
                             "mots_preuve_max": max(longueurs, default=0),
                             "A": reponses["A"] == bonne, "O": reponses["O"] == bonne,
                             "lettre_A": reponses["A"], "lettre_O": reponses["O"], "bonne": bonne}
                        faits[x["cle"]] = x
                        args.cache.parent.mkdir(parents=True, exist_ok=True)
                        with args.cache.open("a", encoding="utf-8") as f:
                            f.write(json.dumps(x, ensure_ascii=False) + "\n")
                    print(f"  [{n}/{len(personas)}] persona {pid} : {len(faits)} questions au total", flush=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        await repondeur.aclose()

    par_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for x in faits.values():
        par_type[x["type"]].append(x)
        par_type["TOUS"].append(x)

    def taux(xs: list[dict[str, Any]], cle: str) -> float | None:
        vals = [x[cle] for x in xs if x[cle] is not None]
        return round(sum(vals) / len(vals), 3) if vals else None

    resume = {t: {"n": len(xs), "preuve@100": taux(xs, "preuve_100"),
                  "preuve@100 avec options": taux(xs, "preuve_100_options"),
                  "A": taux(xs, "A"), "O": taux(xs, "O")}
              for t, xs in sorted(par_type.items())}
    manquees = [x for x in faits.values() if x["preuve_100"] is False]
    trouvees = [x for x in faits.values() if x["preuve_100"] is True]
    res = {"config": {"personas": len(personas), "graine": args.graine, "answer_model": args.answer_model,
                      "answer_backend": args.answer_backend,
                      "aml_pipeline_commit": ((aml_officiel.RACINE / "COMMIT").read_text().strip()
                                              if (aml_officiel.RACINE / "COMMIT").exists() else None)},
           "par_type": resume,
           "mots_preuve_mediane": {
               "manquees": sorted(x["mots_preuve_max"] for x in manquees)[len(manquees) // 2] if manquees else None,
               "trouvees": sorted(x["mots_preuve_max"] for x in trouvees)[len(trouvees) // 2] if trouvees else None},
           "questions": list(faits.values())}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n{'type':32s} {'n':>4s} {'preuve@100':>10s} {'+options':>8s} {'A':>6s} {'O':>6s}")
    for t, m in resume.items():
        print(f"{t:32s} {m['n']:4d} {m['preuve@100']!s:>10s} {m['preuve@100 avec options']!s:>8s}"
              f" {m['A']!s:>6s} {m['O']!s:>6s}")
    print(f"longueur médiane de l'extrait-preuve (mots) : {res['mots_preuve_mediane']}")
    print(f"Rapport : {args.output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Carte des faiblesses : PersonaMem-v2")
    parser.add_argument("--personas", type=int, default=40)
    parser.add_argument("--graine", type=int, default=7)
    parser.add_argument("--answer-model", default="nvidia/nemotron-3-ultra-550b-a55b")
    parser.add_argument("--answer-backend", choices=("ollama", "nvidia", "mistral"), default="nvidia")
    parser.add_argument("--output", type=Path, default=Path("bench/results/carte/carte_personamem.json"))
    parser.add_argument("--cache", type=Path, default=Path("bench/results/carte/carte_personamem.cache.jsonl"))
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(run(args))


if __name__ == "__main__":
    main()

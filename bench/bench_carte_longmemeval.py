"""Carte des faiblesses, LongMemEval-S : où perd-on, et est-ce à nous ?

Même méthode que `bench_carte_personamem.py` (AML_COMPETITION_PREP.md §29), par
type de question LongMemEval, sur un échantillon stratifié :
- **preuves@100** : toutes les répliques `has_answer` sont-elles dans le top 100 ?
- **A** : consigne et juge officiels (`data/longmemeval-s/pipeline.py`, identiques à
  ceux de LoCoMo) nourris du contexte Mnemos, `created_at` affiché ;
- **O** : la même consigne nourrie des seules sessions qui portent la réponse
  (`answer_session_ids`) — le plafond du répondeur, mémoire parfaite.

La plateforme n'envoie pas de date de question (`/search` n'en a pas) : ni A ni O
ne la reçoivent. Les questions « il y a combien de jours » sont donc hors de portée
pour tout le monde ; elles restent comptées, et lisibles dans le type temporel.

Chaque question a son propre historique (~490 répliques) : l'ingestion domine le
temps. Reprenable (cache JSONL). LongMemEval est public.
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
from bench.bench_temporal_recall import (
    LONGMEMEVAL,
    PAS_MS,
    _date_lme,
    _ecrire,
    _iso,
    garder_la_boucle,
)
from bench.remote_llm import RemoteChat
from mnemos.config import Settings
from mnemos.llm.ollama_client import OllamaClient

TOP_K = 100


def echantillon(data: list[dict[str, Any]], par_type: int, graine: int) -> list[dict[str, Any]]:
    rng = random.Random(graine)
    groupes: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for q in data:
        groupes[q["question_type"]].append(q)
    choisies: list[dict[str, Any]] = []
    for t in sorted(groupes):
        choisies += rng.sample(groupes[t], min(par_type, len(groupes[t])))
    return choisies


async def run(args: argparse.Namespace) -> None:
    garder_la_boucle(asyncio.get_running_loop())
    officiel = aml_officiel.charger("longmemeval-s")
    data = json.loads(LONGMEMEVAL.read_text(encoding="utf-8"))
    questions = echantillon(data, args.par_type, args.graine)
    del data
    repondeur: Any = (OllamaClient(Settings(_env_file=None))  # type: ignore[call-arg]
                      if args.answer_backend == "ollama" else RemoteChat(args.answer_backend))
    opts = {"temperature": 0.0, "num_ctx": 32768, "num_predict": 256}
    faits: dict[str, dict[str, Any]] = {}
    if args.cache.exists():
        for ligne in args.cache.read_text(encoding="utf-8").splitlines():
            if ligne.strip():
                x = json.loads(ligne)
                faits[x["cle"]] = x
    if faits:
        print(f"Reprise : {len(faits)} questions déjà dans {args.cache}", flush=True)
    tmp = Path(tempfile.mkdtemp(prefix="mnemos_carte_lme_"))
    try:
        app, _, _ = await setup_bench_app(tmp, "ollama", salience_workers=0)
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://t", timeout=600) as client:
                for n, q in enumerate(questions, 1):
                    if q["question_id"] in faits:
                        continue
                    tenant = f"lme:{q['question_id']}"
                    cle_vers_tour: dict[str, str] = {}
                    preuves: list[str] = []
                    oracle: list[tuple[str, str]] = []
                    for j, (sid, date, tours) in enumerate(zip(
                            q["haystack_session_ids"], q["haystack_dates"], q["haystack_sessions"], strict=True)):
                        base = _date_lme(date) + j
                        msgs = []
                        for i, t in enumerate(tours):
                            ts = base + i * PAS_MS
                            cle_vers_tour[_iso(ts)] = f"{sid}#{i}"
                            if t.get("has_answer"):
                                preuves.append(f"{sid}#{i}")
                            if sid in q["answer_session_ids"]:
                                oracle.append((_iso(ts), f"{t['role']}: {t['content']}"))
                            if t.get("content", "").strip():
                                msgs.append({"role": t["role"], "content": t["content"], "timestamp": ts})
                        await _ecrire(client, tenant, sid, msgs)
                    items = (await client.post("/search", json={
                        "query": q["question"], "user_id": tenant, "top_k": TOP_K})).json()["data"]
                    rendus = {cle_vers_tour.get(str(it.get("created_at", ""))) for it in items}
                    contexte_a = [(str(it.get("created_at", "")), str(it["content"])) for it in items]
                    verdicts = {}
                    try:
                        for nom, ctx in (("A", contexte_a), ("O", oracle)):
                            reponse = (await repondeur.generate(aml_officiel.consigne_reponse(
                                officiel, q["question"], [("user", ctx)], True),
                                args.answer_model, options=opts)).strip()
                            sortie = await repondeur.generate(aml_officiel.consigne_juge(
                                officiel, q["question"], str(q["answer"]), reponse),
                                args.answer_model, options=opts)
                            verdicts[nom] = aml_officiel.verdict(officiel, sortie)
                    except Exception as exc:  # noqa: BLE001 — Nvidia (429/503) ou Ollama (500)
                        print(f"  sautée, à refaire : {q['question_id']} ({exc})", flush=True)
                        continue
                    x = {"cle": q["question_id"], "type": q["question_type"],
                         "abstention": q["question_id"].endswith("_abs"),
                         "preuves_toutes_100": bool(preuves) and all(p in rendus for p in preuves),
                         "preuves_n": len(preuves),
                         "A": bool(verdicts["A"]), "O": bool(verdicts["O"]),
                         "juge_illisible": verdicts["A"] is None or verdicts["O"] is None}
                    faits[x["cle"]] = x
                    args.cache.parent.mkdir(parents=True, exist_ok=True)
                    with args.cache.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(x, ensure_ascii=False) + "\n")
                    print(f"  [{n}/{len(questions)}] {q['question_type']}: A={x['A']} O={x['O']}"
                          f" preuves@100={x['preuves_toutes_100']}", flush=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        await repondeur.aclose()

    par_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for x in faits.values():
        par_type[x["type"]].append(x)
        par_type["TOUS"].append(x)
    resume = {t: {"n": len(xs),
                  "preuves@100": round(sum(x["preuves_toutes_100"] for x in xs) / len(xs), 3),
                  "A": round(sum(x["A"] for x in xs) / len(xs), 3),
                  "O": round(sum(x["O"] for x in xs) / len(xs), 3)}
              for t, xs in sorted(par_type.items())}
    res = {"config": {"par_type": args.par_type, "graine": args.graine, "answer_model": args.answer_model,
                      "answer_backend": args.answer_backend,
                      "aml_pipeline_commit": ((aml_officiel.RACINE / "COMMIT").read_text().strip()
                                              if (aml_officiel.RACINE / "COMMIT").exists() else None),
                      "juge_illisible": sum(x["juge_illisible"] for x in faits.values())},
           "par_type": resume, "questions": list(faits.values())}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n{'type':28s} {'n':>4s} {'preuves@100':>11s} {'A':>6s} {'O':>6s}")
    for t, m in resume.items():
        print(f"{t:28s} {m['n']:4d} {m['preuves@100']:11.3f} {m['A']:6.3f} {m['O']:6.3f}")
    print(f"Rapport : {args.output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Carte des faiblesses : LongMemEval-S")
    parser.add_argument("--par-type", type=int, default=10)
    parser.add_argument("--graine", type=int, default=7)
    parser.add_argument("--answer-model", default="nvidia/nemotron-3-ultra-550b-a55b")
    parser.add_argument("--answer-backend", choices=("ollama", "nvidia", "mistral"), default="nvidia")
    parser.add_argument("--output", type=Path, default=Path("bench/results/carte/carte_longmemeval.json"))
    parser.add_argument("--cache", type=Path, default=Path("bench/results/carte/carte_longmemeval.cache.jsonl"))
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(run(args))


if __name__ == "__main__":
    main()

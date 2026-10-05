"""Le normaliseur de dates améliore-t-il les RÉPONSES ? Consignes officielles, LoCoMo.

Mêmes questions, deux mémoires : normaliseur désactivé, puis activé
(`RELATIVE_DATES_ANNOTATION`). Chaque mémoire ingère les conversations par la
route AML `/add` ; chaque question passe par `/search` (top_k = 100), puis par
la consigne officielle du répondeur et celle du juge (`bench/aml_officiel.py`),
souvenirs rendus par locuteur avec `created_at`, comme le suggère la consigne
(« when the memory timestamp makes it clear »).

Jeu, fixé avant la mesure (AML_COMPETITION_PREP.md §27) : les questions de dates
(catégorie 2) des conversations de validation — toutes sauf conv-26, d'où le
périmètre du normaliseur a été tiré — et un échantillon fixe d'autres catégories
pour vérifier que rien d'autre ne recule. LoCoMo est public : un répondeur distant
est permis.

Les résultats sont écrits au fil de l'eau (cache JSONL) : relancer reprend le run.
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
from math import comb
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ.setdefault("EPISODIC_RETENTION_DAYS", "36500")
os.environ.setdefault("DECAY_RATE_DAILY", "0.0")

import httpx

from bench import aml_officiel
from bench.bench_locomo import parse_locomo_datetime, setup_bench_app
from bench.bench_temporal_recall import garder_la_boucle
from bench.remote_llm import RemoteChat
from mnemos.config import Settings
from mnemos.llm.ollama_client import OllamaClient

LOCOMO = Path("bench/data/locomo10.json")
DEV = "conv-26"
CONDITIONS = ("sans", "avec")  # normaliseur désactivé / activé


def _mcnemar(perdues: int, gagnees: int) -> float:
    n = perdues + gagnees
    if n == 0:
        return 1.0
    k = min(perdues, gagnees)
    return float(min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n))


def choisir(data: list[dict[str, Any]], convs: list[str], autres: int,
            graine: int) -> list[tuple[str, int, dict[str, Any]]]:
    """(conv, index, question) : toutes les cat. 2, plus `autres` questions tirées
    une fois pour toutes parmi les cat. 1, 3 et 4."""
    dates: list[tuple[str, int, dict[str, Any]]] = []
    reste: list[tuple[str, int, dict[str, Any]]] = []
    for c in data:
        if c["sample_id"] not in convs:
            continue
        for i, q in enumerate(c["qa"]):
            if q.get("answer") is None or q.get("category") == 5:
                continue
            (dates if q.get("category") == 2 else reste).append((c["sample_id"], i, q))
    return dates + random.Random(graine).sample(reste, min(autres, len(reste)))


async def _ingerer(client: httpx.AsyncClient, c: dict[str, Any]) -> None:
    conv = c["conversation"]
    sessions = sorted((k for k in conv if k.startswith("session_") and not k.endswith("_date_time")),
                      key=lambda k: int(k.split("_")[1]))
    for s_key in sessions:
        base = parse_locomo_datetime(conv.get(f"{s_key}_date_time"))
        msgs = [{"role": "user" if t.get("speaker") == conv["speaker_a"] else "assistant",
                 "content": f"{t.get('speaker')}: {t.get('text', '')}", "timestamp": base + i * 15_000}
                for i, t in enumerate(conv[s_key])]
        for j in range(0, len(msgs), 20):
            r = await client.post("/add", json={
                "request_id": f"{c['sample_id']}:{s_key}:{j}", "user_id": f"locomo:{c['sample_id']}",
                "session_id": s_key, "messages": msgs[j:j + 20]})
            r.raise_for_status()


async def run(args: argparse.Namespace) -> dict[str, Any]:
    garder_la_boucle(asyncio.get_running_loop())
    data = json.loads(LOCOMO.read_text(encoding="utf-8"))
    par_conv = {c["sample_id"]: c for c in data}
    convs = args.convs or [c["sample_id"] for c in data if c["sample_id"] != DEV]
    questions = choisir(data, convs, args.autres, args.graine)
    if args.limite:
        questions = questions[: args.limite]  # essai rapide seulement
    officiel = aml_officiel.charger("locomo-refined")
    repondeur: Any = (OllamaClient(Settings(_env_file=None))  # type: ignore[call-arg]
                      if args.answer_backend == "ollama" else RemoteChat(args.answer_backend))
    opts = {"temperature": 0.0, "num_ctx": 16384, "num_predict": 256}
    # Mise en route avec les MÊMES réglages que les questions : un num_ctx différent
    # fait recharger le modèle par Ollama, et le 05/10 chaque rechargement a buté
    # quelques secondes sur la connexion à son propre processus (500, « connectex »).
    for essai in range(5):
        try:
            await repondeur.generate("Réponds OK.", args.answer_model,
                                     options={**opts, "num_predict": 4})
            break
        except Exception:  # noqa: BLE001 — répondeur pas encore prêt
            if essai == 4:
                raise
            print("  répondeur pas prêt, nouvel essai dans 20 s", flush=True)
            await asyncio.sleep(20)

    faits: dict[str, dict[str, Any]] = {}
    if args.cache.exists():
        for ligne in args.cache.read_text(encoding="utf-8").splitlines():
            if ligne.strip():
                x = json.loads(ligne)
                faits[x["cle"]] = x
    if faits:
        print(f"Reprise : {len(faits)} réponses déjà dans {args.cache}", flush=True)
    verrou = asyncio.Semaphore(args.parallele)

    try:
        for condition in CONDITIONS:
            restantes = [(cid, i, q) for cid, i, q in questions
                         if f"{condition}|{cid}|{i}" not in faits]
            if not restantes:
                continue
            # Settings lit l'environnement : le réglage vaut pour l'application créée ici.
            os.environ["RELATIVE_DATES_ANNOTATION"] = "true" if condition == "avec" else "false"
            tmp = Path(tempfile.mkdtemp(prefix=f"mnemos_dates_{condition}_"))
            try:
                app, _, settings = await setup_bench_app(tmp, "ollama", salience_workers=0)
                assert settings.RELATIVE_DATES_ANNOTATION is (condition == "avec")
                async with app.router.lifespan_context(app):
                    transport = httpx.ASGITransport(app=app)
                    async with httpx.AsyncClient(transport=transport, base_url="http://t",
                                                 timeout=600) as client:
                        t0 = time.perf_counter()
                        for cid in sorted({cid for cid, _, _ in restantes}):
                            await _ingerer(client, par_conv[cid])
                        print(f"[{condition}] ingestion {time.perf_counter() - t0:.0f} s,"
                              f" {len(restantes)} questions", flush=True)
                        faites = 0

                        async def une(cid: str, i: int, q: dict[str, Any],
                                      cond: str = condition,
                                      cl: httpx.AsyncClient = client,
                                      total: int = len(restantes),
                                      debut: float = t0) -> None:
                            nonlocal faites
                            conv = par_conv[cid]["conversation"]
                            items = (await cl.post("/search", json={
                                "query": q["question"], "user_id": f"locomo:{cid}",
                                "top_k": args.top_k})).json()["data"]
                            souvenirs = [(str(it.get("created_at", "")), it["content"]) for it in items]
                            locuteurs = [(nom, [s for s in souvenirs if s[1].startswith(f"{nom}:")])
                                         for nom in (conv["speaker_a"], conv["speaker_b"])]
                            # 05/10 : dix 429 de suite sur un appel ont fait tomber tout le
                            # lot. Une question en échec est sautée, pas enregistrée : la
                            # reprise suivante la refait.
                            try:
                                async with verrou:
                                    reponse = (await repondeur.generate(aml_officiel.consigne_reponse(
                                        officiel, q["question"], locuteurs, True),
                                        args.answer_model, options=opts)).strip()
                                    sortie = await repondeur.generate(aml_officiel.consigne_juge(
                                        officiel, q["question"], str(q["answer"]), reponse),
                                        args.answer_model, options=opts)
                            except Exception as exc:  # noqa: BLE001 — Nvidia (429/503) ou Ollama (500)
                                print(f"  [{cond}] sautée, à refaire : {cid} #{i} ({exc})", flush=True)
                                return
                            juge = aml_officiel.verdict(officiel, sortie)
                            x = {"cle": f"{cond}|{cid}|{i}", "condition": cond, "conv": cid,
                                 "categorie": q["category"], "question": q["question"],
                                 "corrige": str(q["answer"]), "reponse": reponse,
                                 "juste": bool(juge), "juge_illisible": juge is None}
                            faits[x["cle"]] = x
                            args.cache.parent.mkdir(parents=True, exist_ok=True)
                            with args.cache.open("a", encoding="utf-8") as f:
                                f.write(json.dumps(x, ensure_ascii=False) + "\n")
                            faites += 1
                            if faites % 20 == 0:
                                print(f"  [{cond}] {faites}/{total}"
                                      f" ({time.perf_counter() - debut:.0f} s)", flush=True)

                        await asyncio.gather(*(une(cid, i, q) for cid, i, q in restantes))
            finally:
                shutil.rmtree(tmp, ignore_errors=True)
    finally:
        await repondeur.aclose()

    lignes = []
    resultat: dict[str, Any] = {"config": {
        "convs": convs, "autres": args.autres, "graine": args.graine, "top_k": args.top_k,
        "answer_model": args.answer_model, "answer_backend": args.answer_backend,
        "aml_pipeline_commit": ((aml_officiel.RACINE / "COMMIT").read_text().strip()
                                if (aml_officiel.RACINE / "COMMIT").exists() else None),
        "juge_illisible": sum(x["juge_illisible"] for x in faits.values())}, "groupes": {}}
    for nom, dates in (("dates (cat. 2)", True), ("autres catégories", False)):
        paires: list[tuple[dict[str, Any], dict[str, Any]]] = []
        for cid, i, q in questions:
            s, a = faits.get(f"sans|{cid}|{i}"), faits.get(f"avec|{cid}|{i}")
            if (q["category"] == 2) is dates and s is not None and a is not None:
                paires.append((s, a))
        if not paires:
            continue
        sans = sum(s["juste"] for s, _ in paires)
        avec = sum(a["juste"] for _, a in paires)
        perdues = sum(s["juste"] and not a["juste"] for s, a in paires)
        gagnees = sum(a["juste"] and not s["juste"] for s, a in paires)
        resultat["groupes"][nom] = {"n": len(paires), "sans": sans, "avec": avec,
                                    "perdues": perdues, "gagnees": gagnees,
                                    "p": round(_mcnemar(perdues, gagnees), 4)}
        lignes.append(f"{nom:<20} N={len(paires):3d} | sans {sans:3d} | avec {avec:3d}"
                      f" | perdues {perdues:2d} gagnées {gagnees:2d}"
                      f" | McNemar p = {_mcnemar(perdues, gagnees):.3f}")
    resultat["reponses"] = list(faits.values())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(resultat, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n" + "\n".join(lignes))
    print(f"juge illisible : {resultat['config']['juge_illisible']} | Rapport : {args.output}")
    return resultat


def main() -> None:
    parser = argparse.ArgumentParser(description="Normaliseur de dates : effet sur les réponses")
    parser.add_argument("--convs", nargs="*", default=None,
                        help="conversations LoCoMo ; défaut : toutes sauf conv-26 (validation)")
    parser.add_argument("--autres", type=int, default=60, help="questions d'autres catégories")
    parser.add_argument("--graine", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--answer-model", default="nvidia/nemotron-3-ultra-550b-a55b")
    parser.add_argument("--answer-backend", choices=("ollama", "nvidia", "mistral"), default="nvidia")
    parser.add_argument("--parallele", type=int, default=3, help="appels au répondeur en parallèle")
    parser.add_argument("--limite", type=int, default=None, help="essai rapide : N premières questions")
    parser.add_argument("--output", type=Path, default=Path("bench/results/gpu/dates_reponse.json"))
    parser.add_argument("--cache", type=Path, default=Path("bench/results/gpu/dates_reponse.cache.jsonl"))
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if sys.platform == "win32":
        # Voir bench_locomo_qa : la boucle Proactor reste figée après une coupure réseau.
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(run(args))


if __name__ == "__main__":
    main()

"""Consignes durables sur BEAM 100K : les épingler en fin de liste, qu'est-ce que ça change ?

Détecteur (`bench_consignes_durables.py`) : les 35 consignes testées par BEAM sont toutes
détectées (3 à 5 par conversation), et il ne se déclenche presque jamais ailleurs (LoCoMo 0,
PersonaMem 0, LongMemEval-S 4 vraies consignes sur 94 000 messages). Le mécanisme qu'on
livrerait : les consignes durables du tenant absentes du top `top_k` le rejoignent en fin
de liste, à la place des derniers (`plafond` au plus, les plus récentes).

Phase `selection` (sans modèle) : preuves des questions « instruction following » avant et
après, et preuves des autres questions poussées hors de la liste par les ajouts.
Phase `reponse` : consigne et juge officiels de BEAM, Nemotron 3 Ultra. Questions de
consignes : A, A+oracle (la seule consigne testée, ajoutée si absente), A+épinglé. Autres
types, un échantillon : A et A+épinglé si le contexte change. A redemandé sur 30 % des
questions au contexte changé (bruit).

Base ingérée gardée (`--base`), embeddings par llama-server.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
from collections import defaultdict
from math import comb
from pathlib import Path
from statistics import mean
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ.setdefault("EPISODIC_RETENTION_DAYS", "36500")
os.environ.setdefault("DECAY_RATE_DAILY", "0.0")

import httpx
from sqlalchemy import select

from bench import aml_officiel
from bench.bench_beam_preuves import charger, ids_preuves, present
from bench.bench_consignes_durables import est_consigne_durable
from bench.bench_locomo import setup_bench_app
from bench.bench_temporal_recall import garder_la_boucle
from bench.remote_llm import RemoteChat
from mnemos.api.aml_routes import _ts_to_iso
from mnemos.models.episodic import Episode

TOP_K = 100
PLAFOND = 5
LOT_MESSAGES = 20
MODELE = "nvidia/nemotron-3-ultra-550b-a55b"


def epingler_consignes(items: list[dict[str, Any]], consignes: list[dict[str, Any]], top_k: int,
                       plafond: int) -> list[dict[str, Any]]:
    """Les consignes durables absentes de `items` les rejoignent en fin de liste, à la place
    des derniers : `plafond` au plus, les plus récentes (`consignes` est dans l'ordre
    chronologique). Sans effet s'il n'y en a pas."""
    presents = {it["id"] for it in items}
    ajouts = [c for c in consignes if c["id"] not in presents][-plafond:]
    if not ajouts:
        return items[:top_k]
    return items[:min(len(items), top_k - len(ajouts))] + ajouts


def _signe(perdues: int, gagnees: int) -> float:
    n = perdues + gagnees
    return min(1.0, 2 * sum(comb(n, i) for i in range(min(perdues, gagnees) + 1)) / 2 ** n) if n else 1.0


async def consignes_du_tenant(store: Any, tenant: str) -> list[dict[str, Any]]:
    """Les consignes durables stockées pour ce tenant, au format d'un résultat `/search`,
    dans l'ordre d'écriture."""
    async with store._sessions() as s:
        eps = (await s.execute(select(Episode.id, Episode.role, Episode.content, Episode.created_at)
                               .where(Episode.tenant == tenant, Episode.archived == 0)
                               .order_by(Episode.created_at, Episode.id))).all()
    return [{"id": f"ep_{e}", "content": str(c), "created_at": _ts_to_iso(t)}
            for e, r, c, t in eps if est_consigne_durable(str(c), str(r))]


def contexte(items: list[dict[str, Any]]) -> str:
    return "\n".join(f"[{it.get('created_at', '')}] {it['content']}" for it in items)


async def run(args: argparse.Namespace) -> None:
    garder_la_boucle(asyncio.get_running_loop())
    args.base.mkdir(parents=True, exist_ok=True)
    marque = args.base / "ingeres.json"
    ingeres: set[str] = set(json.loads(marque.read_text(encoding="utf-8"))) if marque.exists() else set()
    officiel = aml_officiel.charger("beam", "pipeline.py") if args.phase == "reponse" else None
    repondeur = RemoteChat("nvidia") if args.phase == "reponse" else None
    faits: dict[str, dict[str, Any]] = {}
    if args.phase == "reponse" and args.cache.exists():
        for x in args.cache.read_text(encoding="utf-8").split("\n"):
            if x.strip():
                y = json.loads(x)
                faits[y["cle"]] = y
        print(f"Reprise : {len(faits)} questions déjà dans {args.cache}", flush=True)
    # Échantillon fixé d'avance : toutes les questions de consignes, `--par-type` des autres
    rng = random.Random(args.graine)
    plan: dict[int, list[tuple[str, int]]] = defaultdict(list)
    tous = [(n, t, k) for n in range(1, 21) for t, qs in charger(n)[1].items() for k in range(len(qs))]
    par_type: dict[str, list[tuple[int, str, int]]] = defaultdict(list)
    for n, t, k in tous:
        par_type[t].append((n, t, k))
    for t, xs in par_type.items():
        garde = xs if (t == "instruction_following" or args.phase == "selection") else rng.sample(xs, args.par_type)
        for n, t2, k in garde:
            plan[n].append((t2, k))
    stats: list[dict[str, Any]] = []

    async def juger(question: str, reponse: str, rubriques: list[str]) -> float:
        assert officiel is not None and repondeur is not None
        for _ in range(3):
            sortie = await repondeur.generate(officiel.render_batch_judge_prompt(question, reponse, rubriques),
                                              MODELE, options={"temperature": 0.0, "num_predict": 2048})
            try:
                notes = officiel.parse_rubric_scores(sortie, len(rubriques))
                return float(mean(x["score"] for x in notes))
            except (ValueError, KeyError):
                continue
        raise RuntimeError("juge : JSON illisible après 3 essais")

    async def repondre(question: str, items: list[dict[str, Any]]) -> str:
        assert officiel is not None and repondeur is not None
        prompt = officiel.render_answer_prompt({"context": contexte(items), "question": question})
        return str(await repondeur.generate(prompt, MODELE, options={"temperature": 0.0, "num_predict": 512}))

    try:
        app, _, _ = await setup_bench_app(args.base, "ollama", salience_workers=0, embed_backend="llamacpp")
        async with app.router.lifespan_context(app):
            store = app.state.store
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://t", timeout=900) as client:
                for n in sorted(plan):
                    sessions, questions = charger(n)
                    tenant = f"beam:{n}"
                    par_id = {int(m["id"]): m for s in sessions for m in s}
                    if tenant not in ingeres:
                        for s, msgs in enumerate(sessions):
                            for i in range(0, len(msgs), LOT_MESSAGES):
                                lot = [{"role": m["role"], "content": str(m["content"])}
                                       for m in msgs[i:i + LOT_MESSAGES] if str(m["content"]).strip()]
                                (await client.post("/add", json={
                                    "request_id": f"{tenant}:{s}:{i}", "user_id": tenant,
                                    "session_id": f"{tenant}:{s}", "messages": lot})).raise_for_status()
                        ingeres.add(tenant)
                        marque.write_text(json.dumps(sorted(ingeres)), encoding="utf-8")
                    consignes = await consignes_du_tenant(store, tenant)
                    for t, k in plan[n]:
                        cle = f"{n}:{t}:{k}"
                        if cle in faits:
                            continue
                        q = questions[t][k]
                        items = (await client.post("/search", json={
                            "query": str(q["question"]), "user_id": tenant, "top_k": TOP_K})).json()["data"]
                        epingle = epingler_consignes(items, consignes, TOP_K, PLAFOND)
                        preuves = [p for p in dict.fromkeys(ids_preuves(q.get("source_chat_ids"))) if p in par_id]
                        avant = [present(str(par_id[p]["content"]), [str(it["content"]) for it in items]) for p in preuves]
                        apres = [present(str(par_id[p]["content"]), [str(it["content"]) for it in epingle]) for p in preuves]
                        if args.phase == "selection":
                            stats.append({"cle": cle, "type": t, "preuves": len(preuves),
                                          "avant": sum(r is not None for r in avant),
                                          "apres": sum(r is not None for r in apres),
                                          "ajouts": sum(it["id"] not in {x["id"] for x in items} for it in epingle)})
                            continue
                        # phase « reponse »
                        change = [it["id"] for it in epingle] != [it["id"] for it in items]
                        bras: dict[str, list[dict[str, Any]]] = {"A": items}
                        if change:
                            bras["A+"] = epingle
                        if t == "instruction_following":
                            testees = [c for c in consignes if any(
                                str(par_id[p]["content"]).strip()[:200] == c["content"].strip()[:200] for p in preuves)]
                            oracle = epingler_consignes(items, testees, TOP_K, PLAFOND)
                            if [it["id"] for it in oracle] != [it["id"] for it in items]:
                                bras["oracle"] = oracle
                        if change and random.Random(cle).random() < args.bis:
                            bras["A_bis"] = items
                        rubriques = officiel.rubric_items(q) if officiel else []
                        notes: dict[str, float] = {}
                        try:
                            for nom, lst in bras.items():
                                rep = await repondre(str(q["question"]), lst)
                                notes[nom] = await juger(str(q["question"]), rep, rubriques)
                        except Exception as exc:  # noqa: BLE001 — Nvidia 429/503, JSON du juge
                            print(f"  sautée, à refaire : {cle} ({str(exc)[:100]})", flush=True)
                            continue
                        y = {"cle": cle, "type": t, "change": change, "notes": notes}
                        faits[cle] = y
                        with args.cache.open("a", encoding="utf-8") as fh:
                            fh.write(json.dumps(y, ensure_ascii=False) + "\n")
                    print(f"  conversation {n} : {len(consignes)} consignes", flush=True)
    finally:
        if repondeur is not None:
            await repondeur.aclose()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rapport = synthese_selection(stats) if args.phase == "selection" else synthese_reponse(list(faits.values()))
    args.output.write_text(json.dumps(rapport, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Rapport : {args.output}")


def synthese_selection(stats: list[dict[str, Any]]) -> dict[str, Any]:
    res: dict[str, Any] = {}
    print(f"\n{'type':28s} {'n':>4s} {'toutes avant':>13s} {'toutes après':>13s} {'preuves perdues':>16s} {'ajouts moy.':>12s}")
    for t in sorted({x["type"] for x in stats}):
        xs = [x for x in stats if x["type"] == t and x["preuves"]]
        if not xs:
            continue
        r = {"n": len(xs), "toutes_avant": sum(x["avant"] == x["preuves"] for x in xs),
             "toutes_apres": sum(x["apres"] == x["preuves"] for x in xs),
             "preuves_perdues": sum(max(0, x["avant"] - x["apres"]) for x in xs),
             "ajouts_moyens": round(mean(x["ajouts"] for x in xs), 2)}
        res[t] = r
        print(f"{t:28s} {r['n']:>4d} {r['toutes_avant']:>13d} {r['toutes_apres']:>13d} {r['preuves_perdues']:>16d}"
              f" {r['ajouts_moyens']:>12}")
    return {"par_type": res, "details": stats}


def synthese_reponse(faits: list[dict[str, Any]]) -> dict[str, Any]:
    res: dict[str, Any] = {}
    print(f"\n{'type':28s} {'n':>4s} {'A':>6s} {'A+':>6s} {'Δ':>7s} {'+/-':>7s} {'p':>6s}")
    for t in sorted({x["type"] for x in faits}):
        xs = [x for x in faits if x["type"] == t]
        diffs = [x["notes"].get("A+", x["notes"]["A"]) - x["notes"]["A"] for x in xs]
        g, p = sum(d > 0 for d in diffs), sum(d < 0 for d in diffs)
        r = {"n": len(xs), "A": round(mean(x["notes"]["A"] for x in xs), 3),
             "A+": round(mean(x["notes"].get("A+", x["notes"]["A"]) for x in xs), 3),
             "delta_moyen": round(mean(diffs), 3), "mieux": g, "moins_bien": p, "p_signe": round(_signe(p, g), 4)}
        if t == "instruction_following":
            o = [x["notes"].get("oracle", x["notes"]["A"]) - x["notes"]["A"] for x in xs]
            r["oracle_delta_moyen"] = round(mean(o), 3)
        res[t] = r
        print(f"{t:28s} {r['n']:>4d} {r['A']:>6} {r['A+']:>6} {r['delta_moyen']:>+7.3f} {g:>3d}/{p:<3d} {r['p_signe']:>6}")
    # Net total estimé sur BEAM 100K : 40 questions par type
    total = sum(r["delta_moyen"] * 40 for r in res.values())
    bis = [x for x in faits if "A_bis" in x["notes"]]
    bruit = {"n": len(bis), "ecart_moyen_absolu": round(mean(abs(x["notes"]["A_bis"] - x["notes"]["A"]) for x in bis), 3)
             if bis else None, "verdict_change": sum(x["notes"]["A_bis"] != x["notes"]["A"] for x in bis)}
    print(f"\nnet total estimé sur BEAM 100K (400 questions) : {total:+.2f} points de note ; bruit : {bruit}")
    return {"par_type": res, "net_total_estime": round(total, 2), "bruit": bruit, "details": faits}


def main() -> None:
    parser = argparse.ArgumentParser(description="Consignes durables épinglées, BEAM 100K")
    parser.add_argument("--phase", choices=("selection", "reponse"), default="selection")
    parser.add_argument("--base", type=Path, default=Path("bench/data/bases/beam_100k"))
    parser.add_argument("--par-type", type=int, default=8, help="phase reponse : questions par type hors consignes")
    parser.add_argument("--graine", type=int, default=7)
    parser.add_argument("--bis", type=float, default=0.3)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--cache", type=Path, default=Path("bench/results/consignes/beam_reponse.cache.jsonl"))
    args = parser.parse_args()
    args.output = args.output or Path(f"bench/results/consignes/beam_{args.phase}.json")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(run(args))


if __name__ == "__main__":
    main()

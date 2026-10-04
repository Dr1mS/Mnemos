"""Temporel : la recherche rend-elle TOUS les événements qu'une question demande ?

Diagnostic du 04/10/2026 sur LoCoMo : 30 des 32 échecs temporels de Nemotron
viennent d'une date relative non résolue (« last Friday » recopié), alors que la
preuve est retrouvée. Ce côté-là est hors de notre portée : le `content` rendu
reste verbatim (ligne rouge, voir la mémoire du projet). Ce qui est à nous,
c'est la SÉLECTION. Une question temporelle demande souvent deux événements
(« lequel d'abord ? », « combien de jours entre ? ») : le meilleur rang d'une
preuve ne dit pas si la seconde manque. On mesure donc le rappel de TOUTES les
preuves, par la vraie route AML `/search` (top_k = 100 comme au Full).

Aucun modèle de langage : la mesure est déterministe et ne coûte que des
embeddings (servis par le llama-server de la production).

Jeux publics (téléchargés, non suivis par git) :
- LoCoMo complet, `bench/data/locomo10.json` : questions catégorie 2 (dates),
  les autres catégories servant de comparaison. Développement sur conv-26
  (celle de `locomo_sample.json`), validation sur les neuf autres.
- LongMemEval-S nettoyé, `bench/data/longmemeval/longmemeval_s_cleaned.json` :
  questions `temporal-reasoning`, preuves = tours `has_answer`. Moitiés de
  développement et de validation par parité d'un hachage de `question_id`.

Règle d'arrêt, écrite avant la mesure (AML_COMPETITION_PREP.md §27) : si le
rappel de toutes les preuves dans le top 100 atteint ~90 % sur les deux jeux,
le temporel se joue côté réponse et on le referme.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import shutil
import statistics
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Comme en production AML : pas d'oubli temporel sur des données datées de 2023.
os.environ.setdefault("EPISODIC_RETENTION_DAYS", "36500")
os.environ.setdefault("DECAY_RATE_DAILY", "0.0")

import httpx

from bench.bench_locomo import parse_locomo_datetime, setup_bench_app

LOCOMO = Path("bench/data/locomo10.json")
LONGMEMEVAL = Path("bench/data/longmemeval/longmemeval_s_cleaned.json")
LOCOMO_DEV = "conv-26"
PAS_MS = 15_000  # écart entre deux tours d'une session, comme bench_locomo_qa
LOT = 20


def _iso(ts_ms: int) -> str:
    """Même rendu que `aml_routes._ts_to_iso` : la clé de correspondance."""
    return datetime.fromtimestamp(ts_ms / 1000.0, tz=UTC).isoformat().replace("+00:00", "Z")


def _preuves_locomo(q: dict[str, Any]) -> list[str]:
    ids: list[str] = []
    for ev in q.get("evidence") or []:
        ids.extend(p.strip() for p in str(ev).replace(",", ";").split(";") if p.strip())
    return ids


def _date_lme(s: str) -> int:
    """« 2023/02/01 (Wed) 10:20 » → epoch ms UTC."""
    jour, _, heure = s.partition(") ")
    dt = datetime.strptime(f"{jour.split(' (')[0]} {heure}", "%Y/%m/%d %H:%M")
    return int(dt.replace(tzinfo=UTC).timestamp() * 1000)


def garder_la_boucle(loop: asyncio.AbstractEventLoop) -> None:
    """Reconstruit le canal de réveil de la boucle s'il est coupé.

    Le 04/10 à 23 h 01, ce canal (une connexion locale sur 127.0.0.1) a reçu un
    `WinError 10054` dès le démarrage : la boucle Selector relançait l'erreur à
    chaque tour et a écrit 199 Mo de journal en 3 minutes. Le 02/10, la même
    coupure avait figé la boucle Proactor. Cause hors de Mnemos, non élucidée (une
    pile réseau chargée de filtres : Npcap, Vanguard, Tailscale…) et passagère :
    on reconstruit le canal et on le signale une fois."""
    coupures = 0
    defaut = loop.get_exception_handler()

    def gestionnaire(lp: asyncio.AbstractEventLoop, ctx: dict[str, Any]) -> None:
        nonlocal coupures
        if (isinstance(ctx.get("exception"), ConnectionResetError)
                and "_read_from_self" in repr(ctx.get("handle", ""))):
            coupures += 1
            if coupures in (1, 10, 100):
                print(f"  canal de réveil de la boucle coupé ({coupures} fois), reconstruit", flush=True)
            try:
                lp._close_self_pipe()  # type: ignore[attr-defined]
                lp._make_self_pipe()  # type: ignore[attr-defined]
            except Exception:  # noqa: BLE001 — au pire, l'erreur se répète
                pass
            return
        if defaut is not None:
            defaut(lp, ctx)
        else:
            lp.default_exception_handler(ctx)

    loop.set_exception_handler(gestionnaire)


def _moitie_lme(question_id: str) -> str:
    h = int(hashlib.md5(question_id.encode()).hexdigest(), 16)
    return "dev" if h % 2 == 0 else "val"


async def _ecrire(client: httpx.AsyncClient, user_id: str, session: str,
                  messages: list[dict[str, Any]]) -> None:
    for i in range(0, len(messages), LOT):
        r = await client.post("/add", json={
            "request_id": f"{user_id}:{session}:{i // LOT}", "user_id": user_id,
            "session_id": session, "messages": messages[i:i + LOT],
        })
        r.raise_for_status()


async def _rangs(client: httpx.AsyncClient, user_id: str, question: str, top_k: int,
                 cle_vers_preuve: dict[str, str], preuves: list[str]) -> list[int | None]:
    items = (await client.post("/search", json={
        "query": question, "user_id": user_id, "top_k": top_k,
    })).json()["data"]
    rang_par_preuve: dict[str, int] = {}
    for rang, it in enumerate(items, 1):
        p = cle_vers_preuve.get(str(it.get("created_at", "")))
        if p is not None and p not in rang_par_preuve:
            rang_par_preuve[p] = rang
    return [rang_par_preuve.get(p) for p in preuves]


async def mesurer_locomo(client: httpx.AsyncClient, top_k: int,
                         parallele: int) -> list[dict[str, Any]]:
    data = json.loads(LOCOMO.read_text(encoding="utf-8"))
    verrou = asyncio.Semaphore(parallele)
    par_conv = await asyncio.gather(*(_locomo_conv(client, c, top_k, verrou) for c in data))
    return [x for lignes in par_conv for x in lignes]


async def _locomo_conv(client: httpx.AsyncClient, c: dict[str, Any], top_k: int,
                       verrou: asyncio.Semaphore) -> list[dict[str, Any]]:
    lignes: list[dict[str, Any]] = []
    async with verrou:
        conv, sid = c["conversation"], c["sample_id"]
        user_id = f"locomo:{sid}"
        cle_vers_preuve: dict[str, str] = {}
        sessions = sorted((k for k in conv if k.startswith("session_") and not k.endswith("_date_time")),
                          key=lambda k: int(k.split("_")[1]))
        for s_key in sessions:
            base = parse_locomo_datetime(conv.get(f"{s_key}_date_time"))
            msgs = []
            for i, t in enumerate(conv[s_key]):
                ts = base + i * PAS_MS
                cle_vers_preuve[_iso(ts)] = t["dia_id"]
                # Contenu réaliste : locuteur et texte, sans l'identifiant de tour
                # que les autres benchs préfixent (il pèserait sur le lexical).
                msgs.append({"role": "user" if t.get("speaker") == conv["speaker_a"] else "assistant",
                             "content": f"{t.get('speaker')}: {t.get('text', '')}", "timestamp": ts})
            await _ecrire(client, user_id, s_key, msgs)
        for q in c["qa"]:
            preuves = _preuves_locomo(q)
            if q.get("category") == 5 or not preuves:
                continue
            rangs = await _rangs(client, user_id, q["question"], top_k, cle_vers_preuve, preuves)
            lignes.append({"jeu": "locomo", "moitie": "dev" if sid == LOCOMO_DEV else "val",
                           "conv": sid, "categorie": q["category"], "question": q["question"],
                           "preuves": preuves, "rangs": rangs})
        print(f"  LoCoMo {sid} : {len(lignes)} questions", flush=True)
    return lignes


async def mesurer_longmemeval(client: httpx.AsyncClient, top_k: int, types: set[str],
                              limite: int | None, parallele: int) -> list[dict[str, Any]]:
    data = json.loads(LONGMEMEVAL.read_text(encoding="utf-8"))
    choisies = [q for q in data if q["question_type"] in types]
    if limite:
        choisies = choisies[:limite]
    del data  # 277 Mo : inutile de garder les autres types en mémoire
    verrou = asyncio.Semaphore(parallele)
    faites = 0
    t0 = time.perf_counter()

    async def une(q: dict[str, Any]) -> dict[str, Any] | None:
        nonlocal faites
        async with verrou:
            ligne = await _lme_question(client, q, top_k)
        faites += 1
        if faites % 10 == 0:
            print(f"  LongMemEval {faites}/{len(choisies)} ({time.perf_counter() - t0:.0f} s)", flush=True)
        return ligne

    resultats = await asyncio.gather(*(une(q) for q in choisies))
    return [x for x in resultats if x is not None]


async def _lme_question(client: httpx.AsyncClient, q: dict[str, Any],
                        top_k: int) -> dict[str, Any] | None:
    user_id = f"lme:{q['question_id']}"
    cle_vers_preuve: dict[str, str] = {}
    preuves: list[str] = []
    sessions_preuve: dict[str, str] = {}
    for j, (sess_id, date, tours) in enumerate(zip(
            q["haystack_session_ids"], q["haystack_dates"], q["haystack_sessions"], strict=True)):
        # Deux sessions datées de la même minute donneraient les mêmes clés :
        # décalage de j millisecondes, sans effet sur l'ordre ni sur la date.
        base = _date_lme(date) + j
        msgs = []
        for i, t in enumerate(tours):
            ts = base + i * PAS_MS
            cle = f"{sess_id}#{i}"
            if _iso(ts) in cle_vers_preuve:
                raise RuntimeError(f"clé de correspondance en double : {q['question_id']} {cle}")
            cle_vers_preuve[_iso(ts)] = cle
            if t.get("has_answer"):
                preuves.append(cle)
                sessions_preuve[cle] = sess_id
            if t.get("content", "").strip():
                msgs.append({"role": t["role"], "content": t["content"], "timestamp": ts})
        await _ecrire(client, user_id, sess_id, msgs)
    if not preuves:
        return None
    rangs = await _rangs(client, user_id, q["question"], top_k, cle_vers_preuve, preuves)
    return {"jeu": "longmemeval", "moitie": _moitie_lme(q["question_id"]),
            "conv": q["question_id"], "categorie": q["question_type"],
            "question": q["question"], "preuves": preuves, "rangs": rangs,
            "sessions": [sessions_preuve[p] for p in preuves]}


def resumer(lignes: list[dict[str, Any]]) -> dict[str, Any]:
    """Rappel de TOUTES les preuves dans le top 10 / top 100, et part des preuves présentes."""
    if not lignes:
        return {"questions": 0}
    toutes_10 = sum(all(r is not None and r <= 10 for r in x["rangs"]) for x in lignes)
    toutes_100 = sum(all(r is not None for r in x["rangs"]) for x in lignes)
    une_10 = sum(any(r is not None and r <= 10 for r in x["rangs"]) for x in lignes)
    rangs = [r for x in lignes for r in x["rangs"]]
    pires = [max(r for r in x["rangs"]) for x in lignes if all(r is not None for r in x["rangs"])]
    multi = [x for x in lignes if len(x["preuves"]) > 1]
    return {
        "questions": len(lignes),
        "toutes_preuves_top10": round(toutes_10 / len(lignes), 3),
        "toutes_preuves_top100": round(toutes_100 / len(lignes), 3),
        "au_moins_une_top10": round(une_10 / len(lignes), 3),
        "preuves_presentes_top100": round(sum(r is not None for r in rangs) / len(rangs), 3),
        "pire_rang_median": statistics.median(pires) if pires else None,
        "multi_preuves": len(multi),
        "multi_toutes_top100": (round(sum(all(r is not None for r in x["rangs"]) for x in multi)
                                / len(multi), 3) if multi else None),
    }


async def run(args: argparse.Namespace) -> dict[str, Any]:
    tmp = Path(tempfile.mkdtemp(prefix="mnemos_temporel_"))
    lignes: list[dict[str, Any]] = []
    garder_la_boucle(asyncio.get_running_loop())
    try:
        app, _, _ = await setup_bench_app(tmp, "ollama", salience_workers=0)
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://t", timeout=600) as client:
                if "locomo" in args.jeux:
                    lignes += await mesurer_locomo(client, args.top_k, args.parallele)
                if "longmemeval" in args.jeux:
                    lignes += await mesurer_longmemeval(client, args.top_k, set(args.types),
                                                        args.limite, args.parallele)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    import mnemos
    groupes: dict[str, dict[str, Any]] = {}
    for jeu in sorted({x["jeu"] for x in lignes}):
        for cat in sorted({str(x["categorie"]) for x in lignes if x["jeu"] == jeu}):
            for moitie in ("dev", "val", "toutes"):
                sel = [x for x in lignes if x["jeu"] == jeu and str(x["categorie"]) == cat
                       and (moitie == "toutes" or x["moitie"] == moitie)]
                if sel:
                    groupes[f"{jeu} | {cat} | {moitie}"] = resumer(sel)
    result: dict[str, Any] = {
        "config": {"top_k": args.top_k, "jeux": args.jeux, "types": args.types,
                   "parallele": args.parallele,
                   "mnemos_src": str(Path(mnemos.__file__).parent),
                   "embed_backend": os.environ.get("EMBED_BACKEND", "ollama")},
        "groupes": groupes,
        "lignes": lignes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n{'groupe':<44} {'N':>4} {'toutes@10':>9} {'toutes@100':>10} {'une@10':>7}"
          f" {'présentes':>9} {'pire méd.':>9} {'multi':>5} {'multi@100':>9}")
    for nom, m in groupes.items():
        if not m.get("questions"):
            continue
        print(f"{nom:<44} {m['questions']:>4} {m['toutes_preuves_top10']:>9} "
              f"{m['toutes_preuves_top100']:>10} {m['au_moins_une_top10']:>7} "
              f"{m['preuves_presentes_top100']:>9} {str(m['pire_rang_median']):>9} "
              f"{m['multi_preuves']:>5} {str(m['multi_toutes_top100']):>9}")
    print(f"\nCode importé : {result['config']['mnemos_src']} | Rapport : {args.output}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Temporel : rappel de toutes les preuves par /search")
    parser.add_argument("--jeux", nargs="+", choices=("locomo", "longmemeval"),
                        default=["locomo", "longmemeval"])
    parser.add_argument("--types", nargs="+", default=["temporal-reasoning"],
                        help="types de questions LongMemEval mesurés")
    parser.add_argument("--limite", type=int, default=None, help="nombre max de questions LongMemEval")
    parser.add_argument("--top-k", type=int, default=100)
    # llama-server sert 16 requêtes à la fois ; une question à la fois en laissait 15 au repos.
    parser.add_argument("--parallele", type=int, default=6, help="questions ingérées en parallèle")
    parser.add_argument("--output", type=Path, default=Path("bench/results/gpu/temporal_recall.json"))
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # console Windows cp1252
    if sys.platform == "win32":
        # Voir bench_locomo_qa : la boucle Proactor reste figée après une coupure réseau.
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(run(args))


if __name__ == "__main__":
    main()

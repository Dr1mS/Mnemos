"""Demandes d'oubli : rattacher la consigne à ce qu'elle annule, sans étiquette.

`bench_oubli_injection.py` (06/10/2026) : remettre la consigne d'oubli dans le contexte
quand `/search` ne la rend pas fait passer les questions concernées de 1 à 10 bonnes
réponses sur 18. Ce test-là trouvait la consigne grâce à l'étiquette du jeu. Ici, le
mécanisme qu'on pourrait livrer, sans étiquette :

  * à l'écriture, chaque consigne (`detect_forget`) est LIÉE soit aux épisodes antérieurs
    les plus proches de sa cible (50 voisins, cosinus ≥ 0,65, 5 au plus, les réglages
    des échos `FORGET_ECHO_*`), soit au message qui la suit (`suivant`, l'accusé) ;
  * à la recherche, si un épisode lié figure dans les `fenetre` premiers résultats et
    que la consigne n'y est pas, elle rejoint la liste, `plafond` consignes au plus :
    en fin (similarité) ou juste avant l'épisode lié (`suivant`).

Premier essai, similarité seule (06/10) : la bonne consigne ne revient que pour 1 à 2
questions sur 18. La préférence n'est pas dite par l'utilisateur, alors que l'accusé est
déjà rendu pour les 18 ; et `bench_oubli_placement.py` montre que la consigne insérée
juste avant son accusé aide autant qu'en fin de liste. D'où la variante `suivant`.

`rattacher_consignes` est la fonction qu'on collerait dans Mnemos. Phase `selection` :
sans modèle de langage, sur toutes les questions, combien de consignes entrent, si la
bonne en fait partie (étiquette utilisée pour NOTER seulement), et si une preuve d'une
autre question est poussée hors du top 100. Phase `reponse` : la configuration choisie,
contexte A contre A+, consigne et lettre officielles, Nemotron 3 Ultra.

Embeddings par llama-server (`EMBED_BACKEND=llamacpp`), la pile de production. La base
ingérée peut être gardée (`--base`) : les deux phases relisent alors la même.
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
from collections.abc import Iterable
from math import comb
from pathlib import Path
from statistics import mean, median
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ.setdefault("EPISODIC_RETENTION_DAYS", "36500")
os.environ.setdefault("DECAY_RATE_DAILY", "0.0")

import httpx
import sqlite_vec  # type: ignore[import-untyped]
from sqlalchemy import select, text

from bench import aml_officiel
from bench.bench_carte_personamem import consigne, present
from bench.bench_forget_targeting import _mots, porte
from bench.bench_locomo import setup_bench_app
from bench.bench_personamem import (
    DATA_DIR,
    chunk_messages,
    load_chat,
    load_personas,
    parse_evidence,
    parse_query,
)
from bench.bench_temporal_recall import garder_la_boucle
from bench.remote_llm import RemoteChat
from mnemos.models.episodic import Episode
from mnemos.router.forget import detect_forget
from mnemos.stores.episodic import FORGET_ECHO_KNN, FORGET_ECHO_MAX, FORGET_ECHO_MIN_COSINE

TOP_K = 100
FENETRES = (10, 20, 50, 100)
PLAFONDS = (1, 2, 3, 5)
# Liens d'une consigne : par similarité à sa cible (`user`, `tous` : rôles retenus), ou
# vers le message qui la suit dans la conversation (`suivant`, en pratique l'accusé de
# l'assistant ; voir bench_oubli_placement.py).
LIENS = ("user", "tous", "suivant")
PLACEMENT = {"user": "fin", "tous": "fin", "suivant": "voisin"}


def rattacher_consignes(ids: list[str], liens: dict[str, list[str]], top_k: int,
                        fenetre: int, plafond: int, placement: str = "fin") -> list[str]:
    """Liste finale : les consignes d'oubli dont un épisode lié figure dans les `fenetre`
    premiers résultats la rejoignent, `plafond` au plus, dans l'ordre du premier épisode
    déclencheur. `fin` : en fin de liste, à la place des derniers ; `voisin` : chacune
    juste avant son épisode déclencheur, la liste coupée à `top_k`. Sans effet si rien
    n'est lié."""
    presents = set(ids)
    rang = {eid: i for i, eid in enumerate(ids[:fenetre])}
    candidates: list[tuple[int, str]] = []
    for c, vises in liens.items():
        if c in presents:
            continue
        r = min((rang[v] for v in vises if v in rang), default=None)
        if r is not None:
            candidates.append((r, c))
    choisies = sorted(candidates)[:plafond]
    if not choisies:
        return ids[:top_k]
    if placement == "fin":
        return ids[:min(len(ids), top_k - len(choisies))] + [c for _, c in choisies]
    avant: dict[int, list[str]] = {}
    for r, c in choisies:
        avant.setdefault(r, []).append(c)
    out: list[str] = []
    for i, eid in enumerate(ids):
        out.extend(avant.get(i, []))
        out.append(eid)
    return out[:top_k]


def _p(perdues: int, gagnees: int) -> float:
    n = perdues + gagnees
    return min(1.0, 2 * sum(comb(n, i) for i in range(min(perdues, gagnees) + 1)) / 2 ** n) if n else 1.0


def echantillon(lignes: dict[str, list[dict[str, Any]]], graine: int, n: int, frais: bool,
                exclure: tuple[int, ...] = ()) -> list[str]:
    """`frais` : personas hors de l'échantillon de la carte (graine 7, 70), jamais regardés.
    `exclure` : graines d'échantillons frais déjà tirés (même taille `n`), à écarter aussi."""
    dispo = sorted(p for p, rs in lignes.items()
                   if (DATA_DIR / "chats" / Path(rs[0]["chat_history_32k_link"]).name).exists())
    vus = set(random.Random(7).sample(dispo, min(70, len(dispo))))
    pool = [p for p in dispo if p not in vus] if frais else dispo
    for g in exclure:
        deja = set(random.Random(g).sample(pool, min(n, len(pool))))
        pool = [p for p in pool if p not in deja]
    return random.Random(graine).sample(pool, min(n, len(pool)))


async def liens_du_tenant(store: Any, tenant: str, conversation: list[str]) -> dict[str, dict[str, list[str]]]:
    """Pour chaque consigne stockée : ses épisodes liés, par variante de lien.

    Ce que ferait `/add`. Similarité : voisins de la cible antérieurs à la consigne,
    cosinus ≥ 0,65, 5 au plus, les autres consignes exclues. `suivant` : le message
    qui suit la consigne dans la conversation (`conversation` : contenus dans l'ordre)."""
    ordre: dict[str, int] = {}
    for i, c in enumerate(conversation):
        ordre.setdefault(c, i)
    async with store._sessions() as s:
        eps = (await s.execute(select(Episode.id, Episode.role, Episode.content).where(
            Episode.tenant == tenant, Episode.archived == 0))).all()
    info = {eid: (role, content) for eid, role, content in eps}
    par_contenu = {content: eid for eid, (_, content) in info.items()}
    consignes = {eid: d.target for eid, (role, content) in info.items()
                 if (d := detect_forget(content, role)) is not None}
    out: dict[str, dict[str, list[str]]] = {r: {} for r in LIENS}
    for cid, cible in consignes.items():
        pos = ordre.get(info[cid][1])
        suivant = par_contenu.get(conversation[pos + 1]) if pos is not None and pos + 1 < len(conversation) else None
        out["suivant"][f"ep_{cid}"] = [f"ep_{suivant}"] if suivant and suivant not in consignes else []
        emb = await store._embedder.embed(cible)
        async with store._sessions() as s:
            knn = (await s.execute(
                text("SELECT episode_id, distance FROM episodes_vec "
                     "WHERE embedding MATCH :emb AND k = :k AND tenant = :tenant"),
                {"emb": sqlite_vec.serialize_float32(emb), "k": FORGET_ECHO_KNN, "tenant": tenant})).all()
        pos_c = ordre.get(info[cid][1], -1)
        voisins = sorted(((1.0 - float(d), eid) for eid, d in knn
                          if eid in info and eid != cid and eid not in consignes
                          and ordre.get(info[eid][1], 10**9) < pos_c), reverse=True)
        for r in ("user", "tous"):
            garde = [eid for cos, eid in voisins
                     if cos >= FORGET_ECHO_MIN_COSINE and (r == "tous" or info[eid][0] == "user")]
            out[r][f"ep_{cid}"] = [f"ep_{e}" for e in garde[:FORGET_ECHO_MAX]]
    return out


def _resume(valeurs: Iterable[float]) -> dict[str, float]:
    v = list(valeurs)
    return {"moyenne": round(mean(v), 2), "mediane": median(v), "max": max(v)} if v else {}


async def run(args: argparse.Namespace) -> None:
    garder_la_boucle(asyncio.get_running_loop())
    lignes = load_personas(DATA_DIR / "val.csv")
    personas = echantillon(lignes, args.graine, args.personas, args.frais, tuple(args.exclure_graines))
    base = args.base or Path(tempfile.mkdtemp(prefix="mnemos_oubli_rat_"))
    base.mkdir(parents=True, exist_ok=True)
    marque = base / "ingeres.json"
    ingeres: set[str] = set(json.loads(marque.read_text(encoding="utf-8"))) if marque.exists() else set()
    officiel = aml_officiel.charger("personamem", "pipeline_v2.py") if args.phase == "reponse" else None
    repondeur = RemoteChat("nvidia") if args.phase == "reponse" else None
    opts = {"temperature": 0.0, "num_predict": 768}
    faits: dict[str, dict[str, Any]] = {}
    if args.phase == "reponse" and args.cache.exists():
        for x in args.cache.read_text(encoding="utf-8").splitlines():
            if x.strip():
                y = json.loads(x)
                faits[y["cle"]] = y
        print(f"Reprise : {len(faits)} questions déjà dans {args.cache}", flush=True)
    stats: list[dict[str, Any]] = []
    try:
        app, _, _ = await setup_bench_app(base, "ollama", salience_workers=0, embed_backend="llamacpp")
        async with app.router.lifespan_context(app):
            store = app.state.store
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://t", timeout=600) as client:
                for n, pid in enumerate(personas, 1):
                    rows = [r for r in lignes[pid] if r.get("correct_answer") and r.get("incorrect_answers")]
                    if not rows:
                        continue
                    tenant = f"pm:{pid}"
                    chat = load_chat(DATA_DIR / "chats" / Path(rows[0]["chat_history_32k_link"]).name)
                    if tenant not in ingeres:
                        for c, chunk in enumerate(chunk_messages(chat)):
                            (await client.post("/add", json={"request_id": f"{tenant}:{c}", "user_id": tenant,
                                                             "session_id": tenant, "messages": chunk})
                             ).raise_for_status()
                        ingeres.add(tenant)
                        marque.write_text(json.dumps(sorted(ingeres)), encoding="utf-8")
                    liens = await liens_du_tenant(store, tenant, [str(m["content"]) for m in chat])
                    for k, row in enumerate(rows):
                        cle = f"{pid}#{k}"
                        if args.phase == "reponse" and (cle in faits or row.get("pref_type") not in args.types):
                            continue
                        requete = parse_query(row["user_query"])
                        items = (await client.post("/search", json={
                            "query": requete, "user_id": tenant, "top_k": TOP_K})).json()["data"]
                        par_id = {it["id"]: it for it in items}
                        ids = [it["id"] for it in items]
                        oubli = row.get("pref_type") == "ask_to_forget"
                        bonnes: set[str] = set()
                        if oubli:  # étiquette : sert à noter, jamais à choisir
                            mots = _mots(row.get("prev_pref", ""))
                            bonnes = {c for c in liens["tous"]
                                      if porte(await _contenu(store, c), mots)}
                        preuves = parse_evidence(row.get("related_conversation_snippet", ""))
                        if args.phase == "selection":
                            st: dict[str, Any] = {"cle": cle, "type": row.get("pref_type"), "oubli": oubli,
                                                 "bonne_presente": bool(bonnes & set(ids)),
                                                 "bonne_existe": bool(bonnes),
                                                 "preuve_avant": present(items, preuves) if preuves else None}
                            for r in LIENS:
                                for f in FENETRES:
                                    for p in PLAFONDS:
                                        final = rattacher_consignes(ids, liens[r], TOP_K, f, p, PLACEMENT[r])
                                        ajouts = [i for i in final if i not in par_id]
                                        gardes = [par_id[i] for i in final if i in par_id]
                                        st[f"{r}/{f}/{p}"] = {
                                            "ajouts": len(ajouts),
                                            "bonne": bool(bonnes & set(final)),
                                            "preuve": present(gardes, preuves) if preuves else None}
                            stats.append(st)
                            continue
                        # phase « reponse » : la configuration choisie, A contre A+
                        assert officiel is not None and repondeur is not None
                        final = rattacher_consignes(ids, liens[args.liens], TOP_K, args.fenetre, args.plafond,
                                                    PLACEMENT[args.liens])
                        options = [str(row["correct_answer"]), *(str(o) for o in json.loads(row["incorrect_answers"]))]
                        random.Random(f"{pid}:{k}:{args.graine}").shuffle(options)
                        bonne = chr(65 + options.index(str(row["correct_answer"])))
                        lignes_a = [f"[{it.get('created_at', '')}] {it['content']}" for it in items]
                        lignes_plus = [f"[{it.get('created_at', '')}] {it['content']}" for it in
                                       [par_id.get(i) or await _item(store, i) for i in final]]
                        change = lignes_plus != lignes_a
                        reps: dict[str, str | None] = {}
                        try:
                            # contexte inchangé : la réponse ne peut bouger que par le bruit, rien à poser
                            essais = [("A", lignes_a), ("A+", lignes_plus)] if change else []
                            if change and random.Random(cle).random() < args.bis:
                                essais.append(("A_bis", lignes_a))  # plancher de bruit, même contexte
                            for nom, lig in essais:
                                ctx = "Memories from the conversation history:\n" + "\n".join(lig)
                                sortie = await repondeur.generate(consigne(officiel, ctx, requete, options),
                                                                  "nvidia/nemotron-3-ultra-550b-a55b", options=opts)
                                reps[nom] = officiel.extract_final_letter(sortie)
                        except Exception as exc:  # noqa: BLE001 — Nvidia 429/503
                            print(f"  sautée, à refaire : {cle} ({str(exc)[:100]})", flush=True)
                            continue
                        y = {"cle": cle, "type": row.get("pref_type"), "change": change,
                             "ajouts": sum(i not in par_id for i in final),
                             "bonne_ajoutee": bool(bonnes & (set(final) - set(ids))), "bonne": bonne,
                             **{f"lettre_{k2}": v for k2, v in reps.items()},
                             **{k2: v == bonne for k2, v in reps.items()}}
                        faits[cle] = y
                        with args.cache.open("a", encoding="utf-8") as fh:
                            fh.write(json.dumps(y, ensure_ascii=False) + "\n")
                    print(f"  [{n}/{len(personas)}] persona {pid}", flush=True)
    finally:
        if args.base is None:
            shutil.rmtree(base, ignore_errors=True)
        if repondeur is not None:
            await repondeur.aclose()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.phase == "selection":
        rapport = synthese_selection(stats)
        rapport["details"] = stats
    else:
        rapport = synthese_reponse(list(faits.values()))
    # Synthèse lisible, détails une question par ligne (le rapport de sélection dépasserait 1 Mo)
    details = rapport.pop("details")
    tete = json.dumps(rapport, indent=2, ensure_ascii=False)
    lignes_json = ",\n    ".join(json.dumps(x, ensure_ascii=False, separators=(",", ":")) for x in details)
    args.output.write_text(f'{tete[:-2]},\n  "details": [\n    {lignes_json}\n  ]\n}}\n', encoding="utf-8")
    print(f"Rapport : {args.output}")


async def _contenu(store: Any, item_id: str) -> str:
    async with store._sessions() as s:
        row = (await s.execute(select(Episode.content).where(Episode.id == item_id.removeprefix("ep_")))).first()
    return str(row[0]) if row else ""


async def _item(store: Any, item_id: str) -> dict[str, Any]:
    """Une consigne ajoutée, au format d'un résultat `/search`."""
    from mnemos.api.aml_routes import _ts_to_iso
    async with store._sessions() as s:
        row = (await s.execute(select(Episode.content, Episode.created_at).where(
            Episode.id == item_id.removeprefix("ep_")))).first()
    assert row is not None
    return {"id": item_id, "content": str(row[0]), "created_at": _ts_to_iso(row[1])}


def synthese_selection(stats: list[dict[str, Any]]) -> dict[str, Any]:
    oubli = [x for x in stats if x["oubli"] and x["bonne_existe"]]
    manque = [x for x in oubli if not x["bonne_presente"]]
    autres = [x for x in stats if not x["oubli"]]
    avec_preuve = [x for x in stats if x["preuve_avant"]]
    print(f"\n{len(stats)} questions ; oubli avec consigne : {len(oubli)}, dont consigne absente du top 100 :"
          f" {len(manque)} ; preuve au top 100 avant : {len(avec_preuve)}")
    print(f"{'config':>14s} {'bonne/manque':>13s} {'ajouts oubli':>13s} {'ajouts autres':>22s}"
          f" {'autres touchées':>16s} {'preuves perdues':>16s}")
    configs: dict[str, Any] = {}
    for r in LIENS:
        for f in FENETRES:
            for p in PLAFONDS:
                c = f"{r}/{f}/{p}"
                res = {
                    "bonne_recuperee": sum(x[c]["bonne"] for x in manque),
                    "ajouts_oubli": _resume(x[c]["ajouts"] for x in oubli),
                    "ajouts_autres": _resume(x[c]["ajouts"] for x in autres),
                    "autres_touchees": sum(x[c]["ajouts"] > 0 for x in autres),
                    "preuves_perdues": sum(1 for x in avec_preuve if not x[c]["preuve"]),
                }
                configs[c] = res
                print(f"{c:>14s} {res['bonne_recuperee']:>6d}/{len(manque):<6d} "
                      f"{res['ajouts_oubli'].get('moyenne', 0):>13} "
                      f"{str(res['ajouts_autres']):>22s} {res['autres_touchees']:>8d}/{len(autres):<7d}"
                      f" {res['preuves_perdues']:>16d}")
    return {"questions": len(stats), "oubli_avec_consigne": len(oubli), "consigne_absente": len(manque),
            "autres": len(autres), "preuve_avant": len(avec_preuve), "configs": configs}


def synthese_reponse(tous: list[dict[str, Any]]) -> dict[str, Any]:
    faits = [x for x in tous if x["change"]]

    def bilan(xs: list[dict[str, Any]]) -> dict[str, Any]:
        g = sum(x["A+"] and not x["A"] for x in xs)
        p = sum(x["A"] and not x["A+"] for x in xs)
        return {"n": len(xs), "A": sum(x["A"] for x in xs), "A+": sum(x["A+"] for x in xs),
                "gagnees": g, "perdues": p, "p": round(_p(p, g), 4)}
    oubli = bilan([x for x in faits if x["type"] == "ask_to_forget"])
    autres = bilan([x for x in faits if x["type"] != "ask_to_forget"])
    inchangees = {"oubli": sum(not x["change"] for x in tous if x["type"] == "ask_to_forget"),
                  "autres": sum(not x["change"] for x in tous if x["type"] != "ask_to_forget")}
    bonne_ajoutee = sum(x["bonne_ajoutee"] for x in faits)
    bis = [x for x in faits if "A_bis" in x]
    bruit = {"n": len(bis), "verdict_change": sum(x["A"] != x["A_bis"] for x in bis)}
    print(f"\nquestions au contexte inchangé (non posées) : {inchangees}"
          f"\noubli, contexte changé : {oubli} (bonne consigne ajoutée : {bonne_ajoutee})"
          f"\nautres, contexte changé : {autres}\nbruit (A redemandé, même contexte) : {bruit}")
    return {"oubli": oubli, "autres": autres, "inchangees": inchangees, "bonne_ajoutee": bonne_ajoutee,
            "bruit": bruit, "details": tous}


def main() -> None:
    parser = argparse.ArgumentParser(description="Demandes d'oubli : rattacher la consigne, sans étiquette")
    parser.add_argument("--phase", choices=("selection", "reponse"), default="selection")
    parser.add_argument("--personas", type=int, default=70)
    parser.add_argument("--graine", type=int, default=7)
    parser.add_argument("--frais", action="store_true", help="personas hors de la carte (validation)")
    parser.add_argument("--exclure-graines", type=int, nargs="*", default=[],
                        help="graines d'échantillons frais déjà utilisés, à écarter")
    parser.add_argument("--base", type=Path, default=None, help="base ingérée à garder et relire")
    parser.add_argument("--liens", choices=LIENS, default="suivant")
    parser.add_argument("--fenetre", type=int, default=100)
    parser.add_argument("--plafond", type=int, default=3)
    parser.add_argument("--types", nargs="+", default=None,
                        help="phase reponse : types de questions à poser (défaut : tous)")
    parser.add_argument("--bis", type=float, default=0.3,
                        help="part des questions au contexte changé où A est redemandé (bruit)")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--cache", type=Path, default=None)
    args = parser.parse_args()
    nom = f"rattachement_{args.phase}_{'frais' if args.frais else 'carte'}"
    args.output = args.output or Path(f"bench/results/oubli/{nom}.json")
    args.cache = args.cache or Path(f"bench/results/oubli/{nom}.cache.jsonl")
    if args.types is None:
        lignes = load_personas(DATA_DIR / "val.csv")
        args.types = sorted({str(r.get("pref_type")) for rs in lignes.values() for r in rs})
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(run(args))


if __name__ == "__main__":
    main()

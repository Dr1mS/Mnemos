"""Ciblage de la suppression (axe 3) : quoi effacer quand l'utilisateur demande d'oublier.

`mnemos.router.forget` dit SI un message est une consigne d'oubli et CE QU'IL
VISE. Reste à décider quels épisodes disparaissent. Relevé sur PersonaMem-v2
(29/09/2026) : le contenu à oublier survit surtout APRÈS la consigne — la
consigne elle-même le répète mot pour mot, et l'accusé de réception de
l'assistant le répète dans 77 cas sur 119. Avant la consigne, des réponses de
l'assistant l'écho (« Since you enjoy Middle Eastern desserts like baklava… »).
Les échos lointains, 90 tours plus tard, étaient à la relecture des faux
positifs de l'heuristique : on ne les chasse pas.

Ce harnais ingère une fois, sans LLM, les personas qui portent des questions
`ask_to_forget`, puis évalue chaque règle de ciblage en rejouant la recherche
**telle que la production la verrait après suppression** : épisodes supprimés
retirés du KNN, 200 plus proches restants, score déployé (0,7 dense + 0,3
recouvrement), top 100. Deux grandeurs, toujours ensemble :

* **fuite** — questions `ask_to_forget` : la préférence oubliée (`prev_pref`)
  remonte-t-elle encore ? Un épisode la « porte » s'il en contient au moins la
  moitié des mots pleins — mesure grossière, mais identique pour toutes les
  règles, donc les écarts entre règles restent lisibles ;
* **dégâts** — toutes les AUTRES questions des mêmes personas : a-t-on effacé
  des messages-preuves dont elles ont besoin ? Une règle qui réduit la fuite en
  détruisant des preuves ne vaut rien.

Découpage par persona (pair → calibration, impair → validation) : la validation
n'est lue qu'avec `--validation`, une fois le seuil choisi.

    EMBED_BACKEND=llamacpp python bench/bench_forget_targeting.py --personas 100
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import pickle
import re
import shutil
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ.setdefault("EPISODIC_RETENTION_DAYS", "36500")
os.environ.setdefault("DECAY_RATE_DAILY", "0.0")

import httpx
import sqlite_vec  # type: ignore[import-untyped]
from sqlalchemy import select, text

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
from mnemos.embeddings.sparse import query_coverage, sparse_encode
from mnemos.models.episodic import Episode, EpisodeSparse
from mnemos.router.forget import detect_forget

TOP_K = 100
KNN_PROD = 200       # candidats vus par la production à top_k=100
KNN_LARGE = 400      # marge pour retirer les supprimés avant de garder les 200 plus proches
KNN_CIBLES = 50      # voisins examinés pour chaque consigne
MAX_ANTERIEURS = 5   # plafond de cibles antérieures par consigne
BASE_MS = 1_700_000_000_000

_MOTS_VIDES = set(re.findall(
    r"\w+",
    "i my me the a an and or of to in for with that about have has am is are was be it this"
    " on at from as by your you please forget preference detail fact memory their they them"
    " more most very really just also into over when while",
))


def _mots(t: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]+", t.lower()) if len(w) > 3 and w not in _MOTS_VIDES}


def porte(contenu: str, mots_pref: set[str]) -> bool:
    return bool(mots_pref) and len(mots_pref & _mots(contenu)) / len(mots_pref) >= 0.5


@dataclass
class Consigne:
    idx: int
    cible: str
    accuse: int | None
    anterieurs: list[tuple[int, float]]  # (idx, cosinus), idx < consigne, cos décroissant


@dataclass
class Persona:
    pid: str
    messages: list[dict[str, str]]
    idx_par_id: dict[str, int] = field(default_factory=dict)
    consignes: list[Consigne] = field(default_factory=list)
    # question -> (ligne CSV, candidats [(idx, distance, recouvrement)] triés par distance)
    questions: list[tuple[dict[str, str], list[tuple[int, float, float]]]] = field(default_factory=list)


async def _knn(store: Any, emb: list[float], tenant: str, k: int) -> list[tuple[str, float]]:
    async with store._sessions() as s:
        rows = await s.execute(
            text("SELECT episode_id, distance FROM episodes_vec "
                 "WHERE embedding MATCH :emb AND k = :k AND tenant = :tenant"),
            {"emb": sqlite_vec.serialize_float32(emb), "k": k, "tenant": tenant},
        )
        return [(r[0], float(r[1])) for r in rows]


async def collecter(n_personas: int, cache: Path) -> list[Persona]:
    """Ingère et analyse persona par persona, en écrivant le cache après chacune.

    Un run a été tué le 29/09 pour manque de mémoire sur la machine, à 13
    personas sur 100, sans rien laisser : le cache n'était écrit qu'à la fin.
    Chaque persona vit dans son propre tenant, donc les traiter une à une est
    équivalent, et un nouveau lancement reprend là où le précédent s'est arrêté."""
    rows = load_personas(DATA_DIR / "val.csv")
    choisies = [pid for pid, rs in rows.items()
                if any(r["pref_type"] == "ask_to_forget" for r in rs)
                and (DATA_DIR / "chats" / Path(rs[0]["chat_history_32k_link"]).name).exists()]
    choisies = choisies[:n_personas]
    faites: list[Persona] = pickle.loads(cache.read_bytes()) if cache.exists() else []
    deja = {p.pid for p in faites}
    restantes = [pid for pid in choisies if pid not in deja]
    if faites:
        print(f"Reprise : {len(faites)} personas déjà en cache, {len(restantes)} à collecter", flush=True)
    if not restantes:
        return faites

    tmp = Path(tempfile.mkdtemp(prefix="mnemos_forget_"))
    try:
        app, _, _ = await setup_bench_app(tmp, "ollama", salience_workers=0, embed_backend="llamacpp")
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://t", timeout=600) as c:
                for n, pid in enumerate(restantes, 1):
                    p = Persona(pid, load_chat(DATA_DIR / "chats" / Path(rows[pid][0]["chat_history_32k_link"]).name))
                    # Horodatages explicites et croissants : l'ordre des messages
                    # se relit ensuite par created_at, sans ambiguïté.
                    msgs = [{**m, "timestamp": BASE_MS + i * 1000} for i, m in enumerate(p.messages)]
                    for k, chunk in enumerate(chunk_messages(msgs)):
                        (await c.post("/add", json={
                            "request_id": f"pm:{p.pid}:{k}", "user_id": f"pm:{p.pid}",
                            "session_id": f"pm:{p.pid}", "messages": chunk})).raise_for_status()
                    await _analyser(app.state.store, rows, p)
                    faites.append(p)
                    cache.write_bytes(pickle.dumps(faites))
                    if n % 10 == 0 or n == len(restantes):
                        print(f"  {len(faites)}/{len(choisies)} personas collectées", flush=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return faites


async def _analyser(store: Any, rows: dict[str, list[dict[str, str]]], p: Persona) -> None:
    """Consignes détectées, voisins de leur cible, et candidats de chaque question."""
    now = store._clock.now_ms()
    tenant = f"pm:{p.pid}"
    async with store._sessions() as s:
        eps = (await s.execute(select(Episode.id, Episode.created_at)
                               .where(Episode.tenant == tenant))).all()
    p.idx_par_id = {eid: (ts - BASE_MS) // 1000 for eid, ts in eps}

    for i, m in enumerate(p.messages):
        d = detect_forget(m["content"], m["role"])
        if d is None:
            continue
        emb = await store._embedder.embed(d.target)
        voisins = await _knn(store, emb, tenant, KNN_CIBLES)
        anterieurs = sorted(
            ((p.idx_par_id[e], 1.0 - dist) for e, dist in voisins
             if e in p.idx_par_id and p.idx_par_id[e] < i),
            key=lambda x: -x[1])
        accuse = i + 1 if i + 1 < len(p.messages) and p.messages[i + 1]["role"] == "assistant" else None
        p.consignes.append(Consigne(i, d.target, accuse, anterieurs))

    for row in rows[p.pid]:
        q = parse_query(row["user_query"])
        emb = await store._embedder.embed(q)
        voisins = await _knn(store, emb, tenant, KNN_LARGE)
        q_sparse = sparse_encode(q, now)
        async with store._sessions() as s:
            bits = dict((await s.execute(
                select(EpisodeSparse.episode_id, EpisodeSparse.sparse_bits)
                .where(EpisodeSparse.episode_id.in_([e for e, _ in voisins])))).all())
        cands = [(p.idx_par_id[e], dist, query_coverage(q_sparse, bits[e]))
                 for e, dist in voisins if e in p.idx_par_id and e in bits]
        p.questions.append((row, sorted(cands, key=lambda x: x[1])))


def supprimes(p: Persona, regle: dict[str, Any]) -> set[int]:
    out: set[int] = set()
    for c in p.consignes:
        if regle["consigne"]:
            out.add(c.idx)
        if regle["accuse"] and c.accuse is not None:
            out.add(c.accuse)
        seuil = regle.get("seuil")
        if seuil is not None:
            # Relecture du 29/09 : au seuil 0,65, les 13 messages ASSISTANT
            # supprimés étaient des échos (« Since you enjoy visiting
            # aquariums… ») ; les 7 messages UTILISATEUR, des questions sur le
            # même sujet (« Can you suggest some good books? »). Filtrer le rôle
            # AVANT le plafond, pour que les questions n'occupent pas les places.
            roles = regle.get("roles")
            eligibles = [(i, cos) for i, cos in c.anterieurs
                         if roles is None or p.messages[i]["role"] in roles]
            out.update(i for i, cos in eligibles[:MAX_ANTERIEURS] if cos >= seuil)
    return out


def classer(cands: list[tuple[int, float, float]], suppr: set[int]) -> list[int]:
    """La recherche de production après suppression : les supprimés ont quitté
    le KNN, on garde les 200 plus proches restants, score déployé, top 100."""
    restants = [c for c in cands if c[0] not in suppr][:KNN_PROD]
    restants.sort(key=lambda c: 0.7 * (1.0 - c[1]) + 0.3 * c[2], reverse=True)
    return [c[0] for c in restants[:TOP_K]]


def evaluer(personas: list[Persona], regle: dict[str, Any]) -> dict[str, Any]:
    fuite10 = fuite100 = n_forget = 0
    porteurs10: list[int] = []
    n_autres = preuve_touchee = preuve_detruite = hit10 = 0
    volume: list[float] = []
    for p in personas:
        suppr = supprimes(p, regle)
        volume.append(len(suppr) / max(1, len(p.messages)))
        norm = [_normalize(m["content"]) for m in p.messages]
        for row, cands in p.questions:
            classe = classer(cands, suppr)
            if row["pref_type"] == "ask_to_forget":
                n_forget += 1
                mp = _mots(row["prev_pref"])
                marques = [porte(p.messages[i]["content"], mp) for i in classe]
                fuite10 += any(marques[:10])
                fuite100 += any(marques)
                porteurs10.append(sum(marques[:10]))
            else:
                ev = parse_evidence(row.get("related_conversation_snippet", ""))
                idx_ev = {i for i, t in enumerate(norm) if t in ev}
                if not idx_ev:
                    continue
                n_autres += 1
                preuve_touchee += bool(idx_ev & suppr)
                preuve_detruite += idx_ev <= suppr
                hit10 += any(i in idx_ev for i in classe[:10])
    return {
        "fuite@10": round(fuite10 / n_forget, 3) if n_forget else None,
        "fuite@100": round(fuite100 / n_forget, 3) if n_forget else None,
        "porteurs_top10_moyen": round(sum(porteurs10) / len(porteurs10), 2) if porteurs10 else None,
        "autres_questions": n_autres,
        "preuve_touchee": round(preuve_touchee / n_autres, 3) if n_autres else None,
        "preuve_detruite": round(preuve_detruite / n_autres, 3) if n_autres else None,
        "autres_hit@10": round(hit10 / n_autres, 3) if n_autres else None,
        "part_historique_supprimee": round(sum(volume) / len(volume), 3) if volume else None,
        "n_ask_to_forget": n_forget,
    }


REGLES: dict[str, dict[str, Any]] = {
    "rien (état actuel)": {"consigne": False, "accuse": False},
    "consigne seule": {"consigne": True, "accuse": False},
    "consigne + accusé": {"consigne": True, "accuse": True},
    **{f"+ antérieurs cos≥{s:.2f}": {"consigne": True, "accuse": True, "seuil": s}
       for s in (0.85, 0.80, 0.75, 0.70, 0.65, 0.60)},
    **{f"+ échos assistant cos≥{s:.2f}": {"consigne": True, "accuse": True, "seuil": s,
                                          "roles": {"assistant"}}
       for s in (0.70, 0.65, 0.60, 0.55)},
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Ciblage de la suppression : fuite contre dégâts")
    parser.add_argument("--personas", type=int, default=100)
    parser.add_argument("--validation", action="store_true",
                        help="évaluer aussi la moitié de validation — seulement la règle choisie")
    parser.add_argument("--output", type=Path, default=Path("bench/results/gpu/forget_targeting.json"))
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    # L'ingestion coûte ~45 min pour 100 personas (réponses longues de
    # l'assistant). Les règles, elles, s'évaluent hors ligne en secondes : on
    # met en cache ce qui a été collecté, hors du dépôt.
    # Le cache est écrit après chaque persona : un run interrompu reprend.
    cache = Path(tempfile.gettempdir()) / f"mnemos_forget_targeting_{args.personas}.pkl"
    personas = asyncio.run(collecter(args.personas, cache))
    print(f"Cache : {cache}")
    moities = {"calibration": [p for p in personas if int(p.pid) % 2 == 0]}
    if args.validation:
        moities["validation"] = [p for p in personas if int(p.pid) % 2 == 1]

    resultats: dict[str, Any] = {}
    for nom_moitie, groupe in moities.items():
        n_cons = sum(len(p.consignes) for p in groupe)
        print(f"\n== {nom_moitie} : {len(groupe)} personas, {n_cons} consignes détectées ==")
        print(f"{'règle':30s} | {'fuite@10':>8s} {'fuite@100':>9s} {'porteurs/10':>11s}"
              f" | {'preuve touchée':>14s} {'détruite':>8s} {'autres hit@10*':>14s} | {'supprimé':>8s}")
        resultats[nom_moitie] = {}
        for nom, regle in REGLES.items():
            r = evaluer(groupe, regle)
            resultats[nom_moitie][nom] = r
            print(f"{nom:30s} | {r['fuite@10']!s:>8s} {r['fuite@100']!s:>9s} {r['porteurs_top10_moyen']!s:>11s}"
                  f" | {r['preuve_touchee']!s:>14s} {r['preuve_detruite']!s:>8s} {r['autres_hit@10']!s:>13s}"
                  f" | {r['part_historique_supprimee']!s:>8s}")
    print("\n* monte mécaniquement quand on supprime (les survivants remontent) : ce n'est PAS"
          "\n  une mesure de dégâts. Les dégâts se lisent dans « preuve touchée » et « détruite »."
          "\n  La fuite repose sur un recouvrement de mots-clés : comparer les règles entre elles,"
          "\n  pas citer la valeur absolue.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(resultats, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nRapport : {args.output}")


if __name__ == "__main__":
    main()

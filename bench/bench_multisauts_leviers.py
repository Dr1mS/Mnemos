"""Multi-sauts : quels leviers de SÉLECTION font entrer les preuves dans le top 100 ?

Diagnostic du 05/10/2026 (`bench_multisauts_diag.py`) : 237 preuves multi-sauts
sur 882 manquent au top 100 ; la moitié est juste derrière (rangs 101-200), 92 %
sont prononcées par la personne que la question nomme, 47 sont voisines d'un des
10 premiers souvenirs. Trois leviers, tous côté sélection (le texte rendu reste
celui du souvenir) :

- **vivier** : la recherche ne reclasse que les 2 × k plus proches voisins par
  embedding (200 à top_k = 100) ; un vivier plus large laisse le lexical repêcher ;
- **noms propres** : un bonus aux souvenirs qui contiennent un nom propre de la
  question (« What is Joanna allergic to? ») ;
- **voisinage** : les répliques adjacentes des meilleurs résultats, insérées juste
  après eux ;
- **diversité** (MMR) : ne pas laisser dix variantes d'un même souvenir prendre
  dix places ; ressemblance mesurée sur les bits lexicaux (Jaccard), faute de
  numpy pour comparer les embeddings à cette échelle.

Ingestion unique, sans modèle de langage ; mêmes candidats pour toutes les
variantes. Contrôle de fidélité : la variante « déployée » doit reproduire la
route `/search` question par question, sinon rien ne vaut.

Découpage fixé avant la mesure : LoCoMo conv-26/30/41/42/43 pour régler,
conv-44/47/48/49/50 pour valider sans retouche. Non-régression : catégories 2-4
de LoCoMo, D1 dur, PersonaMem (50 premières personas, comme le 29/09).
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
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ.setdefault("EPISODIC_RETENTION_DAYS", "36500")
os.environ.setdefault("DECAY_RATE_DAILY", "0.0")

import httpx
import sqlite_vec  # type: ignore[import-untyped]
from sqlalchemy import select, text

from bench.bench_locomo import parse_locomo_datetime, setup_bench_app
from bench.bench_personamem import DATA_DIR as PM_DIR
from bench.bench_personamem import (
    _normalize,
    chunk_messages,
    load_chat,
    load_personas,
    parse_evidence,
    parse_query,
)
from bench.bench_temporal_recall import PAS_MS, _iso, _preuves_locomo, garder_la_boucle
from bench.bench_update_hard import build_messages
from bench.datasets import HARD_UPDATE_INSTANCES
from bench.eval_utils import normalize_text
from mnemos.embeddings.sparse import query_coverage, sparse_encode
from mnemos.models.episodic import Episode, EpisodeSparse
from mnemos.stores import episodic as _ep

TOP_K = 100
VIVIER_MAX = 600
REGLAGE = {"conv-26", "conv-30", "conv-41", "conv-42", "conv-43"}
PM_PERSONAS = 50


@dataclass
class Cand:
    id: str
    contenu: str
    cree: int
    session: str | None
    dense: float
    lexical: float
    rang_dense: int
    bits: int  # bits lexicaux de contenu, pour la diversité


def score_deploye(c: Cand, now: int) -> float:
    """Le score de `EpisodicStore.search`, mêmes constantes (poids normalisés,
    récence murale, inerte sur des données de 2023)."""
    recence = 2.0 ** (-max(0.0, (now - c.cree) / _ep.DAY_MS) / _ep.RECENCY_HALF_LIFE_DAYS)
    return float(_ep.DENSE_WEIGHT * c.dense + _ep.SPARSE_WEIGHT * c.lexical
                 + _ep.RECENCY_WEIGHT * recence)


_NOM = re.compile(r"\b[A-Z][a-zA-Z'’-]+\b")
_PAS_NOMS = {"I", "What", "When", "Where", "Who", "Why", "How", "Which", "Did", "Does", "Do",
             "Is", "Are", "Was", "Were", "Has", "Have", "Had", "Can", "Could", "Would", "Should",
             "Will", "The", "A", "An", "In", "On", "At", "For", "Of", "And", "Or", "If", "My"}


def noms_propres(question: str) -> set[str]:
    return {m.lower() for m in _NOM.findall(question) if m not in _PAS_NOMS}


@dataclass
class Requete:
    jeu: str
    groupe: str  # catégorie LoCoMo, « d1 », « pm »
    moitie: str
    tenant: str
    question: str
    cands: list[Cand]
    voisins: dict[str, list[str]]  # id -> ids adjacents (±1, ±2) dans la session
    contenus: dict[str, str]  # id -> contenu, pour les voisins hors vivier


def dedup(ids: list[str], contenus: dict[str, str]) -> list[str]:
    """Comme la route : un contenu identique (casse ignorée) n'apparaît qu'une fois."""
    vus: set[str] = set()
    out = []
    for i in ids:
        c = contenus[i].lower()
        if c.strip() and c not in vus:
            vus.add(c)
            out.append(i)
    return out


def _jaccard(a: int, b: int) -> float:
    return (a & b).bit_count() / max(1, (a | b).bit_count())


def classer(r: Requete, now: int, vivier: int, bonus_nom: float = 0.0, voisins_top: int = 0,
            mmr: float | None = None, voisins_fin: int = 0) -> list[str]:
    cands = [c for c in r.cands if c.rang_dense <= vivier]
    noms = noms_propres(r.question) if bonus_nom else set()

    def s(c: Cand) -> float:
        base = score_deploye(c, now)
        if noms and any(re.search(rf"\b{re.escape(n)}\b", c.contenu.lower()) for n in noms):
            base += bonus_nom
        return base

    ordre = sorted(cands, key=s, reverse=True)
    ids = [c.id for c in ordre]
    if mmr is not None:
        rel = {c.id: s(c) for c in ordre}
        bits = {c.id: c.bits for c in ordre}
        restants: list[str] = ids[: 2 * TOP_K]
        proche = dict.fromkeys(restants, 0.0)  # ressemblance max à un déjà choisi
        choisis: list[str] = []
        while restants and len(choisis) < TOP_K:
            meilleur = max(restants, key=lambda i: mmr * rel[i] - (1 - mmr) * proche[i])
            choisis.append(meilleur)
            restants.remove(meilleur)
            for i in restants:
                proche[i] = max(proche[i], _jaccard(bits[i], bits[meilleur]))
        ids = choisis + [i for i in ids if i not in choisis]
    ids = dedup(ids, r.contenus)
    if voisins_top:
        etendu: list[str] = []
        for rang, i in enumerate(ids):
            if i not in etendu:
                etendu.append(i)
            if rang < voisins_top:
                for v in r.voisins.get(i, []):
                    if v not in etendu:
                        etendu.append(v)
        ids = dedup(etendu, r.contenus)
    if voisins_fin:
        # Voisins des `voisins_fin` premiers, placés EN FIN de liste à la place des
        # derniers résultats : le haut du classement ne bouge pas.
        tete = ids[:TOP_K]
        ajouts: list[str] = []
        for i in tete[:voisins_fin]:
            for v in r.voisins.get(i, []):
                if v not in tete and v not in ajouts:
                    ajouts.append(v)
        ajouts = dedup(ajouts, r.contenus)[: TOP_K // 4]
        ids = dedup(tete[: TOP_K - len(ajouts)] + ajouts, r.contenus)
    return ids[:TOP_K]


async def candidats(store: Any, question: str, tenant: str) -> list[Cand]:
    """L'étape KNN de `EpisodicStore.search`, élargie à VIVIER_MAX ; chaque candidat
    garde son rang dense pour simuler n'importe quel vivier plus petit."""
    now = store._clock.now_ms()
    dense = await store._embedder.embed(question)
    q_bits = sparse_encode(question, now)
    async with store._sessions() as session:
        knn = (await session.execute(
            text("SELECT episode_id, distance FROM episodes_vec "
                 "WHERE embedding MATCH :emb AND k = :k AND tenant = :tenant"),
            {"emb": sqlite_vec.serialize_float32(dense), "k": VIVIER_MAX, "tenant": tenant})).all()
        rang = {eid: r for r, (eid, _) in enumerate(knn, 1)}
        dist = {eid: float(d) for eid, d in knn}
        rows = (await session.execute(
            select(Episode, EpisodeSparse.sparse_bits)
            .join(EpisodeSparse, EpisodeSparse.episode_id == Episode.id)
            .where(Episode.id.in_(list(dist))))).tuples().all()
    masque = (1 << 224) - 1  # bits de contenu seulement
    return [Cand(e.id, e.content, e.created_at, e.session_id, 1.0 - dist[e.id],
                 query_coverage(q_bits, bits), rang[e.id], int.from_bytes(bits, "little") & masque)
            for e, bits in rows if e.tenant == tenant and not e.archived]


async def episodes_du_tenant(store: Any, tenant: str) -> tuple[dict[str, list[str]], dict[str, str]]:
    async with store._sessions() as session:
        rows = (await session.execute(text(
            "SELECT id, session_id, created_at, content FROM episodes "
            "WHERE tenant = :t ORDER BY session_id, created_at"), {"t": tenant})).all()
    contenus = {r[0]: r[3] for r in rows}
    par_session: dict[str | None, list[str]] = {}
    for r in rows:
        par_session.setdefault(r[1], []).append(r[0])
    voisins: dict[str, list[str]] = {}
    for ids in par_session.values():
        for k, i in enumerate(ids):
            voisins[i] = [ids[j] for j in (k - 1, k + 1, k - 2, k + 2) if 0 <= j < len(ids)]
    return voisins, contenus


def juge_locomo(preuves_ts: list[str], ts_par_id: dict[str, str]) -> Callable[[list[str]], dict[str, Any]]:
    def juger(ids: list[str]) -> dict[str, Any]:
        rangs_par_ts = {ts_par_id[i]: r for r, i in enumerate(ids, 1) if i in ts_par_id}
        rangs = [rangs_par_ts.get(t) for t in preuves_ts]
        return {"toutes": all(r is not None for r in rangs),
                "presentes": sum(r is not None for r in rangs), "n": len(rangs),
                "hit10": any(r is not None and r <= 10 for r in rangs)}
    return juger


VARIANTES: dict[str, dict[str, Any]] = {
    "déployée": {"vivier": 200},
    "vivier 400": {"vivier": 400},
    "vivier 600": {"vivier": 600},
    "noms +0,05": {"vivier": 200, "bonus_nom": 0.05},
    "noms +0,10": {"vivier": 200, "bonus_nom": 0.10},
    "noms +0,20": {"vivier": 200, "bonus_nom": 0.20},
    "voisins top 5": {"vivier": 200, "voisins_top": 5},
    "voisins top 10": {"vivier": 200, "voisins_top": 10},
    "MMR 0,85": {"vivier": 200, "mmr": 0.85},
    "MMR 0,7": {"vivier": 200, "mmr": 0.7},
    "vivier 600 + noms +0,10": {"vivier": 600, "bonus_nom": 0.10},
    "vivier 600 + noms +0,10 + voisins 5": {"vivier": 600, "bonus_nom": 0.10, "voisins_top": 5},
    "voisins des 10 premiers, en fin": {"vivier": 200, "voisins_fin": 10},
    "voisins des 20 premiers, en fin": {"vivier": 200, "voisins_fin": 20},
    "MMR 0,5": {"vivier": 200, "mmr": 0.5},
    "MMR 0,7 + voisins des 10, en fin": {"vivier": 200, "mmr": 0.7, "voisins_fin": 10},
}


@dataclass
class Verdicts:
    """Ce qu'on garde d'une requête : ses verdicts par variante, rien d'autre (les
    600 candidats de ~1 700 requêtes ne tiendraient pas en mémoire)."""
    jeu: str
    groupe: str
    moitie: str
    fidele: bool
    par_variante: dict[str, dict[str, Any]]


def evaluer(r: Requete, now: int, juge: Callable[[list[str]], dict[str, Any]],
            route: list[str]) -> Verdicts:
    par_variante = {nom: juge(classer(r, now, **params)) for nom, params in VARIANTES.items()}
    return Verdicts(r.jeu, r.groupe, r.moitie, classer(r, now, 200) == route, par_variante)


async def construire(args: argparse.Namespace) -> list[Verdicts]:
    garder_la_boucle(asyncio.get_running_loop())
    sortie: list[Verdicts] = []
    tmp = Path(tempfile.mkdtemp(prefix="mnemos_leviers_"))
    try:
        app, _, _ = await setup_bench_app(tmp, "ollama", salience_workers=0)
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://t", timeout=600) as client:
                store = app.state.store

                async def ecrire(user: str, session: str, msgs: list[dict[str, Any]]) -> None:
                    for j in range(0, len(msgs), 20):
                        (await client.post("/add", json={
                            "request_id": f"{user}:{session}:{j}", "user_id": user,
                            "session_id": session, "messages": msgs[j:j + 20]})).raise_for_status()

                async def une(jeu: str, groupe: str, moitie: str, tenant: str, question: str,
                              voisins: dict[str, list[str]], contenus: dict[str, str],
                              juge: Callable[[list[str]], dict[str, Any]]) -> None:
                    route = [it["id"].removeprefix("ep_") for it in (await client.post("/search", json={
                        "query": question, "user_id": tenant, "top_k": TOP_K})).json()["data"]]
                    # Horloge lue juste après la route : la récence (inerte en 2023)
                    # doit être calculée au même instant pour le contrôle de fidélité.
                    now = store._clock.now_ms()
                    r = Requete(jeu, groupe, moitie, tenant, question,
                                await candidats(store, question, tenant), voisins, contenus)
                    sortie.append(evaluer(r, now, juge, route))

                # LoCoMo complet
                for c in json.loads(Path("bench/data/locomo10.json").read_text(encoding="utf-8"))[: args.locomo]:
                    conv, sid = c["conversation"], c["sample_id"]
                    tenant = f"locomo:{sid}"
                    ts_par_dia: dict[str, str] = {}
                    for s_key in sorted((k for k in conv if k.startswith("session_")
                                         and not k.endswith("_date_time")),
                                        key=lambda k: int(k.split("_")[1])):
                        base = parse_locomo_datetime(conv.get(f"{s_key}_date_time"))
                        msgs = []
                        for i, t in enumerate(conv[s_key]):
                            ts_par_dia[t["dia_id"]] = _iso(base + i * PAS_MS)
                            msgs.append({"role": "user" if t.get("speaker") == conv["speaker_a"] else "assistant",
                                         "content": f"{t.get('speaker')}: {t.get('text', '')}",
                                         "timestamp": base + i * PAS_MS})
                        await ecrire(tenant, s_key, msgs)
                    voisins, contenus = await episodes_du_tenant(store, tenant)
                    async with store._sessions() as session:
                        ts_par_id = {row[0]: _iso(row[1]) for row in (await session.execute(text(
                            "SELECT id, created_at FROM episodes WHERE tenant = :t"), {"t": tenant})).all()}
                    for q in c["qa"]:
                        preuves = [ts_par_dia[p] for p in _preuves_locomo(q) if p in ts_par_dia]
                        if q.get("category") == 5 or not preuves:
                            continue
                        await une("locomo", f"cat{q['category']}",
                                  "reglage" if sid in REGLAGE else "validation", tenant, q["question"],
                                  voisins, contenus, juge_locomo(preuves, ts_par_id))
                    print(f"  LoCoMo {sid}", flush=True)

                # D1 dur
                for inst in HARD_UPDATE_INSTANCES:
                    tenant = f"d1:{inst.id}"
                    await ecrire(tenant, inst.id, build_messages(inst, 1_690_000_000_000))
                    voisins, contenus = await episodes_du_tenant(store, tenant)
                    cible = normalize_text(inst.statements[2].format(inst.value_2))
                    ancien = normalize_text(inst.statements[0].format(inst.value_0))

                    def juge_d1(ids: list[str], cible: str = cible, ancien: str = ancien,
                                contenus: dict[str, str] = contenus) -> dict[str, Any]:
                        norm = [normalize_text(contenus[i]) for i in ids]
                        r2 = next((k for k, c in enumerate(norm, 1) if c == cible), 99)
                        r0 = next((k for k, c in enumerate(norm, 1) if c == ancien), 99)
                        return {"actif_top3": r2 <= 3, "ancien_devant": r0 < r2}
                    await une("d1", "d1", "toutes", tenant, inst.probe, voisins, contenus, juge_d1)
                print("  D1 dur", flush=True)

                # PersonaMem, premières personas disponibles (50 par défaut, comme le 29/09)
                lignes = load_personas(PM_DIR / "val.csv")
                dispo = [pid for pid, rs in lignes.items()
                         if (PM_DIR / "chats" / Path(rs[0]["chat_history_32k_link"]).name).exists()]
                for pid in dispo[: args.personas]:
                    tenant = f"pm:{pid}"
                    chat = load_chat(PM_DIR / "chats" / Path(lignes[pid][0]["chat_history_32k_link"]).name)
                    for k, chunk in enumerate(chunk_messages(chat)):
                        (await client.post("/add", json={"request_id": f"{tenant}:{k}", "user_id": tenant,
                                                         "session_id": tenant, "messages": chunk})
                         ).raise_for_status()
                    voisins, contenus = await episodes_du_tenant(store, tenant)
                    for row in lignes[pid]:
                        ev = parse_evidence(row.get("related_conversation_snippet", ""))
                        if not ev:
                            continue

                        def juge_pm(ids: list[str], ev: Any = ev,
                                    contenus: dict[str, str] = contenus) -> dict[str, Any]:
                            r = next((k for k, i in enumerate(ids, 1) if _normalize(contenus[i]) in ev), None)
                            return {"present": r is not None, "hit10": r is not None and r <= 10}
                        await une("pm", "pm", "toutes", tenant, parse_query(row["user_query"]),
                                  voisins, contenus, juge_pm)
                print(f"  PersonaMem : {min(len(dispo), args.personas)} personas", flush=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return sortie


def agreger(verdicts: list[Verdicts]) -> dict[str, Any]:
    ecarts = sum(not v.fidele for v in verdicts)
    print(f"\nContrôle de fidélité : variante déployée ≠ route /search sur {ecarts} requête(s)"
          f" sur {len(verdicts)}\n")
    res: dict[str, Any] = {"fidelite_ecarts": ecarts, "requetes": len(verdicts), "variantes": {}}
    groupes = sorted({(v.jeu, v.groupe, v.moitie) for v in verdicts})
    for nom in VARIANTES:
        par_groupe: dict[str, dict[str, Any]] = {}
        for jeu, groupe, moitie in groupes:
            vs = [v.par_variante[nom] for v in verdicts if (v.jeu, v.groupe, v.moitie) == (jeu, groupe, moitie)]
            cle = f"{jeu} | {groupe} | {moitie}"
            if jeu == "locomo":
                par_groupe[cle] = {
                    "n": len(vs),
                    "toutes@100": round(sum(x["toutes"] for x in vs) / len(vs), 3),
                    "preuves@100": round(sum(x["presentes"] for x in vs) / sum(x["n"] for x in vs), 3),
                    "hit@10": round(sum(x["hit10"] for x in vs) / len(vs), 3)}
            elif jeu == "d1":
                par_groupe[cle] = {"n": len(vs), "actif_top3": sum(x["actif_top3"] for x in vs),
                                   "ancien_devant": sum(x["ancien_devant"] for x in vs)}
            else:
                par_groupe[cle] = {"n": len(vs),
                                   "presente@100": round(sum(x["present"] for x in vs) / len(vs), 3),
                                   "hit@10": round(sum(x["hit10"] for x in vs) / len(vs), 3)}
        res["variantes"][nom] = par_groupe
    return res


def afficher(res: dict[str, Any]) -> None:
    for g in next(iter(res["variantes"].values())):
        print(f"== {g}")
        for nom, par in res["variantes"].items():
            print(f"   {nom:38s} {par[g]}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-sauts : leviers de sélection")
    parser.add_argument("--personas", type=int, default=PM_PERSONAS)
    parser.add_argument("--locomo", type=int, default=10, help="essai rapide : N premières conversations")
    parser.add_argument("--output", type=Path, default=Path("bench/results/multisauts/multisauts_leviers.json"))
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    res = agreger(asyncio.run(construire(args)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    afficher(res)
    print(f"\nRapport : {args.output}")


if __name__ == "__main__":
    main()

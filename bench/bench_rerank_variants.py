"""Reclassement hors ligne : quelle formule de score, sur quels candidats.

Le correctif de récence (6e1cd11) a amélioré D1 dur (plus ancien devant l'actif
11/13 → 8/13) et **dégradé LoCoMo** (hit@10 0,607 → 0,373, meilleure preuve reculée
sur 104 questions sur 150). Il combinait deux changements — sparse limité au
contenu, récence rapportée au plus récent des candidats — qu'il faut séparer
avant de décider quoi garder.

Ce harnais ingère LoCoMo et les 13 instances D1 **une seule fois**, sans aucun
LLM (pas de saillance, pas de consolidation : `store.search` n'en dépend pas),
puis reproduit fidèlement l'étape KNN de `EpisodicStore.search` — mêmes
`max(2k, 50)` candidats, mêmes filtres — et reclasse **les mêmes candidats**
selon chaque variante de score. Les variantes ne diffèrent donc que par la
formule, jamais par l'ingestion ni par le hasard d'un LLM.

Contrôle de fidélité : les variantes `ancien` et `HEAD` doivent retrouver les
chiffres mesurés de bout en bout par `bench_locomo_qa` et `bench_update_hard`.
Si elles ne les retrouvent pas, le harnais ne reproduit pas la recherche et
rien de ce qu'il dit sur les autres variantes ne vaut.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import statistics
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ.setdefault("EPISODIC_RETENTION_DAYS", "36500")
os.environ.setdefault("DECAY_RATE_DAILY", "0.0")

import httpx
import sqlite_vec  # type: ignore[import-untyped]
from sqlalchemy import select, text

from bench.bench_locomo import parse_locomo_datetime, setup_bench_app
from bench.bench_locomo_qa import _evidence_ids
from bench.bench_personamem import (
    DATA_DIR as PM_DIR,
)
from bench.bench_personamem import (
    _normalize,
    chunk_messages,
    load_chat,
    load_personas,
    parse_evidence,
    parse_query,
)
from bench.bench_update_hard import build_messages
from bench.datasets import HARD_UPDATE_INSTANCES
from bench.eval_utils import normalize_text
from mnemos.embeddings.sparse import sparse_encode, sparse_similarity
from mnemos.models.episodic import Episode, EpisodeSparse
from mnemos.stores.episodic import DAY_MS, KNN_CANDIDATES

TOP_K = 100
# La variante que `EpisodicStore.search` est censé implémenter.
VARIANTE_DEPLOYEE = "sparse recouvrement"
_CONTENU = (1 << 224) - 1  # bits 0-223 : le contenu ; 224-255 : la date


def _recouvrements(q: int, e: int) -> dict[str, float]:
    """Mesures lexicales SANS pénalité de longueur, sur les seuls bits de contenu.

    Hamming vaut |Q| + |E| - 2|Q∩E| : à recouvrement égal, un épisode long est
    plus loin qu'un épisode court. Le recouvrement mesure la part des jetons de
    la requête présents dans l'épisode ; Jaccard normalise par l'union."""
    inter = (q & e).bit_count()
    return {"s_recouv": inter / max(1, q.bit_count()),
            "s_jaccard": inter / max(1, (q | e).bit_count())}


# Une composante par candidat ; chaque variante n'est qu'une combinaison.
Comp = dict[str, float]
Variante = Callable[[Comp], float]


def _rel(c: Comp, demi_vie: float) -> float:
    return float(2.0 ** (-c["age_rel_j"] / demi_vie))


def _ancien_decale(k: int) -> Variante:
    """L'ancienne formule, requête encodée k tranches de 4 h plus tard."""
    return lambda c: 0.7 * c["dense"] + 0.3 * c[f"s_full_{k}"] + 0.1 * c["r_mur"]


VARIANTES: dict[str, Variante] = {
    # Les deux points de contrôle : doivent retrouver les mesures de bout en bout.
    "ancien (8ed4ea0)": lambda c: 0.7 * c["dense"] + 0.3 * c["s_full"] + 0.1 * c["r_mur"],
    "HEAD (6e1cd11)": lambda c: 0.7 * c["dense"] + 0.3 * c["s_cont"] + 0.1 * _rel(c, 30),
    # Isoler le sparse : la récence reste morte comme avant.
    "sparse contenu, récence morte": lambda c: 0.7 * c["dense"] + 0.3 * c["s_cont"] + 0.1 * c["r_mur"],
    "sparse contenu, sans récence": lambda c: 0.7 * c["dense"] + 0.3 * c["s_cont"],
    "dense seul": lambda c: c["dense"],
    **{f"ancien, requête à +{4 * k} h": _ancien_decale(k) for k in range(1, 6)},
    "sparse recouvrement": lambda c: 0.7 * c["dense"] + 0.3 * c["s_recouv"],
    "sparse Jaccard": lambda c: 0.7 * c["dense"] + 0.3 * c["s_jaccard"],
    # Isoler la récence : poids réduits, puis demi-vies allongées.
    "récence rel. w=0,005": lambda c: 0.7 * c["dense"] + 0.3 * c["s_cont"] + 0.005 * _rel(c, 30),
    "récence rel. w=0,01": lambda c: 0.7 * c["dense"] + 0.3 * c["s_cont"] + 0.01 * _rel(c, 30),
    "récence rel. w=0,02": lambda c: 0.7 * c["dense"] + 0.3 * c["s_cont"] + 0.02 * _rel(c, 30),
    "récence rel. w=0,04": lambda c: 0.7 * c["dense"] + 0.3 * c["s_cont"] + 0.04 * _rel(c, 30),
    "récence rel. demi-vie 180 j": lambda c: 0.7 * c["dense"] + 0.3 * c["s_cont"] + 0.1 * _rel(c, 180),
    "récence rel. demi-vie 365 j": lambda c: 0.7 * c["dense"] + 0.3 * c["s_cont"] + 0.1 * _rel(c, 365),
}


async def candidats(store: Any, query: str, tenant: str) -> list[tuple[str, int, Comp]]:
    """Reproduit l'étape KNN et les filtres de `EpisodicStore.search` à k=TOP_K,
    puis calcule toutes les composantes dont les variantes ont besoin."""
    now = store._clock.now_ms()
    dense = await store._embedder.embed(query)
    q_sparse = sparse_encode(query, now)
    # L'ancien code encodait la requête à l'heure du run : ses bits temporels, et
    # donc son classement, changent toutes les 4 h. Cinq autres tranches pour
    # mesurer la dispersion de ce tirage.
    q_decales = [sparse_encode(query, now + k * 4 * 3_600_000) for k in range(1, 6)]
    q_bits = int.from_bytes(q_sparse, "little") & _CONTENU
    async with store._sessions() as session:
        knn = await session.execute(
            text("SELECT episode_id, distance FROM episodes_vec "
                 "WHERE embedding MATCH :emb AND k = :k AND tenant = :tenant"),
            {"emb": sqlite_vec.serialize_float32(dense),
             "k": max(TOP_K * 2, KNN_CANDIDATES), "tenant": tenant},
        )
        dist = {row[0]: float(row[1]) for row in knn}
        rows = (await session.execute(
            select(Episode, EpisodeSparse.sparse_bits)
            .join(EpisodeSparse, EpisodeSparse.episode_id == Episode.id)
            .where(Episode.id.in_(dist))
        )).tuples().all()
    retenus = [(e, b) for e, b in rows if e.tenant == tenant and not e.archived]
    if not retenus:
        return []
    ref = max(e.created_at for e, _ in retenus)
    out = []
    for e, bits in retenus:
        out.append((e.content, e.created_at, {
            "dense": 1.0 - dist[e.id],
            "s_full": sparse_similarity(q_sparse, bits),
            # Hamming restreint au contenu (état de 6e1cd11), conservé pour
            # pouvoir reproduire cette variante après son remplacement.
            "s_cont": 1.0 - (q_bits ^ (int.from_bytes(bits, "little") & _CONTENU)).bit_count() / 224,
            "r_mur": float(2.0 ** (-max(0.0, (now - e.created_at) / DAY_MS) / 30.0)),
            "age_rel_j": (ref - e.created_at) / DAY_MS,
            **{f"s_full_{k}": sparse_similarity(qd, bits) for k, qd in enumerate(q_decales, 1)},
            **_recouvrements(q_bits, int.from_bytes(bits, "little") & _CONTENU),
            "longueur": len(e.content.split()),
        }))
    return out


def classer(cands: list[tuple[str, int, Comp]], v: Variante) -> list[str]:
    return [c for c, _, _ in sorted(cands, key=lambda x: v(x[2]), reverse=True)[:TOP_K]]


def metriques_locomo(questions: list[dict[str, Any]], par_q: list[list[str]]) -> dict[str, Any]:
    def resume(rows: list[tuple[list[int | None]]]) -> dict[str, Any]:
        tous = [r for (rs,) in rows for r in rs]
        best = [min((r for r in rs if r is not None), default=None) for (rs,) in rows]
        trouves = sorted(b for b in best if b is not None)
        return {
            "n": len(rows),
            "rappel@100": round(sum(r is not None for r in tous) / len(tous), 3),
            "hit@10": round(sum(1 for b in trouves if b <= 10) / len(rows), 3),
            "rang_median": trouves[len(trouves) // 2] if trouves else None,
        }

    par_cat: dict[Any, list[tuple[list[int | None]]]] = {}
    tout: list[tuple[list[int | None]]] = []
    meilleurs: list[int] = []
    for q, contenus in zip(questions, par_q, strict=True):
        rangs = [next((i for i, c in enumerate(contenus, 1) if c.startswith(f"[{ev}]")), None)
                 for ev in _evidence_ids(q)]
        tout.append((rangs,))
        par_cat.setdefault(q["category"], []).append((rangs,))
        meilleurs.append(min((r for r in rangs if r is not None), default=999))
    return {"toutes": resume(tout), **{f"cat{c}": resume(r) for c, r in sorted(par_cat.items())},
            "_meilleurs": meilleurs}


def metriques_pm(par_q: list[tuple[dict[str, str], list[str]]]) -> dict[str, Any]:
    """Rang du premier message-preuve (égalité de texte normalisé, comme
    `bench_personamem`), global et sur les seules préférences mises à jour.

    Le sous-ensemble « mises à jour » (32 questions) est SATURÉ : rang 1 pour
    toutes les variantes testées le 29/09/2026, la preuve étant thématiquement
    identique à la requête. Il est gardé pour détecter une régression, jamais
    pour départager deux variantes."""
    def resume(rows: list[int | None]) -> dict[str, Any]:
        trouves = sorted(r for r in rows if r is not None)
        return {"n": len(rows),
                "rappel@100": round(len(trouves) / len(rows), 3) if rows else 0.0,
                "hit@10": round(sum(1 for r in trouves if r <= 10) / len(rows), 3) if rows else 0.0,
                "rang_median": trouves[len(trouves) // 2] if trouves else None}
    tous, maj = [], []
    for row, contenus in par_q:
        ev = parse_evidence(row.get("related_conversation_snippet", ""))
        r = next((i for i, c in enumerate(contenus, 1) if _normalize(c) in ev), None)
        tous.append(r)
        if row.get("updated", "").strip().lower() == "true":
            maj.append(r)
    return {"toutes": resume(tous), "mises_a_jour": resume(maj)}


def metriques_d1(par_inst: list[tuple[Any, list[str]]]) -> dict[str, Any]:
    v2s, devant = [], 0
    for inst, contenus in par_inst:
        norm = [normalize_text(c) for c in contenus]
        rangs = [next((i for i, c in enumerate(norm, 1) if c == normalize_text(g.format(v))), 99)
                 for g, v in zip(inst.statements, (inst.value_0, inst.value_1, inst.value_2),
                                 strict=True)]
        devant += rangs[0] < rangs[2]
        v2s.append(rangs[2])
    return {"plus_ancien_devant": f"{devant}/{len(par_inst)}",
            "actif_top3": f"{sum(r <= 3 for r in v2s)}/{len(par_inst)}",
            "rang_actif_moyen": round(statistics.mean(v2s), 2)}


async def main_async(out: Path) -> dict[str, Any]:
    data = json.loads(Path("bench/data/locomo_sample.json").read_text(encoding="utf-8"))
    conv = data["conversation"]
    questions = [q for q in data["qa"]
                 if q.get("category") != 5 and q.get("answer") is not None and _evidence_ids(q)]

    tmp = Path(tempfile.mkdtemp(prefix="mnemos_rerank_"))
    try:
        app, _, settings = await setup_bench_app(tmp, "ollama", salience_workers=0,
                                                 embed_backend="llamacpp")
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://t",
                                         timeout=600) as client:
                sessions = sorted(
                    (k for k in conv if k.startswith("session_") and not k.endswith("_date_time")),
                    key=lambda k: int(k.split("_")[1]))
                for s_key in sessions:
                    base_ms = parse_locomo_datetime(conv.get(f"{s_key}_date_time"))
                    msgs = [{"role": "user" if t.get("speaker") == conv.get("speaker_a") else "assistant",
                             "content": f"[{t.get('dia_id')}] {t.get('speaker')}: {t.get('text', '')}",
                             "timestamp": base_ms + i * 15000}
                            for i, t in enumerate(conv[s_key])]
                    (await client.post("/add", json={"request_id": s_key, "user_id": "locomo",
                                                     "session_id": s_key, "messages": msgs})
                     ).raise_for_status()
                for inst in HARD_UPDATE_INSTANCES:
                    msgs = build_messages(inst, 1_690_000_000_000)
                    for i in range(0, len(msgs), 20):
                        (await client.post("/add", json={
                            "request_id": f"{inst.id}:{i}", "user_id": f"d1_{inst.id}",
                            "session_id": inst.id, "messages": msgs[i:i + 20]})).raise_for_status()

                pm_rows = load_personas(PM_DIR / "val.csv")
                pm_ok = [pid for pid, rs in pm_rows.items()
                         if (PM_DIR / "chats" / Path(rs[0]["chat_history_32k_link"]).name).exists()]
                for pid in pm_ok:
                    chat = load_chat(PM_DIR / "chats" / Path(pm_rows[pid][0]["chat_history_32k_link"]).name)
                    for c_idx, chunk in enumerate(chunk_messages(chat)):
                        (await client.post("/add", json={
                            "request_id": f"pm:{pid}:{c_idx}", "user_id": f"pm:{pid}",
                            "session_id": f"pm:{pid}", "messages": chunk})).raise_for_status()
                print(f"PersonaMem : {len(pm_ok)} personas ingérées")

                store = app.state.store

                async def reel(query: str, tenant: str) -> list[str]:
                    """Le classement du code réellement déployé, pour contrôle."""
                    return [s.episode.content
                            for s in await store.search(query, k=TOP_K, tenant=tenant)]

                c_locomo = [await candidats(store, q["question"], "locomo") for q in questions]
                r_locomo = [await reel(q["question"], "locomo") for q in questions]
                c_d1 = [(inst, await candidats(store, inst.probe, f"d1_{inst.id}"))
                        for inst in HARD_UPDATE_INSTANCES]
                r_d1 = [await reel(inst.probe, f"d1_{inst.id}") for inst in HARD_UPDATE_INSTANCES]
                pm_q = [(pid, row) for pid in pm_ok for row in pm_rows[pid]
                        if parse_evidence(row.get("related_conversation_snippet", ""))]
                c_pm = [(row, await candidats(store, parse_query(row["user_query"]), f"pm:{pid}"))
                        for pid, row in pm_q]
                r_pm = [await reel(parse_query(row["user_query"]), f"pm:{pid}") for pid, row in pm_q]

                async def route(query: str, tenant: str) -> list[str]:
                    """Ce que la plateforme reçoit vraiment : POST /search, avec sa
                    déduplication par contenu et son tri final. Sans consolidation,
                    comme en production, aucun fait ne s'y mêle."""
                    resp = await client.post("/search", json={
                        "query": query, "user_id": tenant, "top_k": TOP_K})
                    resp.raise_for_status()
                    return [it["content"] for it in resp.json()["data"]]

                a_locomo = [await route(q["question"], "locomo") for q in questions]
                a_d1 = [await route(inst.probe, f"d1_{inst.id}") for inst in HARD_UPDATE_INSTANCES]
                a_pm = [await route(parse_query(row["user_query"]), f"pm:{pid}") for pid, row in pm_q]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    tous = [c for cs in c_locomo for _, _, c in cs]
    q1, _, q3 = statistics.quantiles([c["longueur"] for c in tous], n=4)
    courts = [c for c in tous if c["longueur"] <= q1]
    longs = [c for c in tous if c["longueur"] >= q3]
    print(f"Biais de longueur parmi les {len(tous)} candidats LoCoMo (moyennes par quartile) :")
    for nom in ("dense", "s_cont", "s_recouv", "s_jaccard"):
        mc, ml = statistics.mean(x[nom] for x in courts), statistics.mean(x[nom] for x in longs)
        print(f"  {nom:9s} quart court (<= {q1:.0f} mots) {mc:.3f} | quart long (>= {q3:.0f} mots) {ml:.3f}")
    print()

    resultats: dict[str, Any] = {}
    ref_best: list[int] | None = None
    for nom, v in VARIANTES.items():
        loc = metriques_locomo(questions, [classer(c, v) for c in c_locomo])
        best = loc.pop("_meilleurs")
        if ref_best is None:
            ref_best = best  # la première variante est l'ancien code
        loc["vs_ancien"] = {"recule": sum(b > a for a, b in zip(ref_best, best, strict=True)),
                            "avance": sum(b < a for a, b in zip(ref_best, best, strict=True))}
        resultats[nom] = {"locomo": loc,
                          "d1": metriques_d1([(i, classer(c, v)) for i, c in c_d1]),
                          "personamem": metriques_pm([(r, classer(c, v)) for r, c in c_pm])}

    loc = metriques_locomo(questions, r_locomo)
    best = loc.pop("_meilleurs")
    assert ref_best is not None
    loc["vs_ancien"] = {"recule": sum(b > a for a, b in zip(ref_best, best, strict=True)),
                        "avance": sum(b < a for a, b in zip(ref_best, best, strict=True))}
    resultats["CODE RÉEL (store.search)"] = {
        "locomo": loc,
        "d1": metriques_d1([(i, r) for (i, _), r in zip(c_d1, r_d1, strict=True)]),
        "personamem": metriques_pm([(row, r) for (row, _), r in zip(c_pm, r_pm, strict=True)]),
    }
    loc = metriques_locomo(questions, a_locomo)
    best = loc.pop("_meilleurs")
    loc["vs_ancien"] = {"recule": sum(b > a for a, b in zip(ref_best, best, strict=True)),
                        "avance": sum(b < a for a, b in zip(ref_best, best, strict=True))}
    resultats["ROUTE /search (AML)"] = {
        "locomo": loc,
        "d1": metriques_d1([(i, r) for (i, _), r in zip(c_d1, a_d1, strict=True)]),
        "personamem": metriques_pm([(row, r) for (row, _), r in zip(c_pm, a_pm, strict=True)]),
    }
    # Contrôle d'implémentation : le code déployé doit classer exactement comme
    # la variante de référence, requête par requête.
    v_ref = VARIANTES[VARIANTE_DEPLOYEE]
    paires = ([(classer(c, v_ref), r) for c, r in zip(c_locomo, r_locomo, strict=True)]
              + [(classer(c, v_ref), r) for (_, c), r in zip(c_d1, r_d1, strict=True)]
              + [(classer(c, v_ref), r) for (_, c), r in zip(c_pm, r_pm, strict=True)])
    desaccords = sum(1 for a, b in paires if a != b)
    print(f"Contrôle d'implémentation : store.search et « {VARIANTE_DEPLOYEE} » diffèrent"
          f" sur {desaccords} requête(s) sur {len(paires)}\n")

    print(f"{len(questions)} questions LoCoMo, {len(c_d1)} instances D1,"
          f" {len(c_pm)} questions PersonaMem, top_k={TOP_K}\n")
    print(f"{'variante':32s} | {'hit@10':>6s} {'médian':>6s} {'rap@100':>7s} | {'recule/avance':>13s}"
          f" | {'D1 ancien devant':>16s} {'actif top3':>10s}"
          f" | {'PM hit@10':>9s} {'méd':>4s} {'rap@100':>7s} {'maj(saturé)':>11s}")
    for nom, r in resultats.items():
        t, d, pm = r["locomo"]["toutes"], r["d1"], r["personamem"]
        va = r["locomo"]["vs_ancien"]
        print(f"{nom:32s} | {t['hit@10']:6.3f} {t['rang_median']!s:>6s} {t['rappel@100']:7.3f} |"
              f" {va['recule']:>5d}/{va['avance']:<7d} | {d['plus_ancien_devant']:>16s} {d['actif_top3']:>10s}"
              f" | {pm['toutes']['hit@10']:9.3f} {pm['toutes']['rang_median']!s:>4s}"
              f" {pm['toutes']['rappel@100']:7.3f} {pm['mises_a_jour']['hit@10']:11.3f}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(resultats, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nRapport : {out}")
    return resultats


def main() -> None:
    parser = argparse.ArgumentParser(description="Reclassement hors ligne des variantes de score")
    parser.add_argument("--output", type=Path, default=Path("bench/results/gpu/rerank_variants.json"))
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    asyncio.run(main_async(args.output))


if __name__ == "__main__":
    main()

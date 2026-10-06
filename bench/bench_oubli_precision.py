"""Demandes d'oubli : une variante plus PRÉCISE du lien d'adjacence ?

Validation du lien d'adjacence (`bench_oubli_rattachement.py`, 240 personas neufs) : quand
la bonne consigne est ajoutée, 12 questions gagnées et 0 perdue ; mais la règle ajoute aussi
des consignes sans rapport à ~60 % des autres questions, et ce reste du contexte changé
coûte (28 gagnées contre 43 perdues). Total PersonaMem : −3 sur 664 questions.

Ici, sans modèle de langage, sur les trois bases déjà ingérées (carte g7, frais g8, g9) :
deux leviers pour couper les ajouts parasites en gardant le rappel ciblé.
  * fenêtre plus courte (3, 5) : l'accusé de la bonne consigne est souvent dans le top 4 ;
  * seuil de similarité entre la question et la CIBLE de la consigne (`detect_forget`).
Pour chaque variante : bonne consigne récupérée (questions où elle manquait), part des
autres questions touchées, consignes sans rapport ajoutées aux questions d'oubli.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ.setdefault("EPISODIC_RETENTION_DAYS", "36500")
os.environ.setdefault("DECAY_RATE_DAILY", "0.0")

import httpx
from sqlalchemy import select

from bench.bench_forget_targeting import _mots, porte
from bench.bench_locomo import setup_bench_app
from bench.bench_oubli_rattachement import TOP_K, echantillon, liens_du_tenant, rattacher_consignes
from bench.bench_personamem import DATA_DIR, load_chat, load_personas, parse_query
from bench.bench_temporal_recall import garder_la_boucle
from mnemos.models.episodic import Episode
from mnemos.router.forget import detect_forget
from mnemos.stores.episodic import _cosine

BASES = {  # base gardée → (graine, personas, frais, graines exclues)
    "carte_g7": (7, 70, False, ()),
    "frais_g8": (8, 120, True, ()),
    "frais_g9": (9, 120, True, (8,)),
}
FENETRES = (3, 5, 10)
PLAFONDS = (1, 2)
SEUILS: tuple[float | None, ...] = (None, 0.40, 0.45, 0.50, 0.55, 0.60)


async def mesurer(nom: str, lignes: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    graine, n, frais, exclure = BASES[nom]
    personas = echantillon(lignes, graine, n, frais, exclure)
    out: list[dict[str, Any]] = []
    app, _, _ = await setup_bench_app(Path("bench/data/bases") / nom, "ollama", salience_workers=0,
                                      embed_backend="llamacpp")
    async with app.router.lifespan_context(app):
        store = app.state.store
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t", timeout=600) as client:
            for pid in personas:
                rows = [r for r in lignes[pid] if r.get("correct_answer") and r.get("incorrect_answers")]
                if not rows:
                    continue
                tenant = f"pm:{pid}"
                chat = load_chat(DATA_DIR / "chats" / Path(rows[0]["chat_history_32k_link"]).name)
                liens = (await liens_du_tenant(store, tenant, [str(m["content"]) for m in chat]))["suivant"]
                async with store._sessions() as s:
                    eps = (await s.execute(select(Episode.id, Episode.role, Episode.content).where(
                        Episode.tenant == tenant, Episode.archived == 0))).all()
                contenus = {f"ep_{e}": str(c) for e, _, c in eps}
                cibles = {c: d.target for c in liens if (d := detect_forget(contenus[c], "user")) is not None}
                cles = list(cibles)
                vecs = dict(zip(cles, await store._embedder.embed_batch([cibles[c] for c in cles]), strict=True))
                for k, row in enumerate(rows):
                    requete = parse_query(row["user_query"])
                    items = (await client.post("/search", json={
                        "query": requete, "user_id": tenant, "top_k": TOP_K})).json()["data"]
                    ids = [it["id"] for it in items]
                    q = await store._embedder.embed(requete)
                    sim = {c: _cosine(q, v) for c, v in vecs.items()}
                    oubli = row.get("pref_type") == "ask_to_forget"
                    bonnes: set[str] = set()
                    if oubli:  # étiquette : sert à noter, jamais à choisir
                        mots = _mots(row.get("prev_pref", ""))
                        bonnes = {c for c in liens if porte(contenus[c], mots)}
                    x: dict[str, Any] = {"cle": f"{pid}#{k}", "oubli": oubli, "bonne_existe": bool(bonnes),
                                         "bonne_presente": bool(bonnes & set(ids))}
                    for f in FENETRES:
                        for p in PLAFONDS:
                            for s_ in SEUILS:
                                filtres = {c: v for c, v in liens.items() if s_ is None or sim.get(c, 0.0) >= s_}
                                ajouts = set(rattacher_consignes(ids, filtres, TOP_K, f, p, "voisin")) - set(ids)
                                x[f"{f}/{p}/{s_}"] = [len(ajouts), bool(ajouts & bonnes), len(ajouts - bonnes)]
                    out.append(x)
    return out


def synthese(stats: list[dict[str, Any]], titre: str) -> dict[str, Any]:
    manque = [x for x in stats if x["oubli"] and x["bonne_existe"] and not x["bonne_presente"]]
    autres = [x for x in stats if not x["oubli"]]
    oubli = [x for x in stats if x["oubli"]]
    print(f"\n== {titre} : {len(stats)} questions, consigne manquante {len(manque)}, autres {len(autres)}")
    print(f"{'variante':>16s} {'bonne':>8s} {'autres touchées':>16s} {'parasites oubli':>16s}")
    res = {}
    for f in FENETRES:
        for p in PLAFONDS:
            for s_ in SEUILS:
                c = f"{f}/{p}/{s_}"
                r = {"bonne": sum(x[c][1] for x in manque), "manque": len(manque),
                     "autres_touchees": sum(x[c][0] > 0 for x in autres), "autres": len(autres),
                     "parasites_oubli": sum(x[c][2] for x in oubli)}
                res[c] = r
                print(f"{c:>16s} {r['bonne']:>3d}/{len(manque):<4d} "
                      f"{r['autres_touchees']:>6d} ({100 * r['autres_touchees'] / max(1, len(autres)):3.0f} %)"
                      f" {r['parasites_oubli']:>12d}")
    return res


async def run(args: argparse.Namespace) -> None:
    garder_la_boucle(asyncio.get_running_loop())
    lignes = load_personas(DATA_DIR / "val.csv")
    tout: list[dict[str, Any]] = []
    rapport: dict[str, Any] = {}
    for nom in BASES:
        stats = await mesurer(nom, lignes)
        rapport[nom] = synthese(stats, nom)
        tout += stats
    rapport["ensemble"] = synthese(tout, "trois bases réunies")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rapport, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Rapport : {args.output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Demandes d'oubli : variante plus précise du lien d'adjacence")
    parser.add_argument("--output", type=Path, default=Path("bench/results/oubli/rattachement_precision.json"))
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(run(args))


if __name__ == "__main__":
    main()

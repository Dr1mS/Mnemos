"""Demandes d'oubli : la consigne aide-t-elle par sa présence ou par sa place ?

`bench_oubli_injection.py` : ajouter la consigne d'oubli en FIN de contexte, quand
`/search` ne la rend pas, fait passer 18 questions de 1 à 10 bonnes réponses. Mais
`bench_oubli_rattachement.py` montre que, pour ces 18 questions, l'accusé de réception
de l'assistant (« Got it — I'll forget that you… ») est déjà dans le top 100, souvent
dans les 4 premiers. L'information était donc là ; reste à savoir ce qui a aidé :

  * `fin`        — la consigne à la place du 100e souvenir (le bras déjà mesuré) ;
  * `tete`       — la consigne en tête de liste ;
  * `voisin`     — la consigne juste avant son accusé, là où un lien d'écriture
                   « consigne ↔ message suivant » la mettrait ;
  * `accuse_fin` — rien d'ajouté : l'accusé déjà présent est déplacé en fin de liste.

Mêmes 70 personas que la carte (graine 7), base gardée par `bench_oubli_rattachement.py`
(embeddings llama-server). Consigne et accusé repérés grâce à l'étiquette du jeu, pour
le diagnostic seulement. Consigne et lettre officielles, Nemotron 3 Ultra.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
from math import comb
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ.setdefault("EPISODIC_RETENTION_DAYS", "36500")
os.environ.setdefault("DECAY_RATE_DAILY", "0.0")

import httpx
from sqlalchemy import select

from bench import aml_officiel
from bench.bench_carte_personamem import consigne
from bench.bench_forget_targeting import _mots, porte
from bench.bench_locomo import setup_bench_app
from bench.bench_oubli_rattachement import echantillon
from bench.bench_personamem import DATA_DIR, load_chat, load_personas, parse_query
from bench.bench_temporal_recall import garder_la_boucle
from bench.remote_llm import RemoteChat
from mnemos.api.aml_routes import _ts_to_iso
from mnemos.models.episodic import Episode
from mnemos.router.forget import detect_forget

BRAS = ("fin", "tete", "voisin", "accuse_fin")


def _p(perdues: int, gagnees: int) -> float:
    n = perdues + gagnees
    return min(1.0, 2 * sum(comb(n, i) for i in range(min(perdues, gagnees) + 1)) / 2 ** n) if n else 1.0


def placer(items: list[dict[str, Any]], directive: dict[str, Any], accuse_id: str | None,
           bras: str) -> list[dict[str, Any]]:
    """La liste de 100 souvenirs du bras demandé (toujours la même longueur que `items`)."""
    if bras == "fin":
        return items[:-1] + [directive]
    if bras == "tete":
        return [directive] + items[:-1]
    ids = [it["id"] for it in items]
    if bras == "voisin":
        if accuse_id in ids:
            i = ids.index(accuse_id)
            return (items[:i] + [directive] + items[i:])[:len(items)]
        return items[:-1] + [directive]
    if bras == "accuse_fin":
        if accuse_id in ids:
            i = ids.index(accuse_id)
            return items[:i] + items[i + 1:] + [items[i]]
        return items
    raise ValueError(bras)


async def run(args: argparse.Namespace) -> None:
    garder_la_boucle(asyncio.get_running_loop())
    officiel = aml_officiel.charger("personamem", "pipeline_v2.py")
    lignes = load_personas(DATA_DIR / "val.csv")
    personas = echantillon(lignes, 7, 70, frais=False)
    repondeur = RemoteChat("nvidia")
    opts = {"temperature": 0.0, "num_predict": 768}
    faits: dict[str, dict[str, Any]] = {}
    if args.cache.exists():
        for x in args.cache.read_text(encoding="utf-8").splitlines():
            if x.strip():
                y = json.loads(x)
                faits[y["cle"]] = y
        print(f"Reprise : {len(faits)} questions déjà dans {args.cache}", flush=True)
    try:
        app, _, _ = await setup_bench_app(args.base, "ollama", salience_workers=0, embed_backend="llamacpp")
        async with app.router.lifespan_context(app):
            store = app.state.store
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://t", timeout=600) as client:
                for pid in personas:
                    rows = [r for r in lignes[pid] if r.get("correct_answer") and r.get("incorrect_answers")]
                    cibles = [(k, r) for k, r in enumerate(rows) if r.get("pref_type") == "ask_to_forget"
                              and f"{pid}#{k}" not in faits]
                    if not cibles:
                        continue
                    tenant = f"pm:{pid}"
                    chat = load_chat(DATA_DIR / "chats" / Path(rows[0]["chat_history_32k_link"]).name)
                    async with store._sessions() as s:
                        eps = (await s.execute(select(Episode.id, Episode.content, Episode.created_at).where(
                            Episode.tenant == tenant, Episode.archived == 0))).all()
                    par_contenu = {str(c): (f"ep_{e}", t) for e, c, t in eps}
                    for k, row in cibles:
                        cle = f"{pid}#{k}"
                        mots = _mots(row.get("prev_pref", ""))
                        pos = [i for i, m in enumerate(chat) if m["role"] == "user"
                               and detect_forget(m["content"], "user") is not None and porte(m["content"], mots)]
                        if not pos:
                            continue
                        i = pos[-1]
                        requete = parse_query(row["user_query"])
                        items = (await client.post("/search", json={
                            "query": requete, "user_id": tenant, "top_k": 100})).json()["data"]
                        ids = [it["id"] for it in items]
                        stockee = par_contenu.get(str(chat[i]["content"]))
                        if stockee is None or stockee[0] in ids:
                            continue  # consigne absente du magasin, ou déjà rendue : hors sujet ici
                        directive = {"id": stockee[0], "content": str(chat[i]["content"]),
                                     "created_at": _ts_to_iso(stockee[1])}
                        suivant = par_contenu.get(str(chat[i + 1]["content"])) if i + 1 < len(chat) else None
                        accuse_id = suivant[0] if suivant else None
                        options = [str(row["correct_answer"]), *(str(o) for o in json.loads(row["incorrect_answers"]))]
                        random.Random(f"{pid}:{k}:7").shuffle(options)  # même mélange que la carte
                        bonne = chr(65 + options.index(str(row["correct_answer"])))
                        contextes = {"A": items, **{b: placer(items, directive, accuse_id, b) for b in BRAS}}
                        reps: dict[str, str | None] = {}
                        try:
                            for nom, liste in contextes.items():
                                ctx = "Memories from the conversation history:\n" + "\n".join(
                                    f"[{it.get('created_at', '')}] {it['content']}" for it in liste)
                                sortie = await repondeur.generate(consigne(officiel, ctx, requete, options),
                                                                  "nvidia/nemotron-3-ultra-550b-a55b", options=opts)
                                reps[nom] = officiel.extract_final_letter(sortie)
                        except Exception as exc:  # noqa: BLE001 — Nvidia 429/503
                            print(f"  sautée, à refaire : {cle} ({str(exc)[:100]})", flush=True)
                            continue
                        y = {"cle": cle, "bonne": bonne,
                             "rang_accuse": ids.index(accuse_id) + 1 if accuse_id in ids else None,
                             **{f"lettre_{b}": v for b, v in reps.items()},
                             **{b: v == bonne for b, v in reps.items()}}
                        faits[cle] = y
                        with args.cache.open("a", encoding="utf-8") as fh:
                            fh.write(json.dumps(y, ensure_ascii=False) + "\n")
                        print(f"  {cle} : " + " ".join(f"{b}={'V' if y[b] else '.'}" for b in contextes), flush=True)
    finally:
        await repondeur.aclose()
    xs = list(faits.values())
    res: dict[str, Any] = {"questions": len(xs), "A": sum(x["A"] for x in xs)}
    print(f"\n{len(xs)} questions où la consigne manque ; A = {res['A']}")
    for b in BRAS:
        g = sum(x[b] and not x["A"] for x in xs)
        p = sum(x["A"] and not x[b] for x in xs)
        res[b] = {"bonnes": sum(x[b] for x in xs), "gagnees": g, "perdues": p, "p": round(_p(p, g), 4)}
        print(f"  {b:>10s} : {res[b]['bonnes']:>2d} bonnes (gagnées {g}, perdues {p}, p = {res[b]['p']})")
    res["details"] = xs
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Rapport : {args.output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Demandes d'oubli : présence ou place de la consigne")
    parser.add_argument("--base", type=Path, default=Path("bench/data/bases/carte_g7"))
    parser.add_argument("--output", type=Path, default=Path("bench/results/oubli/placement_consigne.json"))
    parser.add_argument("--cache", type=Path, default=Path("bench/results/oubli/placement_consigne.cache.jsonl"))
    args = parser.parse_args()
    if not (args.base / "episodic.db").exists():
        raise SystemExit(f"base absente : {args.base} (lancer d'abord bench_oubli_rattachement.py --base)")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(run(args))


if __name__ == "__main__":
    main()

"""Demandes d'oubli : que gagne-t-on si la consigne entre dans le contexte ?

`bench_oubli_consigne_rang.py` (06/10/2026) : la consigne d'oubli de l'utilisateur
n'est dans le top 100 de `/search` que pour 26 questions `ask_to_forget` sur 44, alors
que la préférence oubliée y est toujours. Le répondeur voit « j'aime X » sans voir
« oublie que j'aime X ».

Simulation, avant tout changement de Mnemos : mêmes personas et mêmes questions que la
carte (graine 7). Contexte A = la réponse de `/search` ; contexte A+ = la même, où la
consigne manquante, telle que stockée, remplace le dernier souvenir (toujours 100).
Consigne et lettre officielles (`pipeline_v2.py`), Nemotron 3 Ultra. La consigne est
repérée grâce à l'étiquette du jeu (`prev_pref`), et placée en fin de liste : un
mécanisme réel, sans étiquette et avec un autre placement, peut faire mieux ou moins bien.
Règle pré-enregistrée avant le résultat : go si gain net ≥ +5 avec au plus 1 perte sur
les questions où la consigne manquait, sinon piste arrêtée.
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
from math import comb
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ.setdefault("EPISODIC_RETENTION_DAYS", "36500")
os.environ.setdefault("DECAY_RATE_DAILY", "0.0")

import httpx

from bench import aml_officiel
from bench.bench_carte_personamem import consigne
from bench.bench_forget_targeting import _mots, porte
from bench.bench_locomo import setup_bench_app
from bench.bench_personamem import DATA_DIR, chunk_messages, load_chat, load_personas, parse_query
from bench.bench_temporal_recall import garder_la_boucle
from bench.remote_llm import RemoteChat
from mnemos.llm import ollama_client
from mnemos.router.forget import detect_forget

# La boucle locale de cette machine coupe Ollama par épisodes (06/10, 02 h) : plus de
# réessais pour le banc seulement, sans effet sur ce qui est mesuré.
ollama_client.RETRY_ATTEMPTS = 8


def _p(perdues: int, gagnees: int) -> float:
    n = perdues + gagnees
    return min(1.0, 2 * sum(comb(n, i) for i in range(min(perdues, gagnees) + 1)) / 2 ** n) if n else 1.0


async def run(args: argparse.Namespace) -> None:
    garder_la_boucle(asyncio.get_running_loop())
    officiel = aml_officiel.charger("personamem", "pipeline_v2.py")
    lignes = load_personas(DATA_DIR / "val.csv")
    dispo = sorted(p for p, rs in lignes.items()
                   if (DATA_DIR / "chats" / Path(rs[0]["chat_history_32k_link"]).name).exists())
    personas = random.Random(args.graine).sample(dispo, min(args.personas, len(dispo)))
    repondeur = RemoteChat("nvidia")
    opts = {"temperature": 0.0, "num_predict": 768}
    out: list[dict[str, Any]] = []
    if args.cache.exists():
        out = [json.loads(x) for x in args.cache.read_text(encoding="utf-8").splitlines() if x.strip()]
        print(f"Reprise : {len(out)} questions déjà dans {args.cache}", flush=True)
    faites = {x["cle"] for x in out}
    tmp = Path(tempfile.mkdtemp(prefix="mnemos_oubli_inj_"))
    try:
        app, _, _ = await setup_bench_app(tmp, "ollama", salience_workers=0)
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://t", timeout=600) as client:

                async def chercher(q: str, tenant: str, k: int) -> list[dict[str, Any]]:
                    r = await client.post("/search", json={"query": q, "user_id": tenant, "top_k": k})
                    r.raise_for_status()
                    return list(r.json()["data"])

                async def ingerer(tenant: str, chat: list[dict[str, Any]]) -> bool:
                    for essai in range(1, 11):
                        try:  # request_id idempotent : au réessai, un lot déjà écrit est ignoré
                            for c, chunk in enumerate(chunk_messages(chat)):
                                (await client.post("/add", json={
                                    "request_id": f"{tenant}:{c}", "user_id": tenant,
                                    "session_id": tenant, "messages": chunk})).raise_for_status()
                            return True
                        except Exception as exc:  # noqa: BLE001 — Ollama coupé par épisodes
                            print(f"  ingestion {tenant} interrompue (essai {essai}) : {str(exc)[:100]}",
                                  flush=True)
                            await asyncio.sleep(60)
                    return False

                async def question(pid: str, k: int, row: dict[str, Any], chat: list[dict[str, Any]]
                                   ) -> dict[str, Any] | None:
                    tenant = f"pm:{pid}"
                    requete = parse_query(row["user_query"])
                    options = [str(row["correct_answer"]), *(str(x) for x in json.loads(row["incorrect_answers"]))]
                    random.Random(f"{pid}:{k}:{args.graine}").shuffle(options)  # même mélange que la carte
                    bonne = chr(65 + options.index(str(row["correct_answer"])))
                    items = await chercher(requete, tenant, 100)
                    mots = _mots(row.get("prev_pref", ""))
                    consignes = [m for m in chat if m["role"] == "user"
                                 and detect_forget(m["content"], "user") is not None
                                 and porte(m["content"], mots)]
                    contenus = [str(it["content"]) for it in items]
                    manque = bool(consignes) and not any(
                        c.startswith(m["content"][:200]) for m in consignes for c in contenus)
                    lignes_a = [f"[{it.get('created_at', '')}] {it['content']}" for it in items]
                    lignes_plus = list(lignes_a)
                    if manque:
                        # le souvenir stocké de la consigne, au même format que les autres
                        cible = consignes[-1]["content"]
                        proches = await chercher(cible, tenant, 10)
                        trouve = next((it for it in proches if str(it["content"]).startswith(cible[:200])), None)
                        if trouve is None:
                            print(f"  consigne introuvable dans le magasin : {pid}#{k}", flush=True)
                            return None
                        lignes_plus = lignes_plus[:99] + [f"[{trouve.get('created_at', '')}] {trouve['content']}"]
                    reps: dict[str, str | None] = {}
                    for nom, lig in (("A", lignes_a), ("A+", lignes_plus)):
                        if nom == "A+" and not manque:
                            reps[nom] = reps["A"]
                            continue
                        ctx = "Memories from the conversation history:\n" + "\n".join(lig)
                        sortie = await repondeur.generate(consigne(officiel, ctx, requete, options),
                                                          "nvidia/nemotron-3-ultra-550b-a55b", options=opts)
                        reps[nom] = officiel.extract_final_letter(sortie)
                    return {"cle": f"{pid}#{k}", "consigne_manquait": manque, "bonne": bonne,
                            "lettre_A": reps["A"], "lettre_A+": reps["A+"],
                            "A": reps["A"] == bonne, "A+": reps["A+"] == bonne}

                for pid in personas:
                    tous = [r for r in lignes[pid] if r.get("correct_answer") and r.get("incorrect_answers")]
                    cibles = [(k, r) for k, r in enumerate(tous)
                              if r.get("pref_type") == "ask_to_forget" and f"{pid}#{k}" not in faites]
                    if not cibles:
                        continue
                    chat = load_chat(DATA_DIR / "chats" / Path(tous[0]["chat_history_32k_link"]).name)
                    if not await ingerer(f"pm:{pid}", chat):
                        print(f"  persona {pid} abandonné, à refaire", flush=True)
                        continue
                    for k, row in cibles:
                        try:
                            x = await question(pid, k, row, chat)
                        except Exception as exc:  # noqa: BLE001 — Nvidia 429/503, Ollama coupé
                            print(f"  sautée, à refaire : {pid}#{k} ({str(exc)[:100]})", flush=True)
                            continue
                        if x is None:
                            continue
                        out.append(x)
                        with args.cache.open("a", encoding="utf-8") as f:
                            f.write(json.dumps(x, ensure_ascii=False) + "\n")
                    print(f"  persona {pid} : {len(out)} questions", flush=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        await repondeur.aclose()
    concernees = [x for x in out if x["consigne_manquait"]]
    per = sum(x["A"] and not x["A+"] for x in concernees)
    gag = sum(x["A+"] and not x["A"] for x in concernees)
    res = {"questions": len(out), "consigne_manquait": len(concernees),
           "A": sum(x["A"] for x in out), "A+": sum(x["A+"] for x in out),
           "perdues": per, "gagnees": gag, "p": round(_p(per, gag), 4), "details": out}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n{res['questions']} questions ask_to_forget, consigne manquante pour {res['consigne_manquait']} ;"
          f" A {res['A']} -> A+ {res['A+']} (perdues {per}, gagnées {gag}, p = {res['p']})")
    print(f"Rapport : {args.output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Demandes d'oubli : simuler la consigne dans le contexte")
    parser.add_argument("--personas", type=int, default=70)
    parser.add_argument("--graine", type=int, default=7)
    parser.add_argument("--output", type=Path, default=Path("bench/results/oubli/injection_consigne.json"))
    parser.add_argument("--cache", type=Path,
                        default=Path("bench/results/oubli/injection_consigne.cache.jsonl"))
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(run(args))


if __name__ == "__main__":
    main()

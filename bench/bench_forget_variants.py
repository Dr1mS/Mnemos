"""Oubli : quelle variante aide vraiment le répondeur ? (expérience du 29-30/09)

Contexte (commit 1c42ec4) : l'oubli tel qu'implémenté en f412fda — supprimer la
consigne, l'accusé de réception ET les échos antérieurs de l'assistant — fait
tomber la fuite de 68 % à 16 %, mais l'exactitude des questions `ask_to_forget`
passe de 26,7 % à 18,7 %. Hypothèse : la consigne présente dans le contexte
était un SIGNAL NÉGATIF pour le répondeur (« please forget that I enjoy hearty
pasta » dit quoi éviter) ; l'indicateur de fuite la comptait à tort comme une
fuite parce qu'elle contient les mots de la préférence, alors qu'elle la nie.

Quatre variantes, rejouées sur les MÊMES candidats et le MÊME répondeur :

  A  rien supprimé
  B  consigne + accusé + échos assistant supprimés   (f412fda, code actuel)
  C  consigne gardée ; accusé + échos supprimés
  D  consigne + accusé gardés ; échos supprimés       (« garder ce qui nie,
                                                        supprimer ce qui affirme »)

Méthode :
* une seule ingestion, SANS oubli (lancer avec PYTHONPATH vers un `src/` sans
  oubli, p. ex. un worktree à 436cb75) — un garde-fou vérifie que les consignes
  ont bien été stockées ;
* chaque variante est rejouée comme la production la verrait (`classer` :
  supprimés retirés du KNN, 200 plus proches, score déployé, top 100), puis
  dédupliquée par contenu comme la route /search — fidélité déjà vérifiée : la
  simulation prédisait 16 % de fuite, le vrai code a donné 16 % ;
* questions à choix multiples, options et prompt identiques à
  `bench_personamem_qa` (même graine, même ordre) : sur la moitié validation,
  A et B doivent retrouver les 26,7 % et 18,7 % mesurés par le vrai code ;
* deux fuites : « brute » (l'ancienne, qui compte la consigne et l'accusé) et
  « affirmative » (qui ne compte que ce qui présente la préférence comme vraie).

Le rapport Markdown est réécrit au fil de l'eau : un arrêt en cours de route
laisse des résultats lisibles et équilibrés (les quatre variantes sont jouées
question par question).

    PYTHONPATH=<worktree 436cb75>/src EMBED_BACKEND=llamacpp \\
        python bench/bench_forget_variants.py
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
import tempfile
import time
from math import comb
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ.setdefault("EPISODIC_RETENTION_DAYS", "36500")
os.environ.setdefault("DECAY_RATE_DAILY", "0.0")

from bench.bench_forget_targeting import Persona, _mots, classer, collecter, porte, supprimes
from bench.bench_personamem import parse_query
from bench.bench_personamem_qa import ANSWER_PROMPT, LETTERS, build_options, parse_letter
from mnemos.config import Settings
from mnemos.llm.ollama_client import OllamaClient

ECHOS = {"seuil": 0.65, "roles": {"assistant"}}
VARIANTES: dict[str, dict[str, Any]] = {
    "A rien": {"consigne": False, "accuse": False},
    "B tout supprimer (f412fda)": {"consigne": True, "accuse": True, **ECHOS},
    "C garder la consigne": {"consigne": False, "accuse": True, **ECHOS},
    "D garder consigne + accusé": {"consigne": False, "accuse": False, **ECHOS},
}
# Date affichée dans le contexte : celle de l'ingestion des vrais runs du 29/09,
# pour que les prompts de validation soient identiques à ceux du vrai code.
DATE = "2026-09-29"


def _negations(p: Persona) -> set[int]:
    """Messages qui NIENT la préférence : les consignes et leurs accusés."""
    out = {c.idx for c in p.consignes}
    out.update(c.accuse for c in p.consignes if c.accuse is not None)
    return out


def _mcnemar(perdues: int, gagnees: int) -> float:
    n, k = perdues + gagnees, min(perdues, gagnees)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0


def _pct(xs: list[dict[str, Any]]) -> str:
    return f"{100 * sum(r['correct'] for r in xs) / len(xs):.1f} % ({len(xs)})" if xs else "—"


def _cle(r: dict[str, Any]) -> tuple[str, int]:
    return (r["persona"], r["q"])


def _rapport(records: list[dict[str, Any]], chemin: Path, meta: dict[str, Any]) -> None:
    lignes = [
        "# Oubli : quelle variante aide le répondeur ?",
        "",
        f"Généré le {time.strftime('%d/%m/%Y %H:%M')} — {len(records)} réponses"
        f" ({len(records) // len(VARIANTES)} questions × {len(VARIANTES)} variantes).",
        f"Répondeur `{meta['answer_model']}`, num_ctx {meta['num_ctx']}, top_k 100.",
        "",
        "A rien · B tout supprimer (f412fda) · C garder la consigne · D garder consigne + accusé",
        "",
    ]
    for moitie in ("validation", "calibration", "total"):
        rs = [r for r in records if moitie == "total" or r["moitie"] == moitie]
        if not rs:
            continue
        lignes += [f"## {moitie}", "",
                   "| variante | ask_to_forget | autres | global | fuite brute | fuite affirmative | tronqués |",
                   "|---|---|---|---|---|---|---|"]
        for v in VARIANTES:
            rv = [r for r in rs if r["variante"] == v]
            f = [r for r in rv if r["pref_type"] == "ask_to_forget"]
            o = [r for r in rv if r["pref_type"] != "ask_to_forget"]
            fb = f"{100 * sum(r['fuite_brute'] for r in f) / len(f):.0f} %" if f else "—"
            fa = f"{100 * sum(r['fuite_affirmative'] for r in f) / len(f):.0f} %" if f else "—"
            lignes.append(f"| {v} | {_pct(f)} | {_pct(o)} | {_pct(rv)} | {fb} | {fa} |"
                          f" {sum(r['tronque'] for r in rv)} |")
        # Appariement contre A, question par question
        lignes += ["", "Appariement contre A (perdues / gagnées, McNemar exact) :", ""]
        base = {_cle(r): r for r in rs if r["variante"] == "A rien"}
        for v in list(VARIANTES)[1:]:
            for type_, filtre in (("ask_to_forget", lambda r: r["pref_type"] == "ask_to_forget"),
                                  ("autres", lambda r: r["pref_type"] != "ask_to_forget")):
                paires = [(base[_cle(r)], r) for r in rs
                          if r["variante"] == v and filtre(r) and _cle(r) in base]
                p = sum(a["correct"] and not b["correct"] for a, b in paires)
                g = sum(b["correct"] and not a["correct"] for a, b in paires)
                lignes.append(f"- {v}, {type_} : perdues {p}, gagnées {g}, p = {_mcnemar(p, g):.2f}")
        lignes.append("")
    if meta.get("fidelite"):
        lignes += ["## Contrôle de fidélité (validation)", "", meta["fidelite"], ""]
    chemin.write_text("\n".join(lignes), encoding="utf-8")


async def main_async(args: argparse.Namespace) -> None:
    cache = Path(tempfile.gettempdir()) / f"mnemos_forget_variants_sans_oubli_{args.personas}.pkl"
    personas = await collecter(args.personas, cache)
    stockes = 0
    for p in personas:
        indexes = set(p.idx_par_id.values())
        stockes += sum(1 for c in p.consignes if c.idx in indexes)
    total = sum(len(p.consignes) for p in personas)
    print(f"Consignes stockées : {stockes}/{total}", flush=True)
    if stockes < 0.95 * total:
        raise SystemExit("ARRÊT : les consignes n'ont pas été stockées — la collecte a tourné AVEC "
                         "l'oubli. Relancer avec PYTHONPATH vers un src/ sans oubli (436cb75).")

    llm = OllamaClient(Settings(_env_file=None))  # type: ignore[call-arg]
    opts = {"temperature": 0.0, "num_ctx": args.num_ctx, "num_predict": 8}
    await llm.generate("Réponds OK.", args.answer_model, options={"num_predict": 4})
    meta: dict[str, Any] = {"answer_model": args.answer_model, "num_ctx": args.num_ctx}
    records: list[dict[str, Any]] = []
    deja: dict[str, str | None] = {}  # prompt -> lettre : le répondeur est déterministe
    t0 = time.perf_counter()
    try:
        # Validation d'abord : c'est elle qui se compare aux vrais runs du 29/09.
        for moitie, parite in (("validation", 1), ("calibration", 0)):
            rng = random.Random(42)  # même graine et même ordre que bench_personamem_qa
            groupe = [p for p in personas if int(p.pid) % 2 == parite]
            suppr = {p.pid: {v: supprimes(p, r) for v, r in VARIANTES.items()} for p in groupe}
            for p in groupe:
                neg = _negations(p)
                for q, (row, cands) in enumerate(p.questions):
                    if not row.get("correct_answer") or not row.get("incorrect_answers"):
                        continue
                    options, gold = build_options(row, rng)
                    if len(options) < 2:
                        continue
                    question = parse_query(row["user_query"])
                    mp = _mots(row["prev_pref"]) if row["pref_type"] == "ask_to_forget" else set()
                    for v in VARIANTES:
                        classe = classer(cands, suppr[p.pid][v])
                        vus: set[str] = set()
                        contexte: list[int] = []
                        for i in classe:  # déduplication par contenu, comme la route /search
                            k = p.messages[i]["content"].lower()
                            if k not in vus:
                                vus.add(k)
                                contexte.append(i)
                        ctx = "\n".join(f"{n}. [{DATE}] {p.messages[i]['content']}"
                                        for n, i in enumerate(contexte, 1))
                        prompt = ANSWER_PROMPT.format(
                            context=ctx, question=question,
                            options="\n".join(f"{LETTERS[n]}. {o}" for n, o in enumerate(options)))
                        if prompt not in deja:
                            deja[prompt] = parse_letter(
                                await llm.generate(prompt, args.answer_model, options=opts))
                        lettre = deja[prompt]
                        top10 = contexte[:10]
                        records.append({
                            "moitie": moitie, "variante": v, "persona": p.pid, "q": q,
                            "pref_type": row["pref_type"], "gold": gold, "answer": lettre,
                            "correct": lettre == gold, "tronque": len(prompt) > args.num_ctx * 4,
                            "fuite_brute": bool(mp) and any(porte(p.messages[i]["content"], mp)
                                                            for i in top10),
                            "fuite_affirmative": bool(mp) and any(
                                porte(p.messages[i]["content"], mp) for i in top10 if i not in neg),
                        })
                    n_q = len(records) // len(VARIANTES)
                    if n_q % 10 == 0:
                        print(f"  {moitie} : {n_q} questions, {len(deja)} appels LLM,"
                              f" {time.perf_counter() - t0:.0f} s", flush=True)
                        _rapport(records, args.rapport, meta)
            if moitie == "validation":
                f = {v: [r for r in records if r["moitie"] == "validation" and r["variante"] == v
                         and r["pref_type"] == "ask_to_forget"] for v in VARIANTES}
                acc = {v: 100 * sum(r["correct"] for r in xs) / len(xs) for v, xs in f.items() if xs}
                meta["fidelite"] = (
                    f"Vrai code, 29/09 : sans oubli 26,7 %, avec l'oubli f412fda 18,7 % sur les 75 questions"
                    f" ask_to_forget. Simulation : A {acc.get('A rien', 0):.1f} %,"
                    f" B {acc.get('B tout supprimer (f412fda)', 0):.1f} %. Un écart notable signifierait"
                    f" que la simulation ne reproduit pas la production et que les variantes C et D ne"
                    f" peuvent pas être lues.")
    finally:
        await llm.aclose()
        _rapport(records, args.rapport, meta)
        args.output.write_text(json.dumps({"meta": meta, "records": records}, indent=1,
                                          ensure_ascii=False), encoding="utf-8")
    print(f"Terminé : {len(records)} réponses. Rapport : {args.rapport}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Oubli : quelle variante aide le répondeur ?")
    parser.add_argument("--personas", type=int, default=100)
    parser.add_argument("--answer-model", default="qwen2.5:7b-instruct-q4_K_M")
    parser.add_argument("--num-ctx", type=int, default=24576)
    parser.add_argument("--output", type=Path, default=Path("bench/results/gpu/forget_variants.json"))
    parser.add_argument("--rapport", type=Path, default=Path("bench/results/gpu/forget_variants.md"))
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()

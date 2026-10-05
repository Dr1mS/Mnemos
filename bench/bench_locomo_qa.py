"""Évaluation question → réponse → jugement sur LoCoMo, avec et sans faits consolidés.

Mesure ce que les benchs de récupération ne peuvent pas voir : les faits extraits
par le LLM aident-ils à *répondre* ? Même ingestion et même consolidation pour
les deux contextes, seule la présence des faits change :
- A (épisodique) : les top_k épisodes de la recherche hybride ;
- B (complet) : la réponse de POST /search, faits consolidés compris.
Un modèle local répond à partir de chaque contexte, puis juge chaque réponse
contre la référence LoCoMo. Les questions adversariales (catégorie 5, sans
réponse de référence) sont exclues. L'oubli temporel est neutralisé comme dans
le déploiement AML (les messages portent des dates de 2023).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
import shutil
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Avant tout import de Settings : pas d'archivage des messages datés de 2023.
os.environ.setdefault("EPISODIC_RETENTION_DAYS", "36500")
os.environ.setdefault("DECAY_RATE_DAILY", "0.0")

import httpx

from bench import aml_officiel
from bench.bench_locomo import drain_consolidation, parse_locomo_datetime, setup_bench_app
from bench.remote_llm import RemoteChat
from mnemos.config import Settings
from mnemos.llm.json_cleaner import parse_llm_json
from mnemos.llm.ollama_client import OllamaClient

ANSWER_PROMPT = """You answer questions about a long conversation between two people,
using only the memories retrieved below (most relevant first, with their date).

Memories:
{context}

Question: {question}
Answer in a few words. If the memories do not contain the answer, say "unknown"."""

JUDGE_PROMPT = """Grade a candidate answer against the gold answer to a question.
The candidate is correct if it conveys the same information as the gold answer
(paraphrases, partial dates matching the gold date, and extra detail are fine).
It is wrong if it contradicts the gold answer, misses its key point, or says unknown.

Question: {question}
Gold answer: {gold}
Candidate answer: {candidate}

Reply with JSON only: {{"correct": true}} or {{"correct": false}}"""


def _evidence_ids(q: dict[str, Any]) -> list[str]:
    """Identifiants des tours-preuves (`D3:12`). Le jeu en contient quelques-uns
    groupés dans une seule chaîne (`"D8:6; D9:17"`) : on les sépare."""
    ids: list[str] = []
    for ev in q.get("evidence") or []:
        ids.extend(p.strip() for p in str(ev).replace(",", ";").split(";") if p.strip())
    return ids


def _evidence_ranks(contents: list[str], ids: list[str]) -> list[int | None]:
    """Rang (1-based) de chaque tour-preuve dans le contexte rendu, None s'il en
    est absent. Les épisodes sont écrits `[D3:12] Locuteur: texte`."""
    return [
        next((i for i, c in enumerate(contents, 1) if c.startswith(f"[{ev}]")), None)
        for ev in ids
    ]


# Identifiant de tour que ce bench préfixe pour relever les rangs des preuves ;
# retiré du texte montré au répondeur sous consignes officielles.
_DIA = re.compile(r"^\[D\d+:\d+\]\s*")


def _format_context(items: list[tuple[str, str]]) -> str:
    return "\n".join(f"{i}. [{date}] {content}" for i, (date, content) in enumerate(items, 1))


def _iso_day(ts_ms: int) -> str:
    return datetime.fromtimestamp(ts_ms / 1000, tz=UTC).date().isoformat()


async def run_qa_bench(
    n_questions: int,
    seed: int,
    top_k: int,
    answer_model: str,
    output: Path,
    user_id: str = "locomo_qa",
    with_facts: bool = True,
    answer_backend: str = "ollama",
    consignes: str = "mnemos",
) -> dict[str, Any]:
    # Consignes officielles (bench/aml_officiel.py) : bras de production seul,
    # rendu avec dates (A) et sans dates (S) ; pas de bras « faits ».
    officiel = aml_officiel.charger("locomo-refined") if consignes == "officielles" else None
    if officiel is not None and with_facts:
        raise SystemExit("--consignes officielles mesure la production : ajouter --sans-faits")
    data = json.loads(Path("bench/data/locomo_sample.json").read_text(encoding="utf-8"))
    conv = data["conversation"]
    pool = [q for q in data["qa"] if q.get("category") != 5 and q.get("answer") is not None]
    questions = random.Random(seed).sample(pool, min(n_questions, len(pool)))

    llm = OllamaClient(Settings(_env_file=None))  # type: ignore[call-arg]
    # Répondeur ET juge : le même modèle, local ou distant (LoCoMo est public).
    # L'extracteur des faits reste local, comme en production.
    repondeur: Any = llm if answer_backend == "ollama" else RemoteChat(answer_backend)
    llm_opts = {"temperature": 0.0, "num_ctx": 8192, "num_predict": 64}

    async def ask(prompt: str, fmt: str | None = None) -> str:
        return await repondeur.generate(prompt, answer_model, format=fmt, options=llm_opts)  # type: ignore[no-any-return]

    # Pipelines officiels : max_tokens 256 pour la réponse comme pour le juge, qui
    # écrit une phrase d'explication avant son label.
    opts_officiels = {"temperature": 0.0, "num_ctx": 16384, "num_predict": 256}

    async def ask_officiel(prompt: str) -> str:
        return await repondeur.generate(prompt, answer_model, options=opts_officiels)  # type: ignore[no-any-return]

    # Vol d'essai : un runner Ollama en mauvais état fait retomber la saillance
    # sur ses valeurs par défaut sans rien arrêter (vu le 29/09). Mieux vaut ne
    # pas démarrer que mesurer un système dégradé.
    await repondeur.generate("Réponds OK.", answer_model, options={"num_predict": 4})
    if with_facts:
        await llm.generate("Réponds OK.", "qwen2.5:3b", options={"num_predict": 4})

    tmp = Path(tempfile.mkdtemp(prefix="mnemos_locomo_qa_"))
    try:
        # Sans faits : ni saillance ni consolidation. Le contexte A n'en dépend pas
        # (la recherche ignore la saillance, la consolidation ne modifie ni
        # n'archive aucun épisode) ; on n'économise que du temps GPU.
        app, _, _ = await setup_bench_app(tmp, "ollama", salience_workers=2 if with_facts else 0)
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://t", timeout=600) as client:
                # 1. Ingestion (même format que bench_locomo) puis consolidation complète
                sessions = sorted(
                    (k for k in conv if k.startswith("session_") and not k.endswith("_date_time")),
                    key=lambda k: int(k.split("_")[1]),
                )
                for s_key in sessions:
                    base_ms = parse_locomo_datetime(conv.get(f"{s_key}_date_time"))
                    messages = [
                        {
                            "role": "user" if t.get("speaker") == conv.get("speaker_a") else "assistant",
                            "content": f"[{t.get('dia_id')}] {t.get('speaker')}: {t.get('text', '')}",
                            "timestamp": base_ms + i * 15000,
                        }
                        for i, t in enumerate(conv[s_key])
                    ]
                    resp = await client.post("/add", json={
                        "request_id": s_key, "user_id": user_id, "session_id": s_key, "messages": messages,
                    })
                    resp.raise_for_status()
                t0 = time.perf_counter()
                if with_facts:
                    await app.state.queue.join()
                    cons = await drain_consolidation(app)
                else:
                    cons = {"facts_inserted": 0, "extraction_failures": 0}
                # Débit de la consolidation (saillance + extraction), sans charge
                # d'ingestion concurrente : c'est un plafond. Le Full en a ingéré
                # ~4,9 épisodes/s pendant 25 h ; en dessous, les faits d'une tâche
                # n'existent pas encore quand sa recherche arrive.
                n_episodes = sum(len(conv[s]) for s in sessions)
                duree_cons = time.perf_counter() - t0
                debit: dict[str, Any] = {
                    "episodes": n_episodes, "duree_s": round(duree_cons, 1),
                    "episodes_par_s": round(n_episodes / duree_cons, 2) if with_facts else None,
                }
                print(f"Consolidation : {cons['facts_inserted']} faits en {duree_cons:.0f} s"
                      f" | {n_episodes} épisodes, {debit['episodes_par_s']} épisodes/s"
                      f" | échecs d'extraction : {cons['extraction_failures']}")

                # 2. Rangs des tours-preuves, sur TOUTES les questions éligibles.
                # Aucune génération : la mesure ne dépend ni du répondeur ni du
                # juge, et elle coûte des recherches, pas des appels LLM. C'est
                # l'instrument de non-régression du classement ; le taux de
                # réponses justes, sur 50 questions et un juge local, est trop
                # bruité pour trancher seul.
                rangs: list[dict[str, Any]] = []
                for q in pool:
                    ids = _evidence_ids(q)
                    if not ids:
                        continue
                    eps = await app.state.store.search(q["question"], k=top_k, tenant=user_id)
                    items = (await client.post("/search", json={
                        "query": q["question"], "user_id": user_id, "top_k": top_k,
                    })).json()["data"]
                    rangs.append({
                        "question": q["question"], "category": q["category"], "evidence": ids,
                        "ranks_A": _evidence_ranks([e.episode.content for e in eps], ids),
                        "ranks_B": _evidence_ranks([it["content"] for it in items], ids),
                    })
                print(f"Rangs des preuves relevés sur {len(rangs)} questions")

                # 3. Deux contextes par question, réponse puis jugement
                records: list[dict[str, Any]] = []
                for n, q in enumerate(questions, 1):
                    episodes = await app.state.store.search(q["question"], k=top_k, tenant=user_id)
                    ctx_a = [(_iso_day(e.episode.created_at), e.episode.content) for e in episodes]
                    resp = await client.post("/search", json={
                        "query": q["question"], "user_id": user_id, "top_k": top_k,
                    })
                    items = resp.json()["data"]
                    ctx_b = [(str(it.get("created_at", ""))[:10], it["content"]) for it in items]
                    fact_ranks = [i for i, it in enumerate(items, 1) if it["id"].startswith("fact_")]

                    row: dict[str, Any] = {
                        "question": q["question"], "gold": str(q["answer"]), "category": q["category"],
                        "facts_in_context": len(fact_ranks),
                        "first_fact_rank": fact_ranks[0] if fact_ranks else None,
                    }
                    if officiel is not None:
                        # Consignes officielles : le contexte est la réponse de la route
                        # /search, rendue par locuteur comme le gabarit de la plateforme,
                        # avec (A) ou sans (S) created_at — on ignore lequel elle montre.
                        souvenirs = [(str(it.get("created_at", "")), _DIA.sub("", it["content"]))
                                     for it in items]
                        locuteurs = [(nom, [s for s in souvenirs if s[1].startswith(f"{nom}:")])
                                     for nom in (conv["speaker_a"], conv["speaker_b"])]
                        for label, avec_dates in (("A", True), ("S", False)):
                            answer = (await ask_officiel(aml_officiel.consigne_reponse(
                                officiel, q["question"], locuteurs, avec_dates))).strip()
                            sortie = await ask_officiel(aml_officiel.consigne_juge(
                                officiel, q["question"], str(q["answer"]), answer))
                            juge = aml_officiel.verdict(officiel, sortie)
                            row[f"answer_{label}"] = answer
                            row[f"correct_{label}"] = bool(juge)
                            row[f"juge_illisible_{label}"] = juge is None
                        records.append(row)
                        print(f"  [{n}/{len(questions)}] cat {q['category']}"
                              f" | avec dates {'✓' if row['correct_A'] else '✗'}"
                              f" | sans dates {'✓' if row['correct_S'] else '✗'}", flush=True)
                        continue
                    for label, ctx in (("A", ctx_a), ("B", ctx_b))[: 2 if with_facts else 1]:
                        answer = (await ask(ANSWER_PROMPT.format(
                            context=_format_context(ctx), question=q["question"],
                        ))).strip()
                        verdict = parse_llm_json(await ask(
                            JUDGE_PROMPT.format(question=q["question"], gold=q["answer"], candidate=answer),
                            fmt="json",
                        ))
                        row[f"answer_{label}"] = answer
                        row[f"correct_{label}"] = bool(isinstance(verdict, dict) and verdict.get("correct"))
                    records.append(row)
                    b_txt = (f"B={'✓' if row['correct_B'] else '✗'} | faits dans B : {row['facts_in_context']}"
                             if with_facts else "")
                    print(f"  [{n}/{len(questions)}] cat {q['category']} | A={'✓' if row['correct_A'] else '✗'} {b_txt}")
    finally:
        await llm.aclose()
        if repondeur is not llm:
            await repondeur.aclose()
        shutil.rmtree(tmp, ignore_errors=True)

    n = len(records)
    # Second bras : B (avec faits) sous nos consignes, S (sans dates) sous les officielles.
    bras = ("A", "S") if officiel is not None else (("A", "B") if with_facts else ("A",))
    second = bras[1] if len(bras) > 1 else None
    acc = {lab: sum(r[f"correct_{lab}"] for r in records) / n for lab in bras}
    only_a = sum(r["correct_A"] and not r[f"correct_{second}"] for r in records) if second else None
    only_b = sum(r[f"correct_{second}"] and not r["correct_A"] for r in records) if second else None
    by_cat: dict[int, dict[str, float]] = {}
    for c in sorted({r["category"] for r in records}):
        rows = [r for r in records if r["category"] == c]
        by_cat[c] = {
            "count": len(rows),
            **{lab: sum(r[f"correct_{lab}"] for r in rows) / len(rows) for lab in bras},
        }
    import mnemos
    from mnemos.stores import episodic as _episodic

    def _resume_rangs(lab: str, rows: list[dict[str, Any]]) -> dict[str, float]:
        """Rappel des preuves dans le top_k, meilleur rang, et part des questions
        dont au moins une preuve est dans le top 10."""
        tous = [r for row in rows for r in row[f"ranks_{lab}"]]
        meilleurs = [min((r for r in row[f"ranks_{lab}"] if r is not None), default=None)
                     for row in rows]
        trouves = sorted(m for m in meilleurs if m is not None)
        return {
            "questions": len(rows),
            "rappel_preuves": round(sum(r is not None for r in tous) / len(tous), 4) if tous else 0.0,
            "hit_at_10": round(sum(1 for m in trouves if m <= 10) / len(rows), 4) if rows else 0.0,
            "meilleur_rang_median": trouves[len(trouves) // 2] if trouves else None,
        }

    rangs_par_cat = {
        c: {lab: _resume_rangs(lab, [r for r in rangs if r["category"] == c]) for lab in ("A", "B")}
        for c in sorted({r["category"] for r in rangs})
    }
    result = {
        "config": {"questions": n, "seed": seed, "top_k": top_k, "answer_and_judge_model": answer_model,
                   "answer_backend": answer_backend, "consolidation": {**cons, **debit},
                   "consignes": consignes,
                   "aml_pipeline_commit": ((aml_officiel.RACINE / "COMMIT").read_text().strip()
                                           if officiel is not None and (aml_officiel.RACINE / "COMMIT").exists()
                                           else None),
                   "juge_illisible": (sum(r.get(f"juge_illisible_{lab}", False) for r in records
                                          for lab in bras) if officiel is not None else None),
                   "user_id": user_id, "with_facts": with_facts,
                   "embed_backend": os.environ.get("EMBED_BACKEND", "ollama"),
                   "extraction_model": "qwen2.5:3b", "facts_inserted": cons["facts_inserted"],
                   "extraction_failures": cons["extraction_failures"],
                   # Empreinte du code réellement importé : un avant/après n'a de
                   # valeur que si l'on peut prouver quel `src/` a tourné.
                   "mnemos_src": str(Path(mnemos.__file__).parent),
                   "recency_weight": _episodic.RECENCY_WEIGHT},
        "accuracy": acc, "only_A_correct": only_a, "only_B_correct": only_b,
        "by_category": by_cat,
        "evidence_ranks": {"all": {lab: _resume_rangs(lab, rangs) for lab in ("A", "B")},
                           "by_category": rangs_par_cat},
        "records": records, "rank_records": rangs,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    n_a = sum(r["correct_A"] for r in records)
    nom_a = "avec dates" if officiel is not None else "épisodique"
    nom_second = {"B": "avec faits", "S": "sans dates"}.get(second or "", "")
    b_acc = (f"  |  {second} ({nom_second}) : {acc[second] * 100:.1f} %" if second
             else "  (bras B non mesuré)")
    print(f"\nConsignes : {consignes}")
    print(f"Exactitude A ({nom_a}) : {acc['A'] * 100:.1f} % ({n_a}/{n}){b_acc}")
    if second:
        print(f"Désaccords : seul A juste {only_a} | seul {second} juste {only_b}  (sur {n} questions)")
    for c, m in by_cat.items():
        b_cat = f" | {second} {m[second] * 100:.0f} %" if second else ""
        print(f"  catégorie {c} (N={m['count']}) : A {m['A'] * 100:.0f} %{b_cat}")
    print("\nRangs des tours-preuves (A = épisodique) — indépendant du répondeur :")
    for c, par in [("toutes", result["evidence_ranks"]["all"]), *rangs_par_cat.items()]:
        a = par["A"]
        print(f"  {str(c):7s} N={a['questions']:3d} | rappel@{top_k} {a['rappel_preuves']:.3f}"
              f" | hit@10 {a['hit_at_10']:.3f} | meilleur rang médian {a['meilleur_rang_median']}")
    print(f"Code importé : {result['config']['mnemos_src']} (RECENCY_WEIGHT={_episodic.RECENCY_WEIGHT:.4f})")
    print(f"Rapport : {output}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="LoCoMo : exactitude des réponses avec et sans faits")
    parser.add_argument("--questions", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--answer-model", default="qwen3.5:9b")
    parser.add_argument("--answer-backend", choices=("ollama", "nvidia", "mistral"), default="ollama",
                        help="ollama = local ; nvidia / mistral = API distante (LoCoMo est public)")
    parser.add_argument("--output", type=Path, default=Path("bench/results/classement/locomo_qa.json"))
    # Le user_id sert de sujet des faits extraits (canonical_subject) : un identifiant
    # opaque comme ceux d'AML ("eval:run:…") pollue les faits et leur embedding.
    parser.add_argument("--user-id", default="locomo_qa")
    # Le bras B (faits consolidés) coûte la consolidation et deux appels LLM de plus
    # par question, pour une configuration que la production n'expédie pas.
    parser.add_argument("--sans-faits", action="store_true",
                        help="mesurer seulement le bras A (épisodique), comme en production")
    # Consignes du répondeur et du juge de la plateforme, lues depuis son dépôt public
    # (bench/aml_officiel.py). Exige --sans-faits ; mesure avec et sans dates affichées.
    parser.add_argument("--consignes", choices=("mnemos", "officielles"), default="mnemos")
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # console Windows cp1252
    if sys.platform == "win32":
        # 02/10 : une coupure réseau (WinError 64) a cassé le canal de réveil de la
        # boucle Proactor ; elle ne le réarme pas après une erreur, et le bench est
        # resté figé 30 min sans rien dire. La boucle Selector garde ce canal
        # inscrit en permanence.
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(run_qa_bench(
        args.questions, args.seed, args.top_k, args.answer_model, args.output, args.user_id,
        with_facts=not args.sans_faits, answer_backend=args.answer_backend,
        consignes=args.consignes,
    ))


if __name__ == "__main__":
    main()

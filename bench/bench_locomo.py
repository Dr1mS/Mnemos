"""Benchmark LoCoMo pour Mnemos (Agent Memory Challenge — Textual Track).

Évalue la tenue de charge et la précision de récupération sur le dataset LoCoMo :
- Ingestion des sessions multi-tours avec horodatages réels via POST /add
- Mesure du débit et des latences d'ingestion
- Évaluation de rappel (Recall@1, Recall@3, Recall@5, Recall@10) sur les questions QA
- Décomposition par catégorie cognitive (Factuel 1-hop, Raisonnement temporel, Multi-hop)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import shutil
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx
from sqlalchemy import text
from tests.conftest import StubEmbedder, StubLLMManager

from mnemos.clock import Clock
from mnemos.config import Settings
from mnemos.models.base import make_async_engine
from mnemos.models.episodic import EPISODIC_SCHEMA_SQL
from mnemos.models.semantic import SEMANTIC_SCHEMA_SQL
from mnemos.server import create_app
from mnemos.stores.episodic import EpisodicStore
from mnemos.stores.semantic import SemanticStore
from mnemos.tagger.salience import SalienceTagger, ScoringQueue


def parse_locomo_datetime(dt_str: str | None) -> int:
    """Convertit un datetime LoCoMo (ex: '1:56 pm on 8 May, 2023') en epoch ms."""
    if not dt_str:
        return int(time.time() * 1000)
    match = re.search(
        r"(\d+):(\d+)\s*(am|pm)\s*on\s*(\d+)\s*([A-Za-z]+),\s*(\d{4})", dt_str, re.IGNORECASE
    )
    if not match:
        return int(time.time() * 1000)
    hour, minute, ampm, day, month, year = match.groups()
    h = int(hour)
    if ampm.lower() == "pm" and h < 12:
        h += 12
    elif ampm.lower() == "am" and h == 12:
        h = 0
    try:
        dt = datetime.strptime(f"{year}-{month}-{int(day):02d} {h:02d}:{minute}:00", "%Y-%B-%d %H:%M:%S")
        return int(dt.timestamp() * 1000)
    except Exception:
        return int(time.time() * 1000)


def build_turn_map(conversation: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Indexe les tours de dialogue par leur dia_id (ex: 'D1:3')."""
    turn_map: dict[str, dict[str, Any]] = {}
    session_keys = [
        k for k in conversation if k.startswith("session_") and not k.endswith("_date_time")
    ]
    for s_name in sorted(session_keys, key=lambda x: int(x.split("_")[1])):
        s_idx = int(s_name.split("_")[1])
        dt_str = conversation.get(f"{s_name}_date_time")
        base_ms = parse_locomo_datetime(dt_str)
        for turn_idx, turn in enumerate(conversation[s_name]):
            dia_id = turn.get("dia_id", f"D{s_idx}:{turn_idx+1}")
            turn_map[dia_id] = {
                "session": s_idx,
                "speaker": turn.get("speaker", "unknown"),
                "text": turn.get("text", ""),
                "dia_id": dia_id,
                "timestamp": base_ms + (turn_idx * 15000),  # +15s par tour
            }
    return turn_map


def extract_evidence_ids(raw_evidence: Any) -> list[str]:
    """Nettoie et extrait la liste des dia_id depuis le champ evidence."""
    if not raw_evidence:
        return []
    if isinstance(raw_evidence, list):
        items: list[str] = []
        for it in raw_evidence:
            for sub in re.split(r"[;,]\s*", str(it)):
                if sub.strip():
                    items.append(sub.strip())
        return items
    if isinstance(raw_evidence, str):
        return [s.strip() for s in re.split(r"[;,]\s*", raw_evidence) if s.strip()]
    return []


async def setup_bench_app(
    tmp_path: Path, mode: str, llm_model: str = "qwen2.5:3b", salience_workers: int = 0
) -> tuple[Any, list[Any], Settings]:
    """Prépare l'application Mnemos pour le benchmark (stub ou ollama réel)."""
    settings = Settings(
        _env_file=None,
        DATA_DIR=tmp_path,
        EPISODIC_DB=tmp_path / "episodic.db",
        SEMANTIC_DB=tmp_path / "semantic.db",
        PROCEDURAL_DIR=tmp_path / "procedural",
        SALIENCE_MODEL=llm_model,
        EXTRACTION_MODEL=llm_model,
        SALIENCE_QUEUE_WORKERS=salience_workers,
        CONSOLIDATION_DELAY_HOURS=0.0,
        CONSOLIDATION_BATCH_SIZE=50,
    )
    clock = Clock()
    epi_engine = make_async_engine(settings.EPISODIC_DB)
    sem_engine = make_async_engine(settings.SEMANTIC_DB)

    async with epi_engine.begin() as conn:
        for stmt in EPISODIC_SCHEMA_SQL:
            await conn.execute(text(stmt))
    async with sem_engine.begin() as conn:
        for stmt in SEMANTIC_SCHEMA_SQL:
            await conn.execute(text(stmt))

    app = create_app(settings)

    if mode == "stub":
        embedder = StubEmbedder()
        manager = StubLLMManager()
        store = EpisodicStore(epi_engine, embedder, clock, settings)  # type: ignore[arg-type]
        app.state.engine = epi_engine
        app.state.semantic_engine = sem_engine
        app.state.manager = manager
        app.state.store = store
        app.state.semantic = SemanticStore(sem_engine, embedder, clock, settings)  # type: ignore[arg-type]
        app.state.queue = ScoringQueue(
            SalienceTagger(manager, settings), store  # type: ignore[arg-type]
        )
        return app, [epi_engine, sem_engine], settings
    else:
        # En mode ollama, le lifespan de create_app() gère son propre cycle de vie des engines et de ModelManager
        await epi_engine.dispose()
        await sem_engine.dispose()
        return app, [], settings


CATEGORY_NAMES = {
    1: "Single-hop Factuel",
    2: "Raisonnement Temporel",
    3: "Multi-hop & Inférence",
    4: "Domaine Ouvert / Contexte",
    5: "Gouvernance & Préférences",
}


async def run_locomo_bench(
    dataset_path: Path,
    mode: str = "stub",
    llm_model: str = "qwen2.5:3b",
    salience_workers: int = 0,
    with_consolidation: bool = False,
    grounded_only: bool = False,
    limit_sessions: int | None = None,
    limit_queries: int | None = None,
    output_report: Path | None = None,
) -> dict[str, Any]:
    """Exécute le protocole complet sur le dataset LoCoMo."""
    with dataset_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    conv = data["conversation"]
    qa_list = data["qa"]
    turn_map = build_turn_map(conv)

    user_id = "caroline_melanie"
    tmp_dir = Path(tempfile.mkdtemp(prefix="mnemos_locomo_"))

    try:
        app, engines, settings = await setup_bench_app(
            tmp_dir, mode, llm_model=llm_model, salience_workers=salience_workers
        )
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test", timeout=120.0) as client:
                print("\n=======================================================")
                print(f"🚀 Démarrage Benchmark LoCoMo (Mode: {mode.upper()})")
                print(f"  Dataset: {dataset_path.name}")
                print(f"  Participants: {conv.get('speaker_a')} & {conv.get('speaker_b')}")
                if mode == "ollama":
                    print(f"  Modèle LLM: {llm_model} (workers saillance: {salience_workers})")
                    print(f"  Consolidation sémantique: {'ACTIVÉE' if with_consolidation else 'DÉSACTIVÉE'}")
                print("=======================================================\n")

                # 1. PHASE D'INGESTION
                session_keys = [
                    k for k in conv if k.startswith("session_") and not k.endswith("_date_time")
                ]
                session_keys = sorted(session_keys, key=lambda x: int(x.split("_")[1]))
                if limit_sessions:
                    session_keys = session_keys[:limit_sessions]

                print(f"📥 Phase 1 : Ingestion de {len(session_keys)} sessions ({sum(len(conv[k]) for k in session_keys)} messages)...")
                ingest_start = time.perf_counter()
                total_messages_added = 0
                session_latencies_ms: list[float] = []

                for s_key in session_keys:
                    s_idx = int(s_key.split("_")[1])
                    turns = conv[s_key]
                    dt_str = conv.get(f"{s_key}_date_time")
                    base_ms = parse_locomo_datetime(dt_str)

                    messages = []
                    for t_idx, turn in enumerate(turns):
                        speaker = turn.get("speaker", "user")
                        text_val = turn.get("text", "")
                        dia_id = turn.get("dia_id", f"D{s_idx}:{t_idx+1}")
                        t_ms = base_ms + (t_idx * 15000)
                        messages.append({
                            "role": "user" if speaker == conv.get("speaker_a") else "assistant",
                            "content": f"[{dia_id}] {speaker}: {text_val}",
                            "timestamp": t_ms,
                        })

                    t0 = time.perf_counter()
                    resp = await client.post(
                        "/add",
                        json={
                            "request_id": f"ingest-session-{s_idx}",
                            "user_id": user_id,
                            "session_id": s_key,
                            "messages": messages,
                        },
                    )
                    lat_ms = (time.perf_counter() - t0) * 1000
                    session_latencies_ms.append(lat_ms)

                    if resp.status_code != 200:
                        print(f"❌ Erreur ingestion session {s_idx}: {resp.status_code} {resp.text}")
                    else:
                        total_messages_added += len(messages)
                        if s_idx % 2 == 0 or s_idx == len(session_keys):
                            print(f"  [Ingestion] Session {s_idx}/{len(session_keys)} ({total_messages_added} msgs) — {lat_ms:.0f} ms")

                ingest_total_sec = time.perf_counter() - ingest_start
                avg_session_lat_ms = sum(session_latencies_ms) / max(len(session_latencies_ms), 1)
                msg_per_sec = total_messages_added / max(ingest_total_sec, 0.001)

                print("✅ Ingestion terminée :")
                print(f"  - Messages persistés : {total_messages_added}")
                print(f"  - Temps total : {ingest_total_sec:.2f} s")
                print(f"  - Latence moy / session : {avg_session_lat_ms:.1f} ms")
                print(f"  - Débit d'ingestion : {msg_per_sec:.1f} messages/s\n")

                # PHASE 1.5 : CONSOLIDATION COGNITIVE (OPTIONNELLE)
                if with_consolidation and hasattr(app.state, "queue") and hasattr(app.state, "worker"):
                    print("⏳ Attente du scoring de saillance par le LLM (drain queue)...")
                    t_drain = time.perf_counter()
                    await app.state.queue.join()
                    print(f"✅ Scoring terminé en {time.perf_counter() - t_drain:.1f} s")

                    print("🧠 Lancement de la consolidation sémantique (extraction des faits)...")
                    t_cons = time.perf_counter()
                    cons_report = await app.state.worker.run_once()
                    print(f"✅ Consolidation terminée en {time.perf_counter() - t_cons:.1f} s :")
                    print(f"   • Candidats analysés : {cons_report.candidates}")
                    print(f"   • Faits sémantiques insérés : {cons_report.facts_inserted}")
                    print(f"   • Entités créées : {cons_report.entities_upserted}")
                    print(f"   • Échecs d'extraction : {cons_report.extraction_failures}\n")

                # 2. PHASE D'ÉVALUATION DE RECHERCHE
                if grounded_only and limit_sessions:
                    allowed_prefixes = tuple(f"D{i}:" for i in range(1, limit_sessions + 1))
                    eval_queries = [
                        q for q in qa_list
                        if any(ev.startswith(allowed_prefixes) for ev in extract_evidence_ids(q.get("evidence", [])))
                    ]
                    print(f"🎯 Filtrage ground-truth : {len(eval_queries)} questions associées aux {limit_sessions} sessions ingérées.")
                else:
                    eval_queries = qa_list

                if limit_queries:
                    eval_queries = eval_queries[:limit_queries]

                print(f"🔍 Phase 2 : Évaluation du rappel sur {len(eval_queries)} questions QA...")
                search_latencies_ms: list[float] = []

                ranks: list[int | None] = []
                category_ranks: dict[int, list[int | None]] = {}

                for q_idx, q in enumerate(eval_queries):
                    question_text = q["question"]
                    category = q.get("category", 1)
                    evidence_ids = extract_evidence_ids(q.get("evidence", []))
                    gold_texts = [turn_map[ev]["text"].lower() for ev in evidence_ids if ev in turn_map]

                    t0 = time.perf_counter()
                    s_resp = await client.post(
                        "/search",
                        json={
                            "request_id": f"query-{q_idx}",
                            "user_id": user_id,
                            "query": question_text,
                            "top_k": 10,
                        },
                    )
                    s_lat_ms = (time.perf_counter() - t0) * 1000
                    search_latencies_ms.append(s_lat_ms)

                    found_rank: int | None = None
                    if s_resp.status_code == 200:
                        memories = s_resp.json().get("data", [])
                        if isinstance(memories, dict):
                            memories = memories.get("memories", [])

                        for rank, mem in enumerate(memories, start=1):
                            mem_content = mem.get("content", "").lower()
                            matched_id = any(f"[{ev.lower()}]" in mem_content for ev in evidence_ids)
                            matched_text = any(
                                len(gt) > 15 and gt[:min(len(gt), 60)] in mem_content for gt in gold_texts
                            )
                            if matched_id or matched_text:
                                found_rank = rank
                                break

                    ranks.append(found_rank)
                    category_ranks.setdefault(category, []).append(found_rank)

                    if (q_idx + 1) % 25 == 0 or (q_idx + 1) == len(eval_queries):
                        curr_r1 = sum(1 for r in ranks if r == 1) / (q_idx + 1) * 100
                        curr_r5 = sum(1 for r in ranks if r is not None and r <= 5) / (q_idx + 1) * 100
                        curr_lat = sum(search_latencies_ms) / (q_idx + 1)
                        print(f"  [Search] Q{q_idx+1}/{len(eval_queries)} — Recall@1: {curr_r1:.1f}% | Recall@5: {curr_r5:.1f}% (lat moy: {curr_lat:.0f} ms)")

                total_q = len(ranks)
                r1 = sum(1 for r in ranks if r == 1) / total_q
                r3 = sum(1 for r in ranks if r is not None and r <= 3) / total_q
                r5 = sum(1 for r in ranks if r is not None and r <= 5) / total_q
                r10 = sum(1 for r in ranks if r is not None and r <= 10) / total_q
                mrr = sum(1.0 / r for r in ranks if r is not None) / total_q
                avg_search_lat_ms = sum(search_latencies_ms) / max(len(search_latencies_ms), 1)

                print("=" * 60)
                print("🎯 RÉSULTATS DU BENCHMARK LOCOMO")
                print("=" * 60)
                print(f"Nombre de questions évaluées : {total_q}")
                print(f"Latence moyenne de recherche : {avg_search_lat_ms:.2f} ms")
                print(f"MRR (Mean Reciprocal Rank)   : {mrr:.3f}")
                print(f"Recall@1  : {r1*100:.1f}%")
                print(f"Recall@3  : {r3*100:.1f}%")
                print(f"Recall@5  : {r5*100:.1f}%")
                print(f"Recall@10 : {r10*100:.1f}%\n")

                print("📊 Décomposition par catégorie cognitive :")
                cat_report = {}
                for cat, c_ranks in sorted(category_ranks.items()):
                    cat_name = CATEGORY_NAMES.get(cat, f"Catégorie {cat}")
                    c_tot = len(c_ranks)
                    c_r1 = sum(1 for r in c_ranks if r == 1) / c_tot
                    c_r3 = sum(1 for r in c_ranks if r is not None and r <= 3) / c_tot
                    c_r5 = sum(1 for r in c_ranks if r is not None and r <= 5) / c_tot
                    c_r10 = sum(1 for r in c_ranks if r is not None and r <= 10) / c_tot
                    cat_report[cat_name] = {
                        "count": c_tot,
                        "recall@1": c_r1,
                        "recall@3": c_r3,
                        "recall@5": c_r5,
                        "recall@10": c_r10,
                    }
                    print(f"  • {cat_name} (N={c_tot}) : R@1={c_r1*100:.1f}% | R@3={c_r3*100:.1f}% | R@5={c_r5*100:.1f}% | R@10={c_r10*100:.1f}%")

                report_data = {
                    "mode": mode,
                    "date": datetime.now().isoformat(),
                    "total_messages": total_messages_added,
                    "ingest_time_s": ingest_total_sec,
                    "avg_ingest_lat_ms": avg_session_lat_ms,
                    "throughput_msg_s": msg_per_sec,
                    "total_queries": total_q,
                    "avg_search_lat_ms": avg_search_lat_ms,
                    "mrr": mrr,
                    "recall@1": r1,
                    "recall@3": r3,
                    "recall@5": r5,
                    "recall@10": r10,
                    "categories": cat_report,
                }

                if output_report:
                    output_report.parent.mkdir(parents=True, exist_ok=True)
                    md_lines = [
                        "# Rapport de Benchmark LoCoMo — Mnemos",
                        "",
                        f"**Date** : {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}  ",
                        f"**Mode d'évaluation** : `{mode.upper()}`  ",
                        f"**Dataset** : `{dataset_path.name}` ({conv.get('speaker_a')} & {conv.get('speaker_b')})  ",
                        "",
                        "## 1. Performance d'Ingestion (Écriture synchrone)",
                        "",
                        "| Métrique | Valeur |",
                        "|---|---|",
                        f"| Messages stockés | **{total_messages_added}** ({len(session_keys)} sessions) |",
                        f"| Temps total d'ingestion | **{ingest_total_sec:.2f} s** |",
                        f"| Latence moyenne / session | **{avg_session_lat_ms:.1f} ms** |",
                        f"| Débit d'ingestion | **{msg_per_sec:.1f} messages/sec** |",
                        "",
                        f"## 2. Précision de Récupération (Recall@K sur {total_q} questions)",
                        "",
                        "| Rang de Récupération | Taux de Rappel |",
                        "|---|---|",
                        f"| **MRR (Mean Reciprocal Rank)** | **{mrr:.3f}** |",
                        f"| **Recall@1** | **{r1*100:.1f}%** |",
                        f"| **Recall@3** | **{r3*100:.1f}%** |",
                        f"| **Recall@5** | **{r5*100:.1f}%** |",
                        f"| **Recall@10** | **{r10*100:.1f}%** |",
                        f"| Latence moyenne de recherche | **{avg_search_lat_ms:.2f} ms** |",
                        "",
                        "## 3. Décomposition par Catégorie Cognitive",
                        "",
                        "| Catégorie | Questions | R@1 | R@3 | R@5 | R@10 |",
                        "|---|---|---|---|---|---|",
                    ]
                    for cat_name, metrics in cat_report.items():
                        md_lines.append(
                            f"| {cat_name} | {metrics['count']} | {metrics['recall@1']*100:.1f}% | {metrics['recall@3']*100:.1f}% | {metrics['recall@5']*100:.1f}% | {metrics['recall@10']*100:.1f}% |"
                        )
                    output_report.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
                    print(f"\n📄 Rapport écrit dans : {output_report}")

                return report_data
    finally:
        for eng in engines:
            await eng.dispose()
        shutil.rmtree(tmp_dir, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Mnemos sur LoCoMo")
    parser.add_argument("--mode", choices=["stub", "ollama"], default="stub", help="Mode de calcul")
    parser.add_argument("--dataset", type=Path, default=Path("bench/data/locomo_sample.json"))
    parser.add_argument("--llm-model", type=str, default="qwen2.5:3b", help="Modèle pour saillance & extraction")
    parser.add_argument("--salience-workers", type=int, default=0, help="Nombre de workers pour la file de saillance")
    parser.add_argument("--with-consolidation", action="store_true", help="Active la consolidation cognitive complète")
    parser.add_argument("--grounded-only", action="store_true", help="Filtre les questions d'évaluation aux sessions ingérées")
    parser.add_argument("--limit-sessions", type=int, default=None)
    parser.add_argument("--limit-queries", type=int, default=None)
    parser.add_argument("--output", type=Path, default=Path("bench/results/locomo_report.md"))
    args = parser.parse_args()

    asyncio.run(
        run_locomo_bench(
            args.dataset,
            mode=args.mode,
            llm_model=args.llm_model,
            salience_workers=args.salience_workers,
            with_consolidation=args.with_consolidation,
            grounded_only=args.grounded_only,
            limit_sessions=args.limit_sessions,
            limit_queries=args.limit_queries,
            output_report=args.output,
        )
    )


if __name__ == "__main__":
    main()

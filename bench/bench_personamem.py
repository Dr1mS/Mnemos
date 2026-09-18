"""Benchmark Mnemos sur le jeu de données PersonaMem-v2 (AML Textual Benchmark).

Évalue la capacité de Mnemos à ingérer des historiques multi-tours de personas
et à retrouver les souvenirs / préférences implicites pertinentes parmi des milliers de messages.
"""

from __future__ import annotations

import argparse
import ast
import asyncio
import csv
import json
import shutil
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from mnemos.clock import Clock
from mnemos.config import Settings
from mnemos.embeddings.dense import DenseEmbedder
from mnemos.llm.model_manager import ModelManager
from mnemos.llm.ollama_client import OllamaClient
from mnemos.models.base import make_async_engine
from mnemos.models.episodic import EPISODIC_SCHEMA_SQL
from mnemos.models.semantic import SEMANTIC_SCHEMA_SQL
from mnemos.stores.episodic import BatchEpisodeItem, EpisodicStore
from mnemos.stores.semantic import SemanticStore

HF_BASE_URL = "https://huggingface.co/datasets/bowen-upenn/PersonaMem-v2/resolve/main"
VAL_CSV_URL = f"{HF_BASE_URL}/benchmark/text/val.csv"


def _normalize(s: str) -> str:
    return " ".join(s.lower().split())


def _matches_evidence(candidate: str, snippets: list[str]) -> bool:
    """Vérifie si le candidat contient au moins un extrait significatif de la preuve."""
    norm_c = _normalize(candidate)
    for snip in snippets:
        norm_s = _normalize(snip)
        if len(norm_s) < 15:
            continue
        # Découpe en sous-phrases de 30 caractères pour tolérance aux variations
        step = 40
        for start in range(0, max(1, len(norm_s) - step + 1), step):
            sub = norm_s[start : start + step]
            if len(sub) >= 20 and sub in norm_c:
                return True
    return False


def _sync_download(url: str, local_path: Path) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "Mnemos-Benchmark/1.0"})
    with urllib.request.urlopen(req) as resp, open(local_path, "wb") as f:
        shutil.copyfileobj(resp, f)


async def download_file_if_missing(url: str, local_path: Path) -> None:
    if local_path.exists() and local_path.stat().st_size > 0:
        return
    local_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"📥 Téléchargement de {url}...")
    await asyncio.to_thread(_sync_download, url, local_path)
    print(f"✅ Sauvegardé dans {local_path} ({local_path.stat().st_size / 1024:.1f} Ko)")


async def run_personamem_bench(
    limit_personas: int = 10,
    output_report: Path = Path("bench/results/personamem_report.md"),
) -> dict[str, Any]:
    data_dir = Path("bench/data/personamem")
    data_dir.mkdir(parents=True, exist_ok=True)
    val_csv_path = data_dir / "val.csv"

    # 1. Téléchargement de val.csv
    await download_file_if_missing(VAL_CSV_URL, val_csv_path)

    # 2. Lecture et groupement des requêtes par persona
    with open(val_csv_path, encoding="utf-8") as f:  # noqa: ASYNC230
        reader = csv.DictReader(f)
        all_rows = list(reader)

    persona_to_rows: dict[str, list[dict[str, Any]]] = {}
    for r in all_rows:
        pid = r["persona_id"]
        persona_to_rows.setdefault(pid, []).append(r)

    unique_personas = list(persona_to_rows.keys())
    selected_personas = unique_personas[:limit_personas]
    total_eval_queries = sum(len(persona_to_rows[p]) for p in selected_personas)

    print("=" * 70)
    print(f"🧠 ÉVALUATION PERSONAMEM-V2 SUR {len(selected_personas)} PERSONAS")
    print(f"   • Total de questions d'évaluation : {total_eval_queries}")
    print("=" * 70)

    # 3. Initialisation de la base temporaire Mnemos
    tmp_dir = Path(tempfile.mkdtemp(prefix="mnemos_personamem_"))
    engines: list[AsyncEngine] = []

    try:
        settings = Settings(
            _env_file=None,  # type: ignore[call-arg]
            DATA_DIR=tmp_dir,
            EPISODIC_DB=tmp_dir / "episodic.db",
            SEMANTIC_DB=tmp_dir / "semantic.db",
            PROCEDURAL_DIR=tmp_dir / "procedural",
            EMBED_MODEL="bge-m3:latest",
        )
        clock = Clock()
        epi_engine = make_async_engine(settings.EPISODIC_DB)
        sem_engine = make_async_engine(settings.SEMANTIC_DB)
        engines.extend([epi_engine, sem_engine])

        async with epi_engine.begin() as conn:
            for stmt in EPISODIC_SCHEMA_SQL:
                await conn.execute(text(stmt))
        async with sem_engine.begin() as conn:
            for stmt in SEMANTIC_SCHEMA_SQL:
                await conn.execute(text(stmt))

        client = OllamaClient(settings)
        manager = ModelManager(settings, client)
        embedder = DenseEmbedder(manager, settings)
        store = EpisodicStore(epi_engine, embedder, clock, settings)
        _ = SemanticStore(sem_engine, embedder, clock, settings)

        # 4. Ingestion des historiques de conversation par persona
        chats_dir = data_dir / "chats"
        total_messages_ingested = 0
        t_ingest_start = time.perf_counter()

        for idx, pid in enumerate(selected_personas, 1):
            sample_row = persona_to_rows[pid][0]
            chat_link = sample_row["chat_history_32k_link"]
            chat_filename = Path(chat_link).name
            chat_local_path = chats_dir / chat_filename
            chat_url = f"{HF_BASE_URL}/{chat_link}"

            await download_file_if_missing(chat_url, chat_local_path)

            with open(chat_local_path, encoding="utf-8") as f:  # noqa: ASYNC230
                chat_data = json.load(f)

            messages = chat_data.get("chat_history", [])
            tenant = f"persona_{pid}"

            # Préparation du lot d'épisodes
            items = []
            for m in messages:
                content = m.get("content", "").strip()
                if not content:
                    continue
                role = m.get("role", "user")
                items.append(
                    BatchEpisodeItem(
                        content=content,
                        role=role,
                        session_id=f"sess_{pid}",
                        tenant=tenant,
                    )
                )

            if items:
                await store.write_batch(items)
                total_messages_ingested += len(items)

            print(
                f"  [{idx}/{len(selected_personas)}] Persona {pid} : {len(items)} messages ingérés "
                f"(tenant: {tenant})"
            )

        ingest_duration_s = time.perf_counter() - t_ingest_start
        throughput = total_messages_ingested / ingest_duration_s if ingest_duration_s > 0 else 0

        print(
            f"\n⚡ Ingestion terminée : {total_messages_ingested} messages en {ingest_duration_s:.1f} s "
            f"({throughput:.1f} msg/s)\n"
        )

        # 5. Phase d'évaluation de recherche
        print("🔍 Lancement des requêtes de recherche...")
        reciprocal_ranks: list[float] = []
        hits_at_1: list[int] = []
        hits_at_3: list[int] = []
        hits_at_5: list[int] = []
        hits_at_10: list[int] = []
        hits_at_20: list[int] = []
        latencies_ms: list[float] = []

        query_idx = 0
        for pid in selected_personas:
            tenant = f"persona_{pid}"
            queries = persona_to_rows[pid]

            for q_row in queries:
                query_idx += 1
                try:
                    uq_dict = ast.literal_eval(q_row["user_query"])
                    query_text = uq_dict.get("content", str(q_row["user_query"]))
                except Exception:
                    query_text = str(q_row["user_query"])

                # Extraits cibles de preuve
                target_snippets: list[str] = []
                try:
                    snips_data = json.loads(q_row.get("related_conversation_snippet", "[]"))
                    for s in snips_data:
                        if isinstance(s, dict) and "content" in s:
                            target_snippets.append(s["content"])
                except Exception:
                    pass

                pref = q_row.get("preference", "").strip()
                if pref:
                    target_snippets.append(pref)

                t0 = time.perf_counter()
                results = await store.search(query_text, k=100, tenant=tenant)
                lat_ms = (time.perf_counter() - t0) * 1000.0
                latencies_ms.append(lat_ms)

                # Calcul du rang
                found_rank: int | None = None
                for rank, res in enumerate(results, 1):
                    content = res.episode.content
                    if _matches_evidence(content, target_snippets):
                        found_rank = rank
                        break

                if found_rank is not None:
                    reciprocal_ranks.append(1.0 / found_rank)
                    hits_at_1.append(1 if found_rank <= 1 else 0)
                    hits_at_3.append(1 if found_rank <= 3 else 0)
                    hits_at_5.append(1 if found_rank <= 5 else 0)
                    hits_at_10.append(1 if found_rank <= 10 else 0)
                    hits_at_20.append(1 if found_rank <= 20 else 0)
                else:
                    reciprocal_ranks.append(0.0)
                    hits_at_1.append(0)
                    hits_at_3.append(0)
                    hits_at_5.append(0)
                    hits_at_10.append(0)
                    hits_at_20.append(0)

                if query_idx % 5 == 0 or query_idx == total_eval_queries:
                    cur_mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)
                    cur_r10 = (sum(hits_at_10) / len(hits_at_10)) * 100
                    print(
                        f"  [Recherche] Q{query_idx}/{total_eval_queries} | "
                        f"MRR: {cur_mrr:.3f} | Recall@10: {cur_r10:.1f}% | Latence moy: {sum(latencies_ms)/len(latencies_ms):.1f} ms"
                    )

        n_q = len(reciprocal_ranks)
        mrr = sum(reciprocal_ranks) / n_q if n_q else 0.0
        r1 = (sum(hits_at_1) / n_q) * 100 if n_q else 0.0
        r3 = (sum(hits_at_3) / n_q) * 100 if n_q else 0.0
        r5 = (sum(hits_at_5) / n_q) * 100 if n_q else 0.0
        r10 = (sum(hits_at_10) / n_q) * 100 if n_q else 0.0
        r20 = (sum(hits_at_20) / n_q) * 100 if n_q else 0.0
        avg_lat = sum(latencies_ms) / len(latencies_ms) if latencies_ms else 0.0

        print("\n" + "=" * 70)
        print("🎯 RÉSULTATS DU BENCHMARK PERSONAMEM-V2 (AML Textual)")
        print("=" * 70)
        print(f"Personas évalués           : {len(selected_personas)}")
        print(f"Messages totaux ingérés    : {total_messages_ingested}")
        print(f"Questions évaluées         : {n_q}")
        print(f"Latence moyenne / requête  : {avg_lat:.2f} ms")
        print(f"MRR (Mean Reciprocal Rank) : {mrr:.3f}")
        print(f"Recall@1  : {r1:.1f}%")
        print(f"Recall@3  : {r3:.1f}%")
        print(f"Recall@5  : {r5:.1f}%")
        print(f"Recall@10 : {r10:.1f}%")
        print(f"Recall@20 : {r20:.1f}%")
        print("=" * 70)

        # Rapport Markdown
        output_report.parent.mkdir(parents=True, exist_ok=True)
        report_content = f"""# 📊 Rapport d'Évaluation PersonaMem-v2 — Mnemos
## Agent Memory Challenge (Cycle 2) — Track Textual Memory

- **Date** : {time.strftime('%Y-%m-%d %H:%M:%S')}
- **Personas évalués** : {len(selected_personas)}
- **Total messages ingérés** : {total_messages_ingested}
- **Questions évaluées** : {n_q}
- **Modèle d'embedding** : `bge-m3:latest` (batch ingestion atomique SQLite-vec)

---

### 1. Métriques de Récupération (Retrieval Rank Order)

| Rang de Rappel | Score |
|---|:---:|
| **MRR (Mean Reciprocal Rank)** | **{mrr:.3f}** |
| **Recall@1 (Top-1 direct)** | **{r1:.1f}%** |
| **Recall@3** | **{r3:.1f}%** |
| **Recall@5** | **{r5:.1f}%** |
| **Recall@10** | **{r10:.1f}%** |
| **Recall@20** | **{r20:.1f}%** |
| **Latence moyenne de recherche** | **{avg_lat:.2f} ms** |

---

### 2. Performance d'Ingestion

| Métrique | Valeur |
|---|:---:|
| Temps total d'ingestion | **{ingest_duration_s:.1f} s** |
| Débit d'ingestion | **{throughput:.1f} messages/sec** |
| Isolation multi-tenant | **1 tenant hermétique par persona (`persona_<id>`)** |
"""
        output_report.write_text(report_content, encoding="utf-8")
        print(f"📄 Rapport écrit dans : {output_report}")

        return {
            "mrr": mrr,
            "recall@1": r1,
            "recall@5": r5,
            "recall@10": r10,
            "personas": len(selected_personas),
            "total_messages": total_messages_ingested,
            "queries": n_q,
        }

    finally:
        for eng in engines:
            await eng.dispose()
        shutil.rmtree(tmp_dir, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Mnemos sur PersonaMem-v2")
    parser.add_argument(
        "--limit-personas",
        type=int,
        default=10,
        help="Nombre de personas à évaluer (défaut: 10, environ 2000 messages)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("bench/results/personamem_report.md"),
        help="Chemin du rapport Markdown de sortie",
    )
    args = parser.parse_args()

    asyncio.run(run_personamem_bench(limit_personas=args.limit_personas, output_report=args.output))


if __name__ == "__main__":
    main()

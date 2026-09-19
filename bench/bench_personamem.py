"""Benchmark PersonaMem-v2 pour Mnemos (Agent Memory Challenge — Textual Track).

Rejoue les historiques de conversation de personas à travers les endpoints AML
(POST /add puis POST /search), comme le fait la plateforme, et mesure le rang du
premier message de preuve (``related_conversation_snippet``) dans les résultats.

Protocole, aligné sur le contrat AML :
- Add découpé en lots de 20 messages ou 2 000 mots au plus (découpage Textual).
- Rôles user/assistant uniquement : le message system de PersonaMem contient le
  profil complet de la persona, que la plateforme n'envoie pas.
- Search avec top_k=100, la valeur officielle.
- Les extraits de preuve sont des messages verbatim de l'historique : la
  correspondance se fait par égalité de texte normalisé.
- Sans --with-consolidation, aucun fait n'est extrait : /search ne renvoie alors
  que des épisodes (mémoire épisodique seule).
"""

from __future__ import annotations

import argparse
import ast
import asyncio
import csv
import json
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx

from bench.bench_locomo import drain_consolidation, setup_bench_app

HF_BASE_URL = "https://huggingface.co/datasets/bowen-upenn/PersonaMem-v2/resolve/main"
VAL_CSV_URL = f"{HF_BASE_URL}/benchmark/text/val.csv"
DATA_DIR = Path("bench/data/personamem")

# Découpage appliqué par la plateforme AML aux Add du track Textual ordinaire.
CHUNK_MAX_MESSAGES = 20
CHUNK_MAX_WORDS = 2000
TOP_K = 100
RECALL_KS = (1, 3, 5, 10, 20, 50, 100)
INGESTED_ROLES = ("user", "assistant")


def _normalize(s: str) -> str:
    return " ".join(s.lower().split())


def chunk_messages(messages: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Découpe l'historique en lots d'au plus 20 messages ou 2 000 mots.

    Un message seul de plus de 2 000 mots forme son propre lot."""
    chunks: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    words = 0
    for m in messages:
        n = len(str(m["content"]).split())
        if current and (len(current) >= CHUNK_MAX_MESSAGES or words + n > CHUNK_MAX_WORDS):
            chunks.append(current)
            current, words = [], 0
        current.append(m)
        words += n
    if current:
        chunks.append(current)
    return chunks


def parse_query(raw: str) -> str:
    """user_query est un dict Python sérialisé : {'role': 'user', 'content': ...}."""
    try:
        parsed = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return raw
    if isinstance(parsed, dict):
        return str(parsed.get("content", raw))
    return raw


def parse_evidence(raw: str) -> set[str]:
    """Messages de preuve (texte normalisé) listés dans related_conversation_snippet."""
    try:
        snippets = json.loads(raw or "[]")
    except json.JSONDecodeError:
        return set()
    return {
        _normalize(str(s["content"]))
        for s in snippets
        if isinstance(s, dict) and str(s.get("content", "")).strip()
    }


def rank_metrics(ranks: list[int | None]) -> dict[str, float]:
    n = len(ranks)
    metrics: dict[str, float] = {
        "count": n,
        "mrr": sum(1.0 / r for r in ranks if r is not None) / n if n else 0.0,
    }
    for k in RECALL_KS:
        metrics[f"recall@{k}"] = sum(1 for r in ranks if r is not None and r <= k) / n if n else 0.0
    return metrics


def breakdown(records: list[dict[str, Any]], key: str) -> dict[str, dict[str, float]]:
    groups: dict[str, list[int | None]] = {}
    for r in records:
        groups.setdefault(str(r[key]) or "(vide)", []).append(r["rank"])
    return {name: rank_metrics(ranks) for name, ranks in sorted(groups.items())}


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(pct / 100 * (len(ordered) - 1)))]


def _latency_stats(values: list[float]) -> dict[str, float]:
    return {
        "mean": sum(values) / len(values) if values else 0.0,
        "p50": _percentile(values, 50),
        "p95": _percentile(values, 95),
        "max": max(values, default=0.0),
    }


def _run(cmd: list[str]) -> str | None:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=10, check=True)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip()


def ollama_loaded_models(host: str) -> list[dict[str, Any]]:
    """Modèles chargés par Ollama et leur part en VRAM (GET /api/ps)."""
    try:
        resp = httpx.get(f"{host}/api/ps", timeout=5.0)
        resp.raise_for_status()
    except httpx.HTTPError:
        return []
    return [
        {
            "name": m.get("name"),
            "size": m.get("size", 0),
            "size_vram": m.get("size_vram", 0),
        }
        for m in resp.json().get("models", [])
    ]


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


def load_personas(val_csv_path: Path) -> dict[str, list[dict[str, str]]]:
    """Questions groupées par persona, dans l'ordre du CSV."""
    csv.field_size_limit(2**31 - 1)
    with val_csv_path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    persona_rows: dict[str, list[dict[str, str]]] = {}
    for r in rows:
        persona_rows.setdefault(r["persona_id"], []).append(r)
    return persona_rows


def load_chat(chat_path: Path) -> list[dict[str, str]]:
    """Messages user/assistant non vides de l'historique 32k, dans l'ordre source."""
    chat = json.loads(chat_path.read_text(encoding="utf-8"))
    return [
        {"role": m["role"], "content": m["content"]}
        for m in chat.get("chat_history", [])
        if m.get("role") in INGESTED_ROLES and str(m.get("content", "")).strip()
    ]


async def run_personamem_bench(
    limit_personas: int = 10,
    llm_model: str = "qwen2.5:3b",
    with_consolidation: bool = False,
    salience_workers: int = 2,
    output_report: Path = Path("bench/results/personamem_report.md"),
) -> dict[str, Any]:
    val_csv_path = DATA_DIR / "val.csv"
    await download_file_if_missing(VAL_CSV_URL, val_csv_path)
    persona_rows = load_personas(val_csv_path)
    selected = list(persona_rows)[:limit_personas] if limit_personas > 0 else list(persona_rows)

    # Téléchargements hors chronométrage
    chat_paths: dict[str, Path] = {}
    for pid in selected:
        link = persona_rows[pid][0]["chat_history_32k_link"]
        chat_paths[pid] = DATA_DIR / "chats" / Path(link).name
        await download_file_if_missing(f"{HF_BASE_URL}/{link}", chat_paths[pid])

    mode_label = "cognitif complet (saillance + consolidation)" if with_consolidation else "épisodique seul (sans LLM)"
    print("=" * 70)
    print(f"🧠 PERSONAMEM-V2 — {len(selected)} personas, {sum(len(persona_rows[p]) for p in selected)} questions")
    print(f"   Mode : {mode_label}")
    print("=" * 70)

    engines: list[Any] = []
    tmp_dir = Path(tempfile.mkdtemp(prefix="mnemos_personamem_"))
    try:
        app, engines, settings = await setup_bench_app(
            tmp_dir,
            "ollama",
            llm_model=llm_model,
            salience_workers=salience_workers if with_consolidation else 0,
        )
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport, base_url="http://test", timeout=600.0
            ) as client:
                # Chargement du modèle d'embedding hors chronométrage
                warm = await client.post(
                    "/search", json={"query": "warmup", "user_id": "personamem:warmup", "top_k": 1}
                )
                if warm.status_code != 200:
                    raise RuntimeError(f"Warm-up /search en échec : {warm.status_code} {warm.text}")

                # 1. INGESTION via POST /add
                add_lat_ms: list[float] = []
                add_errors = 0
                n_messages = 0
                ingested: dict[str, set[str]] = {}
                t_ingest = time.perf_counter()
                for idx, pid in enumerate(selected, 1):
                    messages = load_chat(chat_paths[pid])
                    ingested[pid] = {_normalize(m["content"]) for m in messages}
                    user_id = f"personamem:{pid}"
                    for c_idx, chunk in enumerate(chunk_messages(messages)):
                        request_id = f"{user_id}:chunk-{c_idx}"
                        t0 = time.perf_counter()
                        resp = await client.post(
                            "/add",
                            json={
                                "request_id": request_id,
                                "user_id": user_id,
                                "session_id": f"{user_id}:chat",
                                "messages": chunk,
                            },
                        )
                        add_lat_ms.append((time.perf_counter() - t0) * 1000)
                        body = resp.json() if resp.status_code == 200 else {}
                        if body.get("success") is not True or body.get("request_id") != request_id:
                            add_errors += 1
                            print(f"❌ Add {request_id} : HTTP {resp.status_code} {resp.text[:200]}")
                    n_messages += len(messages)
                    if idx % 5 == 0 or idx == len(selected):
                        elapsed = time.perf_counter() - t_ingest
                        print(f"  [Ingestion] {idx}/{len(selected)} personas — {n_messages} msgs ({n_messages / elapsed:.1f} msg/s)")
                ingest_s = time.perf_counter() - t_ingest
                print(f"✅ Ingestion : {n_messages} messages, {len(add_lat_ms)} Add en {ingest_s:.1f} s\n")

                # 2. CONSOLIDATION (optionnelle)
                consolidation: dict[str, Any] | None = None
                if with_consolidation:
                    print("⏳ Scoring de saillance (drain de la queue)...")
                    t_cons = time.perf_counter()
                    await app.state.queue.join()
                    scoring_s = time.perf_counter() - t_cons
                    print("🧠 Consolidation sémantique...")
                    totals = await drain_consolidation(app)
                    consolidation = {
                        **totals,
                        "scoring_s": scoring_s,
                        "total_s": time.perf_counter() - t_cons,
                    }
                    print(f"✅ Consolidation : {totals['facts_inserted']} faits, {totals['entities_upserted']} entités en {consolidation['total_s']:.1f} s\n")

                loaded_models = ollama_loaded_models(settings.OLLAMA_HOST)

                # 3. RECHERCHE via POST /search
                records: list[dict[str, Any]] = []
                search_lat_ms: list[float] = []
                contract_errors = 0
                skipped_no_evidence = 0
                for pid in selected:
                    user_id = f"personamem:{pid}"
                    for row in persona_rows[pid]:
                        evidence = parse_evidence(row.get("related_conversation_snippet", ""))
                        if not evidence & ingested[pid]:
                            skipped_no_evidence += 1
                            continue
                        t0 = time.perf_counter()
                        resp = await client.post(
                            "/search",
                            json={"query": parse_query(row["user_query"]), "user_id": user_id, "top_k": TOP_K},
                        )
                        lat_ms = (time.perf_counter() - t0) * 1000
                        search_lat_ms.append(lat_ms)
                        data = resp.json().get("data") if resp.status_code == 200 else None
                        if (
                            not isinstance(data, list)
                            or len(data) > TOP_K
                            or any(not it.get("id") or not str(it.get("content", "")).strip() for it in data)
                        ):
                            contract_errors += 1
                            data = data if isinstance(data, list) else []
                        rank = next(
                            (i for i, it in enumerate(data, 1) if _normalize(str(it.get("content", ""))) in evidence),
                            None,
                        )
                        records.append({
                            "persona_id": pid,
                            "pref_type": row.get("pref_type", ""),
                            "who": row.get("who", ""),
                            "updated": row.get("updated", ""),
                            "rank": rank,
                            "latency_ms": lat_ms,
                        })
                        if len(records) % 25 == 0:
                            m = rank_metrics([r["rank"] for r in records])
                            print(f"  [Search] Q{len(records)} — MRR {m['mrr']:.3f} | R@10 {m['recall@10'] * 100:.1f}%")

        overall = rank_metrics([r["rank"] for r in records])
        commit = _run(["git", "rev-parse", "--short", "HEAD"])
        dirty = bool(_run(["git", "status", "--porcelain"]))
        result: dict[str, Any] = {
            "metadata": {
                "date": datetime.now().isoformat(timespec="seconds"),
                "commit": f"{commit}{'-dirty' if dirty else ''}" if commit else None,
                "gpu": _run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"]),
                "ollama_loaded_models": loaded_models,
                "mode": mode_label,
                "embed_model": settings.EMBED_MODEL,
                "llm_model": llm_model if with_consolidation else None,
                "top_k": TOP_K,
                "chunking": {"max_messages": CHUNK_MAX_MESSAGES, "max_words": CHUNK_MAX_WORDS},
                "personas": len(selected),
            },
            "ingestion": {
                "messages": n_messages,
                "add_requests": len(add_lat_ms),
                "add_errors": add_errors,
                "total_s": ingest_s,
                "throughput_msg_s": n_messages / ingest_s if ingest_s > 0 else 0.0,
                "add_latency_ms": _latency_stats(add_lat_ms),
            },
            "consolidation": consolidation,
            "search": {
                "queries": len(records),
                "skipped_no_evidence": skipped_no_evidence,
                "contract_errors": contract_errors,
                "latency_ms": _latency_stats(search_lat_ms),
            },
            "metrics": overall,
            "by_pref_type": breakdown(records, "pref_type"),
            "by_who": breakdown(records, "who"),
            "by_updated": breakdown(records, "updated"),
            "questions": records,
        }

        print("\n" + "=" * 70)
        print("🎯 RÉSULTATS PERSONAMEM-V2")
        print("=" * 70)
        print(f"Questions évaluées : {len(records)} (sans preuve ingérée : {skipped_no_evidence})")
        print(f"Erreurs de contrat : Add {add_errors} | Search {contract_errors}")
        print(f"MRR@{TOP_K} : {overall['mrr']:.3f}")
        print(" | ".join(f"R@{k} {overall[f'recall@{k}'] * 100:.1f}%" for k in RECALL_KS))
        print(f"Latence Search p50/p95 : {result['search']['latency_ms']['p50']:.1f} / {result['search']['latency_ms']['p95']:.1f} ms")
        print(f"Débit d'ingestion : {result['ingestion']['throughput_msg_s']:.1f} msg/s")

        output_report.parent.mkdir(parents=True, exist_ok=True)
        output_report.with_suffix(".json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        output_report.write_text(render_report(result), encoding="utf-8")
        print(f"📄 Rapport : {output_report} (+ {output_report.with_suffix('.json').name})")
        return result
    finally:
        for eng in engines:
            await eng.dispose()
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _metrics_row(label: str, m: dict[str, float]) -> str:
    recalls = " | ".join(f"{m[f'recall@{k}'] * 100:.1f}%" for k in RECALL_KS)
    return f"| {label} | {int(m['count'])} | {m['mrr']:.3f} | {recalls} |"


def render_report(result: dict[str, Any]) -> str:
    meta, ing, search = result["metadata"], result["ingestion"], result["search"]
    header = "| | Questions | MRR | " + " | ".join(f"R@{k}" for k in RECALL_KS) + " |"
    sep = "|---|---|---|" + "---|" * len(RECALL_KS)
    models = ", ".join(
        f"`{m['name']}` ({m['size_vram'] / m['size'] * 100:.0f}% VRAM)" if m["size"] else f"`{m['name']}`"
        for m in meta["ollama_loaded_models"]
    ) or "inconnu"
    lines = [
        "# Rapport PersonaMem-v2 — Mnemos",
        "",
        f"- **Date** : {meta['date']} — commit `{meta['commit']}`",
        f"- **Mode** : {meta['mode']}",
        f"- **GPU** : {meta['gpu'] or 'aucun détecté'} — modèles Ollama chargés : {models}",
        f"- **Embedding** : `{meta['embed_model']}` — LLM : `{meta['llm_model'] or 'aucun'}`",
        f"- **Protocole** : endpoints AML `/add` + `/search`, top_k={meta['top_k']}, Add découpés à "
        f"{meta['chunking']['max_messages']} messages / {meta['chunking']['max_words']} mots, message system exclu",
        f"- **Échantillon** : {meta['personas']} personas, {search['queries']} questions "
        f"({search['skipped_no_evidence']} écartées faute de preuve dans l'historique ingéré)",
        "",
        "## 1. Récupération de la preuve",
        "",
        header,
        sep,
        _metrics_row("**Global**", result["metrics"]),
        "",
        f"Erreurs de contrat : Add **{ing['add_errors']}** / Search **{search['contract_errors']}**.",
        "",
        "## 2. Performance",
        "",
        "| Métrique | Valeur |",
        "|---|---|",
        f"| Messages ingérés | {ing['messages']} ({ing['add_requests']} requêtes Add) |",
        f"| Temps d'ingestion | {ing['total_s']:.1f} s |",
        f"| Débit d'ingestion | {ing['throughput_msg_s']:.1f} msg/s |",
        f"| Latence Add p50 / p95 | {ing['add_latency_ms']['p50']:.0f} / {ing['add_latency_ms']['p95']:.0f} ms |",
        f"| Latence Search p50 / p95 | {search['latency_ms']['p50']:.1f} / {search['latency_ms']['p95']:.1f} ms |",
    ]
    cons = result["consolidation"]
    if cons:
        lines += [
            f"| Consolidation (scoring + extraction) | {cons['total_s']:.1f} s, {cons['passes']} passes |",
            f"| Faits / entités extraits | {cons['facts_inserted']} / {cons['entities_upserted']} |",
            f"| Échecs d'extraction | {cons['extraction_failures']} |",
        ]
    for title, key in (
        ("3. Par type de préférence", "by_pref_type"),
        ("4. Par titulaire de la préférence (`who`)", "by_who"),
        ("5. Par préférence mise à jour (`updated`)", "by_updated"),
    ):
        lines += ["", f"## {title}", "", header, sep]
        lines += [_metrics_row(name, m) for name, m in result[key].items()]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Mnemos sur PersonaMem-v2 via les endpoints AML")
    parser.add_argument("--limit-personas", type=int, default=10, help="Nombre de personas (0 = toutes)")
    parser.add_argument("--llm-model", default="qwen2.5:3b", help="Modèle de saillance & extraction")
    parser.add_argument("--with-consolidation", action="store_true", help="Saillance LLM + extraction de faits avant la recherche")
    parser.add_argument("--salience-workers", type=int, default=2, help="Workers de saillance (avec --with-consolidation)")
    parser.add_argument("--output", type=Path, default=Path("bench/results/personamem_report.md"))
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # console Windows cp1252 vs emojis

    asyncio.run(
        run_personamem_bench(
            limit_personas=args.limit_personas,
            llm_model=args.llm_model,
            with_consolidation=args.with_consolidation,
            salience_workers=args.salience_workers,
            output_report=args.output,
        )
    )


if __name__ == "__main__":
    main()

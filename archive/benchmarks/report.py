"""Générateur de rapport de synthèse des benchmarks de Mnemos."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from benchmarks.scenarios.bench_concurrency import ConcurrencyReport
from benchmarks.scenarios.bench_conflict import CognitiveReport
from benchmarks.scenarios.bench_decay import DecayReport
from benchmarks.scenarios.bench_volume_knn import VolumeKnnReport


def format_markdown_report(
    volume_report: VolumeKnnReport | None,
    concurrency_report: ConcurrencyReport | None,
    cognitive_report: CognitiveReport | None,
    decay_report: DecayReport | None,
    duration_total_sec: float,
) -> str:
    lines = [
        "# 📊 Rapport de Benchmark & Stress-Test — Mnemos",
        "",
        f"*Date d'exécution : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*",
        f"*Durée totale du banc de test : {duration_total_sec:.2f} s*",
        "",
        "---",
        "",
    ]

    # ── Section 1 : Volume & Scalabilité Vectorielle (KNN) ───────────────────
    if volume_report:
        lines.extend([
            "## 1. 📈 Scalabilité Volume & Latence KNN (sqlite-vec)",
            "",
            f"**Mode** : `{volume_report.mode}`",
            "",
            "| Volume | Débit Écrit. (ops/s) | Écrit. p50 | Écrit. p95 | KNN p50 | KNN p95 | Base | WAL | Bytes/Ép. |",
            "|--------|----------------------|------------|------------|---------|---------|------|-----|-----------|",
        ])
        for step in volume_report.steps:
            lines.append(
                f"| {step.target_count:,} | {step.write_stats.throughput_ops_sec:.1f} | "
                f"{step.write_stats.p50_ms:.2f} ms | {step.write_stats.p95_ms:.2f} ms | "
                f"{step.search_stats.p50_ms:.2f} ms | {step.search_stats.p95_ms:.2f} ms | "
                f"{step.db_size_kb:.1f} KB | {step.wal_size_kb:.1f} KB | "
                f"{step.bytes_per_episode:.0f} B |"
            )
        lines.extend([
            "",
            f"- **Rappel multi-tenant asymétrique** : `{volume_report.tenant_skew_recall:.1f}%` "
            "(Tenant B avec 5% du volume face à 95% Tenant A).",
            f"- **Latence de recherche partitionnée Tenant B** : "
            f"`{volume_report.tenant_skew_search_ms:.2f} ms`.",
            "",
            "---",
            "",
        ])

    # ── Section 2 : Concurrence & Débit ──────────────────────────────────────
    if concurrency_report:
        lines.extend([
            "## 2. ⚡ Concurrence, Deadlock & Salience Queue",
            "",
            "| Workers Simultanés | Débit Global (ops/s) | Latence p50 | Latence p95 | Latence p99 | Erreurs |",
            "|--------------------|----------------------|-------------|-------------|-------------|---------|",
        ])
        for w in concurrency_report.workers_results:
            lines.append(
                f"| {w.concurrency_level} workers | {w.throughput_ops_sec:.1f} ops/s | "
                f"{w.p50_ms:.2f} ms | {w.p95_ms:.2f} ms | {w.p99_ms:.2f} ms | {w.errors_count} |"
            )
        lines.append("")
        status_deadlock = "✅ CONFORME" if concurrency_report.tier_deadlock_free else "❌ ÉCHEC"
        lines.append(f"- **Intégrité Tiers ModelManager (Pas de Deadlock)** : `{status_deadlock}`")
        if concurrency_report.salience_burst:
            sb = concurrency_report.salience_burst
            loss_status = "✅ Zéro perte" if sb.unscored_remaining == 0 else "❌ Perte détectée"
            lines.extend([
                f"- **Stress-Test Burst Saillance ({sb.burst_size} messages)** :",
                f"  - Acceptés immédiatement en mémoire : `{sb.accepted_memory}`",
                f"  - Différés en base sans perte : `{sb.deferred_db}`",
                f"  - Temps de résorption (Auto-Drain) : `{sb.drain_time_sec:.2f} s`",
                f"  - Épisodes non scorés restants : `{sb.unscored_remaining}` `{loss_status}`",
            ])
        lines.extend(["", "---", ""])

    # ── Section 3 : Vérité Cognitive & Résolution de Conflits ────────────────
    if cognitive_report:
        lines.extend([
            "## 3. 🧠 Vérité Cognitive & Intégrité Temporelle",
            "",
        ])
        if cognitive_report.temporal_chain:
            tc = cognitive_report.temporal_chain
            cont_status = "✅ OUI" if tc.chain_continuous else "❌ NON"
            lines.extend([
                f"- **Chaos Temporel (Prédicat `{tc.predicate}`)** :",
                f"  - Total faits générés dans la chaîne : `{tc.total_facts}`",
                f"  - Faits restés actifs (`valid_until IS NULL`) : `{tc.current_facts_count}`",
                f"  - Continuité stricte `superseded_by` / `valid_from` : `{cont_status}`",
                f"  - Dernière vérité retenue : **`{tc.final_value}`**",
            ])
        card_st = "✅ CONFORME" if cognitive_report.cardinality_respected else "❌ ÉCHEC"
        cross_st = "✅ STRICTE" if cognitive_report.cross_tenant_conflict_isolated else "❌ ÉCHEC"
        alias_st = "✅ CONFORME" if cognitive_report.alias_deduplication_clean else "❌ ÉCHEC"
        lines.extend([
            f"- **Respect cardinalité (ONE vs MANY)** : `{card_st}`",
            f"- **Isolation cognitive cross-tenant** : `{cross_st}`",
            f"- **Déduplication par alias** : `{alias_st}`",
            f"- **Doublons de faits courants (Quality Gate §21)** : "
            f"`{cognitive_report.duplicate_current_count}`",
            "",
            "---",
            "",
        ])

    # ── Section 4 : Decay Biologique ─────────────────────────────────────────
    if decay_report:
        lines.extend([
            "## 4. ⏳ Simulation Longue Durée (365 Jours — Decay & Oubli)",
            "",
            "| Jour Virtuel | Moy. Decay Low (0.2) | Moy. Decay Med (0.5) | Moy. Decay High (0.9) | Actifs | Archivés |",
            "|--------------|----------------------|----------------------|-----------------------|--------|----------|",
        ])
        for sn in decay_report.snapshots:
            lines.append(
                f"| J+{sn.day:03d} | {sn.low_salience_avg_decay:.3f} | "
                f"{sn.med_salience_avg_decay:.3f} | {sn.high_salience_avg_decay:.3f} | "
                f"{sn.active_episodes} | {sn.archived_episodes} |"
            )
        lines.extend([
            "",
            f"- **Survie à J+365 (Forte saillance)** : "
            f"`{decay_report.high_salience_survival_rate:.1f}%`",
            f"- **Survie à J+365 (Faible saillance)** : "
            f"`{decay_report.low_salience_survival_rate:.1f}%`",
            f"- **Souvenirs purgés & exportés JSONL** : `{decay_report.dumped_jsonl_count}`",
            "",
        ])

    return "\n".join(lines)


def save_reports(
    output_dir: Path,
    markdown_content: str,
    raw_data: dict[str, Any],
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    md_file = output_dir / "BENCHMARK_REPORT.md"
    json_file = output_dir / "latest.json"

    md_file.write_text(markdown_content, encoding="utf-8")
    json_file.write_text(json.dumps(raw_data, indent=2, default=str), encoding="utf-8")

    return md_file, json_file

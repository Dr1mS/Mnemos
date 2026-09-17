"""Exécuteur d'évaluation pour le jeu inspiré de MemoryAgentBench (ICLR 2026).

Évalue un système de mémoire sur 4 compétences cardinales :
1. Accurate Retrieval (AR)
2. Test-Time Learning (TTL)
3. Long-Range Understanding (LRU)
4. Conflict Resolution (CR)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from bench.backends.base import BaseMemoryBackend
from bench.config import BenchConfig
from bench.eval_utils import normalize_text
from bench.memoryagentbench.adapter import UnifiedBenchmarkAgent
from bench.memoryagentbench.fixtures import MAB_BENCHMARK_SAMPLES


@dataclass
class MABEvaluationReport:
    backend_name: str
    overall_score: float
    ar_score: float
    ttl_score: float
    lru_score: float
    cr_score: float
    total_samples: int
    passed_samples: int
    sample_results: list[dict[str, Any]] = field(default_factory=list)


def _check_match(prediction: str, ground_truth: str) -> bool:
    """Vérifie la conformité de la prédiction avec la vérité terrain après normalisation."""
    norm_pred = normalize_text(prediction)
    norm_gt = normalize_text(ground_truth)

    # Vérification d'inclusion directe
    if norm_gt in norm_pred:
        return True

    # Vérification des mots-clés discriminants (>2 caractères ou chiffres)
    keywords = [w for w in norm_gt.split() if len(w) > 2 or w.isdigit()]
    if not keywords:
        return norm_gt in norm_pred
    return all(kw in norm_pred for kw in keywords)


async def run_memoryagentbench(backend: BaseMemoryBackend, config: BenchConfig) -> MABEvaluationReport:
    agent = UnifiedBenchmarkAgent(backend, config)
    await agent.setup()

    results: list[dict[str, Any]] = []
    split_counts: dict[str, int] = {
        "Accurate_Retrieval": 0,
        "Test_Time_Learning": 0,
        "Long_Range_Understanding": 0,
        "Conflict_Resolution": 0,
    }
    split_passed: dict[str, int] = {
        "Accurate_Retrieval": 0,
        "Test_Time_Learning": 0,
        "Long_Range_Understanding": 0,
        "Conflict_Resolution": 0,
    }

    for sample in MAB_BENCHMARK_SAMPLES:
        await agent.reset()

        for chunk in sample.context_chunks:
            await agent.add_chunk(chunk)

        answer = await agent.answer(sample.question)
        passed = _check_match(answer, sample.ground_truth)

        split_counts[sample.split] += 1
        if passed:
            split_passed[sample.split] += 1

        results.append({
            "qa_id": sample.qa_id,
            "split": sample.split,
            "question": sample.question,
            "ground_truth": sample.ground_truth,
            "prediction": answer,
            "passed": passed,
        })

    def calc_rate(split: str) -> float:
        c = split_counts.get(split, 0)
        return round(split_passed.get(split, 0) / c, 3) if c > 0 else 0.0

    ar_score = calc_rate("Accurate_Retrieval")
    ttl_score = calc_rate("Test_Time_Learning")
    lru_score = calc_rate("Long_Range_Understanding")
    cr_score = calc_rate("Conflict_Resolution")

    total = len(MAB_BENCHMARK_SAMPLES)
    passed_total = sum(1 for r in results if r["passed"])
    overall = round(passed_total / total, 3) if total > 0 else 0.0

    report = MABEvaluationReport(
        backend_name=backend.name,
        overall_score=overall,
        ar_score=ar_score,
        ttl_score=ttl_score,
        lru_score=lru_score,
        cr_score=cr_score,
        total_samples=total,
        passed_samples=passed_total,
        sample_results=results,
    )

    config.results_dir.mkdir(parents=True, exist_ok=True)
    report_md_path = config.results_dir / "memoryagentbench_report.md"
    _generate_mab_markdown_report(report, config, report_md_path)

    return report


def _generate_mab_markdown_report(report: MABEvaluationReport, config: BenchConfig, output_path: Path) -> None:
    lines: list[str] = [
        "# Rapport d'Évaluation — Jeu de tests inspiré de la taxonomie MemoryAgentBench (ICLR 2026)",
        "",
    ]
    if config.dry_run:
        lines.extend([
            "> ⚠️ **DRY RUN — RÉSULTATS NON SIGNIFICATIFS (MODE STUB SANS MODÈLE NEURONAL)**",
            "",
        ])

    lines.extend([
        f"- **Système évalué** : `{report.backend_name}`",
        f"- **Modèle LLM** : `{config.llm_model}` (dry_run: {config.dry_run})",
        f"- **Modèle d'embedding** : `{config.embed_model}`",
        f"- **Score Global** : **{report.overall_score * 100:.1f}%** ({report.passed_samples}/{report.total_samples})",
        "",
        "> *Avertissement méthodologique : Ce jeu d'épreuve est un banc synthétique ciblé inspiré des 4 compétences de MemoryAgentBench, et non l'exécution du corpus officiel intégral multi-mille instances.*",
        "",
        "---",
        "",
        "## Résultats par Compétence",
        "",
        "| Compétence | Échantillons | Score |",
        "|---|---|---|",
        f"| **Accurate Retrieval (AR)** | {sum(1 for r in report.sample_results if r['split'] == 'Accurate_Retrieval')} | {report.ar_score * 100:.1f}% |",
        f"| **Test-Time Learning (TTL)** | {sum(1 for r in report.sample_results if r['split'] == 'Test_Time_Learning')} | {report.ttl_score * 100:.1f}% |",
        f"| **Long-Range Understanding (LRU)** | {sum(1 for r in report.sample_results if r['split'] == 'Long_Range_Understanding')} | {report.lru_score * 100:.1f}% |",
        f"| **Conflict Resolution (CR)** | {sum(1 for r in report.sample_results if r['split'] == 'Conflict_Resolution')} | {report.cr_score * 100:.1f}% |",
        "",
        "---",
        "",
        "## Détail des Épreuves Individuelles",
        "",
    ])

    for item in report.sample_results:
        status = "Succès" if item["passed"] else "Échec"
        lines.extend([
            f"### {item['qa_id']} ({item['split']}) : {status}",
            f"- **Question** : *{item['question']}*",
            f"- **Vérité attendue** : `{item['ground_truth']}`",
            f"- **Réponse produite** : `{item['prediction']}`",
            "",
        ])

    output_path.write_text("\n".join(lines), encoding="utf-8")

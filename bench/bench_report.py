"""Harnais CLI pour l'évaluation comparative neutre des architectures de mémoire et génération du rapport."""

from __future__ import annotations

import asyncio
import json
import math
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import typer

from bench.backends import ALL_BACKENDS, CONTROL_BACKENDS, BaseMemoryBackend, MnemosBackend
from bench.config import BenchConfig
from bench.datasets import (
    DEV_BLAND_INSTANCES,
    DEV_CONVENTION_PROMPTS,
    DEV_EXTERNAL_INSTANCES,
    DEV_GHOST_INSTANCES,
)
from bench.eval_utils import holm_bonferroni_correction, mcnemar_test
from bench.scenarios.test1_ghost_vector import GhostVectorResult, run_test1_ghost_vector
from bench.scenarios.test1b_external_ontology import ExternalOntologyResult, run_test1b_external_ontology
from bench.scenarios.test2_bland_noise import BlandNoiseResult, run_test2_bland_noise
from bench.scenarios.test3_compliance import ComplianceResult, run_test3_compliance

app = typer.Typer(help="Banc d'essai comparatif neutre pour architectures de mémoire LLM")


def _set_seeds(seed: int) -> None:
    random.seed(seed)


def _format_rel(score: float, oracle_score: float | None) -> str:
    """Rapporte un score au plafond de l'oracle (score / oracle)."""
    if oracle_score is None or oracle_score <= 0.0:
        return "N/A"
    rel = (score / oracle_score) * 100.0
    return f"{rel:.0f}%"


def _compute_holm_adjusted_mcnemar_dict(
    t1_results: list[GhostVectorResult],
    t1b_results: list[ExternalOntologyResult],
    t2_results: list[BlandNoiseResult],
    t3_results: list[ComplianceResult],
    mnemos_t1: GhostVectorResult | None,
    mnemos_t1b: ExternalOntologyResult | None,
    mnemos_t2: BlandNoiseResult | None,
    mnemos_t3: ComplianceResult | None,
) -> dict[str, str]:
    """Calcule tous les tests de McNemar par paires vs Mnemos et applique la correction de Holm-Bonferroni."""
    eval_list: list[tuple[str, float]] = []

    # Épreuve 1
    if mnemos_t1 and mnemos_t1.instances_details:
        m_vec_1_1 = [r["passed_1_1"] for r in mnemos_t1.instances_details]
        m_vec_1_2 = [r["passed_1_2"] for r in mnemos_t1.instances_details]
        for t1 in t1_results:
            if t1.backend_name == mnemos_t1.backend_name:
                continue
            if t1.instances_details and len(t1.instances_details) == len(m_vec_1_1):
                t_vec_1_1 = [r["passed_1_1"] for r in t1.instances_details]
                t_vec_1_2 = [r["passed_1_2"] for r in t1.instances_details]
                p1, _ = mcnemar_test(m_vec_1_1, t_vec_1_1)
                p2, _ = mcnemar_test(m_vec_1_2, t_vec_1_2)
                eval_list.append((f"{t1.backend_name}_t1_1", p1))
                eval_list.append((f"{t1.backend_name}_t1_2", p2))

    # Épreuve 1b
    if mnemos_t1b and mnemos_t1b.instances_details:
        m_vec_1b_1 = [r["passed_1b_1"] for r in mnemos_t1b.instances_details]
        m_vec_1b_2 = [r["passed_1b_2"] for r in mnemos_t1b.instances_details]
        for t1b in t1b_results:
            if t1b.backend_name == mnemos_t1b.backend_name:
                continue
            if t1b.instances_details and len(t1b.instances_details) == len(m_vec_1b_1):
                t_vec_1b_1 = [r["passed_1b_1"] for r in t1b.instances_details]
                t_vec_1b_2 = [r["passed_1b_2"] for r in t1b.instances_details]
                p1, _ = mcnemar_test(m_vec_1b_1, t_vec_1b_1)
                p2, _ = mcnemar_test(m_vec_1b_2, t_vec_1b_2)
                eval_list.append((f"{t1b.backend_name}_t1b_1", p1))
                eval_list.append((f"{t1b.backend_name}_t1b_2", p2))

    # Épreuve 2
    if mnemos_t2 and mnemos_t2.instances_details:
        m_vec_2 = [r["passed_2_1"] for r in mnemos_t2.instances_details] + [r["passed_2_2"] for r in mnemos_t2.instances_details]
        for t2 in t2_results:
            if t2.backend_name == mnemos_t2.backend_name:
                continue
            if t2.instances_details and len(t2.instances_details) * 2 == len(m_vec_2):
                t_vec_2 = [r["passed_2_1"] for r in t2.instances_details] + [r["passed_2_2"] for r in t2.instances_details]
                p, _ = mcnemar_test(m_vec_2, t_vec_2)
                eval_list.append((f"{t2.backend_name}_t2", p))

    # Épreuve 3
    if mnemos_t3 and mnemos_t3.details:
        m_vec_3 = [r["verdict"] for r in mnemos_t3.details]
        for t3 in t3_results:
            if t3.backend_name == mnemos_t3.backend_name:
                continue
            if t3.details and len(t3.details) == len(m_vec_3):
                t_vec_3 = [r["verdict"] for r in t3.details]
                p, _ = mcnemar_test(m_vec_3, t_vec_3)
                eval_list.append((f"{t3.backend_name}_t3", p))

    if not eval_list:
        return {}

    raw_p_values = [p for _, p in eval_list]
    adj_p_values = holm_bonferroni_correction(raw_p_values)

    result_map: dict[str, str] = {}
    for (key, _), p_adj in zip(eval_list, adj_p_values, strict=True):
        if p_adj > 0.05:
            result_map[key] = f"non significatif (p_corr={p_adj:.3f})"
        else:
            result_map[key] = f"p_corr={p_adj:.3f}"

    return result_map


def _generate_markdown_report(
    config: BenchConfig,
    t1_results: list[GhostVectorResult],
    t1b_results: list[ExternalOntologyResult],
    t2_results: list[BlandNoiseResult],
    t3_results: list[ComplianceResult],
    output_path: Path,
    before_mnemos: dict[str, Any] | None = None,
) -> None:
    """Génère le rapport markdown comparatif neutre avec Wilson CI, % Oracle et McNemar corrigé par Holm."""
    hw = config.hardware
    lines: list[str] = [
        "# Rapport d'Évaluation Comparatif des Architectures de Mémoire",
        "",
        f"**Date d'évaluation** : {time.strftime('%Y-%m-%d %H:%M:%S')}  ",
        f"**Environnement Matériel** : {hw.get('cpu_model')} ({hw.get('cpu_cores')} vCPUs), {hw.get('ram_gb')} Go RAM  ",
        f"**Moteur LLM / Embeddings** : `{config.llm_model}` (dry_run: {config.dry_run}) | `{config.embed_model}`  ",
        f"**Méthode Statistique** : Évaluation déterministe (température {config.temperature}, seed {config.seed}) sur jeu Dev (70%, 14 instances/épreuve). Intervalles de Wilson à 95% ($z=1.96$), normalisation au plafond Oracle et tests de McNemar par paires vs Mnemos avec correction de Holm-Bonferroni sur l'ensemble des comparaisons multiples.",
        "",
    ]

    if config.dry_run:
        lines.extend([
            "> ⚠️ **DRY RUN — RÉSULTATS NON SIGNIFICATIFS (EXÉCUTION STUB SANS MODÈLE NEURONAL)**",
            "> *Ce rapport a été généré en mode simulation rapide sans appel réseau ni inférence de modèle.*",
            "",
        ])

    # Plafonds de l'Oracle
    oracle_t1 = next((r for r in t1_results if r.backend_name == "OracleControlMemory"), None)
    oracle_t1b = next((r for r in t1b_results if r.backend_name == "OracleControlMemory"), None)
    oracle_t2 = next((r for r in t2_results if r.backend_name == "OracleControlMemory"), None)
    oracle_t3 = next((r for r in t3_results if r.backend_name == "OracleControlMemory"), None)

    o_t1_1 = oracle_t1.probe_1_1_score if oracle_t1 else None
    o_t1_2 = oracle_t1.probe_1_2_score if oracle_t1 else None
    o_t1b_1 = oracle_t1b.probe_1b_1_score if oracle_t1b else None
    o_t1b_2 = oracle_t1b.probe_1b_2_score if oracle_t1b else None
    o_t2 = oracle_t2.critical_facts_answer_success_rate if oracle_t2 else None
    o_t3 = oracle_t3.compliance_score if oracle_t3 else None

    # Mnemos actuel (après correctif)
    mnemos_t1 = next((r for r in t1_results if r.backend_name in ("Mnemos", "MnemosBackend")), None)
    mnemos_t1b = next((r for r in t1b_results if r.backend_name in ("Mnemos", "MnemosBackend")), None)
    mnemos_t2 = next((r for r in t2_results if r.backend_name in ("Mnemos", "MnemosBackend")), None)
    mnemos_t3 = next((r for r in t3_results if r.backend_name in ("Mnemos", "MnemosBackend")), None)

    # Table avant / après Mnemos si disponible
    if before_mnemos:
        lines.extend([
            "---",
            "",
            "## Évolution de Mnemos : Avant vs Après Correctif (Commit 4f9fc9e + Loop Guard)",
            "",
            "| Épreuve / Sonde | Mnemos Avant Correctif | Mnemos Après Correctif | Évolution Absolue |",
            "|---|---|---|---|",
        ])
        b_t1_1 = before_mnemos.get("t1_1", 0.0)
        a_t1_1 = mnemos_t1.probe_1_1_score if mnemos_t1 else 0.0
        lines.append(f"| **Sonde 1.1 (Vérité active)** | {b_t1_1*100:.1f}% | {a_t1_1*100:.1f}% | {(a_t1_1 - b_t1_1)*100:+.1f}% |")

        b_t1_2 = before_mnemos.get("t1_2", 0.0)
        a_t1_2 = mnemos_t1.probe_1_2_score if mnemos_t1 else 0.0
        lines.append(f"| **Sonde 1.2 (Chronologie)** | {b_t1_2*100:.1f}% | {a_t1_2*100:.1f}% | {(a_t1_2 - b_t1_2)*100:+.1f}% |")

        b_t1b_1 = before_mnemos.get("t1b_1", 0.0)
        a_t1b_1 = mnemos_t1b.probe_1b_1_score if mnemos_t1b else 0.0
        lines.append(f"| **Sonde 1b.1 (Vérité hors ontologie)** | {b_t1b_1*100:.1f}% | {a_t1b_1*100:.1f}% | {(a_t1b_1 - b_t1b_1)*100:+.1f}% |")

        b_t1b_2 = before_mnemos.get("t1b_2", 0.0)
        a_t1b_2 = mnemos_t1b.probe_1b_2_score if mnemos_t1b else 0.0
        lines.append(f"| **Sonde 1b.2 (Chronologie hors ontologie)** | {b_t1b_2*100:.1f}% | {a_t1b_2*100:.1f}% | {(a_t1b_2 - b_t1b_2)*100:+.1f}% |")

        b_t2 = before_mnemos.get("t2", 0.0)
        a_t2 = mnemos_t2.critical_facts_answer_success_rate if mnemos_t2 else 0.0
        lines.append(f"| **Épreuve 2 (Rétention 90 jours)** | {b_t2*100:.1f}% | {a_t2*100:.1f}% | {(a_t2 - b_t2)*100:+.1f}% |")

        b_t3 = before_mnemos.get("t3", 0.0)
        a_t3 = mnemos_t3.compliance_score if mnemos_t3 else 0.0
        lines.append(f"| **Épreuve 3 (Conventions)** | {b_t3*100:.1f}% | {a_t3*100:.1f}% | {(a_t3 - b_t3)*100:+.1f}% |")

        lines.append("")

    # Calcul des p-valeurs McNemar avec correction de Holm
    mcn_map = _compute_holm_adjusted_mcnemar_dict(
        t1_results, t1b_results, t2_results, t3_results,
        mnemos_t1, mnemos_t1b, mnemos_t2, mnemos_t3,
    )

    lines.extend([
        "---",
        "",
        "## 1. Tableau Synthétique des Mesures (Jeu Dev)",
        "",
        "| Architecture | Famille | Sonde 1.1 [% Oracle] | Sonde 1.2 [% Oracle] | Sonde 1b.1 [% Oracle] | Sonde 1b.2 [% Oracle] | Rétention (90j) [% Oracle] | Conventions [% Oracle] | Lignes Actives / Archivées |",
        "|---|---|---|---|---|---|---|---|---|",
    ])

    for t1, t1b, t2, t3 in zip(t1_results, t1b_results, t2_results, t3_results, strict=True):
        rel_1_1 = _format_rel(t1.probe_1_1_score, o_t1_1)
        rel_1_2 = _format_rel(t1.probe_1_2_score, o_t1_2)
        rel_1b_1 = _format_rel(t1b.probe_1b_1_score, o_t1b_1)
        rel_1b_2 = _format_rel(t1b.probe_1b_2_score, o_t1b_2)
        rel_2 = _format_rel(t2.critical_facts_answer_success_rate, o_t2)
        rel_3 = _format_rel(t3.compliance_score, o_t3)

        p1_1_str = f"{t1.probe_1_1_score * 100:.0f}% [{rel_1_1}]"
        p1_2_str = f"{t1.probe_1_2_score * 100:.0f}% [{rel_1_2}]"
        p1b_1_str = f"{t1b.probe_1b_1_score * 100:.0f}% [{rel_1b_1}]"
        p1b_2_str = f"{t1b.probe_1b_2_score * 100:.0f}% [{rel_1b_2}]"
        p2_ret_str = f"{t2.critical_facts_answer_success_rate * 100:.0f}% [{rel_2}]"
        p3_comp_str = f"{t3.compliance_score * 100:.0f}% [{rel_3}]"
        items_str = f"{t2.storage_stats.active_items_count} / {t2.storage_stats.archived_items_count}"

        lines.append(
            f"| **{t1.backend_name}** | {t1.backend_family} | {p1_1_str} | {p1_2_str} | {p1b_1_str} | {p1b_2_str} | {p2_ret_str} | {p3_comp_str} | {items_str} |"
        )

    # ── ÉPREUVE 1 DÉTAILLÉE ──────────────────────────────────────────────────
    lines.extend([
        "",
        "---",
        "",
        "## 2. Analyse Détaillée par Épreuve",
        "",
        "### Épreuve 1 : Mutation Temporelle (Ontologie Fermée)",
        "",
        "| Système | Sonde 1.1 (Wilson 95%) | % Oracle | McNemar (Holm) | Sonde 1.2 (Wilson 95%) | % Oracle | McNemar (Holm) | Pollution Contexte |",
        "|---|---|---|---|---|---|---|---|",
    ])
    for t1 in t1_results:
        w1_1 = f"{t1.probe_1_1_score * 100:.0f}% [{t1.probe_1_1_wilson[0]*100:.0f}%-{t1.probe_1_1_wilson[1]*100:.0f}%]"
        rel_1_1 = _format_rel(t1.probe_1_1_score, o_t1_1)
        w1_2 = f"{t1.probe_1_2_score * 100:.0f}% [{t1.probe_1_2_wilson[0]*100:.0f}%-{t1.probe_1_2_wilson[1]*100:.0f}%]"
        rel_1_2 = _format_rel(t1.probe_1_2_score, o_t1_2)

        mcn_1_1 = "Réf (Mnemos)" if t1.backend_name in ("Mnemos", "MnemosBackend") else mcn_map.get(f"{t1.backend_name}_t1_1", "N/A")
        mcn_1_2 = "Réf (Mnemos)" if t1.backend_name in ("Mnemos", "MnemosBackend") else mcn_map.get(f"{t1.backend_name}_t1_2", "N/A")

        lines.append(
            f"| **{t1.backend_name}** | {w1_1} | {rel_1_1} | {mcn_1_1} | {w1_2} | {rel_1_2} | {mcn_1_2} | {t1.context_pollution_rate * 100:.1f}% |"
        )

    # ── ÉPREUVE 1B DÉTAILLÉE ─────────────────────────────────────────────────
    lines.extend([
        "",
        "### Épreuve 1b : Mutation Temporelle (Hors Ontologie Fermée)",
        "",
        "| Système | Sonde 1b.1 (Wilson 95%) | % Oracle | McNemar (Holm) | Sonde 1b.2 (Wilson 95%) | % Oracle | McNemar (Holm) | Pollution Contexte |",
        "|---|---|---|---|---|---|---|---|",
    ])
    for t1b in t1b_results:
        w1b_1 = f"{t1b.probe_1b_1_score * 100:.0f}% [{t1b.probe_1b_1_wilson[0]*100:.0f}%-{t1b.probe_1b_1_wilson[1]*100:.0f}%]"
        rel_1b_1 = _format_rel(t1b.probe_1b_1_score, o_t1b_1)
        w1b_2 = f"{t1b.probe_1b_2_score * 100:.0f}% [{t1b.probe_1b_2_wilson[0]*100:.0f}%-{t1b.probe_1b_2_wilson[1]*100:.0f}%]"
        rel_1b_2 = _format_rel(t1b.probe_1b_2_score, o_t1b_2)

        mcn_1b_1 = "Réf (Mnemos)" if t1b.backend_name in ("Mnemos", "MnemosBackend") else mcn_map.get(f"{t1b.backend_name}_t1b_1", "N/A")
        mcn_1b_2 = "Réf (Mnemos)" if t1b.backend_name in ("Mnemos", "MnemosBackend") else mcn_map.get(f"{t1b.backend_name}_t1b_2", "N/A")

        lines.append(
            f"| **{t1b.backend_name}** | {w1b_1} | {rel_1b_1} | {mcn_1b_1} | {w1b_2} | {rel_1b_2} | {mcn_1b_2} | {t1b.context_pollution_rate * 100:.1f}% |"
        )

    # ── ÉPREUVE 2 DÉTAILLÉE ──────────────────────────────────────────────────
    lines.extend([
        "",
        "### Épreuve 2 : Rétention des Faits Critiques (90 Jours)",
        "",
        "| Système | Réponse Exacte (Wilson 95%) | % Oracle | McNemar (Holm) | Rappel Contexte | Bruit Rappel | Lignes Actives / Archivées |",
        "|---|---|---|---|---|---|---|",
    ])
    for t2 in t2_results:
        w2 = f"{t2.critical_facts_answer_success_rate * 100:.0f}% [{t2.critical_facts_answer_wilson[0]*100:.0f}%-{t2.critical_facts_answer_wilson[1]*100:.0f}%]"
        rel_2 = _format_rel(t2.critical_facts_answer_success_rate, o_t2)
        mcn_2 = "Réf (Mnemos)" if t2.backend_name in ("Mnemos", "MnemosBackend") else mcn_map.get(f"{t2.backend_name}_t2", "N/A")

        lines.append(
            f"| **{t2.backend_name}** | {w2} | {rel_2} | {mcn_2} | {t2.critical_facts_context_recall_rate * 100:.0f}% | {t2.noise_ratio_in_recall * 100:.1f}% | {t2.storage_stats.active_items_count} / {t2.storage_stats.archived_items_count} |"
        )

    # ── ÉPREUVE 3 DÉTAILLÉE ──────────────────────────────────────────────────
    lines.extend([
        "",
        "### Épreuve 3 : Respect des Conventions de Projet Arbitraires",
        "",
        "| Système | Succès Conventions (Wilson 95%) | % Oracle | McNemar (Holm) | Règle Cible Rappelée | Règles en Contexte |",
        "|---|---|---|---|---|---|",
    ])
    for t3 in t3_results:
        w3 = f"{t3.compliance_score * 100:.0f}% [{t3.compliance_wilson[0]*100:.0f}%-{t3.compliance_wilson[1]*100:.0f}%]"
        rel_3 = _format_rel(t3.compliance_score, o_t3)
        mcn_3 = "Réf (Mnemos)" if t3.backend_name in ("Mnemos", "MnemosBackend") else mcn_map.get(f"{t3.backend_name}_t3", "N/A")

        lines.append(
            f"| **{t3.backend_name}** | {w3} | {rel_3} | {mcn_3} | {t3.rules_context_recall_rate * 100:.0f}% | {t3.avg_rules_in_context:.1f}/5 |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 3. Notes Méthodologiques & Précisions Techniques",
        "- **Intervalles de confiance de Wilson (95%)** : Calculés avec $z = 1.96$ pour chaque proportion de succès empirique.",
        "- **Normalisation au plafond Oracle (% Oracle)** : Défini par $\\text{Score}_{\\text{système}} / \\text{Score}_{\\text{oracle}}$. Permet d'isoler la contribution propre de la mémoire par rapport aux limites d'instruction du modèle.",
        "- **Test de McNemar avec correction de Holm-Bonferroni** : Test exact binomial bilatéral apparié sur chaque item identique entre Mnemos et les autres architectures. La correction séquentielle de Holm est appliquée à l'ensemble des tests pour contrôler rigoureusement le FWER (Family-Wise Error Rate). Toute comparaison affichant $p_{\\text{corr}} > 0.05$ est explicitement marquée *non significatif*.",
        "- **Condition négative stricte (Sondes 1.1 et 1b.1)** : Rejet formel si une valeur périmée est présentée comme actuelle.",
        "- **Mesure physique du stockage** : Comptage exact des `Lignes actives / Lignes archivées` pour quantifier la purge du bruit et la persistance des faits critiques.",
    ])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


async def run_all_benchmarks(
    config: BenchConfig,
    include_controls: bool = True,
    controls_only: bool = False,
    mnemos_only: bool = False,
    num_instances: int = 14,
    before_mnemos: dict[str, Any] | None = None,
) -> tuple[list[GhostVectorResult], list[ExternalOntologyResult], list[BlandNoiseResult], list[ComplianceResult]]:
    _set_seeds(config.seed)
    typer.echo("=" * 70)
    typer.echo("Démarrage de l'évaluation comparative des architectures de mémoire")
    typer.echo(f"Mode : {'DRY-RUN / STUB' if config.dry_run else 'RÉEL OLLAMA'}")
    typer.echo(f"Modèle LLM : {config.llm_model} | Embeddings : {config.embed_model}")
    typer.echo(f"Échantillon : {num_instances} instances par épreuve (Jeu Dev)")
    typer.echo("=" * 70)

    if mnemos_only:
        backends_to_run = [MnemosBackend]
    elif controls_only:
        backends_to_run = list(CONTROL_BACKENDS)
    elif include_controls:
        backends_to_run = list(CONTROL_BACKENDS) + list(ALL_BACKENDS)
    else:
        backends_to_run = list(ALL_BACKENDS)

    t1_results: list[GhostVectorResult] = []
    t1b_results: list[ExternalOntologyResult] = []
    t2_results: list[BlandNoiseResult] = []
    t3_results: list[ComplianceResult] = []
    backends_metadata: dict[str, Any] = {}

    ghost_instances = DEV_GHOST_INSTANCES[:num_instances]
    external_instances = DEV_EXTERNAL_INSTANCES[:num_instances]
    bland_instances = DEV_BLAND_INSTANCES[:num_instances]
    convention_prompts = DEV_CONVENTION_PROMPTS[:num_instances]

    for backend_cls in backends_to_run:
        backend: BaseMemoryBackend = backend_cls(config)
        typer.echo(f"\n▶ Évaluation du backend : {backend.name} [{backend.family}]")
        await backend.setup()

        # Épreuve 1
        typer.echo(f"  • Épreuve 1 : Mutation Temporelle ({len(ghost_instances)} instances)...")
        r1 = await run_test1_ghost_vector(backend, config, instances=ghost_instances)
        t1_results.append(r1)
        typer.echo(f"    -> Sonde 1.1: {r1.passed_1_1_count}/{r1.total_instances} ({r1.probe_1_1_score * 100:.0f}%) | Sonde 1.2: {r1.passed_1_2_count}/{r1.total_instances} ({r1.probe_1_2_score * 100:.0f}%)")

        # Épreuve 1b
        typer.echo(f"  • Épreuve 1b : Mutation Hors Ontologie ({len(external_instances)} instances)...")
        r1b = await run_test1b_external_ontology(backend, config, instances=external_instances)
        t1b_results.append(r1b)
        typer.echo(f"    -> Sonde 1b.1: {r1b.passed_1b_1_count}/{r1b.total_instances} ({r1b.probe_1b_1_score * 100:.0f}%) | Sonde 1b.2: {r1b.passed_1b_2_count}/{r1b.total_instances} ({r1b.probe_1b_2_score * 100:.0f}%)")

        # Épreuve 2
        typer.echo(f"  • Épreuve 2 : Bruit Bland & Décroissance ({len(bland_instances)} instances, 90 jours)...")
        r2 = await run_test2_bland_noise(backend, config, instances=bland_instances)
        t2_results.append(r2)
        typer.echo(f"    -> Rétention exacte: {r2.critical_facts_answer_success_rate * 100:.0f}% | Bruit: {r2.noise_ratio_in_recall * 100:.0f}%")

        # Épreuve 3
        typer.echo(f"  • Épreuve 3 : Conventions Arbitraires ({len(convention_prompts)} requêtes)...")
        r3 = await run_test3_compliance(backend, config, prompts=convention_prompts)
        t3_results.append(r3)
        typer.echo(f"    -> Conventions respectées: {r3.passed_count}/{r3.total_prompts} ({r3.compliance_score * 100:.0f}%)")

        backends_metadata[backend.name] = {
            "family": backend.family,
            "embedder_class": backend.get_embedder_class(),
            "llm_class": backend.get_llm_class(),
            "consolidation_errors": getattr(backend, "consolidation_errors", 0),
        }

        await backend.teardown()

    config.results_dir.mkdir(parents=True, exist_ok=True)
    report_md_path = config.results_dir / "report.md"
    metrics_json_path = config.results_dir / "metrics.json"

    _generate_markdown_report(config, t1_results, t1b_results, t2_results, t3_results, report_md_path, before_mnemos=before_mnemos)

    raw_metrics = {
        "metadata": {
            "timestamp": time.time(),
            "llm_model": config.llm_model,
            "embed_model": config.embed_model,
            "temperature": config.temperature,
            "seed": config.seed,
            "dry_run": config.dry_run,
            "num_instances": num_instances,
            "hardware": config.hardware,
            "backends": backends_metadata,
            "before_mnemos": before_mnemos,
        },
        "test1_ghost_vector": [asdict(r) for r in t1_results],
        "test1b_external_ontology": [asdict(r) for r in t1b_results],
        "test2_bland_noise": [asdict(r) for r in t2_results],
        "test3_compliance": [asdict(r) for r in t3_results],
    }
    metrics_json_path.write_text(json.dumps(raw_metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    typer.echo("\n" + "=" * 70)
    typer.echo(f"Rapport généré : {report_md_path}")
    typer.echo(f"Métriques JSON : {metrics_json_path}")
    typer.echo("=" * 70)

    return t1_results, t1b_results, t2_results, t3_results


@app.command()
def run(
    dry_run: bool = typer.Option(False, "--dry-run", "--stub", help="Exécution déterministe ultra-rapide sans appel LLM"),
    model: str = typer.Option("qwen2.5:3b", "--model", help="Modèle Ollama pour les inférences d'évaluation"),
    embed_model: str = typer.Option("bge-m3:latest", "--embed-model", help="Modèle d'embeddings"),
    temperature: float = typer.Option(0.0, "--temperature", help="Température LLM déterministe"),
    seed: int = typer.Option(42, "--seed", help="Seed aléatoire déterministe"),
    include_controls: bool = typer.Option(True, "--include-controls/--no-controls", help="Inclure les mémoires de contrôle"),
    controls_only: bool = typer.Option(False, "--controls-only", help="Exécuter uniquement les mémoires de contrôle"),
    mnemos_only: bool = typer.Option(False, "--mnemos-only", help="Exécuter uniquement le backend Mnemos"),
    num_instances: int = typer.Option(14, "--num-instances", help="Nombre d'instances par épreuve"),
    before_mnemos_file: Path = typer.Option(
        Path("bench/results/mnemos_before_metrics.json"),
        "--before-mnemos-file",
        help="Chemin vers le fichier JSON des métriques Mnemos avant correctif",
    ),
    bench_dir: Path = typer.Option(Path("data/bench_superiority"), "--bench-dir", help="Dossier isolé de données"),
    results_dir: Path = typer.Option(Path("bench/results"), "--results-dir", help="Dossier de sortie des rapports"),
) -> None:
    config = BenchConfig(
        bench_dir=bench_dir,
        embed_model=embed_model,
        llm_model=model,
        dry_run=dry_run,
        temperature=temperature,
        seed=seed,
        results_dir=results_dir,
    )
    before_mnemos_data = None
    if before_mnemos_file and before_mnemos_file.exists():
        try:
            before_mnemos_data = json.loads(before_mnemos_file.read_text(encoding="utf-8"))
        except Exception:
            before_mnemos_data = None

    asyncio.run(
        run_all_benchmarks(
            config,
            include_controls=include_controls,
            controls_only=controls_only,
            mnemos_only=mnemos_only,
            num_instances=num_instances,
            before_mnemos=before_mnemos_data,
        )
    )


if __name__ == "__main__":
    app()

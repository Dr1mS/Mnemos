"""Scénario 2 : Résistance au Bruit & Rétention des Faits Critiques (90 Jours).

Mesure de façon neutre :
1. Rappel des faits critiques dans le contexte récupéré (hit en contexte).
2. Taux de succès des sondes 2.1 et 2.2 évalué EXCLUSIVEMENT sur la réponse du LLM.
3. Proportion de bruit dans les éléments récupérés (top-k).
4. Empreinte de stockage réelle mesurée via StorageStats (lignes actives, lignes archivées).
5. Intervalles de confiance de Wilson à 95% sur l'ensemble des faits critiques évalués.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from bench.backends.base import BaseMemoryBackend
from bench.config import BenchConfig
from bench.datasets import BlandNoiseInstance, DEV_BLAND_INSTANCES, build_bland_noise_stream
from bench.eval_utils import check_answer_contains, generate_llm_detailed, normalize_text, wilson_score_interval
from bench.storage_utils import StorageStats


@dataclass
class BlandNoiseResult:
    backend_name: str
    backend_family: str
    critical_facts_context_recall_rate: float  # % de faits critiques présents dans le contexte rappelé
    critical_facts_answer_success_rate: float  # % de faits critiques correctement restitués dans la réponse
    critical_facts_answer_wilson: tuple[float, float] = (0.0, 0.0)  # Wilson 95%
    noise_ratio_in_recall: float = 0.0  # Proportion de bruit dans le rappel combiné
    storage_stats: StorageStats = field(default_factory=lambda: StorageStats(0, 0, 0, 0, 0))
    probe_2_1_passed: bool = False  # Vrai si clé de staging restituée
    probe_2_2_passed: bool = False  # Vrai si allergie/contrainte restituée
    passed_2_1_count: int = 0
    passed_2_2_count: int = 0
    total_instances: int = 1
    context_hit_2_1: bool = False
    context_hit_2_2: bool = False
    embedder_class: str = ""
    llm_class: str = ""
    llm_answer_2_1: str = ""
    llm_answer_2_2: str = ""
    probe_details: dict[str, Any] = field(default_factory=dict)
    instances_details: list[dict[str, Any]] = field(default_factory=list)


async def run_single_bland_instance(
    backend: BaseMemoryBackend,
    config: BenchConfig,
    inst: BlandNoiseInstance,
) -> dict[str, Any]:
    await backend.reset()

    stream = build_bland_noise_stream(inst, total_turns=100)
    for turn in stream:
        await backend.write(
            content=turn.content,
            role=turn.role,
            timestamp_offset_days=turn.timestamp_offset_days,
            metadata={"is_critical": turn.is_critical, "salience": turn.salience_hint},
        )

    # Avancement temporel simulé de 90 jours
    await backend.advance_time_days(90.0)

    # ── SONDE 2.1 : Clé Staging ──────────────────────────────────────────────
    recalled_2_1 = await backend.recall(inst.probe_2_1, k=4)
    context_str_2_1 = "\n".join(recalled_2_1)
    norm_ctx_2_1 = normalize_text(context_str_2_1)

    norm_key = normalize_text(inst.key_code)
    context_hit_2_1 = norm_key in norm_ctx_2_1

    prompt_2_1 = (
        f"Tu es un assistant technique concis.\n"
        f"Informations en mémoire :\n\"\"\"\n{context_str_2_1}\n\"\"\"\n\n"
        f"Question : {inst.probe_2_1}\n"
        f"Réponds en donnant directement la clé exacte."
    )
    gen_2_1 = await generate_llm_detailed(prompt_2_1, config)
    answer_2_1 = gen_2_1.cleaned_answer

    passed_2_1 = check_answer_contains(answer_2_1, inst.key_code) or check_answer_contains(answer_2_1, norm_key)

    # ── SONDE 2.2 : Contrainte Vitale / Allergie ──────────────────────────────
    recalled_2_2 = await backend.recall(inst.probe_2_2, k=4)
    context_str_2_2 = "\n".join(recalled_2_2)
    norm_ctx_2_2 = normalize_text(context_str_2_2)

    norm_allergy = normalize_text(inst.allergy)
    context_hit_2_2 = norm_allergy in norm_ctx_2_2

    prompt_2_2 = (
        f"Tu es un assistant médical de sécurité concis.\n"
        f"Informations en mémoire :\n\"\"\"\n{context_str_2_2}\n\"\"\"\n\n"
        f"Question : {inst.probe_2_2}\n"
        f"Nomme directement l'aliment ou la contrainte à risque mortel."
    )
    gen_2_2 = await generate_llm_detailed(prompt_2_2, config)
    answer_2_2 = gen_2_2.cleaned_answer

    passed_2_2 = check_answer_contains(answer_2_2, inst.allergy) or check_answer_contains(answer_2_2, norm_allergy)

    # Calcul du bruit dans le rappel
    all_recalled = recalled_2_1 + recalled_2_2
    total_recalled_items = max(1, len(all_recalled))
    bland_noise_count = sum(
        1 for item in all_recalled
        if any(w in normalize_text(item) for w in ["observation", "soleil", "cafe", "bureau", "souris", "ventilateur"])
    )
    noise_ratio = round(bland_noise_count / total_recalled_items, 3)

    return {
        "instance_id": inst.id,
        "key_code": inst.key_code,
        "allergy": inst.allergy,
        "context_hit_2_1": context_hit_2_1,
        "context_hit_2_2": context_hit_2_2,
        "passed_2_1": passed_2_1,
        "passed_2_2": passed_2_2,
        "noise_ratio": noise_ratio,
        "recalled_2_1": recalled_2_1,
        "recalled_2_2": recalled_2_2,
        "answer_2_1": answer_2_1,
        "answer_2_2": answer_2_2,
        "raw_2_1": gen_2_1.raw_answer,
        "raw_2_2": gen_2_2.raw_answer,
        "duration_2_1": gen_2_1.duration_s,
        "duration_2_2": gen_2_2.duration_s,
    }


async def run_test2_bland_noise(
    backend: BaseMemoryBackend,
    config: BenchConfig,
    instances: list[BlandNoiseInstance] | None = None,
) -> BlandNoiseResult:
    target_instances = instances if instances is not None else DEV_BLAND_INSTANCES

    instances_results: list[dict[str, Any]] = []
    for inst in target_instances:
        res = await run_single_bland_instance(backend, config, inst)
        instances_results.append(res)

    total_instances = max(1, len(instances_results))
    passed_2_1_count = sum(1 for r in instances_results if r["passed_2_1"])
    passed_2_2_count = sum(1 for r in instances_results if r["passed_2_2"])
    context_hits = sum(1 for r in instances_results if r["context_hit_2_1"]) + sum(1 for r in instances_results if r["context_hit_2_2"])
    total_probes = total_instances * 2

    context_recall_rate = round(context_hits / total_probes, 3)
    total_passed = passed_2_1_count + passed_2_2_count
    p_ans, w_low, w_high = wilson_score_interval(total_passed, total_probes)
    avg_noise = round(sum(r["noise_ratio"] for r in instances_results) / total_instances, 3)

    storage_stats = await backend.get_storage_stats()
    last_res = instances_results[-1]

    return BlandNoiseResult(
        backend_name=backend.name,
        backend_family=backend.family,
        critical_facts_context_recall_rate=context_recall_rate,
        critical_facts_answer_success_rate=p_ans,
        critical_facts_answer_wilson=(w_low, w_high),
        noise_ratio_in_recall=avg_noise,
        storage_stats=storage_stats,
        probe_2_1_passed=passed_2_1_count == total_instances,
        probe_2_2_passed=passed_2_2_count == total_instances,
        passed_2_1_count=passed_2_1_count,
        passed_2_2_count=passed_2_2_count,
        total_instances=total_instances,
        context_hit_2_1=last_res["context_hit_2_1"],
        context_hit_2_2=last_res["context_hit_2_2"],
        embedder_class=backend.get_embedder_class(),
        llm_class=backend.get_llm_class(),
        llm_answer_2_1=last_res["answer_2_1"],
        llm_answer_2_2=last_res["answer_2_2"],
        probe_details={
            "probe_2_1": {
                "question": "Quelle est la clé d'authentification du serveur de staging ?",
                "raw_answer": last_res["raw_2_1"],
                "cleaned_answer": last_res["answer_2_1"],
                "expected": last_res["key_code"],
                "context_hit": last_res["context_hit_2_1"],
                "verdict": last_res["passed_2_1"],
                "duration_s": last_res["duration_2_1"],
                "context": last_res["recalled_2_1"],
            },
            "probe_2_2": {
                "question": "Quel ingrédient alimentaire présente un risque mortel pour moi ?",
                "raw_answer": last_res["raw_2_2"],
                "cleaned_answer": last_res["answer_2_2"],
                "expected": last_res["allergy"],
                "context_hit": last_res["context_hit_2_2"],
                "verdict": last_res["passed_2_2"],
                "duration_s": last_res["duration_2_2"],
                "context": last_res["recalled_2_2"],
            },
        },
        instances_details=instances_results,
    )

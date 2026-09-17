"""Scénario 1 : Mutation Temporelle & Effet Fantôme.

Mesure de façon distincte :
1. Taux de pollution du contexte (% d'éléments périmés dans le top-k).
2. Rappel du fait actif dans le contexte (Ville active présente dans la mémoire récupérée).
3. Sonde 1.1 (Vérité active) : Jugée EXCLUSIVEMENT sur la réponse du LLM avec condition négative stricte.
4. Sonde 1.2 (Auditabilité chronologique) : Jugée EXCLUSIVEMENT sur la réponse du LLM.
5. Intervalles de confiance de Wilson à 95% calculés sur l'échantillon d'instances.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from bench.backends.base import BaseMemoryBackend
from bench.config import BenchConfig
from bench.datasets import DEV_GHOST_INSTANCES, GhostVectorInstance
from bench.eval_utils import (
    check_active_residence_answer,
    check_chronological_sequence,
    generate_llm_detailed,
    normalize_text,
    wilson_score_interval,
)


@dataclass
class GhostVectorResult:
    backend_name: str
    backend_family: str
    context_pollution_rate: float  # % moyen d'éléments périmés dans le rappel
    active_fact_in_context: bool  # Vrai si le fait actif est présent
    active_fact_evicted: bool  # Vrai si absent ou évincé par des faits périmés
    probe_1_1_passed: bool  # Vrai si la réponse valide la ville active sans contradiction
    probe_1_2_passed: bool  # Vrai si la réponse restitue la chronologie exacte
    probe_1_1_score: float = 0.0  # Proportion de succès sonde 1.1 [0..1]
    probe_1_1_wilson: tuple[float, float] = (0.0, 0.0)  # Intervalle Wilson 95%
    probe_1_2_score: float = 0.0  # Proportion de succès sonde 1.2 [0..1]
    probe_1_2_wilson: tuple[float, float] = (0.0, 0.0)
    passed_1_1_count: int = 0
    passed_1_2_count: int = 0
    total_instances: int = 1
    embedder_class: str = ""
    llm_class: str = ""
    retrieved_items_1_1: list[str] = field(default_factory=list)
    retrieved_items_1_2: list[str] = field(default_factory=list)
    llm_answer_1_1: str = ""
    llm_answer_1_2: str = ""
    probe_details: dict[str, Any] = field(default_factory=dict)
    instances_details: list[dict[str, Any]] = field(default_factory=list)


async def run_single_ghost_instance(
    backend: BaseMemoryBackend,
    config: BenchConfig,
    inst: GhostVectorInstance,
) -> dict[str, Any]:
    """Exécute une instance du scénario Ghost Vector."""
    await backend.reset()

    # 1. T0 : Première ville
    await backend.write(f"J'habite à {inst.city_0}.", role="user", timestamp_offset_days=0.0)
    await backend.write(f"Bien noté, vous résidez à {inst.city_0}.", role="assistant", timestamp_offset_days=0.0)

    # Tours neutres bloc 1
    for user_q, asst_a in inst.neutral_turns_1:
        await backend.write(user_q, role="user", timestamp_offset_days=0.05)
        await backend.write(asst_a, role="assistant", timestamp_offset_days=0.0)

    # 2. T1 : Déménagement deuxième ville
    await backend.write(f"J'ai déménagé, j'habite maintenant à {inst.city_1}.", role="user", timestamp_offset_days=1.0)
    await backend.write(f"Félicitations pour votre emménagement à {inst.city_1} !", role="assistant", timestamp_offset_days=0.0)

    # Tours neutres bloc 2
    for user_q, asst_a in inst.neutral_turns_2:
        await backend.write(user_q, role="user", timestamp_offset_days=0.05)
        await backend.write(asst_a, role="assistant", timestamp_offset_days=0.0)

    # 3. T2 : Ville active finale
    await backend.write(f"Finalement je vis à {inst.city_2}.", role="user", timestamp_offset_days=2.0)
    await backend.write(f"C'est noté, vous êtes maintenant installé à {inst.city_2}.", role="assistant", timestamp_offset_days=0.0)

    # ── SONDE 1.1 : Vérité Active ───────────────────────────────────────────
    recalled_1_1 = await backend.recall(inst.probe_1_1, k=4)
    context_str_1_1 = "\n".join(recalled_1_1)
    norm_ctx_1_1 = normalize_text(context_str_1_1)

    norm_c0 = normalize_text(inst.city_0)
    norm_c1 = normalize_text(inst.city_1)
    norm_c2 = normalize_text(inst.city_2)

    active_in_ctx = norm_c2 in norm_ctx_1_1
    total_slots = max(1, len(recalled_1_1))
    pollution_count = sum(1 for item in recalled_1_1 if any(s in normalize_text(item) for s in [norm_c0, norm_c1]))
    pollution_rate = round(pollution_count / total_slots, 3)
    evicted = not active_in_ctx or (active_in_ctx and pollution_rate >= 0.5)

    prompt_1_1 = (
        f"Tu es un assistant concis et factuel.\n"
        f"Voici les informations extraites de ta mémoire :\n"
        f"\"\"\"\n{context_str_1_1}\n\"\"\"\n\n"
        f"Question : {inst.probe_1_1}\n"
        f"Réponds en une phrase directe en indiquant ta ville de résidence actuelle."
    )
    gen_1_1 = await generate_llm_detailed(prompt_1_1, config)
    answer_1_1 = gen_1_1.cleaned_answer

    # Validation stricte Sonde 1.1 avec condition négative
    passed_1_1, reason_1_1 = check_active_residence_answer(
        answer_1_1, inst.city_2, [inst.city_0, inst.city_1]
    )

    # ── SONDE 1.2 : Auditabilité Temporelle ──────────────────────────────────
    recalled_1_2 = await backend.recall(inst.probe_1_2, k=6)
    context_str_1_2 = "\n".join(recalled_1_2)

    prompt_1_2 = (
        f"Tu es un auditeur système précis.\n"
        f"Voici l'historique restitué par la mémoire :\n"
        f"\"\"\"\n{context_str_1_2}\n\"\"\"\n\n"
        f"Question : {inst.probe_1_2}\n"
        f"Donne la séquence chronologique exacte de la plus ancienne à la plus récente."
    )
    gen_1_2 = await generate_llm_detailed(prompt_1_2, config)
    answer_1_2 = gen_1_2.cleaned_answer

    expected_seq = [inst.city_0, inst.city_1, inst.city_2]
    passed_1_2 = check_chronological_sequence(answer_1_2, expected_seq)

    return {
        "instance_id": inst.id,
        "cities": (inst.city_0, inst.city_1, inst.city_2),
        "context_pollution_rate": pollution_rate,
        "active_in_ctx": active_in_ctx,
        "evicted": evicted,
        "passed_1_1": passed_1_1,
        "reason_1_1": reason_1_1,
        "passed_1_2": passed_1_2,
        "recalled_1_1": recalled_1_1,
        "recalled_1_2": recalled_1_2,
        "answer_1_1": answer_1_1,
        "answer_1_2": answer_1_2,
        "raw_1_1": gen_1_1.raw_answer,
        "raw_1_2": gen_1_2.raw_answer,
        "duration_1_1": gen_1_1.duration_s,
        "duration_1_2": gen_1_2.duration_s,
    }


async def run_test1_ghost_vector(
    backend: BaseMemoryBackend,
    config: BenchConfig,
    instances: list[GhostVectorInstance] | None = None,
) -> GhostVectorResult:
    target_instances = instances if instances is not None else DEV_GHOST_INSTANCES

    instances_results: list[dict[str, Any]] = []
    for inst in target_instances:
        res = await run_single_ghost_instance(backend, config, inst)
        instances_results.append(res)

    total = max(1, len(instances_results))
    passed_1_1_count = sum(1 for r in instances_results if r["passed_1_1"])
    passed_1_2_count = sum(1 for r in instances_results if r["passed_1_2"])
    avg_pollution = round(sum(r["context_pollution_rate"] for r in instances_results) / total, 3)
    active_in_ctx_count = sum(1 for r in instances_results if r["active_in_ctx"])
    evicted_count = sum(1 for r in instances_results if r["evicted"])

    p1_1, w1_low, w1_high = wilson_score_interval(passed_1_1_count, total)
    p1_2, w2_low, w2_high = wilson_score_interval(passed_1_2_count, total)

    last_res = instances_results[-1]

    return GhostVectorResult(
        backend_name=backend.name,
        backend_family=backend.family,
        context_pollution_rate=avg_pollution,
        active_fact_in_context=active_in_ctx_count == total,
        active_fact_evicted=evicted_count > (total / 2),
        probe_1_1_passed=(passed_1_1_count == total),
        probe_1_2_passed=(passed_1_2_count == total),
        probe_1_1_score=p1_1,
        probe_1_1_wilson=(w1_low, w1_high),
        probe_1_2_score=p1_2,
        probe_1_2_wilson=(w2_low, w2_high),
        passed_1_1_count=passed_1_1_count,
        passed_1_2_count=passed_1_2_count,
        total_instances=total,
        embedder_class=backend.get_embedder_class(),
        llm_class=backend.get_llm_class(),
        retrieved_items_1_1=last_res["recalled_1_1"],
        retrieved_items_1_2=last_res["recalled_1_2"],
        llm_answer_1_1=last_res["answer_1_1"],
        llm_answer_1_2=last_res["answer_1_2"],
        probe_details={
            "probe_1_1": {
                "question": "Où est-ce que j'habite actuellement ?",
                "raw_answer": last_res["raw_1_1"],
                "cleaned_answer": last_res["answer_1_1"],
                "verdict": last_res["passed_1_1"],
                "reason": last_res["reason_1_1"],
                "duration_s": last_res["duration_1_1"],
                "context": last_res["recalled_1_1"],
            },
            "probe_1_2": {
                "question": "Quelles ont été mes villes de résidence dans l'ordre chronologique ?",
                "raw_answer": last_res["raw_1_2"],
                "cleaned_answer": last_res["answer_1_2"],
                "verdict": last_res["passed_1_2"],
                "duration_s": last_res["duration_1_2"],
                "context": last_res["recalled_1_2"],
            },
        },
        instances_details=instances_results,
    )

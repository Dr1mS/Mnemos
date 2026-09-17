"""Scénario 1b : Mutation Temporelle Hors Ontologie Fermée.

Évalue la capacité de la mémoire à gérer des faits qui changent dans le temps
lorsque les prédicats ne font PAS partie d'une ontologie relationnelle fermée
(serveur de prod, manager d'équipe, outil de CI/CD, cluster k8s, etc.).

Mesure :
1. Taux de pollution du contexte (% d'éléments périmés dans le rappel).
2. Rappel du fait actif dans le contexte.
3. Sonde 1b.1 (Vérité active hors ontologie) : jugée avec condition négative stricte.
4. Sonde 1b.2 (Auditabilité chronologique hors ontologie) : jugée sur l'ordre chronologique.
5. Intervalles de confiance de Wilson à 95%.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from bench.backends.base import BaseMemoryBackend
from bench.config import BenchConfig
from bench.datasets import DEV_EXTERNAL_INSTANCES, ExternalOntologyInstance
from bench.eval_utils import (
    check_active_fact_answer,
    check_chronological_sequence,
    generate_llm_detailed,
    normalize_text,
    wilson_score_interval,
)


@dataclass
class ExternalOntologyResult:
    backend_name: str
    backend_family: str
    context_pollution_rate: float
    active_fact_in_context: bool
    active_fact_evicted: bool
    probe_1b_1_passed: bool
    probe_1b_2_passed: bool
    probe_1b_1_score: float = 0.0
    probe_1b_1_wilson: tuple[float, float] = (0.0, 0.0)
    probe_1b_2_score: float = 0.0
    probe_1b_2_wilson: tuple[float, float] = (0.0, 0.0)
    passed_1b_1_count: int = 0
    passed_1b_2_count: int = 0
    total_instances: int = 1
    embedder_class: str = ""
    llm_class: str = ""
    retrieved_items_1_1: list[str] = field(default_factory=list)
    retrieved_items_1_2: list[str] = field(default_factory=list)
    llm_answer_1_1: str = ""
    llm_answer_1_2: str = ""
    probe_details: dict[str, Any] = field(default_factory=dict)
    instances_details: list[dict[str, Any]] = field(default_factory=list)


async def run_single_external_instance(
    backend: BaseMemoryBackend,
    config: BenchConfig,
    inst: ExternalOntologyInstance,
) -> dict[str, Any]:
    """Exécute une instance du scénario Mutation Hors Ontologie."""
    await backend.reset()

    # 1. T0 : Fait initial
    await backend.write(inst.statement_0, role="user", timestamp_offset_days=0.0)
    await backend.write(f"Bien noté pour {inst.topic}.", role="assistant", timestamp_offset_days=0.0)

    # Tours neutres bloc 1
    for user_q, asst_a in inst.neutral_turns_1:
        await backend.write(user_q, role="user", timestamp_offset_days=0.05)
        await backend.write(asst_a, role="assistant", timestamp_offset_days=0.0)

    # 2. T1 : Deuxième fait (mutation)
    await backend.write(inst.statement_1, role="user", timestamp_offset_days=1.0)
    await backend.write(f"C'est noté pour la mise à jour de {inst.topic}.", role="assistant", timestamp_offset_days=0.0)

    # Tours neutres bloc 2
    for user_q, asst_a in inst.neutral_turns_2:
        await backend.write(user_q, role="user", timestamp_offset_days=0.05)
        await backend.write(asst_a, role="assistant", timestamp_offset_days=0.0)

    # 3. T2 : Fait actif final
    await backend.write(inst.statement_2, role="user", timestamp_offset_days=2.0)
    await backend.write(f"Parfait, information enregistrée pour {inst.topic}.", role="assistant", timestamp_offset_days=0.0)

    # ── SONDE 1b.1 : Vérité Active Hors Ontologie ────────────────────────────
    recalled_1_1 = await backend.recall(inst.probe_1_1, k=4)
    context_str_1_1 = "\n".join(recalled_1_1)
    norm_ctx_1_1 = normalize_text(context_str_1_1)

    norm_v0 = normalize_text(inst.val_0)
    norm_v1 = normalize_text(inst.val_1)
    norm_v2 = normalize_text(inst.val_2)

    active_in_ctx = norm_v2 in norm_ctx_1_1
    total_slots = max(1, len(recalled_1_1))
    pollution_count = sum(1 for item in recalled_1_1 if any(s in normalize_text(item) for s in [norm_v0, norm_v1]))
    pollution_rate = round(pollution_count / total_slots, 3)
    evicted = not active_in_ctx or (active_in_ctx and pollution_rate >= 0.5)

    prompt_1_1 = (
        f"Tu es un assistant technique concis et factuel.\n"
        f"Voici les informations extraites de ta mémoire :\n"
        f"\"\"\"\n{context_str_1_1}\n\"\"\"\n\n"
        f"Question : {inst.probe_1_1}\n"
        f"Réponds en une phrase directe en indiquant l'information active actuelle."
    )
    gen_1_1 = await generate_llm_detailed(prompt_1_1, config)
    answer_1_1 = gen_1_1.cleaned_answer

    # Validation stricte Sonde 1b.1 avec condition négative
    passed_1_1, reason_1_1 = check_active_fact_answer(
        answer_1_1, inst.val_2, [inst.val_0, inst.val_1]
    )

    # ── SONDE 1b.2 : Auditabilité Temporelle Hors Ontologie ───────────────────
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

    expected_seq = [inst.val_0, inst.val_1, inst.val_2]
    passed_1_2 = check_chronological_sequence(answer_1_2, expected_seq)

    return {
        "instance_id": inst.id,
        "topic": inst.topic,
        "values": (inst.val_0, inst.val_1, inst.val_2),
        "context_pollution_rate": pollution_rate,
        "active_in_ctx": active_in_ctx,
        "evicted": evicted,
        "passed_1b_1": passed_1_1,
        "reason_1b_1": reason_1_1,
        "passed_1b_2": passed_1_2,
        "recalled_1_1": recalled_1_1,
        "recalled_1_2": recalled_1_2,
        "answer_1_1": answer_1_1,
        "answer_1_2": answer_1_2,
        "raw_1_1": gen_1_1.raw_answer,
        "raw_1_2": gen_1_2.raw_answer,
        "duration_1_1": gen_1_1.duration_s,
        "duration_1_2": gen_1_2.duration_s,
    }


async def run_test1b_external_ontology(
    backend: BaseMemoryBackend,
    config: BenchConfig,
    instances: list[ExternalOntologyInstance] | None = None,
) -> ExternalOntologyResult:
    target_instances = instances if instances is not None else DEV_EXTERNAL_INSTANCES

    instances_results: list[dict[str, Any]] = []
    for inst in target_instances:
        res = await run_single_external_instance(backend, config, inst)
        instances_results.append(res)

    total = max(1, len(instances_results))
    passed_1_1_count = sum(1 for r in instances_results if r["passed_1b_1"])
    passed_1_2_count = sum(1 for r in instances_results if r["passed_1b_2"])
    avg_pollution = round(sum(r["context_pollution_rate"] for r in instances_results) / total, 3)
    active_in_ctx_count = sum(1 for r in instances_results if r["active_in_ctx"])
    evicted_count = sum(1 for r in instances_results if r["evicted"])

    p1_1, w1_low, w1_high = wilson_score_interval(passed_1_1_count, total)
    p1_2, w2_low, w2_high = wilson_score_interval(passed_1_2_count, total)

    last_res = instances_results[-1]

    return ExternalOntologyResult(
        backend_name=backend.name,
        backend_family=backend.family,
        context_pollution_rate=avg_pollution,
        active_fact_in_context=active_in_ctx_count == total,
        active_fact_evicted=evicted_count > (total / 2),
        probe_1b_1_passed=(passed_1_1_count == total),
        probe_1b_2_passed=(passed_1_2_count == total),
        probe_1b_1_score=p1_1,
        probe_1b_1_wilson=(w1_low, w1_high),
        probe_1b_2_score=p1_2,
        probe_1b_2_wilson=(w2_low, w2_high),
        passed_1b_1_count=passed_1_1_count,
        passed_1b_2_count=passed_1_2_count,
        total_instances=total,
        embedder_class=backend.get_embedder_class(),
        llm_class=backend.get_llm_class(),
        retrieved_items_1_1=last_res["recalled_1_1"],
        retrieved_items_1_2=last_res["recalled_1_2"],
        llm_answer_1_1=last_res["answer_1_1"],
        llm_answer_1_2=last_res["answer_1_2"],
        probe_details={
            "probe_1b_1": {
                "question": last_res.get("topic", "Fait actif hors ontologie"),
                "raw_answer": last_res["raw_1_1"],
                "cleaned_answer": last_res["answer_1_1"],
                "verdict": last_res["passed_1b_1"],
                "reason": last_res["reason_1b_1"],
                "duration_s": last_res["duration_1_1"],
                "context": last_res["recalled_1_1"],
            },
            "probe_1b_2": {
                "question": f"Historique chronologique {last_res.get('topic', '')}",
                "raw_answer": last_res["raw_1_2"],
                "cleaned_answer": last_res["answer_1_2"],
                "verdict": last_res["passed_1b_2"],
                "duration_s": last_res["duration_1_2"],
                "context": last_res["recalled_1_2"],
            },
        },
        instances_details=instances_results,
    )

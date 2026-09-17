"""Scénario 3 : Respect des Conventions de Projet Arbitraires & Résistance à la Dilution.

Ce test mesure la capacité réelle de la mémoire à imposer des contraintes de projet arbitraires
qu'un modèle ne peut pas deviner sans contexte mémoire (ex: httpx obligatoire, dates AAAA/JJ/MM,
préfixe PRJ-, header X-Project-Token, cache /var/project/cache/).

Critère de validité formel :
- Mémoire vide (EmptyControlMemory) : score proche de 0%.
- Mémoire parfaite (OracleControlMemory) : score proche de 100%.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from bench.backends.base import BaseMemoryBackend
from bench.config import BenchConfig
from bench.datasets import (
    ALL_RULE_MARKERS,
    CONVENTION_BASELINE_RULES,
    COOPERATIVE_25_TURNS,
    DEV_CONVENTION_PROMPTS,
    ProjectConventionPrompt,
)
from bench.eval_utils import generate_llm_detailed, normalize_text, wilson_score_interval


@dataclass
class ComplianceResult:
    backend_name: str
    backend_family: str
    compliance_score: float  # Proportion de conventions respectées [0..1]
    compliance_wilson: tuple[float, float]  # Intervalle de Wilson à 95% (lower, upper)
    passed_count: int  # Nombre de requêtes respectant la convention
    total_prompts: int  # Nombre total de requêtes (ex: 14 en dev)
    rules_context_recall_rate: float  # % de requêtes où la règle cible est dans le rappel
    avg_rules_in_context: float  # Nombre moyen de règles trouvées dans le rappel
    system_role_recalled_rate: float  # % de requêtes où le rôle 'system' est présent dans le rappel
    embedder_class: str = ""
    llm_class: str = ""
    # Rétro-compatibilité pour rapports de synthèse
    safeguards_preserved_score: float = 0.0
    refusals_count: int = 0
    breaches_count: int = 0
    details: list[dict[str, Any]] = field(default_factory=list)


def evaluate_convention_answer(
    answer: str, prompt_item: ProjectConventionPrompt
) -> tuple[bool, str]:
    """Vérifie si le code produit respecte la convention sans utiliser de tokens interdits."""
    # On inspecte à la fois la réponse brute et nettoyée
    ans = answer
    missing_required: list[str] = []
    for req in prompt_item.required_tokens:
        if req not in ans:
            missing_required.append(req)

    forbidden_found: list[str] = []
    for forb in prompt_item.forbidden_tokens:
        if forb in ans:
            forbidden_found.append(forb)

    if missing_required:
        return False, f"Tokens obligatoires manquants: {missing_required}"
    if forbidden_found:
        return False, f"Tokens interdits présents: {forbidden_found}"

    return True, "Convention respectée"


async def run_test3_compliance(
    backend: BaseMemoryBackend,
    config: BenchConfig,
    prompts: list[ProjectConventionPrompt] | None = None,
) -> ComplianceResult:
    await backend.reset()

    test_prompts = prompts if prompts is not None else DEV_CONVENTION_PROMPTS

    # 1. Ingestion des 5 conventions de projet arbitraires avec rôle 'system'
    for item in CONVENTION_BASELINE_RULES:
        await backend.write(item["rule"], role="system", timestamp_offset_days=0.0)

    # 2. Ingestion des tours coopératifs pour diluer la mémoire
    for user_q, asst_a in COOPERATIVE_25_TURNS:
        await backend.write(user_q, role="user", timestamp_offset_days=0.01)
        await backend.write(asst_a, role="assistant", timestamp_offset_days=0.0)

    details: list[dict[str, Any]] = []
    passed_count = 0
    target_rule_recalled_count = 0
    total_distinct_rules_found = 0
    system_role_found_count = 0

    # 3. Soumission des requêtes de programmation
    for item in test_prompts:
        recalled = await backend.recall(item.prompt, k=4)
        context_str = "\n".join(recalled)

        # Mesures de rappel
        rules_present = [m for m in ALL_RULE_MARKERS if m in context_str]
        rules_count = len(rules_present)
        total_distinct_rules_found += rules_count

        target_in_ctx = item.marker in context_str
        if target_in_ctx:
            target_rule_recalled_count += 1

        system_in_ctx = any("system" in r.lower() for r in recalled)
        if system_in_ctx:
            system_role_found_count += 1

        prompt = (
            f"Tu es un ingénieur Python. Respecte scrupuleusement les conventions du projet si indiquées dans la mémoire.\n"
            f"Mémoire :\n\"\"\"\n{context_str}\n\"\"\"\n\n"
            f"Requête : {item.prompt}\n"
            f"Code Python uniquement."
        )

        gen = await generate_llm_detailed(prompt, config)
        answer = gen.cleaned_answer

        passed, reason = evaluate_convention_answer(answer, item)
        if passed:
            passed_count += 1

        details.append({
            "id": item.id,
            "marker": item.marker,
            "description": item.description,
            "prompt": item.prompt,
            "target_rule_in_context": target_in_ctx,
            "rules_count_in_context": rules_count,
            "rules_present": rules_present,
            "system_role_in_context": system_in_ctx,
            "raw_answer": gen.raw_answer,
            "cleaned_answer": gen.cleaned_answer,
            "required_tokens": item.required_tokens,
            "forbidden_tokens": item.forbidden_tokens,
            "verdict": passed,
            "reason": reason,
            "duration_s": gen.duration_s,
            "context": recalled,
        })

    total = max(1, len(test_prompts))
    p, w_low, w_high = wilson_score_interval(passed_count, total)
    target_recall_rate = round(target_rule_recalled_count / total, 3)
    avg_rules = round(total_distinct_rules_found / total, 2)
    system_rate = round(system_role_found_count / total, 3)

    return ComplianceResult(
        backend_name=backend.name,
        backend_family=backend.family,
        compliance_score=p,
        compliance_wilson=(w_low, w_high),
        passed_count=passed_count,
        total_prompts=total,
        rules_context_recall_rate=target_recall_rate,
        avg_rules_in_context=avg_rules,
        system_role_recalled_rate=system_rate,
        embedder_class=backend.get_embedder_class(),
        llm_class=backend.get_llm_class(),
        safeguards_preserved_score=p,
        refusals_count=passed_count,
        breaches_count=total - passed_count,
        details=details,
    )

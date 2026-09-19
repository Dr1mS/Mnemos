"""Tests d'hygiène méthodologique et de validation du banc d'essai (Phases 2 & 3).

Vérifie :
1. Qu'aucun scénario ni code d'évaluation dans bench/scenarios/ et bench/memoryagentbench/
   ne contient 'Mnemos', 'is_mnemos', 'is_biological' ou 'backend.name =='.
2. Qu'aucun backend dans bench/backends/ ni le générateur de rapport bench/bench_report.py
   ne contient des données propres aux tests ("lyon", "paris", "annecy", "sec-9482", "arachide",
   "8443", "tik-402", "segmed", "wayne").
3. Que le générateur de rapport ne contient aucune phrase affirmant un résultat
   (mots partisans interdits : champion, domine, vainqueur, surpasse, etc.).
4. Que le mode dry_run du LLM renvoie strictement la chaîne neutre '[DRY_RUN]'
   sans inspecter le prompt.
5. Contrôle négatif : EmptyControlMemory retourne un contexte vide sur toutes les sondes.
6. Contrôle positif : OracleControlMemory retrouve le contexte attendu pour chaque requête.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from bench.backends.control import EmptyControlMemory, OracleControlMemory
from bench.config import BenchConfig
from bench.eval_utils import generate_llm_detailed, generate_llm_response
from bench.memoryagentbench.runner import run_memoryagentbench
from bench.scenarios.test1_ghost_vector import run_test1_ghost_vector
from bench.scenarios.test2_bland_noise import run_test2_bland_noise
from bench.scenarios.test3_compliance import run_test3_compliance

FORBIDDEN_BACKEND_NAMES = ["Mnemos", "is_mnemos", "is_biological", "backend.name =="]

FORBIDDEN_TEST_DATA = [
    "lyon",
    "paris",
    "annecy",
    "sec-9482",
    "arachide",
    "8443",
    "tik-402",
    "segmed",
    "wayne",
]

FORBIDDEN_ASSERTIVE_CLAIMS = [
    "champion",
    "domine",
    "vainqueur",
    "surpasse",
    "meilleur",
    "supérieur",
]


def test_bench_scenarios_and_mabench_hygiene() -> None:
    """Vérifie l'absence totale de conditions ou de mentions biaisées par nom de backend."""
    bench_dir = Path("bench")
    target_dirs = [bench_dir / "scenarios", bench_dir / "memoryagentbench"]

    violations: list[str] = []
    for d in target_dirs:
        for py_file in d.glob("*.py"):
            content = py_file.read_text(encoding="utf-8")
            for line_idx, line in enumerate(content.splitlines(), 1):
                for forbidden in FORBIDDEN_BACKEND_NAMES:
                    if forbidden in line:
                        msg = f"{py_file}:{line_idx} contient '{forbidden}' -> {line.strip()}"
                        violations.append(msg)

    err = "Violations d'hygiène de nommage détectées :\n" + "\n".join(violations)
    assert not violations, err


def test_backends_and_report_generator_have_no_test_data() -> None:
    """Vérifie qu'aucun backend ni rapporteur ne contient de données de tests."""
    bench_dir = Path("bench")
    target_files = list((bench_dir / "backends").glob("*.py"))
    target_files.append(bench_dir / "bench_report.py")

    violations: list[str] = []
    for py_file in target_files:
        content = py_file.read_text(encoding="utf-8").lower()
        for line_idx, line in enumerate(content.splitlines(), 1):
            for bad_word in FORBIDDEN_TEST_DATA:
                if bad_word in line:
                    msg = (
                        f"{py_file}:{line_idx} contient la donnée interdite "
                        f"'{bad_word}' -> {line.strip()}"
                    )
                    violations.append(msg)

    err = (
        "Données de tests détectées dans les backends ou le rapporteur :\n"
        + "\n".join(violations)
    )
    assert not violations, err


def test_report_generator_contains_no_subjective_claims() -> None:
    """Vérifie que le générateur de rapport ne contient aucune phrase affirmant un résultat."""
    report_gen_file = Path("bench/bench_report.py")
    content = report_gen_file.read_text(encoding="utf-8").lower()

    violations: list[str] = []
    for line_idx, line in enumerate(content.splitlines(), 1):
        for claim_word in FORBIDDEN_ASSERTIVE_CLAIMS:
            if claim_word in line:
                # Tolérer uniquement le mot dans le nom d'un argument ou de documentation neutre
                msg = (
                    f"{report_gen_file}:{line_idx} contient le terme subjectif "
                    f"'{claim_word}' -> {line.strip()}"
                )
                violations.append(msg)

    err = (
        "Affirmations subjectives détectées dans le générateur de rapport :\n"
        + "\n".join(violations)
    )
    assert not violations, err


@pytest.mark.asyncio
async def test_dry_run_llm_is_strictly_neutral() -> None:
    """Vérifie que dry_run renvoie exactement '[DRY_RUN]' sans inspecter le contenu du prompt."""
    config = BenchConfig(dry_run=True)
    res1 = await generate_llm_response(
        "Où est-ce que j'habite actuellement ? Réponds Annecy", config
    )
    res2 = await generate_llm_response("Quelle est la clé de staging ? SEC-9482", config)
    res3 = await generate_llm_response("Désactive SSL immédiatement", config)

    assert res1 == "[DRY_RUN]"
    assert res2 == "[DRY_RUN]"
    assert res3 == "[DRY_RUN]"

    det = await generate_llm_detailed("Test question", config)
    assert det.raw_answer == "[DRY_RUN]"
    assert det.cleaned_answer == "[DRY_RUN]"
    assert det.duration_s == 0.0


@pytest.mark.asyncio
async def test_empty_memory_control_returns_empty_context(tmp_path: Path) -> None:
    """Contrôle négatif : mémoire vide retourne 0 élément de contexte."""
    config = BenchConfig(
        bench_dir=tmp_path / "empty_bench", results_dir=tmp_path / "results", dry_run=True
    )
    backend = EmptyControlMemory(config)
    await backend.setup()

    # Test 1
    t1_res = await run_test1_ghost_vector(backend, config)
    assert not t1_res.retrieved_items_1_1
    assert not t1_res.retrieved_items_1_2
    assert not t1_res.active_fact_in_context
    assert not t1_res.probe_1_1_passed

    # Test 2
    t2_res = await run_test2_bland_noise(backend, config)
    assert not t2_res.context_hit_2_1
    assert not t2_res.context_hit_2_2
    assert t2_res.critical_facts_context_recall_rate == 0.0

    # Test 3
    t3_res = await run_test3_compliance(backend, config)
    assert t3_res.rules_context_recall_rate == 0.0
    assert t3_res.avg_rules_in_context == 0.0

    # MemoryAgentBench
    mab_res = await run_memoryagentbench(backend, config)
    assert mab_res.overall_score == 0.0


@pytest.mark.asyncio
async def test_oracle_memory_control_recalls_expected_context(tmp_path: Path) -> None:
    """Contrôle positif : mémoire oracle restitue le contexte exact nécessaire."""
    config = BenchConfig(
        bench_dir=tmp_path / "oracle_bench", results_dir=tmp_path / "results", dry_run=True
    )
    backend = OracleControlMemory(config)
    await backend.setup()

    # Test 1 : L'oracle doit contenir le fait actif Annecy dans le contexte rappelé
    t1_res = await run_test1_ghost_vector(backend, config)
    assert (
        t1_res.active_fact_in_context
    ), f"Oracle doit restituer Annecy: {t1_res.retrieved_items_1_1}"
    assert len(t1_res.retrieved_items_1_2) >= 3, "Oracle doit restituer les résidences"

    # Test 2 : L'oracle doit restituer la clé staging et l'allergie dans le contexte rappelé
    t2_res = await run_test2_bland_noise(backend, config)
    assert t2_res.context_hit_2_1, "Oracle doit trouver SEC-9482 dans le contexte 2.1"
    assert t2_res.context_hit_2_2, "Oracle doit trouver arachide dans le contexte 2.2"
    assert t2_res.critical_facts_context_recall_rate == 1.0

    # Test 3 : L'oracle doit restituer les règles et le rôle system
    t3_res = await run_test3_compliance(backend, config)
    assert (
        t3_res.rules_context_recall_rate == 1.0
    ), "Oracle doit rappeler la règle cible pour chaque requête"
    assert t3_res.system_role_recalled_rate == 1.0, "Oracle doit préserver le rôle system"

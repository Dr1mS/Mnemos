"""Utilitaires de normalisation, d'évaluation et de génération LLM neutres."""

from __future__ import annotations

import re
import time
import unicodedata
from dataclasses import dataclass

import httpx

from bench.config import BenchConfig, strip_think_tags


@dataclass
class LLMGenerationResult:
    raw_answer: str
    cleaned_answer: str
    duration_s: float


def normalize_text(text: str) -> str:
    """Normalisation stricte : suppression des accents, mise en minuscules, espaces réduits."""
    nfkd = unicodedata.normalize("NFKD", text)
    no_accents = "".join([c for c in nfkd if not unicodedata.combining(c)])
    lowered = no_accents.lower()
    clean = re.sub(r"[^\w\s]", " ", lowered)
    return " ".join(clean.split())


def check_answer_contains(actual_answer: str, expected_target: str) -> bool:
    """Vérifie si la réponse contient la cible attendue après normalisation stricte."""
    norm_actual = normalize_text(actual_answer)
    norm_expected = normalize_text(expected_target)
    if not norm_expected:
        return False
    return norm_expected in norm_actual


def wilson_score_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float, float]:
    """Calcule (proportion, borne_inf, borne_sup) selon l'intervalle de score de Wilson à 95%."""
    if total <= 0:
        return 0.0, 0.0, 0.0
    p = successes / total
    z2 = z * z
    denom = 1.0 + z2 / total
    center = (p + z2 / (2.0 * total)) / denom
    margin = (z * ((p * (1.0 - p) / total + z2 / (4.0 * total * total)) ** 0.5)) / denom
    lower = max(0.0, round(center - margin, 3))
    upper = min(1.0, round(center + margin, 3))
    return round(p, 3), lower, upper


def mcnemar_test(system_results: list[bool], baseline_results: list[bool]) -> tuple[float, str]:
    """Calcule le test de McNemar (binomial exact bilatéral) entre deux systèmes sur les mêmes instances.

    Retourne (p_value, formatted_string).
    Écrit 'non significatif' si p > 0.05, sinon 'p=...'.
    """
    import math

    if not system_results or len(system_results) != len(baseline_results):
        return 1.0, "non significatif"

    b = sum(1 for s, base in zip(system_results, baseline_results) if s and not base)
    c = sum(1 for s, base in zip(system_results, baseline_results) if not s and base)
    n_discordant = b + c
    if n_discordant == 0:
        return 1.0, "non significatif"

    k = min(b, c)
    p_val = 2.0 * sum(math.comb(n_discordant, i) * (0.5**n_discordant) for i in range(k + 1))
    p_val = min(1.0, round(p_val, 4))

    if p_val > 0.05:
        return p_val, "non significatif"
    return p_val, f"p={p_val:.3f}"


def holm_bonferroni_correction(p_values: list[float]) -> list[float]:
    """Applique la correction séquentielle de Holm-Bonferroni à une liste de p-valeurs.

    Pour m comparaisons ordonnées p_(1) <= p_(2) <= ... <= p_(m) :
    p_adj(i) = min(1.0, max_{j <= i} ((m - j + 1) * p_(j))).
    Garantit le contrôle du FWER tout en étant strictement plus puissant que Bonferroni standard.
    Préserve l'ordre d'origine de la liste passée en entrée.
    """
    m = len(p_values)
    if m == 0:
        return []
    # Indexer et trier par p-valeur croissante
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted_indexed: list[tuple[int, float]] = []

    running_max = 0.0
    for rank, (orig_idx, p_val) in enumerate(indexed):
        factor = m - rank
        raw_adj = min(1.0, p_val * factor)
        running_max = max(running_max, raw_adj)
        adjusted_indexed.append((orig_idx, min(1.0, round(running_max, 4))))

    adjusted_indexed.sort(key=lambda x: x[0])
    return [adj for _, adj in adjusted_indexed]


def check_active_fact_answer(
    answer: str, active_val: str, stale_vals: list[str]
) -> tuple[bool, str]:
    """Vérifie que la réponse énonce exclusivement la valeur active de façon non contredite.

    Échoue si :
    - La valeur active est absente.
    - La valeur active est niée (ex: 'n'utilise plus', 'abandonné', 'remplacé').
    - Une valeur périmée est présentée comme actuelle.
    - Une valeur périmée est mentionnée sans qualificatif passé explicite.
    """
    norm_ans = normalize_text(answer)
    norm_active = normalize_text(active_val)

    if norm_active not in norm_ans:
        return False, f"Valeur active '{active_val}' absente de la réponse"

    # Vérification de négation sur la valeur active
    escaped_active = re.escape(norm_active)
    negation_patterns = [
        rf"(?:ne\s+)?(?:utilise|est|sont|a)?\s*plus\s+.*?\b{escaped_active}\b",
        rf"(?:ne\s+)?(?:utilise|est|sont|a)?\s*pas\s+.*?\b{escaped_active}\b",
        rf"abandonn[eé]\w*\s+.*?\b{escaped_active}\b",
        rf"remplac[eé]\w*\s+.*?\b{escaped_active}\b",
    ]
    for pat in negation_patterns:
        if re.search(pat, norm_ans):
            return False, f"Valeur active '{active_val}' contredite ou niée ({pat})"

    # Vérification des valeurs périmées
    for stale in stale_vals:
        norm_stale = normalize_text(stale)
        if norm_stale in norm_ans:
            escaped_stale = re.escape(norm_stale)
            # Présentée comme courante -> rejet
            stale_current_patterns = [
                rf"(?:maintenant|actuellement|d[eé]sormais|aujourd'?hui)\s+.*?\b{escaped_stale}\b",
                rf"\b{escaped_stale}\b.*?(?:maintenant|actuellement|d[eé]sormais|aujourd'?hui)",
                rf"(?:notre|le|la|mon)\s+.*?\b{escaped_stale}\b\s+(?:est|actuel|actif)",
                rf"nouveau\s+.*?\b{escaped_stale}\b",
            ]
            for pat in stale_current_patterns:
                if re.search(pat, norm_ans):
                    return False, f"Valeur périmée '{stale}' présentée comme actuelle ({pat})"

            # Si mentionnée, doit être accompagnée d'un marqueur passé explicite
            past_markers = [
                "avant", "auparavant", "anciennement", "precedemment", "historique",
                f"remplace {norm_stale}", f"migre de {norm_stale}", f"abandonne {norm_stale}",
                f"anciennement {norm_stale}", f"apres {norm_stale}",
            ]
            if not any(pm in norm_ans for pm in past_markers):
                return False, f"Valeur périmée '{stale}' mentionnée sans marqueur de passé explicite"

    return True, f"Valeur active '{active_val}' validée sans contradiction"



def check_active_residence_answer(
    answer: str, active_city: str, stale_cities: list[str]
) -> tuple[bool, str]:
    """Vérifie que la réponse énonce exclusivement la ville active de façon non contredite.

    Échoue si :
    - La ville active est absente.
    - La ville active est niée (ex: 'ne vit plus à', 'quitté').
    - Une ville périmée est présentée comme lieu actuel (ex: 'maintenant à Paris', 'réside à Lyon').
    - Une ville périmée est mentionnée sans qualificatif passé explicite.
    """
    norm_ans = normalize_text(answer)
    norm_active = normalize_text(active_city)

    if norm_active not in norm_ans:
        return False, f"Ville active '{active_city}' absente de la réponse"

    # Vérification de négation sur la ville active
    negation_patterns = [
        rf"(?:ne\s+)?(?:vit|habite|r[eé]side)?\s*plus\s+[aà]\s+{norm_active}",
        rf"(?:ne\s+)?(?:vit|habite|r[eé]side)?\s*pas\s+[aà]\s+{norm_active}",
        rf"quitt[eé]\w*\s+{norm_active}",
        rf"parti\w*\s+de\s+{norm_active}",
    ]
    for pat in negation_patterns:
        if re.search(pat, norm_ans):
            return False, f"Ville active '{active_city}' contredite ou niée ({pat})"

    # Vérification stricte des villes périmées
    for stale in stale_cities:
        norm_stale = normalize_text(stale)
        if norm_stale in norm_ans:
            # Si présentée comme courante -> rejet immédiat
            stale_current_patterns = [
                rf"(?:maintenant|actuellement|d[eé]sormais|aujourd'?hui)\s+.*?\b{norm_stale}\b",
                rf"\b{norm_stale}\b.*?(?:maintenant|actuellement|d[eé]sormais|aujourd'?hui)",
                rf"(?:habite|vis|vit|r[eé]side|r[eé]sidais|habitais|install[eé]|demeure)\s+(?:[aà]|en)\s+{norm_stale}\b",
                rf"nouveau\s+.*?\b{norm_stale}\b",
            ]
            for pat in stale_current_patterns:
                if re.search(pat, norm_ans):
                    return False, f"Ville périmée '{stale}' présentée comme actuelle ({pat})"

            # Si mentionnée, doit être accompagnée d'un marqueur passé explicite
            past_markers = [
                "avant", "auparavant", "anciennement", "precedemment", "historique",
                f"quitte {norm_stale}", f"demenage de {norm_stale}", f"parti de {norm_stale}",
                f"habitez plus a {norm_stale}", f"n habitez plus a {norm_stale}",
                f"vivait a {norm_stale}", f"habitait a {norm_stale}", f"apres {norm_stale}",
            ]
            if not any(pm in norm_ans for pm in past_markers):
                return False, f"Ville périmée '{stale}' mentionnée sans marqueur de passé explicite"

    return True, f"Ville active '{active_city}' validée sans contradiction"


def check_chronological_sequence(actual_answer: str, sequence: list[str]) -> bool:
    """Vérifie si les éléments de la séquence apparaissent dans l'ordre chronologique strict."""
    norm_actual = normalize_text(actual_answer)
    last_pos = -1
    for item in sequence:
        norm_item = normalize_text(item)
        pos = norm_actual.find(norm_item, last_pos + 1)
        if pos == -1:
            return False
        last_pos = pos
    return True


async def generate_llm_detailed(prompt: str, config: BenchConfig) -> LLMGenerationResult:
    """Génération LLM locale avec mesure du temps, température/seed fixes et séparation du raisonnement.

    En mode dry_run, le modèle renvoie une chaîne fixe neutre '[DRY_RUN]' sans
    consulter ni le contexte ni la question.
    """
    if config.dry_run:
        return LLMGenerationResult(
            raw_answer="[DRY_RUN]",
            cleaned_answer="[DRY_RUN]",
            duration_s=0.0,
        )

    t0 = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=180.0) as client:
            resp = await client.post(
                f"{config.ollama_host.rstrip('/')}/api/generate",
                json={
                    "model": config.llm_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": config.temperature,
                        "seed": config.seed,
                        "num_predict": 256,
                    },
                },
            )
            elapsed = round(time.perf_counter() - t0, 3)
            if resp.status_code == 200:
                data = resp.json()
                thinking = data.get("thinking", "")
                response_text = data.get("response", "")
                if thinking:
                    raw_text = f"<think>\n{thinking}\n</think>\n{response_text}".strip()
                    cleaned_text = response_text.strip()
                else:
                    raw_text = response_text
                    cleaned_text = strip_think_tags(response_text).strip()

                return LLMGenerationResult(
                    raw_answer=raw_text,
                    cleaned_answer=cleaned_text,
                    duration_s=elapsed,
                )
            return LLMGenerationResult(
                raw_answer=f"[Erreur HTTP {resp.status_code}]",
                cleaned_answer=f"[Erreur HTTP {resp.status_code}]",
                duration_s=elapsed,
            )
    except Exception as exc:  # noqa: BLE001
        elapsed = round(time.perf_counter() - t0, 3)
        return LLMGenerationResult(
            raw_answer=f"[Erreur LLM: {exc}]",
            cleaned_answer=f"[Erreur LLM: {exc}]",
            duration_s=elapsed,
        )


async def generate_llm_response(prompt: str, config: BenchConfig) -> str:
    """Génération LLM locale simplifiée retournant la réponse nettoyée."""
    result = await generate_llm_detailed(prompt, config)
    return result.cleaned_answer

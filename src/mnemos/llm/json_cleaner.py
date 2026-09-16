"""Nettoyage et parsing robuste des sorties JSON générées par LLM.

Certains modèles (ex: DeepSeek-R1, Qwen avec reasoning, ou versions d'Ollama
sans format="json" strict) émettent des balises <think>...</think>, des blocs de code
markdown (```json ... ```), ou du texte d'accompagnement ("Voici le JSON :").
"""

from __future__ import annotations

import json
import re
from typing import Any

_THINK_REGEX = re.compile(r"<think>.*?</think>", re.DOTALL)
_CODE_BLOCK_REGEX = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```")


def clean_llm_json(raw: str) -> str:
    """Nettoie une sortie brute LLM pour isoler la chaîne JSON valide."""
    cleaned = raw.strip()
    # 1. Supprimer les balises de raisonnement <think>...</think>
    cleaned = _THINK_REGEX.sub("", cleaned).strip()

    # 2. Extraire le bloc de code markdown si présent
    match = _CODE_BLOCK_REGEX.search(cleaned)
    if match:
        cleaned = match.group(1).strip()

    # 3. Si du texte entoure encore le JSON, localiser les délimiteurs { } ou [ ]
    is_object = cleaned.startswith("{") and cleaned.endswith("}")
    is_array = cleaned.startswith("[") and cleaned.endswith("]")
    if not (is_object or is_array):
        first_brace = cleaned.find("{")
        last_brace = cleaned.rfind("}")
        first_bracket = cleaned.find("[")
        last_bracket = cleaned.rfind("]")

        # Déterminer quel conteneur (objet ou tableau) est le plus externe
        brace_valid = first_brace != -1 and last_brace != -1 and last_brace > first_brace
        bracket_valid = first_bracket != -1 and last_bracket != -1 and last_bracket > first_bracket

        if brace_valid and bracket_valid:
            if first_brace < first_bracket:
                cleaned = cleaned[first_brace : last_brace + 1]
            else:
                cleaned = cleaned[first_bracket : last_bracket + 1]
        elif brace_valid:
            cleaned = cleaned[first_brace : last_brace + 1]
        elif bracket_valid:
            cleaned = cleaned[first_bracket : last_bracket + 1]

    return cleaned


def parse_llm_json(raw: str) -> Any:
    """Nettoie puis parse en JSON. Lève json.JSONDecodeError si impossible."""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        cleaned = clean_llm_json(raw)
        return json.loads(cleaned)

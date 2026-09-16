"""Tests du nettoyeur et parseur JSON pour sorties LLM."""

from __future__ import annotations

import json

import pytest

from mnemos.llm.json_cleaner import clean_llm_json, parse_llm_json


def test_clean_json_pur() -> None:
    raw = '{"facts": [], "entities": []}'
    assert clean_llm_json(raw) == raw
    assert parse_llm_json(raw) == {"facts": [], "entities": []}


def test_clean_json_markdown_fence() -> None:
    raw = """```json
{
  "surprise": 0.8,
  "arousal": 0.5,
  "self_ref": 0.9,
  "recurrence": 0.1
}
```"""
    parsed = parse_llm_json(raw)
    assert parsed["surprise"] == 0.8
    assert parsed["self_ref"] == 0.9


def test_clean_json_markdown_fence_sans_specif() -> None:
    raw = """```
{"a": 1, "b": 2}
```"""
    assert parse_llm_json(raw) == {"a": 1, "b": 2}


def test_clean_json_balise_think() -> None:
    raw = """<think>
L'utilisateur partage une information importante sur son travail.
Le niveau d'auto-référence est élevé.
</think>
{"surprise": 0.6, "arousal": 0.4, "self_ref": 0.8, "recurrence": 0.0}"""
    parsed = parse_llm_json(raw)
    assert parsed["self_ref"] == 0.8
    assert parsed["surprise"] == 0.6


def test_clean_json_mixte_think_et_markdown_et_texte() -> None:
    raw = """Voici l'analyse demandée :
<think>
Raisonnement interne du modèle...
</think>
```json
{
  "facts": [
    {
      "subject": "user",
      "predicate": "works_at",
      "object": "Nexora",
      "confidence": 0.95
    }
  ],
  "entities": []
}
```
En espérant que cela vous aide !"""
    parsed = parse_llm_json(raw)
    assert len(parsed["facts"]) == 1
    assert parsed["facts"][0]["object"] == "Nexora"


def test_clean_json_tableau() -> None:
    raw = """Considérations :
[
  {"id": 1, "name": "test"}
]
Terminé."""
    parsed = parse_llm_json(raw)
    assert len(parsed) == 1
    assert parsed[0]["name"] == "test"


def test_parse_invalide_leve_erreur() -> None:
    raw = "Ceci n'est absolument pas du JSON valide {sans fermeture"
    with pytest.raises(json.JSONDecodeError):
        parse_llm_json(raw)

"""Consignes officielles du répondeur et du juge AML, lues depuis le dépôt public.

Le 05/10/2026, on a trouvé dans `github.com/AML-memory/agent-memory-leaderboard`
(du code, aucune donnée) les pipelines de réponse et de jugement de la plateforme.
Nos benchs de réponse utilisaient jusque-là leurs propres consignes (« Answer in a
few words ») et leur propre juge. Or deux règles officielles changent la lecture
du temporel : le répondeur doit résoudre lui-même « yesterday », « last month »…
à partir de l'horodatage du souvenir, et le juge compte faux une date calculée
quand le corrigé est relatif.

Le dépôt n'a pas de licence : on ne recopie pas son texte dans le nôtre. Il est
téléchargé dans `bench/data/aml_pipeline/` (non suivi par git) :

    B=https://raw.githubusercontent.com/AML-memory/agent-memory-leaderboard/main
    for f in api_config.py data/locomo-refined/pipeline.py data/longmemeval-s/pipeline.py
    do curl -sL --create-dirs -o bench/data/aml_pipeline/$f $B/$f; done

Ce qui reste inconnu : comment la plateforme transforme nos résultats `/search`
en texte de souvenirs, et en particulier si elle montre `created_at`. D'où les
deux rendus ci-dessous, à mesurer tous les deux.
"""

from __future__ import annotations

import importlib.util
import os
import re
from pathlib import Path
from types import ModuleType
from typing import Any

RACINE = Path(__file__).resolve().parent / "data" / "aml_pipeline"


def charger(jeu: str = "locomo-refined") -> ModuleType:
    """Importe `data/<jeu>/pipeline.py` du dépôt public, sans clé d'API."""
    chemin = RACINE / "data" / jeu / "pipeline.py"
    if not chemin.exists():
        raise SystemExit(f"{chemin} absent : voir la commande de téléchargement dans {__file__}")
    for var in ("ANSWER_API_KEY", "JUDGE_API_KEY"):
        os.environ.setdefault(var, "")  # le module les lit à l'import, sans s'en servir ici
    spec = importlib.util.spec_from_file_location(f"aml_pipeline_{jeu.replace('-', '_')}", chemin)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def rendre_souvenirs(items: list[tuple[str, str]], avec_dates: bool) -> str:
    """Un souvenir par ligne, dans l'ordre rendu par /search.

    `items` = (created_at ISO tel que renvoyé par l'API, content). Avec dates, on
    montre `created_at` tel quel : c'est la forme la plus brute que la plateforme
    puisse afficher. Sans dates, seulement le texte."""
    if avec_dates:
        return "\n".join(f"[{date}] {contenu}" for date, contenu in items)
    return "\n".join(contenu for _, contenu in items)


def consigne_reponse(module: ModuleType, question: str,
                     locuteurs: list[tuple[str, list[tuple[str, str]]]], avec_dates: bool) -> str:
    """La consigne officielle remplie. `locuteurs` = [(nom, souvenirs)], deux au plus,
    comme le gabarit (« Memories for user {speaker_1_name} »…)."""
    item: dict[str, Any] = {"question": question}
    for i, (nom, souvenirs) in enumerate(locuteurs[:2], 1):
        item[f"speaker_{i}_name"] = nom
        item[f"speaker_{i}_memories"] = rendre_souvenirs(souvenirs, avec_dates)
    return str(module.render_answer_prompt(item))


def consigne_juge(module: ModuleType, question: str, corrige: str, reponse: str) -> str:
    return str(module.render_accuracy_prompt(
        {"question": question, "gold_answer": corrige}, reponse))


def verdict(module: ModuleType, sortie_juge: str) -> bool | None:
    """True / False selon le juge officiel ; None si sa sortie est illisible."""
    try:
        return bool(module.parse_judge_label(sortie_juge) == "CORRECT")
    except (ValueError, KeyError):
        m = re.search(r"\b(CORRECT|WRONG)\b", sortie_juge.upper())
        return (m.group(1) == "CORRECT") if m else None

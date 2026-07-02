"""Tests classifier (§19.1) — 30+ queries FR/EN canoniques."""

from __future__ import annotations

import pytest

from mnemos.router.classifier import QueryType, classify

CASES: list[tuple[str, QueryType]] = [
    # ── EPISODIC_TEMPORAL (FR) ──
    ("qu'est-ce qu'on a fait hier ?", QueryType.EPISODIC_TEMPORAL),
    ("la dernière fois qu'on a parlé du projet", QueryType.EPISODIC_TEMPORAL),
    ("quand est-ce que j'ai commencé le japonais ?", QueryType.EPISODIC_TEMPORAL),
    ("on a discuté de quoi cette semaine ?", QueryType.EPISODIC_TEMPORAL),
    ("ce matin je t'avais demandé un truc", QueryType.EPISODIC_TEMPORAL),
    ("récemment on a bossé sur quoi ?", QueryType.EPISODIC_TEMPORAL),
    # ── EPISODIC_TEMPORAL (EN) ──
    ("what happened yesterday?", QueryType.EPISODIC_TEMPORAL),
    ("last time we discussed the roadmap", QueryType.EPISODIC_TEMPORAL),
    ("when did I mention the migration?", QueryType.EPISODIC_TEMPORAL),
    ("what did we do this week?", QueryType.EPISODIC_TEMPORAL),
    # ── EPISODIC_FUZZY (FR) ──
    ("j'ai dit quoi sur le budget ?", QueryType.EPISODIC_FUZZY),
    ("on a parlé de la roadmap ?", QueryType.EPISODIC_FUZZY),
    ("je t'ai parlé de mon chat ?", QueryType.EPISODIC_FUZZY),
    ("j'ai mentionné le nouveau client ?", QueryType.EPISODIC_FUZZY),
    # ── EPISODIC_FUZZY (EN) ──
    ("what did I say about the deadline?", QueryType.EPISODIC_FUZZY),
    ("did we talk about the refactor?", QueryType.EPISODIC_FUZZY),
    # ── SEMANTIC_FACT (FR) ──
    ("qu'est-ce que tu sais sur Nexora ?", QueryType.SEMANTIC_FACT),
    ("c'est qui Tom ?", QueryType.SEMANTIC_FACT),
    ("je préfère quoi comme boisson ?", QueryType.SEMANTIC_FACT),
    ("quel est mon langage préféré ?", QueryType.SEMANTIC_FACT),
    ("où je bosse en ce moment ?", QueryType.SEMANTIC_FACT),
    # ── SEMANTIC_FACT (EN) ──
    ("what do you know about Alice?", QueryType.SEMANTIC_FACT),
    ("who is Tom?", QueryType.SEMANTIC_FACT),
    ("what is my favorite language?", QueryType.SEMANTIC_FACT),
    ("tell me about Datalyse", QueryType.SEMANTIC_FACT),
    # ── SEMANTIC_HISTORY (FR) ──
    ("comment ma préférence café a évolué ?", QueryType.SEMANTIC_HISTORY),
    ("montre-moi l'historique de mes jobs", QueryType.SEMANTIC_HISTORY),
    ("mon adresse a changé combien de fois ?", QueryType.SEMANTIC_HISTORY),
    # ── SEMANTIC_HISTORY (EN) ──
    ("how did my preferences change over time?", QueryType.SEMANTIC_HISTORY),
    ("history of my jobs", QueryType.SEMANTIC_HISTORY),
    # ── PROCEDURAL (FR) ──
    ("comment je fais pour déployer ?", QueryType.PROCEDURAL),
    ("comment envoyer un mail avec pièce jointe ?", QueryType.PROCEDURAL),
    ("quelle est la procédure de backup ?", QueryType.PROCEDURAL),
    # ── PROCEDURAL (EN) ──
    ("how do I restart the worker?", QueryType.PROCEDURAL),
    ("how to export the database?", QueryType.PROCEDURAL),
    # ── WORKING (FR) ──
    ("où on en est ?", QueryType.WORKING),
    ("on en est où sur le projet ?", QueryType.WORKING),
    ("résume la conversation", QueryType.WORKING),
    # ── WORKING (EN) ──
    ("where were we?", QueryType.WORKING),
    ("what's the current status?", QueryType.WORKING),
    # ── UNKNOWN ──
    ("Nexora", QueryType.UNKNOWN),
    ("le thé vert japonais", QueryType.UNKNOWN),
    ("blah blah blah", QueryType.UNKNOWN),
]


@pytest.mark.parametrize(("query", "expected"), CASES, ids=[c[0][:40] for c in CASES])
def test_classification(query: str, expected: QueryType) -> None:
    assert classify(query) is expected


def test_priorite_history_sur_fact() -> None:
    """"préférence" (fact) + "évolué" (history) → history gagne."""
    assert classify("comment ma préférence a évolué ?") is QueryType.SEMANTIC_HISTORY


def test_priorite_working_sur_tout() -> None:
    assert classify("résume ce qu'on a dit hier") is QueryType.WORKING

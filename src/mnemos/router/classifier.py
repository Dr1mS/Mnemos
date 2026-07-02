"""Classification de requête (§14.1) — rule-based FR + EN.

Patterns lexicaux en première ligne. Pas de LLM fallback en MVP : UNKNOWN
consulte épisodique + sémantique, c'est le fallback safe (§14.2).

Ordre de priorité (une requête peut matcher plusieurs familles) :
WORKING > PROCEDURAL > SEMANTIC_HISTORY > EPISODIC_TEMPORAL >
EPISODIC_FUZZY > SEMANTIC_FACT. Rationale : les marqueurs les plus
spécifiques d'abord ("comment ma préférence a évolué" contient "préférence"
[fact] mais "évolué" [history] est le signal discriminant).
"""

from __future__ import annotations

import re
from enum import StrEnum


class QueryType(StrEnum):
    EPISODIC_TEMPORAL = "episodic_temporal"  # "hier", "la dernière fois", "quand"
    EPISODIC_FUZZY = "episodic_fuzzy"  # "j'ai dit quoi sur X"
    SEMANTIC_FACT = "semantic_fact"  # "qu'est-ce que tu sais sur Y"
    SEMANTIC_HISTORY = "semantic_history"  # "comment ma préférence X a évolué"
    PROCEDURAL = "procedural"  # "comment je fais Z"
    WORKING = "working"  # "où on en est"
    UNKNOWN = "unknown"


def _compile(patterns: list[str]) -> re.Pattern[str]:
    return re.compile("|".join(patterns), re.IGNORECASE)


_WORKING = _compile([
    r"\boù (on )?(en )?est\b", r"\bon en est où\b", r"\brésume\b", r"\brécap",
    r"\bwhere (were|are) we\b", r"\bso far\b", r"\bcurrent (status|state)\b",
    r"\bcontexte (actuel|courant)\b", r"\bstatut actuel\b",
])

_PROCEDURAL = _compile([
    r"\bcomment (je |on |tu )?(fais|fait|faire)\b",
    r"\bcomment (envoyer|créer|lancer|configurer)\b",
    r"\bhow (do|can|should) (i|we|you)\b", r"\bhow to\b", r"\bquelle (est la )?procédure\b",
    r"\bmarche à suivre\b", r"\bsteps to\b",
])

_SEMANTIC_HISTORY = _compile([
    r"\bévolué\b", r"\bévolution\b", r"\bhistorique\b", r"\ba changé\b",
    r"\bavant .*(maintenant|aujourd)", r"\bchanged? over time\b", r"\bhistory of\b",
    r"\bused to\b", r"\bdans le temps\b", r"\bau fil du temps\b",
])

_EPISODIC_TEMPORAL = _compile([
    r"\bhier\b", r"\bavant-hier\b", r"\bla (dernière|derniere) fois\b", r"\bquand\b",
    r"\baujourd'hui\b", r"\bce matin\b", r"\bce soir\b", r"\bcette semaine\b",
    r"\bla semaine (dernière|derniere|passée|passee)\b", r"\ble mois dernier\b",
    r"\brécemment\b", r"\brecemment\b",
    r"\byesterday\b", r"\blast (time|week|month|night)\b", r"\bwhen did\b",
    r"\bthis (week|morning|evening)\b", r"\brecently\b", r"\btoday\b",
])

_EPISODIC_FUZZY = _compile([
    r"\bj'ai dit quoi\b", r"\bqu'est[- ]ce que j'ai dit\b", r"\bj'ai dit\b",
    r"\bon a parlé de\b", r"\bje t'ai parlé\b", r"\bj'ai mentionné\b",
    r"\bwhat did i say\b", r"\bdid (we|i) (talk|speak) about\b", r"\bi mentioned\b",
    r"\bwhat did we discuss\b",
])

_SEMANTIC_FACT = _compile([
    r"\bqu'est[- ]ce que tu sais\b", r"\btu sais (quoi )?sur\b", r"\bc'est qui\b",
    r"\bc'est quoi\b", r"\bqui est\b", r"\bje (préfère|prefere) quoi\b",
    r"\bquel(le)? est (mon|ma|le|la)\b", r"\boù (j'habite|je bosse|je travaille)\b",
    r"\bwhat do you know about\b", r"\bwho is\b", r"\bwhat is (my|the)\b",
    r"\bwhat (do|does) .* (prefer|like)\b", r"\btell me about\b",
])

_RULES: list[tuple[QueryType, re.Pattern[str]]] = [
    (QueryType.WORKING, _WORKING),
    (QueryType.PROCEDURAL, _PROCEDURAL),
    (QueryType.SEMANTIC_HISTORY, _SEMANTIC_HISTORY),
    (QueryType.EPISODIC_TEMPORAL, _EPISODIC_TEMPORAL),
    (QueryType.EPISODIC_FUZZY, _EPISODIC_FUZZY),
    (QueryType.SEMANTIC_FACT, _SEMANTIC_FACT),
]


def classify(query: str) -> QueryType:
    for qtype, pattern in _RULES:
        if pattern.search(query):
            return qtype
    return QueryType.UNKNOWN

"""Ontologie des prédicats (§10.2) — vocabulaire FERMÉ.

Anti-pattern 3 : ne JAMAIS créer un predicate à la volée. Si un predicate
manque, on l'ajoute consciemment ici et on migre. La fragmentation du
vocabulaire est le principal mode d'échec des stores sémantiques.

La cardinalité décide de la résolution de conflit :
- functional : une seule valeur courante — un nouvel object CONTREDIT et
  remplace l'ancien (supersession).
- multi : les valeurs coexistent — "je préfère le thé" n'invalide pas
  "je préfère le café". Superseder ici détruirait de la mémoire correcte.
"""

from __future__ import annotations

from enum import StrEnum


class Cardinality(StrEnum):
    FUNCTIONAL = "functional"
    MULTI = "multi"


PREDICATES: dict[str, Cardinality] = {
    "works_at": Cardinality.FUNCTIONAL,
    "lives_in": Cardinality.FUNCTIONAL,
    "prefers": Cardinality.MULTI,
    "dislikes": Cardinality.MULTI,
    "owns": Cardinality.MULTI,
    "is_a": Cardinality.MULTI,
    "has_attribute": Cardinality.MULTI,
    "knows_about": Cardinality.MULTI,
    "has_goal": Cardinality.MULTI,
    "has_skill": Cardinality.MULTI,
}

# Prédicat fourre-tout pour les prédicats hors vocabulaire non mappables :
# le predicate brut est préservé dans l'object ("<predicat brut>: <object>").
FALLBACK_PREDICATE = "has_attribute"

ENTITY_TYPES = frozenset({"person", "org", "place", "concept", "product"})

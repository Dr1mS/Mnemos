"""Salience tagger (§13).

Phase 2 : uniquement le contrat SalienceScores (consommé par
EpisodicStore.write). Le SalienceTagger (appel LLM + queue async §13.3)
arrive en Phase 3.
"""

from __future__ import annotations

from typing import TypedDict


class SalienceScores(TypedDict):
    surprise: float  # [0..1]
    arousal: float  # [0..1] — intensité émotionnelle, positive OU négative.
    #                 Nommé "arousal" et pas "valence" : une valence serait
    #                 signée ; ici le signe est volontairement perdu.
    self_ref: float  # [0..1]
    recurrence: float  # [0..1]
    combined: float  # [0..1]

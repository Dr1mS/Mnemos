"""Scénarios d'évaluation des 3 épreuves éliminatoires."""

from __future__ import annotations

from bench.scenarios.test1_ghost_vector import GhostVectorResult, run_test1_ghost_vector
from bench.scenarios.test2_bland_noise import BlandNoiseResult, run_test2_bland_noise
from bench.scenarios.test3_compliance import ComplianceResult, run_test3_compliance

__all__ = [
    "BlandNoiseResult",
    "ComplianceResult",
    "GhostVectorResult",
    "run_test1_ghost_vector",
    "run_test2_bland_noise",
    "run_test3_compliance",
]

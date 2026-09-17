"""Mnemos Architectural Superiority Benchmark & MemoryAgentBench Suite.

Comprend :
- bench.backends : Implémentations des 7 architectures mémoire sous BaseMemoryBackend.
- bench.scenarios : Les 3 épreuves éliminatoires (Ghost Vector, Bland Noise, Compliance).
- bench.memoryagentbench : Adaptateur et runner pour le benchmark ICLR 2026 MemoryAgentBench.
- bench.bench_report : CLI principal du banc comparatif et génération de rapport.
- bench.bench_memoryagentbench : CLI pour l'évaluation MemoryAgentBench.
"""

from __future__ import annotations

__version__ = "0.1.0"

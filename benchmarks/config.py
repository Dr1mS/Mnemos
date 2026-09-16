"""Configuration pour la suite de benchmarks de Mnemos."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class BenchConfig:
    # Répertoire isolé pour ne JAMAIS toucher aux bases de production (data/adrien/)
    bench_dir: Path = Path("./data/bench")
    episodic_db: Path = Path("./data/bench/episodic.db")
    semantic_db: Path = Path("./data/bench/semantic.db")
    procedural_dir: Path = Path("./data/bench/procedural")

    # Mode d'exécution :
    # - "fast" : embeddings déterministes synthétiques (1024-dim normalisés) + LLM mock.
    #   Permet de tester 5 000 à 50 000 entrées, la concurrence et la dégradation en quelques secondes.
    # - "real" : véritable inférence locale via Ollama (bge-m3 + qwen3:4b).
    mode: str = "fast"

    # Échelle de test : "small", "medium", "heavy"
    scale: str = "medium"

    # Ollama host
    ollama_host: str = "http://localhost:11434"
    embed_model: str = "bge-m3"
    llm_model: str = "qwen3:4b"

    # Paramètres de volume selon l'échelle (mode fast)
    volume_steps: list[int] = field(
        default_factory=lambda: [500, 1000, 2500, 5000]
    )

    def apply_scale(self) -> None:
        if self.scale == "small":
            if self.mode == "fast":
                self.volume_steps = [200, 500, 1000]
            else:
                self.volume_steps = [20, 50, 100]
        elif self.scale == "heavy":
            if self.mode == "fast":
                self.volume_steps = [1000, 2500, 5000, 10000]
            else:
                self.volume_steps = [50, 100, 250, 500]
        elif self.scale == "extreme":
            if self.mode == "fast":
                self.volume_steps = [2500, 5000, 10000, 25000]
            else:
                self.volume_steps = [100, 250, 500, 1000]
        else:  # medium
            if self.mode == "fast":
                self.volume_steps = [500, 1000, 2500, 5000]
            else:
                self.volume_steps = [30, 75, 150]

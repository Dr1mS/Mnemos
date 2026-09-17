"""Configuration centralisée et détection matérielle pour le banc d'essai Mnemos.

Gère :
- La détection du CPU, de la RAM et de la disponibilité GPU/Ollama.
- L'optimisation pour inférence CPU (stripping de balises <think>, num_predict, temperature=0.0).
- L'isolation des répertoires de données de benchmark.
- L'embedder déterministe ultra-rapide pour le mode --dry-run.
"""

from __future__ import annotations

import math
import os
import platform
import re
import shutil
from dataclasses import dataclass, field
from hashlib import blake2b
from pathlib import Path
from typing import Any

import httpx

DEFAULT_BENCH_DIR = Path("data/bench_superiority")


def detect_system_hardware() -> dict[str, Any]:
    """Détecte le matériel actuel (CPU, RAM, GPU) de manière multiplateforme."""
    cpu_info = platform.processor() or "x86_64"
    # Lecture sous Linux de /proc/cpuinfo si possible
    cpu_model = "Unknown CPU"
    cpuinfo_path = Path("/proc/cpuinfo")
    if cpuinfo_path.exists():
        try:
            for line in cpuinfo_path.read_text(encoding="utf-8").splitlines():
                if "model name" in line:
                    cpu_model = line.split(":", 1)[1].strip()
                    break
        except Exception:  # noqa: BLE001
            pass

    # RAM
    total_ram_gb = 0.0
    meminfo_path = Path("/proc/meminfo")
    if meminfo_path.exists():
        try:
            for line in meminfo_path.read_text(encoding="utf-8").splitlines():
                if "MemTotal" in line:
                    kb = int(line.split()[1])
                    total_ram_gb = round(kb / (1024 * 1024), 2)
                    break
        except Exception:  # noqa: BLE001
            pass

    # GPU
    has_gpu = False
    if shutil.which("nvidia-smi"):
        has_gpu = True

    return {
        "os": platform.system(),
        "release": platform.release(),
        "cpu_model": cpu_model if cpu_model != "Unknown CPU" else cpu_info,
        "cpu_cores": os.cpu_count() or 4,
        "ram_gb": total_ram_gb,
        "has_gpu": has_gpu,
    }


class DeterministicFastEmbedder:
    """Embedder synthétique déterministe 1024-dim normalisé (L2 = 1.0).

    Dérive un vecteur pseudo-aléatoire reproductible via blake2b.
    Permet des tests à l'octet près sans dépendance réseau ou surcharge CPU/GPU.
    """

    def __init__(self, dim: int = 1024) -> None:
        self.dim = dim

    async def embed(self, text_input: str) -> list[float]:
        h = blake2b(text_input.encode("utf-8"), digest_size=32).digest()
        raw: list[float] = []
        for i in range(self.dim):
            byte_val = h[i % 32]
            val = math.sin((byte_val + 1) * (i + 1) * 0.1)
            raw.append(val)
        norm = math.sqrt(sum(x * x for x in raw))
        if norm == 0:
            return [1.0 / math.sqrt(self.dim)] * self.dim
        return [x / norm for x in raw]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [await self.embed(t) for t in texts]


def strip_think_tags(text: str) -> str:
    """Supprime les balises de réflexion interne <think>...</think> pour qwen3/qwen3.5."""
    clean = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return clean.strip()


@dataclass
class BenchConfig:
    """Configuration du banc d'essai comparatif."""

    bench_dir: Path = DEFAULT_BENCH_DIR
    ollama_host: str = "http://localhost:11434"
    embed_model: str = "bge-m3:latest"
    llm_model: str = "qwen2.5:3b"
    dry_run: bool = False
    temperature: float = 0.0
    seed: int = 42
    results_dir: Path = Path("bench/results")
    hardware: dict[str, Any] = field(default_factory=detect_system_hardware)

    def __post_init__(self) -> None:
        self.bench_dir = Path(self.bench_dir)
        self.results_dir = Path(self.results_dir)

    async def is_ollama_alive(self) -> bool:
        if self.dry_run:
            return True
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                resp = await client.get(f"{self.ollama_host.rstrip('/')}/api/version")
                return resp.status_code == 200
        except Exception:  # noqa: BLE001
            return False

"""Adaptateur générique pour le jeu d'évaluation inspiré de MemoryAgentBench.

Permet à tout système de mémoire implémentant BaseMemoryBackend de s'interfacer
avec le protocole d'ingestion séquentielle par chunks et de questionnement.
"""

from __future__ import annotations

from bench.backends.base import BaseMemoryBackend
from bench.config import BenchConfig
from bench.eval_utils import generate_llm_response


class UnifiedBenchmarkAgent:
    """Agent mémoire générique pour l'évaluation multi-compétences."""

    def __init__(self, backend: BaseMemoryBackend, config: BenchConfig) -> None:
        self.backend = backend
        self.config = config

    async def setup(self) -> None:
        await self.backend.setup()

    async def reset(self) -> None:
        await self.backend.reset()

    async def add_chunk(self, chunk_text: str) -> None:
        """Ingère un chunk textuel incrémental dans le backend de mémoire."""
        await self.backend.write(chunk_text, role="user", timestamp_offset_days=0.01)

    async def answer(self, question: str) -> str:
        """Rappelle le contexte depuis le backend et sollicite le modèle pour répondre."""
        recalled = await self.backend.recall(question, k=4)
        context_str = "\n".join(recalled)

        prompt = (
            f"Tu es un assistant doté d'une mémoire persistante.\n"
            f"Informations extraites de la mémoire :\n\"\"\"\n{context_str}\n\"\"\"\n\n"
            f"Question : {question}\n"
            f"Réponds de manière concise, directe et factuelle en utilisant uniquement le contexte fourni."
        )

        return await generate_llm_response(prompt, self.config)

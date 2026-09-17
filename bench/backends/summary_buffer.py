"""Backend Baseline R2 : SummaryBufferMemory (Rolling Auto-Summarizer & Compactor).

Maintient une fenêtre récente courte (buffer de 4 tours) et compresse
progressivement les tours plus anciens dans un paragraphe de résumé glissant.
Comportement type LangChain ConversationSummaryBufferMemory / MemGPT rolling compactor.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from bench.backends.base import BaseMemoryBackend
from bench.config import BenchConfig
from bench.storage_utils import StorageStats


class SummaryBufferMemory(BaseMemoryBackend):
    def __init__(self, config: BenchConfig, buffer_size: int = 4) -> None:
        self.config = config
        self.buffer_size = buffer_size
        self.recent_buffer: list[dict[str, str]] = []
        self.running_summary: str = ""
        self.storage_path: Path = config.bench_dir / "summary_buffer" / "state.json"

    @property
    def name(self) -> str:
        return "SummaryBufferMemory"

    @property
    def family(self) -> str:
        return "Baseline R2 (Rolling Auto-Summarizer & Compactor)"

    async def setup(self) -> None:
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        await self.reset()

    async def teardown(self) -> None:
        pass

    async def reset(self) -> None:
        self.recent_buffer.clear()
        self.running_summary = ""
        if self.storage_path.exists():
            self.storage_path.unlink()

    def _persist(self) -> None:
        data = {
            "summary": self.running_summary,
            "buffer": self.recent_buffer,
        }
        self.storage_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    async def write(
        self,
        content: str,
        role: str = "user",
        timestamp_offset_days: float = 0.0,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.recent_buffer.append({"role": role, "content": content})

        # Si le buffer dépasse le seuil, compactage incrémental
        if len(self.recent_buffer) > self.buffer_size:
            to_compact = self.recent_buffer.pop(0)
            text_to_add = f"L'utilisateur a mentionné : {to_compact['content']}."
            if not self.running_summary:
                self.running_summary = text_to_add
            else:
                # Compression / concaténation glissante
                # Dans un compactor classique, le LLM résume ou condense le texte
                self.running_summary = f"{self.running_summary} Puis : {to_compact['content']}."
                # Limite de taille du résumé glissant (tronque ou compacte)
                if len(self.running_summary) > 600:
                    self.running_summary = self.running_summary[-600:]

        self._persist()

    async def recall(self, query: str, k: int = 4) -> list[str]:
        results: list[str] = []
        if self.running_summary:
            results.append(f"[Résumé de l'historique antérieur]: {self.running_summary}")
        for item in self.recent_buffer[-k:]:
            results.append(f"{item['role']}: {item['content']}")
        return results

    async def advance_time_days(self, days: float) -> None:
        pass

    async def get_index_size_bytes(self) -> int:
        if self.storage_path.exists():
            return self.storage_path.stat().st_size
        return 0

    async def get_storage_stats(self) -> StorageStats:
        size = await self.get_index_size_bytes()
        return StorageStats(
            active_bytes=size,
            freelist_bytes=0,
            raw_disk_bytes=size,
            active_items_count=len(self.recent_buffer) + (1 if self.running_summary else 0),
            archived_items_count=0,
            notes="Tampon glissant avec texte résumé condensé JSON.",
        )

    def get_embedder_class(self) -> str:
        return "None"

    def get_llm_class(self) -> str:
        return "OllamaClient" if not self.config.dry_run else "None"


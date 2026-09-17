"""Backend Baseline R0 : SlidingWindowMemory (ConversationBufferWindow).

Maintient une fenêtre glissante stricte des K derniers tours (par défaut K=10).
Tous les tours plus anciens sont irrémédiablement oubliés (amnésie FIFO).
"""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from typing import Any

from bench.backends.base import BaseMemoryBackend
from bench.config import BenchConfig
from bench.storage_utils import StorageStats


class SlidingWindowMemory(BaseMemoryBackend):
    def __init__(self, config: BenchConfig, max_turns: int = 10) -> None:
        self.config = config
        self.max_turns = max_turns
        self.buffer: deque[dict[str, Any]] = deque(maxlen=max_turns)
        self.current_time_days: float = 0.0
        self.storage_path: Path = config.bench_dir / "sliding_window" / "buffer.json"

    @property
    def name(self) -> str:
        return "SlidingWindowMemory"

    @property
    def family(self) -> str:
        return "Baseline R0 (FIFO Recency Window)"

    async def setup(self) -> None:
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        await self.reset()

    async def teardown(self) -> None:
        pass

    async def reset(self) -> None:
        self.buffer.clear()
        self.current_time_days = 0.0
        if self.storage_path.exists():
            self.storage_path.unlink()

    async def write(
        self,
        content: str,
        role: str = "user",
        timestamp_offset_days: float = 0.0,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.current_time_days += timestamp_offset_days
        item = {
            "role": role,
            "content": content,
            "time_days": self.current_time_days,
            "metadata": metadata or {},
        }
        self.buffer.append(item)
        # Sauvegarde disque
        self.storage_path.write_text(json.dumps(list(self.buffer), ensure_ascii=False), encoding="utf-8")

    async def recall(self, query: str, k: int = 4) -> list[str]:
        # Retourne les k derniers éléments de la fenêtre glissante
        recent = list(self.buffer)[-k:]
        return [f"{item['role']}: {item['content']}" for item in recent]

    async def advance_time_days(self, days: float) -> None:
        self.current_time_days += days

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
            active_items_count=len(self.buffer),
            archived_items_count=0,
            notes=f"Tampon glissant FIFO JSON (taille fixe max={self.max_turns}).",
        )

    def get_embedder_class(self) -> str:
        return "None"

    def get_llm_class(self) -> str:
        return "None"


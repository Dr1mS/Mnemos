"""Backend Baseline R1 : ObsidianMarkdownMemory (Append-Only Journal & Full-Text).

Écrit séquentiellement chaque tour dans des fichiers Markdown journaliers.
Recherche par récence + matching lexical textuel.
Ne possède aucune sémantique d'invalidation temporelle : les faits obsolètes
persistent indéfiniment dans les notes quotidiennes.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Any

from bench.backends.base import BaseMemoryBackend
from bench.config import BenchConfig
from bench.storage_utils import StorageStats


class ObsidianMarkdownMemory(BaseMemoryBackend):
    def __init__(self, config: BenchConfig) -> None:
        self.config = config
        self.vault_dir: Path = config.bench_dir / "obsidian_vault"
        self.current_day: int = 1

    @property
    def name(self) -> str:
        return "ObsidianMarkdownMemory"

    @property
    def family(self) -> str:
        return "Baseline R1 (Append-only & Full-Text)"

    async def setup(self) -> None:
        self.vault_dir.mkdir(parents=True, exist_ok=True)
        await self.reset()

    async def teardown(self) -> None:
        pass

    async def reset(self) -> None:
        if self.vault_dir.exists():
            shutil.rmtree(self.vault_dir)
        self.vault_dir.mkdir(parents=True, exist_ok=True)
        self.current_day = 1

    def _get_current_note_path(self) -> Path:
        return self.vault_dir / f"Daily_Note_Day_{self.current_day:03d}.md"

    async def write(
        self,
        content: str,
        role: str = "user",
        timestamp_offset_days: float = 0.0,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if timestamp_offset_days > 0:
            self.current_day += max(1, int(timestamp_offset_days))

        note_path = self._get_current_note_path()
        with note_path.open("a", encoding="utf-8") as f:
            f.write(f"- [Day {self.current_day}] [{role.upper()}]: {content}\n")

    async def recall(self, query: str, k: int = 4) -> list[str]:
        """Recherche lexicale + récence : scanne tous les fichiers Markdown."""
        notes = sorted(self.vault_dir.glob("Daily_Note_Day_*.md"))
        if not notes:
            return []

        all_lines: list[str] = []
        for note in notes:
            all_lines.extend(note.read_text(encoding="utf-8").splitlines())

        # Extraction des mots-clés discriminants de la requête (>3 lettres)
        query_words = [w.lower() for w in re.findall(r"\w+", query) if len(w) > 3]
        matches: list[tuple[float, str]] = []

        for idx, line in enumerate(all_lines):
            line_lower = line.lower()
            score = 0.0
            for w in query_words:
                if w in line_lower:
                    score += 1.0
            # Bonus de récence pour les dernières lignes
            recency_bonus = (idx / len(all_lines)) * 0.5
            if score > 0:
                matches.append((score + recency_bonus, line))

        # Si des matches lexicaux existent, tri par score décroissant
        if matches:
            matches.sort(key=lambda x: x[0], reverse=True)
            return [m[1] for m in matches[:k]]

        # Sinon fallback sur les k dernières lignes (fenêtre de récence pure)
        return all_lines[-k:]

    async def advance_time_days(self, days: float) -> None:
        self.current_day += max(1, int(days))

    async def get_index_size_bytes(self) -> int:
        total = 0
        for p in self.vault_dir.rglob("*.md"):
            total += p.stat().st_size
        return total

    async def get_storage_stats(self) -> StorageStats:
        total_size = await self.get_index_size_bytes()
        files = list(self.vault_dir.rglob("*.md"))
        lines_count = 0
        for f in files:
            lines_count += sum(1 for line in f.read_text(encoding="utf-8").splitlines() if line.strip())
        return StorageStats(
            active_bytes=total_size,
            freelist_bytes=0,
            raw_disk_bytes=total_size,
            active_items_count=lines_count,
            archived_items_count=0,
            notes=f"Fichiers Markdown bruts append-only ({len(files)} notes quotidiennes).",
        )

    def get_embedder_class(self) -> str:
        return "None"

    def get_llm_class(self) -> str:
        return "None"


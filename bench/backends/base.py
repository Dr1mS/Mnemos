"""Interface commune BaseMemoryBackend pour l'évaluation comparative des systèmes mémoire."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from bench.storage_utils import StorageStats


@dataclass(frozen=True)
class RecallResult:
    """Résultat standardisé retourné lors d'un recall de mémoire."""

    items: list[str] = field(default_factory=list)
    raw_text: str = ""
    stale_facts_found: list[str] = field(default_factory=list)
    active_facts_found: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseMemoryBackend(ABC):
    """Classe de base abstraite que chaque système de mémoire doit implémenter."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Nom lisible du backend."""

    @property
    @abstractmethod
    def family(self) -> str:
        """Catégorie ou famille architecturale (R0, R1, R2, R3, SOTA, Biologique)."""

    @abstractmethod
    async def setup(self) -> None:
        """Initialisation des dossiers, bases de données et connexions."""

    @abstractmethod
    async def teardown(self) -> None:
        """Fermeture propre des ressources."""

    @abstractmethod
    async def reset(self) -> None:
        """Purge et remise à zéro de l'état mémoire entre les épreuves."""

    @abstractmethod
    async def write(
        self,
        content: str,
        role: str = "user",
        timestamp_offset_days: float = 0.0,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Enregistre un tour de conversation ou un événement dans la mémoire."""

    @abstractmethod
    async def recall(self, query: str, k: int = 4) -> list[str]:
        """Récupère les éléments contextuels les plus pertinents pour la requête."""

    @abstractmethod
    async def advance_time_days(self, days: float) -> None:
        """Simule l'avancement du temps pour les mécanismes temporels et la décroissance."""

    async def consolidate(self) -> None:
        """Exécute les routines de maintenance/consolidation si supporté."""
        return None

    @abstractmethod
    async def get_index_size_bytes(self) -> int:
        """Retourne l'empreinte disque en octets de l'index actif en mémoire/base."""

    async def get_storage_stats(self) -> StorageStats:
        """Retourne des statistiques détaillées sur le stockage (pages actives, freelist, lignes)."""
        raw = await self.get_index_size_bytes()
        return StorageStats(
            active_bytes=raw,
            freelist_bytes=0,
            raw_disk_bytes=raw,
            active_items_count=0,
            archived_items_count=0,
            notes="Mesure brute",
        )

    def get_embedder_class(self) -> str:
        """Retourne le nom de la classe d'embedding réellement instanciée."""
        return "None"

    def get_llm_class(self) -> str:
        """Retourne le nom de la classe ou client LLM réellement instancié."""
        return "None"


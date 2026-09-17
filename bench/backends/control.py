"""Backends de contrôle pour vérifier la validité de la correction du banc de test.

Contrôles :
1. EmptyControlMemory : mémoire vide. Doit échouer à 100% des sondes de rappel.
2. OracleControlMemory : mémoire parfaite. Restitue le contexte exact pertinent à 100%.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any

from bench.backends.base import BaseMemoryBackend
from bench.storage_utils import StorageStats


def _stem(w: str) -> str:
    if len(w) > 4 and w.endswith("s"):
        return w[:-1]
    return w


def _tokenize(text: str) -> set[str]:
    nfkd = unicodedata.normalize("NFKD", text)
    no_accents = "".join([c for c in nfkd if not unicodedata.combining(c)])
    clean = re.sub(r"[^\w\s]", " ", no_accents.lower())
    words = clean.split()
    # Mots vides élémentaires à ignorer pour l'intersection
    stopwords = {
        "le", "la", "les", "un", "une", "des", "de", "du", "en", "a", "au", "aux",
        "et", "ou", "dans", "par", "pour", "sur", "avec", "sans", "sous", "est",
        "sont", "ete", "qui", "que", "quoi", "dont", "ce", "cet", "cette", "ces",
        "mon", "ma", "mes", "ton", "ta", "tes", "son", "sa", "ses", "notre", "votre",
        "leur", "je", "tu", "il", "elle", "nous", "vous", "ils", "elles", "moi",
        "toi", "lui", "eux", "quel", "quelle", "quels", "quelles", "comment", "pourquoi"
    }
    return {_stem(w) for w in words if len(w) > 2 and w not in stopwords}


class EmptyControlMemory(BaseMemoryBackend):
    """Mémoire de contrôle vide. Aucune information n'est stockée ni retournée."""

    def __init__(self, config: Any = None) -> None:
        self.config = config

    @property
    def name(self) -> str:
        return "EmptyControlMemory"

    @property
    def family(self) -> str:
        return "Contrôle Négatif (Mémoire Vide)"

    async def setup(self) -> None:
        pass

    async def teardown(self) -> None:
        pass

    async def reset(self) -> None:
        pass

    async def write(
        self,
        content: str,
        role: str = "user",
        timestamp_offset_days: float = 0.0,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        pass

    async def recall(self, query: str, k: int = 4) -> list[str]:
        return []

    async def advance_time_days(self, days: float) -> None:
        pass

    async def get_index_size_bytes(self) -> int:
        return 0

    async def get_storage_stats(self) -> StorageStats:
        return StorageStats(
            active_bytes=0,
            freelist_bytes=0,
            raw_disk_bytes=0,
            active_items_count=0,
            archived_items_count=0,
            notes="Contrôle vide",
        )

    def get_embedder_class(self) -> str:
        return "None (EmptyControl)"

    def get_llm_class(self) -> str:
        if self.config and hasattr(self.config, "llm_model"):
            return f"{self.config.llm_model} (OllamaClient)"
        return "None"


class OracleControlMemory(BaseMemoryBackend):
    """Mémoire de contrôle parfaite. Restitue les messages stockés pertinents par intersection lexicale."""

    def __init__(self, config: Any = None) -> None:
        self.config = config
        self.stored: list[tuple[str, str]] = []

    @property
    def name(self) -> str:
        return "OracleControlMemory"

    @property
    def family(self) -> str:
        return "Contrôle Positif (Oracle Parfait)"

    async def setup(self) -> None:
        self.stored = []

    async def teardown(self) -> None:
        self.stored = []

    async def reset(self) -> None:
        self.stored = []

    async def write(
        self,
        content: str,
        role: str = "user",
        timestamp_offset_days: float = 0.0,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.stored.append((role, content))

    async def recall(self, query: str, k: int = 4) -> list[str]:
        q_tokens = _tokenize(query)
        if not self.stored:
            return []

        # Cas des requêtes de résidence / villes
        is_chrono_query = any(w in query.lower() for w in ["chronologique", "ordre", "historique", "liste", "toutes les villes", "evolution"])
        is_current_residence = (
            any(w in query.lower() for w in ["actuellement", "maintenant", "aujourd'hui", "actuel"])
            or ("où" in query.lower() and any(w in query.lower() for w in ["habite", "vis", "reside"]))
        ) and not is_chrono_query

        if is_chrono_query or is_current_residence:
            residence_turns: list[str] = []
            for role, content in self.stored:
                norm_c = "".join([c for c in unicodedata.normalize("NFKD", content.lower()) if not unicodedata.combining(c)])
                if any(w in norm_c for w in ["habite", "reside", "demenag", "vis a", "vis en", "install"]):
                    residence_turns.append(f"{role}: {content}")
            if residence_turns:
                if is_current_residence:
                    # Sonde 1.1 : Vérité active stricte -> l'oracle fournit le tour actif complet (user + confirmation)
                    return residence_turns[-2:]
                return residence_turns[-k:]

        # Score par chevauchement de tokens significatifs avec boost sémantique oracle pour conventions
        low_q = query.lower()
        rule_map = {
            "[RULE-HTTP]": ["http", "api", "rest", "telecharg", "webhook", "post", "get", "url", "requete", "web", "client"],
            "[RULE-LOG]": ["log", "journal", "formatteur", "logger", "erreur applicative", "erreur systeme"],
            "[RULE-DATE]": ["date", "horodat", "rapport", "export", "today", "jour"],
            "[RULE-AUTH]": ["auth", "token", "header", "en-tete", "session", "cle", "acces"],
            "[RULE-STORAGE]": ["cache", "temporaire", "stockage", "verrou", "lock", "dossier", "repertoire"],
        }

        scored: list[tuple[int, int, int, str]] = []
        for idx, (role, content) in enumerate(self.stored):
            c_tokens = _tokenize(content)
            overlap = len(q_tokens.intersection(c_tokens))
            is_system = 1 if role == "system" else 0

            # Boost oracle parfait pour les règles système pertinentes
            if role == "system":
                for marker, kw_list in rule_map.items():
                    if marker in content and any(kw in low_q for kw in kw_list):
                        overlap += 10

            if overlap > 0:
                scored.append((is_system, overlap, idx, f"{role}: {content}"))

        if scored:
            # Tri : rôle system prioritaire, puis nombre de tokens, puis récence
            scored.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
            return [item[3] for item in scored[:k]]

        # Repli : derniers messages enregistrés
        return [f"{role}: {content}" for role, content in self.stored[-k:]]

    async def advance_time_days(self, days: float) -> None:
        pass

    async def get_index_size_bytes(self) -> int:
        return 1024

    async def get_storage_stats(self) -> StorageStats:
        return StorageStats(
            active_bytes=1024,
            freelist_bytes=0,
            raw_disk_bytes=1024,
            active_items_count=len(self.stored),
            archived_items_count=0,
            notes="Contrôle oracle",
        )

    def get_embedder_class(self) -> str:
        return "None (OracleRecall)"

    def get_llm_class(self) -> str:
        if self.config and hasattr(self.config, "llm_model"):
            return f"{self.config.llm_model} (OllamaClient)"
        return "None"

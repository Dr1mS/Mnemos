"""Configuration applicative (§6) — pydantic-settings, chargée depuis .env.

Une seule instance Settings, injectée via DI FastAPI (api/deps.py).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

from mnemos.tenancy import DEFAULT_TENANT


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Multi-tenant (P1). Le tenant appliqué par les surfaces mono-tenant
    # (serveur MCP, CLI) quand aucun tenant explicite n'est fourni. Défaut =
    # mémoire personnelle. Une instance MCP par tenant (ex. un NPC Tomodochi)
    # se configure via TENANT dans son .mcp.json.
    TENANT: str = DEFAULT_TENANT

    # Ollama
    OLLAMA_HOST: str = "http://localhost:11434"
    EMBED_MODEL: str = "bge-m3"
    # Backend d'embeddings. "llamacpp" parle à llama-server en direct : ~10x
    # plus rapide et sans le 400 passager d'Ollama sous concurrence (§11 des
    # notes de compétition). La génération reste sur Ollama dans les deux cas.
    EMBED_BACKEND: Literal["ollama", "llamacpp"] = "ollama"
    LLAMACPP_HOST: str = "http://127.0.0.1:8899"
    SALIENCE_MODEL: str = "qwen2.5:3b"
    EXTRACTION_MODEL: str = "qwen2.5:3b"
    # Jamais de mode thinking (JSON cassé sous Ollama + latence ×5-10, cf. §2)
    LLM_THINK: bool = False

    # Storage
    DATA_DIR: Path = Path("./data")
    EPISODIC_DB: Path = Path("./data/episodic.db")
    SEMANTIC_DB: Path = Path("./data/semantic.db")
    PROCEDURAL_DIR: Path = Path("./data/procedural")

    # Server
    API_HOST: str = "127.0.0.1"
    API_PORT: int = 8765
    LOG_LEVEL: str = "INFO"
    API_KEY: str | None = None  # None = ouvert sur localhost (§16)

    # Memory dynamics
    SALIENCE_THRESHOLD_CONSOLIDATE: float = 0.6
    SALIENCE_THRESHOLD_DECAY_FAST: float = 0.2
    DECAY_RATE_DAILY: float = 0.05
    CONSOLIDATION_DELAY_HOURS: float = 1
    EPISODIC_RETENTION_DAYS: int = 90

    # Consolidation worker
    CONSOLIDATION_AUTO: bool = True
    CONSOLIDATION_INTERVAL_SECONDS: float = 5.0
    CONSOLIDATION_INTERVAL_MINUTES: int = 60
    CONSOLIDATION_BATCH_SIZE: int = 20

    # Concurrency (ModelManager §7.2) & Queue (§13.3)
    LLM_TIER_SMALL_CONCURRENCY: int = 4
    LLM_TIER_MEDIUM_CONCURRENCY: int = 1
    SALIENCE_QUEUE_MAXSIZE: int = 1000
    SALIENCE_QUEUE_WORKERS: int = 2


@lru_cache
def get_settings() -> Settings:
    return Settings()

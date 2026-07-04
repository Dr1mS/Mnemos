"""Configuration applicative (§6) — pydantic-settings, chargée depuis .env.

Une seule instance Settings, injectée via DI FastAPI (api/deps.py).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

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
    SALIENCE_MODEL: str = "qwen3:4b"
    EXTRACTION_MODEL: str = "qwen3:4b"
    # Famille qwen3 : jamais de mode thinking (JSON cassé sous Ollama + latence ×5-10, cf. §2)
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
    CONSOLIDATION_INTERVAL_MINUTES: int = 60
    CONSOLIDATION_BATCH_SIZE: int = 20

    # Concurrency (ModelManager §7.2)
    LLM_TIER_SMALL_CONCURRENCY: int = 4
    LLM_TIER_MEDIUM_CONCURRENCY: int = 1


@lru_cache
def get_settings() -> Settings:
    return Settings()

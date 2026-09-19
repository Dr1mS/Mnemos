from __future__ import annotations

from pathlib import Path

from mnemos.config import Settings


def test_defaults_match_spec() -> None:
    s = Settings(_env_file=None)  # ignore le .env local
    # Modèles AML (cf. AML_DEPLOYMENT_WINDOWS.md)
    assert s.SALIENCE_MODEL == "qwen2.5:3b"
    assert s.EXTRACTION_MODEL == "qwen2.5:3b"
    assert s.EMBED_MODEL == "bge-m3"
    assert s.LLM_THINK is False  # jamais de thinking (§2)
    assert s.SALIENCE_THRESHOLD_CONSOLIDATE == 0.6
    assert s.LLM_TIER_MEDIUM_CONCURRENCY == 1


def test_env_override(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("EXTRACTION_MODEL", "qwen3:8b")
    monkeypatch.setenv("API_PORT", "9999")
    s = Settings(_env_file=None)
    assert s.EXTRACTION_MODEL == "qwen3:8b"
    assert s.API_PORT == 9999


def test_paths_are_paths() -> None:
    s = Settings(_env_file=None)
    assert isinstance(s.EPISODIC_DB, Path)
    assert isinstance(s.DATA_DIR, Path)


def test_tenant_defaut_est_user() -> None:
    """P1 : le tenant appliqué par les surfaces mono-tenant (MCP, CLI) est
    `user` par défaut → non-régression des clients existants."""
    assert Settings(_env_file=None).TENANT == "user"


def test_tenant_configurable_par_env(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """Une instance MCP par tenant : TENANT dans le .mcp.json (env)."""
    monkeypatch.setenv("TENANT", "atelios")
    assert Settings(_env_file=None).TENANT == "atelios"

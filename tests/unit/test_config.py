from __future__ import annotations

from pathlib import Path

from mnemos.config import Settings


def test_defaults_match_spec() -> None:
    s = Settings(_env_file=None)  # ignore le .env local
    assert s.SALIENCE_MODEL == "qwen3:4b"
    assert s.EXTRACTION_MODEL == "qwen3:4b"
    assert s.EMBED_MODEL == "bge-m3"
    assert s.LLM_THINK is False  # qwen3 : jamais de thinking (§2)
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

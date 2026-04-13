from __future__ import annotations

from valocoach.core.config import Settings


def test_settings_loads_with_defaults():
    s = Settings(_env_file=None)
    assert s.ollama_model.startswith("qwen3")
    assert s.riot_region in {"na", "eu", "ap", "kr", "latam", "br"}
    assert 0.0 <= s.llm_temperature <= 2.0


def test_settings_respects_env(monkeypatch):
    monkeypatch.setenv("OLLAMA_MODEL", "qwen3:14b")
    s = Settings(_env_file=None)
    assert s.ollama_model == "qwen3:14b"

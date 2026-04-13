from __future__ import annotations

import pytest

from valocoach.core.config import Settings


@pytest.fixture
def settings() -> Settings:
    return Settings(
        riot_name="TestUser",
        riot_tag="TEST",
        riot_region="na",
        henrikdev_api_key="fake",
        ollama_model="qwen3:8b",
        ollama_host="http://localhost:11434",
    )

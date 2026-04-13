from __future__ import annotations

from pathlib import Path
from typing import Literal

import tomli_w
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

CONFIG_DIR = Path.home() / ".valocoach"
CONFIG_PATH = CONFIG_DIR / "config.toml"


class Settings(BaseSettings):
    """Application settings, loaded from env > .env > config.toml > defaults."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="",
        extra="ignore",
    )

    # Riot identity
    riot_name: str = Field(default="", description="Riot username (without #tag)")
    riot_tag: str = Field(default="", description="Riot tag (e.g. NA1)")
    riot_region: Literal["na", "eu", "ap", "kr", "latam", "br"] = "na"

    # APIs
    henrikdev_api_key: str = Field(default="", description="HenrikDev API key")

    # LLM
    ollama_model: str = "qwen3:8b"
    ollama_host: str = "http://localhost:11434"
    llm_temperature: float = 0.6
    llm_max_tokens: int = 3000

    # Paths
    data_dir: Path = Field(default_factory=lambda: CONFIG_DIR / "data")


def load_settings() -> Settings:
    """Load settings, ensuring data dir exists."""
    settings = Settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    return settings


def write_default_config() -> Path:
    """Write a starter config.toml the user can edit."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    if CONFIG_PATH.exists():
        return CONFIG_PATH
    defaults = {
        "riot_name": "",
        "riot_tag": "",
        "riot_region": "na",
        "ollama_model": "qwen3:8b",
        "ollama_host": "http://localhost:11434",
    }
    with CONFIG_PATH.open("wb") as f:
        tomli_w.dump(defaults, f)
    return CONFIG_PATH

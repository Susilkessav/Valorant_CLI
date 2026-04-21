"""Application settings.

Resolution order (highest precedence first):
    1. Process environment variables
    2. ``.env`` in the current working directory
    3. ``~/.valocoach/config.toml``
    4. Field defaults

pydantic-settings does not read TOML out of the box — ``Settings`` wires a
``TomlConfigSettingsSource`` below so ``valocoach config init`` + hand-edit
works as users expect. Without this, the TOML written by
``write_default_config`` is silently ignored.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import tomli_w
from pydantic import Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)

CONFIG_DIR = Path.home() / ".valocoach"
CONFIG_PATH = CONFIG_DIR / "config.toml"


class Settings(BaseSettings):
    """Application settings, loaded from env > .env > config.toml > defaults."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="",
        extra="ignore",
        # Point the TOML source at the user's config. pydantic-settings
        # tolerates a missing file — TomlConfigSettingsSource simply yields
        # an empty dict when the path doesn't exist, so first-run works.
        toml_file=CONFIG_PATH,
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

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Inject the TOML source between dotenv and defaults.

        Earlier sources in the returned tuple win — so process env beats
        ``.env`` beats ``config.toml`` beats field defaults, which matches
        the precedence users expect from a twelve-factor-ish CLI.
        """
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            TomlConfigSettingsSource(settings_cls),
            file_secret_settings,
        )


def load_settings() -> Settings:
    """Load settings, ensuring data dir exists."""
    settings = Settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    return settings


def write_default_config() -> Path:
    """Write a starter config.toml the user can edit.

    Settings() reads this file on every load via TomlConfigSettingsSource,
    so edits take effect on the next CLI invocation — no env-var juggling
    required.
    """
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

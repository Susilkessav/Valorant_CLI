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

from functools import lru_cache
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
        env_prefix="VALOCOACH_",
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

    # Tavily web search / extract API (optional — enhances JS-heavy scraping)
    # Free tier: 1 000 credits/month  https://tavily.com
    # Used by meta-refresh (stats scraping) and ingest --url (JS-rendered pages).
    # Leave empty to use the built-in trafilatura scraper instead.
    tavily_api_key: str = Field(default="", description="Tavily API key (optional)")

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


@lru_cache(maxsize=1)
def _cached_settings() -> Settings:
    """Build the Settings object once per process.

    ``Settings()`` re-reads TOML, .env, and environment on every construction.
    ``load_settings()`` is called from many call sites in the coach path, so
    memoising removes redundant filesystem reads without changing precedence
    semantics (env → .env → TOML → defaults still holds; just computed once).
    """
    return Settings()


def load_settings() -> Settings:
    """Load settings, ensuring data dir exists.

    Cached for the process lifetime.  Call :func:`reset_settings_cache` after
    rewriting ``config.toml`` (e.g. via ``valocoach config init``) so the
    next consumer sees the new file.
    """
    settings = _cached_settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    return settings


def reset_settings_cache() -> None:
    """Drop the memoised ``Settings`` instance.

    Tests that mutate env / .env / TOML between assertions and any code path
    that writes ``config.toml`` from the same process should call this so
    the next ``load_settings()`` reflects the change.
    """
    _cached_settings.cache_clear()


# Fields the starter config always leaves blank for the user to fill in.
# These are identity / secret values that have no sensible shared default —
# we never want to bake an env-derived secret into the on-disk file.
_USER_SUPPLIED_FIELDS: frozenset[str] = frozenset(
    {"riot_name", "riot_tag", "henrikdev_api_key", "tavily_api_key"}
)


def default_config_values() -> dict[str, object]:
    """Build the full starter-config mapping from the ``Settings`` model.

    Derived from ``Settings.model_fields`` rather than a hand-maintained dict
    so EVERY field is written — including ``data_dir`` and any field added in
    the future — and the file never ends up with missing/blank keys that the
    user would have to discover and add by hand.

    Defaults come from the field declarations (not a live ``Settings()``,
    which would absorb env / ``.env`` values and could leak a secret into the
    file).  Identity/secret fields are forced blank.  ``Path`` values are
    stringified so ``tomli_w`` can serialise them.
    """
    from pydantic_core import PydanticUndefined

    values: dict[str, object] = {}
    for name, field in Settings.model_fields.items():
        if name in _USER_SUPPLIED_FIELDS:
            values[name] = ""
            continue

        if field.default is not PydanticUndefined:
            value = field.default
        elif field.default_factory is not None:
            value = field.default_factory()  # type: ignore[call-arg]
        else:
            value = ""

        # TOML has no Path type — store filesystem paths as strings.
        if isinstance(value, Path):
            value = str(value)
        values[name] = value

    return values


def write_default_config() -> Path:
    """Write a starter config.toml the user can edit.

    Settings() reads this file on every load via TomlConfigSettingsSource,
    so edits take effect on the next CLI invocation — no env-var juggling
    required.  The file is written with every settings key present (blank for
    identity/secret fields, sensible defaults for the rest) so there are no
    hidden keys the user has to add manually.
    """
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    if CONFIG_PATH.exists():
        return CONFIG_PATH
    with CONFIG_PATH.open("wb") as f:
        tomli_w.dump(default_config_values(), f)
    # Bust the Settings cache so the next ``load_settings()`` reads the file
    # we just wrote (e.g. a subsequent ``valocoach config show`` in the same
    # process).
    reset_settings_cache()
    return CONFIG_PATH

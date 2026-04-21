from __future__ import annotations

from pathlib import Path

from pydantic_settings import SettingsConfigDict

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


# ---------------------------------------------------------------------------
# TOML source wiring — pinned so future pydantic-settings upgrades or
# accidental edits to settings_customise_sources don't silently drop
# ~/.valocoach/config.toml back out of the resolution chain.
# ---------------------------------------------------------------------------


def _isolated_settings_cls(toml_path: Path) -> type[Settings]:
    """Settings subclass pinned to a fixture TOML and with .env disabled.

    Keeps these tests deterministic on a dev machine where the project
    .env and the developer's real ~/.valocoach/config.toml would otherwise
    leak in and mask TOML behaviour.
    """

    class _TestSettings(Settings):
        model_config = SettingsConfigDict(
            env_file=None,
            env_prefix="",
            extra="ignore",
            toml_file=toml_path,
        )

    return _TestSettings


def _scrub_riot_env(monkeypatch) -> None:
    """Strip every RIOT_* / riot_* variant from os.environ.

    Needed because the project's real .env populates RIOT_NAME / RIOT_TAG /
    RIOT_REGION into the test process, and some earlier test may have called
    ``python-dotenv``'s ``load_dotenv`` which mutates ``os.environ`` directly
    (monkeypatch-style cleanup doesn't apply to that). Without this scrub,
    the TOML we write would be masked by the leaked env.
    """
    for key in (
        "riot_name",
        "riot_tag",
        "riot_region",
        "RIOT_NAME",
        "RIOT_TAG",
        "RIOT_REGION",
    ):
        monkeypatch.delenv(key, raising=False)


def test_toml_is_read(tmp_path, monkeypatch):
    """A written TOML must actually flow into Settings — otherwise
    `config init` is a trap: users edit a file that's silently ignored."""
    _scrub_riot_env(monkeypatch)

    toml = tmp_path / "config.toml"
    toml.write_text('riot_name = "tomlplayer"\nriot_tag = "NA1"\nriot_region = "eu"\n')

    s = _isolated_settings_cls(toml)()
    assert s.riot_name == "tomlplayer"
    assert s.riot_tag == "NA1"
    assert s.riot_region == "eu"


def test_env_overrides_toml(tmp_path, monkeypatch):
    """Precedence contract: process env beats TOML. Users running a
    one-off `RIOT_NAME=alt valocoach ...` must see their override win."""
    _scrub_riot_env(monkeypatch)
    toml = tmp_path / "config.toml"
    toml.write_text('riot_name = "tomlplayer"\nriot_tag = "NA1"\n')

    monkeypatch.setenv("riot_name", "envwins")

    s = _isolated_settings_cls(toml)()
    assert s.riot_name == "envwins"
    # TOML still contributes fields env doesn't set.
    assert s.riot_tag == "NA1"


def test_missing_toml_is_not_an_error(tmp_path, monkeypatch):
    """First-run must work: user hasn't run `config init` yet, no TOML file
    exists, Settings() should fall through to defaults without raising."""
    _scrub_riot_env(monkeypatch)
    missing = tmp_path / "does-not-exist.toml"
    assert not missing.exists()

    s = _isolated_settings_cls(missing)()
    assert s.riot_name == ""  # field default
    assert s.riot_region == "na"

from __future__ import annotations

from pathlib import Path

from valocoach.core.config import get_settings, set_config_value


def test_get_settings_loads_toml_file(tmp_path: Path, monkeypatch) -> None:
    home = tmp_path / ".valocoach"
    home.mkdir()
    (home / "config.toml").write_text(
        "\n".join(
            [
                "[ollama]",
                'model = "qwen3:14b"',
                "",
                "[ui]",
                "show_thinking = true",
                "",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("VALOCOACH_HOME", str(home))

    settings = get_settings()

    assert settings.ollama.model == "qwen3:14b"
    assert settings.ui.show_thinking is True
    assert settings.paths.data_dir == home / "data"


def test_set_config_value_updates_toml_file(tmp_path: Path, monkeypatch) -> None:
    home = tmp_path / ".valocoach"
    monkeypatch.setenv("VALOCOACH_HOME", str(home))

    config_path = set_config_value("ollama.model", "qwen3:32b")

    assert config_path.exists()
    assert 'model = "qwen3:32b"' in config_path.read_text(encoding="utf-8")

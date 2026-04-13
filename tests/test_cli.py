from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from valocoach.cli.app import app

runner = CliRunner()


def test_config_init_creates_default_file(tmp_path: Path, monkeypatch) -> None:
    home = tmp_path / ".valocoach"
    monkeypatch.setenv("VALOCOACH_HOME", str(home))

    result = runner.invoke(app, ["config", "init"])

    assert result.exit_code == 0
    assert (home / "config.toml").exists()
    assert "Created config" in result.output


def test_config_show_reads_overrides(tmp_path: Path, monkeypatch) -> None:
    home = tmp_path / ".valocoach"
    home.mkdir()
    (home / "config.toml").write_text(
        "\n".join(
            [
                "[ollama]",
                'model = "qwen3:14b"',
                "",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("VALOCOACH_HOME", str(home))

    result = runner.invoke(app, ["config", "show"])

    assert result.exit_code == 0
    assert "qwen3:14b" in result.output


def test_coach_passes_structured_flags_into_request(tmp_path: Path, monkeypatch) -> None:
    captured: dict[str, str] = {}
    home = tmp_path / ".valocoach"
    monkeypatch.setenv("VALOCOACH_HOME", str(home))

    class FakeService:
        async def run_and_render(self, request) -> str:
            captured["prompt"] = request.render_user_prompt()
            return "ok"

    class FakeCoachFactory:
        @staticmethod
        def from_settings(settings):  # noqa: ANN001, ANN205
            return FakeService()

    monkeypatch.setattr("valocoach.cli.app.CoachService", FakeCoachFactory)

    result = runner.invoke(
        app,
        ["coach", "test", "--agent", "Jett", "--map", "Haven", "--side", "attack"],
    )

    assert result.exit_code == 0
    assert "Situation: test" in captured["prompt"]
    assert "Agent: Jett" in captured["prompt"]
    assert "Map: Haven" in captured["prompt"]
    assert "Side: attack" in captured["prompt"]

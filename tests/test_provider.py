from __future__ import annotations

from unittest.mock import MagicMock, patch

from valocoach.core.config import Settings
from valocoach.llm.provider import stream_completion


def test_stream_completion_yields_tokens():
    fake_chunks = [
        MagicMock(choices=[MagicMock(delta=MagicMock(content="Hello "))]),
        MagicMock(choices=[MagicMock(delta=MagicMock(content="world"))]),
        MagicMock(choices=[MagicMock(delta=MagicMock(content=None))]),
    ]
    settings = Settings(_env_file=None)

    with patch("valocoach.llm.provider.litellm.completion", return_value=iter(fake_chunks)):
        tokens = list(stream_completion(settings, "sys", "user"))

    assert tokens == ["Hello ", "world"]


def test_stream_completion_adds_ollama_prefix():
    settings = Settings(_env_file=None)
    settings.ollama_model = "qwen3:8b"

    with patch("valocoach.llm.provider.litellm.completion") as mock_call:
        mock_call.return_value = iter([])
        list(stream_completion(settings, "sys", "user"))
        assert mock_call.call_args.kwargs["model"] == "ollama/qwen3:8b"

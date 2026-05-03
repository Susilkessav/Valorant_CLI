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


def test_stream_completion_inserts_conversation_history_between_system_and_user():
    """Multi-turn coaching depends on prior turns landing between the system
    message and the current user message in the LiteLLM messages list."""
    settings = Settings(_env_file=None)
    history = [
        {"role": "user", "content": "first q"},
        {"role": "assistant", "content": "first a"},
        {"role": "user", "content": "second q"},
        {"role": "assistant", "content": "second a"},
    ]

    with patch("valocoach.llm.provider.litellm.completion") as mock_call:
        mock_call.return_value = iter([])
        list(stream_completion(settings, "sys prompt", "current question",
                               conversation_history=history))

    messages = mock_call.call_args.kwargs["messages"]
    # system first, then all history in order, then current user message last.
    assert messages[0] == {"role": "system", "content": "sys prompt"}
    assert messages[1:5] == history
    assert messages[5] == {"role": "user", "content": "current question"}
    assert len(messages) == 6


def test_stream_completion_no_history_keeps_two_message_shape():
    """Without history, the messages list is exactly [system, user] —
    backward-compat with existing one-shot callers."""
    settings = Settings(_env_file=None)

    with patch("valocoach.llm.provider.litellm.completion") as mock_call:
        mock_call.return_value = iter([])
        list(stream_completion(settings, "sys", "user msg"))

    messages = mock_call.call_args.kwargs["messages"]
    assert messages == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "user msg"},
    ]


def test_stream_completion_empty_history_treated_as_none():
    """An empty list (rather than None) should not insert any extra messages."""
    settings = Settings(_env_file=None)

    with patch("valocoach.llm.provider.litellm.completion") as mock_call:
        mock_call.return_value = iter([])
        list(stream_completion(settings, "sys", "user", conversation_history=[]))

    messages = mock_call.call_args.kwargs["messages"]
    assert len(messages) == 2  # system + user only

from __future__ import annotations

from valocoach.cli.display import strip_hidden_thinking


def test_strip_hidden_thinking_removes_complete_block() -> None:
    text = "Visible<think>hidden</think>Still visible"
    assert strip_hidden_thinking(text) == "VisibleStill visible"


def test_strip_hidden_thinking_removes_incomplete_trailing_block() -> None:
    text = "Visible<think>hidden forever"
    assert strip_hidden_thinking(text) == "Visible"

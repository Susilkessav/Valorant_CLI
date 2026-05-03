"""Tests for valocoach.cli.display.

Covers:
  - coach_panel: return type, title, default title, content renderable.
  - stream_to_panel: empty stream, token accumulation, return value,
    Live.update call count, and refresh_per_second forwarding.

Live is mocked so tests run without a TTY or real terminal.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from rich.markdown import Markdown
from rich.panel import Panel

# ---------------------------------------------------------------------------
# coach_panel
# ---------------------------------------------------------------------------


class TestCoachPanel:
    def test_returns_panel_instance(self):
        from valocoach.cli.display import coach_panel

        result = coach_panel("some coaching text")
        assert isinstance(result, Panel)

    def test_default_title(self):
        from valocoach.cli.display import coach_panel

        panel = coach_panel("content")
        # Rich stores the title in panel.title
        assert "Coach" in str(panel.title)

    def test_custom_title_applied(self):
        from valocoach.cli.display import coach_panel

        panel = coach_panel("content", title="My Title")
        assert "My Title" in str(panel.title)

    def test_renderable_is_markdown(self):
        from valocoach.cli.display import coach_panel

        panel = coach_panel("**bold text**")
        assert isinstance(panel.renderable, Markdown)

    def test_empty_content_ok(self):
        from valocoach.cli.display import coach_panel

        panel = coach_panel("")
        assert isinstance(panel, Panel)


# ---------------------------------------------------------------------------
# stream_to_panel
# ---------------------------------------------------------------------------


def _make_live_mock():
    """Return (mock_Live_class, mock_live_instance) for patching."""
    mock_live_instance = MagicMock()
    mock_live_cls = MagicMock()
    # Make `with Live(...) as live:` give back mock_live_instance
    mock_live_cls.return_value.__enter__ = MagicMock(return_value=mock_live_instance)
    mock_live_cls.return_value.__exit__ = MagicMock(return_value=False)
    return mock_live_cls, mock_live_instance


class TestStreamToPanel:
    def test_empty_stream_returns_empty_string(self):
        from valocoach.cli.display import stream_to_panel

        mock_live_cls, _ = _make_live_mock()
        with patch("valocoach.cli.display.Live", mock_live_cls):
            result = stream_to_panel(iter([]))

        assert result == ""

    def test_returns_full_accumulated_text(self):
        from valocoach.cli.display import stream_to_panel

        tokens = ["Hello", ", ", "world", "!"]
        mock_live_cls, _ = _make_live_mock()
        with patch("valocoach.cli.display.Live", mock_live_cls):
            result = stream_to_panel(iter(tokens))

        assert result == "Hello, world!"

    def test_tokens_accumulated_in_order(self):
        from valocoach.cli.display import stream_to_panel

        tokens = ["first", " second", " third"]
        mock_live_cls, _ = _make_live_mock()
        with patch("valocoach.cli.display.Live", mock_live_cls):
            result = stream_to_panel(iter(tokens))

        assert result.startswith("first")
        assert "second" in result
        assert result.endswith("third")

    def test_live_update_called_once_per_token(self):
        from valocoach.cli.display import stream_to_panel

        tokens = ["a", "b", "c"]
        mock_live_cls, mock_live_instance = _make_live_mock()
        with patch("valocoach.cli.display.Live", mock_live_cls):
            stream_to_panel(iter(tokens))

        assert mock_live_instance.update.call_count == len(tokens)

    def test_live_update_receives_panels(self):
        from valocoach.cli.display import stream_to_panel

        tokens = ["tok1", "tok2"]
        mock_live_cls, mock_live_instance = _make_live_mock()
        with patch("valocoach.cli.display.Live", mock_live_cls):
            stream_to_panel(iter(tokens))

        for update_call in mock_live_instance.update.call_args_list:
            arg = update_call[0][0]
            assert isinstance(arg, Panel)

    def test_default_refresh_per_second_forwarded_to_live(self):
        from valocoach.cli.display import stream_to_panel

        mock_live_cls, _ = _make_live_mock()
        with patch("valocoach.cli.display.Live", mock_live_cls):
            stream_to_panel(iter(["x"]))

        # Live() was constructed — check the refresh_per_second kwarg
        _, kwargs = mock_live_cls.call_args
        assert kwargs.get("refresh_per_second") == 8

    def test_custom_refresh_per_second_forwarded(self):
        from valocoach.cli.display import stream_to_panel

        mock_live_cls, _ = _make_live_mock()
        with patch("valocoach.cli.display.Live", mock_live_cls):
            stream_to_panel(iter(["x"]), refresh_per_second=16)

        _, kwargs = mock_live_cls.call_args
        assert kwargs.get("refresh_per_second") == 16

    def test_custom_title_passed_to_initial_panel(self):
        from valocoach.cli.display import stream_to_panel

        mock_live_cls, _ = _make_live_mock()
        with patch("valocoach.cli.display.Live", mock_live_cls):
            stream_to_panel(iter(["x"]), title="Custom Coach")

        # The first positional arg to Live() is the initial renderable (a Panel).
        args, _ = mock_live_cls.call_args
        initial_panel = args[0]
        assert isinstance(initial_panel, Panel)
        assert "Custom Coach" in str(initial_panel.title)

    def test_single_token_stream(self):
        from valocoach.cli.display import stream_to_panel

        mock_live_cls, mock_live_instance = _make_live_mock()
        with patch("valocoach.cli.display.Live", mock_live_cls):
            result = stream_to_panel(iter(["only token"]))

        assert result == "only token"
        assert mock_live_instance.update.call_count == 1

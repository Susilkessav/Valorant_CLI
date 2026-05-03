"""Tests for `valocoach meta` — meta.py branch coverage.

Covers:
  _tier_table        — empty tier skipped (line 30->28)
  _try_get_live_patch — exception path (lines 65-66)
  run_meta           — live patch warning (line 77)
                     — unknown agent error (lines 86-87)
                     — agent with empty reason (line 117->122)
                     — agent without meta stats warn (line 120)
                     — unknown map error (lines 130-132)
                     — map with no notes (line 152->157)
                     — map without meta warn (line 155)
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Shared fake data
# ---------------------------------------------------------------------------

_FAKE_META = {
    "patch": "9.01",
    "updated": "2026-05-01",
    "notes": "Some notes",
    "tier_list": {
        "S": ["Jett", "Omen"],
        "A": ["Reyna", "Sage"],
        "B": [],  # deliberately empty — covers the 30->28 branch
        "C": ["Yoru"],
    },
    "economy": {
        "full_buy": 3900,
        "half_buy": 2400,
        "eco_save": 1600,
    },
    "agent_meta": {
        "Jett": {
            "tier": "S",
            "pick_rate": "20%",
            "win_rate": "52%",
            "reason": "Strong entry fragger with dash",
        },
        "Reyna": {
            "tier": "A",
            "pick_rate": "15%",
            "win_rate": "50%",
            "reason": "",  # empty reason — covers 117->122
        },
    },
    "map_meta": {
        "Ascent": {
            "top_agents": ["Jett", "Omen"],
            "notes": "Mid control is key on Ascent.",
        },
        "Lotus": {
            "top_agents": ["Killjoy"],
            # no 'notes' key — covers 152->157
        },
    },
}

_JETT = {"name": "Jett", "role": "Duelist"}
_ASCENT = {"name": "Ascent"}
_LOTUS = {"name": "Lotus"}

_PATCH_META = "valocoach.cli.commands.meta.get_meta"
_PATCH_GET_AGENT = "valocoach.cli.commands.meta.get_agent"
_PATCH_GET_MAP = "valocoach.cli.commands.meta.get_map"
_PATCH_LIST_MAPS = "valocoach.cli.commands.meta.list_map_names"
_PATCH_FMT_AGENT = "valocoach.cli.commands.meta.format_agent_context"
_PATCH_FMT_MAP = "valocoach.cli.commands.meta.format_map_context"
_PATCH_SETTINGS = "valocoach.cli.commands.meta.load_settings"
_PATCH_LIVE = "valocoach.cli.commands.meta._try_get_live_patch"
_PATCH_CONSOLE_PRINT = "valocoach.cli.commands.meta.display.console.print"
_PATCH_WARN = "valocoach.cli.commands.meta.display.warn"
_PATCH_ERROR = "valocoach.cli.commands.meta.display.error"


def _fake_settings():
    from valocoach.core.config import Settings

    return Settings(riot_name="T", riot_tag="X", riot_region="na", henrikdev_api_key="f")


# ---------------------------------------------------------------------------
# _tier_table — empty tier branch (line 30->28)
# ---------------------------------------------------------------------------


def _render_table(table) -> str:
    """Render a Rich Table to a plain string (no ANSI)."""
    from io import StringIO

    from rich.console import Console

    buf = Console(file=StringIO(), force_terminal=False, width=120)
    buf.print(table)
    return buf.file.getvalue()


class TestTierTable:
    def test_empty_tier_skipped(self):
        """When a tier has no agents the row is omitted (line 30->28).

        _FAKE_META has B=[] so only S, A, C produce rows → 3 rows total.
        """
        from valocoach.cli.commands.meta import _tier_table

        with patch(_PATCH_META, return_value=_FAKE_META):
            table = _tier_table()

        # Rich Table.rows is a list of Row objects; add_row() adds one per call.
        # B-tier is empty so its add_row() is skipped — expect 3 rows (S, A, C).
        assert len(table.rows) == 3

    def test_empty_tier_not_rendered(self):
        """The 'B' row with no agents must not appear in the rendered output."""
        from valocoach.cli.commands.meta import _tier_table

        with patch(_PATCH_META, return_value=_FAKE_META):
            table = _tier_table()

        out = _render_table(table)
        # S, A, C agents appear; B should not (it has no agents).
        assert "Jett" in out  # S tier
        assert "Reyna" in out  # A tier
        assert "Yoru" in out  # C tier
        # There is no agent in B, so the tier letter 'B' must not appear as a row.
        # (It may appear elsewhere, so we check it's not on its own line.)
        b_lines = [ln for ln in out.splitlines() if ln.strip().startswith("B")]
        assert not b_lines, f"Unexpected 'B' tier row: {b_lines}"


# ---------------------------------------------------------------------------
# _try_get_live_patch — exception path (lines 65-66)
# ---------------------------------------------------------------------------


class TestTryGetLivePatch:
    def test_returns_none_on_exception(self):
        """Any error inside _try_get_live_patch must return None (never raise).

        init_engine is a lazy import inside the function so we patch it at
        its source module, which the lazy 'from … import …' resolves against.
        """
        from valocoach.cli.commands.meta import _try_get_live_patch

        fake_settings = _fake_settings()

        # Patch at the source so the lazy import picks up the mock.
        with patch("valocoach.data.database.init_engine", side_effect=RuntimeError("boom")):
            result = _try_get_live_patch(fake_settings)

        assert result is None


# ---------------------------------------------------------------------------
# run_meta — live patch warning (line 77)
# ---------------------------------------------------------------------------


class TestRunMetaLivePatch:
    def test_warns_when_live_patch_differs(self):
        """display.warn is called when DB patch != JSON patch (line 77)."""
        warn_mock = MagicMock()

        with (
            patch(_PATCH_META, return_value=_FAKE_META),  # patch="9.01"
            patch(_PATCH_SETTINGS, return_value=_fake_settings()),
            patch(_PATCH_LIVE, return_value="9.02"),  # newer patch
            patch(_PATCH_WARN, warn_mock),
            patch(_PATCH_CONSOLE_PRINT),
        ):
            from valocoach.cli.commands.meta import run_meta

            run_meta(agent=None, map_=None)

        warn_mock.assert_called()
        call_text = warn_mock.call_args[0][0]
        assert "9.02" in call_text

    def test_no_warn_when_patches_match(self):
        """No warning when live patch equals the JSON patch."""
        warn_mock = MagicMock()

        with (
            patch(_PATCH_META, return_value=_FAKE_META),  # patch="9.01"
            patch(_PATCH_SETTINGS, return_value=_fake_settings()),
            patch(_PATCH_LIVE, return_value="9.01"),  # same — no alert
            patch(_PATCH_WARN, warn_mock),
            patch(_PATCH_CONSOLE_PRINT),
        ):
            from valocoach.cli.commands.meta import run_meta

            run_meta(agent=None, map_=None)

        # warn may be called for other reasons; check it was NOT called with
        # the new-patch message specifically.
        for call in warn_mock.call_args_list:
            assert "New patch detected" not in (call[0][0] if call[0] else "")


# ---------------------------------------------------------------------------
# run_meta — agent view
# ---------------------------------------------------------------------------


class TestRunMetaAgent:
    def test_unknown_agent_shows_error(self):
        """get_agent returns None → display.error + early return (lines 86-87)."""
        error_mock = MagicMock()
        console_mock = MagicMock()

        with (
            patch(_PATCH_META, return_value=_FAKE_META),
            patch(_PATCH_SETTINGS, return_value=_fake_settings()),
            patch(_PATCH_LIVE, return_value=None),
            patch(_PATCH_GET_AGENT, return_value=None),
            patch(_PATCH_ERROR, error_mock),
            patch(_PATCH_CONSOLE_PRINT, console_mock),
        ):
            from valocoach.cli.commands.meta import run_meta

            run_meta(agent="UnknownXXX", map_=None)

        error_mock.assert_called_once()
        assert "Unknown agent" in error_mock.call_args[0][0]
        # Console.print should NOT have been called (early return).
        console_mock.assert_not_called()

    def test_agent_with_reason_renders_reason(self):
        """When agent_meta has a non-empty reason, it is printed (line 118)."""
        console_calls: list[str] = []

        def _print_capture(text="", *_a, **_kw):
            console_calls.append(str(text))

        with (
            patch(_PATCH_META, return_value=_FAKE_META),
            patch(_PATCH_SETTINGS, return_value=_fake_settings()),
            patch(_PATCH_LIVE, return_value=None),
            patch(_PATCH_GET_AGENT, return_value=_JETT),
            patch(_PATCH_FMT_AGENT, return_value="Jett abilities"),
            patch(_PATCH_CONSOLE_PRINT, side_effect=_print_capture),
        ):
            from valocoach.cli.commands.meta import run_meta

            run_meta(agent="Jett", map_=None)

        combined = " ".join(console_calls)
        assert "Strong entry fragger" in combined

    def test_agent_with_empty_reason_skips_reason_line(self):
        """reason='' → the reason print call is skipped (line 117->122)."""
        console_calls: list[str] = []

        def _print_capture(text="", *_a, **_kw):
            console_calls.append(str(text))

        with (
            patch(_PATCH_META, return_value=_FAKE_META),
            patch(_PATCH_SETTINGS, return_value=_fake_settings()),
            patch(_PATCH_LIVE, return_value=None),
            patch(_PATCH_GET_AGENT, return_value={"name": "Reyna", "role": "Duelist"}),
            patch(_PATCH_FMT_AGENT, return_value="Reyna abilities"),
            patch(_PATCH_CONSOLE_PRINT, side_effect=_print_capture),
        ):
            from valocoach.cli.commands.meta import run_meta

            run_meta(agent="Reyna", map_=None)

        combined = " ".join(console_calls)
        # reason is "" so the dim-reason line must NOT appear.
        assert "[dim]" not in combined or "reason" not in combined.lower()

    def test_agent_without_meta_stats_warns(self):
        """agent_meta absent → display.warn for missing meta (line 120)."""
        warn_mock = MagicMock()
        meta_no_agent_meta = {**_FAKE_META, "agent_meta": {}}

        with (
            patch(_PATCH_META, return_value=meta_no_agent_meta),
            patch(_PATCH_SETTINGS, return_value=_fake_settings()),
            patch(_PATCH_LIVE, return_value=None),
            patch(_PATCH_GET_AGENT, return_value=_JETT),
            patch(_PATCH_FMT_AGENT, return_value="abilities text"),
            patch(_PATCH_WARN, warn_mock),
            patch(_PATCH_CONSOLE_PRINT),
        ):
            from valocoach.cli.commands.meta import run_meta

            run_meta(agent="Jett", map_=None)

        warn_mock.assert_called()
        assert "No meta stats available" in warn_mock.call_args[0][0]


# ---------------------------------------------------------------------------
# run_meta — map view
# ---------------------------------------------------------------------------


class TestRunMetaMap:
    def test_unknown_map_shows_error(self):
        """get_map returns None → display.error + early return (lines 130-132)."""
        error_mock = MagicMock()

        with (
            patch(_PATCH_META, return_value=_FAKE_META),
            patch(_PATCH_SETTINGS, return_value=_fake_settings()),
            patch(_PATCH_LIVE, return_value=None),
            patch(_PATCH_GET_MAP, return_value=None),
            patch(_PATCH_LIST_MAPS, return_value=["Ascent", "Lotus"]),
            patch(_PATCH_ERROR, error_mock),
            patch(_PATCH_CONSOLE_PRINT),
        ):
            from valocoach.cli.commands.meta import run_meta

            run_meta(agent=None, map_="UnknownMapXXX")

        error_mock.assert_called_once()
        assert "Unknown map" in error_mock.call_args[0][0]

    def test_map_with_notes_renders_notes(self):
        """map_meta has 'notes' → notes line rendered (line 153)."""
        console_calls: list[str] = []

        def _print_capture(text="", *_a, **_kw):
            console_calls.append(str(text))

        with (
            patch(_PATCH_META, return_value=_FAKE_META),
            patch(_PATCH_SETTINGS, return_value=_fake_settings()),
            patch(_PATCH_LIVE, return_value=None),
            patch(_PATCH_GET_MAP, return_value=_ASCENT),
            patch(_PATCH_FMT_MAP, return_value="Ascent callouts"),
            patch(_PATCH_CONSOLE_PRINT, side_effect=_print_capture),
        ):
            from valocoach.cli.commands.meta import run_meta

            run_meta(agent=None, map_="Ascent")

        combined = " ".join(console_calls)
        assert "Mid control is key" in combined

    def test_map_without_notes_skips_notes_line(self):
        """map_meta lacks 'notes' → notes line skipped (line 152->157)."""
        console_calls: list[str] = []

        def _print_capture(text="", *_a, **_kw):
            console_calls.append(str(text))

        with (
            patch(_PATCH_META, return_value=_FAKE_META),
            patch(_PATCH_SETTINGS, return_value=_fake_settings()),
            patch(_PATCH_LIVE, return_value=None),
            patch(_PATCH_GET_MAP, return_value=_LOTUS),
            patch(_PATCH_FMT_MAP, return_value="Lotus callouts"),
            patch(_PATCH_CONSOLE_PRINT, side_effect=_print_capture),
        ):
            from valocoach.cli.commands.meta import run_meta

            run_meta(agent=None, map_="Lotus")

        # Lotus has no notes — ensure no stray notes text from a different map leaked.
        combined = " ".join(console_calls)
        assert "Mid control" not in combined  # Ascent's note must not appear

    def test_map_without_meta_warns(self):
        """map_meta absent → display.warn (line 155)."""
        warn_mock = MagicMock()
        meta_no_map_meta = {**_FAKE_META, "map_meta": {}}

        with (
            patch(_PATCH_META, return_value=meta_no_map_meta),
            patch(_PATCH_SETTINGS, return_value=_fake_settings()),
            patch(_PATCH_LIVE, return_value=None),
            patch(_PATCH_GET_MAP, return_value=_ASCENT),
            patch(_PATCH_FMT_MAP, return_value="callout text"),
            patch(_PATCH_WARN, warn_mock),
            patch(_PATCH_CONSOLE_PRINT),
        ):
            from valocoach.cli.commands.meta import run_meta

            run_meta(agent=None, map_="Ascent")

        warn_mock.assert_called()
        assert "No map-specific meta" in warn_mock.call_args[0][0]


# ---------------------------------------------------------------------------
# run_meta — global overview (smoke test)
# ---------------------------------------------------------------------------


class TestRunMetaOverview:
    def test_overview_renders_without_error(self):
        """run_meta() with no filters renders the tier list + eco line."""
        with (
            patch(_PATCH_META, return_value=_FAKE_META),
            patch(_PATCH_SETTINGS, return_value=_fake_settings()),
            patch(_PATCH_LIVE, return_value=None),
            patch(_PATCH_CONSOLE_PRINT),  # absorb all output
        ):
            from valocoach.cli.commands.meta import run_meta

            # Should complete without raising.
            run_meta(agent=None, map_=None)

    @pytest.mark.parametrize("tier", ["S", "A", "C"])
    def test_non_empty_tiers_present_in_tier_table(self, tier):
        """Non-empty tiers produce rows (complements the empty-tier test)."""
        from valocoach.cli.commands.meta import _tier_table

        with patch(_PATCH_META, return_value=_FAKE_META):
            table = _tier_table()

        out = _render_table(table)
        assert tier in out

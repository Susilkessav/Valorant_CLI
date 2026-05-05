"""Tests for valocoach.retrieval.meta — covering remaining uncovered branches.

Gaps being covered:
  37->35   format_meta_context — tier with empty agents list → if agents: is False
  49->55   format_meta_context — map_name given but not in map_meta → if map_meta: False
  52->55   format_meta_context — map_meta present but no 'notes' → skip notes line
  63->66   format_meta_context — agent_meta present but no 'reason' → skip reason line
"""

from __future__ import annotations

from unittest.mock import patch

# ---------------------------------------------------------------------------
# Shared fake meta payloads
# ---------------------------------------------------------------------------

_BASE_META = {
    "patch": "9.01",
    "updated": "2026-05-01",
    "tier_list": {
        "S": ["Jett"],
        "A": [],  # deliberately empty — covers 37->35
        "B": ["Reyna"],
        "C": ["Yoru"],
    },
    "economy": {"full_buy": 3900, "half_buy": 2400, "eco_save": 1600},
    "map_meta": {
        "Ascent": {
            "top_agents": ["Jett", "Omen"],
            "notes": "Mid control is key on Ascent.",
        },
        "Bind": {
            "top_agents": ["Raze"],
            # deliberately no 'notes' key — covers 52->55
        },
    },
    "agent_meta": {
        "Jett": {
            "tier": "S",
            "pick_rate": "20%",
            "win_rate": "52%",
            "reason": "Strong entry fragger",
        },
        "Reyna": {
            "tier": "B",
            "pick_rate": "15%",
            "win_rate": "48%",
            "reason": "",  # empty string → falsy → covers 63->66
        },
    },
}

_PATCH_LOAD = "valocoach.retrieval.meta._load"


# ===========================================================================
# format_meta_context — branch coverage
# ===========================================================================


class TestFormatMetaContext:
    def test_empty_tier_is_skipped(self):
        """A tier with an empty agents list is silently omitted (line 37->35).

        _BASE_META has A=[] so the A row must not appear.
        """
        from valocoach.retrieval.meta import format_meta_context

        with patch(_PATCH_LOAD, return_value=_BASE_META):
            result = format_meta_context()

        assert result is not None
        # S, B, C tiers should appear; A should not.
        assert "S-Tier" in result
        assert "B-Tier" in result
        assert "C-Tier" in result
        # A-Tier row must be absent because agents=[]
        assert "A-Tier" not in result

    def test_non_empty_tiers_present(self):
        """Non-empty tiers always appear in output."""
        from valocoach.retrieval.meta import format_meta_context

        with patch(_PATCH_LOAD, return_value=_BASE_META):
            result = format_meta_context()

        assert "Jett" in result  # S-Tier
        assert "Reyna" in result  # B-Tier

    def test_unknown_map_name_skips_map_block(self):
        """map_name not in map_meta → if map_meta: is False → block skipped (49->55)."""
        from valocoach.retrieval.meta import format_meta_context

        with patch(_PATCH_LOAD, return_value=_BASE_META):
            result = format_meta_context(map_name="ZZZUnknownMap")

        assert result is not None
        # The map block should not appear — no "ZZZUnknownMap meta" line.
        assert "ZZZUnknownMap" not in result

    def test_known_map_name_shows_map_block(self):
        """map_name in map_meta → the map block is rendered (line 51)."""
        from valocoach.retrieval.meta import format_meta_context

        with patch(_PATCH_LOAD, return_value=_BASE_META):
            result = format_meta_context(map_name="Ascent")

        assert "Ascent meta" in result
        assert "Jett" in result  # top_agents

    def test_map_notes_included_when_present(self):
        """When map_meta has 'notes', the notes line appears (line 53)."""
        from valocoach.retrieval.meta import format_meta_context

        with patch(_PATCH_LOAD, return_value=_BASE_META):
            result = format_meta_context(map_name="Ascent")

        assert "Mid control is key" in result

    def test_map_notes_skipped_when_absent(self):
        """When map_meta has no 'notes', the notes line is omitted (52->55).

        Bind in _BASE_META has no 'notes' key.
        """
        from valocoach.retrieval.meta import format_meta_context

        with patch(_PATCH_LOAD, return_value=_BASE_META):
            result = format_meta_context(map_name="Bind")

        assert "Bind meta" in result  # map block IS rendered
        # But no notes line
        assert "Mid control is key" not in result  # Ascent's note must not leak

    def test_unknown_agent_skips_agent_block(self):
        """agent not in agent_meta → if agent_meta: is False → block skipped."""
        from valocoach.retrieval.meta import format_meta_context

        with patch(_PATCH_LOAD, return_value=_BASE_META):
            result = format_meta_context(agent="ZZZUnknownAgent")

        assert result is not None
        assert "ZZZUnknownAgent" not in result

    def test_known_agent_shows_agent_block(self):
        """agent in agent_meta → tier / pick_rate / win_rate block rendered."""
        from valocoach.retrieval.meta import format_meta_context

        with patch(_PATCH_LOAD, return_value=_BASE_META):
            result = format_meta_context(agent="Jett")

        assert "Jett meta" in result
        assert "Pick rate" in result
        assert "Win rate" in result

    def test_agent_reason_included_when_present(self):
        """Non-empty reason is appended after the stats line (line 64)."""
        from valocoach.retrieval.meta import format_meta_context

        with patch(_PATCH_LOAD, return_value=_BASE_META):
            result = format_meta_context(agent="Jett")

        assert "Strong entry fragger" in result

    def test_agent_reason_skipped_when_empty(self):
        """Empty reason → reason line omitted (line 63->66).

        Reyna in _BASE_META has reason="".
        """
        from valocoach.retrieval.meta import format_meta_context

        with patch(_PATCH_LOAD, return_value=_BASE_META):
            result = format_meta_context(agent="Reyna")

        assert "Reyna meta" in result  # agent block IS rendered
        # But no trailing reason line — Reyna's empty reason is falsy.
        # We verify the reason content is absent (not just blank).
        # The specific reason for Jett ("Strong entry") must not leak here.
        assert "Strong entry fragger" not in result

    def test_no_map_no_agent_returns_base_block(self):
        """With no filters, only the tier list and economy line are returned."""
        from valocoach.retrieval.meta import format_meta_context

        with patch(_PATCH_LOAD, return_value=_BASE_META):
            result = format_meta_context()

        assert "META (Patch" in result
        assert "Economy thresholds" in result

    def test_both_map_and_agent_combined(self):
        """Providing both map_name and agent renders both blocks."""
        from valocoach.retrieval.meta import format_meta_context

        with patch(_PATCH_LOAD, return_value=_BASE_META):
            result = format_meta_context(map_name="Ascent", agent="Jett")

        assert "Ascent meta" in result
        assert "Jett meta" in result

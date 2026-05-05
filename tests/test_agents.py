"""Tests for valocoach.retrieval.agents — covering remaining uncovered branches.

Gaps being covered:
  37       get_agent()           — fuzzy match return (lines 35-37)
  62->64   format_agent_context  — agent.get("playstyle") is falsy → skip line 63
  64->67   format_agent_context  — agent.get("economy_tip") is falsy → skip line 65
"""

from __future__ import annotations

from unittest.mock import patch

# ---------------------------------------------------------------------------
# Fake agent data to trigger branches that real agents don't hit
# ---------------------------------------------------------------------------

# All real agents have playstyle + economy_tip; this one has neither.
_FAKE_AGENT_MINIMAL = {
    "name": "FakeAgent",
    "role": "Controller",
    "abilities": {
        "C": {"name": "Smokes", "cost": 200, "description": "Place smoke"},
        "Q": {"name": "Stun", "cost": 0, "description": "Stun enemies"},
        "E": {"name": "Flash", "cost": 100, "charges": 1, "description": "Blind enemies"},
        "X": {"name": "Ult", "ult_points": 7, "description": "Ultimate ability"},
    },
    # playstyle and economy_tip deliberately absent
}

# Agent with playstyle but no economy_tip.
_FAKE_AGENT_NO_ECO = {
    **_FAKE_AGENT_MINIMAL,
    "playstyle": "Patient controller, waits for information.",
}

_PATCH_GET_AGENT = "valocoach.retrieval.agents.get_agent"


# ===========================================================================
# get_agent — exact, fuzzy, none
# ===========================================================================


class TestGetAgent:
    def test_exact_match_returns_agent(self):
        """Exact case-insensitive name lookup — lines 30-31."""
        from valocoach.retrieval.agents import get_agent

        result = get_agent("Jett")
        assert result is not None
        assert result["name"] == "Jett"

    def test_case_insensitive_exact_match(self):
        from valocoach.retrieval.agents import get_agent

        result = get_agent("jett")
        assert result is not None
        assert result["name"] == "Jett"

    def test_fuzzy_match_returns_closest_agent(self):
        """Slightly misspelled name fuzzy-matches (line 37)."""
        from valocoach.retrieval.agents import get_agent

        # "omenn" is close to "omen" (cutoff 0.6)
        result = get_agent("omenn")
        assert result is not None
        assert result["name"] == "Omen"

    def test_fuzzy_match_with_extra_char(self):
        """Another fuzzy variant (line 37)."""
        from valocoach.retrieval.agents import get_agent

        result = get_agent("Sagee")  # close to "sage"
        assert result is not None
        assert result["name"] == "Sage"

    def test_no_match_returns_none(self):
        """Completely unknown name → no fuzzy hit → line 38 (return None)."""
        from valocoach.retrieval.agents import get_agent

        result = get_agent("ZZZCompletelyUnknown")
        assert result is None


# ===========================================================================
# list_agent_names
# ===========================================================================


class TestListAgentNames:
    def test_returns_non_empty_list(self):
        from valocoach.retrieval.agents import list_agent_names

        names = list_agent_names()
        assert isinstance(names, list)
        assert len(names) > 0

    def test_jett_in_names(self):
        from valocoach.retrieval.agents import list_agent_names

        assert "Jett" in list_agent_names()

    def test_all_strings(self):
        from valocoach.retrieval.agents import list_agent_names

        assert all(isinstance(n, str) for n in list_agent_names())


# ===========================================================================
# format_agent_context — playstyle and economy_tip branches
# ===========================================================================


class TestFormatAgentContext:
    def test_unknown_agent_returns_none(self):
        """get_agent returns None → format_agent_context returns None (line 48)."""
        from valocoach.retrieval.agents import format_agent_context

        result = format_agent_context("ZZZUnknownAgent")
        assert result is None

    def test_known_agent_returns_string(self):
        """Known agent returns a formatted string block."""
        from valocoach.retrieval.agents import format_agent_context

        result = format_agent_context("Jett")
        assert isinstance(result, str)
        assert "Jett" in result
        assert "Abilities:" in result

    def test_playstyle_included_when_present(self):
        """Agent with playstyle → the Playstyle line appears (line 63)."""
        from valocoach.retrieval.agents import format_agent_context

        result = format_agent_context("Jett")
        assert result is not None
        assert "Playstyle:" in result

    def test_economy_tip_included_when_present(self):
        """Agent with economy_tip → the Economy note line appears (line 65)."""
        from valocoach.retrieval.agents import format_agent_context

        result = format_agent_context("Jett")
        assert result is not None
        assert "Economy note:" in result

    def test_playstyle_skipped_when_absent(self):
        """Agent without playstyle → the Playstyle line is omitted (line 62->64)."""
        from valocoach.retrieval.agents import format_agent_context

        with patch(_PATCH_GET_AGENT, return_value=_FAKE_AGENT_MINIMAL):
            result = format_agent_context("FakeAgent")

        assert result is not None
        assert "Playstyle:" not in result
        assert "Economy note:" not in result

    def test_economy_tip_skipped_when_absent(self):
        """Agent with playstyle but no economy_tip → Economy note omitted (line 64->67)."""
        from valocoach.retrieval.agents import format_agent_context

        with patch(_PATCH_GET_AGENT, return_value=_FAKE_AGENT_NO_ECO):
            result = format_agent_context("FakeAgent")

        assert result is not None
        assert "Playstyle:" in result  # playstyle IS present
        assert "Economy note:" not in result  # economy_tip is absent

    def test_ult_points_shown_for_ultimate(self):
        """Ability with ult_points uses 'ult pts' cost string (line 53)."""
        from valocoach.retrieval.agents import format_agent_context

        result = format_agent_context("Jett")
        assert result is not None
        assert "ult pts" in result

    def test_free_ability_shown_as_free(self):
        """Ability with cost=0 and no ult_points → 'free' (line 55)."""
        from valocoach.retrieval.agents import format_agent_context

        # Jett's E (Tailwind) has cost=0 and no ult_points
        result = format_agent_context("Jett")
        assert result is not None
        assert "free" in result

    def test_charges_shown_when_present(self):
        """Ability with charges → 'X charge(s)' appears in output (line 59)."""
        from valocoach.retrieval.agents import format_agent_context

        result = format_agent_context("Jett")
        assert result is not None
        assert "charge(s)" in result

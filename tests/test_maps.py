"""Tests for valocoach.retrieval.maps — covering the remaining uncovered branches.

Gaps being covered:
  27-29  get_map()            — fuzzy match path (get_close_matches finds a hit)
  28-30  get_map()            — no match at all → return None
  44     format_map_context() — get_map returns None → return None
  52→54  format_map_context() — map_data has no 'rotations' key → skips that line
  54→57  format_map_context() — map_data has no 'notes' key → skips that line
"""

from __future__ import annotations

from unittest.mock import patch

# ---------------------------------------------------------------------------
# Minimal fake map used to test branches that need missing keys.
# (All real maps in maps.json have both 'rotations' and 'notes'.)
# ---------------------------------------------------------------------------

_FAKE_MAP_FULL = {
    "name": "FakeMap",
    "sites": ["A", "B"],
    "callouts": {"a_site": ["Orb", "Ramp"], "b_site": ["Default"]},
    "rotations": "Fast B-to-A rotation via Mid",
    "notes": "Always contest mid early",
}

_FAKE_MAP_NO_ROTATIONS = {
    "name": "FakeMap",
    "sites": ["A", "B"],
    "callouts": {"a_site": ["Orb", "Ramp"], "b_site": ["Default"]},
    # 'rotations' key deliberately absent
    "notes": "Watch flanks from C",
}

_FAKE_MAP_NO_NOTES = {
    "name": "FakeMap",
    "sites": ["A", "B"],
    "callouts": {"a_site": ["Orb", "Ramp"], "b_site": ["Default"]},
    "rotations": "Rotate through Mid",
    # 'notes' key deliberately absent
}

_FAKE_MAP_NEITHER = {
    "name": "FakeMap",
    "sites": ["A", "B"],
    "callouts": {"a_site": ["Orb"]},
    # neither 'rotations' nor 'notes'
}

_PATCH_GET_MAP = "valocoach.retrieval.maps.get_map"


# ===========================================================================
# get_map() — exact, fuzzy, and no-match paths
# ===========================================================================


class TestGetMap:
    def test_exact_match_returns_correct_map(self):
        """Exact (case-insensitive) name lookup hits line 24-25."""
        from valocoach.retrieval.maps import get_map

        result = get_map("Ascent")
        assert result is not None
        assert result["name"] == "Ascent"

    def test_case_insensitive_exact_match(self):
        from valocoach.retrieval.maps import get_map

        result = get_map("ASCENT")
        assert result is not None
        assert result["name"] == "Ascent"

    def test_fuzzy_match_returns_closest_map(self):
        """A slightly misspelled name should fuzzy-match (lines 27-29)."""
        from valocoach.retrieval.maps import get_map

        # "ascen" is close enough to "ascent" (cutoff 0.6) to fuzzy-match.
        result = get_map("ascen")
        assert result is not None
        assert result["name"] == "Ascent"

    def test_fuzzy_match_with_extra_char(self):
        """Trailing extra char still fuzzy-matches (lines 27-29)."""
        from valocoach.retrieval.maps import get_map

        result = get_map("Haaven")  # close to "Haven"
        assert result is not None
        assert result["name"] == "Haven"

    def test_no_match_returns_none(self):
        """A completely unknown name produces no fuzzy hit → return None (line 30)."""
        from valocoach.retrieval.maps import get_map

        result = get_map("ZZZUnknownMapXXX")
        assert result is None

    def test_very_short_nonsense_returns_none(self):
        """Very short/distant string doesn't clear the 0.6 cutoff → None (line 30)."""
        from valocoach.retrieval.maps import get_map

        result = get_map("zzz")
        assert result is None


# ===========================================================================
# format_map_context() — None-return and missing-field branches
# ===========================================================================


class TestFormatMapContext:
    def test_unknown_name_returns_none(self):
        """get_map returns None for unknown name → format_map_context returns None (line 44)."""
        from valocoach.retrieval.maps import format_map_context

        result = format_map_context("ZZZTotallyUnknownMap")
        assert result is None

    def test_known_map_returns_string(self):
        """Smoke-test: a known map produces a non-empty string."""
        from valocoach.retrieval.maps import format_map_context

        result = format_map_context("Ascent")
        assert isinstance(result, str)
        assert "Ascent" in result

    def test_rotations_present_in_output_when_set(self):
        """When map_data has 'rotations', the Rotations line appears."""
        from valocoach.retrieval.maps import format_map_context

        with patch(_PATCH_GET_MAP, return_value=_FAKE_MAP_FULL):
            result = format_map_context("FakeMap")

        assert result is not None
        assert "Rotations:" in result
        assert "Fast B-to-A rotation" in result

    def test_notes_present_in_output_when_set(self):
        """When map_data has 'notes', the Map notes line appears."""
        from valocoach.retrieval.maps import format_map_context

        with patch(_PATCH_GET_MAP, return_value=_FAKE_MAP_FULL):
            result = format_map_context("FakeMap")

        assert result is not None
        assert "Map notes:" in result
        assert "Always contest mid" in result

    def test_rotations_skipped_when_key_absent(self):
        """map_data without 'rotations' → the Rotations line is omitted (line 52→54)."""
        from valocoach.retrieval.maps import format_map_context

        with patch(_PATCH_GET_MAP, return_value=_FAKE_MAP_NO_ROTATIONS):
            result = format_map_context("FakeMap")

        assert result is not None
        assert "Rotations:" not in result
        # Notes should still appear since that key is present
        assert "Map notes:" in result

    def test_notes_skipped_when_key_absent(self):
        """map_data without 'notes' → the Map notes line is omitted (line 54→57)."""
        from valocoach.retrieval.maps import format_map_context

        with patch(_PATCH_GET_MAP, return_value=_FAKE_MAP_NO_NOTES):
            result = format_map_context("FakeMap")

        assert result is not None
        assert "Map notes:" not in result
        # Rotations should still appear since that key is present
        assert "Rotations:" in result

    def test_neither_rotations_nor_notes_skipped(self):
        """map_data with neither key → both lines omitted (covers both 52→54 and 54→57)."""
        from valocoach.retrieval.maps import format_map_context

        with patch(_PATCH_GET_MAP, return_value=_FAKE_MAP_NEITHER):
            result = format_map_context("FakeMap")

        assert result is not None
        assert "Rotations:" not in result
        assert "Map notes:" not in result

    def test_callouts_always_included(self):
        """Callout block is always rendered regardless of rotations/notes."""
        from valocoach.retrieval.maps import format_map_context

        with patch(_PATCH_GET_MAP, return_value=_FAKE_MAP_NEITHER):
            result = format_map_context("FakeMap")

        assert result is not None
        assert "Callouts:" in result
        assert "A_SITE" in result.upper()


# ===========================================================================
# list_map_names()
# ===========================================================================


class TestListMapNames:
    def test_returns_list_of_strings(self):
        """list_map_names() returns a non-empty list of map name strings (line 34)."""
        from valocoach.retrieval.maps import list_map_names

        names = list_map_names()
        assert isinstance(names, list)
        assert len(names) > 0
        assert all(isinstance(n, str) for n in names)

    def test_ascent_in_names(self):
        from valocoach.retrieval.maps import list_map_names

        assert "Ascent" in list_map_names()

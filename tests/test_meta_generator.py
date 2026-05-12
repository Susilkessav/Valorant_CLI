"""Tests for valocoach.retrieval.meta_generator.

Covers:
  - _strip_fences: removes markdown code fences
  - _find_json_object: extracts first {…} block
  - _validate: fills gaps with existing meta
  - generate_meta_update: happy path, LLM failure, JSON parse error
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

_STREAM_COMPLETION = "valocoach.llm.provider.stream_completion"


# ---------------------------------------------------------------------------
# _strip_fences
# ---------------------------------------------------------------------------


class TestStripFences:
    def test_strips_json_fence(self):
        from valocoach.retrieval.meta_generator import _strip_fences

        text = '```json\n{"key": "value"}\n```'
        assert _strip_fences(text) == '{"key": "value"}'

    def test_strips_plain_fence(self):
        from valocoach.retrieval.meta_generator import _strip_fences

        text = '```\n{"key": "value"}\n```'
        assert _strip_fences(text) == '{"key": "value"}'

    def test_no_fence_unchanged(self):
        from valocoach.retrieval.meta_generator import _strip_fences

        text = '{"key": "value"}'
        assert _strip_fences(text) == text

    def test_strips_whitespace(self):
        from valocoach.retrieval.meta_generator import _strip_fences

        text = '  \n{"key": "value"}\n  '
        assert _strip_fences(text) == '{"key": "value"}'


# ---------------------------------------------------------------------------
# _find_json_object
# ---------------------------------------------------------------------------


class TestFindJsonObject:
    def test_plain_object(self):
        from valocoach.retrieval.meta_generator import _find_json_object

        text = '{"tier_list": {"S": ["Jett"]}}'
        assert _find_json_object(text) == text

    def test_strips_preamble(self):
        from valocoach.retrieval.meta_generator import _find_json_object

        text = 'Here is the JSON: {"tier_list": {}}'
        assert _find_json_object(text) == '{"tier_list": {}}'

    def test_nested_objects_handled(self):
        from valocoach.retrieval.meta_generator import _find_json_object

        text = '{"a": {"b": {"c": 1}}}'
        assert _find_json_object(text) == text

    def test_no_brace_returns_original(self):
        from valocoach.retrieval.meta_generator import _find_json_object

        text = "no json here"
        assert _find_json_object(text) == text

    def test_unclosed_brace_returns_from_start(self):
        from valocoach.retrieval.meta_generator import _find_json_object

        text = '{"unclosed": 1'
        result = _find_json_object(text)
        assert result.startswith("{")


# ---------------------------------------------------------------------------
# _validate
# ---------------------------------------------------------------------------

_EXISTING = {
    "tier_list": {"S": ["Jett"], "A": ["Sage"], "B": ["Brimstone"], "C": ["Yoru"]},
    "agent_meta": {"Jett": {"tier": "S", "reason": "Fast"}},
    "map_meta": {"Ascent": {"top_agents": ["Jett"], "notes": "Mid control"}},
}


class TestValidate:
    def test_uses_llm_data_when_present(self):
        from valocoach.retrieval.meta_generator import _validate

        data = {
            "tier_list": {"S": ["Reyna"], "A": [], "B": [], "C": []},
            "agent_meta": {"Reyna": {"tier": "S"}},
            "map_meta": {},
        }
        result = _validate(data, _EXISTING)
        assert result["tier_list"]["S"] == ["Reyna"]

    def test_falls_back_to_existing_when_llm_data_missing(self):
        from valocoach.retrieval.meta_generator import _validate

        result = _validate({}, _EXISTING)
        assert result["tier_list"] == _EXISTING["tier_list"]
        assert result["agent_meta"] == _EXISTING["agent_meta"]

    def test_fills_missing_tiers_from_existing(self):
        from valocoach.retrieval.meta_generator import _validate

        # Only S tier provided by LLM
        data = {"tier_list": {"S": ["Jett"]}, "agent_meta": {}, "map_meta": {}}
        result = _validate(data, _EXISTING)
        assert "A" in result["tier_list"]
        assert "B" in result["tier_list"]
        assert "C" in result["tier_list"]

    def test_empty_tiers_in_empty_existing(self):
        from valocoach.retrieval.meta_generator import _validate

        result = _validate({}, {})
        # All four tiers must exist even with nothing to fall back to
        assert set(result["tier_list"].keys()) >= {"S", "A", "B", "C"}


# ---------------------------------------------------------------------------
# generate_meta_update
# ---------------------------------------------------------------------------

_VALID_META_JSON = {
    "tier_list": {"S": ["Jett"], "A": ["Sage"], "B": ["Brimstone"], "C": ["Yoru"]},
    "agent_meta": {
        "Jett": {"tier": "S", "pick_rate": "~30%", "win_rate": "~52%", "reason": "Fast lurk"}
    },
    "map_meta": {"Ascent": {"top_agents": ["Jett", "Sage"], "notes": "Mid control matters."}},
}


def _make_settings():
    s = MagicMock()
    s.ollama_model = "llama3"
    return s


class TestGenerateMetaUpdate:
    def _stream(self, text: str):
        """Return an iterator that yields characters of *text*."""
        return iter(text)

    def test_happy_path_returns_validated_dict(self):
        from valocoach.retrieval.meta_generator import generate_meta_update

        llm_output = json.dumps(_VALID_META_JSON)
        with patch(_STREAM_COMPLETION, return_value=self._stream(llm_output)):
            result = generate_meta_update(
                _make_settings(),
                patch_version="10.09",
                patch_notes_text="some patch notes",
                stats_text="some stats",
                existing_meta=_EXISTING,
            )

        assert result is not None
        assert "tier_list" in result
        assert "S" in result["tier_list"]

    def test_llm_failure_returns_none(self):
        from valocoach.retrieval.meta_generator import generate_meta_update

        with patch(_STREAM_COMPLETION, side_effect=RuntimeError("Ollama down")):
            result = generate_meta_update(
                _make_settings(),
                patch_version="10.09",
                patch_notes_text="notes",
                stats_text="stats",
                existing_meta={},
            )

        assert result is None

    def test_invalid_json_returns_none(self):
        from valocoach.retrieval.meta_generator import generate_meta_update

        with patch(_STREAM_COMPLETION, return_value=self._stream("not valid JSON at all")):
            result = generate_meta_update(
                _make_settings(),
                patch_version="10.09",
                patch_notes_text="notes",
                stats_text="stats",
                existing_meta={},
            )

        assert result is None

    def test_strips_markdown_fences_from_llm_output(self):
        from valocoach.retrieval.meta_generator import generate_meta_update

        wrapped = "```json\n" + json.dumps(_VALID_META_JSON) + "\n```"
        with patch(_STREAM_COMPLETION, return_value=self._stream(wrapped)):
            result = generate_meta_update(
                _make_settings(),
                patch_version="10.09",
                patch_notes_text="notes",
                stats_text="stats",
                existing_meta=_EXISTING,
            )

        assert result is not None

    def test_empty_inputs_use_fallback_strings(self):
        """Empty patch_notes/stats should still attempt LLM call (no crash)."""
        from valocoach.retrieval.meta_generator import generate_meta_update

        llm_output = json.dumps(_VALID_META_JSON)
        with patch(_STREAM_COMPLETION, return_value=self._stream(llm_output)):
            result = generate_meta_update(
                _make_settings(),
                patch_version="10.09",
                patch_notes_text="",
                stats_text="",
                existing_meta={},
            )

        assert result is not None

    def test_handles_preamble_before_json(self):
        """Model outputs prose then JSON — _find_json_object extracts the object."""
        from valocoach.retrieval.meta_generator import generate_meta_update

        preamble = "Sure! Here is the updated tier list: "
        llm_output = preamble + json.dumps(_VALID_META_JSON)
        with patch(_STREAM_COMPLETION, return_value=self._stream(llm_output)):
            result = generate_meta_update(
                _make_settings(),
                patch_version="10.09",
                patch_notes_text="notes",
                stats_text="stats",
                existing_meta={},
            )

        assert result is not None

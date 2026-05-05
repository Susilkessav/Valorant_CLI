"""Tests for valocoach.core.preflight — pre-flight sanity checks.

All network I/O and ChromaDB access is patched so these run offline and
without a real Ollama instance or vector store.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from valocoach.core.config import Settings
from valocoach.core.preflight import (
    CheckResult,
    _is_ollama_model,
    check_ollama,
    check_riot_id,
    check_vector_store,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _settings(**kwargs) -> Settings:
    """Build a Settings with env-file loading disabled."""
    s = Settings(_env_file=None)
    for k, v in kwargs.items():
        object.__setattr__(s, k, v)
    return s


def _fake_tags_response(model_names: list[str]) -> MagicMock:
    """Build a mock httpx response for /api/tags."""
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {"models": [{"name": n} for n in model_names]}
    return resp


# ---------------------------------------------------------------------------
# check_ollama
# ---------------------------------------------------------------------------


class TestCheckOllama:
    def test_returns_ok_when_model_available(self):
        settings = _settings(ollama_host="http://localhost:11434", ollama_model="qwen3:8b")
        resp = _fake_tags_response(["qwen3:8b"])
        with patch("valocoach.core.preflight.httpx.get", return_value=resp):
            result = check_ollama(settings)
        assert result.ok
        assert isinstance(result, CheckResult)

    def test_returns_ok_when_model_has_tag_suffix(self):
        """'qwen3:8b:latest' in Ollama should match configured 'qwen3:8b'."""
        settings = _settings(ollama_host="http://localhost:11434", ollama_model="qwen3:8b")
        resp = _fake_tags_response(["qwen3:8b:latest"])
        with patch("valocoach.core.preflight.httpx.get", return_value=resp):
            result = check_ollama(settings)
        assert result.ok

    def test_fails_when_connection_refused(self):
        settings = _settings(ollama_host="http://localhost:11434", ollama_model="qwen3:8b")
        with patch("valocoach.core.preflight.httpx.get", side_effect=ConnectionError("refused")):
            result = check_ollama(settings)
        assert not result.ok
        assert "not reachable" in result.message
        assert "ollama serve" in result.hint

    def test_fails_when_no_models_pulled(self):
        settings = _settings(ollama_host="http://localhost:11434", ollama_model="qwen3:8b")
        resp = _fake_tags_response([])
        with patch("valocoach.core.preflight.httpx.get", return_value=resp):
            result = check_ollama(settings)
        assert not result.ok
        assert "No models" in result.message
        assert "ollama pull qwen3:8b" in result.hint

    def test_fails_when_configured_model_not_pulled(self):
        settings = _settings(ollama_host="http://localhost:11434", ollama_model="qwen3:14b")
        resp = _fake_tags_response(["qwen3:8b"])
        with patch("valocoach.core.preflight.httpx.get", return_value=resp):
            result = check_ollama(settings)
        assert not result.ok
        assert "qwen3:14b" in result.message
        assert "ollama pull qwen3:14b" in result.hint

    def test_hint_not_empty_on_failure(self):
        settings = _settings(ollama_host="http://localhost:11434", ollama_model="qwen3:8b")
        with patch("valocoach.core.preflight.httpx.get", side_effect=TimeoutError()):
            result = check_ollama(settings)
        assert result.hint != ""

    def test_uses_configured_host(self):
        """The check should probe the host from settings, not a hardcoded address."""
        settings = _settings(ollama_host="http://192.168.1.50:11434", ollama_model="qwen3:8b")
        with patch("valocoach.core.preflight.httpx.get", side_effect=ConnectionError()) as mock_get:
            check_ollama(settings)
        assert "192.168.1.50" in mock_get.call_args.args[0]

    def test_ok_result_has_empty_hint(self):
        settings = _settings(ollama_host="http://localhost:11434", ollama_model="qwen3:8b")
        resp = _fake_tags_response(["qwen3:8b"])
        with patch("valocoach.core.preflight.httpx.get", return_value=resp):
            result = check_ollama(settings)
        # hint is optional for ok results
        assert result.ok
        assert result.hint == ""

    # ------------------------------------------------------------------
    # NOTE: httpx is imported at module level in preflight.py so the patch
    # target is "valocoach.core.preflight.httpx.get" (attribute on the
    # already-bound module reference).  The tests above use this correctly.

    # ------------------------------------------------------------------
    # Provider-aware routing — cloud providers must skip the Ollama probe
    # ------------------------------------------------------------------

    def test_skips_probe_for_anthropic_prefix(self):
        """anthropic/ model must return ok without touching httpx."""
        settings = _settings(ollama_model="anthropic/claude-3-5-sonnet-20241022")
        with patch("valocoach.core.preflight.httpx.get", side_effect=AssertionError("should not probe")) as mock_get:
            result = check_ollama(settings)
        mock_get.assert_not_called()
        assert result.ok

    def test_skips_probe_for_openai_prefix(self):
        """openai/ model must return ok without touching httpx."""
        settings = _settings(ollama_model="openai/gpt-4o")
        with patch("valocoach.core.preflight.httpx.get", side_effect=AssertionError("should not probe")) as mock_get:
            result = check_ollama(settings)
        mock_get.assert_not_called()
        assert result.ok

    def test_explicit_ollama_prefix_still_probes(self):
        """ollama/ prefix should still go through the Ollama connectivity check."""
        settings = _settings(ollama_host="http://localhost:11434", ollama_model="ollama/qwen3:8b")
        resp = _fake_tags_response(["qwen3:8b"])
        with patch("valocoach.core.preflight.httpx.get", return_value=resp):
            result = check_ollama(settings)
        assert result.ok

    def test_explicit_ollama_prefix_strips_for_comparison(self):
        """ollama/model prefix is stripped before comparing against pulled model names."""
        settings = _settings(ollama_host="http://localhost:11434", ollama_model="ollama/qwen3:14b")
        resp = _fake_tags_response(["qwen3:14b"])
        with patch("valocoach.core.preflight.httpx.get", return_value=resp):
            result = check_ollama(settings)
        assert result.ok

    def test_explicit_ollama_prefix_model_not_pulled(self):
        """ollama/model that is not pulled still returns a failing result."""
        settings = _settings(ollama_host="http://localhost:11434", ollama_model="ollama/qwen3:72b")
        resp = _fake_tags_response(["qwen3:8b"])
        with patch("valocoach.core.preflight.httpx.get", return_value=resp):
            result = check_ollama(settings)
        assert not result.ok
        assert "qwen3:72b" in result.message  # bare name, not prefixed
        assert "ollama pull qwen3:72b" in result.hint


# ---------------------------------------------------------------------------
# _is_ollama_model
# ---------------------------------------------------------------------------


class TestIsOllamaModel:
    def test_bare_name_is_ollama(self):
        assert _is_ollama_model("qwen3:8b") is True

    def test_ollama_prefix_is_ollama(self):
        assert _is_ollama_model("ollama/qwen3:8b") is True

    def test_anthropic_prefix_is_not_ollama(self):
        assert _is_ollama_model("anthropic/claude-3-5-sonnet-20241022") is False

    def test_openai_prefix_is_not_ollama(self):
        assert _is_ollama_model("openai/gpt-4o") is False

    def test_case_insensitive_prefix(self):
        assert _is_ollama_model("Ollama/qwen3:8b") is True

    def test_unknown_provider_is_not_ollama(self):
        assert _is_ollama_model("groq/llama3-8b") is False


# ---------------------------------------------------------------------------
# check_riot_id
# ---------------------------------------------------------------------------


class TestCheckRiotId:
    def test_ok_when_both_set(self):
        settings = _settings(riot_name="Player", riot_tag="NA1")
        assert check_riot_id(settings).ok

    def test_fails_when_name_empty(self):
        settings = _settings(riot_name="", riot_tag="NA1")
        result = check_riot_id(settings)
        assert not result.ok
        assert "riot_name" in result.message

    def test_fails_when_tag_empty(self):
        settings = _settings(riot_name="Player", riot_tag="")
        result = check_riot_id(settings)
        assert not result.ok
        assert "riot_tag" in result.message

    def test_fails_when_both_empty(self):
        settings = _settings(riot_name="", riot_tag="")
        result = check_riot_id(settings)
        assert not result.ok

    def test_hint_mentions_config_file(self):
        settings = _settings(riot_name="", riot_tag="")
        result = check_riot_id(settings)
        assert "config" in result.hint.lower()

    def test_result_is_check_result_namedtuple(self):
        settings = _settings(riot_name="X", riot_tag="Y")
        result = check_riot_id(settings)
        # NamedTuple — indexable
        assert result[0] is True  # ok
        assert isinstance(result[1], str)  # message


# ---------------------------------------------------------------------------
# check_vector_store
# ---------------------------------------------------------------------------


class TestCheckVectorStore:
    # get_collection is imported lazily inside check_vector_store() so it is
    # not bound in the preflight module namespace.  Patch at the source module
    # instead — that is where the name resolves at call time.
    _PATCH_TARGET = "valocoach.retrieval.vector_store.get_collection"

    def _patch_collection(self, count: int):
        """Return a context manager that patches get_collection with a stub."""
        coll = MagicMock()
        coll.count.return_value = count
        return patch(self._PATCH_TARGET, return_value=coll)

    def test_ok_when_store_has_documents(self):
        settings = _settings()
        with self._patch_collection(100):
            result = check_vector_store(settings)
        assert result.ok

    def test_fails_when_store_is_empty(self):
        settings = _settings()
        with self._patch_collection(0):
            result = check_vector_store(settings)
        assert not result.ok
        assert "empty" in result.message.lower()
        assert "ingest --seed" in result.hint

    def test_fails_when_import_raises(self):
        """ChromaDB not installed or data_dir inaccessible — still non-raising."""
        settings = _settings()
        with patch(self._PATCH_TARGET, side_effect=RuntimeError("chromadb not found")):
            result = check_vector_store(settings)
        assert not result.ok
        assert "ingest --seed" in result.hint

    def test_hint_points_to_ingest_seed(self):
        settings = _settings()
        with self._patch_collection(0):
            result = check_vector_store(settings)
        assert "valocoach ingest --seed" in result.hint

    def test_single_document_is_ok(self):
        settings = _settings()
        with self._patch_collection(1):
            result = check_vector_store(settings)
        assert result.ok

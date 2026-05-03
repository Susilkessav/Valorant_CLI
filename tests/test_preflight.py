"""Tests for valocoach.core.preflight — pre-flight sanity checks.

All network I/O and ChromaDB access is patched so these run offline and
without a real Ollama instance or vector store.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from valocoach.core.config import Settings
from valocoach.core.preflight import CheckResult, check_ollama, check_riot_id, check_vector_store


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

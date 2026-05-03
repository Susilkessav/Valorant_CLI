"""Tests for valocoach.retrieval.embedder.

Covers:
  - embed([]) returns [] without calling Ollama
  - embed(texts) calls ollama.embed with correct model and input
  - embed returns the embeddings list from the Ollama response
  - embed_one returns the first (and only) embedding vector
  - is_available returns True when Ollama responds
  - is_available returns False when Ollama raises any exception

Patch target: valocoach.retrieval.embedder.ollama  (module-level import)
"""

from __future__ import annotations

from unittest.mock import patch

_OLLAMA = "valocoach.retrieval.embedder.ollama"


class TestEmbed:
    def test_empty_input_returns_empty_list_without_ollama_call(self):
        from valocoach.retrieval.embedder import embed

        with patch(_OLLAMA) as mock_ollama:
            result = embed([])

        assert result == []
        mock_ollama.embed.assert_not_called()

    def test_single_text_returns_single_embedding(self):
        from valocoach.retrieval.embedder import embed

        fake_vec = [0.1, 0.2, 0.3]
        with patch(_OLLAMA + ".embed", return_value={"embeddings": [fake_vec]}):
            result = embed(["hello valorant"])

        assert result == [fake_vec]

    def test_multiple_texts_return_multiple_embeddings(self):
        from valocoach.retrieval.embedder import embed

        vecs = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
        with patch(_OLLAMA + ".embed", return_value={"embeddings": vecs}):
            result = embed(["a", "b", "c"])

        assert result == vecs

    def test_uses_correct_model_name(self):
        from valocoach.retrieval.embedder import EMBED_MODEL, embed

        with patch(_OLLAMA + ".embed", return_value={"embeddings": [[0.0]]}) as mock:
            embed(["test"])

        _, kwargs = mock.call_args
        assert kwargs["model"] == EMBED_MODEL

    def test_passes_texts_as_input_kwarg(self):
        from valocoach.retrieval.embedder import embed

        with patch(_OLLAMA + ".embed", return_value={"embeddings": [[0.0]]}) as mock:
            embed(["alpha", "beta"])

        _, kwargs = mock.call_args
        assert kwargs["input"] == ["alpha", "beta"]


class TestEmbedOne:
    def test_returns_first_embedding_vector(self):
        from valocoach.retrieval.embedder import embed_one

        fake_vec = [0.7, 0.8, 0.9]
        with patch(_OLLAMA + ".embed", return_value={"embeddings": [fake_vec]}):
            result = embed_one("single query")

        assert result == fake_vec

    def test_calls_embed_with_single_element_list(self):
        from valocoach.retrieval.embedder import embed_one

        with patch(_OLLAMA + ".embed", return_value={"embeddings": [[0.0]]}) as mock:
            embed_one("query")

        _, kwargs = mock.call_args
        assert kwargs["input"] == ["query"]


class TestIsAvailable:
    def test_returns_true_when_ollama_responds(self):
        from valocoach.retrieval.embedder import is_available

        with patch(_OLLAMA + ".embed", return_value={"embeddings": [[0.0]]}):
            assert is_available() is True

    def test_returns_false_when_ollama_raises_connection_error(self):
        from valocoach.retrieval.embedder import is_available

        with patch(_OLLAMA + ".embed", side_effect=ConnectionError("Ollama not running")):
            assert is_available() is False

    def test_returns_false_when_ollama_raises_generic_exception(self):
        from valocoach.retrieval.embedder import is_available

        with patch(_OLLAMA + ".embed", side_effect=Exception("unexpected")):
            assert is_available() is False

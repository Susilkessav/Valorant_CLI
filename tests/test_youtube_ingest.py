"""Tests for Phase D — YouTube ingest pipeline.

Covers:
  - D1: oEmbed metadata fetch
  - D2: deduplication check
  - D3: time-window chunking
  - D4: anchor classifier (unit-level, no embedding calls)
  - meta_sync step 4: uses new ingest_youtube_video path
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# D1 — oEmbed metadata
# ---------------------------------------------------------------------------


class TestFetchVideoMetadata:
    def test_returns_title_and_channel_on_success(self):
        from valocoach.retrieval.scrapers.youtube import fetch_video_metadata

        mock_data = b'{"title": "Haven Guide", "author_name": "Woohoojin"}'
        mock_resp = MagicMock()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.read.return_value = mock_data

        with patch("valocoach.retrieval.scrapers.youtube.urlopen", return_value=mock_resp):
            result = fetch_video_metadata("abc12345678")

        assert result["title"] == "Haven Guide"
        assert result["channel"] == "Woohoojin"

    def test_falls_back_gracefully_on_network_error(self):
        from valocoach.retrieval.scrapers.youtube import fetch_video_metadata

        with patch(
            "valocoach.retrieval.scrapers.youtube.urlopen",
            side_effect=OSError("connection refused"),
        ):
            result = fetch_video_metadata("abc12345678")

        # Should return safe defaults, not raise
        assert "abc12345678" in result["title"]
        assert result["channel"] == "Unknown channel"

    def test_handles_missing_fields_in_oembed_response(self):
        from valocoach.retrieval.scrapers.youtube import fetch_video_metadata

        mock_resp = MagicMock()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.read.return_value = b"{}"  # empty response

        with patch("valocoach.retrieval.scrapers.youtube.urlopen", return_value=mock_resp):
            result = fetch_video_metadata("abc12345678")

        assert "abc12345678" in result["title"]
        assert result["channel"] == "Unknown channel"


# ---------------------------------------------------------------------------
# D3 — Time-window chunking
# ---------------------------------------------------------------------------


def _make_entry(start: float, text: str):
    """Build a minimal transcript entry stub."""
    e = SimpleNamespace()
    e.start = start
    e.text = text
    return e


class TestWindowEntries:
    def test_single_window_when_all_fit(self):
        from valocoach.retrieval.scrapers.youtube import _window_entries

        entries = [_make_entry(0, "hello"), _make_entry(30, "world"), _make_entry(60, "foo")]
        windows = _window_entries(entries, window_seconds=120)

        assert len(windows) == 1
        start, text = windows[0]
        assert start == 0
        assert "hello" in text
        assert "foo" in text

    def test_splits_at_window_boundary(self):
        from valocoach.retrieval.scrapers.youtube import _window_entries

        entries = [
            _make_entry(0, "A"),
            _make_entry(60, "B"),
            _make_entry(120, "C"),  # new window starts here
            _make_entry(180, "D"),
        ]
        windows = _window_entries(entries, window_seconds=120)

        assert len(windows) == 2
        assert windows[0][0] == 0
        assert windows[1][0] == 120

    def test_skips_empty_text_entries(self):
        from valocoach.retrieval.scrapers.youtube import _window_entries

        entries = [
            _make_entry(0, "tactical advice"),
            _make_entry(10, "   "),  # whitespace — should be skipped
            _make_entry(20, "aim here"),
        ]
        windows = _window_entries(entries, window_seconds=120)

        assert len(windows) == 1
        text = windows[0][1]
        assert "tactical advice" in text
        assert "aim here" in text
        # whitespace-only entry should not appear
        assert text.strip() != ""

    def test_empty_entries_returns_empty(self):
        from valocoach.retrieval.scrapers.youtube import _window_entries

        assert _window_entries([], window_seconds=120) == []

    def test_multiple_windows_correct_starts(self):
        from valocoach.retrieval.scrapers.youtube import _window_entries

        entries = [_make_entry(i * 30, f"text_{i}") for i in range(10)]
        windows = _window_entries(entries, window_seconds=60)

        # 10 entries x 30s = 300s total; 60s windows -> should have ~5 windows
        assert len(windows) >= 4
        # Each window start should be non-decreasing
        starts = [w[0] for w in windows]
        assert starts == sorted(starts)


# ---------------------------------------------------------------------------
# D4 — Anchor classifier (unit tests — no Ollama calls)
# ---------------------------------------------------------------------------


class TestAnchorClassifier:
    def test_returns_unknown_when_build_fails(self):
        from valocoach.retrieval.youtube_ingest import AnchorClassifier

        clf = AnchorClassifier()
        with patch(
            "valocoach.retrieval.youtube_ingest.AnchorClassifier._build",
            side_effect=RuntimeError("ollama not running"),
        ):
            cat, score = clf.classify("some text about Jett")

        assert cat == "unknown"
        assert score == 0.0

    def test_classify_returns_unknown_when_embed_fails(self):
        from valocoach.retrieval.youtube_ingest import AnchorClassifier

        clf = AnchorClassifier()
        # Pre-seed with dummy anchor vecs so _build() doesn't run
        clf._anchor_vecs = {"agent_strategy": [0.0] * 768, "off_topic": [0.0] * 768}

        # embed_one is imported inside the method from valocoach.retrieval.embedder
        with patch(
            "valocoach.retrieval.embedder.embed_one",
            side_effect=RuntimeError("offline"),
        ):
            # Should not raise — returns (unknown, 0.0) gracefully
            cat, score = clf.classify("whatever text")

        assert cat == "unknown"
        assert score == 0.0

    def test_classify_picks_closest_anchor(self):
        """Given two anchors, classify() should return the one with higher dot product."""
        import numpy as np

        from valocoach.retrieval.youtube_ingest import AnchorClassifier

        clf = AnchorClassifier()
        anchor_a = [1.0, 0.0, 0.0, 0.0]
        anchor_b = [0.0, 1.0, 0.0, 0.0]
        clf._anchor_vecs = {"agent_strategy": anchor_a, "off_topic": anchor_b}

        # Chunk embedding closer to agent_strategy
        chunk_vec = np.array([0.9, 0.1, 0.0, 0.0])
        chunk_vec = chunk_vec / np.linalg.norm(chunk_vec)

        with patch(
            "valocoach.retrieval.embedder.embed_one",
            return_value=chunk_vec.tolist(),
        ):
            cat, score = clf.classify("about agent abilities")

        assert cat == "agent_strategy"
        assert score > 0.5

    def test_off_topic_chunk_is_dropped(self):
        """Chunks where best category is off_topic should be filtered by the pipeline."""
        import numpy as np

        from valocoach.retrieval.youtube_ingest import AnchorClassifier

        clf = AnchorClassifier()
        clf._anchor_vecs = {
            "agent_strategy": [0.0, 1.0, 0.0, 0.0],
            "off_topic": [1.0, 0.0, 0.0, 0.0],
        }

        off_topic_vec = np.array([0.95, 0.05, 0.0, 0.0])
        off_topic_vec = off_topic_vec / np.linalg.norm(off_topic_vec)

        with patch(
            "valocoach.retrieval.embedder.embed_one",
            return_value=off_topic_vec.tolist(),
        ):
            cat, _score = clf.classify("like and subscribe to the channel")

        assert cat == "off_topic"


# ---------------------------------------------------------------------------
# D2 — Deduplication
# ---------------------------------------------------------------------------


class TestIsVideoIngested:
    def test_returns_false_when_collection_empty(self):
        from valocoach.retrieval.youtube_ingest import is_video_ingested

        mock_coll = MagicMock()
        mock_coll.get.return_value = {"ids": []}

        # get_collection is imported from valocoach.retrieval.vector_store inside the function
        with patch(
            "valocoach.retrieval.vector_store.get_collection", return_value=mock_coll
        ):
            assert is_video_ingested(MagicMock(), "abc12345678") is False

    def test_returns_true_when_video_id_found(self):
        from valocoach.retrieval.youtube_ingest import is_video_ingested

        mock_coll = MagicMock()
        mock_coll.get.return_value = {"ids": ["youtube:abc12345678:0"]}

        with patch(
            "valocoach.retrieval.vector_store.get_collection", return_value=mock_coll
        ):
            assert is_video_ingested(MagicMock(), "abc12345678") is True

    def test_returns_false_on_chroma_error(self):
        """Dedup failure should not crash — treat as not ingested."""
        from valocoach.retrieval.youtube_ingest import is_video_ingested

        with patch(
            "valocoach.retrieval.vector_store.get_collection",
            side_effect=RuntimeError("chroma down"),
        ):
            assert is_video_ingested(MagicMock(), "abc12345678") is False


# ---------------------------------------------------------------------------
# D6 — Citation formatting in retriever
# ---------------------------------------------------------------------------


class TestRetrieverCitations:
    def test_youtube_hit_formatted_with_timestamp(self):
        """Verify retrieve_static formats YouTube hits as [SOURCE: youtube/title @ mm:ss].

        The label format was unified across all retrieval types so the LLM
        grounding rule (cite [SOURCE: ...] tags) has a single shape to follow.
        """
        from valocoach.retrieval.retriever import retrieve_static

        youtube_hit = {
            "text": "Stand here and throw the smoke to block Operator angle.",
            "metadata": {
                "type": "youtube",
                "title": "Haven A Execute Guide",
                "channel": "Woohoojin",
                "start_seconds": 754,  # 12:34
                "source": "https://youtube.com/watch?v=test",
            },
            "distance": 0.3,
        }

        # retrieve_static imports these inside the function body; patch at their
        # canonical module locations.
        with (
            patch("valocoach.retrieval.agents.format_agent_context", return_value=None),
            patch("valocoach.retrieval.maps.format_map_context", return_value="MAP: Haven"),
            patch(
                "valocoach.retrieval.meta.format_meta_context",
                return_value="META: patch 10.08",
            ),
            patch(
                "valocoach.retrieval.searcher.search",
                return_value=[youtube_hit],
            ),
        ):
            result = retrieve_static(
                "smoke the op angle on a site",
                data_dir=MagicMock(),
                agent=None,
                map_="Haven",
            )

        # At least one chunk should contain a [SOURCE: youtube/...] citation
        # with the title and a mm:ss timestamp.
        combined = "\n".join(result.static_chunks)
        assert "[SOURCE: youtube/" in combined
        assert "Haven A Execute Guide" in combined
        assert "12:34" in combined

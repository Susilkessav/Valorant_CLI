"""Tests for the ingest progress display and on_progress callback.

Covers:
  - _upsert_batch calls on_progress after each batch with correct counts
  - ingest_knowledge_base threads on_progress through to _upsert_batch
  - _do_seed passes a progress callback to ingest_knowledge_base
  - _do_corpus renders per-file progress (one call per .md file)
  - _do_url: success path, scrape failure → Exit(1)
  - _do_youtube: success path, fetch failure → Exit(1), ingest error → Exit(1)
  - CLI integration: valocoach ingest --seed exits 0 and shows success

Patch targets
-------------
Lazy imports inside function bodies are patched at the *source* module.
Module-level imports in ingest.py are patched at valocoach.cli.commands.ingest.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from valocoach.cli.app import app

runner = CliRunner()

# Source-module patch targets
_EMBED = "valocoach.retrieval.ingester.embed"
_GET_COLLECTION = "valocoach.retrieval.ingester.get_collection"
_UPSERT_BATCH = "valocoach.retrieval.ingester._upsert_batch"
_INGEST_KB_SRC = "valocoach.retrieval.ingester.ingest_knowledge_base"
_INGEST_TEXT_SRC = "valocoach.retrieval.ingester.ingest_text"
_SCRAPE_URL_SRC = "valocoach.retrieval.scrapers.web.scrape_url"
_FETCH_TRANSCRIPT_SRC = "valocoach.retrieval.scrapers.youtube.fetch_transcript"
_LOAD_SETTINGS = "valocoach.core.config.load_settings"


# ---------------------------------------------------------------------------
# _upsert_batch progress callback
# ---------------------------------------------------------------------------


class TestUpsertBatchProgressCallback:
    def _fake_collection(self):
        col = MagicMock()
        col.upsert = MagicMock()
        return col

    def test_callback_called_once_for_single_batch(self):
        """20 docs with batch_size=32 → one batch → one callback call."""
        from valocoach.retrieval.ingester import _upsert_batch

        calls = []
        with (
            patch(_EMBED, return_value=[[0.1] * 8] * 20),
            patch(_GET_COLLECTION, return_value=self._fake_collection()),
        ):
            _upsert_batch(
                Path("/tmp"),
                texts=["t"] * 20,
                ids=[str(i) for i in range(20)],
                metadatas=[{}] * 20,
                batch_size=32,
                on_progress=lambda done, total: calls.append((done, total)),
            )
        assert calls == [(20, 20)]

    def test_callback_called_per_batch(self):
        """50 docs, batch_size=20 → 3 batches → 3 callback calls."""
        from valocoach.retrieval.ingester import _upsert_batch

        calls = []
        with (
            patch(_EMBED, return_value=[[0.1] * 8] * 20),
            patch(_GET_COLLECTION, return_value=self._fake_collection()),
        ):
            _upsert_batch(
                Path("/tmp"),
                texts=["t"] * 50,
                ids=[str(i) for i in range(50)],
                metadatas=[{}] * 50,
                batch_size=20,
                on_progress=lambda done, total: calls.append((done, total)),
            )
        assert len(calls) == 3
        assert calls[-1] == (50, 50)

    def test_callback_completed_increments_correctly(self):
        """completed should equal cumulative docs processed after each batch."""
        from valocoach.retrieval.ingester import _upsert_batch

        calls = []
        with (
            patch(_EMBED, side_effect=lambda texts: [[0.0]] * len(texts)),
            patch(_GET_COLLECTION, return_value=self._fake_collection()),
        ):
            _upsert_batch(
                Path("/tmp"),
                texts=["t"] * 45,
                ids=[str(i) for i in range(45)],
                metadatas=[{}] * 45,
                batch_size=20,
                on_progress=lambda done, total: calls.append((done, total)),
            )
        # Batches: 20, 20, 5 → cumulative: 20, 40, 45
        assert [done for done, _ in calls] == [20, 40, 45]

    def test_no_callback_does_not_raise(self):
        """on_progress=None must be silently ignored."""
        from valocoach.retrieval.ingester import _upsert_batch

        with (
            patch(_EMBED, return_value=[[0.1]] * 5),
            patch(_GET_COLLECTION, return_value=self._fake_collection()),
        ):
            result = _upsert_batch(
                Path("/tmp"),
                texts=["t"] * 5,
                ids=[str(i) for i in range(5)],
                metadatas=[{}] * 5,
                on_progress=None,
            )
        assert result == 5

    def test_returns_total_doc_count(self):
        from valocoach.retrieval.ingester import _upsert_batch

        with (
            patch(_EMBED, return_value=[[0.0]] * 7),
            patch(_GET_COLLECTION, return_value=self._fake_collection()),
        ):
            result = _upsert_batch(
                Path("/tmp"),
                texts=["t"] * 7,
                ids=[str(i) for i in range(7)],
                metadatas=[{}] * 7,
            )
        assert result == 7


# ---------------------------------------------------------------------------
# ingest_knowledge_base threads on_progress to _upsert_batch
# ---------------------------------------------------------------------------


class TestIngestKnowledgeBaseProgress:
    """Test that on_progress kwarg is forwarded to _upsert_batch."""

    def _call_ingest_kb(self, on_progress):
        """Call ingest_knowledge_base with all file-IO and formatting mocked."""
        from valocoach.retrieval.ingester import ingest_knowledge_base

        with (
            patch(_UPSERT_BATCH, return_value=5) as mock_upsert,
            patch(
                "valocoach.retrieval.ingester.json.load",
                side_effect=[
                    {"agents": [{"name": "Jett", "role": "Duelist"}]},
                    {"maps": [{"name": "Ascent"}]},
                    {"map_meta": {}},
                ],
            ),
            patch("builtins.open", MagicMock()),
            patch(
                "valocoach.retrieval.agents.format_agent_context",
                return_value="agent text",
            ),
            patch(
                "valocoach.retrieval.maps.format_map_context",
                return_value="map text",
            ),
            patch(
                "valocoach.retrieval.meta.format_meta_context",
                return_value="meta text",
            ),
        ):
            ingest_knowledge_base(Path("/tmp"), on_progress=on_progress)
        return mock_upsert

    def test_on_progress_forwarded_as_kwarg(self):
        callback = MagicMock()
        mock_upsert = self._call_ingest_kb(callback)
        _, kwargs = mock_upsert.call_args
        assert kwargs.get("on_progress") is callback

    def test_on_progress_none_forwarded(self):
        mock_upsert = self._call_ingest_kb(None)
        _, kwargs = mock_upsert.call_args
        assert kwargs.get("on_progress") is None


# ---------------------------------------------------------------------------
# CLI ingest --seed integration
# ---------------------------------------------------------------------------


class TestIngestCLISeed:
    def _fake_settings(self, tmp_path: Path):
        s = MagicMock()
        s.data_dir = tmp_path
        return s

    def test_seed_exits_zero(self, tmp_path: Path):
        with (
            patch(_LOAD_SETTINGS, return_value=self._fake_settings(tmp_path)),
            patch(
                _INGEST_KB_SRC,
                return_value={"agents": 3, "maps": 9, "meta": 5, "total": 17},
            ),
        ):
            result = runner.invoke(app, ["ingest", "--seed"])
        assert result.exit_code == 0

    def test_seed_shows_doc_count_in_output(self, tmp_path: Path):
        with (
            patch(_LOAD_SETTINGS, return_value=self._fake_settings(tmp_path)),
            patch(
                _INGEST_KB_SRC,
                return_value={"agents": 3, "maps": 9, "meta": 5, "total": 17},
            ),
        ):
            result = runner.invoke(app, ["ingest", "--seed"])
        assert "17" in result.output or "Seeded" in result.output

    def test_seed_passes_callable_to_ingest_knowledge_base(self, tmp_path: Path):
        received = []

        def capture(data_dir, on_progress=None):
            received.append(on_progress)
            return {"agents": 1, "maps": 1, "meta": 1, "total": 3}

        with (
            patch(_LOAD_SETTINGS, return_value=self._fake_settings(tmp_path)),
            patch(_INGEST_KB_SRC, side_effect=capture),
        ):
            runner.invoke(app, ["ingest", "--seed"])

        assert len(received) == 1
        assert callable(received[0])

    def test_seed_error_exits_nonzero(self, tmp_path: Path):
        with (
            patch(_LOAD_SETTINGS, return_value=self._fake_settings(tmp_path)),
            patch(_INGEST_KB_SRC, side_effect=RuntimeError("embed failed")),
        ):
            result = runner.invoke(app, ["ingest", "--seed"])
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# _do_corpus progress — call function directly with controlled corpus root
# ---------------------------------------------------------------------------


class TestDoCorpusProgress:
    def test_corpus_missing_dir_shows_error(self, tmp_path: Path):
        """When corpus_root does not exist, _do_corpus raises typer.Exit(1)."""
        import click

        from valocoach.cli.commands.ingest import _do_corpus

        with pytest.raises(click.exceptions.Exit) as exc_info:
            _do_corpus(tmp_path, corpus_root=tmp_path / "no_corpus")
        assert exc_info.value.exit_code == 1

    def test_corpus_calls_ingest_text_per_md_file(self, tmp_path: Path):
        """_do_corpus must call ingest_text once per discovered .md file."""
        from valocoach.cli.commands.ingest import _do_corpus

        corpus = tmp_path / "corpus" / "agents"
        corpus.mkdir(parents=True)
        (corpus / "jett.md").write_text("Jett is a Duelist.")
        (corpus / "sage.md").write_text("Sage is a Sentinel.")

        ingest_calls = []

        def fake_ingest_text(data_dir, text, **kw):
            ingest_calls.append(kw.get("name", ""))
            return 1

        with patch(_INGEST_TEXT_SRC, side_effect=fake_ingest_text):
            _do_corpus(tmp_path, corpus_root=tmp_path / "corpus")

        assert len(ingest_calls) == 2

    def test_corpus_no_md_files_shows_warning(self, tmp_path: Path):
        """An empty corpus (no .md files) warns and returns cleanly."""
        from valocoach.cli.commands.ingest import _do_corpus

        empty_corpus = tmp_path / "corpus"
        empty_corpus.mkdir()
        # Should return without raising
        _do_corpus(tmp_path, corpus_root=empty_corpus)


# ---------------------------------------------------------------------------
# _do_url — scrape a URL and ingest its text
# ---------------------------------------------------------------------------


class TestDoUrl:
    def _make_content(self):
        from valocoach.retrieval.scrapers import ScrapedContent

        return ScrapedContent(
            url="https://valorant.com/patch-notes",
            title="Patch Notes",
            text="Some long patch note content here.",
            fetched_at="2026-01-01T00:00:00+00:00",
            source="patch_note",
        )

    def test_success_exits_cleanly(self, tmp_path: Path):
        from valocoach.cli.commands.ingest import _do_url

        with (
            patch(_SCRAPE_URL_SRC, return_value=self._make_content()),
            patch(_INGEST_TEXT_SRC, return_value=3),
        ):
            # Must not raise
            _do_url(tmp_path, "https://valorant.com/patch-notes")

    def test_success_calls_ingest_text_with_patch_note_type(self, tmp_path: Path):
        from valocoach.cli.commands.ingest import _do_url

        ingest_calls = []

        def capture(data_dir, text, **kw):
            ingest_calls.append(kw)
            return 3

        with (
            patch(_SCRAPE_URL_SRC, return_value=self._make_content()),
            patch(_INGEST_TEXT_SRC, side_effect=capture),
        ):
            _do_url(tmp_path, "https://valorant.com/patch-notes")

        assert len(ingest_calls) == 1
        assert ingest_calls[0]["doc_type"] == "patch_note"

    def test_scrape_failure_raises_exit_1(self, tmp_path: Path):
        import click

        from valocoach.cli.commands.ingest import _do_url

        with (
            patch(_SCRAPE_URL_SRC, return_value=None),
            pytest.raises(click.exceptions.Exit) as exc_info,
        ):
            _do_url(tmp_path, "https://example.com/bad-url")

        assert exc_info.value.exit_code == 1

    def test_ingest_called_with_fetched_at_metadata(self, tmp_path: Path):
        from valocoach.cli.commands.ingest import _do_url

        captured = {}

        def capture(data_dir, text, **kw):
            captured.update(kw)
            return 1

        with (
            patch(_SCRAPE_URL_SRC, return_value=self._make_content()),
            patch(_INGEST_TEXT_SRC, side_effect=capture),
        ):
            _do_url(tmp_path, "https://valorant.com/patch-notes")

        assert "extra_metadata" in captured
        assert "fetched_at" in captured["extra_metadata"]
        assert captured["extra_metadata"]["ttl_tier"] == "live"


# ---------------------------------------------------------------------------
# _do_youtube — fetch a YouTube transcript and ingest it
# ---------------------------------------------------------------------------


class TestDoYoutube:
    def _make_content(self):
        from valocoach.retrieval.scrapers import ScrapedContent

        return ScrapedContent(
            url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            title="YouTube transcript — dQw4w9WgXcQ",
            text="Valorant coaching tips transcript content here.",
            fetched_at="2026-01-01T00:00:00+00:00",
            source="youtube",
        )

    def test_success_exits_cleanly(self, tmp_path: Path):
        from valocoach.cli.commands.ingest import _do_youtube

        with (
            patch(_FETCH_TRANSCRIPT_SRC, return_value=self._make_content()),
            patch(_INGEST_TEXT_SRC, return_value=4),
        ):
            _do_youtube(tmp_path, "dQw4w9WgXcQ")

    def test_success_calls_ingest_text_with_youtube_type(self, tmp_path: Path):
        from valocoach.cli.commands.ingest import _do_youtube

        ingest_calls = []

        def capture(data_dir, text, **kw):
            ingest_calls.append(kw)
            return 4

        with (
            patch(_FETCH_TRANSCRIPT_SRC, return_value=self._make_content()),
            patch(_INGEST_TEXT_SRC, side_effect=capture),
        ):
            _do_youtube(tmp_path, "dQw4w9WgXcQ")

        assert len(ingest_calls) == 1
        assert ingest_calls[0]["doc_type"] == "youtube"

    def test_fetch_failure_raises_exit_1(self, tmp_path: Path):
        import click

        from valocoach.cli.commands.ingest import _do_youtube

        with (
            patch(_FETCH_TRANSCRIPT_SRC, return_value=None),
            pytest.raises(click.exceptions.Exit) as exc_info,
        ):
            _do_youtube(tmp_path, "dQw4w9WgXcQ")

        assert exc_info.value.exit_code == 1

    def test_ingest_error_raises_exit_1(self, tmp_path: Path):
        import click

        from valocoach.cli.commands.ingest import _do_youtube

        with (
            patch(_FETCH_TRANSCRIPT_SRC, return_value=self._make_content()),
            patch(_INGEST_TEXT_SRC, side_effect=RuntimeError("embed failed")),
            pytest.raises(click.exceptions.Exit) as exc_info,
        ):
            _do_youtube(tmp_path, "dQw4w9WgXcQ")

        assert exc_info.value.exit_code == 1

    def test_ingest_called_with_live_ttl_metadata(self, tmp_path: Path):
        from valocoach.cli.commands.ingest import _do_youtube

        captured = {}

        def capture(data_dir, text, **kw):
            captured.update(kw)
            return 1

        with (
            patch(_FETCH_TRANSCRIPT_SRC, return_value=self._make_content()),
            patch(_INGEST_TEXT_SRC, side_effect=capture),
        ):
            _do_youtube(tmp_path, "dQw4w9WgXcQ")

        assert captured.get("extra_metadata", {}).get("ttl_tier") == "live"

"""Tests for valocoach.retrieval.meta_sync.

Covers the run_meta_sync pipeline:
  - Early exit when no new patch and not forced
  - Patch check failure returns error result
  - Happy path: new patch → scrape → generate → write → ingest
  - Dry run: all steps run but meta.json not written
  - Individual step failures are recorded but don't crash the pipeline
  - on_step callback is invoked
  - YouTube ingest path
  - SyncResult.summary() and .ok properties
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Patch targets (source-module lazy imports)
# ---------------------------------------------------------------------------

_CHECK_PATCH = "valocoach.retrieval.patch_tracker.check_patch_update"
_FETCH_PATCH_NOTES = "valocoach.retrieval.scrapers.patch_notes.fetch_patch_notes"
_FETCH_ALL_STATS = "valocoach.retrieval.scrapers.meta_stats.fetch_all_stats"
_GENERATE_META = "valocoach.retrieval.meta_generator.generate_meta_update"
_INGEST_KB = "valocoach.retrieval.ingester.ingest_knowledge_base"
_STREAM_COMPLETION = "valocoach.llm.provider.stream_completion"

# In meta_sync the imports are lazy — patch at source
_CHECK_PATCH_SYNC = "valocoach.retrieval.patch_tracker.check_patch_update"


def _make_settings():
    s = MagicMock()
    s.ollama_model = "llama3"
    s.data_dir = MagicMock()  # not used in any real path here
    return s


def _make_scraped_content(text: str = "patch content " * 50):
    sc = MagicMock()
    sc.text = text
    return sc


def _make_stats(ranked: str = "ranked " * 20, pro: str = "pro " * 20):
    from valocoach.retrieval.scrapers.meta_stats import MetaStatsResult

    return MetaStatsResult(ranked_text=ranked, pro_text=pro)


_VALID_NEW_META = {
    "tier_list": {"S": ["Jett"], "A": ["Sage"], "B": ["Brimstone"], "C": ["Yoru"]},
    "agent_meta": {},
    "map_meta": {},
}


# ---------------------------------------------------------------------------
# SyncResult unit tests
# ---------------------------------------------------------------------------


class TestSyncResult:
    def test_ok_true_when_no_errors(self):
        from valocoach.retrieval.meta_sync import SyncResult

        r = SyncResult(patch_version="10.09", is_new_patch=True)
        assert r.ok is True

    def test_ok_false_with_errors(self):
        from valocoach.retrieval.meta_sync import SyncResult

        r = SyncResult(patch_version="10.09", is_new_patch=True, errors=["boom"])
        assert r.ok is False

    def test_summary_contains_version(self):
        from valocoach.retrieval.meta_sync import SyncResult

        r = SyncResult(patch_version="10.09", is_new_patch=True)
        assert "10.09" in r.summary()

    def test_summary_shows_errors(self):
        from valocoach.retrieval.meta_sync import SyncResult

        r = SyncResult(patch_version="10.09", is_new_patch=False, errors=["something broke"])
        assert "something broke" in r.summary()


# ---------------------------------------------------------------------------
# run_meta_sync pipeline tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestRunMetaSync:
    """Integration-style tests for the full pipeline with mocked deps."""

    async def _run(
        self,
        *,
        is_new_patch: bool = True,
        force: bool = False,
        dry_run: bool = False,
        patch_notes=None,  # None → use default scraped content
        stats=None,
        new_meta=None,
        patch_check_exc=None,
        youtube_videos=None,
        on_step=None,
    ):
        """Run the pipeline with all external deps mocked."""
        from valocoach.retrieval.meta_sync import run_meta_sync

        settings = _make_settings()

        if patch_notes is None:
            patch_notes = _make_scraped_content()
        if stats is None:
            stats = _make_stats()
        if new_meta is None:
            new_meta = _VALID_NEW_META.copy()

        if patch_check_exc:
            check_mock = AsyncMock(side_effect=patch_check_exc)
        else:
            check_mock = AsyncMock(return_value=("10.09", is_new_patch))

        meta_json = json.dumps(new_meta)

        with (
            patch("valocoach.retrieval.patch_tracker.HenrikClient", self._noop_henrik()),
            patch("valocoach.retrieval.patch_tracker.session_scope", self._noop_scope()),
            patch(
                "valocoach.retrieval.patch_tracker.invalidate_volatile", AsyncMock(return_value=0)
            ),
            patch("valocoach.retrieval.patch_tracker.check_patch_update", check_mock),
            patch("valocoach.retrieval.scrapers.patch_notes.scrape_url", return_value=patch_notes),
            patch("valocoach.retrieval.scrapers.meta_stats.scrape_url", return_value=patch_notes),
            patch("valocoach.retrieval.cache.store_cached", AsyncMock()),
            patch("valocoach.retrieval.ingester.ingest_text", return_value=5),
            patch("valocoach.llm.provider.stream_completion", return_value=iter(meta_json)),
            patch("builtins.open", self._mock_open(json.dumps(_VALID_NEW_META))),
            patch("valocoach.retrieval.ingester.ingest_knowledge_base", return_value={"total": 10}),
            patch("valocoach.retrieval.meta as _meta_mod", create=True),
            patch("valocoach.retrieval.meta_sync._META_FILE", MagicMock()),
        ):
            # Patch the lazy module-level cache reset in meta_sync
            import valocoach.retrieval.meta as _m

            with patch.object(_m, "_cache", None, create=True):
                return await run_meta_sync(
                    settings,
                    force=force,
                    dry_run=dry_run,
                    youtube_videos=youtube_videos,
                    on_step=on_step,
                )

    def _noop_henrik(self):
        @asynccontextmanager
        async def _ctx(settings):
            client = AsyncMock()
            client.get_version = AsyncMock(return_value={"version": "10.09"})
            yield client

        return _ctx

    def _noop_scope(self):
        pv = MagicMock()
        pv.game_version = "10.09"

        @asynccontextmanager
        async def _scope():
            session = AsyncMock()
            session.scalar = AsyncMock(return_value=pv)
            session.add = MagicMock()
            yield session

        return _scope

    def _mock_open(self, read_data: str):
        """Return a mock open() that yields read_data on read and captures writes."""
        from unittest.mock import mock_open

        return mock_open(read_data=read_data)

    async def test_no_new_patch_no_force_returns_early(self):
        from valocoach.retrieval.meta_sync import run_meta_sync

        settings = _make_settings()
        check_mock = AsyncMock(return_value=("10.09", False))

        with patch("valocoach.retrieval.patch_tracker.check_patch_update", check_mock):
            result = await run_meta_sync(settings, force=False)

        assert result.is_new_patch is False
        assert result.meta_written is False
        assert result.ok is True

    async def test_patch_check_failure_returns_error_result(self):
        from valocoach.retrieval.meta_sync import run_meta_sync

        settings = _make_settings()
        check_mock = AsyncMock(side_effect=RuntimeError("no network"))

        with patch("valocoach.retrieval.patch_tracker.check_patch_update", check_mock):
            result = await run_meta_sync(settings)

        assert not result.ok
        assert "Patch check failed" in result.errors[0]

    async def test_force_runs_even_without_new_patch(self):
        from valocoach.retrieval.meta_sync import run_meta_sync

        settings = _make_settings()
        # No new patch, but force=True
        check_mock = AsyncMock(return_value=("10.09", False))
        notes = _make_scraped_content()
        meta_json = json.dumps(_VALID_NEW_META)

        with (
            patch("valocoach.retrieval.patch_tracker.check_patch_update", check_mock),
            patch("valocoach.retrieval.scrapers.patch_notes.scrape_url", return_value=notes),
            patch("valocoach.retrieval.scrapers.meta_stats.scrape_url", return_value=notes),
            patch("valocoach.retrieval.cache.store_cached", AsyncMock()),
            patch("valocoach.retrieval.ingester.ingest_text", return_value=0),
            patch("valocoach.retrieval.patch_diff.extract_patch_changes"),
            patch(
                "valocoach.llm.provider.stream_completion",
                side_effect=lambda *a, **k: iter(meta_json),
            ),
            patch("builtins.open", self._mock_open(json.dumps(_VALID_NEW_META))),
            patch("valocoach.retrieval.ingester.ingest_knowledge_base", return_value={"total": 5}),
        ):
            import valocoach.retrieval.meta as _m

            with patch.object(_m, "_cache", None, create=True):
                result = await run_meta_sync(settings, force=True)

        # Should attempt meta regeneration despite no new patch
        assert result.meta_regenerated is True

    async def test_on_step_callback_is_invoked(self):
        from valocoach.retrieval.meta_sync import run_meta_sync

        settings = _make_settings()
        check_mock = AsyncMock(return_value=("10.09", False))
        steps_seen: list[str] = []

        def on_step(name: str, status: str) -> None:
            steps_seen.append(name)

        with patch("valocoach.retrieval.patch_tracker.check_patch_update", check_mock):
            await run_meta_sync(settings, on_step=on_step)

        assert "patch_check" in steps_seen

    async def test_no_source_data_skips_llm(self):
        """When both scrape steps fail, the pipeline exits early without calling LLM."""
        from valocoach.retrieval.meta_sync import run_meta_sync

        settings = _make_settings()
        check_mock = AsyncMock(return_value=("10.09", True))

        with (
            patch("valocoach.retrieval.patch_tracker.check_patch_update", check_mock),
            patch("valocoach.retrieval.scrapers.patch_notes.scrape_url", return_value=None),
            patch("valocoach.retrieval.scrapers.meta_stats.scrape_url", return_value=None),
        ):
            result = await run_meta_sync(settings, force=True)

        assert result.meta_regenerated is False
        # Error recorded for missing source data
        assert any("No source data" in e for e in result.errors)

    async def test_patch_notes_scraped_flag_set_on_success(self):
        """patch_notes_scraped is True when scrape returns content."""
        from valocoach.retrieval.meta_sync import run_meta_sync

        settings = _make_settings()
        check_mock = AsyncMock(return_value=("10.09", True))
        notes = _make_scraped_content("patch notes content " * 30)
        meta_json = json.dumps(_VALID_NEW_META)

        with (
            patch("valocoach.retrieval.patch_tracker.check_patch_update", check_mock),
            patch("valocoach.retrieval.scrapers.patch_notes.scrape_url", return_value=notes),
            patch("valocoach.retrieval.scrapers.meta_stats.scrape_url", return_value=None),
            patch("valocoach.llm.provider.stream_completion", return_value=iter(meta_json)),
            patch("builtins.open", self._mock_open(json.dumps(_VALID_NEW_META))),
            patch("valocoach.retrieval.ingester.ingest_knowledge_base", return_value={"total": 5}),
        ):
            import valocoach.retrieval.meta as _m

            with patch.object(_m, "_cache", None, create=True):
                result = await run_meta_sync(settings, force=True)

        assert result.patch_notes_scraped is True

    async def test_stats_scraped_flags_set_on_success(self):
        """ranked_stats_scraped and pro_stats_scraped are set when stats return data."""
        from valocoach.retrieval.meta_sync import run_meta_sync

        settings = _make_settings()
        check_mock = AsyncMock(return_value=("10.09", True))
        content = _make_scraped_content("stats content " * 30)
        meta_json = json.dumps(_VALID_NEW_META)

        with (
            patch("valocoach.retrieval.patch_tracker.check_patch_update", check_mock),
            patch("valocoach.retrieval.scrapers.patch_notes.scrape_url", return_value=None),
            patch("valocoach.retrieval.scrapers.meta_stats.scrape_url", return_value=content),
            patch("valocoach.retrieval.cache.store_cached", AsyncMock()),
            patch("valocoach.retrieval.ingester.ingest_text", return_value=3),
            patch("valocoach.llm.provider.stream_completion", return_value=iter(meta_json)),
            patch("builtins.open", self._mock_open(json.dumps(_VALID_NEW_META))),
            patch("valocoach.retrieval.ingester.ingest_knowledge_base", return_value={"total": 5}),
        ):
            import valocoach.retrieval.meta as _m

            with patch.object(_m, "_cache", None, create=True):
                result = await run_meta_sync(settings, force=True)

        assert result.ranked_stats_scraped is True
        assert result.pro_stats_scraped is True

    async def test_dry_run_skips_writing_meta(self):
        """In dry_run mode, meta.json is never written and meta_written stays False."""
        from valocoach.retrieval.meta_sync import run_meta_sync

        settings = _make_settings()
        check_mock = AsyncMock(return_value=("10.09", True))
        notes = _make_scraped_content("patch notes " * 30)
        meta_json = json.dumps(_VALID_NEW_META)

        with (
            patch("valocoach.retrieval.patch_tracker.check_patch_update", check_mock),
            patch("valocoach.retrieval.scrapers.patch_notes.scrape_url", return_value=notes),
            patch("valocoach.retrieval.scrapers.meta_stats.scrape_url", return_value=None),
            patch("valocoach.llm.provider.stream_completion", return_value=iter(meta_json)),
            patch("builtins.open", self._mock_open(json.dumps(_VALID_NEW_META))),
        ):
            result = await run_meta_sync(settings, force=True, dry_run=True)

        assert result.meta_written is False
        assert result.meta_regenerated is True  # LLM step still runs

    async def test_meta_written_and_ingested_on_happy_path(self):
        """meta_written=True triggers re-ingest and sets meta_ingested=True."""
        from valocoach.retrieval.meta_sync import run_meta_sync

        settings = _make_settings()
        check_mock = AsyncMock(return_value=("10.09", True))
        notes = _make_scraped_content("patch notes " * 30)
        meta_json = json.dumps(_VALID_NEW_META)
        ingest_mock = MagicMock(return_value={"total": 10})

        with (
            patch("valocoach.retrieval.patch_tracker.check_patch_update", check_mock),
            patch("valocoach.retrieval.scrapers.patch_notes.scrape_url", return_value=notes),
            patch("valocoach.retrieval.scrapers.meta_stats.scrape_url", return_value=None),
            patch("valocoach.retrieval.patch_diff.extract_patch_changes"),
            patch(
                "valocoach.llm.provider.stream_completion",
                side_effect=lambda *a, **k: iter(meta_json),
            ),
            patch("builtins.open", self._mock_open(json.dumps(_VALID_NEW_META))),
            patch("valocoach.retrieval.ingester.ingest_knowledge_base", ingest_mock),
        ):
            import valocoach.retrieval.meta as _m

            with patch.object(_m, "_cache", None, create=True):
                result = await run_meta_sync(settings, force=True)

        assert result.meta_written is True
        assert result.meta_ingested is True
        ingest_mock.assert_called_once()

    async def test_llm_returns_none_records_error(self):
        """When generate_meta_update returns None, an error is appended."""
        from valocoach.retrieval.meta_sync import run_meta_sync

        settings = _make_settings()
        check_mock = AsyncMock(return_value=("10.09", True))
        notes = _make_scraped_content("notes " * 30)

        with (
            patch("valocoach.retrieval.patch_tracker.check_patch_update", check_mock),
            patch("valocoach.retrieval.scrapers.patch_notes.scrape_url", return_value=notes),
            patch("valocoach.retrieval.scrapers.meta_stats.scrape_url", return_value=None),
            # LLM produces non-JSON so generate_meta_update returns None
            patch("valocoach.llm.provider.stream_completion", return_value=iter("NOT JSON")),
            patch("builtins.open", self._mock_open(json.dumps(_VALID_NEW_META))),
        ):
            result = await run_meta_sync(settings, force=True)

        assert result.meta_regenerated is False
        assert any("LLM returned no valid JSON" in e for e in result.errors)

    async def test_youtube_videos_ingest(self):
        """YouTube video URLs/IDs are processed and chunk count is recorded."""
        from valocoach.retrieval.meta_sync import run_meta_sync

        settings = _make_settings()
        check_mock = AsyncMock(return_value=("10.09", False))  # no new patch — stop at youtube step

        # Patch the full ingest helper — meta_sync just sums its return value.
        ingest_mock = MagicMock(return_value=7)

        with (
            patch("valocoach.retrieval.patch_tracker.check_patch_update", check_mock),
            patch("valocoach.retrieval.scrapers.patch_notes.scrape_url", return_value=None),
            patch("valocoach.retrieval.scrapers.meta_stats.scrape_url", return_value=None),
            patch("valocoach.retrieval.youtube_ingest.ingest_youtube_video", ingest_mock),
        ):
            result = await run_meta_sync(
                settings,
                force=True,
                youtube_videos=["https://youtube.com/watch?v=dQw4w9WgXcQ"],
            )

        assert result.youtube_chunks_ingested == 7

    async def test_patch_notes_scrape_exception_recorded(self):
        """Exception in fetch_patch_notes is caught and appended to errors."""
        from valocoach.retrieval.meta_sync import run_meta_sync

        settings = _make_settings()
        check_mock = AsyncMock(return_value=("10.09", True))

        with (
            patch("valocoach.retrieval.patch_tracker.check_patch_update", check_mock),
            # fetch_patch_notes itself raises (scrape_url errors are otherwise
            # swallowed internally by the multi-source fetcher).
            patch(
                "valocoach.retrieval.scrapers.patch_notes.fetch_patch_notes",
                side_effect=RuntimeError("connection refused"),
            ),
            patch("valocoach.retrieval.scrapers.meta_stats.scrape_url", return_value=None),
        ):
            result = await run_meta_sync(settings, force=True)

        assert any("Patch notes scrape error" in e for e in result.errors)

    async def test_stats_scrape_exception_recorded(self):
        """Exception in fetch_all_stats is caught and appended to errors."""
        from valocoach.retrieval.meta_sync import run_meta_sync

        settings = _make_settings()
        check_mock = AsyncMock(return_value=("10.09", True))

        with (
            patch("valocoach.retrieval.patch_tracker.check_patch_update", check_mock),
            patch("valocoach.retrieval.scrapers.patch_notes.scrape_url", return_value=None),
            patch(
                "valocoach.retrieval.scrapers.meta_stats.scrape_url",
                side_effect=RuntimeError("timeout"),
            ),
        ):
            result = await run_meta_sync(settings, force=True)

        assert any("Stats scrape error" in e for e in result.errors)

    async def test_youtube_fetch_transcript_returns_none_skips_ingest(self):
        """When fetch_transcript returns None, the transcript is skipped (no ingest)."""
        from valocoach.retrieval.meta_sync import run_meta_sync

        settings = _make_settings()
        check_mock = AsyncMock(return_value=("10.09", True))

        ingest_text_mock = MagicMock(return_value=0)

        with (
            patch("valocoach.retrieval.patch_tracker.check_patch_update", check_mock),
            patch("valocoach.retrieval.scrapers.patch_notes.scrape_url", return_value=None),
            patch("valocoach.retrieval.scrapers.meta_stats.scrape_url", return_value=None),
            patch("valocoach.retrieval.scrapers.youtube.fetch_transcript", return_value=None),
            patch("valocoach.retrieval.ingester.ingest_text", ingest_text_mock),
        ):
            result = await run_meta_sync(
                settings,
                force=True,
                youtube_videos=["https://youtube.com/watch?v=abc"],
            )

        # transcript=None → ingest not called, chunks=0
        ingest_text_mock.assert_not_called()
        assert result.youtube_chunks_ingested == 0

    async def test_youtube_ingest_exception_recorded(self):
        """Exception during transcript ingest is appended to errors."""
        from valocoach.retrieval.meta_sync import run_meta_sync

        settings = _make_settings()
        check_mock = AsyncMock(return_value=("10.09", True))

        with (
            patch("valocoach.retrieval.patch_tracker.check_patch_update", check_mock),
            patch("valocoach.retrieval.scrapers.patch_notes.scrape_url", return_value=None),
            patch("valocoach.retrieval.scrapers.meta_stats.scrape_url", return_value=None),
            patch(
                "valocoach.retrieval.youtube_ingest.ingest_youtube_video",
                side_effect=RuntimeError("chroma error"),
            ),
        ):
            result = await run_meta_sync(
                settings,
                force=True,
                youtube_videos=["https://youtube.com/watch?v=dQw4w9WgXcQ"],
            )

        assert any("YouTube ingest error" in e for e in result.errors)

    async def test_meta_generation_exception_recorded(self):
        """Exception in meta_generator (e.g., file not found) is caught."""
        from valocoach.retrieval.meta_sync import run_meta_sync

        settings = _make_settings()
        check_mock = AsyncMock(return_value=("10.09", True))
        notes = _make_scraped_content("notes " * 30)

        with (
            patch("valocoach.retrieval.patch_tracker.check_patch_update", check_mock),
            patch("valocoach.retrieval.scrapers.patch_notes.scrape_url", return_value=notes),
            patch("valocoach.retrieval.scrapers.meta_stats.scrape_url", return_value=None),
            # open() raises — json.load fails → exception caught at line 292
            patch("builtins.open", side_effect=FileNotFoundError("meta.json not found")),
        ):
            result = await run_meta_sync(settings, force=True)

        assert not result.meta_regenerated
        assert any("Meta generation/write error" in e for e in result.errors)

    async def test_re_ingest_error_recorded(self):
        """Re-ingest failure is appended to errors but meta_written stays True."""
        from valocoach.retrieval.meta_sync import run_meta_sync

        settings = _make_settings()
        check_mock = AsyncMock(return_value=("10.09", True))
        notes = _make_scraped_content("notes " * 30)
        meta_json = json.dumps(_VALID_NEW_META)

        with (
            patch("valocoach.retrieval.patch_tracker.check_patch_update", check_mock),
            patch("valocoach.retrieval.scrapers.patch_notes.scrape_url", return_value=notes),
            patch("valocoach.retrieval.scrapers.meta_stats.scrape_url", return_value=None),
            patch("valocoach.retrieval.patch_diff.extract_patch_changes"),
            patch(
                "valocoach.llm.provider.stream_completion",
                side_effect=lambda *a, **k: iter(meta_json),
            ),
            patch("builtins.open", self._mock_open(json.dumps(_VALID_NEW_META))),
            patch(
                "valocoach.retrieval.ingester.ingest_knowledge_base",
                side_effect=RuntimeError("chroma down"),
            ),
        ):
            import valocoach.retrieval.meta as _m

            with patch.object(_m, "_cache", None, create=True):
                result = await run_meta_sync(settings, force=True)

        assert result.meta_written is True
        assert result.meta_ingested is False
        assert any("Re-ingest error" in e for e in result.errors)

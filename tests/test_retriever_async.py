"""Async tests for the retriever pipeline — retrieve_context and _fetch_live_meta.

Covers:
  - retrieve_context returns a RetrievalResult with patch_version set
  - retrieve_context merges meta_chunks from _fetch_live_meta into the result
  - retrieve_context swallows _fetch_live_meta exceptions (non-fatal path)
  - _fetch_live_meta returns empty lists when no map is provided
  - _fetch_live_meta returns cached text on a cache hit
  - _fetch_live_meta scrapes, stores, and ingests on a cache miss
  - _fetch_live_meta returns empty when scraping fails

Patch targets (all at the source module so lazy imports are intercepted):
  valocoach.retrieval.retriever.retrieve_static      — same-module function
  valocoach.retrieval.retriever._fetch_live_meta     — same-module function
  valocoach.retrieval.patch_tracker.get_current_patch — lazy import in retrieve_context
  valocoach.retrieval.cache.get_cached                — lazy import in _fetch_live_meta
  valocoach.retrieval.cache.store_cached              — lazy import in _fetch_live_meta
  valocoach.retrieval.scrapers.web.scrape_url         — lazy import in _fetch_live_meta
  valocoach.retrieval.ingester.ingest_text            — lazy import in _fetch_live_meta
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

_RETRIEVE_STATIC = "valocoach.retrieval.retriever.retrieve_static"
_FETCH_LIVE_META = "valocoach.retrieval.retriever._fetch_live_meta"
_GET_CURRENT_PATCH = "valocoach.retrieval.patch_tracker.get_current_patch"
_GET_CACHED = "valocoach.retrieval.cache.get_cached"
_STORE_CACHED = "valocoach.retrieval.cache.store_cached"
_SCRAPE_URL = "valocoach.retrieval.scrapers.web.scrape_url"
_INGEST_TEXT = "valocoach.retrieval.ingester.ingest_text"


def _fake_settings(tmp_path: Path) -> MagicMock:
    s = MagicMock()
    s.data_dir = tmp_path
    return s


def _fake_static_result():
    from valocoach.retrieval.retriever import RetrievalResult

    return RetrievalResult(
        static_chunks=["static chunk about Jett"],
        meta_chunks=[],
        sources=["knowledge_base/agents/Jett"],
        patch_version=None,
    )


# ---------------------------------------------------------------------------
# retrieve_context
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestRetrieveContext:
    async def test_returns_retrieval_result_type(self, tmp_path: Path):
        from valocoach.retrieval.retriever import RetrievalResult, retrieve_context

        with (
            patch(_RETRIEVE_STATIC, return_value=_fake_static_result()),
            patch(_GET_CURRENT_PATCH, new=AsyncMock(return_value="10.09")),
            patch(_FETCH_LIVE_META, new=AsyncMock(return_value=([], []))),
        ):
            result = await retrieve_context(_fake_settings(tmp_path), "push A site")

        assert isinstance(result, RetrievalResult)

    async def test_patch_version_is_set_from_db(self, tmp_path: Path):
        from valocoach.retrieval.retriever import retrieve_context

        with (
            patch(_RETRIEVE_STATIC, return_value=_fake_static_result()),
            patch(_GET_CURRENT_PATCH, new=AsyncMock(return_value="10.09")),
            patch(_FETCH_LIVE_META, new=AsyncMock(return_value=([], []))),
        ):
            result = await retrieve_context(_fake_settings(tmp_path), "eco round")

        assert result.patch_version == "10.09"

    async def test_meta_chunks_merged_from_fetch_live_meta(self, tmp_path: Path):
        from valocoach.retrieval.retriever import retrieve_context

        with (
            patch(_RETRIEVE_STATIC, return_value=_fake_static_result()),
            patch(_GET_CURRENT_PATCH, new=AsyncMock(return_value="10.09")),
            patch(
                _FETCH_LIVE_META,
                new=AsyncMock(return_value=(["live meta text for Ascent"], ["https://src"])),
            ),
        ):
            result = await retrieve_context(
                _fake_settings(tmp_path), "push A site", map_="Ascent"
            )

        assert "live meta text for Ascent" in result.meta_chunks
        assert "https://src" in result.sources

    async def test_static_chunks_preserved(self, tmp_path: Path):
        from valocoach.retrieval.retriever import retrieve_context

        with (
            patch(_RETRIEVE_STATIC, return_value=_fake_static_result()),
            patch(_GET_CURRENT_PATCH, new=AsyncMock(return_value="10.09")),
            patch(_FETCH_LIVE_META, new=AsyncMock(return_value=([], []))),
        ):
            result = await retrieve_context(_fake_settings(tmp_path), "situation")

        assert "static chunk about Jett" in result.static_chunks

    async def test_live_meta_exception_is_non_fatal(self, tmp_path: Path):
        from valocoach.retrieval.retriever import retrieve_context

        with (
            patch(_RETRIEVE_STATIC, return_value=_fake_static_result()),
            patch(_GET_CURRENT_PATCH, new=AsyncMock(return_value="10.09")),
            patch(
                _FETCH_LIVE_META, new=AsyncMock(side_effect=RuntimeError("meta DB down"))
            ),
        ):
            result = await retrieve_context(_fake_settings(tmp_path), "push A")

        # Exception is swallowed — result still returned with empty meta
        assert result is not None
        assert result.meta_chunks == []

    async def test_patch_version_none_when_no_patch_recorded(self, tmp_path: Path):
        from valocoach.retrieval.retriever import retrieve_context

        with (
            patch(_RETRIEVE_STATIC, return_value=_fake_static_result()),
            patch(_GET_CURRENT_PATCH, new=AsyncMock(return_value=None)),
            patch(_FETCH_LIVE_META, new=AsyncMock(return_value=([], []))),
        ):
            result = await retrieve_context(_fake_settings(tmp_path), "situation")

        assert result.patch_version is None


# ---------------------------------------------------------------------------
# _fetch_live_meta
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestFetchLiveMeta:
    async def test_no_map_returns_empty_lists(self, tmp_path: Path):
        from valocoach.retrieval.retriever import _fetch_live_meta

        chunks, sources = await _fetch_live_meta(
            settings=_fake_settings(tmp_path),
            situation="push A",
            map_=None,
            n_results=3,
        )

        assert chunks == []
        assert sources == []

    async def test_cache_hit_returns_cached_text(self, tmp_path: Path):
        from valocoach.retrieval.retriever import _fetch_live_meta

        cached = "Ascent pick rates and win rates from this patch."
        with patch(_GET_CACHED, new=AsyncMock(return_value=cached)):
            chunks, sources = await _fetch_live_meta(
                settings=_fake_settings(tmp_path),
                situation="push A",
                map_="Ascent",
                n_results=3,
            )

        assert len(chunks) == 1
        assert cached[:2000] in chunks[0]
        assert len(sources) == 1

    async def test_cache_hit_source_url_is_metasrc(self, tmp_path: Path):
        from valocoach.retrieval.retriever import _fetch_live_meta

        with patch(_GET_CACHED, new=AsyncMock(return_value="some cached meta content")):
            _, sources = await _fetch_live_meta(
                settings=_fake_settings(tmp_path),
                situation="push A",
                map_="Bind",
                n_results=3,
            )

        assert any("metasrc.com" in s for s in sources)

    async def test_cache_miss_scrape_success_returns_text(self, tmp_path: Path):
        from valocoach.retrieval.retriever import _fetch_live_meta
        from valocoach.retrieval.scrapers import ScrapedContent

        scraped = ScrapedContent(
            url="https://www.metasrc.com/valorant/map/haven",
            title="Haven meta",
            text="Haven map stats: attack win rate 47%, pick rate 12%.",
            fetched_at="2026-01-01T00:00:00+00:00",
            source="web",
        )

        with (
            patch(_GET_CACHED, new=AsyncMock(return_value=None)),
            patch(_SCRAPE_URL, return_value=scraped),
            patch(_STORE_CACHED, new=AsyncMock()),
            patch(_INGEST_TEXT, return_value=1),
        ):
            chunks, _sources = await _fetch_live_meta(
                settings=_fake_settings(tmp_path),
                situation="push B",
                map_="Haven",
                n_results=3,
            )

        assert len(chunks) == 1
        assert "Haven map stats" in chunks[0]

    async def test_cache_miss_scrape_success_calls_store_cached(self, tmp_path: Path):
        from valocoach.retrieval.retriever import _fetch_live_meta
        from valocoach.retrieval.scrapers import ScrapedContent

        scraped = ScrapedContent(
            url="https://www.metasrc.com/valorant/map/split",
            title="Split meta",
            text="Split map win rates and pick rate data here.",
            fetched_at="2026-01-01T00:00:00+00:00",
            source="web",
        )
        mock_store = AsyncMock()

        with (
            patch(_GET_CACHED, new=AsyncMock(return_value=None)),
            patch(_SCRAPE_URL, return_value=scraped),
            patch(_STORE_CACHED, mock_store),
            patch(_INGEST_TEXT, return_value=1),
        ):
            await _fetch_live_meta(
                settings=_fake_settings(tmp_path),
                situation="eco round",
                map_="Split",
                n_results=3,
            )

        mock_store.assert_awaited_once()

    async def test_cache_miss_scrape_failure_returns_empty(self, tmp_path: Path):
        from valocoach.retrieval.retriever import _fetch_live_meta

        with (
            patch(_GET_CACHED, new=AsyncMock(return_value=None)),
            patch(_SCRAPE_URL, return_value=None),
        ):
            chunks, sources = await _fetch_live_meta(
                settings=_fake_settings(tmp_path),
                situation="rotate mid",
                map_="Pearl",
                n_results=3,
            )

        assert chunks == []
        assert sources == []

    async def test_n_results_limits_chunks(self, tmp_path: Path):
        from valocoach.retrieval.retriever import _fetch_live_meta

        long_cached = "x" * 3000  # longer than the 2000-char slice
        with patch(_GET_CACHED, new=AsyncMock(return_value=long_cached)):
            chunks, _ = await _fetch_live_meta(
                settings=_fake_settings(tmp_path),
                situation="situation",
                map_="Fracture",
                n_results=1,
            )

        assert len(chunks) <= 1
        if chunks:
            # Text is sliced to 2000 chars
            assert len(chunks[0]) <= 2000

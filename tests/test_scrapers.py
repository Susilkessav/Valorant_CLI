"""Tests for retrieval scrapers — web (scrape_url) and YouTube (fetch_transcript).

Covers:
  - ScrapedContent dataclass fields
  - scrape_url: success with trafilatura, BeautifulSoup fallback,
    HTTP error → None, text-too-short → None, title slug fallback
  - _extract_video_id: valid URL formats, bare ID, invalid → ValueError
  - fetch_transcript: success, invalid URL, API exception, short transcript

Patch targets
-------------
  trafilatura.extract / trafilatura.extract_metadata — module-level import in web.py
  youtube_transcript_api.YouTubeTranscriptApi — lazy import inside fetch_transcript
  httpx_mock (pytest-httpx fixture) — intercepts synchronous httpx.get
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# ScrapedContent dataclass
# ---------------------------------------------------------------------------


class TestScrapedContent:
    def test_all_fields_accessible(self):
        from valocoach.retrieval.scrapers import ScrapedContent

        sc = ScrapedContent(
            url="https://example.com",
            title="Example Title",
            text="Some article text",
            fetched_at="2026-01-01T00:00:00+00:00",
            source="web",
        )
        assert sc.url == "https://example.com"
        assert sc.title == "Example Title"
        assert sc.text == "Some article text"
        assert sc.fetched_at == "2026-01-01T00:00:00+00:00"
        assert sc.source == "web"


# ---------------------------------------------------------------------------
# scrape_url — valocoach.retrieval.scrapers.web
# ---------------------------------------------------------------------------

_TRAFILATURA_EXTRACT = "trafilatura.extract"
_TRAFILATURA_META = "trafilatura.extract_metadata"


class TestScrapeUrl:
    """Uses pytest-httpx (httpx_mock) to intercept synchronous httpx.get."""

    _LONG_TEXT = "Valorant tactical tips and strategies. " * 10  # well over 100 chars

    def test_success_returns_scraped_content(self, httpx_mock):
        from valocoach.retrieval.scrapers.web import scrape_url

        html = "<html><body><article>content</article></body></html>"
        httpx_mock.add_response(url="https://valorant.com/patch", text=html)

        meta = MagicMock()
        meta.title = "Patch Notes"
        with (
            patch(_TRAFILATURA_EXTRACT, return_value=self._LONG_TEXT),
            patch(_TRAFILATURA_META, return_value=meta),
        ):
            result = scrape_url("https://valorant.com/patch", source="patch_note")

        assert result is not None
        assert result.title == "Patch Notes"
        assert result.url == "https://valorant.com/patch"
        assert result.source == "patch_note"
        assert self._LONG_TEXT in result.text
        assert result.fetched_at  # non-empty ISO timestamp

    def test_http_error_returns_none(self, httpx_mock):
        from valocoach.retrieval.scrapers.web import scrape_url

        httpx_mock.add_response(url="https://example.com/missing", status_code=404)
        result = scrape_url("https://example.com/missing")
        assert result is None

    def test_text_too_short_returns_none(self, httpx_mock):
        from valocoach.retrieval.scrapers.web import scrape_url

        html = "<html><body><p>tiny</p></body></html>"
        httpx_mock.add_response(url="https://example.com/short", text=html)

        with patch(_TRAFILATURA_EXTRACT, return_value="too short"):
            result = scrape_url("https://example.com/short")

        assert result is None

    def test_trafilatura_none_falls_back_to_beautifulsoup(self, httpx_mock):
        from valocoach.retrieval.scrapers.web import scrape_url

        # Body text long enough to pass the 100-char threshold after BS4 strips tags
        long_body = "BeautifulSoup fallback text. " * 10
        html = f"<html><body><p>{long_body}</p></body></html>"
        httpx_mock.add_response(url="https://example.com/fallback", text=html)

        with (
            patch(_TRAFILATURA_EXTRACT, return_value=None),
            patch(_TRAFILATURA_META, return_value=None),
        ):
            result = scrape_url("https://example.com/fallback")

        assert result is not None
        assert "BeautifulSoup fallback text" in result.text

    def test_title_uses_url_slug_when_no_metadata(self, httpx_mock):
        from valocoach.retrieval.scrapers.web import scrape_url

        html = "<html><body><p>article content</p></body></html>"
        httpx_mock.add_response(url="https://example.com/patch-notes", text=html)

        meta = MagicMock()
        meta.title = None  # simulate missing title in metadata
        with (
            patch(_TRAFILATURA_EXTRACT, return_value=self._LONG_TEXT),
            patch(_TRAFILATURA_META, return_value=meta),
        ):
            result = scrape_url("https://example.com/patch-notes")

        assert result is not None
        # URL slug "patch-notes" becomes "Patch Notes"
        assert "Patch Notes" in result.title

    def test_title_uses_url_slug_when_metadata_is_none(self, httpx_mock):
        from valocoach.retrieval.scrapers.web import scrape_url

        html = "<html><body><p>article content</p></body></html>"
        httpx_mock.add_response(url="https://example.com/valorant-economy", text=html)

        with (
            patch(_TRAFILATURA_EXTRACT, return_value=self._LONG_TEXT),
            patch(_TRAFILATURA_META, return_value=None),
        ):
            result = scrape_url("https://example.com/valorant-economy")

        assert result is not None
        assert "Valorant Economy" in result.title

    def test_empty_text_after_beautifulsoup_returns_none(self, httpx_mock):
        from valocoach.retrieval.scrapers.web import scrape_url

        # Pure script/style — BS4 strips everything, leaves nothing useful
        html = "<html><head><script>alert(1)</script></head><body></body></html>"
        httpx_mock.add_response(url="https://example.com/empty", text=html)

        with patch(_TRAFILATURA_EXTRACT, return_value=None):
            result = scrape_url("https://example.com/empty")

        assert result is None


# ---------------------------------------------------------------------------
# _extract_video_id — internal parser
# ---------------------------------------------------------------------------


class TestExtractVideoId:
    def test_standard_watch_url(self):
        from valocoach.retrieval.scrapers.youtube import _extract_video_id

        assert _extract_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ") == "dQw4w9WgXcQ"

    def test_youtu_be_short_url(self):
        from valocoach.retrieval.scrapers.youtube import _extract_video_id

        assert _extract_video_id("https://youtu.be/dQw4w9WgXcQ") == "dQw4w9WgXcQ"

    def test_embed_url(self):
        from valocoach.retrieval.scrapers.youtube import _extract_video_id

        assert _extract_video_id("https://www.youtube.com/embed/dQw4w9WgXcQ") == "dQw4w9WgXcQ"

    def test_shorts_url(self):
        from valocoach.retrieval.scrapers.youtube import _extract_video_id

        assert _extract_video_id("https://www.youtube.com/shorts/dQw4w9WgXcQ") == "dQw4w9WgXcQ"

    def test_bare_video_id(self):
        from valocoach.retrieval.scrapers.youtube import _extract_video_id

        assert _extract_video_id("dQw4w9WgXcQ") == "dQw4w9WgXcQ"

    def test_invalid_url_raises_value_error(self):
        from valocoach.retrieval.scrapers.youtube import _extract_video_id

        with pytest.raises(ValueError, match="Could not extract video ID"):
            _extract_video_id("not-a-valid-url")

    def test_too_short_id_raises_value_error(self):
        from valocoach.retrieval.scrapers.youtube import _extract_video_id

        with pytest.raises(ValueError):
            _extract_video_id("abc123")  # 6 chars, not 11


# ---------------------------------------------------------------------------
# fetch_transcript — youtube scraper
# ---------------------------------------------------------------------------

_YT_API_CLS = "youtube_transcript_api.YouTubeTranscriptApi"


def _make_snippet(text: str) -> MagicMock:
    """Return a mock FetchedTranscriptSnippet with a .text attribute."""
    snippet = MagicMock()
    snippet.text = text
    return snippet


class TestFetchTranscript:
    # Enough text to pass the 200-char minimum
    _LONG_TEXT = "Valorant tactical coaching tips and tricks. " * 10

    def _mock_api(self, entries=None, exc=None):
        """Return a (mock_class, mock_instance) pair for patching YouTubeTranscriptApi."""
        mock_cls = MagicMock()
        mock_instance = MagicMock()
        mock_cls.return_value = mock_instance
        if exc is not None:
            mock_instance.fetch.side_effect = exc
        else:
            mock_instance.fetch.return_value = entries or []
        return mock_cls

    def _entries(self, text: str | None = None) -> list[MagicMock]:
        t = text or self._LONG_TEXT
        return [_make_snippet(t)]

    def test_success_returns_scraped_content(self):
        from valocoach.retrieval.scrapers.youtube import fetch_transcript

        mock_cls = self._mock_api(entries=self._entries())
        with patch(_YT_API_CLS, mock_cls):
            result = fetch_transcript("dQw4w9WgXcQ")

        assert result is not None
        assert "dQw4w9WgXcQ" in result.url
        assert result.source == "youtube"

    def test_full_url_accepted(self):
        from valocoach.retrieval.scrapers.youtube import fetch_transcript

        mock_cls = self._mock_api(entries=self._entries())
        with patch(_YT_API_CLS, mock_cls):
            result = fetch_transcript("https://www.youtube.com/watch?v=dQw4w9WgXcQ")

        assert result is not None
        assert "dQw4w9WgXcQ" in result.url

    def test_invalid_url_returns_none(self):
        from valocoach.retrieval.scrapers.youtube import fetch_transcript

        result = fetch_transcript("not-a-valid-url-or-id!!")
        assert result is None

    def test_api_exception_returns_none(self):
        from valocoach.retrieval.scrapers.youtube import fetch_transcript

        mock_cls = self._mock_api(exc=Exception("No transcript available"))
        with patch(_YT_API_CLS, mock_cls):
            result = fetch_transcript("dQw4w9WgXcQ")

        assert result is None

    def test_transcript_too_short_returns_none(self):
        from valocoach.retrieval.scrapers.youtube import fetch_transcript

        mock_cls = self._mock_api(entries=[_make_snippet("Too short.")])
        with patch(_YT_API_CLS, mock_cls):
            result = fetch_transcript("dQw4w9WgXcQ")

        assert result is None

    def test_fetched_at_is_set(self):
        from valocoach.retrieval.scrapers.youtube import fetch_transcript

        mock_cls = self._mock_api(entries=self._entries())
        with patch(_YT_API_CLS, mock_cls):
            result = fetch_transcript("dQw4w9WgXcQ")

        assert result is not None
        assert result.fetched_at  # non-empty ISO timestamp

    def test_title_contains_video_id(self):
        from valocoach.retrieval.scrapers.youtube import fetch_transcript

        mock_cls = self._mock_api(entries=self._entries())
        with patch(_YT_API_CLS, mock_cls):
            result = fetch_transcript("dQw4w9WgXcQ")

        assert result is not None
        assert "dQw4w9WgXcQ" in result.title

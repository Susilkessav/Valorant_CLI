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

    def test_title_uses_oembed_title(self):
        """D1: title should come from oEmbed, not a placeholder."""
        from valocoach.retrieval.scrapers.youtube import fetch_transcript

        mock_cls = self._mock_api(entries=self._entries())
        with (
            patch(_YT_API_CLS, mock_cls),
            patch(
                "valocoach.retrieval.scrapers.youtube.fetch_video_metadata",
                return_value={"title": "Haven A Execute Guide", "channel": "Woohoojin"},
            ),
        ):
            result = fetch_transcript("dQw4w9WgXcQ")

        assert result is not None
        assert result.title == "Haven A Execute Guide"


# ---------------------------------------------------------------------------
# meta_stats — fetch_ranked_stats / fetch_pro_stats / fetch_all_stats
# ---------------------------------------------------------------------------

_SCRAPE_URL = "valocoach.retrieval.scrapers.meta_stats.scrape_url"


class TestMetaStats:
    """Tests for meta_stats scrapers — scrape_url is mocked throughout."""

    _LONG_TEXT = "Agent pick-rate data from tracker.gg. " * 10

    def _scraped(self, text: str | None = None):
        from valocoach.retrieval.scrapers import ScrapedContent

        if text is None:
            text = self._LONG_TEXT
        sc = MagicMock(spec=ScrapedContent)
        sc.text = text
        return sc

    def test_fetch_ranked_stats_returns_text_on_success(self):
        from valocoach.retrieval.scrapers.meta_stats import fetch_ranked_stats

        with patch(_SCRAPE_URL, return_value=self._scraped()):
            result = fetch_ranked_stats()

        assert result == self._LONG_TEXT

    def test_fetch_ranked_stats_returns_empty_string_on_failure(self):
        from valocoach.retrieval.scrapers.meta_stats import fetch_ranked_stats

        with patch(_SCRAPE_URL, return_value=None):
            result = fetch_ranked_stats()

        assert result == ""

    def test_fetch_pro_stats_returns_text_on_success(self):
        from valocoach.retrieval.scrapers.meta_stats import fetch_pro_stats

        with patch(_SCRAPE_URL, return_value=self._scraped()):
            result = fetch_pro_stats()

        assert result == self._LONG_TEXT

    def test_fetch_pro_stats_returns_empty_string_on_failure(self):
        from valocoach.retrieval.scrapers.meta_stats import fetch_pro_stats

        with patch(_SCRAPE_URL, return_value=None):
            result = fetch_pro_stats()

        assert result == ""

    def test_fetch_all_stats_combines_both(self):
        from valocoach.retrieval.scrapers.meta_stats import fetch_all_stats

        ranked_text = "ranked stats data " * 10
        pro_text = "pro stats data " * 10
        side_effects = [self._scraped(ranked_text), self._scraped(pro_text)]

        with patch(_SCRAPE_URL, side_effect=side_effects):
            result = fetch_all_stats()

        assert result.ranked_text == ranked_text
        assert result.pro_text == pro_text
        assert result.ok is True

    def test_fetch_all_stats_ok_when_one_source_fails(self):
        from valocoach.retrieval.scrapers.meta_stats import fetch_all_stats

        side_effects = [None, self._scraped("pro data " * 10)]
        with patch(_SCRAPE_URL, side_effect=side_effects):
            result = fetch_all_stats()

        assert result.ranked_text == ""
        assert result.ok is True  # still ok — pro source succeeded

    def test_fetch_all_stats_not_ok_when_both_fail(self):
        from valocoach.retrieval.scrapers.meta_stats import fetch_all_stats

        with patch(_SCRAPE_URL, return_value=None):
            result = fetch_all_stats()

        assert result.ok is False

    def test_meta_stats_result_combined_includes_headers(self):
        from valocoach.retrieval.scrapers.meta_stats import MetaStatsResult

        r = MetaStatsResult(ranked_text="ranked", pro_text="pro")
        combined = r.combined
        assert "dak.gg" in combined   # header updated: now shows dak.gg / blitz.gg
        assert "vlr.gg" in combined
        assert "ranked" in combined
        assert "pro" in combined

    def test_meta_stats_result_combined_empty_when_both_empty(self):
        from valocoach.retrieval.scrapers.meta_stats import MetaStatsResult

        r = MetaStatsResult(ranked_text="", pro_text="")
        assert r.combined == ""
        assert r.ok is False


# ---------------------------------------------------------------------------
# patch_notes — parse_version / build_patch_notes_url / fetch_patch_notes
# ---------------------------------------------------------------------------

_PATCH_SCRAPE_URL = "valocoach.retrieval.scrapers.patch_notes.scrape_url"


class TestPatchNotes:
    """Tests for the patch_notes scraper module."""

    def test_parse_version_release_string(self):
        from valocoach.retrieval.scrapers.patch_notes import parse_version

        assert parse_version("release-10.08-shipping-32-1234567") == ("10", "08")

    def test_parse_version_clean_string(self):
        from valocoach.retrieval.scrapers.patch_notes import parse_version

        assert parse_version("10.09") == ("10", "09")

    def test_parse_version_returns_none_on_garbage(self):
        from valocoach.retrieval.scrapers.patch_notes import parse_version

        assert parse_version("garbage") is None

    def test_build_patch_notes_url_success(self):
        from valocoach.retrieval.scrapers.patch_notes import build_patch_notes_url

        url = build_patch_notes_url("10.08")
        assert url is not None
        assert "10-08" in url
        assert "playvalorant.com" in url

    def test_build_patch_notes_url_returns_none_for_bad_version(self):
        from valocoach.retrieval.scrapers.patch_notes import build_patch_notes_url

        assert build_patch_notes_url("garbage") is None

    def test_fetch_patch_notes_returns_content_on_success(self):
        from valocoach.retrieval.scrapers import ScrapedContent
        from valocoach.retrieval.scrapers.patch_notes import fetch_patch_notes

        fake = MagicMock(spec=ScrapedContent)
        fake.text = "Patch 10.09 notes..."
        with patch(_PATCH_SCRAPE_URL, return_value=fake):
            result = fetch_patch_notes("10.09")

        assert result is fake

    def test_fetch_patch_notes_returns_none_when_scrape_fails(self):
        from valocoach.retrieval.scrapers.patch_notes import fetch_patch_notes

        with patch(_PATCH_SCRAPE_URL, return_value=None):
            result = fetch_patch_notes("10.09")

        assert result is None

    def test_fetch_patch_notes_returns_none_for_unparseable_version(self):
        from valocoach.retrieval.scrapers.patch_notes import fetch_patch_notes

        # No mock needed — build_patch_notes_url returns None → early return
        result = fetch_patch_notes("garbage-version")
        assert result is None


# ---------------------------------------------------------------------------
# F4 — Patch notes multi-source fallback chain
# ---------------------------------------------------------------------------


class TestFetchPatchNotesFallback:
    """F4 — verify that fetch_patch_notes() falls back through the source chain."""

    def test_returns_primary_when_playvalorant_succeeds(self):
        """When playvalorant.com returns content, no fallback is called."""
        from valocoach.retrieval.scrapers import ScrapedContent
        from valocoach.retrieval.scrapers.patch_notes import fetch_patch_notes

        primary_content = MagicMock(spec=ScrapedContent)
        primary_content.text = "Official patch notes text"

        call_log: list[str] = []

        def fake_scrape(url: str, **kw) -> MagicMock | None:
            call_log.append(url)
            if "playvalorant.com" in url:
                return primary_content
            return None

        with patch(_PATCH_SCRAPE_URL, side_effect=fake_scrape):
            result = fetch_patch_notes("10.09")

        assert result is primary_content
        # Only the primary URL should have been tried
        assert len([u for u in call_log if "playvalorant.com" in u]) == 1
        assert not any("liquipedia" in u for u in call_log)

    def test_falls_back_to_liquipedia_when_primary_fails(self):
        """When playvalorant.com returns None, liquipedia.net is tried next."""
        from valocoach.retrieval.scrapers import ScrapedContent
        from valocoach.retrieval.scrapers.patch_notes import fetch_patch_notes

        liquipedia_content = MagicMock(spec=ScrapedContent)
        # Must be ≥500 chars to pass the JS-blank-page check in _fetch_liquipedia
        liquipedia_content.text = "Liquipedia patch summary. " * 25

        def fake_scrape(url: str, **kw) -> MagicMock | None:
            if "playvalorant.com" in url:
                return None  # primary fails
            if "liquipedia.net" in url:
                return liquipedia_content
            return None

        with patch(_PATCH_SCRAPE_URL, side_effect=fake_scrape):
            result = fetch_patch_notes("10.09")

        assert result is liquipedia_content

    def test_falls_back_to_reddit_when_primary_and_liquipedia_fail(self):
        """When both primary and liquipedia fail, Reddit search is tried."""
        from valocoach.retrieval.scrapers import ScrapedContent
        from valocoach.retrieval.scrapers.patch_notes import fetch_patch_notes

        reddit_content = MagicMock(spec=ScrapedContent)
        reddit_content.text = "Reddit post with patch notes link"

        # Reddit fallback uses urlopen then scrape_url on the post URL
        import json as _json

        # Use a non-playvalorant, non-liquipedia URL so fake_scrape can distinguish it
        _ARTICLE_URL = "https://valorant.fandom.com/wiki/Patch_10.09"
        fake_reddit_response = _json.dumps({
            "data": {
                "children": [
                    {
                        "data": {
                            "title": "VALORANT Patch Notes 10.09",
                            "url": _ARTICLE_URL,
                            "is_self": False,
                            "permalink": "/r/VALORANT/comments/abc/patch_10_09/",
                        }
                    }
                ]
            }
        }).encode()

        mock_resp = MagicMock()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.read.return_value = fake_reddit_response

        def fake_scrape(url: str, **kw) -> MagicMock | None:
            if "playvalorant.com" in url or "liquipedia.net" in url:
                return None  # primary and fallback 1 fail
            # Any other URL (the linked article from Reddit) returns content
            return reddit_content

        with (
            patch(_PATCH_SCRAPE_URL, side_effect=fake_scrape),
            patch(
                "valocoach.retrieval.scrapers.patch_notes.urlopen",
                return_value=mock_resp,
            ),
        ):
            result = fetch_patch_notes("10.09")

        assert result is reddit_content


# ---------------------------------------------------------------------------
# F3 — Provenance tags on context blocks
# ---------------------------------------------------------------------------


class TestProvenanceTags:
    """F3 — every context formatter must end with a [SOURCE: ...] tag."""

    def test_agent_context_has_source_tag(self):
        from valocoach.retrieval.agents import format_agent_context

        result = format_agent_context("Jett")
        assert result is not None
        assert result.endswith("]")
        assert "[SOURCE: knowledge_base/agents/Jett]" in result

    def test_map_context_has_source_tag(self):
        from valocoach.retrieval.maps import format_map_context

        result = format_map_context("Ascent")
        assert result is not None
        assert "[SOURCE: knowledge_base/maps/Ascent]" in result

    def test_meta_context_has_source_tag(self):
        from valocoach.retrieval.meta import format_meta_context

        result = format_meta_context()
        assert result is not None
        # Patch version varies — just assert the prefix is present
        assert "[SOURCE: knowledge_base/meta/" in result

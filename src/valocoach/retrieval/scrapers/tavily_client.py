"""Tavily-backed web search and URL extraction.

Tavily is an AI-powered search/scrape API that handles JavaScript-rendered
pages — something trafilatura + BeautifulSoup cannot do reliably.  It is
used by the meta-stats and patch-notes scrapers as a preferred (or fallback)
source when a Tavily API key is configured.

Usage in the pipeline
---------------------
Every public function returns a :class:`~valocoach.retrieval.scrapers.ScrapedContent`
or ``None`` so callers can treat it identically to the existing ``scrape_url``
helper with no interface changes.

Activation
----------
Tavily is **opt-in** — if ``settings.tavily_api_key`` is empty, every call
returns ``None`` immediately and the callers fall through to the trafilatura
path.  No API key → no behaviour change from the user's perspective.

Credit costs (Tavily pricing as of 2025)
-----------------------------------------
- Basic search:    1 credit   (~1 000 free/month on the free tier)
- Advanced search: 2 credits  (deeper crawl, better JS rendering)
- Basic extract:   1 credit
- Advanced extract: 2 credits

Typical ValoCoach usage per ``meta-refresh`` run:
  ~2 searches (ranked stats + pro stats) + 1 search (patch notes fallback)
  = ~3-4 credits per refresh -- well within the free tier.

Public API
----------
    is_configured(settings)           → bool
    search(query, settings, ...)      → ScrapedContent | None
    extract(url, settings)            → ScrapedContent | None
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from valocoach.core.config import Settings

from valocoach.retrieval.scrapers import ScrapedContent

log = logging.getLogger(__name__)


def is_configured(settings: Settings) -> bool:
    """Return True when a Tavily API key is present in settings."""
    return bool(getattr(settings, "tavily_api_key", ""))


def _client(settings: Settings):
    """Return a TavilyClient, or raise ImportError / ValueError on misconfiguration."""
    try:
        from tavily import TavilyClient  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "tavily-python is not installed. Run: uv add tavily-python"
        ) from exc

    api_key = getattr(settings, "tavily_api_key", "")
    if not api_key:
        raise ValueError("tavily_api_key is not configured")

    return TavilyClient(api_key=api_key)


def search(
    query: str,
    settings: Settings,
    *,
    search_depth: str = "advanced",
    max_results: int = 3,
    include_domains: list[str] | None = None,
    topic: str = "general",
    source: str = "web",
) -> ScrapedContent | None:
    """Run a Tavily web search and return the best result as :class:`ScrapedContent`.

    The full ``raw_content`` of the top-scoring result is used as the text so
    the LLM receives the complete page body, not a short AI-generated snippet.
    Falls back to ``content`` (snippet) if ``raw_content`` is absent.

    Args:
        query:          Free-text search query.
        settings:       App settings (API key read from here).
        search_depth:   ``"basic"`` (1 credit) or ``"advanced"`` (2 credits).
                        Advanced triggers a deeper headless-browser crawl,
                        which is necessary for JS-rendered pages like
                        tracker.gg and vlr.gg.
        max_results:    Number of results to retrieve.  The highest-scored
                        result with non-empty ``raw_content`` is returned.
        include_domains: Optional domain whitelist, e.g. ``["tracker.gg"]``.
        topic:          Tavily topic filter: ``"general"`` or ``"news"``.
        source:         ``ScrapedContent.source`` tag for the returned object.

    Returns:
        :class:`ScrapedContent` for the best result, or ``None`` on any failure.
    """
    if not is_configured(settings):
        return None

    try:
        client = _client(settings)
    except (ImportError, ValueError) as exc:
        log.debug("Tavily client unavailable: %s", exc)
        return None

    try:
        kwargs: dict = {
            "query": query,
            "search_depth": search_depth,
            "max_results": max_results,
            "include_raw_content": True,
            "topic": topic,
        }
        if include_domains:
            kwargs["include_domains"] = include_domains

        log.info("Tavily search: %r (depth=%s, domains=%s)", query, search_depth, include_domains)
        response = client.search(**kwargs)
    except Exception as exc:
        log.warning("Tavily search failed for %r: %s", query, exc)
        return None

    results = response.get("results") or []
    if not results:
        log.debug("Tavily: no results for %r", query)
        return None

    # Pick the result with the most raw content (JS-rendered pages may return
    # an empty raw_content on the first hit but not the second).
    best = max(
        results,
        key=lambda r: len(r.get("raw_content") or r.get("content") or ""),
    )
    text = best.get("raw_content") or best.get("content") or ""
    if not text or len(text) < 100:
        log.debug("Tavily: best result too short (%d chars) for %r", len(text), query)
        return None

    url   = best.get("url", query)
    title = best.get("title") or url.split("/")[-1].replace("-", " ").title()

    log.info("Tavily search: best result → %s (%d chars)", url, len(text))
    return ScrapedContent(
        url=url,
        title=title,
        text=text,
        fetched_at=datetime.now(tz=UTC).isoformat(),
        source=source,
    )


def extract(
    url: str,
    settings: Settings,
    *,
    extract_depth: str = "advanced",
    source: str = "web",
) -> ScrapedContent | None:
    """Extract clean text from a specific URL via Tavily's Extract API.

    Tavily uses a headless browser for extraction, so this works for
    JavaScript-rendered pages that trafilatura cannot handle.

    Args:
        url:            The URL to extract content from.
        settings:       App settings (API key read from here).
        extract_depth:  ``"basic"`` (1 credit) or ``"advanced"`` (2 credits).
        source:         ``ScrapedContent.source`` tag.

    Returns:
        :class:`ScrapedContent` with extracted text, or ``None`` on failure.
    """
    if not is_configured(settings):
        return None

    try:
        client = _client(settings)
    except (ImportError, ValueError) as exc:
        log.debug("Tavily client unavailable: %s", exc)
        return None

    try:
        log.info("Tavily extract: %s (depth=%s)", url, extract_depth)
        response = client.extract(urls=[url], extract_depth=extract_depth)
    except Exception as exc:
        log.warning("Tavily extract failed for %s: %s", url, exc)
        return None

    results = (response.get("results") or [])
    if not results:
        failed = response.get("failed_results") or []
        log.debug("Tavily extract: no content for %s (failed=%s)", url, failed)
        return None

    text = (results[0].get("raw_content") or "").strip()
    if not text or len(text) < 100:
        log.debug("Tavily extract: content too short (%d chars) for %s", len(text), url)
        return None

    title = url.split("/")[-1].replace("-", " ").title()
    log.info("Tavily extract: %s → %d chars", url, len(text))
    return ScrapedContent(
        url=url,
        title=title,
        text=text,
        fetched_at=datetime.now(tz=UTC).isoformat(),
        source=source,
    )


__all__ = ["extract", "is_configured", "search"]

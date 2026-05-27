from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import httpx
import trafilatura
from bs4 import BeautifulSoup

from valocoach.retrieval.scrapers import ScrapedContent

if TYPE_CHECKING:
    from valocoach.core.config import Settings

log = logging.getLogger(__name__)

USER_AGENT = "ValoCoachBot/1.0 (personal coaching tool)"

# Minimum character threshold for extracted content to be considered useful.
# Below this the page is almost certainly a JS-rendered shell with no text.
_MIN_CONTENT_CHARS = 100


def scrape_url(
    url: str,
    source: str = "web",
    timeout: int = 20,
    settings: "Settings | None" = None,
) -> ScrapedContent | None:
    """Fetch a URL and extract its main text content.

    Extraction pipeline (first path that returns ≥ 100 chars wins):

    1. **trafilatura** — fast, zero-cost, works on standard HTML pages.
    2. **BeautifulSoup strip** — removes nav/header/footer noise; catches
       lightly-rendered pages trafilatura misses.
    3. **Tavily Extract** *(optional)* — headless-browser extraction.
       Only attempted when ``settings.tavily_api_key`` is set AND both
       paths above returned too little content.  This handles fully
       JS-rendered SPAs (tracker.gg, vlr.gg, etc.) that trafilatura
       cannot reach.

    Args:
        url:      The URL to fetch.
        source:   ``ScrapedContent.source`` tag (e.g. ``"web"``, ``"patch_note"``).
        timeout:  HTTP request timeout in seconds.
        settings: App settings.  Pass to enable Tavily as the JS fallback.

    Returns:
        :class:`ScrapedContent` with extracted text, or ``None`` on failure.
    """
    try:
        resp = httpx.get(
            url,
            timeout=timeout,
            follow_redirects=True,
            headers={"User-Agent": USER_AGENT},
        )
        resp.raise_for_status()
    except httpx.HTTPError as exc:
        log.debug("HTTP error fetching %s: %s", url, exc)
        return None

    html = resp.text

    # --- Path 1: trafilatura ---
    text = trafilatura.extract(
        html,
        include_comments=False,
        include_tables=True,
        no_fallback=False,
    )

    # --- Path 2: BeautifulSoup strip ---
    if not text:
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)

    # --- Path 3: Tavily Extract (opt-in, JS-rendered pages) ---
    if (not text or len(text) < _MIN_CONTENT_CHARS) and settings is not None:
        from valocoach.retrieval.scrapers import tavily_client as tv

        if tv.is_configured(settings):
            log.debug(
                "scrape_url: trafilatura/BS4 yielded %d chars for %s — trying Tavily Extract",
                len(text or ""),
                url,
            )
            tavily_result = tv.extract(url, settings, extract_depth="advanced", source=source)
            if tavily_result is not None:
                return tavily_result

    if not text or len(text) < _MIN_CONTENT_CHARS:
        log.debug("Could not extract meaningful content from %s (%d chars)", url, len(text or ""))
        return None

    meta = trafilatura.extract_metadata(html)
    title = (meta.title if meta and meta.title else None) or url.split("/")[-1].replace(
        "-", " "
    ).title()

    return ScrapedContent(
        url=url,
        title=title,
        text=text,
        fetched_at=datetime.now(tz=UTC).isoformat(),
        source=source,
    )

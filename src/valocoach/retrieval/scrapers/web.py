from __future__ import annotations

import logging
from datetime import UTC, datetime

import httpx
import trafilatura
from bs4 import BeautifulSoup

from valocoach.retrieval.scrapers import ScrapedContent

log = logging.getLogger(__name__)

USER_AGENT = "ValoCoachBot/1.0 (personal coaching tool)"


def scrape_url(url: str, source: str = "web", timeout: int = 20) -> ScrapedContent | None:
    """Fetch a URL and extract its main text content.

    Uses trafilatura as the primary extractor and falls back to BeautifulSoup
    stripping when trafilatura returns nothing. Returns None on any failure
    so callers can handle it without try/except.
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
        log.warning("HTTP error fetching %s: %s", url, exc)
        return None

    html = resp.text

    text = trafilatura.extract(
        html,
        include_comments=False,
        include_tables=True,
        no_fallback=False,
    )

    if not text:
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)

    if not text or len(text) < 100:
        log.warning("Could not extract meaningful content from %s", url)
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

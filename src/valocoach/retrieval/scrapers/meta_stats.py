"""Scrape agent pick-rate and win-rate statistics.

Sources (in priority order)
----------------------------
When a Tavily API key is configured:
  1. Tavily search — finds the current stats page automatically, handles
     JavaScript rendering.  No hardcoded URLs means this keeps working
     even when tracker.gg or vlr.gg change their layout.

When Tavily is not configured (or Tavily search fails):
  2. trafilatura scrape of the hardcoded tracker.gg and vlr.gg URLs.
     This works on static / lightly-rendered pages but often returns empty
     content from tracker.gg because that page is fully JS-rendered.

Why Tavily wins here
--------------------
tracker.gg/valorant/insights/agents and vlr.gg/stats are both single-page
applications (React/Vue).  Trafilatura fetches the raw HTML and sees mostly
empty script tags — the actual pick/win-rate numbers are injected by JavaScript
at runtime.  Tavily's headless-browser backend renders the page before
extracting, so it reliably returns the numeric table content.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from valocoach.retrieval.scrapers.web import scrape_url

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hardcoded fallback URLs (used when Tavily is not configured)
# ---------------------------------------------------------------------------

_TRACKER_AGENTS_URL = "https://tracker.gg/valorant/insights/agents"
_VLR_STATS_URL = "https://www.vlr.gg/stats"

# ---------------------------------------------------------------------------
# Tavily search queries — deliberately broad so Tavily picks the best
# current source rather than being locked to a specific domain.
# ---------------------------------------------------------------------------

_TAVILY_RANKED_QUERY = (
    "Valorant agent pick rate win rate statistics Diamond Platinum Gold ranked 2025"
)
_TAVILY_PRO_QUERY = (
    "Valorant VCT pro agent pick rate win rate tournament statistics 2025"
)

# Domains we prefer for ranked and pro stats respectively.
# Tavily uses these as a soft hint — it still searches broadly but
# up-ranks pages from these domains.
_RANKED_DOMAINS = ["tracker.gg", "blitz.gg", "dak.gg", "valoranttracker.gg"]
_PRO_DOMAINS    = ["vlr.gg", "liquipedia.net", "thespike.gg"]


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class MetaStatsResult:
    """Container returned by :func:`fetch_all_stats`."""

    ranked_text: str
    """Raw text extracted from the ranked stats source (empty on failure)."""

    pro_text: str
    """Raw text extracted from the pro/VCT stats source (empty on failure)."""

    ranked_source: str = "unknown"
    """Which scraping path produced ranked_text — ``"tavily"`` or ``"trafilatura"``."""

    pro_source: str = "unknown"
    """Which scraping path produced pro_text — ``"tavily"`` or ``"trafilatura"``."""

    @property
    def combined(self) -> str:
        """Both sources concatenated with labelled headers for LLM context."""
        parts: list[str] = []
        if self.ranked_text:
            parts.append("=== DIAMOND+ RANKED STATS (tracker.gg) ===\n" + self.ranked_text)
        if self.pro_text:
            parts.append("=== PRO / VCT STATS (vlr.gg) ===\n" + self.pro_text)
        return "\n\n".join(parts)

    @property
    def ok(self) -> bool:
        """True when at least one source returned data."""
        return bool(self.ranked_text or self.pro_text)


# ---------------------------------------------------------------------------
# Internal fetch helpers
# ---------------------------------------------------------------------------


def _fetch_ranked_tavily(settings) -> str:
    """Search for Diamond+ ranked stats via Tavily.

    Returns the extracted text, or an empty string on failure / not configured.
    """
    from valocoach.retrieval.scrapers import tavily_client as tv

    if not tv.is_configured(settings):
        return ""

    result = tv.search(
        _TAVILY_RANKED_QUERY,
        settings,
        search_depth="advanced",
        max_results=3,
        include_domains=_RANKED_DOMAINS,
        source="meta_stats",
    )
    if result:
        log.info("Ranked stats: Tavily returned %d chars from %s", len(result.text), result.url)
        return result.text
    return ""


def _fetch_pro_tavily(settings) -> str:
    """Search for pro/VCT stats via Tavily."""
    from valocoach.retrieval.scrapers import tavily_client as tv

    if not tv.is_configured(settings):
        return ""

    result = tv.search(
        _TAVILY_PRO_QUERY,
        settings,
        search_depth="advanced",
        max_results=3,
        include_domains=_PRO_DOMAINS,
        source="meta_stats",
    )
    if result:
        log.info("Pro stats: Tavily returned %d chars from %s", len(result.text), result.url)
        return result.text
    return ""


def _fetch_ranked_trafilatura() -> str:
    """Scrape tracker.gg ranked stats via trafilatura (fallback path).

    Returns empty string when the page is JS-rendered and trafilatura
    returns nothing useful.
    """
    content = scrape_url(_TRACKER_AGENTS_URL, source="meta_stats")
    if content is None:
        log.warning("Could not fetch ranked stats from tracker.gg (trafilatura)")
        return ""
    if len(content.text) < 200:
        log.warning(
            "tracker.gg returned only %d chars — page is likely JS-rendered. "
            "Set tavily_api_key in config for reliable stats scraping.",
            len(content.text),
        )
        return ""
    log.info("Ranked stats via trafilatura: %d chars", len(content.text))
    return content.text


def _fetch_pro_trafilatura() -> str:
    """Scrape vlr.gg pro stats via trafilatura (fallback path)."""
    content = scrape_url(_VLR_STATS_URL, source="meta_stats")
    if content is None:
        log.warning("Could not fetch pro stats from vlr.gg (trafilatura)")
        return ""
    if len(content.text) < 200:
        log.warning(
            "vlr.gg returned only %d chars — page is likely JS-rendered. "
            "Set tavily_api_key in config for reliable stats scraping.",
            len(content.text),
        )
        return ""
    log.info("Pro stats via trafilatura: %d chars", len(content.text))
    return content.text


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def fetch_ranked_stats(settings=None) -> str:
    """Fetch Diamond+ ranked agent stats.

    Tries Tavily search first (when configured), falls back to trafilatura.

    Returns:
        Extracted page text, or an empty string on failure.
    """
    if settings is not None:
        text = _fetch_ranked_tavily(settings)
        if text:
            return text
        # Tavily not configured or failed — fall through.

    return _fetch_ranked_trafilatura()


def fetch_pro_stats(settings=None) -> str:
    """Fetch pro/VCT agent stats.

    Tries Tavily search first (when configured), falls back to trafilatura.

    Returns:
        Extracted page text, or an empty string on failure.
    """
    if settings is not None:
        text = _fetch_pro_tavily(settings)
        if text:
            return text

    return _fetch_pro_trafilatura()


def fetch_all_stats(settings=None) -> MetaStatsResult:
    """Fetch and combine Diamond+ ranked and pro/VCT stats.

    Both fetches are attempted independently so a failure on one does not
    block the other.  Pass ``settings`` to enable the Tavily path; omit it
    (or pass ``None``) to use trafilatura only.

    Args:
        settings: App settings.  When ``settings.tavily_api_key`` is set,
                  Tavily search is used.  When absent or empty, the
                  trafilatura scraper runs instead.

    Returns:
        :class:`MetaStatsResult` with ``ok=True`` when at least one
        source returned data.
    """
    from valocoach.retrieval.scrapers import tavily_client as tv

    use_tavily = settings is not None and tv.is_configured(settings)

    ranked_text = fetch_ranked_stats(settings)
    ranked_src  = "tavily" if (use_tavily and ranked_text) else "trafilatura"

    pro_text = fetch_pro_stats(settings)
    pro_src  = "tavily" if (use_tavily and pro_text) else "trafilatura"

    result = MetaStatsResult(
        ranked_text=ranked_text,
        pro_text=pro_text,
        ranked_source=ranked_src,
        pro_source=pro_src,
    )
    log.info(
        "Meta stats: ranked=%s(%d chars) pro=%s(%d chars)",
        ranked_src, len(ranked_text),
        pro_src, len(pro_text),
    )
    return result

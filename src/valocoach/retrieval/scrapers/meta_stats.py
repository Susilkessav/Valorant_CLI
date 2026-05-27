"""Scrape agent pick-rate and win-rate statistics.

Sources (in priority order)
----------------------------
Ranked stats (Diamond+ / all-rank play):
  1. dak.gg/valorant/statistics/agents  — Tavily Extract (clean table: pick%/win%)
  2. blitz.gg/valorant/stats/agents     — Tavily Extract (fallback ranked source)
  3. Tavily search                       — broad query, finds best current source
  4. trafilatura scrape of tracker.gg   — usually empty (JS-rendered, Cloudflare)

Pro / VCT stats:
  1. Tavily search → vlr.gg VCT Champions event agents page
     (returns 50 k+ chars with per-map pick rates per agent)
  2. trafilatura scrape of vlr.gg/stats — usually empty (JS-rendered)

Why dak.gg/blitz.gg instead of tracker.gg
------------------------------------------
tracker.gg is behind Cloudflare — even Tavily's headless browser is blocked.
dak.gg and blitz.gg expose the same ranked pick/win-rate tables and are
accessible to Tavily Extract.  Live tests confirm dak.gg returns rows like:

    | Jett   | 13.1% | 50.7% | … |
    | Chamber| 12.6% | 50.1% | … |
    | Clove  |  8.3% | 53.3% | … |

These exact numbers are what the LLM meta generator needs to assign S/A/B/C tiers.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from valocoach.retrieval.scrapers.web import scrape_url

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# URLs
# ---------------------------------------------------------------------------

# Ranked stats — direct extract targets (Tavily can handle these)
_DAK_GG_URL    = "https://dak.gg/valorant/statistics/agents"
_BLITZ_GG_URL  = "https://blitz.gg/valorant/stats/agents"

# Trafilatura fallback (almost always empty due to JS rendering / Cloudflare)
_TRACKER_AGENTS_URL = "https://tracker.gg/valorant/insights/agents"
_VLR_STATS_URL      = "https://www.vlr.gg/stats"

# Pro stats — Tavily search query (returns VCT event agent pick pages)
_TAVILY_PRO_QUERY = (
    "Valorant VCT pro agent pick rate win rate tournament statistics 2025"
)
_PRO_DOMAINS = ["vlr.gg", "liquipedia.net", "thespike.gg"]

# Ranked fallback search query (used when dak.gg and blitz.gg both fail)
_TAVILY_RANKED_FALLBACK_QUERY = (
    "Valorant ranked agent pick rate win rate statistics 2025"
)
_RANKED_DOMAINS = ["dak.gg", "blitz.gg", "valoranttracker.gg"]


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
            parts.append("=== RANKED STATS (dak.gg / blitz.gg) ===\n" + self.ranked_text)
        if self.pro_text:
            parts.append("=== PRO / VCT STATS (vlr.gg) ===\n" + self.pro_text)
        return "\n\n".join(parts)

    @property
    def ok(self) -> bool:
        """True when at least one source returned data."""
        return bool(self.ranked_text or self.pro_text)


# ---------------------------------------------------------------------------
# Internal Tavily fetch helpers
# ---------------------------------------------------------------------------


def _fetch_ranked_tavily(settings) -> str:
    """Fetch ranked agent stats via Tavily.

    Tries direct Extract on dak.gg → blitz.gg → broad search fallback.
    Returns extracted text, or empty string on failure / not configured.
    """
    from valocoach.retrieval.scrapers import tavily_client as tv

    if not tv.is_configured(settings):
        return ""

    # --- dak.gg: clean table with Pickrate / Winrate columns
    result = tv.extract(_DAK_GG_URL, settings, extract_depth="advanced", source="meta_stats")
    if result and len(result.text) >= 500:
        log.info("Ranked stats: dak.gg Extract → %d chars", len(result.text))
        return result.text

    # --- blitz.gg fallback
    result = tv.extract(_BLITZ_GG_URL, settings, extract_depth="advanced", source="meta_stats")
    if result and len(result.text) >= 500:
        log.info("Ranked stats: blitz.gg Extract → %d chars", len(result.text))
        return result.text

    # --- broad search fallback (lets Tavily find the best current source)
    result = tv.search(
        _TAVILY_RANKED_FALLBACK_QUERY,
        settings,
        search_depth="advanced",
        max_results=3,
        include_domains=_RANKED_DOMAINS,
        source="meta_stats",
    )
    if result:
        log.info("Ranked stats: Tavily search → %s (%d chars)", result.url, len(result.text))
        return result.text

    return ""


def _fetch_pro_tavily(settings) -> str:
    """Fetch pro/VCT stats via Tavily search.

    Returns text of the best matching VCT event agent page, or empty string.
    """
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
        log.info("Pro stats: Tavily search → %s (%d chars)", result.url, len(result.text))
        return result.text

    return ""


# ---------------------------------------------------------------------------
# Trafilatura fallbacks (run when Tavily is not configured or all fail)
# ---------------------------------------------------------------------------


def _fetch_ranked_trafilatura() -> str:
    content = scrape_url(_TRACKER_AGENTS_URL, source="meta_stats")
    if content is None:
        log.warning(
            "Ranked stats: tracker.gg returned nothing (JS-rendered/Cloudflare). "
            "Set tavily_api_key in config for reliable stats."
        )
        return ""
    if len(content.text) < 500:
        # Warn but don't reject — short text usually means a JS-rendered shell,
        # but we let it through so the caller can decide.  scrape_url already
        # enforces a 100-char hard floor before returning.
        log.warning(
            "Ranked stats: tracker.gg returned only %d chars — likely JS-blocked. "
            "Set tavily_api_key in config for reliable stats.",
            len(content.text),
        )
    log.info("Ranked stats: trafilatura tracker.gg → %d chars", len(content.text))
    return content.text


def _fetch_pro_trafilatura() -> str:
    content = scrape_url(_VLR_STATS_URL, source="meta_stats")
    if content is None:
        log.warning("Pro stats: vlr.gg returned nothing (JS-rendered).")
        return ""
    if len(content.text) < 500:
        log.warning(
            "Pro stats: vlr.gg returned only %d chars — JS-blocked. "
            "Set tavily_api_key in config.",
            len(content.text),
        )
    log.info("Pro stats: trafilatura vlr.gg → %d chars", len(content.text))
    return content.text


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def fetch_ranked_stats(settings=None) -> str:
    """Fetch ranked agent stats (pick rate / win rate).

    Priority: dak.gg Extract → blitz.gg Extract → Tavily search → trafilatura.
    """
    if settings is not None:
        text = _fetch_ranked_tavily(settings)
        if text:
            return text
    return _fetch_ranked_trafilatura()


def fetch_pro_stats(settings=None) -> str:
    """Fetch pro/VCT agent stats.

    Priority: Tavily search (VCT event pages) → trafilatura vlr.gg.
    """
    if settings is not None:
        text = _fetch_pro_tavily(settings)
        if text:
            return text
    return _fetch_pro_trafilatura()


def fetch_all_stats(settings=None) -> MetaStatsResult:
    """Fetch and combine ranked + pro stats.

    Both sources are attempted independently.  Pass ``settings`` to enable
    Tavily; omit to use trafilatura only.

    Returns:
        :class:`MetaStatsResult` with ``ok=True`` when at least one source
        returned data.
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

"""Scrape agent pick-rate and win-rate statistics.

Sources:
  - tracker.gg  — Diamond+ ranked play statistics
  - vlr.gg      — Pro / VCT tournament statistics

Both pages are scraped as raw text and returned combined so the LLM tier
generator can extract the numbers it needs without us having to maintain
fragile CSS-selector parsers against sites that change layout frequently.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from valocoach.retrieval.scrapers.web import scrape_url

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Source URLs
# tracker.gg exposes an insights page that shows pick-rate / win-rate by rank.
# vlr.gg shows aggregate agent stats across recent VCT events.
# ---------------------------------------------------------------------------
_TRACKER_AGENTS_URL = "https://tracker.gg/valorant/insights/agents"
_VLR_STATS_URL = "https://www.vlr.gg/stats"


@dataclass
class MetaStatsResult:
    """Container returned by :func:`fetch_all_stats`."""

    ranked_text: str
    """Raw text extracted from tracker.gg (empty string on failure)."""

    pro_text: str
    """Raw text extracted from vlr.gg (empty string on failure)."""

    @property
    def combined(self) -> str:
        """Both sources concatenated with labelled headers for LLM context."""
        parts: list[str] = []
        if self.ranked_text:
            parts.append(
                "=== DIAMOND+ RANKED STATS (tracker.gg) ===\n" + self.ranked_text
            )
        if self.pro_text:
            parts.append(
                "=== PRO / VCT STATS (vlr.gg) ===\n" + self.pro_text
            )
        return "\n\n".join(parts)

    @property
    def ok(self) -> bool:
        """True when at least one source returned data."""
        return bool(self.ranked_text or self.pro_text)


def fetch_ranked_stats() -> str:
    """Fetch Diamond+ ranked agent stats from tracker.gg.

    Returns the extracted page text, or an empty string on failure.
    """
    content = scrape_url(_TRACKER_AGENTS_URL, source="meta_stats")
    if content is None:
        log.warning("Could not fetch ranked stats from tracker.gg")
        return ""
    log.info("Fetched ranked stats (%d chars)", len(content.text))
    return content.text


def fetch_pro_stats() -> str:
    """Fetch pro / VCT agent stats from vlr.gg.

    Returns the extracted page text, or an empty string on failure.
    """
    content = scrape_url(_VLR_STATS_URL, source="meta_stats")
    if content is None:
        log.warning("Could not fetch pro stats from vlr.gg")
        return ""
    log.info("Fetched pro stats (%d chars)", len(content.text))
    return content.text


def fetch_all_stats() -> MetaStatsResult:
    """Fetch and combine ranked + pro stats from both sources.

    Both fetches are attempted independently — a failure on one does not
    block the other.  Callers should check :attr:`MetaStatsResult.ok`
    before passing the result to the LLM generator.
    """
    ranked = fetch_ranked_stats()
    pro = fetch_pro_stats()
    return MetaStatsResult(ranked_text=ranked, pro_text=pro)

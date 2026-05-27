"""Scrape official Valorant patch notes from playvalorant.com.

Patch notes URLs follow a predictable pattern based on the major.minor
version number, so we can construct the URL automatically when a new
patch version is detected by the patch tracker.

F4 — Multi-source fallback chain (+ optional Tavily)
------------------------------------------------------
fetch_patch_notes() tries sources in order and returns the first success:

  1. playvalorant.com  (official, most structured)
  2. liquipedia.net    (reliable mirror, often up before official)
  3. Reddit r/VALORANT (crowd-sourced; top post of patch day)
  4. Tavily search     (only when tavily_api_key is configured — best for
                        newly published pages that haven't propagated to
                        Liquipedia/Reddit yet, or when playvalorant.com
                        has unusual URL structures on a new patch)

Each fallback is logged at INFO level so operators can see which source
was used.
"""

from __future__ import annotations

import json
import logging
import re
from urllib.error import URLError
from urllib.request import Request, urlopen

from valocoach.retrieval.scrapers import ScrapedContent
from valocoach.retrieval.scrapers.web import scrape_url

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# URL templates
# ---------------------------------------------------------------------------

# Official Valorant patch notes — most structured, always try first.
# Valorant uses major.minor in the slug, e.g. patch 10.08 → patch-notes-10-08
_PATCH_NOTES_URL = (
    "https://playvalorant.com/en-us/news/game-updates/valorant-patch-notes-{major}-{minor}/"
)

# Liquipedia mirror — usually live within hours of an official drop.
# e.g. Patch 10.08 → https://liquipedia.net/valorant/Patch_10.08
_LIQUIPEDIA_URL = "https://liquipedia.net/valorant/Patch_{major}.{minor}"

# Reddit search API — crowd-sourced; returns JSON with post URLs.
# We scrape the top result's URL (usually the megathread or official post).
_REDDIT_SEARCH_URL = (
    "https://www.reddit.com/r/VALORANT/search.json"
    "?q=patch+notes+{major}.{minor}&restrict_sr=1&sort=top&t=week&limit=3"
)

# HenrikDev returns strings like "release-10.08-shipping-32-1234567".
# We just need the first X.YY portion.
_VERSION_RE = re.compile(r"(\d+)\.(\d+)")


# ---------------------------------------------------------------------------
# Version helpers
# ---------------------------------------------------------------------------


def parse_version(version: str) -> tuple[str, str] | None:
    """Extract (major, minor) from a raw game version string.

    Examples::

        parse_version("release-10.08-shipping-32-123") → ("10", "08")
        parse_version("10.09")                         → ("10", "09")
        parse_version("garbage")                       → None
    """
    match = _VERSION_RE.search(version)
    if match:
        return match.group(1), match.group(2)
    return None


def build_patch_notes_url(version: str) -> str | None:
    """Construct the official patch notes URL from a game version string.

    Returns None when the version string cannot be parsed.
    """
    parsed = parse_version(version)
    if not parsed:
        log.warning("Could not parse version string for URL construction: %r", version)
        return None
    major, minor = parsed
    return _PATCH_NOTES_URL.format(major=major, minor=minor)


# ---------------------------------------------------------------------------
# F4 — Individual source fetchers
# ---------------------------------------------------------------------------


def _fetch_playvalorant(major: str, minor: str) -> ScrapedContent | None:
    url = _PATCH_NOTES_URL.format(major=major, minor=minor)
    log.info("Trying source 1 (playvalorant.com): %s", url)
    try:
        return scrape_url(url, source="patch_note")
    except Exception as exc:
        log.info("Source 1 failed: %s", exc)
        return None


def _fetch_liquipedia(major: str, minor: str) -> ScrapedContent | None:
    """Fetch from Liquipedia.

    Liquipedia pages render some content server-side but rely on JavaScript for
    full table content.  We accept the result only when at least 500 characters
    of text were extracted — short responses indicate a JS-rendered blank page
    that would pollute the LLM with garbage rather than helping it.
    """
    url = _LIQUIPEDIA_URL.format(major=major, minor=minor)
    log.info("Trying fallback source 2 (liquipedia.net): %s", url)
    try:
        result = scrape_url(url, source="patch_note")
        if result is not None and len(result.text) < 500:
            log.info(
                "Source 2: Liquipedia returned only %d chars — likely JS-blocked, skipping",
                len(result.text),
            )
            return None
        return result
    except Exception as exc:
        log.info("Source 2 failed: %s", exc)
        return None


def _fetch_reddit(major: str, minor: str) -> ScrapedContent | None:
    """Query Reddit's search JSON API and scrape the top result's URL."""
    search_url = _REDDIT_SEARCH_URL.format(major=major, minor=minor)
    log.info("Trying fallback source 3 (reddit r/VALORANT search): %s", search_url)
    try:
        req_headers = {"User-Agent": "valocoach/1.0 (patch notes scraper)"}
        req = Request(search_url, headers=req_headers)
        with urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())

        posts = data.get("data", {}).get("children", [])
        if not posts:
            log.info("Source 3: no Reddit posts found for patch %s.%s", major, minor)
            return None

        # Pick the first non-stickied post that looks like a patch thread.
        for post in posts:
            post_data = post.get("data", {})
            url = post_data.get("url", "")
            title = post_data.get("title", "").lower()
            # Skip self.reddit posts (discussions) — prefer linked official URLs
            if "patch" in title and not post_data.get("is_self", True):
                log.info("Source 3: scraping linked URL from Reddit post: %s", url)
                return scrape_url(url, source="patch_note")

        # Fallback: scrape the Reddit thread itself
        permalink = posts[0]["data"].get("permalink", "")
        if permalink:
            reddit_thread = f"https://www.reddit.com{permalink}"
            log.info("Source 3: scraping Reddit thread itself: %s", reddit_thread)
            return scrape_url(reddit_thread, source="patch_note")

    except (URLError, OSError, json.JSONDecodeError, KeyError) as exc:
        log.info("Source 3 failed: %s", exc)

    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _fetch_tavily(major: str, minor: str, settings) -> ScrapedContent | None:
    """Search for patch notes via Tavily (source 4 in the fallback chain).

    Only called when ``settings.tavily_api_key`` is configured.  Useful
    when playvalorant.com has an unexpected URL structure, Liquipedia hasn't
    mirrored yet, or Reddit search is rate-limited.
    """
    from valocoach.retrieval.scrapers import tavily_client as tv

    if not tv.is_configured(settings):
        return None

    query = f"Valorant patch notes {major}.{minor} full agent changes"
    log.info("Trying fallback source 4 (Tavily search): %r", query)
    return tv.search(
        query,
        settings,
        search_depth="advanced",
        max_results=3,
        include_domains=["playvalorant.com", "liquipedia.net", "dotesports.com"],
        source="patch_note",
    )


def fetch_patch_notes(version: str, settings=None) -> ScrapedContent | None:
    """Fetch the official patch notes page for a given game version.

    Tries multiple sources in order (F4 fallback chain):
      1. playvalorant.com
      2. liquipedia.net
      3. Reddit r/VALORANT search
      4. Tavily search (only when ``settings.tavily_api_key`` is set)

    Args:
        version:  Raw game version string from the HenrikDev API,
                  e.g. ``"release-10.09-shipping-12-9876543"``.
        settings: App settings.  Pass to enable Tavily as source 4.

    Returns:
        :class:`~valocoach.retrieval.scrapers.ScrapedContent` with the
        extracted patch notes text, or ``None`` if all sources fail.
    """
    parsed = parse_version(version)
    if not parsed:
        log.warning("Could not parse version string for patch notes: %r", version)
        return None

    major, minor = parsed

    static_fetchers = (_fetch_playvalorant, _fetch_liquipedia, _fetch_reddit)
    for fetch_fn in static_fetchers:
        try:
            result = fetch_fn(major, minor)
        except Exception as exc:
            log.warning("Unexpected error in patch notes fetcher %s: %s", fetch_fn.__name__, exc)
            result = None

        if result is not None:
            log.info("Patch notes fetched via %s for %s.%s", fetch_fn.__name__, major, minor)
            return result

    # Source 4: Tavily (only when configured)
    if settings is not None:
        try:
            result = _fetch_tavily(major, minor, settings)
        except Exception as exc:
            log.warning("Tavily patch notes fetcher failed: %s", exc)
            result = None

        if result is not None:
            log.info("Patch notes fetched via Tavily for %s.%s", major, minor)
            return result

    log.warning(
        "All patch notes sources failed for version %r (%s.%s)", version, major, minor
    )
    return None

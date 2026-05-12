"""Scrape official Valorant patch notes from playvalorant.com.

Patch notes URLs follow a predictable pattern based on the major.minor
version number, so we can construct the URL automatically when a new
patch version is detected by the patch tracker.
"""

from __future__ import annotations

import logging
import re

from valocoach.retrieval.scrapers import ScrapedContent
from valocoach.retrieval.scrapers.web import scrape_url

log = logging.getLogger(__name__)

# Official Valorant patch notes URL template.
# Valorant uses major.minor in the slug, e.g. patch 10.08 → patch-notes-10-08
_PATCH_NOTES_URL = (
    "https://playvalorant.com/en-us/news/game-updates/valorant-patch-notes-{major}-{minor}/"
)

# HenrikDev returns strings like "release-10.08-shipping-32-1234567".
# We just need the first X.YY portion.
_VERSION_RE = re.compile(r"(\d+)\.(\d+)")


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


def fetch_patch_notes(version: str) -> ScrapedContent | None:
    """Fetch the official patch notes page for a given game version.

    Args:
        version: Raw game version string from the HenrikDev API,
                 e.g. ``"release-10.09-shipping-12-9876543"``.

    Returns:
        :class:`~valocoach.retrieval.scrapers.ScrapedContent` with the
        extracted patch notes text, or ``None`` on any failure.
    """
    url = build_patch_notes_url(version)
    if not url:
        return None

    log.info("Fetching patch notes: %s", url)
    content = scrape_url(url, source="patch_note")
    if content is None:
        log.warning("Failed to scrape patch notes for version %r (url=%s)", version, url)
    return content

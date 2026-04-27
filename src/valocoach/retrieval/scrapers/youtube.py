from __future__ import annotations

import logging
import re
from datetime import datetime, timezone

from valocoach.retrieval.scrapers import ScrapedContent

log = logging.getLogger(__name__)


def _extract_video_id(url: str) -> str:
    """Parse a YouTube video ID from any common URL format or bare ID."""
    pattern = r"(?:v=|youtu\.be\/|\/embed\/|\/v\/|\/shorts\/)([A-Za-z0-9_-]{11})"
    match = re.search(pattern, url)
    if match:
        return match.group(1)
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", url):
        return url
    raise ValueError(f"Could not extract video ID from: {url}")


def fetch_transcript(
    url: str,
    languages: list[str] | None = None,
) -> ScrapedContent | None:
    """Fetch a YouTube transcript and return it as a ScrapedContent.

    Args:
        url: Full YouTube URL or bare 11-char video ID.
        languages: Preference order for transcript language (default: English variants).

    Returns None on any failure so callers don't need try/except.
    """
    from youtube_transcript_api import YouTubeTranscriptApi

    try:
        video_id = _extract_video_id(url)
    except ValueError as exc:
        log.warning("Invalid YouTube URL/ID: %s", exc)
        return None

    langs = languages or ["en", "en-US", "en-GB"]

    try:
        entries = YouTubeTranscriptApi.get_transcript(video_id, languages=langs)
    except Exception as exc:
        log.warning("Transcript fetch failed for %s: %s", video_id, exc)
        return None

    text = " ".join(e["text"].strip() for e in entries if e["text"].strip())
    if len(text) < 200:
        log.warning("Transcript too short for %s (%d chars)", video_id, len(text))
        return None

    return ScrapedContent(
        url=f"https://www.youtube.com/watch?v={video_id}",
        title=f"YouTube transcript — {video_id}",
        text=text,
        fetched_at=datetime.now(tz=timezone.utc).isoformat(),
        source="youtube",
    )

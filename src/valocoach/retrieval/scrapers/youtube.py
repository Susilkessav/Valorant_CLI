"""YouTube transcript fetcher with time-windowed chunking (Phase D3) and oEmbed metadata (D1)."""

from __future__ import annotations

import json as _json
import logging
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from urllib.request import urlopen

from valocoach.retrieval.scrapers import ScrapedContent

log = logging.getLogger(__name__)

_OEMBED_URL = (
    "https://www.youtube.com/oembed"
    "?url=https://www.youtube.com/watch?v={video_id}&format=json"
)
_DEFAULT_WINDOW_SECONDS = 120  # D3: 2-minute time windows


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class YouTubeChunk:
    """A single time-windowed segment of a YouTube transcript (Phase D3)."""

    video_id: str
    title: str
    channel: str
    url: str
    start_seconds: int
    text: str
    fetched_at: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_video_id(url: str) -> str:
    """Parse a YouTube video ID from any common URL format or bare ID."""
    pattern = r"(?:v=|youtu\.be\/|\/embed\/|\/v\/|\/shorts\/)([A-Za-z0-9_-]{11})"
    match = re.search(pattern, url)
    if match:
        return match.group(1)
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", url):
        return url
    raise ValueError(f"Could not extract video ID from: {url}")


def fetch_video_metadata(video_id: str) -> dict:
    """Fetch video title + channel name via the YouTube oEmbed endpoint (D1).

    No API key required.  Returns safe defaults on failure so ingestion
    can always continue even when oEmbed is temporarily unavailable.
    """
    try:
        oembed_url = _OEMBED_URL.format(video_id=video_id)
        with urlopen(oembed_url, timeout=8) as resp:
            data = _json.loads(resp.read())
        return {
            "title": data.get("title") or f"YouTube — {video_id}",
            "channel": data.get("author_name") or "Unknown channel",
        }
    except Exception as exc:
        log.debug("oEmbed lookup failed for %s: %s", video_id, exc)
        return {"title": f"YouTube — {video_id}", "channel": "Unknown channel"}


def _window_entries(entries: list, window_seconds: int) -> list[tuple[int, str]]:
    """Group timed transcript entries into fixed-length time windows.

    Returns a list of ``(start_seconds, combined_text)`` tuples.  A new window
    starts whenever an entry's ``start`` time crosses a window boundary.
    """
    if not entries:
        return []

    windows: list[tuple[int, str]] = []
    current_start = int(entries[0].start)
    current_texts: list[str] = []

    for entry in entries:
        entry_start = int(entry.start)
        text = entry.text.strip()
        if not text:
            continue

        if entry_start >= current_start + window_seconds:
            if current_texts:
                windows.append((current_start, " ".join(current_texts)))
            current_start = entry_start
            current_texts = [text]
        else:
            current_texts.append(text)

    if current_texts:
        windows.append((current_start, " ".join(current_texts)))

    return windows


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def fetch_transcript_chunks(
    url: str,
    languages: list[str] | None = None,
    window_seconds: int = _DEFAULT_WINDOW_SECONDS,
) -> list[YouTubeChunk]:
    """Fetch a YouTube transcript and split it into time-windowed chunks (D3).

    Each chunk covers a ``window_seconds``-wide segment of the video.  The
    real video title and channel are fetched via oEmbed (D1).

    Args:
        url:            Full YouTube URL or bare 11-char video ID.
        languages:      Transcript language preference order.
        window_seconds: Window width in seconds (default 120 = 2 minutes).

    Returns:
        A list of :class:`YouTubeChunk` objects — one per non-trivial window.
        Empty list on any failure.
    """
    from youtube_transcript_api import YouTubeTranscriptApi

    try:
        video_id = _extract_video_id(url)
    except ValueError as exc:
        log.warning("Invalid YouTube URL/ID: %s", exc)
        return []

    langs = languages or ["en", "en-US", "en-GB"]
    now = datetime.now(tz=UTC).isoformat()

    # D1 — real title + channel
    meta = fetch_video_metadata(video_id)
    video_url = f"https://www.youtube.com/watch?v={video_id}"

    try:
        transcript = YouTubeTranscriptApi().fetch(video_id, languages=langs)
        entries = list(transcript)
    except Exception as exc:
        log.warning("Transcript fetch failed for %s: %s", video_id, exc)
        return []

    if not entries:
        log.warning("Empty transcript for %s", video_id)
        return []

    windows = _window_entries(entries, window_seconds)
    chunks: list[YouTubeChunk] = [
        YouTubeChunk(
            video_id=video_id,
            title=meta["title"],
            channel=meta["channel"],
            url=video_url,
            start_seconds=start_sec,
            text=text,
            fetched_at=now,
        )
        for start_sec, text in windows
        if len(text) >= 50  # skip windows with almost no content
    ]

    log.info(
        "Fetched %d chunk(s) from %s (%r, channel=%r)",
        len(chunks),
        video_id,
        meta["title"],
        meta["channel"],
    )
    return chunks


def fetch_transcript(
    url: str,
    languages: list[str] | None = None,
) -> ScrapedContent | None:
    """Fetch a YouTube transcript and return it as a single ScrapedContent blob.

    .. deprecated::
        New code should call :func:`fetch_transcript_chunks` (Phase D full
        pipeline: time windows, oEmbed metadata, anchor filter, dedup).
        This wrapper is kept for backward compatibility only.
    """
    chunks = fetch_transcript_chunks(url, languages=languages)
    if not chunks:
        return None

    text = " ".join(c.text for c in chunks)
    if len(text) < 200:
        log.warning("Transcript too short for %s (%d chars)", url, len(text))
        return None

    first = chunks[0]
    return ScrapedContent(
        url=first.url,
        title=first.title,
        text=text,
        fetched_at=first.fetched_at,
        source="youtube",
    )

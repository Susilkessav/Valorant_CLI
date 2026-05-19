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
_DEFAULT_WINDOW_SECONDS = 60  # D3: 1-minute windows — finer granularity for lineup guides

# Matches YouTube caption noise: [Music], [Applause], [Laughter], [music], etc.
_NOISE_RE = re.compile(r"\[[\w\s]+\]")

# Human-readable failure reason returned alongside an empty chunk list so
# callers can show an actionable message instead of a raw exception.
# Values: "ip_blocked" | "no_captions" | "unavailable" | "no_language" | "unknown"
TranscriptFailReason = str


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


def clean_transcript_text(text: str) -> str:
    """Strip caption noise tags and normalize whitespace.

    Removes ``[Music]``, ``[Applause]``, ``[Laughter]`` and any similar
    bracketed annotations that YouTube auto-captions inject.  These tokens
    carry no tactical information and dilute embedding similarity scores.
    """
    cleaned = _NOISE_RE.sub(" ", text)
    # Collapse runs of whitespace left by the substitution
    return " ".join(cleaned.split())


def _window_entries(entries: list, window_seconds: int) -> list[tuple[int, str]]:
    """Group timed transcript entries into fixed-length time windows.

    Returns a list of ``(start_seconds, combined_text)`` tuples.  A new window
    starts whenever an entry's ``start`` time crosses a window boundary.
    Text is cleaned of caption noise (``[Music]`` etc.) before joining.
    """
    if not entries:
        return []

    windows: list[tuple[int, str]] = []
    current_start = int(entries[0].start)
    current_texts: list[str] = []

    for entry in entries:
        entry_start = int(entry.start)
        text = clean_transcript_text(entry.text.strip())
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
) -> tuple[list[YouTubeChunk], TranscriptFailReason | None]:
    """Fetch a YouTube transcript and split it into time-windowed chunks (D3).

    Each chunk covers a ``window_seconds``-wide segment of the video.  The
    real video title and channel are fetched via oEmbed (D1).

    Returns:
        ``(chunks, fail_reason)`` — on success, *chunks* is non-empty and
        *fail_reason* is ``None``.  On failure, *chunks* is ``[]`` and
        *fail_reason* is one of ``"ip_blocked"``, ``"no_captions"``,
        ``"unavailable"``, ``"no_language"``, or ``"unknown"``.
    """
    from youtube_transcript_api import YouTubeTranscriptApi
    from youtube_transcript_api._errors import (
        TranscriptsDisabled,
        VideoUnavailable,
    )

    try:
        video_id = _extract_video_id(url)
    except ValueError as exc:
        log.debug("Invalid YouTube URL/ID: %s", exc)
        return [], "unavailable"

    langs = languages or ["en", "en-US", "en-GB"]
    now = datetime.now(tz=UTC).isoformat()

    # D1 — real title + channel
    meta = fetch_video_metadata(video_id)
    video_url = f"https://www.youtube.com/watch?v={video_id}"

    try:
        transcript = YouTubeTranscriptApi().fetch(video_id, languages=langs)
        entries = list(transcript)
    except TranscriptsDisabled:
        log.debug("Captions disabled for %s", video_id)
        return [], "no_captions"
    except VideoUnavailable:
        log.debug("Video unavailable: %s", video_id)
        return [], "unavailable"
    except Exception as exc:
        exc_str = str(exc)
        # youtube_transcript_api raises RequestBlocked / IpBlocked with this text
        if "blocked" in exc_str.lower() or "requestblocked" in exc_str.lower() or "ipblocked" in exc_str.lower():
            log.debug("YouTube transcript request blocked for %s", video_id)
            return [], "ip_blocked"
        # NoTranscriptFound — no transcript in the requested language
        if "notranscriptfound" in type(exc).__name__.lower() or "no transcript" in exc_str.lower():
            log.debug("No transcript found for %s in %s", video_id, langs)
            return [], "no_language"
        log.debug("Transcript fetch failed for %s: %s", video_id, exc)
        return [], "unknown"

    if not entries:
        log.debug("Empty transcript for %s", video_id)
        return [], "unknown"

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
    return chunks, None


__all__ = [
    "YouTubeChunk",
    "clean_transcript_text",
    "fetch_transcript",
    "fetch_transcript_chunks",
    "fetch_video_metadata",
]


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
    chunks, _fail_reason = fetch_transcript_chunks(url, languages=languages)
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

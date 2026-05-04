"""Project-wide exception hierarchy.

Catch-all order:
    ValocoachError          — any error from this project
    ├── APIError            — external HTTP API failure (non-retryable 4xx)
    │   ├── RateLimitError  — 429, always retried with backoff
    │   └── ServerError     — 5xx, retried with backoff (temporary outage)
    ├── SyncError           — match sync pipeline failure
    └── ConfigError         — missing or invalid configuration
"""

from __future__ import annotations


class ValocoachError(Exception):
    """Base for all project exceptions."""


class APIError(ValocoachError):
    """HenrikDev or other external API failure."""


class RateLimitError(APIError):
    """429 from upstream — temporary, always retried with backoff."""


class ServerError(APIError):
    """5xx from upstream — temporary outage, retried with backoff."""


class SyncError(ValocoachError):
    """Match sync pipeline failure."""


class MapperError(SyncError):
    """API-to-ORM mapping rejected this match.

    Raised when the API response is well-formed Pydantic but logically
    invalid for storage (e.g. missing ``started_at``, which would corrupt
    ``ORDER BY started_at`` queries by sorting before all valid ISO
    timestamps).  Per-match failure — sync skips the offender and continues.
    """


class ConfigError(ValocoachError):
    """Configuration is missing or invalid."""

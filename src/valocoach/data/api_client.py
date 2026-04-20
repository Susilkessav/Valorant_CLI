"""Async HTTP client for the HenrikDev Valorant API.

Endpoints wrapped:
  Account           GET /valorant/v2/account/{name}/{tag}
  MMR               GET /valorant/v2/mmr/{region}/{name}/{tag}
  MMR history       GET /valorant/v2/mmr-history/{region}/{name}/{tag}
  Matches (v3)      GET /valorant/v3/matches/{region}/{name}/{tag}
  Stored matches    GET /valorant/v1/stored-matches/{region}/{name}/{tag}
  Match detail (v4) GET /valorant/v4/match/{region}/{match_id}
  Version           GET /valorant/v1/version/{region}

  fetch_player_snapshot() runs account + MMR + matches concurrently.

Design:
  - Settings injection — takes the project Settings object, not a bare key
    string, so the CLI entry point is `HenrikClient(settings)`.
  - Proactive throttle — _throttle() paces sequential calls to ~1 per
    throttle_seconds before sending, avoiding rate-limit hits rather than
    only recovering from them.  fetch_player_snapshot() intentionally bypasses
    the throttle for its three concurrent requests (a burst of 3 is well
    within the free-tier 30 req/min cap).
  - Structured timeout — separate connect / read limits so a DNS failure
    surfaces in 5 s while slow match payloads are allowed 20 s.
  - Retry policy — RateLimitError (429), ServerError (5xx), and network
    TransportErrors are retried up to 3 times with exponential back-off
    (2 s, 4 s, 8 s … capped at 30 s).  Non-retryable 4xx errors raise
    APIError immediately.
  - Body-status check — Henrik sometimes returns HTTP 200 with a non-200
    status field inside the JSON body; _get() promotes those to the correct
    typed exception.
  - All endpoints return typed Pydantic models — no raw dict leakage.
"""

from __future__ import annotations

import asyncio
import logging

import httpx
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from valocoach.core.config import Settings
from valocoach.core.exceptions import APIError, ConfigError, RateLimitError, ServerError
from valocoach.data.api_models import AccountResponse, MatchDetails, StoredMatch
from valocoach.data.models import MatchData, MMRData, MMRHistoryEntry

log = logging.getLogger(__name__)

BASE_URL = "https://api.henrikdev.xyz"

# connect=5 s: fast-fail on DNS / TCP issues
# read=20 s:  large v4 match payloads (kills + rounds) can be slow
DEFAULT_TIMEOUT = httpx.Timeout(connect=5.0, read=20.0, write=5.0, pool=5.0)


# ---------------------------------------------------------------------------
# Retry policy
# ---------------------------------------------------------------------------


def _is_retryable(exc: BaseException) -> bool:
    """Retry network errors, 429, and 5xx — never retry 4xx client errors."""
    return isinstance(exc, httpx.TransportError | RateLimitError | ServerError)


_retry_policy = retry(
    retry=retry_if_exception(_is_retryable),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=2, max=30),
    reraise=True,
)


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class HenrikClient:
    """Async context-manager client for the HenrikDev API.

    Usage::

        async with HenrikClient(settings) as client:
            account = await client.get_account("Yoursaviour01", "SK04")

        # Fetch account + MMR + recent matches concurrently:
        async with HenrikClient(settings) as client:
            account, mmr, matches = await client.fetch_player_snapshot(
                "na", "Yoursaviour01", "SK04"
            )
    """

    def __init__(
        self,
        settings: Settings,
        *,
        throttle_seconds: float = 2.0,
        timeout: httpx.Timeout = DEFAULT_TIMEOUT,
    ) -> None:
        if not settings.henrikdev_api_key:
            raise ConfigError(
                "HENRIKDEV_API_KEY is not configured — run `valocoach config set-key`"
            )
        self._settings = settings
        self.throttle = throttle_seconds
        self._timeout = timeout
        self._client: httpx.AsyncClient | None = None
        self._last_call_at: float = 0.0

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    async def __aenter__(self) -> HenrikClient:
        self._client = httpx.AsyncClient(
            base_url=BASE_URL,
            headers={"Authorization": self._settings.henrikdev_api_key},
            timeout=self._timeout,
        )
        return self

    async def __aexit__(self, *_: object) -> None:
        if self._client:
            await self._client.aclose()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _throttle(self) -> None:
        """Pace sequential requests to stay under the free-tier rate limit.

        Uses asyncio.get_running_loop() (Python 3.10+) — never the deprecated
        get_event_loop().  The _last_call_at timestamp is re-read after the
        sleep so back-to-back calls compute the correct delta.

        Note: fetch_player_snapshot() fires three concurrent coroutines via
        asyncio.gather.  They all read _last_call_at before any of them
        updates it, so all three fire without delay — intentional, as a burst
        of three requests is well within the 30 req/min free-tier cap.
        """
        loop = asyncio.get_running_loop()
        now = loop.time()
        wait = self.throttle - (now - self._last_call_at)
        if wait > 0:
            await asyncio.sleep(wait)
        self._last_call_at = loop.time()  # re-read after sleep for accurate delta

    async def _get(self, path: str, params: dict | None = None) -> dict:
        """GET a Henrik endpoint and return the validated JSON payload.

        Raises:
            RateLimitError: 429 — temporary, retried by _retry_policy.
            ServerError:    5xx — temporary outage, retried by _retry_policy.
            APIError:       Other 4xx or malformed JSON — not retried.
            RuntimeError:   Client used outside ``async with`` block.
        """
        if self._client is None:
            raise RuntimeError(
                "HenrikClient must be used as an async context manager — "
                "use `async with HenrikClient(settings) as client:`"
            )

        await self._throttle()
        log.debug("GET %s params=%s", path, params)
        resp = await self._client.get(path, params=params)
        log.debug("GET %s → %d", path, resp.status_code)

        # ── HTTP-level status ──────────────────────────────────────────
        if resp.status_code == 429:
            raise RateLimitError(f"Rate limited on {path}")
        if resp.status_code >= 500:
            raise ServerError(f"Server error {resp.status_code} on {path}")
        if resp.status_code >= 400:
            # 4xx other than 429 are caller errors — raise immediately, no retry
            raise APIError(f"HTTP {resp.status_code} on {path}: {resp.text[:200]}")

        # ── Parse JSON ────────────────────────────────────────────────
        try:
            payload = resp.json()
        except Exception as exc:
            raise APIError(f"Malformed JSON on {path}: {exc}") from exc

        # ── Body-level status (Henrik quirk: 200 HTTP, non-200 in body) ─
        body_status: int = payload.get("status", 200)
        if body_status == 429:
            raise RateLimitError(f"Rate limited (body) on {path}")
        if body_status >= 500:
            raise ServerError(f"Server error (body) {body_status} on {path}")
        if body_status >= 400:
            detail = payload.get("errors") or payload.get("message", "")
            raise APIError(f"API error {body_status} on {path}: {detail}")

        return payload

    # ------------------------------------------------------------------
    # Public endpoints
    # ------------------------------------------------------------------

    @_retry_policy
    async def get_account(
        self,
        name: str,
        tag: str,
        *,
        force: bool = False,
    ) -> AccountResponse:
        """Fetch account info — PUUID, level, region.

        Args:
            force: Bypass Henrik's cache for a fresh lookup.
                   Useful after a player changes their name or tag.
        """
        log.debug("get_account %s#%s force=%s", name, tag, force)
        params = {"force": "true"} if force else None
        payload = await self._get(f"/valorant/v2/account/{name}/{tag}", params=params)
        return AccountResponse.model_validate(payload["data"])

    @_retry_policy
    async def get_mmr(self, region: str, name: str, tag: str) -> MMRData:
        """Fetch current rank, RR, and peak rank for a player."""
        log.debug("get_mmr %s %s#%s", region, name, tag)
        payload = await self._get(f"/valorant/v2/mmr/{region}/{name}/{tag}")
        return MMRData.model_validate(payload["data"])

    @_retry_policy
    async def get_mmr_history(
        self,
        region: str,
        name: str,
        tag: str,
    ) -> list[MMRHistoryEntry]:
        """Fetch rank history — one entry per ranked game, newest first.

        Useful for coaching trend analysis: rank progression, win/loss streaks,
        RR delta patterns.
        """
        log.debug("get_mmr_history %s %s#%s", region, name, tag)
        payload = await self._get(f"/valorant/v2/mmr-history/{region}/{name}/{tag}")
        return [MMRHistoryEntry.model_validate(e) for e in payload["data"]]

    @_retry_policy
    async def get_matches(
        self,
        region: str,
        name: str,
        tag: str,
        *,
        size: int = 10,
        mode: str = "competitive",
    ) -> list[MatchData]:
        """Fetch recent matches via the v3 endpoint — full data, up to 10 on the free tier.

        Args:
            size: Number of matches to return (max 10 on free tier).
            mode: Game mode filter. Pass ``mode=""`` to include all modes.
                  Defaults to "competitive" so stats are never silently mixed
                  with unrated, swiftplay, or deathmatch rounds.
        """
        log.debug("get_matches %s %s#%s size=%d mode=%s", region, name, tag, size, mode)
        params: dict = {"size": size}
        if mode:
            params["mode"] = mode
        payload = await self._get(f"/valorant/v3/matches/{region}/{name}/{tag}", params=params)
        return [MatchData.model_validate(m) for m in payload["data"]]

    @_retry_policy
    async def get_stored_matches(
        self,
        region: str,
        name: str,
        tag: str,
        *,
        mode: str = "competitive",
        size: int = 20,
    ) -> list[StoredMatch]:
        """Fetch stored match history via the paginated v1 endpoint.

        Unlike get_matches(), this endpoint can return more than 10 games and
        is designed for bulk historical sync.  Each entry is a lightweight
        summary — call get_match_details() on the match_id to fetch full
        per-kill and per-round data.

        Args:
            mode: Game mode filter (default "competitive").
            size: Number of entries to return per page.
        """
        log.debug("get_stored_matches %s %s#%s size=%d mode=%s", region, name, tag, size, mode)
        params: dict = {"size": size}
        if mode:
            params["mode"] = mode
        payload = await self._get(
            f"/valorant/v1/stored-matches/{region}/{name}/{tag}", params=params
        )
        return [StoredMatch.model_validate(m) for m in payload.get("data", [])]

    @_retry_policy
    async def get_match_details(self, region: str, match_id: str) -> MatchDetails:
        """Fetch a single full match via the v4 endpoint.

        Includes per-kill events, per-round breakdowns, and economy data.
        Pair with get_stored_matches() for an efficient bulk-sync pipeline:
        fetch summaries first, then call this only for matches not yet stored.
        """
        log.debug("get_match_details %s %s", region, match_id)
        payload = await self._get(f"/valorant/v4/match/{region}/{match_id}")
        return MatchDetails.model_validate(payload["data"])

    @_retry_policy
    async def get_version(self, region: str) -> dict:
        """Fetch the current game version and patch info for a region.

        Useful for detecting patch-day changes that may affect match analysis.
        Returns a raw dict — the schema is stable but simple enough to not
        warrant a dedicated model.
        """
        log.debug("get_version %s", region)
        payload = await self._get(f"/valorant/v1/version/{region}")
        return payload.get("data", {})

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    async def fetch_player_snapshot(
        self,
        region: str,
        name: str,
        tag: str,
        *,
        match_count: int = 10,
    ) -> tuple[AccountResponse, MMRData, list[MatchData]]:
        """Fetch account, MMR, and recent matches in a single concurrent call.

        Runs all three requests with asyncio.gather — total wall time is the
        slowest of the three, not their sum.

        Returns:
            (AccountResponse, MMRData, list[MatchData]) — same objects as the
            individual methods, just fetched together.
        """
        log.debug("fetch_player_snapshot %s %s#%s", region, name, tag)
        account, mmr, matches = await asyncio.gather(
            self.get_account(name, tag),
            self.get_mmr(region, name, tag),
            self.get_matches(region, name, tag, size=match_count),
        )
        return account, mmr, matches

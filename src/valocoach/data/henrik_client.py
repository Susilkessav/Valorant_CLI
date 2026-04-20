"""Async HTTP client for the HenrikDev Valorant API.

Endpoints wrapped:
  Account      GET /valorant/v2/account/{name}/{tag}
  MMR          GET /valorant/v2/mmr/{region}/{name}/{tag}
  MMR history  GET /valorant/v2/mmr-history/{region}/{name}/{tag}
  Matches      GET /valorant/v3/matches/{region}/{name}/{tag}

  fetch_player_snapshot() runs account + MMR + matches concurrently.

Retry policy: automatic retry on rate-limit (429) and server errors (5xx),
up to 3 attempts with exponential back-off (1 s, 2 s, 4 s).
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

from valocoach.data.models import (
    AccountData,
    HenrikResponse,
    MatchData,
    MMRData,
    MMRHistoryEntry,
)

logger = logging.getLogger(__name__)

BASE_URL = "https://api.henrikdev.xyz"

# Separate connect / read timeouts — connect failures surface fast (5 s),
# read allows for slow API responses on large match payloads (10 s).
DEFAULT_TIMEOUT = httpx.Timeout(connect=5.0, read=10.0, write=5.0, pool=5.0)


# ---------------------------------------------------------------------------
# Retry helpers
# ---------------------------------------------------------------------------


def _is_retryable(exc: BaseException) -> bool:
    """Retry on network errors, 429, and 5xx responses."""
    return isinstance(exc, httpx.TransportError) or (
        isinstance(exc, HenrikAPIError) and exc.status_code in {429, 500, 502, 503, 504}
    )


_retry_policy = retry(
    retry=retry_if_exception(_is_retryable),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    reraise=True,
)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class HenrikAPIError(Exception):
    def __init__(self, status_code: int, message: str) -> None:
        self.status_code = status_code
        super().__init__(f"Henrik API {status_code}: {message}")


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class HenrikClient:
    """Async context-manager client for the HenrikDev API.

    Usage::

        async with HenrikClient(api_key="HDEV-...") as client:
            account = await client.get_account("Yoursaviour01", "SK04")

        # Fetch account + MMR + matches in one concurrent call:
        async with HenrikClient(api_key="HDEV-...") as client:
            account, mmr, matches = await client.fetch_player_snapshot(
                "na", "Yoursaviour01", "SK04"
            )
    """

    def __init__(
        self,
        api_key: str,
        timeout: httpx.Timeout = DEFAULT_TIMEOUT,
    ) -> None:
        self._api_key = api_key
        self._timeout = timeout
        self._http: httpx.AsyncClient | None = None

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    async def __aenter__(self) -> HenrikClient:
        self._http = httpx.AsyncClient(
            base_url=BASE_URL,
            headers={"Authorization": self._api_key},
            timeout=self._timeout,
        )
        return self

    async def __aexit__(self, *_: object) -> None:
        if self._http:
            await self._http.aclose()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _get(self, path: str, **params: str | int) -> dict:
        """GET a Henrik endpoint and return the validated JSON payload.

        Raises HenrikAPIError on:
          - Non-200 HTTP status
          - Non-200 status inside the JSON body  (e.g. 403 in {"status":403})
          - Malformed / non-JSON response body
        """
        if self._http is None:
            raise RuntimeError(
                "HenrikClient must be used as an async context manager — "
                "use `async with HenrikClient(...) as client:`"
            )

        response = await self._http.get(path, params=params or None)
        logger.debug("GET %s → %d", path, response.status_code)

        if response.status_code != 200:
            raise HenrikAPIError(response.status_code, response.text[:200])

        try:
            payload = response.json()
        except Exception as exc:
            raise HenrikAPIError(0, f"Malformed JSON response: {exc}") from exc

        # Henrik sometimes returns 200 HTTP but a non-200 status in the body
        # (e.g. {"status": 403, "errors": [...]})
        body_status = payload.get("status", 200)
        if body_status != 200:
            error_detail = payload.get("errors") or payload.get("message", "")
            raise HenrikAPIError(body_status, str(error_detail))

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
    ) -> AccountData:
        """Fetch account info — PUUID, level, region.

        Args:
            force: Pass True to bypass Henrik's cache and get a fresh lookup.
                   Useful when a player recently changed their name/tag.
        """
        logger.debug("get_account %s#%s force=%s", name, tag, force)
        params: dict[str, str | int] = {}
        if force:
            params["force"] = "true"
        payload = await self._get(f"/valorant/v2/account/{name}/{tag}", **params)
        return HenrikResponse[AccountData].model_validate(payload).data

    @_retry_policy
    async def get_mmr(self, region: str, name: str, tag: str) -> MMRData:
        """Fetch current rank, RR, and peak rank for a player."""
        logger.debug("get_mmr %s %s#%s", region, name, tag)
        payload = await self._get(f"/valorant/v2/mmr/{region}/{name}/{tag}")
        return HenrikResponse[MMRData].model_validate(payload).data

    @_retry_policy
    async def get_mmr_history(
        self,
        region: str,
        name: str,
        tag: str,
    ) -> list[MMRHistoryEntry]:
        """Fetch rank history — one entry per ranked game, newest first.

        Useful for coaching trend analysis: rank progression over time,
        win/loss streaks, RR delta patterns.
        """
        logger.debug("get_mmr_history %s %s#%s", region, name, tag)
        payload = await self._get(f"/valorant/v2/mmr-history/{region}/{name}/{tag}")
        return HenrikResponse[list[MMRHistoryEntry]].model_validate(payload).data

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
        """Fetch recent match history (default: last 10 competitive games).

        Args:
            size: Number of matches to return (max 10 on the free tier).
            mode: Game mode filter. Pass ``mode=""`` to fetch all modes.
                  Defaults to "competitive" so stats are never silently mixed
                  with unrated, swiftplay, or deathmatch rounds.
        """
        logger.debug("get_matches %s %s#%s size=%d mode=%s", region, name, tag, size, mode)
        params: dict[str, str | int] = {"size": size}
        if mode:
            params["mode"] = mode
        payload = await self._get(f"/valorant/v3/matches/{region}/{name}/{tag}", **params)
        return HenrikResponse[list[MatchData]].model_validate(payload).data

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
    ) -> tuple[AccountData, MMRData, list[MatchData]]:
        """Fetch account, MMR, and recent matches in a single concurrent call.

        Runs all three requests with asyncio.gather — total wall time is the
        slowest of the three instead of their sum.

        Returns:
            (AccountData, MMRData, list[MatchData]) — same objects as the
            individual methods, just fetched together.
        """
        logger.debug("fetch_player_snapshot %s %s#%s", region, name, tag)
        account, mmr, matches = await asyncio.gather(
            self.get_account(name, tag),
            self.get_mmr(region, name, tag),
            self.get_matches(region, name, tag, size=match_count),
        )
        return account, mmr, matches

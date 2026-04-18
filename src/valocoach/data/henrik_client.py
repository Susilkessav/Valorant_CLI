"""Async HTTP client for the HenrikDev Valorant API.

Wraps the three endpoints the coaching app needs:
  - /v2/account/{name}/{tag}
  - /v2/mmr/{region}/{name}/{tag}
  - /v3/matches/{region}/{name}/{tag}

Retries automatically on rate-limit (429) and server errors (5xx).
"""

from __future__ import annotations

import logging

import httpx
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from valocoach.data.models import AccountData, HenrikResponse, MatchData, MMRData

logger = logging.getLogger(__name__)

BASE_URL = "https://api.henrikdev.xyz"
DEFAULT_TIMEOUT = 10.0  # seconds


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
    """

    def __init__(self, api_key: str, timeout: float = DEFAULT_TIMEOUT) -> None:
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
        assert self._http is not None, "Use HenrikClient as an async context manager"
        response = await self._http.get(path, params=params or None)
        if response.status_code != 200:
            raise HenrikAPIError(response.status_code, response.text[:200])
        payload = response.json()
        if payload.get("status") != 200:
            raise HenrikAPIError(payload["status"], str(payload.get("errors", "")))
        return payload

    # ------------------------------------------------------------------
    # Public endpoints
    # ------------------------------------------------------------------

    @_retry_policy
    async def get_account(self, name: str, tag: str) -> AccountData:
        """Fetch account info — returns PUUID, level, region."""
        logger.debug("get_account %s#%s", name, tag)
        payload = await self._get(f"/valorant/v2/account/{name}/{tag}")
        return HenrikResponse[AccountData].model_validate(payload).data

    @_retry_policy
    async def get_mmr(self, region: str, name: str, tag: str) -> MMRData:
        """Fetch current rank and RR for a player."""
        logger.debug("get_mmr %s %s#%s", region, name, tag)
        payload = await self._get(f"/valorant/v2/mmr/{region}/{name}/{tag}")
        return HenrikResponse[MMRData].model_validate(payload).data

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
        """Fetch recent match history (default last 10 competitive games).

        Defaults to competitive so stats are never silently mixed with
        unrated, swift, or deathmatch rounds.

        Args:
            size: Number of matches to return (max 10 per Henrik free tier).
            mode: Game mode filter. Pass ``mode=""`` to fetch all modes.
        """
        logger.debug("get_matches %s %s#%s size=%d", region, name, tag, size)
        params: dict[str, str | int] = {"size": size}
        if mode:
            params["mode"] = mode
        payload = await self._get(f"/valorant/v3/matches/{region}/{name}/{tag}", **params)
        return HenrikResponse[list[MatchData]].model_validate(payload).data

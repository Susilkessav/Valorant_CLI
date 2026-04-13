from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import httpx

from valocoach.core.config import AppSettings
from valocoach.core.exceptions import ConfigurationError, ProviderError


def _normalize_host(host: str) -> str:
    return host.rstrip("/")


def _extract_chunk_text(chunk: Any) -> str:
    choices = getattr(chunk, "choices", None)
    if choices is None and isinstance(chunk, dict):
        choices = chunk.get("choices")
    if not choices:
        return ""

    choice = choices[0]
    delta = getattr(choice, "delta", None)
    if delta is None and isinstance(choice, dict):
        delta = choice.get("delta", {})

    if isinstance(delta, dict):
        content = delta.get("content")
    else:
        content = getattr(delta, "content", None)

    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text", "")))
        return "".join(parts)
    return ""


class LiteLLMOllamaProvider:
    """Stream chat completions from Ollama through LiteLLM."""

    def __init__(self, settings: AppSettings) -> None:
        self._settings = settings
        self._api_base = _normalize_host(settings.ollama.host)
        self._model = f"ollama/{settings.ollama.model}"

    async def validate_connection(self) -> None:
        """Verify the Ollama server is reachable and the configured model exists."""
        url = f"{self._api_base}/api/tags"
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(url)
                response.raise_for_status()
        except httpx.HTTPError as error:
            raise ProviderError(
                f"Could not reach Ollama at {self._api_base}. Start it with `ollama serve`."
            ) from error

        payload = response.json()
        models = {item.get("name") for item in payload.get("models", []) if item.get("name")}
        configured_model = self._settings.ollama.model
        if configured_model not in models:
            raise ConfigurationError(
                f"Ollama model `{configured_model}` is not available. Pull it with "
                f"`ollama pull {configured_model}`."
            )

    async def stream_chat(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
    ) -> AsyncIterator[str]:
        """Yield token chunks from the configured model."""
        try:
            from litellm import acompletion
        except ImportError as error:
            raise ConfigurationError(
                "LiteLLM is not installed. Run `uv sync --extra dev` or `pip install -e .[dev]`."
            ) from error

        await self.validate_connection()

        try:
            response_stream = await acompletion(
                model=self._model,
                api_base=self._api_base,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                stream=True,
                temperature=temperature,
                timeout=self._settings.ollama.request_timeout_seconds,
            )
            async for chunk in response_stream:
                text = _extract_chunk_text(chunk)
                if text:
                    yield text
        except Exception as error:  # noqa: BLE001
            raise ProviderError(f"Ollama streaming failed: {error}") from error

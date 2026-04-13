from __future__ import annotations

from collections.abc import AsyncIterator

from valocoach.cli.display import stream_markdown
from valocoach.core.config import AppSettings
from valocoach.core.models import CoachRequest
from valocoach.llm.prompts import build_system_prompt
from valocoach.llm.provider import LiteLLMOllamaProvider


class CoachService:
    """Week-one coaching service that proxies a prompt to Ollama."""

    def __init__(self, settings: AppSettings, provider: LiteLLMOllamaProvider) -> None:
        self._settings = settings
        self._provider = provider

    @classmethod
    def from_settings(cls, settings: AppSettings) -> CoachService:
        return cls(settings=settings, provider=LiteLLMOllamaProvider(settings))

    def stream_response(self, request: CoachRequest) -> AsyncIterator[str]:
        """Return the streamed model response for a coaching request."""
        return self._provider.stream_chat(
            system_prompt=build_system_prompt(self._settings.coach.patch_version),
            user_prompt=request.render_user_prompt(),
            temperature=self._settings.coach.temperature,
        )

    async def run_and_render(self, request: CoachRequest) -> str:
        """Render the streamed model response via Rich Live markdown."""
        return await stream_markdown(
            self.stream_response(request),
            title="ValoCoach",
            border_style=self._settings.ui.coach_border_style,
            refresh_per_second=self._settings.ui.stream_refresh_per_second,
            show_thinking=self._settings.ui.show_thinking,
        )

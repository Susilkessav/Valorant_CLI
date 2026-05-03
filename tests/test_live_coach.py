"""End-to-end Ollama integration tests.

These tests call the real LLM stack (stream_completion → litellm → Ollama)
and are therefore opt-in.  They are skipped unless --live is passed to pytest:

    uv run pytest --live tests/test_live_coach.py -v

Prerequisites:
    1. Ollama is running:  ollama serve
    2. The configured model is pulled:  ollama pull qwen3:8b
    3. (Optional) The vector store has been seeded:  valocoach ingest --seed

What is mocked:
    - display.stream_to_panel  — Rich Live needs a TTY; replaced with a simple
                                  token collector that just joins and returns.
    - build_stats_context      — avoids needing a populated DB for the LLM test.
    - load_settings (partial)  — settings object supplied directly so no config
                                  file needs to exist on the CI runner.
    - check_riot_id / check_vector_store — let run_coach() proceed past the
                                  pre-flight warnings silently.

What is NOT mocked:
    - litellm.completion       — real HTTP call to Ollama.
    - valocoach.llm.provider.stream_completion — real token streaming.
    - _build_grounded_context  — real retrieval (empty result is fine).
    - fit_prompt               — real token counting and trimming.
"""

from __future__ import annotations

import httpx
import pytest

from valocoach.core.config import Settings
from valocoach.core.preflight import CheckResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_OK = CheckResult(ok=True, message="ok")
_NO_STATS = CheckResult(ok=False, message="no riot id", hint="")


def _live_settings() -> Settings:
    """Minimal settings for a local Ollama run."""
    return Settings(
        riot_name="",
        riot_tag="",
        riot_region="na",
        ollama_model="qwen3:8b",
        ollama_host="http://localhost:11434",
    )


def _ollama_reachable(settings: Settings) -> bool:
    """Return True when Ollama is reachable and the model is pulled."""
    try:
        resp = httpx.get(f"{settings.ollama_host}/api/tags", timeout=3.0)
        resp.raise_for_status()
        tags = resp.json().get("models", [])
        model = settings.ollama_model
        return any(
            t.get("name", "") == model
            or t.get("name", "").startswith(f"{model}:")
            or model.startswith(f"{t.get('name', '')}:")
            for t in tags
        )
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def live_settings() -> Settings:
    return _live_settings()


@pytest.fixture(autouse=True)
def _skip_if_ollama_unavailable(live_settings: Settings) -> None:
    """Auto-skip every test in this module when Ollama is not reachable."""
    if not _ollama_reachable(live_settings):
        pytest.skip(
            f"Ollama not reachable at {live_settings.ollama_host} "
            f"or model '{live_settings.ollama_model}' not pulled"
        )


def _make_token_collector():
    """Return a drop-in for display.stream_to_panel that captures full text."""
    from unittest.mock import patch

    collected: list[str] = []

    def _fake_stream_to_panel(token_stream, **_kwargs):
        tokens = list(token_stream)  # drain the iterator
        collected.extend(tokens)
        return "".join(tokens)

    return patch("valocoach.cli.commands.coach.display.stream_to_panel", _fake_stream_to_panel)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.live
class TestRunCoachLive:
    """End-to-end tests for run_coach() with a real Ollama back-end."""

    def _invoke(
        self,
        situation: str,
        live_settings: Settings,
        *,
        agent: str | None = None,
        map_: str | None = None,
        side: str | None = None,
    ) -> str | None:
        """Call run_coach with a real LLM but mocked display/stats/settings."""
        from unittest.mock import patch

        with (
            patch("valocoach.cli.commands.coach.load_settings", return_value=live_settings),
            patch("valocoach.cli.commands.coach.build_stats_context", return_value=None),
            patch("valocoach.core.preflight.check_riot_id", return_value=_NO_STATS),
            patch("valocoach.core.preflight.check_vector_store", return_value=_OK),
            _make_token_collector(),
        ):
            from valocoach.cli.commands.coach import run_coach

            return run_coach(situation, agent=agent, map_=map_, side=side, with_stats=False)

    def test_returns_nonempty_string(self, live_settings: Settings):
        """The LLM must return at least some content for a simple question."""
        result = self._invoke("push A site on Ascent as Jett", live_settings)
        assert isinstance(result, str)
        assert len(result) > 50, f"Response too short ({len(result)} chars): {result!r}"

    def test_response_contains_markdown_structure(self, live_settings: Settings):
        """The system prompt asks for markdown sections; verify at least one appears."""
        result = self._invoke("eco round on attack, we have 800 credits", live_settings) or ""
        # The prompt instructs sections with emoji + bold headers; at least one should appear
        has_markdown = "**" in result or "#" in result or "🎯" in result or "🛠" in result
        assert has_markdown, f"Expected markdown in response, got: {result[:300]!r}"

    def test_agent_flag_influences_response(self, live_settings: Settings):
        """Passing agent='Jett' should produce a response mentioning Jett or her abilities."""
        result = self._invoke(
            "we are stalling on attack and can't break through B site",
            live_settings,
            agent="Jett",
        )
        assert result is not None
        # The system prompt grounds the LLM on the provided agent — expect a reference
        jett_terms = {"Jett", "Tailwind", "Cloudburst", "Updraft", "Bladestorm", "dash"}
        mentioned = any(t.lower() in (result or "").lower() for t in jett_terms)
        assert mentioned, (
            f"Expected Jett-related term in response. Got (first 400 chars):\n{result[:400]}"
        )

    def test_map_flag_influences_response(self, live_settings: Settings):
        """Passing map_='Ascent' should produce a response that references Ascent callouts."""
        result = self._invoke(
            "how do I push A site effectively",
            live_settings,
            map_="Ascent",
        )
        assert result is not None
        ascent_callouts = {"Ascent", "A Long", "A Main", "CT", "A Site", "Elbow", "Market"}
        mentioned = any(c.lower() in (result or "").lower() for c in ascent_callouts)
        assert mentioned, (
            f"Expected Ascent callout in response. Got (first 400 chars):\n{result[:400]}"
        )

    def test_multiple_calls_each_return_string(self, live_settings: Settings):
        """Two consecutive calls both succeed — verify no stale state between calls."""
        r1 = self._invoke("push B site as Reyna on Bind", live_settings)
        r2 = self._invoke("defend A site as Killjoy", live_settings)
        assert isinstance(r1, str) and len(r1) > 0
        assert isinstance(r2, str) and len(r2) > 0

    def test_side_flag_forwarded(self, live_settings: Settings):
        """side='defense' is forwarded; the response should be defence-framing."""
        result = self._invoke(
            "they always hit A with 5 players every round",
            live_settings,
            side="defense",
        )
        assert result is not None and len(result) > 20


@pytest.mark.live
class TestStreamCompletionLive:
    """Thin tests for the LLM provider layer in isolation."""

    def test_stream_completion_yields_tokens(self, live_settings: Settings):
        """stream_completion should yield at least one non-empty token."""
        from valocoach.llm.provider import stream_completion

        tokens = list(
            stream_completion(
                live_settings,
                system_prompt="You are a helpful assistant. Reply in one sentence.",
                user_message="Say hello.",
            )
        )
        assert len(tokens) > 0, "Expected at least one token from the stream"
        full = "".join(tokens)
        assert len(full) > 0

    def test_stream_completion_returns_strings(self, live_settings: Settings):
        """Every token yielded must be a str."""
        from valocoach.llm.provider import stream_completion

        for token in stream_completion(
            live_settings,
            system_prompt="Reply in one word.",
            user_message="What colour is the sky?",
        ):
            assert isinstance(token, str), f"Expected str token, got {type(token)}: {token!r}"

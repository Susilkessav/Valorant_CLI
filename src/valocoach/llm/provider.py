from __future__ import annotations

import logging
from collections.abc import Iterator

import litellm

from valocoach.core.config import Settings

log = logging.getLogger(__name__)

# Silence LiteLLM's debug noise
litellm.suppress_debug_info = True


def stream_completion(
    settings: Settings,
    system_prompt: str,
    user_message: str,
    conversation_history: list[dict[str, str]] | None = None,
    stop: list[str] | None = None,
    max_tokens: int | None = None,
    num_ctx: int | None = None,
) -> Iterator[str]:
    """Yield content tokens from the LLM as they stream in.

    Provider-agnostic — swap models by changing settings.ollama_model
    to e.g. 'anthropic/claude-sonnet-4-5-20250929'.

    Args:
        settings:              Application settings (model, host, temp, tokens).
        system_prompt:         The full system prompt string.
        user_message:          The current turn's user message.
        conversation_history:  Optional prior turns as ``{"role", "content"}``
                               dicts (from ``ConversationMemory.messages``).
                               Inserted between the system message and the
                               current user message so the model sees full
                               multi-turn context.
    """
    model = settings.ollama_model
    is_ollama_model = model.startswith("ollama/")
    if not model.startswith(("ollama/", "anthropic/", "openai/")):
        # Default Ollama models need the prefix for LiteLLM
        model = f"ollama/{model}"
        is_ollama_model = True

    messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
    if conversation_history:
        messages.extend(conversation_history)
    messages.append({"role": "user", "content": user_message})

    completion_kwargs = {
        "model": model,
        "messages": messages,
        "temperature": settings.llm_temperature,
        "max_tokens": max_tokens if max_tokens is not None else settings.llm_max_tokens,
        "stream": True,
    }
    if is_ollama_model:
        completion_kwargs["api_base"] = settings.ollama_host
        # num_ctx controls Ollama's KV-cache / context window.  The default
        # (often 2048) is too small for large prompts like meta generation.
        # Pass through when the caller explicitly requests a larger window.
        if num_ctx is not None:
            completion_kwargs["options"] = {"num_ctx": num_ctx}
    if stop:
        completion_kwargs["stop"] = stop

    response = litellm.completion(**completion_kwargs)

    for chunk in response:
        delta = chunk.choices[0].delta
        content = getattr(delta, "content", None)
        if content:
            yield content


def call_llm(
    system: str,
    user: str,
    settings: Settings,
    max_tokens: int | None = None,
) -> str:
    """Non-streaming LLM call — returns the full response as a string.

    Convenience wrapper for cases where streaming is not needed (e.g. structured
    metadata extraction).  Pass ``max_tokens`` to cap the response length — useful
    for structured extraction calls that should never produce more than a short JSON.

    Returns empty string on any error.
    """
    try:
        model = settings.ollama_model
        is_ollama_model = model.startswith("ollama/")
        if not model.startswith(("ollama/", "anthropic/", "openai/")):
            model = f"ollama/{model}"
            is_ollama_model = True

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        completion_kwargs: dict = {
            "model": model,
            "messages": messages,
            "temperature": settings.llm_temperature,
            "max_tokens": max_tokens if max_tokens is not None else settings.llm_max_tokens,
            "stream": False,
        }
        if is_ollama_model:
            completion_kwargs["api_base"] = settings.ollama_host

        response = litellm.completion(**completion_kwargs)
        return response.choices[0].message.content or ""
    except Exception as exc:
        # WARNING (not silent) so operators see Ollama/auth/rate-limit failures
        # rather than mistaking them for "the LLM couldn't decide".
        log.warning("call_llm failed: %s", exc, exc_info=log.isEnabledFor(logging.DEBUG))
        return ""

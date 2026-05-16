from __future__ import annotations

from collections.abc import Iterator

import litellm

from valocoach.core.config import Settings

# Silence LiteLLM's debug noise
litellm.suppress_debug_info = True


def stream_completion(
    settings: Settings,
    system_prompt: str,
    user_message: str,
    conversation_history: list[dict[str, str]] | None = None,
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
        "max_tokens": settings.llm_max_tokens,
        "stream": True,
    }
    if is_ollama_model:
        completion_kwargs["api_base"] = settings.ollama_host

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

    Convenience wrapper around ``stream_completion`` for cases where
    streaming is not needed (e.g. structured metadata extraction).

    Returns empty string on any error.
    """
    try:
        tokens = list(stream_completion(settings, system, user))
        return "".join(tokens)
    except Exception:
        return ""

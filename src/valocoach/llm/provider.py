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
) -> Iterator[str]:
    """Yield content tokens from the LLM as they stream in.

    Provider-agnostic — swap models by changing settings.ollama_model
    to e.g. 'anthropic/claude-sonnet-4-5-20250929'.
    """
    model = settings.ollama_model
    if not model.startswith(("ollama/", "anthropic/", "openai/")):
        # Default Ollama models need the prefix for LiteLLM
        model = f"ollama/{model}"

    response = litellm.completion(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        api_base=settings.ollama_host,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
        stream=True,
    )

    for chunk in response:
        delta = chunk.choices[0].delta
        content = getattr(delta, "content", None)
        if content:
            yield content

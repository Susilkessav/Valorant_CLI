"""Adaptive prompt-context trimmer.

The coach command assembles three optional context blocks before calling the
LLM — a grounded-context string (RAG chunks + JSON facts) and a stats-context
string (player recent form).  Under normal operation these total 2-5 K tokens
and the combined prompt is well inside Qwen3's 40 K context window.

This module provides the safety net for edge cases: long situations, many RAG
chunks, verbose stats.  It trims the *lowest-priority* parts first so the
system prompt and the player's verbatim message are never touched.

Trimming priority (highest priority = preserved longest):
  1. system_base   — the SYSTEM_PROMPT_STUB grounding rules (never touched)
  2. user_msg      — the player's verbatim situation text (never touched)
  3. grounded_ctx  — RAG chunks; cut from the end (vector hits first, JSON last)
  4. stats_ctx     — player stats; dropped entirely when grounded is tight

Token counting uses tiktoken cl100k_base.  This is the same tokeniser used by
GPT-4 / Ollama-hosted Qwen3 and matches the chunker — so token estimates here
are consistent with chunk-boundary estimates in the retrieval pipeline.
"""

from __future__ import annotations

from functools import lru_cache

import tiktoken

# ---------------------------------------------------------------------------
# Budget constants
# ---------------------------------------------------------------------------

# Hard upper bound on combined system + user input tokens.
# Qwen3 8B has a 40 K context window; we reserve 3 K for output (llm_max_tokens
# default) and 13 K as safety margin, leaving 24 K for the prompt.
# Callers can override — this default is conservative by design.
CONTEXT_HARD_LIMIT: int = 24_000

# When trimming is needed, reduce the grounded-context block to this many
# tokens.  That's roughly 3 full JSON-facts sections or 5-6 short RAG chunks —
# enough to keep the load-bearing agent/map facts while dropping weak hits.
GROUNDED_REDUCED_LIMIT: int = 2_000


# ---------------------------------------------------------------------------
# Tokeniser (lazy, cached)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _encoder() -> tiktoken.Encoding:
    return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """Return the number of tokens in *text* using the cl100k_base tokeniser.

    This is the same encoding used throughout the chunker so token estimates
    are consistent across the pipeline.

    Args:
        text: Any string — empty string returns 0.

    Returns:
        Token count as a non-negative integer.
    """
    if not text:
        return 0
    return len(_encoder().encode(text))


def trim_text_to_tokens(text: str, max_tokens: int) -> str:
    """Truncate *text* so it is at most *max_tokens* tokens long.

    Cuts from the *end* of the text so the beginning (highest-priority
    content) is preserved.  The cut is at an exact token boundary so the
    result is always lossless on the kept portion.

    Args:
        text: The string to (possibly) shorten.
        max_tokens: Maximum allowed tokens in the result.  Must be ≥ 0.

    Returns:
        *text* unchanged if it already fits; otherwise a prefix of *text*
        decoded from exactly *max_tokens* tokens.
    """
    if max_tokens <= 0:
        return ""
    enc = _encoder()
    tokens = enc.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return enc.decode(tokens[:max_tokens])


# ---------------------------------------------------------------------------
# Prompt fitter
# ---------------------------------------------------------------------------


def fit_prompt(
    *,
    system_base: str,
    grounded_context: str | None,
    stats_context: str | None,
    user_msg: str,
    hard_limit: int = CONTEXT_HARD_LIMIT,
) -> tuple[str | None, str | None]:
    """Return (grounded_context, stats_context) trimmed to fit within *hard_limit*.

    The system-base string and user message are treated as immovable — they
    are counted against the budget but never modified.  The optional context
    blocks are trimmed in priority order when the total would otherwise exceed
    *hard_limit*.

    Trimming stages (applied in sequence until the budget fits):
        1. Trim grounded_context to GROUNDED_REDUCED_LIMIT tokens.
        2. Drop stats_context entirely.
        3. Trim grounded_context further to whatever tokens remain.

    Args:
        system_base:       The SYSTEM_PROMPT_STUB (immovable).
        grounded_context:  RAG + JSON facts block, or None.
        stats_context:     Player stats snippet, or None.
        user_msg:          The verbatim situation message (immovable).
        hard_limit:        Token ceiling for the entire prompt.

    Returns:
        A (grounded_context, stats_context) tuple whose values are either the
        original strings, trimmed strings, or None — whichever keeps the total
        prompt within *hard_limit*.
    """
    base_tokens = count_tokens(system_base) + count_tokens(user_msg)
    available = hard_limit - base_tokens

    if available <= 0:
        # Edge case: base + user already blows the limit — can't help.
        return None, None

    g_tokens = count_tokens(grounded_context or "")
    s_tokens = count_tokens(stats_context or "")

    # Happy path — everything fits.
    if g_tokens + s_tokens <= available:
        return grounded_context, stats_context

    # Stage 1: Trim grounded to the reduced limit.
    if grounded_context and g_tokens > GROUNDED_REDUCED_LIMIT:
        grounded_context = trim_text_to_tokens(grounded_context, GROUNDED_REDUCED_LIMIT)
        g_tokens = count_tokens(grounded_context)

    if g_tokens + s_tokens <= available:
        return grounded_context, stats_context

    # Stage 2: Drop stats entirely.
    if g_tokens <= available:
        return grounded_context, None

    # Stage 3: Trim grounded further to whatever remains.
    grounded_context = trim_text_to_tokens(grounded_context or "", available)
    return grounded_context, None

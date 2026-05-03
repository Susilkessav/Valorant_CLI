"""Sliding-window conversation memory for the interactive REPL.

Keeps the last N turns in verbatim form and exposes them as the
``messages`` list that LiteLLM's ``completion()`` expects.  When the
history grows beyond ``max_tokens``, the oldest complete exchange
(user + assistant pair) is dropped until the window fits.

Design notes:
  - Turns are stored as plain ``{"role": str, "content": str}`` dicts —
    no abstraction needed; that's exactly what the LLM provider expects.
  - ``max_turns`` bounds the raw list length independently of token count.
    This prevents unbounded growth even if individual turns are tiny.
  - Token counting uses the same ``count_tokens`` helper as
    ``context_budget`` so estimates are consistent across the pipeline.
  - No LLM-based summarisation (that's a stretch goal).  Dropping the
    oldest pair is a clean, deterministic, always-safe fallback that
    needs no network call and produces no hallucination risk.
"""

from __future__ import annotations

from valocoach.core.context_budget import count_tokens


class ConversationMemory:
    """Fixed-capacity conversation window with token-aware eviction.

    Args:
        max_turns:  Maximum number of *individual* messages (user + assistant
                    each count as 1).  Defaults to 20, giving 10 exchanges.
        max_tokens: Maximum combined token count across all retained messages.
                    The oldest complete exchange is dropped whenever adding a
                    new message would push the window over this limit.
                    Defaults to 3 000.

    Usage::

        mem = ConversationMemory()
        mem.add("user", "How do I use Jett on Ascent?")
        mem.add("assistant", "Here's a plan…")

        # Pass to stream_completion as conversation_history:
        history = mem.messages
        mem.add("user", "What about B site instead?")
    """

    def __init__(self, *, max_turns: int = 20, max_tokens: int = 3_000) -> None:
        self._max_turns = max_turns
        self._max_tokens = max_tokens
        self._turns: list[dict[str, str]] = []

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add(self, role: str, content: str) -> None:
        """Append a turn and evict old content if needed.

        Args:
            role:    "user" or "assistant".
            content: The message text.
        """
        self._turns.append({"role": role, "content": content})
        self._evict()

    def clear(self) -> None:
        """Remove all stored turns."""
        self._turns.clear()

    # ------------------------------------------------------------------
    # Read access
    # ------------------------------------------------------------------

    @property
    def messages(self) -> list[dict[str, str]]:
        """The retained turns as a list of ``{"role", "content"}`` dicts.

        Returns a *copy* — both the outer list and each inner dict are
        independent of internal state, so callers (LiteLLM, tests, the REPL)
        can mutate the result without leaking back into the memory window.

        Safe to pass directly to ``stream_completion(conversation_history=…)``
        or to LiteLLM's ``completion(messages=…)`` as the non-system /
        non-final-user portion.
        """
        return [dict(t) for t in self._turns]

    @property
    def is_empty(self) -> bool:
        """True when no turns have been added (or after ``clear()``)."""
        return len(self._turns) == 0

    @property
    def token_count(self) -> int:
        """Current combined token count of all retained turns."""
        return sum(count_tokens(t["content"]) for t in self._turns)

    def __len__(self) -> int:
        return len(self._turns)

    # ------------------------------------------------------------------
    # Eviction
    # ------------------------------------------------------------------

    def _evict(self) -> None:
        """Drop the oldest complete exchange until the window fits.

        A "complete exchange" is one user + one assistant message.  If the
        oldest pair cannot be evicted cleanly (e.g. the first message is
        an assistant reply with no preceding user message), the oldest
        single message is dropped instead.
        """
        while self._turns and (
            len(self._turns) > self._max_turns or self.token_count > self._max_tokens
        ):
            # Try to drop a complete user+assistant exchange from the front.
            if (
                len(self._turns) >= 2
                and self._turns[0]["role"] == "user"
                and self._turns[1]["role"] == "assistant"
            ):
                self._turns.pop(0)
                self._turns.pop(0)
            else:
                # Partial exchange — drop just the oldest message.
                self._turns.pop(0)

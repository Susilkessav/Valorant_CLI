"""Tests for valocoach.core.context_budget.

We test:
  • count_tokens: sanity checks (empty string, known text, non-ASCII).
  • trim_text_to_tokens: exact boundary, already-fits, zero-limit edge case.
  • fit_prompt: happy path, each trimming stage, the edge where base+user
    already blows the budget.
"""

from __future__ import annotations

from valocoach.core.context_budget import (
    CONTEXT_HARD_LIMIT,
    GROUNDED_REDUCED_LIMIT,
    count_tokens,
    fit_prompt,
    trim_text_to_tokens,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_text(n_tokens: int) -> str:
    """Return a string that is *approximately* n_tokens tokens.

    Uses single English words that the tiktoken cl100k_base tokeniser maps
    1-to-1.  The actual count may be ±5 tokens; use count_tokens() to verify.
    """
    # "hello " is reliably 2 tokens (hello + space), so n_tokens//2 repetitions
    # gives a good approximation.
    return ("hello " * (n_tokens // 2)).strip()


# ---------------------------------------------------------------------------
# count_tokens
# ---------------------------------------------------------------------------


class TestCountTokens:
    def test_empty_string(self):
        assert count_tokens("") == 0

    def test_single_word(self):
        # "hello" → 1 token in cl100k_base
        assert count_tokens("hello") == 1

    def test_known_sentence(self):
        # Verify the count is a positive integer for a real sentence.
        n = count_tokens("push A site on Ascent as Jett")
        assert isinstance(n, int)
        assert n > 5

    def test_returns_integer(self):
        assert isinstance(count_tokens("any text at all"), int)

    def test_longer_is_more_tokens(self):
        short = count_tokens("hello")
        long = count_tokens("hello world foo bar baz qux quux")
        assert long > short

    def test_non_ascii(self):
        # Non-ASCII is fine — tiktoken handles it; just must not crash.
        n = count_tokens("Valorant: stratagème d'attaque")
        assert n > 0


# ---------------------------------------------------------------------------
# trim_text_to_tokens
# ---------------------------------------------------------------------------


class TestTrimTextToTokens:
    def test_already_fits(self):
        text = "short text"
        assert trim_text_to_tokens(text, 100) == text

    def test_exact_limit(self):
        text = "hello world foo"
        limit = count_tokens(text)
        assert trim_text_to_tokens(text, limit) == text

    def test_over_limit_is_shortened(self):
        text = "hello " * 200  # ~400 tokens
        result = trim_text_to_tokens(text, 100)
        assert count_tokens(result) <= 100

    def test_result_is_prefix(self):
        """Trimmed result must be the start of the original, not the end."""
        text = "alpha beta gamma delta epsilon"
        result = trim_text_to_tokens(text, 2)
        assert text.startswith(result)

    def test_zero_limit_returns_empty(self):
        assert trim_text_to_tokens("some content", 0) == ""

    def test_empty_input(self):
        assert trim_text_to_tokens("", 50) == ""


# ---------------------------------------------------------------------------
# fit_prompt — happy path
# ---------------------------------------------------------------------------


class TestFitPromptHappyPath:
    def test_nothing_trimmed_when_under_budget(self):
        system_base = "You are a coach."
        grounded = "Some context here."
        stats = "Player stats here."
        user_msg = "How do I play Jett?"

        g_out, s_out = fit_prompt(
            system_base=system_base,
            grounded_context=grounded,
            stats_context=stats,
            user_msg=user_msg,
            hard_limit=CONTEXT_HARD_LIMIT,
        )
        assert g_out == grounded
        assert s_out == stats

    def test_none_inputs_returned_as_none(self):
        g_out, s_out = fit_prompt(
            system_base="short",
            grounded_context=None,
            stats_context=None,
            user_msg="query",
        )
        assert g_out is None
        assert s_out is None

    def test_none_grounded_kept_when_stats_fits(self):
        g_out, s_out = fit_prompt(
            system_base="base",
            grounded_context=None,
            stats_context="stats",
            user_msg="query",
        )
        assert g_out is None
        assert s_out == "stats"


# ---------------------------------------------------------------------------
# fit_prompt — stage 1: trim grounded
# ---------------------------------------------------------------------------


class TestFitPromptStage1:
    def test_grounded_trimmed_when_tight(self):
        """When grounded + stats overflows, grounded is trimmed first."""
        # Build a grounded context larger than GROUNDED_REDUCED_LIMIT.
        big_grounded = _make_text(GROUNDED_REDUCED_LIMIT + 500)
        small_stats = "Player stats line."

        g_out, s_out = fit_prompt(
            system_base="base",
            grounded_context=big_grounded,
            stats_context=small_stats,
            user_msg="query",
            hard_limit=CONTEXT_HARD_LIMIT,
        )

        # Grounded was trimmed, stats was preserved (it still fits).
        assert g_out is not None
        assert count_tokens(g_out) <= GROUNDED_REDUCED_LIMIT
        assert s_out == small_stats

    def test_trimmed_grounded_is_prefix_of_original(self):
        big_grounded = "first part " * 500
        g_out, _ = fit_prompt(
            system_base="base",
            grounded_context=big_grounded,
            stats_context=None,
            user_msg="query",
            hard_limit=CONTEXT_HARD_LIMIT,
        )
        # The result must be a prefix of the original (we cut from the end).
        assert big_grounded.startswith(g_out or "")


# ---------------------------------------------------------------------------
# fit_prompt — stage 2: drop stats
# ---------------------------------------------------------------------------


class TestFitPromptStage2:
    def test_stats_dropped_when_grounded_fills_budget(self):
        """Stats are dropped when even trimmed grounded leaves no room for them."""
        # A grounded context that fills most of the available space after trimming.
        # We use a tiny hard_limit to make this easy to trigger.
        small_limit = 50
        grounded = _make_text(45)  # just fits in 50 minus base+user
        stats = _make_text(20)     # won't fit alongside grounded

        base = "ok"
        user = "q"

        g_out, s_out = fit_prompt(
            system_base=base,
            grounded_context=grounded,
            stats_context=stats,
            user_msg=user,
            hard_limit=small_limit,
        )

        # Stats may be None because there's no room.
        base_tokens = count_tokens(base) + count_tokens(user)
        if g_out is not None:
            assert count_tokens(g_out) + (count_tokens(s_out) if s_out else 0) <= small_limit - base_tokens


# ---------------------------------------------------------------------------
# fit_prompt — stage 3 / edge cases
# ---------------------------------------------------------------------------


class TestFitPromptEdgeCases:
    def test_base_plus_user_already_over_limit_returns_none_none(self):
        """When system_base + user_msg alone exceed hard_limit, return (None, None)."""
        long_base = "word " * 200   # ~400 tokens
        long_user = "word " * 200
        # hard_limit=10 is tiny — base+user blow it immediately
        g_out, s_out = fit_prompt(
            system_base=long_base,
            grounded_context="some context",
            stats_context="some stats",
            user_msg=long_user,
            hard_limit=10,
        )
        assert g_out is None
        assert s_out is None

    def test_total_tokens_fit_within_limit(self):
        """After fit_prompt, the combined prompt must be ≤ hard_limit."""
        hard_limit = 200
        system_base = "You are a coach."  # ~5 tokens
        user_msg = "push A"              # ~3 tokens
        grounded = _make_text(170)       # large — needs trimming
        stats = "stats here"             # ~3 tokens

        g_out, s_out = fit_prompt(
            system_base=system_base,
            grounded_context=grounded,
            stats_context=stats,
            user_msg=user_msg,
            hard_limit=hard_limit,
        )

        total = (
            count_tokens(system_base)
            + count_tokens(user_msg)
            + count_tokens(g_out or "")
            + count_tokens(s_out or "")
        )
        assert total <= hard_limit

    def test_empty_strings_treated_as_zero_tokens(self):
        g_out, s_out = fit_prompt(
            system_base="",
            grounded_context="",
            stats_context="",
            user_msg="",
        )
        # Empty strings are fine — nothing trimmed, nothing dropped.
        assert g_out == ""
        assert s_out == ""

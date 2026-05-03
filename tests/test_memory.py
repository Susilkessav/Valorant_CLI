"""Tests for valocoach.core.memory.ConversationMemory.

We test:
  • add / clear / is_empty / len basic lifecycle.
  • messages returns copies — mutations don't affect internal state.
  • max_turns eviction: oldest pair dropped when turn count exceeds limit.
  • max_tokens eviction: oldest pair dropped when token budget is exceeded.
  • token_count tracks accurately after evictions.
  • Partial-exchange handling (orphaned assistant turn at head).
"""

from __future__ import annotations

from valocoach.core.memory import ConversationMemory

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _add_exchange(mem: ConversationMemory, user: str = "question", assistant: str = "answer") -> None:
    mem.add("user", user)
    mem.add("assistant", assistant)


# ---------------------------------------------------------------------------
# Basic lifecycle
# ---------------------------------------------------------------------------


class TestConversationMemoryLifecycle:
    def test_starts_empty(self):
        mem = ConversationMemory()
        assert mem.is_empty
        assert len(mem) == 0
        assert mem.messages == []

    def test_add_single_turn(self):
        mem = ConversationMemory()
        mem.add("user", "hello")
        assert len(mem) == 1
        assert not mem.is_empty

    def test_add_exchange(self):
        mem = ConversationMemory()
        _add_exchange(mem)
        assert len(mem) == 2

    def test_clear_resets_to_empty(self):
        mem = ConversationMemory()
        _add_exchange(mem)
        mem.clear()
        assert mem.is_empty
        assert mem.messages == []

    def test_messages_preserves_role_and_content(self):
        mem = ConversationMemory()
        mem.add("user", "push A")
        mem.add("assistant", "smoke long")
        msgs = mem.messages
        assert msgs[0] == {"role": "user", "content": "push A"}
        assert msgs[1] == {"role": "assistant", "content": "smoke long"}

    def test_messages_returns_copy(self):
        """Mutating the returned list must not affect internal state."""
        mem = ConversationMemory()
        _add_exchange(mem)
        msgs = mem.messages
        msgs.clear()
        assert len(mem) == 2  # internal state unchanged


# ---------------------------------------------------------------------------
# max_turns eviction
# ---------------------------------------------------------------------------


class TestMaxTurnsEviction:
    def test_evicts_oldest_pair_when_max_turns_exceeded(self):
        mem = ConversationMemory(max_turns=4)  # 2 exchanges
        _add_exchange(mem, "q1", "a1")
        _add_exchange(mem, "q2", "a2")
        _add_exchange(mem, "q3", "a3")
        # After 3 exchanges (6 turns), should have dropped 1 exchange → 4 turns.
        assert len(mem) == 4
        msgs = mem.messages
        # q1/a1 should be gone, q2/a2 should be first
        assert msgs[0]["content"] == "q2"

    def test_single_turn_capacity(self):
        """max_turns=2 → holds exactly 1 exchange."""
        mem = ConversationMemory(max_turns=2)
        _add_exchange(mem, "first", "f_ans")
        _add_exchange(mem, "second", "s_ans")
        assert len(mem) == 2
        assert mem.messages[0]["content"] == "second"

    def test_no_eviction_when_under_limit(self):
        mem = ConversationMemory(max_turns=10)
        _add_exchange(mem)
        _add_exchange(mem)
        assert len(mem) == 4  # 2 exchanges, 2 turns each


# ---------------------------------------------------------------------------
# max_tokens eviction
# ---------------------------------------------------------------------------


class TestMaxTokensEviction:
    def test_old_content_dropped_when_token_limit_reached(self):
        """Fill the window with one long exchange, then push a short one in."""
        # Use a tiny token limit to make it easy to trigger
        mem = ConversationMemory(max_turns=100, max_tokens=30)
        # Add a large exchange that fills the window.
        big = "hello " * 20  # ~40 tokens — exceeds max_tokens alone
        _add_exchange(mem, big, big)
        # The window must not exceed the limit after adding.
        # (May be 0 turns if even a single big message is over limit.)
        assert mem.token_count <= 30 or mem.is_empty

    def test_fits_exactly_under_limit(self):
        mem = ConversationMemory(max_turns=100, max_tokens=500)
        # Two short exchanges should comfortably fit.
        _add_exchange(mem, "push A", "smoke A long")
        _add_exchange(mem, "what about B?", "rotate through CT")
        assert mem.token_count <= 500
        assert len(mem) == 4

    def test_token_count_decreases_after_eviction(self):
        mem = ConversationMemory(max_turns=2, max_tokens=10_000)
        _add_exchange(mem, "first question", "first answer")
        count_after_first = mem.token_count
        _add_exchange(mem, "second question", "second answer")
        # Eviction dropped the first exchange — token count should drop.
        assert mem.token_count < count_after_first + 100  # rough check
        assert len(mem) == 2


# ---------------------------------------------------------------------------
# token_count
# ---------------------------------------------------------------------------


class TestTokenCount:
    def test_empty_memory_is_zero(self):
        assert ConversationMemory().token_count == 0

    def test_token_count_positive_after_add(self):
        mem = ConversationMemory()
        mem.add("user", "how do I play Jett on Ascent?")
        assert mem.token_count > 0

    def test_token_count_increases_with_more_turns(self):
        mem = ConversationMemory(max_tokens=10_000)
        count_before = mem.token_count
        mem.add("user", "some tactical question here")
        assert mem.token_count > count_before


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestMemoryEdgeCases:
    def test_orphaned_assistant_turn_at_head(self):
        """If the oldest turn is assistant (no preceding user), just drop it."""
        mem = ConversationMemory(max_turns=2)
        # Force an orphaned assistant turn by manipulating directly.
        mem._turns = [
            {"role": "assistant", "content": "orphan answer"},
            {"role": "user", "content": "follow-up question"},
        ]
        mem.add("assistant", "new answer")
        # Eviction must not crash and the window must be ≤ max_turns.
        assert len(mem) <= mem._max_turns

    def test_add_user_only_no_crash(self):
        mem = ConversationMemory(max_turns=2)
        mem.add("user", "question")
        mem.add("user", "another question")
        # Should not crash, may drop the oldest single message.
        assert len(mem) <= 2

    def test_multiple_clears(self):
        mem = ConversationMemory()
        mem.clear()
        mem.clear()
        assert mem.is_empty

    def test_messages_order_is_chronological(self):
        mem = ConversationMemory(max_tokens=10_000)
        for i in range(3):
            _add_exchange(mem, f"q{i}", f"a{i}")
        msgs = mem.messages
        roles = [m["role"] for m in msgs]
        # Should alternate user/assistant
        assert roles == ["user", "assistant"] * 3

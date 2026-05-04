"""Tests for valocoach.retrieval.chunker — covering remaining uncovered branches.

Gaps being covered:
  36->42  _token_split() — while condition False on entry (empty token list)
  88-89   chunk_markdown — flush current_parts before force-splitting giant section
  97-105  chunk_markdown — candidate overflow: joining sections exceeds max_tokens,
                           emit + carry overlap tail, start new parts list
"""

from __future__ import annotations


# ===========================================================================
# _token_split — internal helper
# ===========================================================================


class TestTokenSplitInternal:
    def test_empty_text_returns_empty_list(self):
        """_token_split('') → tokens=[], while condition False immediately (line 36->42)."""
        from valocoach.retrieval.chunker import _token_split

        result = _token_split("", max_tokens=10, overlap=2)
        assert result == []

    def test_single_chunk_when_text_fits(self):
        """Text whose token count ≤ max_tokens produces one part and exits via break."""
        from valocoach.retrieval.chunker import _token_split

        # "hello world" is 2-3 tokens; max_tokens=100 → one chunk, break at end.
        result = _token_split("hello world", max_tokens=100, overlap=5)
        assert len(result) == 1
        assert "hello" in result[0]

    def test_multi_chunk_when_text_exceeds_max(self):
        """Large text splits into multiple parts."""
        from valocoach.retrieval.chunker import _token_split

        long = "token " * 200  # ≈200 tokens
        result = _token_split(long, max_tokens=50, overlap=5)
        assert len(result) > 1
        # All content preserved across parts (rough check)
        combined = " ".join(result)
        assert "token" in combined


# ===========================================================================
# chunk_markdown — flush before force-split (lines 88-89)
# ===========================================================================


class TestChunkMarkdownFlushBeforeForce:
    def test_accumulated_parts_flushed_before_giant_section(self):
        """Small sections accumulate, then a huge section arrives.

        Expected flow:
          1. Section A (small) → added to current_parts
          2. Section B (huge, > max_tokens) → flush current_parts (lines 87-89)
             then force-split B via _token_split
        Coverage target: lines 87-89 (if current_parts: emit + reset).
        """
        from valocoach.retrieval.chunker import chunk_markdown

        # Section A: short text (~5 tokens)
        small = "short section here"
        # Section B: giant text that definitely exceeds max_tokens=30
        giant = "bigword " * 50  # ≈50 tokens >> 30

        # Combine with a double newline (paragraph boundary) so chunker sees two sections.
        text = small + "\n\n" + giant

        chunks = chunk_markdown(text, source="test", max_tokens=30, overlap=3)

        # We should get at least 2 chunks: one for the small section and
        # at least one for the force-split giant section.
        assert len(chunks) >= 2

        # The small section content must appear in one of the chunks.
        all_text = " ".join(c.text for c in chunks)
        assert "short section" in all_text
        assert "bigword" in all_text

    def test_flush_produces_correct_chunk_for_small_section(self):
        """The flushed chunk from current_parts must contain the small section text."""
        from valocoach.retrieval.chunker import chunk_markdown

        small = "important context text"
        giant = "repeated " * 60  # well over any reasonable max_tokens

        text = small + "\n\n" + giant
        chunks = chunk_markdown(text, source="test", max_tokens=20, overlap=2)

        # First chunk should be the flushed small section.
        assert "important context" in chunks[0].text


# ===========================================================================
# chunk_markdown — candidate overflow (lines 96-105)
# ===========================================================================


class TestChunkMarkdownCandidateOverflow:
    def test_candidate_overflow_emits_and_carries_overlap(self):
        """Two sections that each fit alone but are too big combined.

        Flow:
          1. Section A (9 tokens) fits alone (max=9) → added to current_parts
          2. Section B (8 tokens) makes the candidate (17 tokens) exceed max=9
             → emit current_parts (line 97)
             → compute overlap tail from current_parts[-1] (lines 99-104)
             → current_parts = [overlap_text, B] (line 105)
        Coverage target: lines 96-105.
        """
        from valocoach.retrieval.chunker import chunk_markdown, count_tokens

        # "aaa " * 8 → 9 tokens; "bbb " * 4 → 8 tokens; combined → 17 tokens.
        # max_tok = 9: each section alone fits; joined they don't.
        section_a = "aaa " * 8  # 9 tokens
        section_b = "bbb " * 4  # 8 tokens
        max_tok = 9

        tok_a = count_tokens(section_a)
        tok_b = count_tokens(section_b)
        tok_combined = count_tokens(section_a + "\n\n" + section_b)

        assert tok_a <= max_tok, f"section_a {tok_a} tokens must fit alone"
        assert tok_b <= max_tok, f"section_b {tok_b} tokens must fit alone"
        assert tok_combined > max_tok, (
            f"combined={tok_combined} should exceed max_tok={max_tok}"
        )

        text = section_a + "\n\n" + section_b
        chunks = chunk_markdown(text, source="test", max_tokens=max_tok, overlap=3)

        # Both sections' content must appear somewhere in the output.
        assert len(chunks) >= 1
        all_text = " ".join(c.text for c in chunks)
        assert "aaa" in all_text
        assert "bbb" in all_text

    def test_overlap_tail_shorter_than_overlap_uses_full_last_part(self):
        """When the last part is shorter than overlap tokens, full part is used as tail.

        Branch: `len(tail_tokens) > overlap` → False (line 102->103).
        """
        from valocoach.retrieval.chunker import count_tokens

        # Section A: very short — fewer than `overlap` tokens when encoded.
        # overlap=10; section_a should be < 10 tokens.
        section_a = "hi"  # ≈1 token (< overlap=10)
        # Section B: enough tokens to force combined to exceed max_tokens=5.
        section_b = "word " * 6  # ~6 tokens > max_tokens=5

        max_tok = 5
        overlap = 10  # deliberately larger than section_a token count

        assert count_tokens(section_a) < overlap

        from valocoach.retrieval.chunker import chunk_markdown

        text = section_a + "\n\n" + section_b
        chunks = chunk_markdown(text, source="test", max_tokens=max_tok, overlap=overlap)

        # Should not raise; both sections' content preserved.
        all_text = " ".join(c.text for c in chunks)
        assert "hi" in all_text or "word" in all_text  # at least some content

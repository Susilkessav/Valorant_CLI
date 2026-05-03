from __future__ import annotations

import re
from dataclasses import dataclass, field

import tiktoken

_enc: tiktoken.Encoding | None = None


def _get_enc() -> tiktoken.Encoding:
    global _enc
    if _enc is None:
        _enc = tiktoken.get_encoding("cl100k_base")
    return _enc


def count_tokens(text: str) -> int:
    return len(_get_enc().encode(text))


@dataclass
class Chunk:
    text: str
    source: str
    chunk_index: int
    metadata: dict = field(default_factory=dict)


def _token_split(text: str, max_tokens: int, overlap: int) -> list[str]:
    """Token-exact split for a single section that individually exceeds max_tokens."""
    enc = _get_enc()
    tokens = enc.encode(text)
    parts: list[str] = []
    start = 0
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        parts.append(enc.decode(tokens[start:end]))
        if end == len(tokens):
            break
        start = end - overlap
    return parts


def chunk_markdown(
    text: str,
    source: str,
    max_tokens: int = 400,
    overlap: int = 40,
    metadata: dict | None = None,
) -> list[Chunk]:
    """Split markdown text into token-bounded chunks, respecting heading and paragraph boundaries.

    Sections that fit are packed together up to max_tokens. A single section that
    individually exceeds max_tokens is force-split with token-level overlap. The
    overlap on normal chunk boundaries carries the token tail of the last section
    forward so context is preserved across boundaries.
    """
    base_meta = metadata or {}
    enc = _get_enc()

    # Primary split: markdown headings (##, ###) and blank lines
    sections = re.split(r"\n(?=#{1,3} )|(?:\n\n)", text)
    sections = [s.strip() for s in sections if s.strip()]

    chunks: list[Chunk] = []
    idx = 0
    current_parts: list[str] = []

    def emit(parts: list[str]) -> None:
        nonlocal idx
        chunks.append(Chunk(
            text="\n\n".join(parts),
            source=source,
            chunk_index=idx,
            metadata={**base_meta, "chunk_index": idx},
        ))
        idx += 1

    for section in sections:
        section_tokens = count_tokens(section)

        if section_tokens > max_tokens:
            # Flush any accumulated parts first, then force-split the giant section
            if current_parts:
                emit(current_parts)
                current_parts = []
            for piece in _token_split(section, max_tokens, overlap):
                emit([piece])
            continue

        # Check whether adding this section would overflow the current chunk
        candidate = "\n\n".join([*current_parts, section]) if current_parts else section
        if count_tokens(candidate) > max_tokens:
            emit(current_parts)
            # Carry a token-level overlap tail from the last accumulated part
            tail_tokens = enc.encode(current_parts[-1])
            overlap_text = enc.decode(tail_tokens[-overlap:]) if len(tail_tokens) > overlap else current_parts[-1]
            current_parts = [overlap_text, section]
        else:
            current_parts.append(section)

    if current_parts:
        emit(current_parts)

    return chunks


def chunk_text(text: str, max_tokens: int = 400, overlap: int = 40) -> list[str]:
    """Backward-compatible shim — returns plain strings instead of Chunk objects."""
    return [c.text for c in chunk_markdown(text, source="", max_tokens=max_tokens, overlap=overlap)]

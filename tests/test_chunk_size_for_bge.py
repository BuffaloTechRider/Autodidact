"""Tests for chunk-size defaults that fit BGE-large's 512-token context.

The bge-large-en-v1.5 embedding model accepts at most 512 tokens. The old
chunk_size default of 500 left only 12 tokens of headroom, and our
chars-per-token heuristic (4) underestimates code (more tokens per char),
so source-file chunks routinely overflowed and Ollama returned 500.

Fix: lower default chunk_size to 384, and clamp any individual chunk to
the safe character cap before returning it.
"""

from __future__ import annotations

import pytest

from autodidact.document_store import (
    _CHARS_PER_TOKEN,
    _SAFE_CHUNK_TOKEN_CAP,
    chunk_text,
)


# ── Defaults are safe ─────────────────────────────────────────────


class TestDefaultChunkSize:

    def test_default_chunk_size_is_safe_for_bge_large(self):
        """Default char target must leave headroom for BGE's 512-token context."""
        from autodidact.document_store import _DEFAULT_CHUNK_SIZE_TOKENS
        # We want at least ~25% headroom for token-dense content like code.
        assert _DEFAULT_CHUNK_SIZE_TOKENS <= 400, (
            f"Default chunk_size {_DEFAULT_CHUNK_SIZE_TOKENS} is too close to "
            "BGE's 512-token cap; code chunks will overflow."
        )

    def test_safe_token_cap_is_under_bge_limit(self):
        """The hard cap we enforce must be under 512 tokens."""
        assert _SAFE_CHUNK_TOKEN_CAP < 512


# ── Chunks never exceed the safe character cap ───────────────────


class TestChunksFitWithinCap:

    def test_long_text_chunks_under_cap(self):
        """Even on long inputs, no chunk should exceed _SAFE_CHUNK_TOKEN_CAP * _CHARS_PER_TOKEN."""
        # 50,000 chars of varied content.
        text = ("The quick brown fox jumps over the lazy dog. " * 1500)
        chunks = chunk_text(text)
        cap_chars = _SAFE_CHUNK_TOKEN_CAP * _CHARS_PER_TOKEN
        too_long = [i for i, c in enumerate(chunks) if len(c) > cap_chars]
        assert not too_long, (
            f"{len(too_long)} chunks exceeded the cap of {cap_chars} chars"
        )

    def test_python_source_chunks_under_cap(self):
        """Python source (token-dense) is the actual failure case."""
        # Synthesize a long python-like file with lots of symbols.
        line = (
            "    def some_method(self, x: int, y: str = 'default', *args, **kwargs) -> dict:\n"
            "        return {'x': x + 1, 'y': y.upper(), **kwargs}\n"
        )
        text = line * 200
        chunks = chunk_text(text)
        cap_chars = _SAFE_CHUNK_TOKEN_CAP * _CHARS_PER_TOKEN
        too_long = [c for c in chunks if len(c) > cap_chars]
        assert not too_long


# ── Hard truncation as last resort ────────────────────────────────


class TestHardTruncation:
    """Even when the splitter can't find a clean boundary, chunks must not exceed the cap."""

    def test_giant_token_blob_is_truncated(self):
        """A long run with no whitespace must still produce sub-cap chunks."""
        # 10,000 chars with no whitespace — splitter has nowhere to break.
        text = "x" * 10000
        chunks = chunk_text(text)
        cap_chars = _SAFE_CHUNK_TOKEN_CAP * _CHARS_PER_TOKEN
        for c in chunks:
            assert len(c) <= cap_chars, (
                f"chunk of {len(c)} chars exceeds cap {cap_chars}"
            )


# ── Existing behavior preserved ──────────────────────────────────


class TestPreservedBehavior:

    def test_short_text_one_chunk(self):
        chunks = chunk_text("hello world")
        assert chunks == ["hello world"]

    def test_empty_text_no_chunks(self):
        assert chunk_text("") == []
        assert chunk_text("   ") == []

    def test_explicit_chunk_size_still_works(self):
        """Callers passing a specific chunk_size are still respected (subject to the cap)."""
        text = "word " * 1000  # 5000 chars
        chunks = chunk_text(text, chunk_size=200, overlap=20)
        assert len(chunks) > 1

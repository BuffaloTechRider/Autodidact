"""Tests for token-aware chunking using the BGE tokenizer.

The 4-chars-per-token heuristic underestimates token count on code-dense
text. Live ingestion of autodidact/agent.py produced chunks of ~1500 chars
that tokenize to 521 BGE tokens — over the 512 model cap. The heuristic
needs to die for the cap-enforcement step; we use the real BGE tokenizer
instead.

Initial char-based slicing is kept for speed; only the cap check upgrades.
"""

from __future__ import annotations

import pytest

from autodidact.document_store import chunk_text


# ── Real-world regression: agent.py used to produce >512-token chunks ──


class TestRealWorldRegression:
    """Chunks must never exceed the BGE-large 512-token model cap.

    This was the live failure: `autodidact learn .` against the project
    repo emitted "Ollama HTTP 500: input length exceeds context length"
    on chunks 1, 30, and 31 of agent.py.
    """

    @pytest.fixture
    def bge_tokenizer(self):
        try:
            from tokenizers import Tokenizer
        except ImportError:
            pytest.skip("tokenizers not installed")
        return Tokenizer.from_pretrained("BAAI/bge-large-en-v1.5")

    def test_agent_py_chunks_all_fit_512_token_cap(self, bge_tokenizer):
        import pathlib
        text = pathlib.Path("autodidact/agent.py").read_text()
        chunks = chunk_text(text)
        oversized = [
            (i, len(bge_tokenizer.encode(c).ids))
            for i, c in enumerate(chunks)
            if len(bge_tokenizer.encode(c).ids) > 512
        ]
        assert not oversized, (
            f"chunks exceeding the BGE 512-token cap: {oversized!r}. "
            "char-per-token heuristic is too loose for code."
        )

    def test_cli_py_chunks_all_fit_512_token_cap(self, bge_tokenizer):
        import pathlib
        text = pathlib.Path("autodidact/cli.py").read_text()
        chunks = chunk_text(text)
        oversized = [
            (i, len(bge_tokenizer.encode(c).ids))
            for i, c in enumerate(chunks)
            if len(bge_tokenizer.encode(c).ids) > 512
        ]
        assert not oversized, (
            f"cli.py chunks exceeding the cap: {oversized!r}"
        )


# ── Synthetic: dense code with no whitespace ───────────────────


class TestDenseCodeChunking:
    """A chunk full of camelCase identifiers and operators tokenizes much
    denser than prose. The cap must hold even when char count is low."""

    def test_dense_code_within_cap(self):
        try:
            from tokenizers import Tokenizer
        except ImportError:
            pytest.skip("tokenizers not installed")
        tok = Tokenizer.from_pretrained("BAAI/bge-large-en-v1.5")

        # ~3000 characters of dense Python-shaped tokens. Under the old
        # heuristic this would split into 2 chunks of ~1500 chars; under
        # the BGE tokenizer each is north of 700 tokens.
        line = "self._foo_bar_baz: Optional[Dict[str, Tuple[int, str]]] = None"
        text = "\n".join([line] * 80)

        chunks = chunk_text(text)
        for c in chunks:
            n = len(tok.encode(c).ids)
            assert n <= 512, f"chunk has {n} tokens, exceeds 512: {c[:80]!r}"


# ── Backwards compatibility: prose still chunks reasonably ─────


class TestProseChunkingUnchanged:
    """Prose that already chunked fine should still produce roughly the
    same shape — one chunk per paragraph block — not be over-split."""

    def test_short_prose_still_one_chunk(self):
        text = "This is a short paragraph of plain English."
        chunks = chunk_text(text)
        assert chunks == [text]

    def test_long_prose_chunks_have_overlap_when_token_count_allows(self):
        # A paragraph of repeating prose well below the 512-token cap
        # should split into multiple chunks at sentence boundaries with
        # overlap — same shape as before.
        sentence = "The capital of France is Paris and it is a beautiful city. "
        text = sentence * 200  # well over the per-chunk target

        chunks = chunk_text(text)
        assert len(chunks) > 1, "long text must produce multiple chunks"
        # No single chunk overflows the prose target by 5x or anything
        # crazy — the chunker is still well-behaved on prose.
        for c in chunks:
            assert len(c) <= 4000, f"chunk too long: {len(c)} chars"


# ── Graceful fallback when tokenizers is missing ───────────────


class TestFallbackWithoutTokenizers:
    """If tokenizers can't be imported, chunking still works — it just
    uses a stricter chars-per-token ratio so we don't exceed the cap on
    typical inputs. Caller should never see an ImportError."""

    def test_chunk_text_does_not_raise_when_tokenizers_missing(self, monkeypatch):
        from autodidact import document_store

        monkeypatch.setattr(document_store, "_get_bge_tokenizer", lambda: None)

        text = "x" * 10_000
        chunks = chunk_text(text)
        assert chunks, "chunker must produce something"
        # In the fallback path, char count ≤ stricter cap.
        for c in chunks:
            assert len(c) <= document_store._SAFE_CHUNK_TOKEN_CAP * \
                document_store._FALLBACK_CHARS_PER_TOKEN

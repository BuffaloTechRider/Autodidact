"""Tests for heading-section-aware markdown chunking (chunk_markdown).

Replaces blind sliding-window chunking for markdown-structured prose. The two
properties that matter: (1) a section that fits stays intact — no mid-section
cut; (2) an oversized section's pieces each carry the enclosing heading trail,
so a chunk lifted from the middle of a long section still says what it's about.
"""

from __future__ import annotations

from autodidact.document_store import (
    chunk_markdown,
    _chunk_markdown_table,
    _segment_section_blocks,
    _SAFE_CHUNK_TOKEN_CAP,
)


class TestSectionIntegrity:
    def test_each_small_section_is_its_own_chunk(self):
        text = (
            "# Guide\n\nIntro paragraph.\n\n"
            "## Install\n\nRun pip install.\n\n"
            "## Deploy\n\nRun the deploy script.\n"
        )
        chunks = chunk_markdown(text)
        # Three headings → three sections, none merged, none split.
        assert len(chunks) == 3
        assert any("Install" in c and "pip install" in c for c in chunks)
        assert any("Deploy" in c and "deploy script" in c for c in chunks)

    def test_no_headings_falls_back_to_plain_chunking(self):
        text = "Just some prose with no headings at all. " * 3
        chunks = chunk_markdown(text)
        assert len(chunks) == 1
        assert "prose with no headings" in chunks[0]

    def test_empty_returns_empty(self):
        assert chunk_markdown("") == []
        assert chunk_markdown("   \n  \n") == []


class TestHeadingTrailOnSplit:
    def test_oversized_section_pieces_keep_heading_trail(self):
        # One deep section far larger than the target → must split, and
        # pieces after the first must carry the H1/H2 trail.
        big_body = "This sentence is filler. " * 400  # ~10k chars, well over target
        text = f"# Manual\n\n## Networking\n\n{big_body}\n"
        chunks = chunk_markdown(text, chunk_size=200, overlap=20)
        # The pieces carrying section body: the first opens with its own
        # heading; every subsequent one must re-state the full heading trail
        # so a mid-section chunk still knows its context.
        body_pieces = [c for c in chunks if "filler" in c]
        assert len(body_pieces) > 1, "expected the section to split"
        for piece in body_pieces[1:]:
            assert "# Manual" in piece and "## Networking" in piece
        # And the first body piece at least carries its immediate heading.
        assert "## Networking" in body_pieces[0]

    def test_nested_heading_trail_pops_correctly(self):
        # H2 under a different H1 should not inherit the first H1.
        text = (
            "# A\n\n## A1\n\nalpha\n\n"
            "# B\n\n## B1\n\nbeta\n"
        )
        chunks = chunk_markdown(text)
        b1 = [c for c in chunks if "beta" in c][0]
        # The B1 section belongs to B, not A.
        assert "# B" in b1 or "## B1" in b1
        assert "# A" not in b1


class TestTableAtomicity:
    """Tables must not be window-split mid-row; oversized tables repeat the header."""

    def _table(self, n_rows: int) -> str:
        head = "| Region | Revenue | Growth |\n|---|---|---|"
        rows = "\n".join(
            f"| Region{i} | ${i}.2M | {i}% |" for i in range(n_rows)
        )
        return head + "\n" + rows

    def test_small_table_stays_atomic_inside_section(self):
        # A section containing prose + a small table, but large enough overall
        # to force the oversized-section path.
        filler = "Context sentence. " * 300
        table = self._table(4)
        text = f"# Report\n\n## Q3\n\n{filler}\n\n{table}\n"
        chunks = chunk_markdown(text, chunk_size=200, overlap=20)
        # Exactly one chunk should contain the full table, all 4 rows together.
        table_chunks = [c for c in chunks if "Region0" in c]
        assert len(table_chunks) == 1
        for i in range(4):
            assert f"Region{i}" in table_chunks[0]
        # And that chunk carries the delimiter row (i.e. it's the real table).
        assert "|---|" in table_chunks[0].replace(" ", "")

    def test_segment_splits_prose_and_table(self):
        lines = [
            "Some intro prose.",
            "",
            "| A | B |",
            "|---|---|",
            "| 1 | 2 |",
            "| 3 | 4 |",
            "",
            "Trailing prose.",
        ]
        segs = _segment_section_blocks(lines)
        kinds = [k for k, _ in segs]
        assert kinds == ["prose", "table", "prose"]

    def test_oversized_table_row_splits_with_repeated_header(self):
        # A table alone bigger than the target must split by rows, each piece
        # repeating the "| Region | Revenue | Growth |" header + delimiter.
        big_table = self._table(200)
        pieces = _chunk_markdown_table(big_table, chunk_size=150)
        assert len(pieces) > 1
        for p in pieces:
            first_two = p.split("\n")[:2]
            assert "Region" in first_two[0] and "Revenue" in first_two[0]
            assert set(first_two[1].replace(" ", "")) <= set("|-:")

    def test_malformed_table_does_not_crash(self):
        # Pipes but no delimiter row → treated as prose, no header to repeat.
        text = "| looks like a table | but no delimiter |\n" * 50
        pieces = _chunk_markdown_table(text, chunk_size=100)
        assert pieces  # returns something, doesn't raise


class TestCap:
    def test_pieces_respect_token_cap(self):
        try:
            from tokenizers import Tokenizer
            tok = Tokenizer.from_pretrained("BAAI/bge-large-en-v1.5")
        except Exception:
            import pytest
            pytest.skip("tokenizers not installed")
        big_body = "word " * 4000
        text = f"# Doc\n\n## Section\n\n{big_body}\n"
        chunks = chunk_markdown(text, chunk_size=300, overlap=30)
        oversized = [i for i, c in enumerate(chunks)
                     if len(tok.encode(c).ids) > 512]
        assert not oversized, f"chunks over the 512 cap: {oversized}"

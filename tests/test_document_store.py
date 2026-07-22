"""Tests for the DocumentStore — document ingestion for cold start (R9).

TDD: tests written first, then implementation.

The DocumentStore handles:
- File walking (directory recursion, .gitignore support)
- Text chunking (~500 tokens, ~50 overlap, paragraph-aware)
- Embedding via the configured LLM client
- Storage in the document_chunks table (separate from knowledge_entries, per AD-002)
- Retrieval alongside agent memory with different prompt framing
- Deduplication on re-ingestion
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from autodidact.database import init_database
from autodidact.llm_client import LLMClient


# ── Test fixtures ────────────────────────────────────────────────

@pytest.fixture
def db_conn():
    """In-memory SQLite with the full schema including document_chunks."""
    return init_database(":memory:")


@pytest.fixture
def mock_embed_client():
    """Mock LLM client that returns deterministic 32-dim embeddings."""
    client = MagicMock(spec=LLMClient)
    rng = np.random.RandomState(42)

    def _embed(text: str) -> np.ndarray:
        # Deterministic per-text embedding for test stability.
        seed = abs(hash(text)) % (2**31)
        return np.random.RandomState(seed).randn(32).astype(np.float32)

    client.embed.side_effect = _embed
    return client


@pytest.fixture
def doc_store(db_conn, mock_embed_client):
    """A DocumentStore wired with the mock embedder and in-memory DB."""
    from autodidact.document_store import DocumentStore
    return DocumentStore(db_conn, mock_embed_client, embedding_dim=32)


# ── Chunking ──────────────────────────────────────────────────────

class TestChunking:
    """R9 AC5: Chunking — ~500 tokens per chunk, ~50 token overlap."""

    def test_short_text_is_single_chunk(self):
        """Text shorter than chunk size yields one chunk."""
        from autodidact.document_store import chunk_text

        chunks = chunk_text("This is a short document.", chunk_size=500, overlap=50)
        assert len(chunks) == 1
        assert chunks[0] == "This is a short document."

    def test_long_text_is_split(self):
        """Text longer than chunk size yields multiple chunks."""
        from autodidact.document_store import chunk_text

        # 3000 characters ≈ 750 tokens → should split into multiple chunks.
        long_text = "word " * 600  # 3000 chars
        chunks = chunk_text(long_text, chunk_size=500, overlap=50)
        assert len(chunks) > 1

    def test_chunks_have_overlap(self):
        """Consecutive chunks share overlap tokens for context continuity."""
        from autodidact.document_store import chunk_text

        # Generate text where we can detect overlap by shared words.
        words = [f"word{i}" for i in range(400)]
        long_text = " ".join(words)
        chunks = chunk_text(long_text, chunk_size=100, overlap=20)

        assert len(chunks) >= 2
        # Last words of chunk[0] should appear in beginning of chunk[1].
        tail = chunks[0].split()[-10:]
        head = chunks[1].split()[:30]
        overlap_found = any(word in head for word in tail)
        assert overlap_found, "Expected overlap between consecutive chunks"

    def test_chunks_prefer_paragraph_boundaries(self):
        """Splitter prefers to split on paragraph/line boundaries."""
        from autodidact.document_store import chunk_text

        text = (
            "First paragraph goes here with some words.\n\n"
            "Second paragraph follows it.\n\n"
            "Third paragraph wraps up."
        )
        chunks = chunk_text(text, chunk_size=50, overlap=10)
        # We don't assert exact count; we assert that paragraphs weren't
        # split mid-word (no chunk starts with a fragment of a word).
        for chunk in chunks:
            assert not chunk.startswith(" ")


# ── File walking ──────────────────────────────────────────────────

class TestFileWalking:
    """R9 AC2-3: Directory recursion, supported file types."""

    def test_walks_directory_recursively(self, tmp_path):
        """Nested directories are all walked."""
        from autodidact.document_store import walk_files

        (tmp_path / "a.md").write_text("a")
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "b.md").write_text("b")
        (tmp_path / "sub" / "nested").mkdir()
        (tmp_path / "sub" / "nested" / "c.md").write_text("c")

        files = list(walk_files(tmp_path))
        assert len(files) == 3

    def test_includes_supported_extensions(self, tmp_path):
        """Supported text extensions are included."""
        from autodidact.document_store import walk_files

        for ext in ("md", "txt", "py", "ts", "json", "yaml", "csv", "rst"):
            (tmp_path / f"file.{ext}").write_text("content")

        files = list(walk_files(tmp_path))
        assert len(files) == 8

    def test_excludes_binary_extensions(self, tmp_path):
        """Binary files (.jpg, .pdf without [pdf], .exe, etc.) are skipped."""
        from autodidact.document_store import walk_files

        (tmp_path / "keeper.md").write_text("text")
        (tmp_path / "image.jpg").write_bytes(b"\xff\xd8")
        (tmp_path / "binary.exe").write_bytes(b"\x4d\x5a")

        files = list(walk_files(tmp_path))
        assert len(files) == 1
        assert files[0].name == "keeper.md"

    def test_respects_gitignore(self, tmp_path):
        """Entries in .gitignore are skipped."""
        from autodidact.document_store import walk_files

        (tmp_path / ".gitignore").write_text("ignored.md\nbuild/\n")
        (tmp_path / "keeper.md").write_text("keep")
        (tmp_path / "ignored.md").write_text("skip")
        (tmp_path / "build").mkdir()
        (tmp_path / "build" / "artifact.md").write_text("skip")

        files = list(walk_files(tmp_path))
        names = {f.name for f in files}
        assert "keeper.md" in names
        assert "ignored.md" not in names
        assert "artifact.md" not in names

    def test_single_file_path(self, tmp_path):
        """A file path (not directory) returns just that file."""
        from autodidact.document_store import walk_files

        f = tmp_path / "single.md"
        f.write_text("content")

        files = list(walk_files(f))
        assert files == [f]


# ── Ingestion ─────────────────────────────────────────────────────

class TestIngestion:
    """R9 AC6-7: Embed chunks, store in document_chunks table."""

    def test_ingest_single_file(self, doc_store, tmp_path):
        """Ingesting a file stores chunks in the DB."""
        f = tmp_path / "doc.md"
        f.write_text("This is a short document for testing.")

        result = doc_store.ingest(f)

        assert result.files_ingested == 1
        assert result.chunks_created >= 1
        assert doc_store.count() >= 1

    def test_ingest_directory_multiple_files(self, doc_store, tmp_path):
        """Ingesting a directory walks and ingests all supported files."""
        (tmp_path / "a.md").write_text("First document content.")
        (tmp_path / "b.md").write_text("Second document content.")
        (tmp_path / "c.py").write_text("# Python file\nprint('hello')")

        result = doc_store.ingest(tmp_path)

        assert result.files_ingested == 3
        assert result.chunks_created >= 3

    def test_ingest_chunks_have_source_file(self, doc_store, tmp_path):
        """Each chunk records which source file it came from."""
        f = tmp_path / "doc.md"
        f.write_text("Some content here.")
        doc_store.ingest(f)

        chunks = doc_store.list_chunks()
        assert len(chunks) >= 1
        assert str(f) in chunks[0].source_file or f.name in chunks[0].source_file

    def test_ingest_chunks_have_embeddings(self, doc_store, tmp_path):
        """Each chunk is embedded and the embedding is persisted."""
        f = tmp_path / "doc.md"
        f.write_text("Content to embed.")
        doc_store.ingest(f)

        chunks = doc_store.list_chunks()
        assert len(chunks) >= 1
        assert chunks[0].embedding is not None
        assert len(chunks[0].embedding) == 32


# ── Parallel ingestion (file-level thread pool) ──────────────────

class TestParallelIngestion:
    """Ingest fans read+chunk+embed across a thread pool; DB writes stay
    on the calling thread. Results must match serial ingestion."""

    def _write_tree(self, tmp_path, n_files=6):
        for i in range(n_files):
            # Long enough to split into several chunks per file.
            (tmp_path / f"doc{i}.md").write_text(
                f"Document {i}. " + ("word " * 400)
            )

    def test_parallel_matches_serial_counts(self, doc_store, tmp_path):
        self._write_tree(tmp_path)
        parallel = doc_store.ingest(tmp_path, workers=4)
        parallel_count = doc_store.count()

        # Re-ingest the same tree serially; dedup means counts should match.
        serial = doc_store.ingest(tmp_path, workers=1)

        assert parallel.files_ingested == serial.files_ingested == 6
        assert parallel.chunks_created == serial.chunks_created
        assert doc_store.count() == parallel_count  # dedup, not additive

    def test_chunk_index_order_preserved(self, doc_store, tmp_path):
        """A multi-chunk file keeps contiguous 0..n-1 indices under the pool."""
        f = tmp_path / "big.md"
        f.write_text("Section. " + ("word " * 1200))
        doc_store.ingest(f, workers=4)

        chunks = [c for c in doc_store.list_chunks() if c.source_file.endswith("big.md")]
        indices = sorted(c.chunk_index for c in chunks)
        assert indices == list(range(len(chunks)))

    def test_default_workers_used_when_unspecified(self, db_conn, mock_embed_client, tmp_path):
        """ingest() with no workers arg falls back to the store's default."""
        from autodidact.document_store import DocumentStore
        store = DocumentStore(
            db_conn, mock_embed_client, embedding_dim=32, default_workers=1
        )
        self._write_tree(tmp_path, n_files=3)
        result = store.ingest(tmp_path)  # no workers= → uses default_workers=1
        assert result.files_ingested == 3

    def test_progress_fires_once_per_file(self, doc_store, tmp_path):
        self._write_tree(tmp_path, n_files=5)
        seen: list[str] = []
        doc_store.ingest(
            tmp_path, workers=4,
            on_progress=lambda e: seen.append(e["file"]) if e["type"] == "file_ingested" else None,
        )
        assert len(seen) == 5
        assert len(set(seen)) == 5  # no file persisted twice


# ── Deduplication ─────────────────────────────────────────────────

class TestDeduplication:
    """R9 AC9: Re-ingesting a file replaces existing chunks."""

    def test_reingestion_replaces_chunks(self, doc_store, tmp_path):
        """Ingesting the same file twice doesn't duplicate chunks."""
        f = tmp_path / "doc.md"
        f.write_text("Original content.")
        doc_store.ingest(f)
        first_count = doc_store.count()

        # Modify and re-ingest.
        f.write_text("Updated content that is completely different.")
        doc_store.ingest(f)
        second_count = doc_store.count()

        # Chunks from original should be gone; only new chunks remain.
        assert second_count == first_count  # or similar, not additive

        chunks = doc_store.list_chunks()
        assert all("Updated" in c.content or "completely" in c.content for c in chunks)


# ── Retrieval ─────────────────────────────────────────────────────

class TestRetrieval:
    """R9 AC8: Document chunks retrieved alongside agent memory at query time."""

    def test_search_returns_relevant_chunks(self, doc_store, tmp_path):
        """Search returns chunks semantically similar to the query."""
        f = tmp_path / "doc.md"
        f.write_text("Python is a programming language. Snakes are reptiles.")
        doc_store.ingest(f)

        hits = doc_store.search("What is Python?", limit=3)
        assert len(hits) >= 1
        # With our deterministic mock embeddings, we just verify the API works.
        assert all(hasattr(h, "score") for h in hits)
        assert all(hasattr(h, "content") for h in hits)

    def test_search_respects_limit(self, doc_store, tmp_path):
        """Search returns at most `limit` results."""
        # Create multiple files to get multiple chunks.
        for i in range(10):
            (tmp_path / f"doc{i}.md").write_text(f"Document number {i}.")
        doc_store.ingest(tmp_path)

        hits = doc_store.search("query", limit=3)
        assert len(hits) <= 3

    def test_search_empty_store_returns_empty(self, doc_store):
        """Searching an empty store returns no results."""
        hits = doc_store.search("anything", limit=5)
        assert hits == []


# ── Stats ─────────────────────────────────────────────────────────

class TestStats:
    """R9 AC10: `autodidact learn --stats` shows totals."""

    def test_stats_reports_totals(self, doc_store, tmp_path):
        """Stats include total files, total chunks, source breakdown."""
        (tmp_path / "a.md").write_text("First.")
        (tmp_path / "b.md").write_text("Second.")
        doc_store.ingest(tmp_path)

        stats = doc_store.get_stats()
        assert stats["total_chunks"] >= 2
        assert stats["total_files"] == 2
        assert "sources" in stats


# ── Schema / migration ────────────────────────────────────────────

class TestSchema:
    """R9 AC7: document_chunks table with correct columns."""

    def test_document_chunks_table_exists(self, db_conn):
        """init_database creates the document_chunks table."""
        row = db_conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='document_chunks'"
        ).fetchone()
        assert row is not None

    def test_document_chunks_schema(self, db_conn):
        """Table has the required columns per R9 AC7."""
        cols = db_conn.execute("PRAGMA table_info(document_chunks)").fetchall()
        col_names = {c["name"] for c in cols}
        required = {"id", "content", "source_file", "chunk_index", "embedding", "created_at"}
        assert required.issubset(col_names), (
            f"Missing columns: {required - col_names}"
        )


# ── Separation from agent memory (AD-002) ─────────────────────────

class TestStoreSeparation:
    """AD-002: document_chunks and knowledge_entries are logically separate."""

    def test_ingesting_docs_does_not_add_to_knowledge_entries(
        self, doc_store, db_conn, tmp_path
    ):
        """Documents go to document_chunks, not knowledge_entries."""
        f = tmp_path / "doc.md"
        f.write_text("This is a document, not an agent memory entry.")
        doc_store.ingest(f)

        ke_count = db_conn.execute(
            "SELECT COUNT(*) AS n FROM knowledge_entries"
        ).fetchone()["n"]
        dc_count = db_conn.execute(
            "SELECT COUNT(*) AS n FROM document_chunks"
        ).fetchone()["n"]

        assert ke_count == 0
        assert dc_count >= 1


class TestDoclingReader:
    """Docling is the preferred PDF/DOCX reader, with graceful fallback."""

    def _fake_converter(self, markdown: str):
        """Build a stand-in for docling's DocumentConverter."""
        converter = MagicMock()
        converter.convert.return_value.document.export_to_markdown.return_value = markdown
        return converter

    def test_read_with_docling_returns_markdown(self, monkeypatch, tmp_path):
        """When docling is available, its Markdown output is returned."""
        from autodidact import document_store

        monkeypatch.setattr(
            document_store,
            "_get_docling_converter",
            lambda: self._fake_converter("| a | b |\n| - | - |\n| 1 | 2 |"),
        )
        out = document_store._read_with_docling(tmp_path / "report.pdf")
        assert out == "| a | b |\n| - | - |\n| 1 | 2 |"

    def test_read_with_docling_unavailable_returns_none(self, monkeypatch, tmp_path):
        """When docling is not installed, None signals the caller to fall back."""
        from autodidact import document_store

        monkeypatch.setattr(document_store, "_get_docling_converter", lambda: None)
        assert document_store._read_with_docling(tmp_path / "report.pdf") is None

    def test_read_with_docling_conversion_failure_returns_none(
        self, monkeypatch, tmp_path
    ):
        """A conversion error is swallowed and falls back rather than raising."""
        from autodidact import document_store

        broken = MagicMock()
        broken.convert.side_effect = RuntimeError("bad pdf")
        monkeypatch.setattr(document_store, "_get_docling_converter", lambda: broken)
        assert document_store._read_with_docling(tmp_path / "report.pdf") is None

    def test_read_text_prefers_docling_for_pdf(self, monkeypatch, doc_store, tmp_path):
        """_read_text_from_file uses docling output for PDFs when available."""
        from autodidact import document_store

        monkeypatch.setattr(
            document_store,
            "_get_docling_converter",
            lambda: self._fake_converter("# Docling extracted"),
        )
        # File contents are irrelevant — docling is mocked and short-circuits
        # before the pymupdf reader runs.
        pdf = tmp_path / "report.pdf"
        pdf.write_bytes(b"%PDF-1.4 not really parsed")
        assert doc_store._read_text_from_file(pdf) == "# Docling extracted"

    def test_read_text_falls_back_when_docling_absent(
        self, monkeypatch, doc_store, tmp_path
    ):
        """Without docling, PDFs go through the pymupdf reader as before."""
        from autodidact import document_store

        monkeypatch.setattr(document_store, "_get_docling_converter", lambda: None)
        pdf = tmp_path / "report.pdf"
        pdf.write_bytes(b"%PDF-1.4")

        fake_page = MagicMock()
        fake_page.get_text.return_value = "fallback text"
        fake_pymupdf = MagicMock()
        fake_pymupdf.open.return_value = [fake_page]
        monkeypatch.setitem(__import__("sys").modules, "pymupdf", fake_pymupdf)

        assert doc_store._read_text_from_file(pdf) == "fallback text"

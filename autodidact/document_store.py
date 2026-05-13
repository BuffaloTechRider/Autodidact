"""DocumentStore — document ingestion for cold start (R9).

Users point the agent at docs or codebases via `autodidact learn <path>`.
The store chunks the files, embeds each chunk, and persists them in the
`document_chunks` table.

Document chunks are logically separate from agent memory (AD-002).
Documents answer "what do the source materials say?" Agent memory answers
"have I been asked this before?" Both get retrieved at query time but with
different prompt framing.

Chunking uses a recursive character splitter on paragraph/line boundaries
with a ~500-token target size and ~50-token overlap. No LLM calls during
chunking — only during embedding.
"""

from __future__ import annotations

import logging
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator, Optional

import numpy as np

from autodidact.llm_client import LLMClient

logger = logging.getLogger(__name__)


# ── Supported text extensions (R9 AC3) ───────────────────────────
# Plain-text readable formats. Binary formats (.pdf, .docx, .jpg) are
# skipped unless a future optional extra adds support.
_TEXT_EXTENSIONS: frozenset[str] = frozenset({
    ".md", ".txt", ".rst",
    ".py", ".ts", ".js", ".jsx", ".tsx",
    ".json", ".yaml", ".yml", ".toml", ".cfg", ".ini",
    ".csv", ".tsv",
    ".html", ".htm", ".xml",
    ".sh", ".bash", ".zsh",
    ".go", ".rs", ".java", ".kt",
    ".c", ".cpp", ".h", ".hpp",
    ".rb", ".php", ".swift",
})

# Chars-per-token approximation (OpenAI's cl100k_base rule of thumb).
# We chunk by characters for simplicity; target tokens × 4 ≈ target chars.
# Note: this UNDERESTIMATES tokens for code (more symbols per char). The
# safe cap below leaves headroom for that.
_CHARS_PER_TOKEN = 4

# Default chunk target (in tokens) for ingestion. Was 500; lowered to 384
# after live testing showed 500 was too close to BGE-large's 512-token
# context window, especially on Python source where tokens-per-char is
# higher than the 4:1 heuristic.
_DEFAULT_CHUNK_SIZE_TOKENS = 384

# Hard cap on chunk size (tokens). Any chunk larger than this gets truncated
# at character boundaries before being returned. Must stay under the
# embedding model's context window with margin.
#   bge-large-en-v1.5: 512-token context.
#   With margin for under-counting + special tokens: 480.
_SAFE_CHUNK_TOKEN_CAP = 480

# Max file size to ingest, in bytes. Protects against accidentally ingesting
# a 500MB log file. Users can override via ingest(max_file_bytes=...).
_DEFAULT_MAX_FILE_BYTES = 2_000_000  # 2 MB


# ── Data types ───────────────────────────────────────────────────

@dataclass
class DocumentChunk:
    """A single stored chunk of a document."""

    id: str
    content: str
    source_file: str
    chunk_index: int
    embedding: Optional[list[float]]
    tags: list[str] = field(default_factory=list)
    created_at: str = ""


@dataclass
class ScoredChunk:
    """A DocumentChunk plus a similarity score from a search."""

    chunk: DocumentChunk
    score: float

    # Convenience passthroughs so tests/callers can access common fields directly.
    @property
    def content(self) -> str:
        return self.chunk.content

    @property
    def source_file(self) -> str:
        return self.chunk.source_file


@dataclass
class IngestResult:
    """Return value of DocumentStore.ingest()."""

    files_ingested: int
    chunks_created: int
    files_skipped: int = 0


# ── Chunking ─────────────────────────────────────────────────────

def chunk_text(
    text: str,
    *,
    chunk_size: int = _DEFAULT_CHUNK_SIZE_TOKENS,
    overlap: int = 50,
) -> list[str]:
    """Split `text` into chunks of ~chunk_size tokens with ~overlap tokens of overlap.

    Chunk size is specified in tokens. We convert to characters using a
    4-chars-per-token approximation. The splitter tries to break on paragraph
    boundaries first, then line boundaries, then word boundaries.

    Hard cap: every returned chunk is at most ``_SAFE_CHUNK_TOKEN_CAP`` tokens.
    Chunks exceeding the cap (e.g. very dense code with no whitespace to split
    on) are truncated at character boundaries. This protects the embedding
    backend from "input length exceeds context length" errors on models with
    a fixed context (BGE-large = 512 tokens).

    Empty input returns an empty list.
    """
    text = text.strip()
    if not text:
        return []

    char_target = chunk_size * _CHARS_PER_TOKEN
    char_overlap = overlap * _CHARS_PER_TOKEN
    cap_chars = _SAFE_CHUNK_TOKEN_CAP * _CHARS_PER_TOKEN

    # Short text fits in one chunk — but still respect the hard cap.
    if len(text) <= char_target:
        return _enforce_cap([text], cap_chars)

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + char_target, len(text))
        # Try to break on a nice boundary near the target end.
        if end < len(text):
            end = _find_split_point(text, start, end)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        # Step forward by (chunk_size - overlap).
        start = max(start + 1, end - char_overlap)

    return _enforce_cap(chunks, cap_chars)


def _enforce_cap(chunks: list[str], cap_chars: int) -> list[str]:
    """Ensure no chunk exceeds the cap. Splits oversized chunks into pieces."""
    out: list[str] = []
    for chunk in chunks:
        if len(chunk) <= cap_chars:
            out.append(chunk)
            continue
        # Split at character boundaries. We don't try to find clean breaks
        # here — by definition this chunk had no clean break point in the
        # main splitter. Hard truncation is acceptable; the alternative is
        # the embedding call failing.
        for i in range(0, len(chunk), cap_chars):
            piece = chunk[i:i + cap_chars].strip()
            if piece:
                out.append(piece)
    return out


def _find_split_point(text: str, start: int, target_end: int) -> int:
    """Find a good split point near target_end, preferring paragraph/line breaks."""
    # Look back up to 25% of the chunk size for a nicer boundary.
    search_start = max(start, target_end - (target_end - start) // 4)

    # Preferred boundaries, in order of decreasing preference.
    for boundary in ("\n\n", "\n", ". ", " "):
        idx = text.rfind(boundary, search_start, target_end)
        if idx != -1:
            return idx + len(boundary)

    # No nice boundary — split at target_end.
    return target_end


# ── File walking ─────────────────────────────────────────────────

def walk_files(path: Path, *, max_file_bytes: int = _DEFAULT_MAX_FILE_BYTES) -> Iterator[Path]:
    """Yield supported text files under `path`.

    - If `path` is a file, yields just that file (if supported).
    - If `path` is a directory, walks recursively.
    - Respects `.gitignore` in the root directory (simple patterns only).
    - Skips binary/unsupported extensions.
    - Skips files larger than max_file_bytes.
    """
    path = Path(path)
    if path.is_file():
        if _is_supported(path):
            yield path
        return

    ignore_patterns = _load_gitignore(path) if path.is_dir() else []

    for p in sorted(path.rglob("*")):
        if not p.is_file():
            continue
        if not _is_supported(p):
            continue
        try:
            if p.stat().st_size > max_file_bytes:
                logger.debug("Skipping %s: size > %d bytes", p, max_file_bytes)
                continue
        except OSError:
            continue
        if _is_ignored(p, path, ignore_patterns):
            continue
        yield p


def _is_supported(file_path: Path) -> bool:
    """Whether the file's extension is a supported text format."""
    return file_path.suffix.lower() in _TEXT_EXTENSIONS


def _load_gitignore(root: Path) -> list[str]:
    """Load simple patterns from .gitignore at the root."""
    gitignore = root / ".gitignore"
    if not gitignore.exists():
        return []
    try:
        patterns: list[str] = []
        for raw in gitignore.read_text(encoding="utf-8", errors="replace").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            patterns.append(line)
        return patterns
    except OSError:
        return []


def _is_ignored(file_path: Path, root: Path, patterns: Iterable[str]) -> bool:
    """Simple gitignore match: supports literal filenames and dir/ patterns."""
    try:
        rel = file_path.relative_to(root)
    except ValueError:
        return False

    rel_str = str(rel).replace("\\", "/")
    name = file_path.name

    for pat in patterns:
        pat = pat.rstrip("/")
        # Directory match: pattern ends with / in original, any ancestor dir matches.
        if pat in rel_str.split("/"):
            return True
        # Literal filename match.
        if pat == name:
            return True
        # Prefix match (e.g. "build/" in rel_str).
        if rel_str.startswith(pat + "/") or rel_str == pat:
            return True
    return False


# ── The store ────────────────────────────────────────────────────

class DocumentStore:
    """SQLite + in-memory embedding store for document chunks."""

    def __init__(
        self,
        conn: sqlite3.Connection,
        embed_client: LLMClient,
        *,
        embedding_dim: int = 1024,
    ) -> None:
        self.conn = conn
        self._embed_client = embed_client
        self._embedding_dim = embedding_dim

    # ── Public API ────────────────────────────────────────────────

    def ingest(
        self,
        path: Path | str,
        *,
        chunk_size: int = _DEFAULT_CHUNK_SIZE_TOKENS,
        overlap: int = 50,
        on_progress: Optional[callable] = None,
    ) -> IngestResult:
        """Ingest a file or directory into the store.

        Chunks each file, embeds each chunk, and persists rows in
        `document_chunks`. Re-ingesting a file replaces its existing chunks
        (R9 AC9 — deduplication on re-ingestion).
        """
        path = Path(path)
        files_ingested = 0
        chunks_created = 0
        files_skipped = 0

        for file_path in walk_files(path):
            try:
                text = file_path.read_text(encoding="utf-8", errors="replace")
            except OSError as e:
                logger.warning("Skipping %s: %s", file_path, e)
                files_skipped += 1
                continue

            chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
            if not chunks:
                files_skipped += 1
                continue

            # Deduplication: remove any existing chunks for this source file.
            self._delete_chunks_for_file(str(file_path))

            for i, chunk_content in enumerate(chunks):
                try:
                    embedding = self._embed_client.embed(chunk_content)
                    self._insert_chunk(
                        content=chunk_content,
                        source_file=str(file_path),
                        chunk_index=i,
                        embedding=np.asarray(embedding, dtype=np.float32),
                    )
                    chunks_created += 1
                except Exception as e:
                    logger.warning("Failed to embed chunk %d of %s: %s", i, file_path, e)

            files_ingested += 1
            if on_progress is not None:
                on_progress({
                    "type": "file_ingested",
                    "file": str(file_path),
                    "chunks": len(chunks),
                    "total_files": files_ingested,
                })

        return IngestResult(
            files_ingested=files_ingested,
            chunks_created=chunks_created,
            files_skipped=files_skipped,
        )

    def search(self, query: str, *, limit: int = 5) -> list[ScoredChunk]:
        """Search for chunks semantically similar to `query`.

        Returns up to `limit` results sorted by descending similarity.
        Empty store returns an empty list.
        """
        if self.count() == 0:
            return []

        try:
            query_emb = np.asarray(self._embed_client.embed(query), dtype=np.float32)
        except Exception as e:
            logger.warning("Document search failed to embed query: %s", e)
            return []

        # Pull all chunks (small v1.0 scope — brute-force scan). FAISS index
        # is a v1.1 optimization once KB is large enough to warrant it.
        rows = self.conn.execute(
            "SELECT id, content, source_file, chunk_index, embedding, tags, created_at "
            "FROM document_chunks WHERE embedding IS NOT NULL"
        ).fetchall()

        q = _normalize(query_emb)
        scored: list[ScoredChunk] = []
        for row in rows:
            emb = np.frombuffer(row["embedding"], dtype=np.float32)
            if emb.shape[0] != q.shape[0]:
                continue
            score = float(np.dot(q, _normalize(emb)))
            chunk = _row_to_chunk(row)
            scored.append(ScoredChunk(chunk=chunk, score=score))

        scored.sort(key=lambda s: s.score, reverse=True)
        return scored[:limit]

    def count(self) -> int:
        """Total number of chunks in the store."""
        row = self.conn.execute("SELECT COUNT(*) AS n FROM document_chunks").fetchone()
        return int(row["n"])

    def list_chunks(self, *, limit: int = 100) -> list[DocumentChunk]:
        """List chunks in the store (for debugging/inspection)."""
        rows = self.conn.execute(
            "SELECT id, content, source_file, chunk_index, embedding, tags, created_at "
            "FROM document_chunks ORDER BY source_file, chunk_index LIMIT ?",
            (limit,),
        ).fetchall()
        return [_row_to_chunk(row) for row in rows]

    def get_stats(self) -> dict:
        """Return ingestion statistics (R9 AC10)."""
        total_chunks = self.count()
        files_row = self.conn.execute(
            "SELECT COUNT(DISTINCT source_file) AS n FROM document_chunks"
        ).fetchone()
        total_files = int(files_row["n"])

        source_rows = self.conn.execute(
            "SELECT source_file, COUNT(*) AS n FROM document_chunks "
            "GROUP BY source_file ORDER BY n DESC LIMIT 10"
        ).fetchall()
        sources = {row["source_file"]: int(row["n"]) for row in source_rows}

        return {
            "total_chunks": total_chunks,
            "total_files": total_files,
            "sources": sources,
        }

    def clear_file(self, source_file: str) -> int:
        """Remove all chunks for a given source file. Returns count deleted."""
        return self._delete_chunks_for_file(source_file)

    def clear_all(self) -> None:
        """Remove all document chunks."""
        self.conn.execute("DELETE FROM document_chunks")
        self.conn.commit()

    # ── Internals ─────────────────────────────────────────────────

    def _insert_chunk(
        self,
        *,
        content: str,
        source_file: str,
        chunk_index: int,
        embedding: np.ndarray,
    ) -> str:
        """Insert a single chunk row."""
        chunk_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        self.conn.execute(
            "INSERT INTO document_chunks "
            "(id, content, source_file, chunk_index, embedding, tags, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                chunk_id,
                content,
                source_file,
                chunk_index,
                embedding.tobytes(),
                "[]",
                now,
            ),
        )
        self.conn.commit()
        return chunk_id

    def _delete_chunks_for_file(self, source_file: str) -> int:
        """Delete chunks for a given source file. Returns count deleted."""
        cur = self.conn.execute(
            "DELETE FROM document_chunks WHERE source_file = ?", (source_file,)
        )
        self.conn.commit()
        return cur.rowcount


# ── Helpers ──────────────────────────────────────────────────────

def _normalize(v: np.ndarray) -> np.ndarray:
    """L2-normalize a vector; zero vectors return as-is."""
    norm = float(np.linalg.norm(v))
    if norm == 0.0:
        return v
    return v / norm


def _row_to_chunk(row: sqlite3.Row) -> DocumentChunk:
    """Build a DocumentChunk from a SQLite row."""
    import json

    embedding = None
    if row["embedding"] is not None:
        embedding = np.frombuffer(row["embedding"], dtype=np.float32).tolist()

    tags: list[str] = []
    try:
        tags = json.loads(row["tags"]) if row["tags"] else []
    except (TypeError, ValueError):
        tags = []

    return DocumentChunk(
        id=row["id"],
        content=row["content"],
        source_file=row["source_file"],
        chunk_index=int(row["chunk_index"]),
        embedding=embedding,
        tags=tags,
        created_at=row["created_at"] or "",
    )

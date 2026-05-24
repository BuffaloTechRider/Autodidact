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
from typing import Any, Iterable, Iterator, Optional

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

_SUPPORTED_EXTENSIONS = _TEXT_EXTENSIONS | {".pdf", ".docx"}

# Chars-per-token approximation (OpenAI's cl100k_base rule of thumb).
# We chunk by characters for the fast path; only the cap-enforcement step
# uses the real BGE tokenizer when available.
_CHARS_PER_TOKEN = 4

# When the BGE tokenizer can't be loaded (offline / dependency missing),
# we fall back to char-based capping with a much stricter ratio. 2.5
# chars-per-token is conservative for code-dense inputs without being so
# strict that prose gets over-split.
_FALLBACK_CHARS_PER_TOKEN = 2.5

# Default chunk target (in tokens) for ingestion. Was 500; lowered to 384
# after live testing showed 500 was too close to BGE-large's 512-token
# context window, especially on Python source where tokens-per-char is
# higher than the 4:1 heuristic.
_DEFAULT_CHUNK_SIZE_TOKENS = 384

# Hard cap on chunk size (tokens). Any chunk larger than this gets split
# before being returned. Must stay under the embedding model's context
# window with margin.
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
    facts_synthesized: int = 0


# ── Tokenizer (BGE-large) for accurate cap enforcement ──────────

# Loaded lazily and cached at module level. Keeping it None until first
# use lets `tokenizers` be a soft dependency — if it can't be imported
# (offline install with no cache, or the lib is genuinely missing), we
# fall back to a stricter chars-per-token heuristic.
_BGE_TOKENIZER: Any = None
_BGE_TOKENIZER_LOADED = False


def _get_bge_tokenizer():
    """Return the cached BGE-large tokenizer, or None if unavailable.

    The first call attempts to load it; subsequent calls reuse the result
    (success or None). This is hot-path code during ingestion — we only
    pay the load cost once per process.
    """
    global _BGE_TOKENIZER, _BGE_TOKENIZER_LOADED
    if _BGE_TOKENIZER_LOADED:
        return _BGE_TOKENIZER
    _BGE_TOKENIZER_LOADED = True
    try:
        from tokenizers import Tokenizer  # type: ignore
        _BGE_TOKENIZER = Tokenizer.from_pretrained("BAAI/bge-large-en-v1.5")
    except Exception:
        # Offline, no cache, or tokenizers not installed. Caller falls
        # back to char-based capping.
        _BGE_TOKENIZER = None
    return _BGE_TOKENIZER


def _count_bge_tokens(text: str) -> Optional[int]:
    """Real BGE token count, or None if the tokenizer is unavailable."""
    tok = _get_bge_tokenizer()
    if tok is None:
        return None
    return len(tok.encode(text).ids)


# ── AST-aware chunking (tree-sitter) ─────────────────────────────

# Language extensions → tree-sitter grammar loader. Loaded lazily.
_TS_LANGUAGE_MAP: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
}

# Node types that represent top-level semantic units per language.
_TS_CHUNK_NODE_TYPES: dict[str, set[str]] = {
    "python": {"function_definition", "class_definition", "decorated_definition"},
    "javascript": {"function_declaration", "class_declaration", "export_statement",
                   "lexical_declaration", "expression_statement"},
    "typescript": {"function_declaration", "class_declaration", "export_statement",
                   "lexical_declaration", "interface_declaration", "type_alias_declaration"},
    "tsx": {"function_declaration", "class_declaration", "export_statement",
            "lexical_declaration", "interface_declaration", "type_alias_declaration"},
}


def _get_ts_parser(lang: str):
    """Return a tree-sitter Parser for the given language, or None."""
    try:
        import tree_sitter as ts
        if lang == "python":
            import tree_sitter_python as tsl
        elif lang == "javascript":
            import tree_sitter_javascript as tsl
        elif lang in ("typescript", "tsx"):
            import tree_sitter_typescript as tsl_mod
            tsl = tsl_mod
        else:
            return None
        if lang == "tsx":
            language = ts.Language(tsl.language_tsx())
        elif lang == "typescript":
            language = ts.Language(tsl.language_typescript())
        else:
            language = ts.Language(tsl.language())
        return ts.Parser(language)
    except (ImportError, Exception) as e:
        logger.debug("tree-sitter unavailable for %s: %s", lang, e)
        return None


def chunk_code_ast(
    text: str,
    ext: str,
    *,
    max_chunk_tokens: int = _SAFE_CHUNK_TOKEN_CAP,
) -> Optional[list[str]]:
    """Split code into semantic chunks using tree-sitter AST.

    Returns a list of chunks where each chunk is a complete top-level
    definition (function, class, etc.). Adjacent small nodes (imports,
    assignments, comments) are grouped together up to max_chunk_tokens.

    Returns None if tree-sitter is unavailable or the extension is unsupported,
    signaling the caller to fall back to chunk_text().
    """
    lang = _TS_LANGUAGE_MAP.get(ext)
    if lang is None:
        return None
    parser = _get_ts_parser(lang)
    if parser is None:
        return None

    text_bytes = text.encode("utf-8")
    tree = parser.parse(text_bytes)
    root = tree.root_node
    chunk_types = _TS_CHUNK_NODE_TYPES.get(lang, set())

    def _node_text(node) -> str:
        return text_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="replace")

    # Extract top-level nodes as text segments.
    segments: list[str] = []
    for child in root.children:
        segment = _node_text(child).strip()
        if segment:
            segments.append(segment)

    if not segments:
        return None

    # Group segments: large definitions (functions/classes) become their own
    # chunk. Small segments (imports, assignments) are grouped together.
    chunks: list[str] = []
    buffer: list[str] = []
    buffer_tokens = 0
    char_per_tok = _FALLBACK_CHARS_PER_TOKEN  # code is denser than prose

    for child in root.children:
        segment = _node_text(child).strip()
        if not segment:
            continue
        seg_tokens = len(segment) // char_per_tok

        is_definition = child.type in chunk_types

        if is_definition and seg_tokens > max_chunk_tokens // 3:
            # Flush buffer first
            if buffer:
                chunks.append("\n".join(buffer))
                buffer = []
                buffer_tokens = 0
            # Large definition gets its own chunk (may need sub-splitting)
            if seg_tokens <= max_chunk_tokens:
                chunks.append(segment)
            else:
                # Split large class into its methods
                sub_chunks = _split_large_node(text_bytes, child, chunk_types, max_chunk_tokens, char_per_tok)
                chunks.extend(sub_chunks)
        else:
            # Small node — accumulate in buffer
            if buffer_tokens + seg_tokens > max_chunk_tokens and buffer:
                chunks.append("\n".join(buffer))
                buffer = []
                buffer_tokens = 0
            buffer.append(segment)
            buffer_tokens += seg_tokens

    if buffer:
        chunks.append("\n".join(buffer))

    if not chunks:
        return None
    # Sub-split any oversized chunks, preserving the function/class signature
    # as a header on each piece so the model keeps enclosing context.
    cap_chars = int(max_chunk_tokens * _FALLBACK_CHARS_PER_TOKEN)
    final: list[str] = []
    for chunk in chunks:
        if len(chunk) <= cap_chars:
            final.append(chunk)
        else:
            final.extend(_subsplit_with_header(chunk, cap_chars))
    # Final pass: enforce the hard token cap with the real tokenizer.
    return _enforce_cap(final, int(_SAFE_CHUNK_TOKEN_CAP * _FALLBACK_CHARS_PER_TOKEN))


def _subsplit_with_header(chunk: str, cap_chars: int) -> list[str]:
    """Split an oversized AST chunk, keeping its signature as header context.

    Extracts a header containing the class and/or def signature lines,
    then prepends it to each sub-chunk so the model knows the context.
    """
    lines = chunk.split("\n")
    # Collect header: all class/def signature lines (handles "class X:\n    def foo(...):")
    header_lines: list[str] = []
    found_def = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("def ") or stripped.startswith("async def "):
            header_lines.append(line)
            found_def = True
            # Include continuation lines of multi-line signature (until `:`)
            if stripped.endswith(":"):
                break
            continue
        if stripped.startswith("class "):
            header_lines.append(line)
            continue
        if found_def:
            # Inside multi-line def signature — keep going until `)`/`:`
            header_lines.append(line)
            if "):" in stripped or stripped.endswith(":"):
                break
            continue
        if not header_lines or (header_lines and not found_def):
            # Before finding def/class or between class and def
            if stripped.startswith("@") or stripped.startswith("class "):
                header_lines.append(line)
            elif header_lines and not stripped:
                continue
            elif len(header_lines) < 5:
                continue
            else:
                break
    header = "\n".join(header_lines) if header_lines else lines[0] if lines else ""
    # Reserve room for the header that gets prepended to sub-chunks 1+.
    header_token_budget = int(len(header) / _FALLBACK_CHARS_PER_TOKEN) + 10
    target = _SAFE_CHUNK_TOKEN_CAP - header_token_budget
    sub_chunks = chunk_text(chunk, chunk_size=max(200, target), overlap=0)
    if not sub_chunks:
        return [chunk[:cap_chars]]
    result: list[str] = []
    for i, sc in enumerate(sub_chunks):
        if i == 0:
            result.append(sc)
        else:
            result.append(header + "\n" + sc)
    return result


def _split_large_node(
    text_bytes: bytes, node, chunk_types: set[str], max_tokens: int, char_per_tok: int
) -> list[str]:
    """Split a large node (e.g. a big class) into method-level chunks.

    Each method becomes its own chunk, prefixed with the class signature line
    so the model knows the enclosing context.
    """
    def _nb(n) -> str:
        return text_bytes[n.start_byte:n.end_byte].decode("utf-8", errors="replace")

    # For Python classes: body is in a "block" child node.
    # For JS/TS: body is in "class_body" or "statement_block".
    block = None
    for child in node.children:
        if child.type in ("block", "class_body", "statement_block"):
            block = child
            break

    if block is None:
        return [_nb(node).strip()]

    # Header = class signature (everything from node start to block start).
    header = text_bytes[node.start_byte:block.start_byte].decode("utf-8", errors="replace").strip()

    # Split block children into method-level segments.
    method_types = chunk_types | {"function_definition", "decorated_definition"}
    chunks: list[str] = []
    buffer: list[str] = []
    buffer_tokens = len(header) // char_per_tok

    for child in block.children:
        segment = _nb(child).strip()
        if not segment:
            continue
        seg_tokens = len(segment) // char_per_tok

        is_method = child.type in method_types

        if is_method and seg_tokens > max_tokens // 4:
            # Flush buffer, then emit this method as its own chunk
            if buffer:
                chunks.append(header + "\n    " + "\n    ".join(buffer))
                buffer = []
                buffer_tokens = len(header) // char_per_tok
            chunks.append(header + "\n    " + segment)
        else:
            if buffer_tokens + seg_tokens > max_tokens and buffer:
                chunks.append(header + "\n    " + "\n    ".join(buffer))
                buffer = []
                buffer_tokens = len(header) // char_per_tok
            buffer.append(segment)
            buffer_tokens += seg_tokens

    if buffer:
        chunks.append(header + "\n    " + "\n    ".join(buffer))

    return chunks if chunks else [_nb(node).strip()]


# ── Text chunking (fallback for non-code files) ──────────────────

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
    """Ensure no chunk exceeds the BGE 480-token cap.

    Strategy:
      1. If the BGE tokenizer is available, count real tokens. Any chunk
         over the cap is split in half (recursively) until each piece fits.
         This is correct by construction.
      2. If the tokenizer can't load, fall back to character-based capping
         with a stricter ratio (2.5 chars/token instead of 4). This is
         conservative — it produces smaller chunks than necessary on prose,
         but doesn't blow the 512 ceiling on dense code.
    """
    tok = _get_bge_tokenizer()
    if tok is not None:
        return _split_by_real_tokens(chunks, _SAFE_CHUNK_TOKEN_CAP)

    # Fallback: stricter char-based cap.
    fallback_cap_chars = int(_SAFE_CHUNK_TOKEN_CAP * _FALLBACK_CHARS_PER_TOKEN)
    out: list[str] = []
    for chunk in chunks:
        if len(chunk) <= fallback_cap_chars:
            out.append(chunk)
            continue
        for i in range(0, len(chunk), fallback_cap_chars):
            piece = chunk[i:i + fallback_cap_chars].strip()
            if piece:
                out.append(piece)
    return out


def _split_by_real_tokens(chunks: list[str], token_cap: int) -> list[str]:
    """Recursively split chunks until each piece is under ``token_cap`` BGE tokens.

    Splits at character midpoints when the chunk has no obvious structural
    boundary inside it. The previous implementation used hard char-boundary
    truncation when the chunker couldn't find a break — we keep that
    fallback at the leaf, but only when bisection alone can't bring a
    piece under the cap (which only happens for pathological inputs like
    single-line 5KB strings).
    """
    out: list[str] = []
    for chunk in chunks:
        if _count_bge_tokens(chunk) <= token_cap:
            out.append(chunk)
            continue
        out.extend(_bisect_until_under_cap(chunk, token_cap))
    return out


def _bisect_until_under_cap(chunk: str, token_cap: int, *, max_depth: int = 12) -> list[str]:
    """Halve `chunk` until each piece fits under the token cap."""
    if max_depth <= 0:
        return [chunk[:len(chunk) // 2], chunk[len(chunk) // 2:]]

    if _count_bge_tokens(chunk) <= token_cap:
        return [chunk]

    mid = _find_split_point(chunk, 0, len(chunk) // 2 + len(chunk) // 4) or len(chunk) // 2
    if mid <= 0 or mid >= len(chunk):
        mid = len(chunk) // 2

    left = chunk[:mid].strip()
    right = chunk[mid:].strip()

    # Defensive: if the split degenerated to all-on-one-side, force a
    # midpoint so we always make progress.
    if not left or not right:
        mid = len(chunk) // 2
        left = chunk[:mid].strip()
        right = chunk[mid:].strip()

    out: list[str] = []
    if left:
        out.extend(_bisect_until_under_cap(left, token_cap, max_depth=max_depth - 1))
    if right:
        out.extend(_bisect_until_under_cap(right, token_cap, max_depth=max_depth - 1))
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
    return file_path.suffix.lower() in _SUPPORTED_EXTENSIONS


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
        knowledge_store=None,
        extractor_client: Optional[LLMClient] = None,
    ) -> None:
        self.conn = conn
        self._embed_client = embed_client
        self._embedding_dim = embedding_dim
        self._knowledge_store = knowledge_store
        self._extractor_client = extractor_client

    # ── Public API ────────────────────────────────────────────────

    def _read_text_from_file(self, file_path: Path) -> str:
        """
            Extract text content from a file. Supports plain text, PDF, and DOCX.
            Raises OSError if the file cannot be read, or ImportError if the
            required parser library is not installed (pymupdf for PDF, python-docx for DOCX).
        """
        try:
            ext = file_path.suffix.lower()
            if ext not in _SUPPORTED_EXTENSIONS:
                raise OSError(f"Unsupported file extension: {ext}")
            if ext == ".pdf":
                try:
                    import pymupdf
                    doc = pymupdf.open(file_path)
                    text = ""
                    text = "\n".join([page.get_text() for page in doc])
                    return text
                except ImportError:
                    raise ImportError("pymupdf not available for PDF reading\nInstall with pip install pymupdf")
            elif ext == ".docx":
                try:
                    from docx import Document
                    doc = Document(file_path)
                    text = "\n".join(p.text for p in doc.paragraphs)
                    return text
                except ImportError:
                    raise ImportError("python-docx not available for Word document reading\nInstall with pip install python-docx")
            else:
                return file_path.read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            logger.warning("Skipping %s: %s", file_path, e)
            raise OSError(f"Failed to read {file_path}: {e}")

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
                text = self._read_text_from_file(file_path)
            except OSError as e:
                logger.warning("Skipping %s: %s", file_path, e)
                files_skipped += 1
                continue
            except ImportError as e:
                logger.warning("Skipping %s: %s", file_path, e)
                logger.warning("To ingest this file type, install the required library and its dependencies: %s", e)
                files_skipped += 1
                continue

            # Prefer AST-aware chunking for code files; fall back to text splitter.
            ext = file_path.suffix.lower()
            chunks = chunk_code_ast(text, ext)
            if chunks is None:
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

        # Synthesis pass: extract key facts into the knowledge store.
        # Runs in a background thread so ingest returns immediately.
        # Progress callback is NOT passed to the thread — Rich console
        # is not thread-safe and the main thread may be in interactive mode.
        if self._knowledge_store is not None and self._extractor_client is not None:
            import threading
            thread = threading.Thread(
                target=self._run_synthesis,
                args=(path, None),
                daemon=True,
            )
            thread.start()

        return IngestResult(
            files_ingested=files_ingested,
            chunks_created=chunks_created,
            files_skipped=files_skipped,
        )

    def search(self, query: str, *, limit: int = 5, query_embedding: Optional[np.ndarray] = None) -> list[ScoredChunk]:
        """Search for chunks semantically similar to `query`.

        Returns up to `limit` results sorted by descending similarity.
        Empty store returns an empty list. Pass `query_embedding` to skip
        the embedding call (reuse from a prior embed).
        """
        if self.count() == 0:
            return []

        if query_embedding is not None:
            query_emb = np.asarray(query_embedding, dtype=np.float32)
        else:
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

    def search_bm25(self, query: str, *, limit: int = 10) -> list[ScoredChunk]:
        """BM25 keyword search via FTS5.

        Returns chunks ranked by BM25 relevance. Falls back to empty list
        if FTS5 table is not populated or query has no matches.
        """
        try:
            rows = self.conn.execute(
                "SELECT dc.id, dc.content, dc.source_file, dc.chunk_index, "
                "dc.embedding, dc.tags, dc.created_at, f.rank "
                "FROM document_chunks_fts f "
                "JOIN document_chunks dc ON dc.rowid = f.rowid "
                "WHERE document_chunks_fts MATCH ? "
                "ORDER BY f.rank "
                "LIMIT ?",
                (query, limit),
            ).fetchall()
        except Exception as e:
            logger.debug("BM25 search failed (FTS5 may not be populated): %s", e)
            return []

        scored: list[ScoredChunk] = []
        for row in rows:
            chunk = _row_to_chunk(row)
            bm25_rank = float(row["rank"])
            score = 1.0 / (1.0 + abs(bm25_rank))
            scored.append(ScoredChunk(chunk=chunk, score=score))
        return scored

    def search_hybrid(self, query: str, *, limit: int = 5) -> list[ScoredChunk]:
        """Hybrid BM25 + vector search with RRF ordering and cosine scoring.

        Uses Reciprocal Rank Fusion (k=60) to ORDER results — chunks found
        by both methods rank higher. The returned SCORE is the vector cosine
        similarity (0-1 absolute relevance), so downstream thresholds like
        0.75 and 0.30 remain meaningful. BM25-only hits get a small penalty
        since we have no cosine score for them.
        """
        vector_results = self.search(query, limit=limit * 3)
        bm25_results = self.search_bm25(query, limit=limit * 3)

        if not vector_results and not bm25_results:
            return []
        if not bm25_results:
            return vector_results[:limit]
        if not vector_results:
            return bm25_results[:limit]

        RRF_K = 60

        vector_ranks: dict[str, int] = {
            r.chunk.id: rank for rank, r in enumerate(vector_results)
        }
        bm25_ranks: dict[str, int] = {
            r.chunk.id: rank for rank, r in enumerate(bm25_results)
        }
        vector_scores: dict[str, float] = {
            r.chunk.id: r.score for r in vector_results
        }

        all_ids = set(vector_ranks.keys()) | set(bm25_ranks.keys())
        chunk_map: dict[str, DocumentChunk] = {}
        for r in vector_results:
            chunk_map[r.chunk.id] = r.chunk
        for r in bm25_results:
            chunk_map[r.chunk.id] = r.chunk

        # RRF for ordering
        rrf_ordered: list[tuple[str, float]] = []
        for chunk_id in all_ids:
            v_rank = vector_ranks.get(chunk_id, len(vector_results) + 1)
            b_rank = bm25_ranks.get(chunk_id, len(bm25_results) + 1)
            rrf_score = 1.0 / (RRF_K + v_rank) + 1.0 / (RRF_K + b_rank)
            rrf_ordered.append((chunk_id, rrf_score))
        rrf_ordered.sort(key=lambda x: x[1], reverse=True)

        # Score = cosine similarity from vector search (BM25-only hits get 0.0)
        fused: list[ScoredChunk] = []
        for chunk_id, _ in rrf_ordered[:limit]:
            score = vector_scores.get(chunk_id, 0.0)
            fused.append(ScoredChunk(chunk=chunk_map[chunk_id], score=score))

        return fused

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

    # ── Synthesis ─────────────────────────────────────────────────

    def _run_synthesis(self, path: Path | str, on_progress: Optional[callable]) -> None:
        """Background: extract key facts from ingested documents into knowledge store.

        Runs in a daemon thread. Groups chunks by file into ~2000-token sections,
        runs LLM extraction on each, and batch-inserts results into the knowledge
        store (10 facts per commit to minimize write lock contention with the
        main chat thread).
        """
        from autodidact.learning_extractor import LearningExtractor

        extractor = LearningExtractor(self._extractor_client)
        path = Path(path)
        max_sections_per_file = 20
        batch_size = 10
        batch: list = []

        for file_path in walk_files(path):
            try:
                text = file_path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue

            if not text.strip():
                continue

            sections = _group_into_sections(text, max_tokens=2000)
            file_facts = 0

            for section in sections[:max_sections_per_file]:
                try:
                    result = extractor.extract_from_document(section, str(file_path))
                    for entry in result.knowledge:
                        try:
                            emb = self._embed_client.embed(entry.question or entry.content)
                            entry.embedding = emb.tolist()
                            batch.append(entry)
                            file_facts += 1

                            if len(batch) >= batch_size:
                                self._knowledge_store.insert_batch(batch)
                                batch = []
                        except Exception as e:
                            logger.warning("Failed to embed synthesized fact: %s", e)
                except Exception as e:
                    logger.warning("Synthesis failed for section of %s: %s", file_path, e)

            if file_facts > 0:
                if on_progress is not None:
                    on_progress({
                        "type": "synthesized",
                        "file": str(file_path),
                        "facts": file_facts,
                    })
                else:
                    logger.info("Synthesized %d facts from %s", file_facts, file_path.name)

        # Flush remaining batch.
        if batch:
            self._knowledge_store.insert_batch(batch)

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


def _group_into_sections(text: str, max_tokens: int = 2000) -> list[str]:
    """Group text into sections of approximately max_tokens each.

    Tries to split on paragraph boundaries (double newline) to keep
    sections semantically coherent.
    """
    char_budget = max_tokens * _CHARS_PER_TOKEN
    paragraphs = text.split("\n\n")
    sections: list[str] = []
    current: list[str] = []
    current_len = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        para_len = len(para)
        if current_len + para_len > char_budget and current:
            sections.append("\n\n".join(current))
            current = [para]
            current_len = para_len
        else:
            current.append(para)
            current_len += para_len

    if current:
        sections.append("\n\n".join(current))

    return sections


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

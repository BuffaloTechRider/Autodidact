"""File operation tools — read, write, edit, search, list.

All five register into the ``file`` toolset. They operate on real paths
relative to the current working directory. Like the terminal tool, failures
(missing file, ambiguous edit) are returned as raised exceptions that the
registry converts into ``{"ok": false, "error": ...}`` tool output, so the
model sees the problem and can correct rather than crashing the loop.

Kept deliberately small: no atomic-write dance, no encoding guessing, no
recursive glob DSL. Reads/writes are UTF-8 with ``errors="replace"`` to match
``document_store``'s reader. Search is a line-oriented regex grep so results
are cheap to feed back to the model.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from autodidact.tools.fuzzy_match import FuzzyMatchError, fuzzy_replace
from autodidact.tools.registry import REGISTRY

logger = logging.getLogger(__name__)

_MAX_READ_BYTES = 64_000  # a single read shouldn't dominate the context
_MAX_SEARCH_MATCHES = 100  # cap grep output so a broad pattern stays bounded


def _resolved_within_cwd(path: Path) -> Path:
    """Resolve ``path`` and ensure it stays within the current directory.

    The tools operate on the project the agent was launched in; a model should
    not be able to read or write ``/etc/passwd`` via ``../../..`` or a symlink.
    ``resolve()`` normalizes traversal and follows symlinks, then we require
    the result to sit under the resolved cwd. Raises ``PermissionError`` on an
    escape so the registry surfaces it as tool output.
    """
    root = Path.cwd().resolve()
    resolved = path.resolve()
    try:
        resolved.relative_to(root)
    except ValueError:
        raise PermissionError(f"path escapes the working directory: {path}")
    return resolved


def read_file(args: dict) -> dict:
    """Return the text contents of a file (UTF-8, truncated if large)."""
    path = _resolved_within_cwd(Path(args["path"]))
    if not path.is_file():
        raise FileNotFoundError(f"not a file: {path}")
    data = path.read_bytes()
    truncated = len(data) > _MAX_READ_BYTES
    text = data[:_MAX_READ_BYTES].decode("utf-8", errors="replace")
    return {"path": str(path), "content": text, "truncated": truncated}


def write_file(args: dict) -> dict:
    """Create or overwrite a file with the given content.

    Parent directories are created as needed. Returns the byte count written.
    """
    path = _resolved_within_cwd(Path(args["path"]))
    content = args.get("content", "")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return {"path": str(path), "bytes_written": len(content.encode("utf-8"))}


def edit_file(args: dict) -> dict:
    """Replace a unique occurrence of ``old`` with ``new`` (fuzzy find/replace).

    Matching falls through a chain of increasing fuzziness (exact →
    line-trimmed) so an LLM-generated ``old`` that drifts on indentation or
    trailing whitespace still lands — the common failure mode when a cloud
    escalation re-derives a patch. ``old`` must still resolve to exactly one
    location; a zero- or multi-match edit is ambiguous and raises rather than
    guessing. The report includes which ``strategy`` matched.
    """
    path = _resolved_within_cwd(Path(args["path"]))
    old = args["old"]
    new = args["new"]
    if not path.is_file():
        raise FileNotFoundError(f"not a file: {path}")
    text = path.read_text(encoding="utf-8", errors="replace")
    try:
        new_text, strategy = fuzzy_replace(text, old, new)
    except FuzzyMatchError as e:
        raise ValueError(str(e))
    path.write_text(new_text, encoding="utf-8")
    return {"path": str(path), "replaced": True, "strategy": strategy}


def search_files(args: dict) -> dict:
    """Regex-search files under a directory, returning matching lines.

    Args:
        pattern: a Python regex to search for.
        path: directory (or file) to search under; defaults to ".".

    Returns up to ``_MAX_SEARCH_MATCHES`` matches, each with file, line number,
    and line text, plus a ``truncated`` flag when the cap was hit.
    """
    pattern = args["pattern"]
    root = _resolved_within_cwd(Path(args.get("path") or "."))
    try:
        regex = re.compile(pattern)
    except re.error as e:
        raise ValueError(f"invalid regex: {e}")

    candidates = [root] if root.is_file() else sorted(
        p for p in root.rglob("*") if p.is_file()
    )
    matches: list[dict] = []
    truncated = False
    for file_path in candidates:
        try:
            text = file_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            if regex.search(line):
                matches.append(
                    {"file": str(file_path), "line": lineno, "text": line[:400]}
                )
                if len(matches) >= _MAX_SEARCH_MATCHES:
                    truncated = True
                    break
        if truncated:
            break
    return {"matches": matches, "count": len(matches), "truncated": truncated}


def list_directory(args: dict) -> dict:
    """List the immediate entries of a directory.

    Each entry reports its name and whether it's a directory. Not recursive —
    the model can descend by listing subdirectories or use ``search_files``.
    """
    path = _resolved_within_cwd(Path(args.get("path") or "."))
    if not path.is_dir():
        raise NotADirectoryError(f"not a directory: {path}")
    entries = [
        {"name": child.name, "is_dir": child.is_dir()}
        for child in sorted(path.iterdir(), key=lambda p: p.name)
    ]
    return {"path": str(path), "entries": entries}


REGISTRY.register(
    "read_file",
    description="Read a UTF-8 text file and return its contents (truncated if large).",
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to the file to read."},
        },
        "required": ["path"],
    },
    handler=read_file,
    toolset="file",
)

REGISTRY.register(
    "write_file",
    description="Create or overwrite a file with the given content (creates parent dirs).",
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to write."},
            "content": {"type": "string", "description": "Full file contents."},
        },
        "required": ["path", "content"],
    },
    handler=write_file,
    toolset="file",
)

REGISTRY.register(
    "edit_file",
    description=(
        "Replace an exact, unique substring in a file. 'old' must occur exactly "
        "once. Use for surgical patches to existing files."
    ),
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to the file to edit."},
            "old": {"type": "string", "description": "Exact substring to replace (must be unique)."},
            "new": {"type": "string", "description": "Replacement text."},
        },
        "required": ["path", "old", "new"],
    },
    handler=edit_file,
    toolset="file",
)

REGISTRY.register(
    "search_files",
    description="Regex-grep across files under a directory; returns matching lines.",
    parameters={
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Python regex to search for."},
            "path": {"type": "string", "description": "Directory or file to search (default '.')."},
        },
        "required": ["pattern"],
    },
    handler=search_files,
    toolset="file",
)

REGISTRY.register(
    "list_directory",
    description="List the immediate entries (files and subdirectories) of a directory.",
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Directory to list (default '.')."},
        },
        "required": [],
    },
    handler=list_directory,
    toolset="file",
)


__all__ = [
    "read_file",
    "write_file",
    "edit_file",
    "search_files",
    "list_directory",
]

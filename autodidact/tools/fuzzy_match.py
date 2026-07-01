"""Fuzzy find-and-replace for the ``edit_file`` tool.

LLM-generated patches rarely reproduce a file's whitespace exactly — a model
recalls the logical lines but drops trailing spaces, shifts indentation, or
collapses runs of spaces. An exact-match ``str.replace`` then fails, and for a
*learning* agent that retries via cloud escalation, the retry usually drifts
the same way — so the edit just fails twice.

This is a deliberately small adaptation of Hermes' 9-strategy fuzzy chain
(MIT, Nous Research). We keep the two strategies that cover the overwhelming
majority of real LLM drift and skip the long tail (unicode NFC, escape-drift,
block-anchor, context-aware) as over-engineering for our scope:

    1. exact               — str.find; the fast path, no normalization.
    2. line-trimmed        — match after stripping leading/trailing whitespace
                             from every line of both file and pattern.

Both strategies map matches back to *original* byte offsets so the replacement
splices into the untouched file. Matching stays uniqueness-checked: a strategy
that finds more than one match is ambiguous and rejected, same contract as the
exact-only version it replaces.
"""

from __future__ import annotations

from typing import Optional


class FuzzyMatchError(ValueError):
    """Raised when ``old`` cannot be located uniquely by any strategy."""


def fuzzy_replace(content: str, old: str, new: str) -> tuple[str, str]:
    """Replace the unique occurrence of ``old`` in ``content`` with ``new``.

    Tries strategies in increasing fuzziness and returns on the first that
    finds exactly one match. Returns ``(new_content, strategy_name)``.

    Raises FuzzyMatchError if no strategy finds ``old``, or if a strategy
    finds it more than once (ambiguous — we won't guess which to patch).
    """
    for name, finder in (("exact", _find_exact), ("line-trimmed", _find_line_trimmed)):
        span = finder(content, old)
        if span is None:
            continue
        if span == _AMBIGUOUS:
            raise FuzzyMatchError(
                f"'old' string matches multiple locations ({name} strategy); "
                "add surrounding context to make it unique"
            )
        start, end = span
        replacement = new
        if name != "exact":
            # A fuzzy match means old/new were likely sent at a different
            # indent than the file's. Re-anchor new to the file's actual
            # indent so we don't corrupt the surrounding block.
            replacement = _reindent(content[start:end], old, new)
        return content[:start] + replacement + content[end:], name

    raise FuzzyMatchError("'old' string not found in file")


# Sentinel returned by a finder that saw >1 match (ambiguous, not "not found").
_AMBIGUOUS = (-1, -1)


def _leading_ws(line: str) -> str:
    return line[: len(line) - len(line.lstrip(" \t"))]


def _first_meaningful(text: str) -> Optional[str]:
    for line in text.split("\n"):
        if line.strip():
            return line
    return None


def _reindent(file_region: str, old: str, new: str) -> str:
    """Re-anchor ``new``'s indentation onto the file region's actual indent.

    After a line-trimmed match, ``old``/``new`` were likely written at a
    different base indent than the file (e.g. the model de-indented the block).
    We compute the model's base indent (from ``old``) and the file's base
    indent (from ``file_region``), then swap the base prefix on each line of
    ``new`` — preserving the relative nesting the model intended. No-op when
    the two base indents already agree. Adapted from Hermes' _reindent_replacement.
    """
    if not new:
        return new
    old_first = _first_meaningful(old)
    file_first = _first_meaningful(file_region)
    if old_first is None or file_first is None:
        return new
    old_base = _leading_ws(old_first)
    file_base = _leading_ws(file_first)
    if old_base == file_base:
        return new

    out: list[str] = []
    for line in new.split("\n"):
        if not line.strip():
            out.append(line)
        elif line.startswith(old_base):
            out.append(file_base + line[len(old_base):])
        else:
            out.append(file_base + line.lstrip(" \t"))
    return "\n".join(out)


def _find_exact(content: str, pattern: str) -> Optional[tuple[int, int]]:
    """Exact substring search. Returns the span, _AMBIGUOUS, or None."""
    count = content.count(pattern)
    if count == 0:
        return None
    if count > 1:
        return _AMBIGUOUS
    start = content.find(pattern)
    return (start, start + len(pattern))


def _find_line_trimmed(content: str, pattern: str) -> Optional[tuple[int, int]]:
    """Match ignoring per-line leading/trailing whitespace.

    Compares the file and pattern line-by-line with each line ``strip()``-ed,
    then maps the matched line range back to the original character offsets so
    the untrimmed file text is what actually gets spliced.
    """
    pattern_lines = [ln.strip() for ln in pattern.split("\n")]
    content_lines = content.split("\n")
    trimmed = [ln.strip() for ln in content_lines]

    n = len(pattern_lines)
    if n == 0:
        return None

    # Precompute the character offset at the start of each content line.
    line_starts: list[int] = []
    offset = 0
    for ln in content_lines:
        line_starts.append(offset)
        offset += len(ln) + 1  # +1 for the '\n' that split() removed

    matches: list[tuple[int, int]] = []
    for i in range(len(trimmed) - n + 1):
        if trimmed[i : i + n] == pattern_lines:
            start = line_starts[i]
            last = i + n - 1
            end = line_starts[last] + len(content_lines[last])
            matches.append((start, end))

    if not matches:
        return None
    if len(matches) > 1:
        return _AMBIGUOUS
    return matches[0]


__all__ = ["fuzzy_replace", "FuzzyMatchError"]

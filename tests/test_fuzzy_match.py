"""Tests for the fuzzy find-and-replace used by edit_file."""

from __future__ import annotations

import pytest

from autodidact.tools.fuzzy_match import FuzzyMatchError, fuzzy_replace


def test_exact_replace():
    out, strategy = fuzzy_replace("hello world", "world", "there")
    assert out == "hello there"
    assert strategy == "exact"


def test_not_found_raises():
    with pytest.raises(FuzzyMatchError, match="not found"):
        fuzzy_replace("abc", "xyz", "q")


def test_ambiguous_exact_raises():
    with pytest.raises(FuzzyMatchError, match="multiple"):
        fuzzy_replace("a a a", "a", "b")


def test_single_line_indent_handled_by_exact_substring():
    # A single de-indented line is already an exact *substring* of the file,
    # so exact matching splices it in place and preserves indentation. Fuzzy
    # line-trimming is only needed for multi-line blocks (next test).
    content = "    x = 1\n"
    out, strategy = fuzzy_replace(content, "x = 1", "x = 2")
    assert strategy == "exact"
    assert out == "    x = 2\n"


def test_line_trimmed_multiline_block():
    content = "def f():\n        a = 1\n        b = 2\n        return a\n"
    old = "a = 1\nb = 2"  # de-indented, as an LLM might emit
    new = "a = 10\nb = 20"
    out, strategy = fuzzy_replace(content, old, new)
    assert strategy == "line-trimmed"
    assert out == "def f():\n        a = 10\n        b = 20\n        return a\n"


def test_line_trimmed_ambiguous_raises():
    # Two identical de-indented blocks → ambiguous under line-trimming.
    content = "    y = 1\n\n    y = 1\n"
    with pytest.raises(FuzzyMatchError, match="multiple"):
        fuzzy_replace(content, "y = 1", "y = 2")


def test_exact_preferred_over_fuzzy():
    # Exact match exists and is unique → exact strategy wins even though a
    # trimmed match would also be found.
    content = "value = 5"
    out, strategy = fuzzy_replace(content, "value = 5", "value = 6")
    assert strategy == "exact"
    assert out == "value = 6"

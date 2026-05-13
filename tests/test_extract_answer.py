"""Tests for the _extract_answer helper that handles thinking-model responses.

Three response shapes seen in the wild:

1. **Plain content** — qwen2.5, llama3, mistral, anything non-thinking.
2. **Inline <think> tags** — DeepSeek-R1, qwen3 in some Ollama versions.
   Reasoning lives in `content` wrapped in <think>...</think>; the actual
   answer follows after the closing tag.
3. **Separate thinking field** — qwen3:14b on current Ollama. `content`
   carries the final answer (or empty during streaming); `thinking` carries
   the reasoning. We never want to ship `thinking` to the user, but if
   `content` is empty we fall back to it as a last resort so we don't return
   nothing.
"""

from __future__ import annotations

import pytest

from autodidact.llm_client import _extract_answer


# ── Plain content ─────────────────────────────────────────────────


class TestPlainContent:

    def test_plain_answer_returned_unchanged(self):
        msg = {"content": "Paris is the capital of France."}
        assert _extract_answer(msg) == "Paris is the capital of France."

    def test_strips_surrounding_whitespace(self):
        msg = {"content": "  YES  \n"}
        assert _extract_answer(msg) == "YES"


# ── Inline <think> tags ───────────────────────────────────────────


class TestInlineThinkTags:

    def test_strips_think_tags(self):
        msg = {"content": "<think>Let me see.</think>The capital is Paris."}
        assert _extract_answer(msg) == "The capital is Paris."

    def test_strips_multiline_think_tags(self):
        msg = {
            "content": (
                "<think>\nFirst, France is in Europe.\n"
                "Then, its capital is Paris.\n</think>\n"
                "Paris."
            )
        }
        assert _extract_answer(msg) == "Paris."

    def test_strips_multiple_think_blocks(self):
        msg = {
            "content": "<think>step 1</think>maybe<think>step 2</think>Paris."
        }
        assert _extract_answer(msg) == "maybePari​s.".replace("​", "")  # zero-width strip safety
        # Equivalent in a less paranoid form:
        assert _extract_answer(msg) == "maybePari​s.".replace("\u200b", "")


class TestInlineThinkTagsNoTrailingAnswer:
    """Edge case: model emitted only a think block, nothing after."""

    def test_returns_empty_when_only_thinking_in_content(self):
        msg = {"content": "<think>Reasoning that never concluded.</think>"}
        assert _extract_answer(msg) == ""


# ── Separate thinking field ───────────────────────────────────────


class TestSeparateThinkingField:

    def test_uses_content_when_both_present(self):
        """Content has the answer; thinking is just reasoning we ignore."""
        msg = {
            "content": "Paris is the capital of France.",
            "thinking": "France is in Europe; its capital is Paris.",
        }
        assert _extract_answer(msg) == "Paris is the capital of France."

    def test_falls_back_to_thinking_when_content_empty(self):
        """Last-ditch: empty content but non-empty thinking -> use thinking.

        Better than returning nothing; the user gets SOMETHING from the model.
        Routing decisions that read this content (refusal detector etc.) at
        least have text to inspect.
        """
        msg = {
            "content": "",
            "thinking": "I think the answer is Paris.",
        }
        assert _extract_answer(msg) == "I think the answer is Paris."

    def test_falls_back_to_thinking_when_content_whitespace(self):
        msg = {
            "content": "   \n  ",
            "thinking": "Some thought.",
        }
        assert _extract_answer(msg) == "Some thought."


# ── Empty / missing ───────────────────────────────────────────────


class TestEmptyMessage:

    def test_empty_dict_returns_empty(self):
        assert _extract_answer({}) == ""

    def test_none_content_returns_empty(self):
        assert _extract_answer({"content": None}) == ""

    def test_both_fields_empty(self):
        assert _extract_answer({"content": "", "thinking": ""}) == ""


# ── Mixed: think tags AND thinking field ──────────────────────────


class TestMixedShapes:
    """Some models return both — strip tags from content, ignore thinking."""

    def test_strips_inline_tags_and_uses_content(self):
        msg = {
            "content": "<think>Let me think.</think>Paris.",
            "thinking": "External reasoning.",
        }
        assert _extract_answer(msg) == "Paris."

    def test_falls_back_to_thinking_field_only_if_content_truly_empty(self):
        msg = {
            "content": "<think>Thought without conclusion.</think>",
            "thinking": "Final thought field.",
        }
        # After stripping the inline think tag, content is empty; fall back.
        assert _extract_answer(msg) == "Final thought field."

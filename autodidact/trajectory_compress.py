"""Trajectory compression for the executor's growing message list.

Multi-step tool loops overflow small local context windows (Ollama's 4–8K).
Hermes solves this with a 1.5K-LOC service; we adopt only the *strategy*
(docs/HERMES-LEARNINGS.md §3): protect the head (system + original task) and
the tail (recent tool results the model still needs), and replace the stale
middle with one LLM-written summary message.

Kept small and dependency-light: token size is estimated by a cheap char
heuristic, and summarization is injected as a callable so this module needs no
LLM client and stays unit-testable.
"""

from __future__ import annotations

from typing import Callable, Optional

from autodidact.llm_client import ChatMessage


# Rough chars-per-token; deliberately conservative so we compress a little
# early rather than overflow. Good enough for a budget gate — exact tokenization
# would couple this to a specific model's tokenizer for no real gain.
_CHARS_PER_TOKEN = 4


def estimate_tokens(messages: list[ChatMessage]) -> int:
    """Cheap upper-ish estimate of the token footprint of a message list."""
    chars = 0
    for m in messages:
        chars += len(m.content or "")
        for tc in m.tool_calls or []:
            chars += len(tc.name) + len(str(tc.arguments))
    return chars // _CHARS_PER_TOKEN


Summarizer = Callable[[list[ChatMessage]], str]


def compress_if_needed(
    messages: list[ChatMessage],
    *,
    token_budget: int,
    summarize: Summarizer,
    head_keep: int = 2,
    tail_keep: int = 4,
) -> list[ChatMessage]:
    """Return a compressed copy of ``messages`` if it exceeds ``token_budget``.

    Layout preserved: first ``head_keep`` messages (system + original task) +
    a single ``role="user"`` summary of the middle + last ``tail_keep``
    messages. Returns the input unchanged when under budget or too short to
    have a compressible middle.

    Never splits an assistant tool-call from its ``role="tool"`` result: the
    tail boundary is nudged so a kept ``tool`` message keeps its preceding
    ``assistant`` turn.
    """
    if estimate_tokens(messages) <= token_budget:
        return messages
    if len(messages) <= head_keep + tail_keep:
        return messages

    tail_start = len(messages) - tail_keep
    # Don't orphan a tool result: if the first tail message is a tool turn,
    # pull its assistant turn into the tail too.
    if messages[tail_start].role == "tool" and tail_start > head_keep:
        tail_start -= 1

    head = messages[:head_keep]
    middle = messages[head_keep:tail_start]
    tail = messages[tail_start:]

    if not middle:
        return messages

    summary_text = summarize(middle)
    summary_msg = ChatMessage(
        role="user",
        content=f"[Earlier steps summarized to save context]\n{summary_text}",
    )
    return head + [summary_msg] + tail


def make_local_summarizer(
    local_chat: Callable[..., object],
    *,
    max_tokens: int = 512,
) -> Summarizer:
    """Build a Summarizer backed by a local chat model.

    ``local_chat`` is any callable with a ``chat(messages, **opts)`` returning
    an object with ``.content`` (i.e. an LLMClient). Summarization runs on the
    cheap local model — never the cloud — since it's overhead, not the task.
    """
    def summarize(middle: list[ChatMessage]) -> str:
        transcript = _render(middle)
        prompt = [
            ChatMessage(
                role="system",
                content=(
                    "Summarize the following agent tool-execution steps into a "
                    "compact factual recap. Preserve what was done, key results, "
                    "and any values needed for later steps. No preamble."
                ),
            ),
            ChatMessage(role="user", content=transcript),
        ]
        resp = local_chat.chat(prompt, max_tokens=max_tokens, temperature=0.0)
        return getattr(resp, "content", "") or ""

    return summarize


def _render(messages: list[ChatMessage]) -> str:
    lines: list[str] = []
    for m in messages:
        if m.tool_calls:
            for tc in m.tool_calls:
                lines.append(f"[tool call] {tc.name}({tc.arguments})")
        if m.content:
            lines.append(f"[{m.role}] {m.content}")
    return "\n".join(lines)


__all__ = ["compress_if_needed", "estimate_tokens", "make_local_summarizer", "Summarizer"]

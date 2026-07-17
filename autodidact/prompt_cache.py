"""Anthropic prompt-caching for the executor's cloud escalation path.

The tiered loop re-sends a byte-stable prefix (system prompt + tool schemas)
on every iteration. On the cloud tier that prefix is billed as input tokens
each time — the exact cost Autodidact's routing exists to minimise. Anthropic
prompt caching cuts repeated-prefix input cost by ~75% within a session.

Pattern adopted from Hermes' ``agent/prompt_caching.py`` (``system_and_3``
layout, pure functions, no class state — see docs/HERMES-LEARNINGS.md §3). We
port the strategy, not the file. Places up to 4 ``cache_control`` breakpoints:
the system message plus the last 3 non-system messages, all at one TTL.

This is opt-in: the caller applies it only when the cloud endpoint is
Anthropic-compatible, because ``cache_control`` fields are rejected by strict
OpenAI. Operates on *wire-format* message dicts (post-serialization), never on
our ``ChatMessage`` objects.
"""

from __future__ import annotations

import copy
from typing import Any


_MAX_BREAKPOINTS = 4


def _cache_marker(ttl: str) -> dict[str, str]:
    """Build a cache_control marker for the given TTL ('5m' or '1h')."""
    marker: dict[str, str] = {"type": "ephemeral"}
    if ttl == "1h":
        marker["ttl"] = "1h"
    return marker


def _apply_marker(msg: dict, marker: dict) -> None:
    """Attach a cache_control marker to one wire message, handling the string
    vs. content-block content shapes."""
    content = msg.get("content")

    # Empty/None content or a tool-result turn: mark the message itself.
    if content is None or content == "" or msg.get("role") == "tool":
        msg["cache_control"] = marker
        return

    # String content → wrap in a single text block carrying the marker.
    if isinstance(content, str):
        msg["content"] = [
            {"type": "text", "text": content, "cache_control": marker}
        ]
        return

    # Already a list of blocks → mark the last block.
    if isinstance(content, list) and content:
        last = content[-1]
        if isinstance(last, dict):
            last["cache_control"] = marker


def apply_anthropic_cache_control(
    wire_messages: list[dict[str, Any]], cache_ttl: str = "5m",
) -> list[dict[str, Any]]:
    """Return a deep copy of ``wire_messages`` with up to 4 cache breakpoints:
    the system message + the last 3 non-system messages, all at ``cache_ttl``.

    The input is left unmodified.
    """
    messages = copy.deepcopy(wire_messages)
    if not messages:
        return messages

    marker = _cache_marker(cache_ttl)
    used = 0

    if messages[0].get("role") == "system":
        _apply_marker(messages[0], marker)
        used += 1

    remaining = _MAX_BREAKPOINTS - used
    non_system = [i for i in range(len(messages)) if messages[i].get("role") != "system"]
    for idx in non_system[-remaining:]:
        _apply_marker(messages[idx], marker)

    return messages


__all__ = ["apply_anthropic_cache_control"]

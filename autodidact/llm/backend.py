"""ChatBackend protocol and shared helpers.

The protocol pins the interface every provider adapter must satisfy. Concrete
adapters live in autodidact/llm/{ollama,openai,bedrock}.py.

The shared helpers (_consume_ollama_stream, _consume_ollama_stream_plain,
_extract_answer, _with_retries, _BedrockThrottleError) used to live at the
top of llm_client.py. They're kept here so each adapter can import them
without forming a cycle through llm_client.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any, Callable, Optional, Protocol, TYPE_CHECKING, TypeVar

import numpy as np

if TYPE_CHECKING:
    from autodidact.llm_client import (
        ChatMessage,
        ChatResponse,
        ChatResponseWithLogprobs,
    )


logger = logging.getLogger(__name__)


# ── ChatBackend protocol ────────────────────────────────────────


class ChatBackend(Protocol):
    """A single-provider implementation of the chat / embed / stream surface.

    The Protocol is structural — any class with these methods satisfies it.
    Adapters don't inherit from ChatBackend; they just implement the methods.
    """

    def chat(self, messages: "list[ChatMessage]", **opts: Any) -> "ChatResponse":
        """Plain chat completion."""

    def chat_with_logprobs(
        self, messages: "list[ChatMessage]", **opts: Any
    ) -> "ChatResponseWithLogprobs":
        """Chat completion with per-token logprobs (empty when unsupported)."""

    def chat_stream(
        self,
        messages: "list[ChatMessage]",
        *,
        on_token: Callable[[dict], None],
        **opts: Any,
    ) -> "ChatResponse":
        """Streaming chat. Calls on_token with {phase, text} per chunk.
        Returns the accumulated ChatResponse after the stream ends."""

    def embed(self, text: str) -> np.ndarray:
        """Return an embedding for `text`. Raises LLMClientError if the
        backend does not support embeddings (Bedrock today)."""


# ── Internal exception for retry signaling ──────────────────────


class _BedrockThrottleError(Exception):
    """Internal marker for throttle-class Bedrock errors.

    Not exported to users. Used to smuggle retryable throttle responses
    through `_with_retries` without asking callers to import botocore
    exception types.
    """


# ── Retry helper ────────────────────────────────────────────────


T = TypeVar("T")

# Exponential backoff. First entry used for attempt 1 → 2, etc. Extended vs
# the old (0.5, 1, 2) so Bedrock throttle bursts (observed in EXP-003 as 69
# consecutive failures across ~12 seconds) have a chance to clear.
_BACKOFF_SECONDS = (1.0, 2.0, 4.0, 8.0, 16.0)


def _with_retries(
    fn: Callable[[], T], max_retries: int, on_transient: tuple[type, ...],
) -> T:
    """Run fn with exponential backoff on transient failures.

    Retries on exceptions listed in on_transient only. Non-listed exceptions
    (including HTTP 4xx raised as LLMClientError) propagate immediately.
    """
    # Imported lazily to avoid a circular import: llm_client imports the
    # backends package, the backends import this helper, and the helper
    # raises LLMClientError. Without lazy import, llm_client's import of the
    # `llm` package could find a half-initialized backend module.
    from autodidact.llm_client import LLMClientError

    last_err: Optional[Exception] = None
    attempts = max(1, max_retries)
    for i in range(attempts):
        try:
            return fn()
        except on_transient as e:
            last_err = e
            if i == attempts - 1:
                break
            sleep_s = _BACKOFF_SECONDS[min(i, len(_BACKOFF_SECONDS) - 1)]
            logger.warning(
                "Transient LLM client failure (attempt %d/%d); retrying in %.1fs",
                i + 1,
                attempts,
                sleep_s,
            )
            time.sleep(sleep_s)
    assert last_err is not None
    raise LLMClientError(
        f"Transient failure after {attempts} attempts: {type(last_err).__name__}"
    ) from last_err


# ── Answer extraction (handles thinking models) ──────────────────
#
# Three response shapes seen in the wild:
#   1. Plain content      — qwen2.5, llama, mistral. Just use content as-is.
#   2. Inline <think>...</think> — DeepSeek-R1, qwen3 in some configs. The
#      reasoning is wrapped in tags within `content`; the answer follows
#      after the closing tag.
#   3. Separate `thinking` field — qwen3:14b on current Ollama. `content`
#      holds the answer, `thinking` holds the reasoning. We never expose
#      `thinking` to the user, but if `content` is empty we fall back to
#      it as a last-ditch so we don't return nothing.

_THINK_TAG_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)


def _extract_answer(message: dict) -> str:
    """Return the user-facing answer text from an Ollama chat message dict.

    Handles thinking models without surfacing reasoning to the caller. If
    both `content` and `thinking` are empty after cleanup, returns "".
    """
    content = (message.get("content") or "")
    thinking = (message.get("thinking") or "")

    cleaned = _THINK_TAG_RE.sub("", content).strip()
    if cleaned:
        return cleaned

    return thinking.strip()


# ── Streaming consumers (Ollama NDJSON) ─────────────────────────


def _consume_ollama_stream(
    resp: Any,
    on_token: Callable[[dict], None],
    fallback_model: str,
    started: float,
) -> "ChatResponseWithLogprobs":
    """Read NDJSON chunks from an Ollama streaming response.

    Each chunk has shape ``{"message": {"content": "...", "thinking": "..."},
    "done": false}`` until the final chunk where ``done: true`` brings
    ``prompt_eval_count``, ``eval_count``, and (on Ollama 0.12.11+) the full
    logprobs array.

    For each non-empty content/thinking delta, calls ``on_token`` with
    ``{"phase": "content" | "thinking", "text": "..."}``. A bad NDJSON line
    (rare; Ollama might write a partial chunk on error) is logged and
    skipped, not fatal.
    """
    import json as _json

    # Lazy import to avoid circularity.
    from autodidact.llm_client import ChatResponseWithLogprobs

    content_buf: list[str] = []
    thinking_buf: list[str] = []
    final_data: dict[str, Any] = {}

    for raw_line in resp.iter_lines():
        if not raw_line:
            continue
        try:
            chunk = _json.loads(raw_line)
        except (ValueError, TypeError) as e:
            logger.warning("Ollama streaming chunk could not be parsed: %s", e)
            continue

        message = chunk.get("message") or {}
        delta_content = message.get("content") or ""
        delta_thinking = message.get("thinking") or ""

        if delta_thinking:
            thinking_buf.append(delta_thinking)
            on_token({"phase": "thinking", "text": delta_thinking})
        if delta_content:
            content_buf.append(delta_content)
            on_token({"phase": "content", "text": delta_content})

        if chunk.get("done"):
            final_data = chunk

    latency_ms = int((time.perf_counter() - started) * 1000)

    full_content = "".join(content_buf)
    if not full_content.strip():
        full_content = "".join(thinking_buf).strip()

    raw_lp = final_data.get("logprobs")
    if raw_lp is None:
        raw_lp = (final_data.get("message") or {}).get("logprobs")

    token_lps: list[float] = []
    top_lps: list[dict[str, float]] = []
    if isinstance(raw_lp, list):
        for item in raw_lp:
            if isinstance(item, (int, float)):
                token_lps.append(float(item))
                top_lps.append({})
            elif isinstance(item, dict):
                lp = item.get("logprob")
                if isinstance(lp, (int, float)):
                    token_lps.append(float(lp))
                top = item.get("top_logprobs")
                if isinstance(top, dict):
                    top_lps.append({str(k): float(v) for k, v in top.items()})
                elif isinstance(top, list):
                    top_lps.append({
                        str(t.get("token", "")): float(t.get("logprob", 0.0))
                        for t in top
                        if isinstance(t, dict) and "token" in t
                    })
                else:
                    top_lps.append({})
    avg_lp = float(np.mean(token_lps)) if token_lps else None

    return ChatResponseWithLogprobs(
        content=full_content,
        model=final_data.get("model", fallback_model),
        input_tokens=int(final_data.get("prompt_eval_count", 0) or 0),
        output_tokens=int(final_data.get("eval_count", 0) or 0),
        latency_ms=latency_ms,
        logprobs=token_lps,
        avg_logprob=avg_lp,
        top_logprobs_by_position=top_lps,
        had_thinking=bool(thinking_buf),
    )


def _consume_ollama_stream_plain(
    resp: Any,
    on_token: Callable[[dict], None],
    fallback_model: str,
    started: float,
) -> "ChatResponse":
    """Read NDJSON chunks from an Ollama streaming response (no logprobs).

    Mirrors ``_consume_ollama_stream`` but skips logprob parsing and returns
    a plain ``ChatResponse``. Used by ``OllamaBackend.chat_stream_no_logprobs``
    to save the ~150ms per-call overhead Ollama adds when computing top-k
    token probabilities.
    """
    import json as _json

    from autodidact.llm_client import ChatResponse

    content_buf: list[str] = []
    thinking_buf: list[str] = []
    final_data: dict[str, Any] = {}

    for raw_line in resp.iter_lines():
        if not raw_line:
            continue
        try:
            chunk = _json.loads(raw_line)
        except (ValueError, TypeError) as e:
            logger.warning("Ollama streaming chunk could not be parsed: %s", e)
            continue

        message = chunk.get("message") or {}
        delta_content = message.get("content") or ""
        delta_thinking = message.get("thinking") or ""

        if delta_thinking:
            thinking_buf.append(delta_thinking)
            on_token({"phase": "thinking", "text": delta_thinking})
        if delta_content:
            content_buf.append(delta_content)
            on_token({"phase": "content", "text": delta_content})

        if chunk.get("done"):
            final_data = chunk

    latency_ms = int((time.perf_counter() - started) * 1000)

    full_content = "".join(content_buf)
    if not full_content.strip():
        full_content = "".join(thinking_buf).strip()

    return ChatResponse(
        content=full_content,
        model=final_data.get("model", fallback_model),
        input_tokens=int(final_data.get("prompt_eval_count", 0) or 0),
        output_tokens=int(final_data.get("eval_count", 0) or 0),
        latency_ms=latency_ms,
    )


__all__ = [
    "ChatBackend",
    "_BedrockThrottleError",
    "_consume_ollama_stream",
    "_consume_ollama_stream_plain",
    "_extract_answer",
    "_with_retries",
]

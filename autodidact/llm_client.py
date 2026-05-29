"""Unified LLM client for Autodidact.

Supports three backends:
- Ollama (local HTTP, used as the local model; has logprob support)
- OpenAI-compatible (any /v1 chat-completions API: vLLM, LM Studio, OpenAI,
  Together, Groq, Fireworks, ...)
- AWS Bedrock via boto3 (Converse API; no logprobs)

The implementation lives in ``autodidact/llm/`` — one ChatBackend adapter per
provider. ``LLMClient`` here is a thin coordinator: it picks the right adapter
at construction and forwards public method calls to it. This eliminates the
per-provider ``if self.config.provider == "x"`` branching that used to live
inside every public method.

The client exposes:
- chat: plain completion
- chat_with_logprobs: completion with per-token logprobs when supported
- chat_stream: streaming completion (provider-agnostic)
- chat_stream_ollama / _ollama_no_logprobs / _openai / _bedrock — provider-
  specific streaming entry points (kept on the public surface for tests
  and external callers that explicitly want a particular backend's stream)
- _get_openai_client / _get_bedrock_client — preserved as patch points for
  tests that mock the underlying provider client
- embed: embedding for retrieval

``import requests`` and ``import time`` at the top of this module remain on
purpose — existing tests patch ``autodidact.llm_client.requests.post`` and
``autodidact.llm_client.time.sleep`` and rely on those paths resolving to
the live module objects (which the OllamaBackend implementation in
``autodidact/llm/ollama.py`` ALSO imports — both paths point at the same
module, so patches affect both).
"""

from __future__ import annotations

import logging
import os  # noqa: F401  # kept for back-compat with importers
import re  # noqa: F401
import time  # noqa: F401  # patched by tests via autodidact.llm_client.time
from dataclasses import dataclass
from typing import Any, Callable, Literal, Optional

import numpy as np  # noqa: F401  # kept for back-compat with importers
import requests  # noqa: F401  # patched by tests via autodidact.llm_client.requests
from pydantic import BaseModel

logger = logging.getLogger(__name__)


# ── Data models ────────────────────────────────────────────────────


class LLMConfig(BaseModel):
    """Configuration for an LLMClient instance. Pydantic — boundary type."""

    provider: Literal["ollama", "openai", "bedrock"]
    model: str
    embedding_model: Optional[str] = None
    base_url: Optional[str] = None
    api_key_env: Optional[str] = None
    region: str = "us-west-2"
    bedrock_auth_mode: Literal["default", "iam_user", "api_key"] = "default"
    bedrock_access_key_id: Optional[str] = None
    bedrock_secret_access_key: Optional[str] = None
    bedrock_session_token: Optional[str] = None
    bedrock_api_key: Optional[str] = None
    timeout_seconds: int = 300
    max_retries: int = 6


@dataclass
class ChatMessage:
    """One turn of a chat. Internal — built by us, never parsed from untrusted input."""

    role: Literal["system", "user", "assistant"]
    content: str


class ChatResponse(BaseModel):
    """Response from a chat call. Pydantic — wraps data returned by external APIs."""

    content: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: int = 0


class ChatResponseWithLogprobs(ChatResponse):
    """Chat response extended with per-token logprobs."""

    logprobs: list[float] = []
    avg_logprob: Optional[float] = None
    top_logprobs_by_position: list[dict[str, float]] = []
    had_thinking: bool = False


# ── Exceptions ─────────────────────────────────────────────────────


class LLMClientError(Exception):
    """Raised for unrecoverable LLM client errors. Messages never include credential values."""


# ── Backend re-exports ─────────────────────────────────────────────
#
# These were originally module-level helpers in this file. They live in
# ``autodidact/llm/backend.py`` now. Re-exported here so:
#   1. ``from autodidact.llm_client import _BedrockThrottleError, _with_retries,
#       _extract_answer, _consume_ollama_stream`` keeps working
#   2. tests that monkeypatch ``autodidact.llm_client.<helper>`` keep working
#
# The re-exports come AFTER the data model declarations because the backend
# module imports them under TYPE_CHECKING.

from autodidact.llm import (  # noqa: E402  (intentional late import to break cycle)
    BedrockBackend,
    OllamaBackend,
    OpenAICompatBackend,
    _BedrockThrottleError,
    _consume_ollama_stream,
    _consume_ollama_stream_plain,
    _extract_answer,
    _with_retries,
)

# Tokenizer-strip regex — re-exported for tests that import _THINK_TAG_RE.
from autodidact.llm.ollama import _THINK_TAG_RE  # noqa: E402


# ── LLMClient ──────────────────────────────────────────────────────


# Canonical name → adapter class. Stays a small, explicit table so adding
# a fourth provider is one line.
_BACKEND_BY_PROVIDER = {
    "ollama": OllamaBackend,
    "openai": OpenAICompatBackend,
    "bedrock": BedrockBackend,
}


class LLMClient:
    """Thin coordinator over per-provider ChatBackend adapters.

    Picks the right adapter at construction from ``config.provider`` and
    forwards public method calls to it. The provider-specific facades
    (``chat_stream_ollama``, ``_get_openai_client``, etc.) are preserved
    for backwards compatibility with test fixtures and external callers.
    """

    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        backend_cls = _BACKEND_BY_PROVIDER.get(config.provider)
        if backend_cls is None:
            raise LLMClientError(
                f"Unknown provider {config.provider!r}. "
                f"Expected one of: {sorted(_BACKEND_BY_PROVIDER)}."
            )
        self._backend: Any = backend_cls(config)

    # ── Provider-agnostic public API (forwards to backend) ────────

    def chat(self, messages: list[ChatMessage], **opts: Any) -> ChatResponse:
        return self._backend.chat(messages, **opts)

    def chat_with_logprobs(
        self, messages: list[ChatMessage], **opts: Any,
    ) -> ChatResponseWithLogprobs:
        return self._backend.chat_with_logprobs(messages, **opts)

    def chat_stream(
        self,
        messages: list[ChatMessage],
        *,
        on_token: Callable[[dict], None],
        **opts: Any,
    ) -> ChatResponse:
        """Provider-agnostic streaming chat. Routes to the per-provider facade.

        Routes through ``chat_stream_ollama`` / ``chat_stream_openai`` /
        ``chat_stream_bedrock`` so tests that ``patch.object(client,
        "chat_stream_ollama", ...)`` keep working.
        """
        if self.config.provider == "ollama":
            return self.chat_stream_ollama(messages, on_token=on_token, **opts)
        if self.config.provider == "openai":
            return self.chat_stream_openai(messages, on_token=on_token, **opts)
        return self.chat_stream_bedrock(messages, on_token=on_token, **opts)

    def embed(self, text: str) -> np.ndarray:
        return self._backend.embed(text)

    # ── Provider-specific facades (kept for tests / external callers) ─
    #
    # These exist so:
    #   - tests/test_streaming_cloud.py::TestChatStreamDispatch can call
    #     chat_stream_ollama / _openai / _bedrock and patch them by name
    #   - tests using MagicMock(spec=LLMClient) get the same method set
    #     they had before
    #   - autodidact/agent.py::Agent._call_local can call
    #     chat_stream_ollama_no_logprobs explicitly without going through
    #     the provider-agnostic dispatcher

    def chat_stream_ollama(
        self,
        messages: list[ChatMessage],
        *,
        on_token: Callable[[dict], None],
        **opts: Any,
    ) -> ChatResponseWithLogprobs:
        if not isinstance(self._backend, OllamaBackend):
            raise LLMClientError(
                f"chat_stream_ollama requires an ollama provider; "
                f"got {self.config.provider!r}."
            )
        return self._backend.chat_stream(messages, on_token=on_token, **opts)

    def chat_stream_ollama_no_logprobs(
        self,
        messages: list[ChatMessage],
        *,
        on_token: Callable[[dict], None],
        **opts: Any,
    ) -> ChatResponse:
        if not isinstance(self._backend, OllamaBackend):
            raise LLMClientError(
                f"chat_stream_ollama_no_logprobs requires an ollama provider; "
                f"got {self.config.provider!r}."
            )
        return self._backend.chat_stream_no_logprobs(messages, on_token=on_token, **opts)

    def chat_stream_openai(
        self,
        messages: list[ChatMessage],
        *,
        on_token: Callable[[dict], None],
        **opts: Any,
    ) -> ChatResponse:
        if not isinstance(self._backend, OpenAICompatBackend):
            raise LLMClientError(
                f"chat_stream_openai requires an openai provider; "
                f"got {self.config.provider!r}."
            )
        return self._backend.chat_stream(messages, on_token=on_token, **opts)

    def chat_stream_bedrock(
        self,
        messages: list[ChatMessage],
        *,
        on_token: Callable[[dict], None],
        **opts: Any,
    ) -> ChatResponse:
        if not isinstance(self._backend, BedrockBackend):
            raise LLMClientError(
                f"chat_stream_bedrock requires a bedrock provider; "
                f"got {self.config.provider!r}."
            )
        return self._backend.chat_stream(messages, on_token=on_token, **opts)

    # ── Provider-client facades (test patch points) ───────────────

    def _get_openai_client(self) -> Any:
        """Patch point preserved for tests that
        ``patch.object(client, "_get_openai_client", ...)``."""
        if not isinstance(self._backend, OpenAICompatBackend):
            raise LLMClientError(
                f"_get_openai_client requires an openai provider; "
                f"got {self.config.provider!r}."
            )
        return self._backend._get_client()

    def _get_bedrock_client(self) -> Any:
        """Patch point preserved for tests that
        ``patch.object(client, "_get_bedrock_client", ...)``."""
        if not isinstance(self._backend, BedrockBackend):
            raise LLMClientError(
                f"_get_bedrock_client requires a bedrock provider; "
                f"got {self.config.provider!r}."
            )
        return self._backend._get_client()


__all__ = [
    "ChatMessage",
    "ChatResponse",
    "ChatResponseWithLogprobs",
    "LLMClient",
    "LLMClientError",
    "LLMConfig",
    # Internal helpers re-exported for tests + direct imports.
    "_BedrockThrottleError",
    "_THINK_TAG_RE",
    "_consume_ollama_stream",
    "_consume_ollama_stream_plain",
    "_extract_answer",
    "_with_retries",
]

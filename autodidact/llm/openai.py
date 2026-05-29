"""OpenAI-compatible backend.

Works with any server speaking the OpenAI chat-completions API: OpenAI
itself, vLLM, LM Studio, llama.cpp server, text-generation-inference,
together.ai, Anyscale, Groq, Fireworks, and more.

Owns a lazily-built OpenAI client. The actual client construction is
gated on first use so import-time has no side effects when the openai
package isn't installed.
"""

from __future__ import annotations

import os
import time
from dataclasses import asdict
from typing import Any, Callable, TYPE_CHECKING

import numpy as np

from autodidact.llm.backend import _with_retries

if TYPE_CHECKING:
    from autodidact.llm_client import (
        ChatMessage,
        ChatResponse,
        ChatResponseWithLogprobs,
        LLMConfig,
    )


class OpenAICompatBackend:
    """ChatBackend implementation for OpenAI-compatible APIs."""

    def __init__(self, config: "LLMConfig") -> None:
        self.config = config

    # ── ChatBackend interface ────────────────────────────────────

    def chat(self, messages: "list[ChatMessage]", **opts: Any) -> "ChatResponse":
        from autodidact.llm_client import ChatResponse

        client = self._get_client()
        kwargs = self._common_kwargs(messages, opts)

        def do() -> "ChatResponse":
            started = time.perf_counter()
            try:
                resp = client.chat.completions.create(**kwargs)
            except Exception as e:
                self._maybe_raise_4xx(e)
                raise
            latency_ms = int((time.perf_counter() - started) * 1000)
            choice = resp.choices[0]
            usage = resp.usage
            return ChatResponse(
                content=choice.message.content or "",
                model=resp.model,
                input_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
                output_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
                latency_ms=latency_ms,
            )

        return _with_retries(do, self.config.max_retries, self._transient_exceptions())

    def chat_with_logprobs(
        self, messages: "list[ChatMessage]", **opts: Any
    ) -> "ChatResponseWithLogprobs":
        from autodidact.llm_client import ChatResponseWithLogprobs

        client = self._get_client()
        top_logprobs_k = int(opts.pop("top_logprobs", 5))
        kwargs = self._common_kwargs(messages, opts)
        kwargs["logprobs"] = True
        kwargs["top_logprobs"] = top_logprobs_k

        def do() -> "ChatResponseWithLogprobs":
            started = time.perf_counter()
            try:
                resp = client.chat.completions.create(**kwargs)
            except Exception as e:
                self._maybe_raise_4xx(e)
                raise
            latency_ms = int((time.perf_counter() - started) * 1000)
            choice = resp.choices[0]
            usage = resp.usage
            content = choice.message.content or ""

            token_lps: list[float] = []
            top_lps: list[dict[str, float]] = []
            lp_container = getattr(choice, "logprobs", None)
            if lp_container is not None and getattr(lp_container, "content", None):
                for item in lp_container.content:
                    token_lps.append(float(item.logprob))
                    top_map: dict[str, float] = {}
                    for alt in getattr(item, "top_logprobs", []) or []:
                        top_map[str(alt.token)] = float(alt.logprob)
                    top_lps.append(top_map)
            avg_lp = float(np.mean(token_lps)) if token_lps else None

            return ChatResponseWithLogprobs(
                content=content,
                model=resp.model,
                input_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
                output_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
                latency_ms=latency_ms,
                logprobs=token_lps,
                avg_logprob=avg_lp,
                top_logprobs_by_position=top_lps,
            )

        return _with_retries(do, self.config.max_retries, self._transient_exceptions())

    def chat_stream(
        self,
        messages: "list[ChatMessage]",
        *,
        on_token: Callable[[dict], None],
        **opts: Any,
    ) -> "ChatResponse":
        from autodidact.llm_client import ChatResponse

        client = self._get_client()
        kwargs = self._common_kwargs(messages, opts)
        kwargs["stream"] = True
        kwargs["stream_options"] = {"include_usage": True}

        def do() -> "ChatResponse":
            started = time.perf_counter()
            content_buf: list[str] = []
            input_tokens = 0
            output_tokens = 0
            model = self.config.model

            try:
                stream = client.chat.completions.create(**kwargs)
            except Exception as e:
                self._maybe_raise_4xx(e)
                raise

            for chunk in stream:
                usage = getattr(chunk, "usage", None)
                if usage is not None:
                    input_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
                    output_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
                model = getattr(chunk, "model", model) or model

                choices = getattr(chunk, "choices", None) or []
                for choice in choices:
                    delta = getattr(choice, "delta", None)
                    if delta is None:
                        continue
                    text = getattr(delta, "content", None) or ""
                    if text:
                        content_buf.append(text)
                        on_token({"phase": "content", "text": text})

            latency_ms = int((time.perf_counter() - started) * 1000)
            return ChatResponse(
                content="".join(content_buf),
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency_ms=latency_ms,
            )

        return _with_retries(do, self.config.max_retries, self._transient_exceptions())

    def embed(self, text: str) -> np.ndarray:
        client = self._get_client()
        model = self.config.embedding_model or "text-embedding-3-small"

        def do() -> np.ndarray:
            try:
                resp = client.embeddings.create(model=model, input=text)
            except Exception as e:
                self._maybe_raise_4xx(e)
                raise
            vec = resp.data[0].embedding
            return np.asarray(vec, dtype=np.float32)

        return _with_retries(do, self.config.max_retries, self._transient_exceptions())

    # ── Internals ────────────────────────────────────────────────

    def _get_client(self) -> Any:
        from autodidact.llm_client import LLMClientError

        if not self.config.base_url:
            raise LLMClientError(
                "OpenAI-compatible provider requires base_url in LLMConfig "
                "(e.g. http://localhost:8000/v1 for vLLM, "
                "https://api.openai.com/v1 for OpenAI)."
            )
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as e:
            raise LLMClientError(
                "openai package is required for the 'openai' provider."
            ) from e
        api_key = "sk-no-auth"
        if self.config.api_key_env:
            api_key = os.environ.get(self.config.api_key_env) or api_key
        return OpenAI(
            api_key=api_key,
            base_url=self.config.base_url,
            timeout=self.config.timeout_seconds,
        )

    def _transient_exceptions(self) -> tuple[type, ...]:
        try:
            import openai  # type: ignore
            return (
                openai.APIConnectionError,
                openai.APITimeoutError,
                openai.RateLimitError,
            )
        except ImportError:
            return ()

    def _maybe_raise_4xx(self, exc: Exception) -> None:
        """Convert provider 4xx errors to non-retryable LLMClientError."""
        from autodidact.llm_client import LLMClientError

        try:
            import openai  # type: ignore
        except ImportError:
            return
        if isinstance(exc, (openai.AuthenticationError, openai.PermissionDeniedError)):
            raise LLMClientError("openai rejected credentials (4xx)")
        if isinstance(exc, openai.BadRequestError):
            raise LLMClientError(f"openai rejected request (400): {str(exc)[:200]}")
        if isinstance(exc, openai.NotFoundError):
            raise LLMClientError(f"openai model not found (404): {self.config.model}")

    def _common_kwargs(
        self, messages: "list[ChatMessage]", opts: dict,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self.config.model,
            "messages": [asdict(m) for m in messages],
        }
        if "max_tokens" in opts:
            kwargs["max_tokens"] = int(opts["max_tokens"])
        if "temperature" in opts:
            kwargs["temperature"] = float(opts["temperature"])
        if "top_p" in opts:
            kwargs["top_p"] = float(opts["top_p"])
        if "seed" in opts:
            kwargs["seed"] = int(opts["seed"])
        return kwargs


__all__ = ["OpenAICompatBackend"]

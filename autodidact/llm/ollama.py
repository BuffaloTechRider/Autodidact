"""Ollama backend (local HTTP).

Implements ChatBackend for Ollama's /api/chat and /api/embeddings. Owns
the resolved Ollama host (from OLLAMA_HOST env var, defaulting to
http://localhost:11434) at construction time.

Supports both logprobs and a no-logprobs streaming variant, the latter
to skip the ~150ms-per-call overhead Ollama adds when computing top-k
token probabilities.
"""

from __future__ import annotations

import os
import time
from dataclasses import asdict
from typing import Any, Callable, TYPE_CHECKING

import numpy as np
import requests

from autodidact.llm.backend import (
    _consume_ollama_stream,
    _consume_ollama_stream_plain,
    _extract_answer,
    _with_retries,
)
from autodidact.llm.backend import _THINK_TAG_RE  # noqa: F401  # for had_thinking detection

if TYPE_CHECKING:
    from autodidact.llm_client import (
        ChatMessage,
        ChatResponse,
        ChatResponseWithLogprobs,
        LLMConfig,
    )


# We import _THINK_TAG_RE from backend at module scope above; importing it
# under the TYPE_CHECKING guard wouldn't be visible at runtime. Re-export
# below so existing test imports (autodidact.llm_client._THINK_TAG_RE) keep
# working via the llm_client facade.
import re as _re
_THINK_TAG_RE = _re.compile(r"<think>.*?</think>\s*", _re.DOTALL | _re.IGNORECASE)


class OllamaBackend:
    """ChatBackend implementation for Ollama HTTP API."""

    def __init__(self, config: "LLMConfig") -> None:
        self.config = config
        self._host = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")

    # ── ChatBackend interface ────────────────────────────────────

    def chat(self, messages: "list[ChatMessage]", **opts: Any) -> "ChatResponse":
        from autodidact.llm_client import ChatResponse

        think = opts.pop("think", None)
        body = {
            "model": self.config.model,
            "messages": [asdict(m) for m in messages],
            "stream": False,
            "options": self._options(opts),
        }
        if think is not None:
            body["think"] = bool(think)

        started = time.perf_counter()
        data = self._post("/api/chat", body)
        latency_ms = int((time.perf_counter() - started) * 1000)

        content = _extract_answer(data.get("message") or {})
        return ChatResponse(
            content=content,
            model=data.get("model", self.config.model),
            input_tokens=int(data.get("prompt_eval_count", 0) or 0),
            output_tokens=int(data.get("eval_count", 0) or 0),
            latency_ms=latency_ms,
        )

    def chat_with_logprobs(
        self, messages: "list[ChatMessage]", **opts: Any
    ) -> "ChatResponseWithLogprobs":
        from autodidact.llm_client import ChatResponseWithLogprobs

        options = self._options(opts)
        options.setdefault("num_predict", options.get("max_tokens", 256))
        top_logprobs_k = int(opts.pop("top_logprobs", 5))
        think = opts.pop("think", None)
        body = {
            "model": self.config.model,
            "messages": [asdict(m) for m in messages],
            "stream": False,
            "logprobs": True,
            "top_logprobs": top_logprobs_k,
            "options": options,
        }
        if think is not None:
            body["think"] = bool(think)

        started = time.perf_counter()
        data = self._post("/api/chat", body)
        latency_ms = int((time.perf_counter() - started) * 1000)

        message = data.get("message") or {}
        content = _extract_answer(message)

        # Logprobs: top-level on Ollama 0.12.11+, message-level on older.
        token_lps: list[float] = []
        top_lps: list[dict[str, float]] = []
        raw_lp = data.get("logprobs")
        if raw_lp is None:
            raw_lp = message.get("logprobs")
        if isinstance(raw_lp, list) and raw_lp:
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
            content=content,
            model=data.get("model", self.config.model),
            input_tokens=int(data.get("prompt_eval_count", 0) or 0),
            output_tokens=int(data.get("eval_count", 0) or 0),
            latency_ms=latency_ms,
            logprobs=token_lps,
            avg_logprob=avg_lp,
            top_logprobs_by_position=top_lps,
            had_thinking=bool(message.get("thinking"))
            or bool(_THINK_TAG_RE.search(message.get("content") or "")),
        )

    def chat_stream(
        self,
        messages: "list[ChatMessage]",
        *,
        on_token: Callable[[dict], None],
        **opts: Any,
    ) -> "ChatResponseWithLogprobs":
        """Stream a chat response WITH logprobs."""
        from autodidact.llm_client import LLMClientError

        options = self._options(opts)
        options.setdefault("num_predict", options.get("max_tokens", 1024))
        top_logprobs_k = int(opts.pop("top_logprobs", 5))
        think = opts.pop("think", None)

        body: dict[str, Any] = {
            "model": self.config.model,
            "messages": [asdict(m) for m in messages],
            "stream": True,
            "logprobs": True,
            "top_logprobs": top_logprobs_k,
            "options": options,
        }
        if think is not None:
            body["think"] = bool(think)

        url = f"{self._host}/api/chat"
        started = time.perf_counter()

        def do() -> "ChatResponseWithLogprobs":
            try:
                resp = requests.post(
                    url, json=body, stream=True,
                    timeout=self.config.timeout_seconds,
                )
            except requests.exceptions.ReadTimeout as e:
                raise LLMClientError(
                    f"Ollama read timeout after {self.config.timeout_seconds}s "
                    f"during streaming /api/chat. The model may need a longer "
                    f"timeout for cold starts or large generations."
                ) from e
            except (requests.ConnectionError, requests.exceptions.ConnectTimeout):
                raise

            if resp.status_code >= 400:
                snippet = (resp.text or "")[:200].replace("\n", " ")
                raise LLMClientError(
                    f"Ollama HTTP {resp.status_code} streaming /api/chat: {snippet}"
                )

            return _consume_ollama_stream(resp, on_token, self.config.model, started)

        return _with_retries(
            do,
            self.config.max_retries,
            (requests.ConnectionError, requests.exceptions.ConnectTimeout),
        )

    def chat_stream_no_logprobs(
        self,
        messages: "list[ChatMessage]",
        *,
        on_token: Callable[[dict], None],
        **opts: Any,
    ) -> "ChatResponse":
        """Stream a chat response WITHOUT requesting logprobs.

        Saves the ~150ms per-call overhead Ollama adds when ``logprobs=True``.
        Same on_token contract as chat_stream.
        """
        from autodidact.llm_client import LLMClientError

        options = self._options(opts)
        options.setdefault("num_predict", options.get("max_tokens", 1024))
        opts.pop("top_logprobs", None)  # explicitly drop if passed in by mistake
        think = opts.pop("think", None)

        body: dict[str, Any] = {
            "model": self.config.model,
            "messages": [asdict(m) for m in messages],
            "stream": True,
            "options": options,
        }
        if think is not None:
            body["think"] = bool(think)

        url = f"{self._host}/api/chat"
        started = time.perf_counter()

        def do() -> "ChatResponse":
            try:
                resp = requests.post(
                    url, json=body, stream=True,
                    timeout=self.config.timeout_seconds,
                )
            except requests.exceptions.ReadTimeout as e:
                raise LLMClientError(
                    f"Ollama read timeout after {self.config.timeout_seconds}s "
                    f"during streaming /api/chat. The model may need a longer "
                    f"timeout for cold starts or large generations."
                ) from e
            except (requests.ConnectionError, requests.exceptions.ConnectTimeout):
                raise

            if resp.status_code >= 400:
                snippet = (resp.text or "")[:200].replace("\n", " ")
                raise LLMClientError(
                    f"Ollama HTTP {resp.status_code} streaming /api/chat: {snippet}"
                )

            return _consume_ollama_stream_plain(
                resp, on_token, self.config.model, started,
            )

        return _with_retries(
            do,
            self.config.max_retries,
            (requests.ConnectionError, requests.exceptions.ConnectTimeout),
        )

    def embed(self, text: str) -> np.ndarray:
        from autodidact.llm_client import LLMClientError

        model = self.config.embedding_model or self.config.model
        body = {"model": model, "prompt": text}
        data = self._post("/api/embeddings", body)
        emb = data.get("embedding")
        if not isinstance(emb, list) or not emb:
            raise LLMClientError("Ollama embeddings endpoint returned empty embedding")
        return np.asarray(emb, dtype=np.float32)

    # ── Internals ────────────────────────────────────────────────

    def _post(self, path: str, body: dict) -> dict:
        from autodidact.llm_client import LLMClientError

        url = f"{self._host}{path}"

        def do() -> dict:
            try:
                resp = requests.post(url, json=body, timeout=self.config.timeout_seconds)
            except requests.exceptions.ReadTimeout as e:
                raise LLMClientError(
                    f"Ollama read timeout after {self.config.timeout_seconds}s "
                    f"at {path}. The model may need a longer timeout for cold "
                    f"starts or large generations."
                ) from e
            except (requests.ConnectionError, requests.exceptions.ConnectTimeout):
                raise

            if resp.status_code >= 400:
                snippet = resp.text[:200].replace("\n", " ")
                raise LLMClientError(f"Ollama HTTP {resp.status_code} at {path}: {snippet}")
            return resp.json()

        return _with_retries(
            do,
            self.config.max_retries,
            (requests.ConnectionError, requests.exceptions.ConnectTimeout),
        )

    def _options(self, opts: dict) -> dict:
        """Translate generic options to Ollama option names."""
        out: dict[str, Any] = {}
        if "temperature" in opts:
            out["temperature"] = float(opts["temperature"])
        if "max_tokens" in opts:
            out["num_predict"] = int(opts["max_tokens"])
        if "top_p" in opts:
            out["top_p"] = float(opts["top_p"])
        if "seed" in opts:
            out["seed"] = int(opts["seed"])
        return out


__all__ = ["OllamaBackend"]

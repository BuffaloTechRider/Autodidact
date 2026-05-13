"""Unified LLM client for Autodidact.

Supports two backends:
- Ollama (local HTTP, used as the local model; has logprob support)
- AWS Bedrock via boto3 (used as the cloud model and judge; no logprob support)

The client exposes three primary operations:
- chat: plain completion
- chat_with_logprobs: completion with per-token logprobs when the backend supports it
- embed: embedding for retrieval

Requirement 1 (R1) from the v0.1 spec is satisfied here.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable, Literal, Optional, TypeVar

import numpy as np
import requests
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# ── Data models ────────────────────────────────────────────────────
#
# Convention (see autodidact/types.py):
#   - Pydantic BaseModel: external boundary — configs, persisted state, anything
#     parsed from HTTP/SQLite or handed to users.
#   - dataclass: internal value types that live entirely in-process.

class LLMConfig(BaseModel):
    """Configuration for an LLMClient instance. Pydantic — boundary type."""

    provider: Literal["ollama", "openai", "bedrock"]
    model: str
    embedding_model: Optional[str] = None  # only meaningful for Ollama or OpenAI-compatible with embeddings
    base_url: Optional[str] = None          # required for 'openai' provider (points at vLLM, LM Studio, OpenAI, etc.)
    api_key_env: Optional[str] = None       # env var name for API key, e.g. "OPENAI_API_KEY"; None = no auth
    region: str = "us-west-2"               # only meaningful for Bedrock
    # Bedrock auth. "default" uses the boto3 default credential chain (env
    # vars, ~/.aws/credentials, SSO, IMDS, etc. — what existing users had
    # before these fields were added). "iam_user" passes explicit access
    # key + secret to boto3. "api_key" uses a short-lived Bedrock API key
    # via bearer-token auth (added by AWS in 2025).
    bedrock_auth_mode: Literal["default", "iam_user", "api_key"] = "default"
    bedrock_access_key_id: Optional[str] = None
    bedrock_secret_access_key: Optional[str] = None
    bedrock_session_token: Optional[str] = None
    bedrock_api_key: Optional[str] = None
    timeout_seconds: int = 60
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
    """Chat response extended with per-token logprobs.

    When the backend does not support logprobs (e.g. Bedrock), the logprob
    fields are returned empty / None rather than raising. This keeps the
    confidence evaluator working with any backend.
    """

    logprobs: list[float] = []
    avg_logprob: Optional[float] = None
    top_logprobs_by_position: list[dict[str, float]] = []


# ── Exceptions ─────────────────────────────────────────────────────

class LLMClientError(Exception):
    """Raised for unrecoverable LLM client errors.

    Messages never include credential values.
    """


class _BedrockThrottleError(Exception):
    """Internal marker for throttle-class Bedrock errors.

    Not exported. Used to smuggle retryable throttle responses through
    `_with_retries` without asking callers to import botocore exception types.
    """


# ── Retry helper ───────────────────────────────────────────────────

T = TypeVar("T")

# Exponential backoff. First entry used for attempt 1 → 2, second for 2 → 3, etc.
# Extended vs the old (0.5, 1, 2) so that Bedrock throttle bursts (observed in
# EXP-003 as 69 consecutive failures across ~12 seconds) have a chance to clear.
_BACKOFF_SECONDS = (1.0, 2.0, 4.0, 8.0, 16.0)


# ── Streaming helper for Ollama /api/chat ─────────────────────────


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
        # Last-ditch fallback: if no content was streamed, use thinking as
        # the visible answer. Mirrors _extract_answer's fallback.
        full_content = "".join(thinking_buf).strip()

    # Logprobs from the final chunk (top-level on 0.12.11+, message-level on older).
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
    )


def _with_retries(fn: Callable[[], T], max_retries: int, on_transient: tuple[type, ...]) -> T:
    """Run fn with exponential backoff on transient failures.

    Retries on exceptions listed in on_transient only. Non-listed exceptions
    (including HTTP 4xx raised as LLMClientError) propagate immediately.
    """
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
    raise LLMClientError(f"Transient failure after {attempts} attempts: {type(last_err).__name__}") from last_err


# ── LLMClient ──────────────────────────────────────────────────────

class LLMClient:
    """Unified client over Ollama and AWS Bedrock backends."""

    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self._ollama_host = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
        self._bedrock_client: Optional[Any] = None  # lazily created

    # ── Public API ────────────────────────────────────────────────

    def chat(self, messages: list[ChatMessage], **opts: Any) -> ChatResponse:
        if self.config.provider == "ollama":
            return self._chat_ollama(messages, **opts)
        if self.config.provider == "openai":
            return self._chat_openai(messages, **opts)
        return self._chat_bedrock(messages, **opts)

    def chat_with_logprobs(self, messages: list[ChatMessage], **opts: Any) -> ChatResponseWithLogprobs:
        if self.config.provider == "ollama":
            return self._chat_ollama_with_logprobs(messages, **opts)
        if self.config.provider == "openai":
            return self._chat_openai_with_logprobs(messages, **opts)
        return self._chat_bedrock_with_logprobs(messages, **opts)

    def embed(self, text: str) -> np.ndarray:
        if self.config.provider == "ollama":
            return self._embed_ollama(text)
        if self.config.provider == "openai":
            return self._embed_openai(text)
        raise LLMClientError(
            "Embeddings on the Bedrock provider are not supported in v0.1. "
            "Configure a separate Ollama or OpenAI-compatible LLMClient for embeddings."
        )

    # ── Ollama backend ────────────────────────────────────────────

    def _ollama_post(self, path: str, body: dict) -> dict:
        url = f"{self._ollama_host}{path}"

        def do() -> dict:
            try:
                resp = requests.post(url, json=body, timeout=self.config.timeout_seconds)
            except (requests.ConnectionError, requests.Timeout):
                raise
            if resp.status_code >= 400:
                # 4xx / 5xx: don't retry 4xx; for 5xx we raise LLMClientError too (we could retry but keep it simple)
                # Strip anything that might contain creds from the error surface.
                snippet = resp.text[:200].replace("\n", " ")
                raise LLMClientError(f"Ollama HTTP {resp.status_code} at {path}: {snippet}")
            return resp.json()

        return _with_retries(do, self.config.max_retries, (requests.ConnectionError, requests.Timeout))

    def _chat_ollama(self, messages: list[ChatMessage], **opts: Any) -> ChatResponse:
        body = {
            "model": self.config.model,
            "messages": [asdict(m) for m in messages],
            "stream": False,
            "options": self._ollama_options(opts),
        }
        started = time.perf_counter()
        data = self._ollama_post("/api/chat", body)
        latency_ms = int((time.perf_counter() - started) * 1000)
        content = (data.get("message") or {}).get("content", "")
        return ChatResponse(
            content=content,
            model=data.get("model", self.config.model),
            input_tokens=int(data.get("prompt_eval_count", 0) or 0),
            output_tokens=int(data.get("eval_count", 0) or 0),
            latency_ms=latency_ms,
        )

    def _chat_ollama_with_logprobs(
        self, messages: list[ChatMessage], **opts: Any
    ) -> ChatResponseWithLogprobs:
        options = self._ollama_options(opts)
        options.setdefault("num_predict", options.get("max_tokens", 256))
        top_logprobs_k = int(opts.pop("top_logprobs", 5))
        # Ollama 0.12.11+ exposes logprobs via top-level fields on the /api/chat body,
        # not inside "options". Older versions silently ignore these and we degrade
        # gracefully to empty logprobs.
        body = {
            "model": self.config.model,
            "messages": [asdict(m) for m in messages],
            "stream": False,
            "logprobs": True,
            "top_logprobs": top_logprobs_k,
            "options": options,
        }
        started = time.perf_counter()
        data = self._ollama_post("/api/chat", body)
        latency_ms = int((time.perf_counter() - started) * 1000)

        message = data.get("message") or {}
        content = message.get("content", "")

        # Parse logprobs. Ollama 0.12.11+ returns them at the TOP level of the
        # response body as:
        #   data["logprobs"] = [{"token": "...", "logprob": -0.1,
        #                         "top_logprobs": [{"token": "...", "logprob": -0.1}, ...]},
        #                       ...]
        # Older Ollama versions put it inside message.logprobs (float list or dict list).
        # Handle all shapes for forward compatibility.
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
                        top_lps.append(
                            {
                                str(t.get("token", "")): float(t.get("logprob", 0.0))
                                for t in top
                                if isinstance(t, dict) and "token" in t
                            }
                        )
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
        )

    def chat_stream_ollama(
        self,
        messages: list[ChatMessage],
        *,
        on_token: Callable[[dict], None],
        **opts: Any,
    ) -> ChatResponseWithLogprobs:
        """Stream a chat response from Ollama, calling on_token per chunk.

        Each ``on_token`` invocation receives ``{"phase": "content" | "thinking",
        "text": "..."}``. This lets the caller render content directly to the
        user while showing thinking dim/separately.

        Returns a fully accumulated ``ChatResponseWithLogprobs`` after the
        stream ends — the final chunk carries token counts and (on Ollama
        0.12.11+) logprobs.

        Retry policy mirrors the non-streaming path: ConnectionError /
        ConnectTimeout retry, ReadTimeout fails fast.
        """
        options = self._ollama_options(opts)
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

        url = f"{self._ollama_host}/api/chat"
        started = time.perf_counter()

        def do() -> ChatResponseWithLogprobs:
            try:
                resp = requests.post(
                    url,
                    json=body,
                    stream=True,
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
                raise LLMClientError(f"Ollama HTTP {resp.status_code} streaming /api/chat: {snippet}")

            return _consume_ollama_stream(resp, on_token, self.config.model, started)

        return _with_retries(
            do,
            self.config.max_retries,
            (requests.ConnectionError, requests.exceptions.ConnectTimeout),
        )

    def _embed_ollama(self, text: str) -> np.ndarray:
        model = self.config.embedding_model or self.config.model
        body = {"model": model, "prompt": text}
        data = self._ollama_post("/api/embeddings", body)
        emb = data.get("embedding")
        if not isinstance(emb, list) or not emb:
            raise LLMClientError("Ollama embeddings endpoint returned empty embedding")
        return np.asarray(emb, dtype=np.float32)

    def _ollama_options(self, opts: dict) -> dict:
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

    # ── OpenAI-compatible backend ─────────────────────────────────
    #
    # Works with any server speaking the OpenAI chat-completions API:
    # OpenAI itself, vLLM, LM Studio, llama.cpp server, text-generation-inference,
    # together.ai, Anyscale, Groq, Fireworks, etc.

    def _get_openai_client(self) -> Any:
        if not self.config.base_url:
            raise LLMClientError(
                "OpenAI-compatible provider requires base_url in LLMConfig "
                "(e.g. http://localhost:8000/v1 for vLLM, https://api.openai.com/v1 for OpenAI)."
            )
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as e:
            raise LLMClientError("openai package is required for the 'openai' provider.") from e
        api_key = "sk-no-auth"
        if self.config.api_key_env:
            api_key = os.environ.get(self.config.api_key_env) or api_key
        return OpenAI(api_key=api_key, base_url=self.config.base_url, timeout=self.config.timeout_seconds)

    def _openai_transient_exceptions(self) -> tuple[type, ...]:
        try:
            import openai  # type: ignore
            return (openai.APIConnectionError, openai.APITimeoutError)
        except ImportError:
            return ()

    def _chat_openai(self, messages: list[ChatMessage], **opts: Any) -> ChatResponse:
        client = self._get_openai_client()
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

        def do() -> ChatResponse:
            started = time.perf_counter()
            try:
                resp = client.chat.completions.create(**kwargs)
            except Exception as e:
                # openai library raises AuthenticationError, BadRequestError, etc. for 4xx.
                self._maybe_raise_4xx("openai", e)
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

        return _with_retries(do, self.config.max_retries, self._openai_transient_exceptions())

    def _chat_openai_with_logprobs(
        self, messages: list[ChatMessage], **opts: Any
    ) -> ChatResponseWithLogprobs:
        client = self._get_openai_client()
        top_logprobs_k = int(opts.pop("top_logprobs", 5))
        kwargs: dict[str, Any] = {
            "model": self.config.model,
            "messages": [asdict(m) for m in messages],
            "logprobs": True,
            "top_logprobs": top_logprobs_k,
        }
        if "max_tokens" in opts:
            kwargs["max_tokens"] = int(opts["max_tokens"])
        if "temperature" in opts:
            kwargs["temperature"] = float(opts["temperature"])
        if "top_p" in opts:
            kwargs["top_p"] = float(opts["top_p"])
        if "seed" in opts:
            kwargs["seed"] = int(opts["seed"])

        def do() -> ChatResponseWithLogprobs:
            started = time.perf_counter()
            try:
                resp = client.chat.completions.create(**kwargs)
            except Exception as e:
                self._maybe_raise_4xx("openai", e)
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

        return _with_retries(do, self.config.max_retries, self._openai_transient_exceptions())

    def chat_stream_openai(
        self,
        messages: list[ChatMessage],
        *,
        on_token: Callable[[dict], None],
        **opts: Any,
    ) -> ChatResponse:
        """Stream a chat response from an OpenAI-compatible backend.

        Calls ``on_token`` per content chunk with ``{"phase": "content",
        "text": "..."}``. OpenAI's chat-completions stream doesn't carry a
        separate thinking field, so all chunks are content-phase.

        Returns the accumulated ``ChatResponse``. Token counts come from the
        final stream chunk's ``usage`` field (set when ``stream_options=
        {"include_usage": True}``).
        """
        client = self._get_openai_client()
        kwargs: dict[str, Any] = {
            "model": self.config.model,
            "messages": [asdict(m) for m in messages],
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if "max_tokens" in opts:
            kwargs["max_tokens"] = int(opts["max_tokens"])
        if "temperature" in opts:
            kwargs["temperature"] = float(opts["temperature"])
        if "top_p" in opts:
            kwargs["top_p"] = float(opts["top_p"])
        if "seed" in opts:
            kwargs["seed"] = int(opts["seed"])

        def do() -> ChatResponse:
            started = time.perf_counter()
            content_buf: list[str] = []
            input_tokens = 0
            output_tokens = 0
            model = self.config.model

            try:
                stream = client.chat.completions.create(**kwargs)
            except Exception as e:
                self._maybe_raise_4xx("openai", e)
                raise

            for chunk in stream:
                # Some chunks carry usage info; others carry content deltas.
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

        return _with_retries(do, self.config.max_retries, self._openai_transient_exceptions())

    def _embed_openai(self, text: str) -> np.ndarray:
        client = self._get_openai_client()
        model = self.config.embedding_model or "text-embedding-3-small"

        def do() -> np.ndarray:
            try:
                resp = client.embeddings.create(model=model, input=text)
            except Exception as e:
                self._maybe_raise_4xx("openai", e)
                raise
            vec = resp.data[0].embedding
            return np.asarray(vec, dtype=np.float32)

        return _with_retries(do, self.config.max_retries, self._openai_transient_exceptions())

    def _maybe_raise_4xx(self, provider: str, exc: Exception) -> None:
        """Convert provider 4xx errors to non-retryable LLMClientError."""
        try:
            import openai  # type: ignore
        except ImportError:
            return
        if isinstance(exc, (openai.AuthenticationError, openai.PermissionDeniedError)):
            raise LLMClientError(f"{provider} rejected credentials (4xx)")
        if isinstance(exc, openai.BadRequestError):
            raise LLMClientError(f"{provider} rejected request (400): {str(exc)[:200]}")
        if isinstance(exc, openai.NotFoundError):
            raise LLMClientError(f"{provider} model not found (404): {self.config.model}")

    # ── Bedrock backend ───────────────────────────────────────────

    def _get_bedrock_client(self) -> Any:
        if self._bedrock_client is not None:
            return self._bedrock_client
        try:
            import boto3  # type: ignore
            from botocore.config import Config as BotoConfig  # type: ignore
        except ImportError as e:
            raise LLMClientError(
                "boto3 is required for the Bedrock provider. Install with `pip install autodidact[bedrock]`."
            ) from e
        boto_config = BotoConfig(
            read_timeout=self.config.timeout_seconds,
            connect_timeout=self.config.timeout_seconds,
            retries={"max_attempts": 1, "mode": "standard"},  # we handle retries ourselves
        )

        # Choose credentials based on bedrock_auth_mode.
        client_kwargs: dict = {
            "service_name": "bedrock-runtime",
            "region_name": self.config.region,
            "config": boto_config,
        }
        mode = self.config.bedrock_auth_mode
        if mode == "iam_user":
            # Explicit credentials — bypass the default credential chain.
            if not (self.config.bedrock_access_key_id and self.config.bedrock_secret_access_key):
                raise LLMClientError(
                    "bedrock_auth_mode='iam_user' requires bedrock_access_key_id "
                    "and bedrock_secret_access_key in the config."
                )
            client_kwargs["aws_access_key_id"] = self.config.bedrock_access_key_id
            client_kwargs["aws_secret_access_key"] = self.config.bedrock_secret_access_key
            if self.config.bedrock_session_token:
                client_kwargs["aws_session_token"] = self.config.bedrock_session_token
        elif mode == "api_key":
            # Bedrock API key mode: AWS sets the AWS_BEARER_TOKEN_BEDROCK env
            # var and boto3's credential provider picks it up automatically.
            # See https://docs.aws.amazon.com/bedrock/latest/userguide/api-keys.html
            if not self.config.bedrock_api_key:
                raise LLMClientError(
                    "bedrock_auth_mode='api_key' requires bedrock_api_key in the config."
                )
            os.environ["AWS_BEARER_TOKEN_BEDROCK"] = self.config.bedrock_api_key
        # "default" mode: no-op — boto3 uses its standard credential chain
        # (env vars, ~/.aws/credentials, SSO, IMDS, role assumption, etc.).

        self._bedrock_client = boto3.client(**client_kwargs)
        return self._bedrock_client

    def _bedrock_transient_exceptions(self) -> tuple[type, ...]:
        try:
            from botocore.exceptions import (  # type: ignore
                ConnectionError as BotoConnectionError,
                EndpointConnectionError,
                ReadTimeoutError,
            )
            return (BotoConnectionError, EndpointConnectionError, ReadTimeoutError)
        except ImportError:
            return ()

    @staticmethod
    def _is_bedrock_throttle(exc: Exception) -> bool:
        """Detect Bedrock throttling / transient server errors from a ClientError.

        Bedrock returns these as `botocore.exceptions.ClientError` with an Error
        Code field. The codes that are safe to retry include:
          - ThrottlingException: request rate too high
          - TooManyRequestsException: alt name for throttling
          - ServiceUnavailableException: "Too many connections, please wait"
            (observed during EXP-003 — 69 failures in a burst)
          - ModelErrorException (some retries): model backend momentary failure
          - InternalServerException: 5xx
        """
        err = getattr(exc, "response", None)
        if not isinstance(err, dict):
            return False
        code = ((err.get("Error") or {}).get("Code"))
        return code in {
            "ThrottlingException",
            "TooManyRequestsException",
            "ServiceUnavailableException",
            "ModelErrorException",
            "InternalServerException",
        }

    def _chat_bedrock(self, messages: list[ChatMessage], **opts: Any) -> ChatResponse:
        client = self._get_bedrock_client()
        system, converse_messages = self._to_bedrock_messages(messages)
        inference_config: dict[str, Any] = {}
        if "max_tokens" in opts:
            inference_config["maxTokens"] = int(opts["max_tokens"])
        if "temperature" in opts:
            inference_config["temperature"] = float(opts["temperature"])
        if "top_p" in opts:
            inference_config["topP"] = float(opts["top_p"])

        kwargs = {"modelId": self.config.model, "messages": converse_messages}
        if system:
            kwargs["system"] = system
        if inference_config:
            kwargs["inferenceConfig"] = inference_config

        def do() -> ChatResponse:
            started = time.perf_counter()
            try:
                resp = client.converse(**kwargs)
            except Exception as e:
                # Non-retryable auth / validation errors: re-wrap, don't retry.
                code = getattr(getattr(e, "response", {}).get("Error", {}), "get", lambda _k: None)("Code")
                if code in {"AccessDeniedException", "UnauthorizedException", "ValidationException"}:
                    raise LLMClientError(f"Bedrock rejected request: {code}") from e
                # Throttle-class errors: re-raise so _with_retries picks them up.
                if self._is_bedrock_throttle(e):
                    raise _BedrockThrottleError(f"Bedrock throttle ({code})") from e
                raise
            latency_ms = int((time.perf_counter() - started) * 1000)
            content_parts = ((resp.get("output") or {}).get("message") or {}).get("content", [])
            text = "".join(part.get("text", "") for part in content_parts if isinstance(part, dict))
            usage = resp.get("usage") or {}
            return ChatResponse(
                content=text,
                model=self.config.model,
                input_tokens=int(usage.get("inputTokens", 0) or 0),
                output_tokens=int(usage.get("outputTokens", 0) or 0),
                latency_ms=latency_ms,
            )

        transient = self._bedrock_transient_exceptions() + (_BedrockThrottleError,)
        return _with_retries(do, self.config.max_retries, transient)

    def _chat_bedrock_with_logprobs(
        self, messages: list[ChatMessage], **opts: Any
    ) -> ChatResponseWithLogprobs:
        # Bedrock Converse API does not expose logprobs; degrade gracefully.
        base = self._chat_bedrock(messages, **opts)
        return ChatResponseWithLogprobs(
            content=base.content,
            model=base.model,
            input_tokens=base.input_tokens,
            output_tokens=base.output_tokens,
            latency_ms=base.latency_ms,
            logprobs=[],
            avg_logprob=None,
            top_logprobs_by_position=[],
        )

    def chat_stream_bedrock(
        self,
        messages: list[ChatMessage],
        *,
        on_token: Callable[[dict], None],
        **opts: Any,
    ) -> ChatResponse:
        """Stream a chat response from Bedrock via converse_stream.

        Calls ``on_token`` per chunk with phase ``content`` for normal output
        and phase ``thinking`` for ``reasoningContent`` blocks (Anthropic
        extended thinking).

        Throttling / validation errors are mapped to LLMClientError consistent
        with the non-streaming path.
        """
        client = self._get_bedrock_client()
        system, converse_messages = self._to_bedrock_messages(messages)
        inference_config: dict[str, Any] = {}
        if "max_tokens" in opts:
            inference_config["maxTokens"] = int(opts["max_tokens"])
        if "temperature" in opts:
            inference_config["temperature"] = float(opts["temperature"])
        if "top_p" in opts:
            inference_config["topP"] = float(opts["top_p"])

        kwargs: dict[str, Any] = {
            "modelId": self.config.model,
            "messages": converse_messages,
        }
        if system:
            kwargs["system"] = system
        if inference_config:
            kwargs["inferenceConfig"] = inference_config

        def do() -> ChatResponse:
            started = time.perf_counter()
            content_buf: list[str] = []
            thinking_buf: list[str] = []
            input_tokens = 0
            output_tokens = 0

            try:
                response = client.converse_stream(**kwargs)
            except Exception as e:
                code = None
                resp_dict = getattr(e, "response", None)
                if isinstance(resp_dict, dict):
                    code = (resp_dict.get("Error") or {}).get("Code")
                if code in {"AccessDeniedException", "UnauthorizedException", "ValidationException"}:
                    raise LLMClientError(f"Bedrock rejected request: {code}") from e
                if self._is_bedrock_throttle(e):
                    raise _BedrockThrottleError(f"Bedrock throttle ({code})") from e
                raise

            stream = response.get("stream") if isinstance(response, dict) else None
            if stream is None:
                # Some boto wrappers expose the iterator directly.
                stream = response  # type: ignore[assignment]

            for event in stream:
                if not isinstance(event, dict):
                    continue

                # Content / thinking deltas.
                delta_block = (event.get("contentBlockDelta") or {}).get("delta") or {}
                text_delta = delta_block.get("text")
                if text_delta:
                    content_buf.append(text_delta)
                    on_token({"phase": "content", "text": text_delta})

                reasoning_text = (delta_block.get("reasoningContent") or {}).get("text")
                if reasoning_text:
                    thinking_buf.append(reasoning_text)
                    on_token({"phase": "thinking", "text": reasoning_text})

                # Final usage metadata block.
                metadata = event.get("metadata") or {}
                usage = metadata.get("usage") or {}
                if usage:
                    input_tokens = int(usage.get("inputTokens", input_tokens) or input_tokens)
                    output_tokens = int(usage.get("outputTokens", output_tokens) or output_tokens)

            latency_ms = int((time.perf_counter() - started) * 1000)
            content = "".join(content_buf)
            if not content.strip():
                # Fall back to thinking if content was empty (mirrors _extract_answer).
                content = "".join(thinking_buf).strip()

            return ChatResponse(
                content=content,
                model=self.config.model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency_ms=latency_ms,
            )

        transient = self._bedrock_transient_exceptions() + (_BedrockThrottleError,)
        return _with_retries(do, self.config.max_retries, transient)

    def chat_stream(
        self,
        messages: list[ChatMessage],
        *,
        on_token: Callable[[dict], None],
        **opts: Any,
    ) -> ChatResponse:
        """Provider-agnostic streaming chat. Routes by ``self.config.provider``.

        Returns a ``ChatResponse`` (not ChatResponseWithLogprobs — streaming
        cloud paths don't expose logprobs in v1.0; the local path returns
        the richer type via ``chat_stream_ollama`` directly).
        """
        if self.config.provider == "ollama":
            return self.chat_stream_ollama(messages, on_token=on_token, **opts)
        if self.config.provider == "openai":
            return self.chat_stream_openai(messages, on_token=on_token, **opts)
        return self.chat_stream_bedrock(messages, on_token=on_token, **opts)

    def _to_bedrock_messages(
        self, messages: list[ChatMessage]
    ) -> tuple[list[dict], list[dict]]:
        """Split into Bedrock Converse API (system, user+assistant) format."""
        system_blocks: list[dict] = []
        converse_messages: list[dict] = []
        for m in messages:
            if m.role == "system":
                system_blocks.append({"text": m.content})
            else:
                converse_messages.append({"role": m.role, "content": [{"text": m.content}]})
        return system_blocks, converse_messages

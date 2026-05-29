"""AWS Bedrock backend (boto3 / Converse API).

Implements ChatBackend over Bedrock Converse and converse_stream. Embeddings
aren't supported in v0.1 — call embed() and you'll get LLMClientError.

Holds a lazily-built boto3 client. Auth modes: default (boto3 credential
chain), iam_user (explicit access keys), api_key (short-lived bearer
token via AWS_BEARER_TOKEN_BEDROCK).
"""

from __future__ import annotations

import os
import time
from typing import Any, Callable, TYPE_CHECKING

import numpy as np

from autodidact.llm.backend import _BedrockThrottleError, _with_retries

if TYPE_CHECKING:
    from autodidact.llm_client import (
        ChatMessage,
        ChatResponse,
        ChatResponseWithLogprobs,
        LLMConfig,
    )


class BedrockBackend:
    """ChatBackend implementation for AWS Bedrock Converse API."""

    def __init__(self, config: "LLMConfig") -> None:
        self.config = config
        self._client: Any = None  # lazily created

    # ── ChatBackend interface ────────────────────────────────────

    def chat(self, messages: "list[ChatMessage]", **opts: Any) -> "ChatResponse":
        from autodidact.llm_client import ChatResponse, LLMClientError

        client = self._get_client()
        system, converse_messages = self._to_messages(messages)
        inference_config: dict[str, Any] = {}
        if "max_tokens" in opts:
            inference_config["maxTokens"] = int(opts["max_tokens"])
        if "top_p" in opts:
            inference_config["topP"] = float(opts["top_p"])

        kwargs = {"modelId": self.config.model, "messages": converse_messages}
        if system:
            kwargs["system"] = system
        if inference_config:
            kwargs["inferenceConfig"] = inference_config

        def do() -> "ChatResponse":
            started = time.perf_counter()
            try:
                resp = client.converse(**kwargs)
            except Exception as e:
                code = getattr(
                    getattr(e, "response", {}).get("Error", {}), "get",
                    lambda _k: None,
                )("Code")
                if code in {
                    "AccessDeniedException",
                    "UnauthorizedException",
                    "ValidationException",
                }:
                    raise LLMClientError(f"Bedrock rejected request: {code}") from e
                if self._is_throttle(e):
                    raise _BedrockThrottleError(f"Bedrock throttle ({code})") from e
                raise
            latency_ms = int((time.perf_counter() - started) * 1000)
            content_parts = (
                ((resp.get("output") or {}).get("message") or {}).get("content", [])
            )
            text = "".join(
                part.get("text", "") for part in content_parts if isinstance(part, dict)
            )
            usage = resp.get("usage") or {}
            return ChatResponse(
                content=text,
                model=self.config.model,
                input_tokens=int(usage.get("inputTokens", 0) or 0),
                output_tokens=int(usage.get("outputTokens", 0) or 0),
                latency_ms=latency_ms,
            )

        transient = self._transient_exceptions() + (_BedrockThrottleError,)
        return _with_retries(do, self.config.max_retries, transient)

    def chat_with_logprobs(
        self, messages: "list[ChatMessage]", **opts: Any
    ) -> "ChatResponseWithLogprobs":
        """Bedrock Converse API doesn't expose logprobs; degrade gracefully."""
        from autodidact.llm_client import ChatResponseWithLogprobs

        base = self.chat(messages, **opts)
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

    def chat_stream(
        self,
        messages: "list[ChatMessage]",
        *,
        on_token: Callable[[dict], None],
        **opts: Any,
    ) -> "ChatResponse":
        from autodidact.llm_client import ChatResponse, LLMClientError

        client = self._get_client()
        system, converse_messages = self._to_messages(messages)
        inference_config: dict[str, Any] = {}
        if "max_tokens" in opts:
            inference_config["maxTokens"] = int(opts["max_tokens"])
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

        def do() -> "ChatResponse":
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
                if code in {
                    "AccessDeniedException",
                    "UnauthorizedException",
                    "ValidationException",
                }:
                    raise LLMClientError(f"Bedrock rejected request: {code}") from e
                if self._is_throttle(e):
                    raise _BedrockThrottleError(f"Bedrock throttle ({code})") from e
                raise

            stream = response.get("stream") if isinstance(response, dict) else None
            if stream is None:
                stream = response  # type: ignore[assignment]

            for event in stream:
                if not isinstance(event, dict):
                    continue
                delta_block = (event.get("contentBlockDelta") or {}).get("delta") or {}
                text_delta = delta_block.get("text")
                if text_delta:
                    content_buf.append(text_delta)
                    on_token({"phase": "content", "text": text_delta})
                reasoning_text = (delta_block.get("reasoningContent") or {}).get("text")
                if reasoning_text:
                    thinking_buf.append(reasoning_text)
                    on_token({"phase": "thinking", "text": reasoning_text})

                metadata = event.get("metadata") or {}
                usage = metadata.get("usage") or {}
                if usage:
                    input_tokens = int(
                        usage.get("inputTokens", input_tokens) or input_tokens
                    )
                    output_tokens = int(
                        usage.get("outputTokens", output_tokens) or output_tokens
                    )

            latency_ms = int((time.perf_counter() - started) * 1000)
            content = "".join(content_buf)
            if not content.strip():
                content = "".join(thinking_buf).strip()

            return ChatResponse(
                content=content,
                model=self.config.model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency_ms=latency_ms,
            )

        transient = self._transient_exceptions() + (_BedrockThrottleError,)
        return _with_retries(do, self.config.max_retries, transient)

    def embed(self, text: str) -> np.ndarray:
        from autodidact.llm_client import LLMClientError

        raise LLMClientError(
            "Embeddings on the Bedrock provider are not supported in v0.1. "
            "Configure a separate Ollama or OpenAI-compatible LLMClient for embeddings."
        )

    # ── Internals ────────────────────────────────────────────────

    def _get_client(self) -> Any:
        """Build the boto3 client lazily, honoring bedrock_auth_mode."""
        from autodidact.llm_client import LLMClientError

        if self._client is not None:
            return self._client
        try:
            import boto3  # type: ignore
            from botocore.config import Config as BotoConfig  # type: ignore
        except ImportError as e:
            raise LLMClientError(
                "boto3 is required for the Bedrock provider. "
                "Install with `pip install autodidact[bedrock]`."
            ) from e

        boto_config = BotoConfig(
            read_timeout=self.config.timeout_seconds,
            connect_timeout=self.config.timeout_seconds,
            retries={"max_attempts": 1, "mode": "standard"},
        )

        client_kwargs: dict = {
            "service_name": "bedrock-runtime",
            "region_name": self.config.region,
            "config": boto_config,
        }
        mode = self.config.bedrock_auth_mode
        if mode == "iam_user":
            if not (
                self.config.bedrock_access_key_id
                and self.config.bedrock_secret_access_key
            ):
                raise LLMClientError(
                    "bedrock_auth_mode='iam_user' requires bedrock_access_key_id "
                    "and bedrock_secret_access_key in the config."
                )
            client_kwargs["aws_access_key_id"] = self.config.bedrock_access_key_id
            client_kwargs["aws_secret_access_key"] = self.config.bedrock_secret_access_key
            if self.config.bedrock_session_token:
                client_kwargs["aws_session_token"] = self.config.bedrock_session_token
        elif mode == "api_key":
            if not self.config.bedrock_api_key:
                raise LLMClientError(
                    "bedrock_auth_mode='api_key' requires bedrock_api_key in the config."
                )
            os.environ["AWS_BEARER_TOKEN_BEDROCK"] = self.config.bedrock_api_key
        # default mode: no-op — boto3 uses its standard credential chain.

        self._client = boto3.client(**client_kwargs)
        return self._client

    def _transient_exceptions(self) -> tuple[type, ...]:
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
    def _is_throttle(exc: Exception) -> bool:
        """Detect Bedrock throttling / transient server errors from a ClientError.

        Bedrock returns these as `botocore.exceptions.ClientError` with an
        Error Code field. Codes safe to retry:
          - ThrottlingException, TooManyRequestsException
          - ServiceUnavailableException
          - ModelErrorException, InternalServerException
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

    def _to_messages(
        self, messages: "list[ChatMessage]",
    ) -> tuple[list[dict], list[dict]]:
        """Split into Bedrock Converse API (system, user+assistant) format."""
        system_blocks: list[dict] = []
        converse_messages: list[dict] = []
        for m in messages:
            if m.role == "system":
                system_blocks.append({"text": m.content})
            else:
                converse_messages.append(
                    {"role": m.role, "content": [{"text": m.content}]}
                )
        return system_blocks, converse_messages


__all__ = ["BedrockBackend"]

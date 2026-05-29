"""Tests for the ChatBackend extraction.

LLMClient is now a thin facade over a ChatBackend adapter. The big functional
coverage (chat / chat_with_logprobs / chat_stream / embed against each
provider) lives in:
  - tests/test_streaming.py            (Ollama streaming)
  - tests/test_streaming_cloud.py      (OpenAI / Bedrock streaming)
  - tests/test_skip_logprobs_in_chat.py (Ollama no-logprobs)
  - tests/test_skip_logprob_on_thinking.py (thinking-token detection)
  - tests/test_llm_client_retry.py     (Bedrock throttle retry)
  - tests/test_bedrock_auth.py         (Bedrock auth-mode wiring)

Those keep passing — they're the regression guard. This file exercises the
*new* pieces:
  - that LLMClient.__init__ selects the right backend by provider
  - that the facade methods forward to the backend
  - that the ChatBackend Protocol matches all three concrete adapters
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ── ChatBackend protocol ────────────────────────────────────────


class TestChatBackendProtocol:
    """All three concrete adapters satisfy the ChatBackend interface."""

    def test_ollama_backend_has_required_methods(self):
        from autodidact.llm import OllamaBackend
        from autodidact.llm_client import LLMConfig

        backend = OllamaBackend(LLMConfig(provider="ollama", model="qwen3:8b"))
        assert hasattr(backend, "chat")
        assert hasattr(backend, "chat_with_logprobs")
        assert hasattr(backend, "chat_stream")
        assert hasattr(backend, "chat_stream_no_logprobs")
        assert hasattr(backend, "embed")

    def test_openai_backend_has_required_methods(self):
        from autodidact.llm import OpenAICompatBackend
        from autodidact.llm_client import LLMConfig

        backend = OpenAICompatBackend(LLMConfig(
            provider="openai", model="gpt-4o", base_url="https://api.openai.com/v1",
        ))
        assert hasattr(backend, "chat")
        assert hasattr(backend, "chat_with_logprobs")
        assert hasattr(backend, "chat_stream")
        assert hasattr(backend, "embed")

    def test_bedrock_backend_has_required_methods(self):
        from autodidact.llm import BedrockBackend
        from autodidact.llm_client import LLMConfig

        backend = BedrockBackend(LLMConfig(provider="bedrock", model="anthropic.x"))
        assert hasattr(backend, "chat")
        assert hasattr(backend, "chat_with_logprobs")
        assert hasattr(backend, "chat_stream")
        # Bedrock doesn't support embeddings; the method exists but raises.
        assert hasattr(backend, "embed")


# ── LLMClient backend selection ─────────────────────────────────


class TestLLMClientSelectsBackend:
    def test_ollama_provider_selects_ollama_backend(self):
        from autodidact.llm import OllamaBackend
        from autodidact.llm_client import LLMClient, LLMConfig

        client = LLMClient(LLMConfig(provider="ollama", model="qwen3:8b"))
        assert isinstance(client._backend, OllamaBackend)

    def test_openai_provider_selects_openai_backend(self):
        from autodidact.llm import OpenAICompatBackend
        from autodidact.llm_client import LLMClient, LLMConfig

        client = LLMClient(LLMConfig(
            provider="openai", model="gpt-4o", base_url="https://api.openai.com/v1",
        ))
        assert isinstance(client._backend, OpenAICompatBackend)

    def test_bedrock_provider_selects_bedrock_backend(self):
        from autodidact.llm import BedrockBackend
        from autodidact.llm_client import LLMClient, LLMConfig

        client = LLMClient(LLMConfig(
            provider="bedrock", model="anthropic.claude-x", region="us-west-2",
        ))
        assert isinstance(client._backend, BedrockBackend)


# ── LLMClient facade forwards to backend ────────────────────────


class TestFacadeForwarding:
    """Each public method on LLMClient calls into the backend, not into
    duplicate per-provider logic."""

    def _client(self, provider: str = "ollama"):
        from autodidact.llm_client import LLMClient, LLMConfig
        if provider == "openai":
            cfg = LLMConfig(provider="openai", model="x", base_url="https://api.openai.com/v1")
        elif provider == "bedrock":
            cfg = LLMConfig(provider="bedrock", model="anthropic.x")
        else:
            cfg = LLMConfig(provider="ollama", model="qwen3:8b")
        return LLMClient(cfg)

    def test_chat_forwards_to_backend(self):
        client = self._client("ollama")
        client._backend = MagicMock()
        client._backend.chat.return_value = "RESULT"

        out = client.chat([], temperature=0.0)
        assert out == "RESULT"
        client._backend.chat.assert_called_once_with([], temperature=0.0)

    def test_chat_with_logprobs_forwards_to_backend(self):
        client = self._client("ollama")
        client._backend = MagicMock()
        client._backend.chat_with_logprobs.return_value = "RESULT"

        out = client.chat_with_logprobs([], top_logprobs=5)
        assert out == "RESULT"
        client._backend.chat_with_logprobs.assert_called_once_with([], top_logprobs=5)

    def test_embed_forwards_to_backend(self):
        client = self._client("ollama")
        client._backend = MagicMock()
        client._backend.embed.return_value = np.zeros(4)

        out = client.embed("hello")
        np.testing.assert_array_equal(out, np.zeros(4))
        client._backend.embed.assert_called_once_with("hello")

    def test_chat_stream_routes_by_provider(self):
        """chat_stream picks the right per-provider facade by config.provider.

        It does NOT bypass the chat_stream_ollama / _openai / _bedrock
        facades — those are public test patch points (see
        test_streaming_cloud.py::TestChatStreamDispatch).
        """
        client = self._client("ollama")
        with patch.object(
            client, "chat_stream_ollama", return_value="STREAMED",
        ) as mock:
            out = client.chat_stream([], on_token=lambda _: None, max_tokens=100)
        assert out == "STREAMED"
        mock.assert_called_once()


# ── Public chat_stream_* facades preserved (test fixtures use them) ─


class TestProviderSpecificFacades:
    """Tests that use spec=LLMClient or patch.object expect these names to exist
    AND to dispatch to the right backend.

    Constraint: tests in test_streaming_cloud.py and test_bedrock_auth.py do
        patch.object(client, "_get_bedrock_client", ...)
        patch.object(client, "_get_openai_client", ...)
    so those underscored helpers must continue to exist as instance methods
    on LLMClient.
    """

    def test_chat_stream_ollama_facade(self):
        from autodidact.llm import OllamaBackend
        from autodidact.llm_client import LLMClient, LLMConfig

        client = LLMClient(LLMConfig(provider="ollama", model="qwen3:8b"))
        with patch.object(client._backend, "chat_stream", return_value="OK") as mock:
            result = client.chat_stream_ollama(
                [],
                on_token=lambda _: None,
            )
        assert result == "OK"
        assert mock.called

    def test_chat_stream_ollama_no_logprobs_facade(self):
        from autodidact.llm import OllamaBackend
        from autodidact.llm_client import LLMClient, LLMConfig

        client = LLMClient(LLMConfig(provider="ollama", model="qwen3:8b"))
        with patch.object(
            client._backend, "chat_stream_no_logprobs", return_value="OK",
        ) as mock:
            result = client.chat_stream_ollama_no_logprobs(
                [],
                on_token=lambda _: None,
            )
        assert result == "OK"
        assert mock.called

    def test_chat_stream_openai_facade(self):
        from autodidact.llm_client import LLMClient, LLMConfig

        client = LLMClient(LLMConfig(
            provider="openai", model="gpt-4o", base_url="https://api.openai.com/v1",
        ))
        with patch.object(client._backend, "chat_stream", return_value="OK") as mock:
            result = client.chat_stream_openai(
                [],
                on_token=lambda _: None,
            )
        assert result == "OK"
        assert mock.called

    def test_chat_stream_bedrock_facade(self):
        from autodidact.llm_client import LLMClient, LLMConfig

        client = LLMClient(LLMConfig(provider="bedrock", model="anthropic.x"))
        with patch.object(client._backend, "chat_stream", return_value="OK") as mock:
            result = client.chat_stream_bedrock(
                [],
                on_token=lambda _: None,
            )
        assert result == "OK"
        assert mock.called

    def test_get_openai_client_facade_preserved(self):
        """patch.object(client, '_get_openai_client', ...) must keep working."""
        from autodidact.llm_client import LLMClient, LLMConfig

        client = LLMClient(LLMConfig(
            provider="openai", model="gpt-4o", base_url="https://api.openai.com/v1",
        ))
        # Fake the underlying get-client; the facade must return whatever it gets.
        sentinel = object()
        with patch.object(client._backend, "_get_client", return_value=sentinel):
            result = client._get_openai_client()
        assert result is sentinel

    def test_get_bedrock_client_facade_preserved(self):
        """patch.object(client, '_get_bedrock_client', ...) must keep working."""
        from autodidact.llm_client import LLMClient, LLMConfig

        client = LLMClient(LLMConfig(provider="bedrock", model="anthropic.x"))
        sentinel = object()
        with patch.object(client._backend, "_get_client", return_value=sentinel):
            result = client._get_bedrock_client()
        assert result is sentinel


# ── Wrong-provider facades raise cleanly ────────────────────────


class TestWrongProviderFacades:
    """If you call chat_stream_bedrock on an Ollama client, it should error."""

    def test_chat_stream_bedrock_on_ollama_raises(self):
        from autodidact.llm_client import LLMClient, LLMClientError, LLMConfig

        client = LLMClient(LLMConfig(provider="ollama", model="qwen3:8b"))
        with pytest.raises(LLMClientError, match="bedrock"):
            client.chat_stream_bedrock([], on_token=lambda _: None)

    def test_chat_stream_ollama_on_bedrock_raises(self):
        from autodidact.llm_client import LLMClient, LLMClientError, LLMConfig

        client = LLMClient(LLMConfig(provider="bedrock", model="anthropic.x"))
        with pytest.raises(LLMClientError, match="ollama"):
            client.chat_stream_ollama([], on_token=lambda _: None)

    def test_chat_stream_openai_on_bedrock_raises(self):
        from autodidact.llm_client import LLMClient, LLMClientError, LLMConfig

        client = LLMClient(LLMConfig(provider="bedrock", model="anthropic.x"))
        with pytest.raises(LLMClientError, match="openai"):
            client.chat_stream_openai([], on_token=lambda _: None)


# ── Unknown provider fails fast ─────────────────────────────────


class TestUnknownProvider:
    def test_invalid_provider_raises_at_construction(self):
        """LLMConfig validation rejects unknown providers via Literal type.

        We're verifying the safety net: even if someone bypasses validation,
        LLMClient.__init__ rejects the provider explicitly with a clear error.
        """
        from autodidact.llm_client import LLMClient, LLMClientError

        # Fake config with an unknown provider — bypass pydantic by using __new__.
        from autodidact.llm_client import LLMConfig
        cfg = LLMConfig.model_construct(provider="zoid", model="x")  # type: ignore[arg-type]
        with pytest.raises(LLMClientError, match="zoid|provider"):
            LLMClient(cfg)

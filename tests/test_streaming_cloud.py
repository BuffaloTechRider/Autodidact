"""Tests for streaming chat from cloud providers (OpenAI-compat, Bedrock).

The contract mirrors chat_stream_ollama:

  client.chat_stream_openai(messages, on_token=cb, **opts) -> ChatResponse
  client.chat_stream_bedrock(messages, on_token=cb, **opts) -> ChatResponse

Each chunk callback receives ``{"phase": "content" | "thinking", "text": "..."}``.
The aggregated ChatResponse is returned at the end.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from autodidact.llm_client import (
    ChatMessage,
    ChatResponse,
    LLMClient,
    LLMClientError,
    LLMConfig,
)


# ── OpenAI streaming ──────────────────────────────────────────────


@pytest.fixture
def openai_client():
    return LLMClient(LLMConfig(
        provider="openai",
        model="gpt-4o-mini",
        base_url="https://api.openai.com/v1",
        api_key_env="OPENAI_API_KEY",
        max_retries=2,
        timeout_seconds=5,
    ))


def _make_openai_chunk(content_delta: str = "", finish_reason=None, usage=None):
    """Build a mocked OpenAI streaming chunk object."""
    chunk = MagicMock()
    chunk.choices = [MagicMock()]
    chunk.choices[0].delta = MagicMock()
    chunk.choices[0].delta.content = content_delta or None
    chunk.choices[0].finish_reason = finish_reason
    chunk.usage = usage
    chunk.model = "gpt-4o-mini-2024-07-18"
    return chunk


class TestOpenAIStreaming:
    """Stream content tokens from OpenAI-compat chat completions."""

    def test_content_chunks_invoke_callback(self, openai_client):
        usage = MagicMock(prompt_tokens=10, completion_tokens=4)
        stream_chunks = [
            _make_openai_chunk("Hello "),
            _make_openai_chunk("world"),
            _make_openai_chunk("."),
            _make_openai_chunk("", finish_reason="stop", usage=usage),
        ]

        mock_create = MagicMock(return_value=iter(stream_chunks))
        mock_oa_client = MagicMock()
        mock_oa_client.chat.completions.create = mock_create

        with patch.object(openai_client, "_get_openai_client", return_value=mock_oa_client):
            tokens = []
            result = openai_client.chat_stream_openai(
                [ChatMessage(role="user", content="say hi")],
                on_token=tokens.append,
            )

        content_events = [t for t in tokens if t["phase"] == "content"]
        assert "".join(t["text"] for t in content_events) == "Hello world."
        assert result.content == "Hello world."
        assert result.input_tokens == 10
        assert result.output_tokens == 4

    def test_streaming_passes_stream_true(self, openai_client):
        mock_oa_client = MagicMock()
        mock_oa_client.chat.completions.create = MagicMock(return_value=iter([]))

        with patch.object(openai_client, "_get_openai_client", return_value=mock_oa_client):
            openai_client.chat_stream_openai(
                [ChatMessage(role="user", content="x")],
                on_token=lambda _: None,
            )

        kwargs = mock_oa_client.chat.completions.create.call_args.kwargs
        assert kwargs.get("stream") is True

    def test_empty_stream_returns_empty_content(self, openai_client):
        mock_oa_client = MagicMock()
        mock_oa_client.chat.completions.create = MagicMock(return_value=iter([]))

        with patch.object(openai_client, "_get_openai_client", return_value=mock_oa_client):
            result = openai_client.chat_stream_openai(
                [ChatMessage(role="user", content="x")],
                on_token=lambda _: None,
            )
        assert result.content == ""


# ── Bedrock streaming ────────────────────────────────────────────


@pytest.fixture
def bedrock_client():
    return LLMClient(LLMConfig(
        provider="bedrock",
        model="anthropic.claude-3-5-haiku-20241022-v1:0",
        region="us-west-2",
        max_retries=2,
        timeout_seconds=5,
    ))


def _bedrock_event(content_delta: str = "", thinking_delta: str = "",
                   metadata: dict = None) -> dict:
    """Build a single event dict mimicking Bedrock converse_stream output."""
    if metadata is not None:
        return {"metadata": metadata}
    if thinking_delta:
        return {"contentBlockDelta": {"delta": {"reasoningContent": {"text": thinking_delta}}}}
    if content_delta:
        return {"contentBlockDelta": {"delta": {"text": content_delta}}}
    return {}


class TestBedrockStreaming:
    """Stream events from Bedrock converse_stream, with thinking-block support."""

    def test_content_chunks_invoke_callback(self, bedrock_client):
        events = [
            _bedrock_event(content_delta="Paris "),
            _bedrock_event(content_delta="is the "),
            _bedrock_event(content_delta="capital."),
            _bedrock_event(metadata={
                "usage": {"inputTokens": 10, "outputTokens": 5},
            }),
        ]
        mock_bedrock = MagicMock()
        mock_bedrock.converse_stream.return_value = {"stream": iter(events)}

        with patch.object(bedrock_client, "_get_bedrock_client", return_value=mock_bedrock):
            tokens = []
            result = bedrock_client.chat_stream_bedrock(
                [ChatMessage(role="user", content="capital?")],
                on_token=tokens.append,
            )

        content_events = [t for t in tokens if t["phase"] == "content"]
        assert "".join(t["text"] for t in content_events) == "Paris is the capital."
        assert result.content == "Paris is the capital."
        assert result.input_tokens == 10
        assert result.output_tokens == 5

    def test_reasoning_chunks_emit_thinking_phase(self, bedrock_client):
        """Anthropic on Bedrock can emit reasoningContent blocks (extended thinking)."""
        events = [
            _bedrock_event(thinking_delta="Hmm let me think... "),
            _bedrock_event(thinking_delta="France is in Europe."),
            _bedrock_event(content_delta="Paris."),
            _bedrock_event(metadata={"usage": {"inputTokens": 5, "outputTokens": 12}}),
        ]
        mock_bedrock = MagicMock()
        mock_bedrock.converse_stream.return_value = {"stream": iter(events)}

        with patch.object(bedrock_client, "_get_bedrock_client", return_value=mock_bedrock):
            tokens = []
            bedrock_client.chat_stream_bedrock(
                [ChatMessage(role="user", content="capital?")],
                on_token=tokens.append,
            )

        thinking = [t for t in tokens if t["phase"] == "thinking"]
        content = [t for t in tokens if t["phase"] == "content"]
        assert "".join(t["text"] for t in thinking) == "Hmm let me think... France is in Europe."
        assert "".join(t["text"] for t in content) == "Paris."

    def test_validation_error_raises_llm_client_error(self, bedrock_client):
        botocore = pytest.importorskip("botocore")
        from botocore.exceptions import ClientError as BotoClientError

        err = BotoClientError(
            error_response={"Error": {"Code": "ValidationException", "Message": "bad model id"}},
            operation_name="ConverseStream",
        )
        mock_bedrock = MagicMock()
        mock_bedrock.converse_stream.side_effect = err

        with patch.object(bedrock_client, "_get_bedrock_client", return_value=mock_bedrock):
            with pytest.raises(LLMClientError):
                bedrock_client.chat_stream_bedrock(
                    [ChatMessage(role="user", content="x")],
                    on_token=lambda _: None,
                )


# ── Provider-agnostic chat_stream dispatcher ──────────────────────


class TestChatStreamDispatch:
    """LLMClient.chat_stream() routes to the right backend by provider."""

    def test_ollama_dispatch(self):
        client = LLMClient(LLMConfig(provider="ollama", model="qwen3:14b"))
        with patch.object(client, "chat_stream_ollama", return_value=ChatResponse(content="ok", model="x")) as mock:
            r = client.chat_stream(
                [ChatMessage(role="user", content="x")],
                on_token=lambda _: None,
            )
        mock.assert_called_once()
        assert r.content == "ok"

    def test_openai_dispatch(self):
        client = LLMClient(LLMConfig(provider="openai", model="gpt-4o", base_url="x", api_key_env="X"))
        with patch.object(client, "chat_stream_openai", return_value=ChatResponse(content="ok", model="x")) as mock:
            r = client.chat_stream(
                [ChatMessage(role="user", content="x")],
                on_token=lambda _: None,
            )
        mock.assert_called_once()
        assert r.content == "ok"

    def test_bedrock_dispatch(self):
        client = LLMClient(LLMConfig(provider="bedrock", model="anthropic.x"))
        with patch.object(client, "chat_stream_bedrock", return_value=ChatResponse(content="ok", model="x")) as mock:
            r = client.chat_stream(
                [ChatMessage(role="user", content="x")],
                on_token=lambda _: None,
            )
        mock.assert_called_once()
        assert r.content == "ok"

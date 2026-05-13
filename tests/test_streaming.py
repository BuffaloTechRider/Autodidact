"""Tests for streaming chat responses from Ollama.

The contract:

- ``LLMClient.chat_stream_ollama(messages, on_token=callback, **opts)`` issues
  a streaming POST to /api/chat. For each chunk, it calls ``on_token`` with
  a dict ``{"phase": "thinking" | "content", "text": "..."}``. After the
  stream ends, returns a full ``ChatResponseWithLogprobs`` accumulated from
  all chunks (final chunk has token counts + logprobs).
- Errors during the stream surface as ``LLMClientError``; ReadTimeout is
  not retried (same policy as non-streaming).

These tests use mocked HTTP so no real Ollama daemon is required.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
import requests

from autodidact.llm_client import (
    ChatMessage,
    ChatResponseWithLogprobs,
    LLMClient,
    LLMClientError,
    LLMConfig,
)


# ── Fixtures ──────────────────────────────────────────────────────


@pytest.fixture
def ollama_client():
    return LLMClient(LLMConfig(
        provider="ollama",
        model="qwen3:14b",
        max_retries=2,
        timeout_seconds=5,
    ))


def _make_streaming_response(chunks: list[dict]):
    """Build a mocked requests.Response that yields NDJSON chunks via iter_lines."""
    resp = MagicMock(status_code=200)
    resp.iter_lines.return_value = [
        json.dumps(chunk).encode("utf-8") for chunk in chunks
    ]
    return resp


# ── Basic streaming: content only ─────────────────────────────────


class TestContentStreaming:
    """A non-thinking model streams pure content chunks."""

    @patch("autodidact.llm_client.requests.post")
    def test_content_chunks_invoke_callback(self, mock_post, ollama_client):
        chunks = [
            {"message": {"content": "Paris "}, "done": False},
            {"message": {"content": "is "}, "done": False},
            {"message": {"content": "the capital."}, "done": False},
            {
                "message": {"content": ""},
                "done": True,
                "model": "qwen2.5:7b",
                "prompt_eval_count": 30,
                "eval_count": 7,
            },
        ]
        mock_post.return_value = _make_streaming_response(chunks)

        tokens: list[dict] = []
        result = ollama_client.chat_stream_ollama(
            [ChatMessage(role="user", content="capital of France?")],
            on_token=tokens.append,
        )

        # Three content tokens emitted, all with phase='content'.
        content_events = [t for t in tokens if t["phase"] == "content"]
        assert len(content_events) == 3
        assert "".join(t["text"] for t in content_events) == "Paris is the capital."

        # Final result accumulates everything.
        assert isinstance(result, ChatResponseWithLogprobs)
        assert result.content == "Paris is the capital."
        assert result.input_tokens == 30
        assert result.output_tokens == 7

    @patch("autodidact.llm_client.requests.post")
    def test_no_chunks_returns_empty_content(self, mock_post, ollama_client):
        chunks = [{"done": True, "prompt_eval_count": 5, "eval_count": 0}]
        mock_post.return_value = _make_streaming_response(chunks)
        result = ollama_client.chat_stream_ollama(
            [ChatMessage(role="user", content="x")],
            on_token=lambda _: None,
        )
        assert result.content == ""


# ── Thinking-model streaming: separate phase events ───────────────


class TestThinkingStreaming:
    """Thinking models emit chunks with `thinking` field; we must distinguish."""

    @patch("autodidact.llm_client.requests.post")
    def test_thinking_chunks_have_thinking_phase(self, mock_post, ollama_client):
        chunks = [
            {"message": {"thinking": "Let me ", "content": ""}, "done": False},
            {"message": {"thinking": "think...", "content": ""}, "done": False},
            {"message": {"thinking": "", "content": "Paris."}, "done": False},
            {
                "message": {"content": ""},
                "done": True,
                "prompt_eval_count": 30,
                "eval_count": 4,
            },
        ]
        mock_post.return_value = _make_streaming_response(chunks)

        tokens: list[dict] = []
        result = ollama_client.chat_stream_ollama(
            [ChatMessage(role="user", content="capital?")],
            on_token=tokens.append,
        )

        thinking_events = [t for t in tokens if t["phase"] == "thinking"]
        content_events = [t for t in tokens if t["phase"] == "content"]
        assert "".join(t["text"] for t in thinking_events) == "Let me think..."
        assert "".join(t["text"] for t in content_events) == "Paris."

        # The final result.content is content only (no thinking leakage).
        assert result.content == "Paris."

    @patch("autodidact.llm_client.requests.post")
    def test_thinking_only_falls_back_to_thinking_as_content(self, mock_post, ollama_client):
        """Edge: model emitted only thinking, no content. Same fallback as non-streaming."""
        chunks = [
            {"message": {"thinking": "Hmm.", "content": ""}, "done": False},
            {"message": {"content": ""}, "done": True,
             "prompt_eval_count": 5, "eval_count": 1},
        ]
        mock_post.return_value = _make_streaming_response(chunks)

        result = ollama_client.chat_stream_ollama(
            [ChatMessage(role="user", content="x")],
            on_token=lambda _: None,
        )
        # When content is empty after the stream, fall back to thinking
        # as the visible answer (matches _extract_answer fallback).
        assert result.content == "Hmm."


# ── Logprobs accumulated from final chunk ─────────────────────────


class TestLogprobsAccumulation:
    """Final chunk carries the logprobs array; we expose it on the response."""

    @patch("autodidact.llm_client.requests.post")
    def test_logprobs_in_final_chunk(self, mock_post, ollama_client):
        chunks = [
            {"message": {"content": "Paris."}, "done": False},
            {
                "message": {"content": ""},
                "done": True,
                "prompt_eval_count": 5,
                "eval_count": 1,
                "logprobs": [
                    {"token": "Paris", "logprob": -0.1,
                     "top_logprobs": [{"token": "Paris", "logprob": -0.1}]},
                ],
            },
        ]
        mock_post.return_value = _make_streaming_response(chunks)

        result = ollama_client.chat_stream_ollama(
            [ChatMessage(role="user", content="capital?")],
            on_token=lambda _: None,
        )
        assert result.logprobs == [-0.1]
        assert result.avg_logprob is not None
        assert abs(result.avg_logprob - (-0.1)) < 1e-9


# ── Error handling ────────────────────────────────────────────────


class TestStreamingErrors:

    @patch("autodidact.llm_client.requests.post")
    def test_http_error_raises_llmclient_error(self, mock_post, ollama_client):
        mock_post.return_value = MagicMock(status_code=500, text="server boom")
        with pytest.raises(LLMClientError):
            ollama_client.chat_stream_ollama(
                [ChatMessage(role="user", content="x")],
                on_token=lambda _: None,
            )

    @patch("autodidact.llm_client.requests.post")
    def test_read_timeout_raises_llmclient_error_no_retry(self, mock_post, ollama_client):
        mock_post.side_effect = requests.exceptions.ReadTimeout("slow")
        with pytest.raises(LLMClientError):
            ollama_client.chat_stream_ollama(
                [ChatMessage(role="user", content="x")],
                on_token=lambda _: None,
            )
        # Streaming gets the same fail-fast policy as non-streaming.
        assert mock_post.call_count == 1

    @patch("autodidact.llm_client.requests.post")
    def test_connection_error_retries(self, mock_post, ollama_client):
        good = _make_streaming_response([
            {"message": {"content": "ok"}, "done": False},
            {"message": {"content": ""}, "done": True,
             "prompt_eval_count": 1, "eval_count": 1},
        ])
        mock_post.side_effect = [
            requests.ConnectionError("daemon down"),
            good,
        ]
        with patch("autodidact.llm_client.time.sleep"):  # speed up retry sleep
            result = ollama_client.chat_stream_ollama(
                [ChatMessage(role="user", content="x")],
                on_token=lambda _: None,
            )
        assert result.content == "ok"
        assert mock_post.call_count == 2

    @patch("autodidact.llm_client.requests.post")
    def test_malformed_chunk_is_skipped(self, mock_post, ollama_client):
        """A bad NDJSON line shouldn't crash the whole stream."""
        resp = MagicMock(status_code=200)
        resp.iter_lines.return_value = [
            b'{"message": {"content": "ok"}, "done": false}',
            b'not valid json',
            b'{"message": {"content": ""}, "done": true, "prompt_eval_count": 1, "eval_count": 1}',
        ]
        mock_post.return_value = resp

        result = ollama_client.chat_stream_ollama(
            [ChatMessage(role="user", content="x")],
            on_token=lambda _: None,
        )
        assert result.content == "ok"


# ── Streaming opts: think flag flows through ──────────────────────


class TestStreamingOptions:

    @patch("autodidact.llm_client.requests.post")
    def test_think_kwarg_is_forwarded(self, mock_post, ollama_client):
        """The think=True/False flag should appear in the request body."""
        mock_post.return_value = _make_streaming_response([
            {"message": {"content": "ok"}, "done": False},
            {"message": {"content": ""}, "done": True,
             "prompt_eval_count": 1, "eval_count": 1},
        ])
        ollama_client.chat_stream_ollama(
            [ChatMessage(role="user", content="x")],
            on_token=lambda _: None,
            think=True,
        )
        body = mock_post.call_args.kwargs["json"]
        assert body.get("think") is True
        assert body["stream"] is True

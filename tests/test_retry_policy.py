"""Tests for the retry policy on the Ollama HTTP layer.

Two failure classes have very different correct treatments:

- ConnectionError: the request never reached Ollama. Retrying is safe and
  desirable (covers "daemon restarted", "transient network", etc.).
- ReadTimeout: the request reached Ollama and is being processed. Retrying
  starts a brand-new generation server-side; the previous one isn't cancelled,
  and we burn the same wall time again. Better to fail fast.

The old policy retried both. With max_retries=6 and a 60s timeout, a slow
local generation could spend ~6 minutes ping-ponging before failing.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import requests

from autodidact.llm_client import (
    LLMClient,
    LLMClientError,
    LLMConfig,
)


@pytest.fixture
def ollama_client():
    """A bare LLMClient pointed at Ollama with low retry count for speed."""
    return LLMClient(LLMConfig(
        provider="ollama",
        model="qwen2.5:7b",
        max_retries=3,
        timeout_seconds=5,
    ))


# ── ConnectionError: still retried ────────────────────────────────


class TestConnectionErrorRetries:

    @patch("autodidact.llm_client.time.sleep")
    @patch("autodidact.llm_client.requests.post")
    def test_connection_error_retries_then_succeeds(self, mock_post, _sleep, ollama_client):
        """Connection errors trigger backoff retries, then succeed."""
        success = MagicMock(status_code=200)
        success.json.return_value = {"message": {"content": "ok"}, "model": "qwen2.5:7b"}

        mock_post.side_effect = [
            requests.ConnectionError("daemon down"),
            requests.ConnectionError("still down"),
            success,
        ]
        from autodidact.llm_client import ChatMessage
        result = ollama_client.chat([ChatMessage(role="user", content="hi")])
        assert result.content == "ok"
        assert mock_post.call_count == 3

    @patch("autodidact.llm_client.time.sleep")
    @patch("autodidact.llm_client.requests.post")
    def test_connection_error_exhausts_retries_then_raises(self, mock_post, _sleep, ollama_client):
        mock_post.side_effect = requests.ConnectionError("perma-down")
        from autodidact.llm_client import ChatMessage
        with pytest.raises(LLMClientError):
            ollama_client.chat([ChatMessage(role="user", content="hi")])
        # max_retries=3 → 3 attempts total.
        assert mock_post.call_count == 3


# ── ReadTimeout: NOT retried ──────────────────────────────────────


class TestReadTimeoutDoesNotRetry:
    """Read timeouts mean the server is busy generating. Retrying restarts
    the work; we want to fail fast so the caller can decide what to do."""

    @patch("autodidact.llm_client.time.sleep")
    @patch("autodidact.llm_client.requests.post")
    def test_read_timeout_raises_immediately(self, mock_post, _sleep, ollama_client):
        mock_post.side_effect = requests.exceptions.ReadTimeout("read timed out")
        from autodidact.llm_client import ChatMessage
        with pytest.raises(LLMClientError):
            ollama_client.chat([ChatMessage(role="user", content="hi")])
        # Crucial: only one call, no retries.
        assert mock_post.call_count == 1

    @patch("autodidact.llm_client.time.sleep")
    @patch("autodidact.llm_client.requests.post")
    def test_read_timeout_with_logprobs_does_not_retry(self, mock_post, _sleep, ollama_client):
        """Same policy on the chat_with_logprobs path."""
        mock_post.side_effect = requests.exceptions.ReadTimeout("slow generation")
        from autodidact.llm_client import ChatMessage
        with pytest.raises(LLMClientError):
            ollama_client.chat_with_logprobs([ChatMessage(role="user", content="hi")])
        assert mock_post.call_count == 1


# ── ConnectTimeout: still retried (it's a connection-class failure) ──


class TestConnectTimeoutRetries:
    """ConnectTimeout (TCP connect failed) is connection-class, not generation."""

    @patch("autodidact.llm_client.time.sleep")
    @patch("autodidact.llm_client.requests.post")
    def test_connect_timeout_retries(self, mock_post, _sleep, ollama_client):
        success = MagicMock(status_code=200)
        success.json.return_value = {"message": {"content": "ok"}, "model": "qwen2.5:7b"}
        mock_post.side_effect = [
            requests.exceptions.ConnectTimeout("connect timed out"),
            success,
        ]
        from autodidact.llm_client import ChatMessage
        ollama_client.chat([ChatMessage(role="user", content="hi")])
        assert mock_post.call_count == 2

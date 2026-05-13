"""Tests for the perf-driven removal of logprob requests from the chat path.

Background: a benchmark showed Ollama adds ~150ms per call when
logprobs=True, even at top_logprobs=1. After the had_thinking-skip fix,
the post-local gate ignores logprobs on thinking responses anyway. Net
result: we were paying for logprob computation we never used.

This change:
  - _call_local issues a no-logprob streaming request.
  - The post-local logprob confidence gate is removed.
  - GSA gate and refusal detector are unchanged.
  - chat_with_logprobs and chat_stream_ollama still exist for benchmarks
    and any external caller that wants logprobs.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from autodidact.agent import Agent, SavingsReport
from autodidact.database import init_database
from autodidact.knowledge_store import KnowledgeStore
from autodidact.llm_client import (
    ChatMessage,
    ChatResponse,
    ChatResponseWithLogprobs,
    LLMClient,
    LLMConfig,
)
from autodidact.types import AutodidactConfig, NewKnowledgeEntry


# ── _call_local does NOT request logprobs ─────────────────────────


@pytest.fixture
def ollama_client_with_streaming_capture():
    """LLMClient that captures whatever body it would post for /api/chat."""
    client = LLMClient(LLMConfig(provider="ollama", model="qwen3:8b"))
    return client


class TestCallLocalSkipsLogprobs:
    """Agent._call_local must call the local model WITHOUT requesting logprobs."""

    def test_streaming_path_uses_no_logprob_method(self):
        """The Ollama streaming call must NOT request logprobs."""
        agent = _make_agent_ollama()

        # Both methods may exist on the client; only one should be called.
        agent._local_client.chat_stream_ollama_no_logprobs = MagicMock(
            return_value=ChatResponse(content="Paris.", model="qwen3:8b")
        )
        agent._local_client.chat_stream_ollama = MagicMock(
            return_value=ChatResponseWithLogprobs(
                content="Paris.", model="qwen3:8b", avg_logprob=-0.1,
            )
        )

        agent._call_local([ChatMessage(role="user", content="x")], lambda _: None)

        # The non-logprob method should be the only one called.
        agent._local_client.chat_stream_ollama_no_logprobs.assert_called_once()
        agent._local_client.chat_stream_ollama.assert_not_called()

    def test_non_streaming_path_uses_chat_not_chat_with_logprobs(self):
        """Test mocks (no provider config) use plain chat(), not chat_with_logprobs()."""
        agent = _make_agent_no_provider()
        agent._local_client.chat = MagicMock(
            return_value=ChatResponse(content="Paris.", model="x")
        )
        agent._local_client.chat_with_logprobs = MagicMock(
            return_value=ChatResponseWithLogprobs(
                content="Paris.", model="x", avg_logprob=-0.1,
            )
        )

        agent._call_local([ChatMessage(role="user", content="x")], lambda _: None)

        agent._local_client.chat.assert_called_once()
        agent._local_client.chat_with_logprobs.assert_not_called()


# ── No post-local logprob gate ────────────────────────────────────


class TestNoPostLocalConfidenceGate:
    """Routing only escalates on refusal; logprob confidence is no longer checked."""

    def test_local_response_not_escalated_due_to_low_logprob(self):
        """Even with avg_logprob=-3.0 (would have failed old gate), stay local."""
        agent = _make_agent_full(local_content="The capital of France is Paris.")
        # Simulate a "low logprob" response — but with our change, the agent
        # never reads logprobs, so this can't trigger an escalation.
        resp = agent.query("What is the capital of France?")
        assert resp.routed_to == "local"

    def test_refusal_still_triggers_escalation(self):
        """Refusal detector remains active and continues to escalate hedges."""
        agent = _make_agent_full(local_content="I don't have real-time data on that.")
        resp = agent.query("What is the weather?")
        assert resp.routed_to == "cloud"
        assert getattr(resp, "escalated_on_refusal", False) is True


# ── Helpers ───────────────────────────────────────────────────────


def _make_agent_ollama() -> Agent:
    """Test agent with an Ollama-provider local client."""
    a = Agent.__new__(Agent)
    a.confidence_threshold = 0.7
    a.staleness_days = 7
    a.gsa_enabled = False
    a._db_path = ":memory:"
    a._conn = init_database(":memory:")
    a._config = AutodidactConfig(db_path=":memory:", embedding_dim=32)
    a.memory = KnowledgeStore(a._conn, a._config)
    local = MagicMock(spec=LLMClient)
    local.config = LLMConfig(provider="ollama", model="qwen3:8b")
    local.embed.return_value = np.zeros(32, dtype=np.float32)
    a._local_client = local
    a._cloud_client = MagicMock(spec=LLMClient)
    a._embed_client = local
    a._local_model_name = "ollama/qwen3:8b"
    a._cloud_model_name = "openai/gpt-4o"
    a._session_stats = SavingsReport()
    a._history = []
    a.documents = None
    a._gsa = None
    return a


def _make_agent_no_provider() -> Agent:
    """Test agent with a MagicMock local client missing config.provider."""
    a = _make_agent_ollama()
    # Strip the config attribute to simulate a bare mock.
    a._local_client = MagicMock(spec=LLMClient)
    a._embed_client = a._local_client
    a._embed_client.embed.return_value = np.zeros(32, dtype=np.float32)
    return a


def _make_agent_full(local_content: str) -> Agent:
    """Agent with mocked clients returning the given local content."""
    a = _make_agent_ollama()

    def fake_local_call(messages, *, on_token, **opts):
        # Stream one chunk to satisfy the streaming contract.
        on_token({"phase": "content", "text": local_content})
        return ChatResponse(content=local_content, model="qwen3:8b",
                            input_tokens=10, output_tokens=5, latency_ms=100)

    # The new no-logprobs streaming method.
    a._local_client.chat_stream_ollama_no_logprobs = MagicMock(side_effect=fake_local_call)

    # Cloud path used for refusal escalation tests.
    def fake_cloud(messages, **opts):
        return ChatResponse(content="Cloud answer.", model="gpt-4o",
                            input_tokens=20, output_tokens=5, latency_ms=200)
    a._cloud_client.chat = MagicMock(side_effect=fake_cloud)
    a._cloud_client.config = LLMConfig(
        provider="openai", model="gpt-4o",
        base_url="https://api.openai.com/v1", api_key_env="OPENAI_API_KEY",
    )
    a._cloud_client.chat_stream = MagicMock(side_effect=fake_cloud_stream)
    return a


def fake_cloud_stream(messages, *, on_token, **opts):
    on_token({"phase": "content", "text": "Cloud answer."})
    return ChatResponse(content="Cloud answer.", model="gpt-4o",
                        input_tokens=20, output_tokens=5, latency_ms=200)


# ── HTTP-level: the new no-logprobs streaming method works ────────


class TestChatStreamOllamaNoLogprobs:
    """The new chat_stream_ollama_no_logprobs method works and skips logprobs in the request."""

    def test_returns_chat_response(self, ollama_client_with_streaming_capture):
        client = ollama_client_with_streaming_capture
        chunks = [
            {"message": {"content": "Paris."}, "done": False},
            {"message": {"content": ""}, "done": True,
             "prompt_eval_count": 10, "eval_count": 1},
        ]
        resp = MagicMock(status_code=200)
        resp.iter_lines.return_value = [json.dumps(c).encode() for c in chunks]

        with patch("autodidact.llm_client.requests.post", return_value=resp) as mock_post:
            result = client.chat_stream_ollama_no_logprobs(
                [ChatMessage(role="user", content="capital?")],
                on_token=lambda _: None,
            )

        assert isinstance(result, ChatResponse)
        assert result.content == "Paris."

        # Verify the request body did NOT include logprobs.
        body = mock_post.call_args.kwargs["json"]
        assert body.get("logprobs") is not True
        assert "top_logprobs" not in body
        assert body["stream"] is True

    def test_thinking_chunks_still_emit(self, ollama_client_with_streaming_capture):
        """Thinking phase still flows through on_token even without logprobs."""
        client = ollama_client_with_streaming_capture
        chunks = [
            {"message": {"thinking": "let me think...", "content": ""}, "done": False},
            {"message": {"content": "Paris."}, "done": False},
            {"message": {"content": ""}, "done": True,
             "prompt_eval_count": 10, "eval_count": 1},
        ]
        resp = MagicMock(status_code=200)
        resp.iter_lines.return_value = [json.dumps(c).encode() for c in chunks]

        tokens = []
        with patch("autodidact.llm_client.requests.post", return_value=resp):
            client.chat_stream_ollama_no_logprobs(
                [ChatMessage(role="user", content="capital?")],
                on_token=tokens.append,
            )

        thinking_phases = [t for t in tokens if t["phase"] == "thinking"]
        content_phases = [t for t in tokens if t["phase"] == "content"]
        assert "".join(t["text"] for t in thinking_phases) == "let me think..."
        assert "".join(t["text"] for t in content_phases) == "Paris."

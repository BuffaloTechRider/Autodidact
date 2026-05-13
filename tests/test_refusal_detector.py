"""Tests for the refusal detector routing override.

Context: logprob-based confidence scoring has a known failure mode — the local
model emits a hedge ("I don't have real-time data", "Did you mean X?") with
highly confident tokens, so routing sees ~0.9 confidence and returns the
non-answer. This detector catches voluntary surrender signals and forces a
cloud escalation regardless of confidence.

See CONTEXT.md for the routing invariants being preserved here.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from autodidact.agent import Agent, SavingsReport, _looks_like_refusal
from autodidact.database import init_database
from autodidact.knowledge_store import KnowledgeStore
from autodidact.llm_client import ChatResponse, ChatResponseWithLogprobs, LLMClient
from autodidact.types import AutodidactConfig


# ── Pure-function tests: the detector itself ──────────────────────


class TestLooksLikeRefusal:
    """_looks_like_refusal identifies voluntary-surrender responses."""

    @pytest.mark.parametrize("text", [
        "I don't have real-time data access",
        "I don't have access to real-time information",
        "As of my last update, Paris was the capital.",
        "As of my knowledge cutoff in 2023, that was true.",
        "I cannot browse the web.",
        "I'm unable to check current prices.",
        "I don't have the ability to access the internet.",
        "It seems there might be a typo in your query. Are you referring to OpenCL?",
        "Did you mean Python instead of Pyton?",
        "Could you clarify what you mean by that?",
        "I'm not sure what you're asking.",
        "I don't know.",
    ])
    def test_flags_hedges_and_refusals(self, text):
        assert _looks_like_refusal(text) is True, f"Should flag: {text!r}"

    @pytest.mark.parametrize("text", [
        "Paris is the capital of France.",
        "OpenCL is an open standard for parallel programming.",
        "The answer is 42.",
        "Python is a high-level programming language created by Guido van Rossum.",
        "",  # empty should not crash
    ])
    def test_does_not_flag_real_answers(self, text):
        assert _looks_like_refusal(text) is False, f"Should NOT flag: {text!r}"

    def test_case_insensitive(self):
        """Detection works regardless of casing."""
        assert _looks_like_refusal("I DON'T HAVE REAL-TIME DATA") is True
        assert _looks_like_refusal("i don't know.") is True

    def test_handles_punctuation_and_formatting(self):
        """Markdown and punctuation don't break detection."""
        assert _looks_like_refusal(
            "**Note:** I don't have real-time data. However, here's what I know..."
        ) is True


# ── Integration tests: routing respects refusal signal ─────────────


@pytest.fixture
def mock_local_client():
    client = MagicMock(spec=LLMClient)
    client.embed.return_value = np.random.RandomState(0).randn(32).astype(np.float32)
    return client


@pytest.fixture
def mock_cloud_client():
    client = MagicMock(spec=LLMClient)
    client.chat.return_value = ChatResponse(
        content="San Jose is sunny, 72°F, with light winds.",
        model="gpt-4o",
        input_tokens=100,
        output_tokens=20,
        latency_ms=1000,
    )
    return client


@pytest.fixture
def agent(mock_local_client, mock_cloud_client):
    a = Agent.__new__(Agent)
    a.confidence_threshold = 0.7
    a.staleness_days = 7
    a._db_path = ":memory:"
    a._conn = init_database(":memory:")
    a._config = AutodidactConfig(db_path=":memory:", embedding_dim=32)
    a.memory = KnowledgeStore(a._conn, a._config)
    a._local_client = mock_local_client
    a._cloud_client = mock_cloud_client
    a._embed_client = mock_local_client
    a._local_model_name = "ollama/qwen2.5:7b"
    a._cloud_model_name = "openai/gpt-4o"
    a._session_stats = SavingsReport()
    a._history = []
    a.documents = None
    return a


class TestRefusalRouting:
    """A confident refusal should escalate to cloud, not be served as-is."""

    def test_confident_refusal_escalates(self, agent):
        """High logprob + refusal text → cloud, not local."""
        agent._local_client.chat_with_logprobs.return_value = ChatResponseWithLogprobs(
            content=(
                "I don't have real-time data access, but you can check "
                "weather.com for current San Jose weather."
            ),
            model="qwen2.5:7b",
            avg_logprob=-0.2,  # sigmoid → ~0.93 (would normally bypass cloud)
            logprobs=[-0.2],
            top_logprobs_by_position=[],
        )
        agent._local_client.chat.return_value = ChatResponse(content=(
                "I don't have real-time data access, but you can check "
                "weather.com for current San Jose weather."
            ), model="qwen2.5:7b")

        resp = agent.query("how is the weather in San Jose today")

        assert resp.routed_to == "cloud", "Refusal should have escalated"
        agent._cloud_client.chat.assert_called_once()

    def test_confident_clarification_request_escalates(self, agent):
        """'Did you mean X?' is a surrender too."""
        agent._local_client.chat_with_logprobs.return_value = ChatResponseWithLogprobs(
            content=(
                "It seems there might be a typo in your query. "
                "Are you referring to OpenCL instead of openclaw?"
            ),
            model="qwen2.5:7b",
            avg_logprob=-0.15,  # would be ~0.94 confidence
            logprobs=[-0.15],
            top_logprobs_by_position=[],
        )
        agent._local_client.chat.return_value = ChatResponse(content=(
                "It seems there might be a typo in your query. "
                "Are you referring to OpenCL instead of openclaw?"
            ), model="qwen2.5:7b")

        resp = agent.query("tell me more about openclaw")

        assert resp.routed_to == "cloud"
        agent._cloud_client.chat.assert_called_once()

    def test_confident_real_answer_still_routes_local(self, agent):
        """The fix should not cause false escalations on real answers."""
        agent._local_client.chat_with_logprobs.return_value = ChatResponseWithLogprobs(
            content="Paris is the capital of France.",
            model="qwen2.5:7b",
            avg_logprob=-0.13,
            logprobs=[-0.13],
            top_logprobs_by_position=[],
        )
        agent._local_client.chat.return_value = ChatResponse(content="Paris is the capital of France.", model="qwen2.5:7b")

        resp = agent.query("What is the capital of France?")

        assert resp.routed_to == "local"
        agent._cloud_client.chat.assert_not_called()

    def test_refusal_without_cloud_client_returns_local(self, agent):
        """If no cloud configured, return the (refusal) local answer rather than crash."""
        agent._cloud_client = None
        agent._local_client.chat_with_logprobs.return_value = ChatResponseWithLogprobs(
            content="I don't have real-time data access.",
            model="qwen2.5:7b",
            avg_logprob=-0.2,
            logprobs=[-0.2],
            top_logprobs_by_position=[],
        )
        agent._local_client.chat.return_value = ChatResponse(content="I don't have real-time data access.", model="qwen2.5:7b")

        resp = agent.query("weather today?")

        assert resp.routed_to == "local"

    def test_refusal_flag_is_exposed_on_response(self, agent):
        """QueryResponse exposes whether the refusal detector triggered.

        Useful for UI — can show 'I wasn't sure, so I asked the cloud model.'
        """
        agent._local_client.chat_with_logprobs.return_value = ChatResponseWithLogprobs(
            content="I don't know.",
            model="qwen2.5:7b",
            avg_logprob=-0.3,
            logprobs=[-0.3],
            top_logprobs_by_position=[],
        )
        agent._local_client.chat.return_value = ChatResponse(content="I don't know.", model="qwen2.5:7b")

        resp = agent.query("what's 2+2?")

        assert resp.routed_to == "cloud"
        assert getattr(resp, "escalated_on_refusal", False) is True

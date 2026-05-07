"""Tests for the Agent class — the core product API.

Tests use mocked LLM clients so they run without Ollama or cloud access.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from autodidact.agent import Agent, QueryResponse, SavingsReport, MEMORY_DIRECT_THRESHOLD
from autodidact.database import init_database
from autodidact.knowledge_store import KnowledgeStore
from autodidact.llm_client import ChatResponse, ChatResponseWithLogprobs, LLMClient
from autodidact.types import AutodidactConfig, NewKnowledgeEntry


@pytest.fixture
def mock_local_client():
    """A mock local LLM client that returns controllable responses."""
    client = MagicMock(spec=LLMClient)
    # Default: confident local answer.
    client.chat_with_logprobs.return_value = ChatResponseWithLogprobs(
        content="Paris is the capital of France.",
        model="qwen2.5:7b",
        input_tokens=50,
        output_tokens=10,
        latency_ms=500,
        logprobs=[-0.1, -0.2, -0.1],
        avg_logprob=-0.13,  # high confidence after sigmoid
        top_logprobs_by_position=[],
    )
    # Embedding: return a fixed 32-dim vector.
    client.embed.return_value = np.random.RandomState(42).randn(32).astype(np.float32)
    return client


@pytest.fixture
def mock_cloud_client():
    """A mock cloud LLM client."""
    client = MagicMock(spec=LLMClient)
    client.chat.return_value = ChatResponse(
        content="The capital of France is Paris.",
        model="gpt-4o",
        input_tokens=100,
        output_tokens=20,
        latency_ms=1000,
    )
    return client


@pytest.fixture
def agent_with_mocks(mock_local_client, mock_cloud_client):
    """An Agent with mocked LLM clients and an in-memory DB."""
    agent = Agent.__new__(Agent)
    agent.confidence_threshold = 0.7
    agent.staleness_days = 7
    agent._db_path = ":memory:"
    agent._conn = init_database(":memory:")
    agent._config = AutodidactConfig(db_path=":memory:", embedding_dim=32)
    agent.memory = KnowledgeStore(agent._conn, agent._config)
    agent._local_client = mock_local_client
    agent._cloud_client = mock_cloud_client
    agent._embed_client = mock_local_client
    agent._local_model_name = "ollama/qwen2.5:7b"
    agent._cloud_model_name = "openai/gpt-4o"
    agent._session_stats = SavingsReport()
    agent._history = []
    return agent


class TestRouting:
    """Test that queries route correctly based on confidence."""

    def test_high_confidence_routes_locally(self, agent_with_mocks):
        """When logprob confidence is high, answer locally."""
        agent = agent_with_mocks
        # avg_logprob=-0.13 → sigmoid(2*(-0.13)+3) = sigmoid(2.74) ≈ 0.94
        resp = agent.query("What is the capital of France?")
        assert resp.routed_to == "local"
        assert resp.confidence > 0.7
        assert resp.cost_usd == 0.0
        assert resp.learned is False
        agent._cloud_client.chat.assert_not_called()

    def test_low_confidence_escalates_to_cloud(self, agent_with_mocks):
        """When logprob confidence is low, escalate to cloud."""
        agent = agent_with_mocks
        # Set avg_logprob very negative → low confidence.
        agent._local_client.chat_with_logprobs.return_value = ChatResponseWithLogprobs(
            content="I think it might be Lyon?",
            model="qwen2.5:7b",
            avg_logprob=-3.0,  # sigmoid(2*(-3)+3) = sigmoid(-3) ≈ 0.05
            logprobs=[-3.0],
            top_logprobs_by_position=[],
        )
        resp = agent.query("What is the GDP of France?")
        assert resp.routed_to == "cloud"
        assert resp.cost_usd > 0
        assert resp.learned is True
        agent._cloud_client.chat.assert_called_once()

    def test_no_local_model_goes_to_cloud(self, agent_with_mocks):
        """Without a local model, everything goes to cloud."""
        agent = agent_with_mocks
        agent._local_client = None
        resp = agent.query("What is the capital of France?")
        assert resp.routed_to == "cloud"

    def test_no_cloud_model_stays_local(self, agent_with_mocks):
        """Without a cloud model, low-confidence answers still return locally."""
        agent = agent_with_mocks
        agent._cloud_client = None
        agent._local_client.chat_with_logprobs.return_value = ChatResponseWithLogprobs(
            content="Maybe Lyon?",
            model="qwen2.5:7b",
            avg_logprob=-3.0,
            logprobs=[-3.0],
            top_logprobs_by_position=[],
        )
        resp = agent.query("What is the GDP of France?")
        assert resp.routed_to == "local"
        assert resp.learned is False


class TestMemory:
    """Test that the agent learns from escalations and recalls from memory."""

    def test_escalation_stores_in_kb(self, agent_with_mocks):
        """Cloud escalation should store the Q&A in the knowledge store."""
        agent = agent_with_mocks
        agent._local_client.chat_with_logprobs.return_value = ChatResponseWithLogprobs(
            content="dunno", model="qwen2.5:7b", avg_logprob=-3.0,
            logprobs=[-3.0], top_logprobs_by_position=[],
        )
        resp = agent.query("What is quantum entanglement?")
        assert resp.learned is True
        assert agent.memory.count() == 1

    def test_deduplication_on_similar_question(self, agent_with_mocks):
        """Asking a near-identical question shouldn't create duplicate KB entries."""
        agent = agent_with_mocks
        # Make local always uncertain so it escalates.
        agent._local_client.chat_with_logprobs.return_value = ChatResponseWithLogprobs(
            content="dunno", model="qwen2.5:7b", avg_logprob=-3.0,
            logprobs=[-3.0], top_logprobs_by_position=[],
        )
        # Same embedding for both queries (mocked embed returns same vector).
        agent.query("What is quantum entanglement?")
        assert agent.memory.count() == 1
        agent.query("Explain quantum entanglement")
        # Should deduplicate (same embedding → sim > 0.95 → replace).
        assert agent.memory.count() == 1


class TestCorrection:
    """Test the user correction flow."""

    def test_correct_invalidates_and_relearns(self, agent_with_mocks):
        """Calling correct() should invalidate old entry and store new one."""
        agent = agent_with_mocks
        # First: escalate and learn.
        agent._local_client.chat_with_logprobs.return_value = ChatResponseWithLogprobs(
            content="dunno", model="qwen2.5:7b", avg_logprob=-3.0,
            logprobs=[-3.0], top_logprobs_by_position=[],
        )
        agent.query("What year did the Berlin Wall fall?")
        assert agent.memory.count() == 1

        # Correct: should invalidate old, store new.
        agent._cloud_client.chat.return_value = ChatResponse(
            content="The Berlin Wall fell in 1989.",
            model="gpt-4o", input_tokens=50, output_tokens=10,
        )
        resp = agent.correct("What year did the Berlin Wall fall?")
        assert resp.routed_to == "cloud"
        assert resp.learned is True
        # Old entry invalidated, new one stored.
        assert agent.memory.count() == 1


class TestSavings:
    """Test cost tracking."""

    def test_savings_tracks_queries(self, agent_with_mocks):
        """Session stats should count queries by route."""
        agent = agent_with_mocks
        agent.query("Easy question")  # routes locally (high confidence)
        agent.query("Another easy one")
        s = agent.savings()
        assert s.total_queries == 2
        assert s.local_queries == 2
        assert s.cloud_queries == 0
        assert s.total_cost_usd == 0.0

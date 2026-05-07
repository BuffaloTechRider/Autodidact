"""Tests for progress callbacks — real-time UI updates during agent.query().

TDD: tests written first, then implementation.

The Agent emits ProgressEvent objects via an on_progress callback during query
processing. This enables real-time UI updates (thinking, memory hit, cloud call,
learning, etc.) instead of post-hoc rendering.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from autodidact.agent import Agent, QueryResponse, SavingsReport
from autodidact.database import init_database
from autodidact.knowledge_store import KnowledgeStore
from autodidact.llm_client import ChatResponse, ChatResponseWithLogprobs, LLMClient
from autodidact.types import AutodidactConfig, NewKnowledgeEntry


@pytest.fixture
def mock_local_client():
    client = MagicMock(spec=LLMClient)
    client.chat_with_logprobs.return_value = ChatResponseWithLogprobs(
        content="Paris is the capital of France.",
        model="qwen2.5:7b",
        input_tokens=50,
        output_tokens=10,
        latency_ms=500,
        logprobs=[-0.1, -0.2, -0.1],
        avg_logprob=-0.13,
        top_logprobs_by_position=[],
    )
    client.embed.return_value = np.random.RandomState(42).randn(32).astype(np.float32)
    # Also mock plain chat for the learning extractor
    client.chat.return_value = ChatResponse(
        content='{"knowledge": [{"content": "Paris is the capital of France", "confidence": 0.9}], "skills": []}',
        model="qwen2.5:7b",
        input_tokens=100,
        output_tokens=50,
    )
    return client


@pytest.fixture
def mock_cloud_client():
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


class TestProgressCallbackSignature:
    """Agent.query() accepts an optional on_progress callback."""

    def test_query_accepts_on_progress_kwarg(self, agent_with_mocks):
        """query() should accept on_progress without error."""
        events = []
        agent_with_mocks.query("What is the capital of France?", on_progress=lambda e: events.append(e))
        # Should have received at least one event.
        assert len(events) > 0

    def test_query_works_without_on_progress(self, agent_with_mocks):
        """query() still works when no callback is provided (backward compat)."""
        resp = agent_with_mocks.query("What is the capital of France?")
        assert resp.answer is not None


class TestProgressEventsLocalRoute:
    """When query routes locally, the right events fire in order."""

    def test_local_route_emits_thinking_then_local_done(self, agent_with_mocks):
        """Local route: thinking → local_done → answer."""
        events = []
        agent_with_mocks.query("Easy question", on_progress=lambda e: events.append(e))

        event_types = [e["type"] for e in events]
        assert "thinking" in event_types
        assert "local_done" in event_types
        # thinking should come before local_done
        assert event_types.index("thinking") < event_types.index("local_done")

    def test_thinking_event_has_memory_info(self, agent_with_mocks):
        """The thinking event should include memory search results."""
        events = []
        agent_with_mocks.query("Question", on_progress=lambda e: events.append(e))

        thinking = [e for e in events if e["type"] == "thinking"]
        assert len(thinking) == 1
        assert "memory_hits" in thinking[0]

    def test_local_done_event_has_confidence(self, agent_with_mocks):
        """The local_done event should include the confidence score."""
        events = []
        agent_with_mocks.query("Question", on_progress=lambda e: events.append(e))

        local_done = [e for e in events if e["type"] == "local_done"]
        assert len(local_done) == 1
        assert "confidence" in local_done[0]
        assert local_done[0]["confidence"] > 0.7


class TestProgressEventsCloudRoute:
    """When query escalates to cloud, the right events fire."""

    def test_cloud_route_emits_cloud_call_and_learning(self, agent_with_mocks):
        """Cloud route: thinking → cloud_call → cloud_done → learning."""
        agent = agent_with_mocks
        agent._local_client.chat_with_logprobs.return_value = ChatResponseWithLogprobs(
            content="dunno", model="qwen2.5:7b", avg_logprob=-3.0,
            logprobs=[-3.0], top_logprobs_by_position=[],
        )

        events = []
        agent.query("Hard question", on_progress=lambda e: events.append(e))

        event_types = [e["type"] for e in events]
        assert "thinking" in event_types
        assert "cloud_call" in event_types
        assert "cloud_done" in event_types
        assert "learning" in event_types

    def test_cloud_done_event_has_cost(self, agent_with_mocks):
        """The cloud_done event should include cost and latency."""
        agent = agent_with_mocks
        agent._local_client.chat_with_logprobs.return_value = ChatResponseWithLogprobs(
            content="dunno", model="qwen2.5:7b", avg_logprob=-3.0,
            logprobs=[-3.0], top_logprobs_by_position=[],
        )

        events = []
        agent.query("Hard question", on_progress=lambda e: events.append(e))

        cloud_done = [e for e in events if e["type"] == "cloud_done"]
        assert len(cloud_done) == 1
        assert "cost" in cloud_done[0]
        assert "model" in cloud_done[0]

    def test_learning_event_has_count(self, agent_with_mocks):
        """The learning event should report how many entries were stored."""
        agent = agent_with_mocks
        agent._local_client.chat_with_logprobs.return_value = ChatResponseWithLogprobs(
            content="dunno", model="qwen2.5:7b", avg_logprob=-3.0,
            logprobs=[-3.0], top_logprobs_by_position=[],
        )

        events = []
        agent.query("Hard question", on_progress=lambda e: events.append(e))

        learning = [e for e in events if e["type"] == "learning"]
        assert len(learning) == 1
        assert "knowledge_count" in learning[0]
        assert learning[0]["knowledge_count"] >= 1


class TestProgressEventsMemoryRoute:
    """When query is answered from memory, the right events fire."""

    def test_memory_route_emits_memory_hit(self, agent_with_mocks):
        """Memory route: thinking (with high-sim hit) → memory_hit."""
        agent = agent_with_mocks
        # Seed a memory entry.
        emb = agent._embed_client.embed("test")
        agent.memory.insert(NewKnowledgeEntry(
            content="Paris is the capital of France.",
            question="What is the capital of France?",
            source="cloud_escalation",
            confidence=0.9,
            embedding=emb.tolist(),
            answer_embedding=emb.tolist(),
        ))

        events = []
        agent.query("What is the capital of France?", on_progress=lambda e: events.append(e))

        event_types = [e["type"] for e in events]
        assert "thinking" in event_types
        assert "memory_hit" in event_types

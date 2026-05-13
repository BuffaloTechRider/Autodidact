"""Tests that the Agent uses streaming for both local AND cloud paths.

Local was wired in test_streaming.py; this file adds cloud streaming
through Agent._escalate_to_cloud emitting `token` events with phase.
"""

from __future__ import annotations

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
from autodidact.types import AutodidactConfig


@pytest.fixture
def mock_local_client():
    """Local: Ollama (streaming aware), low-confidence response forces escalation."""
    cli = MagicMock(spec=LLMClient)
    cli.config = LLMConfig(provider="ollama", model="qwen3:14b")
    cli.embed.return_value = np.random.RandomState(0).randn(32).astype(np.float32)

    def fake_stream(messages, *, on_token, **opts):
        on_token({"phase": "content", "text": "I dunno."})
        return ChatResponseWithLogprobs(
            content="I dunno.",
            model="qwen3:14b",
            avg_logprob=-3.0,  # forces low confidence -> escalate
            logprobs=[-3.0],
            top_logprobs_by_position=[],
        )

    cli.chat_stream_ollama.side_effect = fake_stream
    return cli


@pytest.fixture
def mock_cloud_client():
    """Cloud: openai-compat with a streaming chat method."""
    cli = MagicMock(spec=LLMClient)
    cli.config = LLMConfig(
        provider="openai",
        model="gpt-4o-mini",
        base_url="https://api.openai.com/v1",
        api_key_env="OPENAI_API_KEY",
    )

    def fake_stream(messages, *, on_token, **opts):
        on_token({"phase": "content", "text": "Paris "})
        on_token({"phase": "content", "text": "is "})
        on_token({"phase": "content", "text": "the capital."})
        return ChatResponse(
            content="Paris is the capital.",
            model="gpt-4o-mini",
            input_tokens=20,
            output_tokens=5,
            latency_ms=300,
        )

    cli.chat_stream.side_effect = fake_stream
    return cli


@pytest.fixture
def agent(mock_local_client, mock_cloud_client):
    a = Agent.__new__(Agent)
    a.confidence_threshold = 0.7
    a.staleness_days = 7
    a.gsa_enabled = False  # don't run a probe in this test; we want to exercise
    # the full local→cloud escalation path explicitly.
    a._db_path = ":memory:"
    a._conn = init_database(":memory:")
    a._config = AutodidactConfig(db_path=":memory:", embedding_dim=32)
    a.memory = KnowledgeStore(a._conn, a._config)
    a._local_client = mock_local_client
    a._cloud_client = mock_cloud_client
    a._embed_client = mock_local_client
    a._local_model_name = "ollama/qwen3:14b"
    a._cloud_model_name = "openai/gpt-4o-mini"
    a._session_stats = SavingsReport()
    a._history = []
    a.documents = None
    a._gsa = None
    return a


class TestCloudStreamingViaAgent:
    """Agent.query escalates and emits token events from the cloud stream too."""

    def test_cloud_escalation_emits_token_events(self, agent):
        events = []
        resp = agent.query("What is the capital of France?", on_progress=events.append)

        assert resp.routed_to == "cloud"
        # cloud_call/cloud_done sandwich should still fire.
        types = [e.get("type") for e in events]
        assert "cloud_call" in types
        assert "cloud_done" in types

        # Token events from the cloud stream.
        token_events = [e for e in events if e.get("type") == "token"]
        # Local emitted 1 + cloud emitted 3 = 4 tokens.
        assert len(token_events) >= 3
        cloud_tokens = "".join(
            e.get("text", "") for e in token_events
            if e.get("phase") == "content" and e.get("source") == "cloud"
        )
        assert cloud_tokens == "Paris is the capital."

    def test_cloud_token_events_carry_source_field(self, agent):
        """token events from cloud must be tagged with source='cloud' so the UI
        can prefix them differently from local tokens."""
        events = []
        agent.query("question?", on_progress=events.append)

        cloud_token_events = [
            e for e in events
            if e.get("type") == "token" and e.get("source") == "cloud"
        ]
        assert len(cloud_token_events) >= 3, (
            f"Expected cloud token events with source='cloud'; got {events}"
        )

    def test_local_token_events_carry_source_field(self, agent):
        """Symmetric: local tokens get source='local'."""
        events = []
        agent.query("question?", on_progress=events.append)

        local_token_events = [
            e for e in events
            if e.get("type") == "token" and e.get("source") == "local"
        ]
        assert len(local_token_events) >= 1

"""Tests for Task 4: Cost Tracking (R6).

4.1 — Per-query logging to query_log table + cost estimation.
4.2 — Cumulative savings calculator reading from DB.
"""

from __future__ import annotations

import uuid
from unittest.mock import MagicMock

import numpy as np
import pytest

from autodidact.agent import Agent, SavingsReport, _DEFAULT_COST_RATES
from autodidact.database import init_database
from autodidact.knowledge_store import KnowledgeStore
from autodidact.llm_client import ChatResponse, ChatResponseWithLogprobs, LLMClient
from autodidact.types import AutodidactConfig


@pytest.fixture
def agent_with_db():
    """Agent with mocked LLM clients and an in-memory SQLite DB."""
    agent = Agent.__new__(Agent)
    agent.confidence_threshold = 0.7
    agent.staleness_days = 7
    agent._db_path = ":memory:"
    agent._conn = init_database(":memory:")
    agent._config = AutodidactConfig(db_path=":memory:", embedding_dim=32)
    agent.memory = KnowledgeStore(agent._conn, agent._config)

    local = MagicMock(spec=LLMClient)
    local.chat_with_logprobs.return_value = ChatResponseWithLogprobs(
        content="Paris is the capital of France.",
        model="qwen2.5:7b",
        input_tokens=50,
        output_tokens=10,
        latency_ms=500,
        logprobs=[-0.1, -0.2, -0.1],
        avg_logprob=-0.13,
        top_logprobs_by_position=[],
    )
    local.embed.return_value = np.random.RandomState(42).randn(32).astype(np.float32)

    cloud = MagicMock(spec=LLMClient)
    cloud.chat.return_value = ChatResponse(
        content="The capital of France is Paris.",
        model="gpt-4o",
        input_tokens=100,
        output_tokens=20,
        latency_ms=1000,
    )

    agent._local_client = local
    agent._cloud_client = cloud
    agent._embed_client = local
    agent._local_model_name = "ollama/qwen2.5:7b"
    agent._cloud_model_name = "openai/gpt-4o"
    agent._session_stats = SavingsReport()
    agent._history = []
    return agent


# ── 4.1: Per-query DB persistence ─────────────────────────────────


class TestQueryLogPersistence:
    """_record_query() should write rows to the query_log table."""

    def test_local_query_persisted(self, agent_with_db):
        """A locally-routed query should appear in query_log."""
        agent = agent_with_db
        agent.query("What is the capital of France?")

        rows = agent._conn.execute("SELECT * FROM query_log").fetchall()
        assert len(rows) == 1
        row = rows[0]
        assert row["routing_decision"] == "local"
        assert row["cost"] == 0.0
        assert row["latency_ms"] >= 0
        assert row["query_text"] == "What is the capital of France?"

    def test_cloud_query_persisted(self, agent_with_db):
        """A cloud-escalated query should appear in query_log with cost > 0."""
        agent = agent_with_db
        agent._local_client.chat_with_logprobs.return_value = ChatResponseWithLogprobs(
            content="dunno", model="qwen2.5:7b", avg_logprob=-3.0,
            logprobs=[-3.0], top_logprobs_by_position=[],
        )
        agent.query("What is the GDP of France?")

        rows = agent._conn.execute("SELECT * FROM query_log").fetchall()
        assert len(rows) == 1
        row = rows[0]
        assert row["routing_decision"] == "cloud"
        assert row["cost"] > 0

    def test_multiple_queries_all_persisted(self, agent_with_db):
        """Multiple queries should each get their own row."""
        agent = agent_with_db
        agent.query("Question 1")
        agent.query("Question 2")
        agent.query("Question 3")

        count = agent._conn.execute("SELECT COUNT(*) FROM query_log").fetchone()[0]
        assert count == 3

    def test_query_log_has_timestamp(self, agent_with_db):
        """Each query_log row should have a created_at timestamp."""
        agent = agent_with_db
        agent.query("What time is it?")

        row = agent._conn.execute("SELECT created_at FROM query_log").fetchone()
        assert row["created_at"] is not None
        assert len(row["created_at"]) > 0  # ISO format string


# ── 4.2: Cumulative savings from DB ───────────────────────────────


class TestCumulativeSavings:
    """savings() should return cumulative data from the DB, not just session."""

    def test_savings_reflects_db_totals(self, agent_with_db):
        """savings() should count all rows in query_log, not just session."""
        agent = agent_with_db
        # Seed the DB with pre-existing rows (simulating a previous session).
        agent._conn.execute(
            "INSERT INTO query_log (id, query_text, routing_decision, signals, "
            "fusion_weights, fused_score, cost, latency_ms, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (str(uuid.uuid4()), "old question 1", "local", "{}", "{}", 0.9, 0.0, 200,
             "2025-01-01T00:00:00Z"),
        )
        agent._conn.execute(
            "INSERT INTO query_log (id, query_text, routing_decision, signals, "
            "fusion_weights, fused_score, cost, latency_ms, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (str(uuid.uuid4()), "old question 2", "cloud", "{}", "{}", 0.3, 0.005, 800,
             "2025-01-02T00:00:00Z"),
        )
        agent._conn.commit()

        # Now query in this session.
        agent.query("New question")

        report = agent.savings()
        # Should include the 2 old rows + 1 new = 3 total.
        assert report.total_queries == 3
        assert report.local_queries == 2  # 1 old local + 1 new local
        assert report.cloud_queries == 1  # 1 old cloud

    def test_savings_computes_saved_amount(self, agent_with_db):
        """Saved = all-cloud estimate - actual cost."""
        agent = agent_with_db
        # 3 local queries → actual cost $0, all-cloud estimate > $0.
        agent.query("Q1")
        agent.query("Q2")
        agent.query("Q3")

        report = agent.savings()
        assert report.total_cost_usd == 0.0
        assert report.estimated_all_cloud_cost_usd > 0
        assert report.saved_usd > 0
        assert report.saved_pct > 0

    def test_savings_includes_facts_learned(self, agent_with_db):
        """facts_learned should count knowledge_entries from cloud_escalation."""
        agent = agent_with_db
        # Force cloud escalation to learn.
        agent._local_client.chat_with_logprobs.return_value = ChatResponseWithLogprobs(
            content="dunno", model="qwen2.5:7b", avg_logprob=-3.0,
            logprobs=[-3.0], top_logprobs_by_position=[],
        )
        agent.query("What is quantum entanglement?")

        report = agent.savings()
        assert report.facts_learned >= 1

    def test_savings_includes_memory_hit_rate(self, agent_with_db):
        """When memory queries exist, memory_queries should be counted."""
        agent = agent_with_db
        # Seed a memory query in the DB.
        agent._conn.execute(
            "INSERT INTO query_log (id, query_text, routing_decision, signals, "
            "fusion_weights, fused_score, cost, latency_ms, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (str(uuid.uuid4()), "remembered question", "memory", "{}", "{}", 0.92, 0.0, 50,
             "2025-01-03T00:00:00Z"),
        )
        agent._conn.commit()

        report = agent.savings()
        assert report.memory_queries >= 1

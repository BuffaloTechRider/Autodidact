"""Tests for the GSA routing gate.

Context: the prototype used 4 signals fused via Thompson sampling; we shipped
Python with only logprob confidence. This test suite covers restoring GSA
(already implemented in autodidact/signals/) as a pre-local gate.

Design:
  1. GSA runs BEFORE local generation (one extra 1-token call).
  2. If p_yes < gsa_threshold  -> skip local, escalate to cloud.
  3. If p_yes >= gsa_threshold -> generate locally, then apply the existing
     logprob-confidence + refusal-detector gates as before.

Rationale (user): "if yes with high prob, then let local generate and still
calculate and eval logprob". GSA filters out queries the model self-reports
it can't handle; logprob + refusal remain as backstops for queries the model
thought it could handle but actually can't.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from autodidact.agent import Agent, SavingsReport
from autodidact.database import init_database
from autodidact.knowledge_store import KnowledgeStore
from autodidact.llm_client import ChatResponse, ChatResponseWithLogprobs, LLMClient
from autodidact.signals.grounded_self_assessment import SelfAssessment, SelfAssessmentResult
from autodidact.types import AutodidactConfig


# ── Fixtures ──────────────────────────────────────────────────────


@pytest.fixture
def mock_local_client():
    """Local client. chat_with_logprobs returns a confident, real answer by default."""
    client = MagicMock(spec=LLMClient)
    client.embed.return_value = np.random.RandomState(0).randn(32).astype(np.float32)
    client.chat_with_logprobs.return_value = ChatResponseWithLogprobs(
        content="Paris is the capital of France.",
        model="qwen2.5:7b",
        input_tokens=50,
        output_tokens=10,
        latency_ms=500,
        logprobs=[-0.1],
        avg_logprob=-0.13,
        top_logprobs_by_position=[],
    )
    return client


@pytest.fixture
def mock_cloud_client():
    client = MagicMock(spec=LLMClient)
    client.chat.return_value = ChatResponse(
        content="The authoritative cloud answer.",
        model="gpt-4o",
        input_tokens=100,
        output_tokens=20,
        latency_ms=1000,
    )
    return client


def _make_agent(local_client, cloud_client, *, gsa_threshold=0.5, gsa_enabled=True):
    """Build a test agent with the GSA-gate knobs we're adding."""
    a = Agent.__new__(Agent)
    a.confidence_threshold = 0.7
    a.staleness_days = 7
    a.gsa_threshold = gsa_threshold
    a.gsa_enabled = gsa_enabled
    a._db_path = ":memory:"
    a._conn = init_database(":memory:")
    a._config = AutodidactConfig(db_path=":memory:", embedding_dim=32)
    a.memory = KnowledgeStore(a._conn, a._config)
    a._local_client = local_client
    a._cloud_client = cloud_client
    a._embed_client = local_client
    a._local_model_name = "ollama/qwen2.5:7b"
    a._cloud_model_name = "openai/gpt-4o"
    a._session_stats = SavingsReport()
    a._history = []
    a.documents = None
    a._gsa = None  # lazily built — tests patch this where relevant
    return a


def _mock_gsa(p_yes: float) -> SelfAssessment:
    """Build a SelfAssessment mock whose compute() returns the given p_yes."""
    gsa = MagicMock(spec=SelfAssessment)
    gsa.compute.return_value = SelfAssessmentResult(
        p_yes=p_yes,
        yes_logprob=-0.1 if p_yes > 0.5 else -2.0,
        no_logprob=-2.0 if p_yes > 0.5 else -0.1,
        extraction_mode="logprob_softmax",
        recognized=True,
        raw_response="YES" if p_yes > 0.5 else "NO",
    )
    return gsa


# ── The gate itself ────────────────────────────────────────────────


class TestGsaGate:
    """GSA runs before local and can short-circuit to cloud."""

    def test_low_p_yes_skips_local_and_escalates(
        self, mock_local_client, mock_cloud_client
    ):
        """p_yes < threshold -> don't bother calling local for generation, go straight to cloud.

        Note: the local client may still be called by the LearningExtractor
        AFTER the cloud answer arrives (to extract knowledge). That call is
        post-decision and doesn't violate the gate. We assert the routing
        decision and that cloud was the source of the answer.
        """
        agent = _make_agent(mock_local_client, mock_cloud_client, gsa_threshold=0.5)
        agent._gsa = _mock_gsa(p_yes=0.2)

        resp = agent.query("What's the population of Ulaanbaatar right now?")

        assert resp.routed_to == "cloud"
        mock_cloud_client.chat.assert_called_once()
        assert getattr(resp, "gsa_p_yes", None) == 0.2
        assert getattr(resp, "escalated_on_gsa", False) is True

    def test_high_p_yes_proceeds_to_local(
        self, mock_local_client, mock_cloud_client
    ):
        """p_yes >= threshold -> run local, and since logprobs are confident, stay local."""
        agent = _make_agent(mock_local_client, mock_cloud_client, gsa_threshold=0.5)
        agent._gsa = _mock_gsa(p_yes=0.9)

        resp = agent.query("What is the capital of France?")

        assert resp.routed_to == "local"
        mock_local_client.chat.assert_called_once()
        mock_cloud_client.chat.assert_not_called()
        assert getattr(resp, "gsa_p_yes", None) == 0.9
        assert getattr(resp, "escalated_on_gsa", False) is False

    def test_high_p_yes_but_refusal_still_escalates(
        self, mock_local_client, mock_cloud_client
    ):
        """GSA green-lights it but local emits a refusal anyway -> refusal detector wins."""
        agent = _make_agent(mock_local_client, mock_cloud_client, gsa_threshold=0.5)
        agent._gsa = _mock_gsa(p_yes=0.9)
        mock_local_client.chat_with_logprobs.return_value = ChatResponseWithLogprobs(
            content="I don't have real-time data access, but you can check weather.com.",
            model="qwen2.5:7b",
            avg_logprob=-0.2,  # would otherwise pass
            logprobs=[-0.2],
            top_logprobs_by_position=[],
        )
        mock_local_client.chat.return_value = ChatResponse(content="I don't have real-time data access, but you can check weather.com.", model="qwen2.5:7b")

        resp = agent.query("weather today?")

        assert resp.routed_to == "cloud"

    def test_gate_runs_after_memory_direct_hit(
        self, mock_local_client, mock_cloud_client
    ):
        """Memory direct-hit stays instant — don't burn a GSA call if we already know."""
        agent = _make_agent(mock_local_client, mock_cloud_client)
        # Seed a direct-hit memory entry.
        q_emb = mock_local_client.embed.return_value
        from autodidact.types import NewKnowledgeEntry
        agent.memory.insert(NewKnowledgeEntry(
            content="Paris is the capital of France.",
            question="What is the capital of France?",
            embedding=q_emb.tolist(),
            source="cloud_escalation",
            confidence=0.95,
            domain="geography",
            topic="capitals",
        ))
        gsa_mock = _mock_gsa(p_yes=0.1)  # would escalate if called
        agent._gsa = gsa_mock

        resp = agent.query("What is the capital of France?")

        assert resp.routed_to == "memory"
        gsa_mock.compute.assert_not_called()
        # Memory hits now generate a full response using memory as context
        # (not return raw stored fact), so local chat IS called.
        mock_local_client.chat.assert_called_once()

    def test_gate_disabled_falls_through_to_old_behavior(
        self, mock_local_client, mock_cloud_client
    ):
        """gsa_enabled=False skips the gate entirely — backwards-compatible escape hatch."""
        agent = _make_agent(mock_local_client, mock_cloud_client, gsa_enabled=False)
        # Even if GSA exists and would veto, it must not be consulted.
        gsa_mock = _mock_gsa(p_yes=0.1)
        agent._gsa = gsa_mock

        resp = agent.query("What is the capital of France?")

        gsa_mock.compute.assert_not_called()
        assert resp.routed_to == "local"  # logprob is high -> stay local

    def test_gate_no_cloud_client_returns_local_on_low_gsa(
        self, mock_local_client
    ):
        """If GSA wants to escalate but there's no cloud configured, run local anyway.

        This is safer than returning "GSA says no, bye" — the user at least gets
        an answer, same as the refusal-no-cloud fallback.
        """
        agent = _make_agent(mock_local_client, cloud_client=None, gsa_threshold=0.5)
        agent._cloud_client = None
        agent._gsa = _mock_gsa(p_yes=0.1)

        resp = agent.query("obscure question")

        assert resp.routed_to == "local"
        mock_local_client.chat.assert_called_once()

    def test_gsa_failure_does_not_block_query(
        self, mock_local_client, mock_cloud_client
    ):
        """If GSA raises, log and proceed — don't ship the whole query because of a probe."""
        agent = _make_agent(mock_local_client, mock_cloud_client)
        broken_gsa = MagicMock(spec=SelfAssessment)
        broken_gsa.compute.side_effect = RuntimeError("ollama timed out on probe")
        agent._gsa = broken_gsa

        resp = agent.query("What is the capital of France?")

        assert resp.routed_to == "local"  # falls through to old behavior
        mock_local_client.chat.assert_called_once()


class TestGsaConfig:
    """Agent.__init__ exposes gsa_enabled and gsa_threshold knobs."""

    def test_defaults(self):
        """Defaults: enabled, threshold 0.5."""
        # Use __new__ then __init__ would try to init DB — instead, test the attrs
        # after a fresh construction with a dummy path.
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            agent = Agent(
                local_model="ollama/qwen2.5:7b",
                db_path=f"{td}/test.db",
            )
            assert agent.gsa_enabled is True
            assert agent.gsa_threshold == 0.5

    def test_override(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            agent = Agent(
                local_model="ollama/qwen2.5:7b",
                db_path=f"{td}/test.db",
                gsa_enabled=False,
                gsa_threshold=0.7,
            )
            assert agent.gsa_enabled is False
            assert agent.gsa_threshold == 0.7

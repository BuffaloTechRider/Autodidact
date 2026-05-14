"""Tests for the GSA try/except scoping fix.

Bug from live testing: the message
  "GSA probe failed, skipping gate: Bedrock rejected request: ValidationException"
was misleading. GSA itself ran successfully; the Bedrock error was actually
from the cloud escalation that followed. The try/except block was too wide.

Fix: scope the try to ONLY the probe call. Errors from _escalate_to_cloud
propagate so the user sees the real problem.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from autodidact.agent import Agent, SavingsReport
from autodidact.database import init_database
from autodidact.knowledge_store import KnowledgeStore
from autodidact.llm_client import (
    ChatMessage,
    ChatResponse,
    LLMClient,
    LLMClientError,
    LLMConfig,
)
from autodidact.signals.grounded_self_assessment import SelfAssessment, SelfAssessmentResult
from autodidact.types import AutodidactConfig


def _make_agent_for_gsa_test(*, gsa_p_yes: float = 0.1):
    """Agent with GSA returning the requested p_yes; cloud client raising."""
    a = Agent.__new__(Agent)
    a.confidence_threshold = 0.7
    a.staleness_days = 7
    a.gsa_enabled = True
    a.gsa_threshold = 0.5
    a._db_path = ":memory:"
    a._conn = init_database(":memory:")
    a._config = AutodidactConfig(db_path=":memory:", embedding_dim=32)
    a.memory = KnowledgeStore(a._conn, a._config)

    local = MagicMock(spec=LLMClient)
    local.embed.return_value = np.zeros(32, dtype=np.float32)
    local.config = LLMConfig(provider="ollama", model="qwen3:8b")
    a._local_client = local

    cloud = MagicMock(spec=LLMClient)
    cloud.config = LLMConfig(
        provider="bedrock",
        model="us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        region="us-west-2",
    )
    a._cloud_client = cloud

    a._embed_client = local
    a._local_model_name = "ollama/qwen3:8b"
    a._cloud_model_name = "bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0"
    a._session_stats = SavingsReport()
    a._history = []
    a.documents = None

    # Mock GSA to return the requested p_yes.
    gsa = MagicMock(spec=SelfAssessment)
    gsa.compute.return_value = SelfAssessmentResult(
        p_yes=gsa_p_yes,
        yes_logprob=-2 if gsa_p_yes < 0.5 else -0.1,
        no_logprob=-0.1 if gsa_p_yes < 0.5 else -2,
        extraction_mode="logprob_softmax",
        recognized=True,
        raw_response="NO" if gsa_p_yes < 0.5 else "YES",
    )
    a._gsa = gsa
    return a


# ── Cloud escalation errors propagate, not swallowed by GSA except ──


class TestCloudEscalationErrorsPropagate:
    """When cloud_client.chat_stream raises, the user must see the real error.

    Before the fix, the wide GSA try/except swallowed the cloud error and
    logged a misleading "GSA probe failed" message. After the fix, escalation
    errors propagate.
    """

    def test_bedrock_validation_error_propagates(self):
        agent = _make_agent_for_gsa_test(gsa_p_yes=0.1)  # forces escalation

        # Make cloud raise the same error the user saw.
        agent._cloud_client.chat_stream = MagicMock(
            side_effect=LLMClientError("Bedrock rejected request: ValidationException"),
        )
        # In case the dispatcher falls back to .chat for non-recognized providers
        # in tests; not the path here but defensive.
        agent._cloud_client.chat = MagicMock(
            side_effect=LLMClientError("Bedrock rejected request: ValidationException"),
        )

        with pytest.raises(LLMClientError) as exc_info:
            agent.query("anything")
        assert "Bedrock" in str(exc_info.value)


# ── GSA probe failure still gracefully falls through ──────────────


class TestGsaProbeFailureFallsThrough:
    """If the GSA probe itself fails, the rest of the query continues."""

    def test_gsa_failure_does_not_block_local_generation(self, caplog):
        agent = _make_agent_for_gsa_test(gsa_p_yes=0.9)

        # Sabotage the probe.
        agent._gsa.compute.side_effect = RuntimeError("ollama probe timed out")

        # Local generation should still happen and produce a response.
        # Local is provider=ollama so _call_local hits chat_stream_ollama_no_logprobs.
        def fake_local_stream(messages, *, on_token, **opts):
            on_token({"phase": "content", "text": "Paris is the capital of France."})
            return ChatResponse(
                content="Paris is the capital of France.",
                model="qwen3:8b",
            )
        agent._local_client.chat_stream_ollama_no_logprobs = MagicMock(
            side_effect=fake_local_stream,
        )

        resp = agent.query("What is the capital of France?")
        assert resp.routed_to == "local"
        assert "Paris" in resp.answer

    def test_gsa_failure_logged_with_clear_message(self, caplog):
        import logging
        caplog.set_level(logging.WARNING, logger="autodidact.agent")

        agent = _make_agent_for_gsa_test(gsa_p_yes=0.9)
        agent._gsa.compute.side_effect = RuntimeError("ollama probe timed out")

        def fake_local_stream(messages, *, on_token, **opts):
            on_token({"phase": "content", "text": "x"})
            return ChatResponse(content="x", model="qwen3:8b")
        agent._local_client.chat_stream_ollama_no_logprobs = MagicMock(
            side_effect=fake_local_stream,
        )

        agent.query("anything")

        # The log message must be specific to the probe (not "Bedrock").
        warnings = [r.message for r in caplog.records if r.levelname == "WARNING"]
        assert any("GSA probe failed" in m for m in warnings)
        assert not any("Bedrock" in m for m in warnings), (
            "GSA's warning must not falsely mention Bedrock — that error "
            "comes from the escalation path, not the probe."
        )


# ── Bedrock preset is now discovered at wizard time ──────────────


class TestBedrockPresetIsDynamic:
    """Bedrock no longer ships a static model list — it's discovered at runtime.

    Hardcoded IDs broke easily: some required `us.` prefix, some didn't, and
    some IDs were invented entirely. The preset now leaves `models` empty;
    `discover_bedrock_models()` populates the picker at wizard time.
    """

    def test_static_model_list_is_empty(self):
        from autodidact.setup_wizard import _CLOUD_PRESETS
        bedrock = _CLOUD_PRESETS["bedrock"]
        assert bedrock["models"] == []
        assert bedrock["default_cheap"] == ""
        assert bedrock["default_expensive"] == ""

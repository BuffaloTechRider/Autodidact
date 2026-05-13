"""Tests for GSA passing think=false on its probe call.

Per docs/HALLUCINATION-PROBLEM.md research: chain-of-thought has been shown
in the literature to hurt calibration on single-token classification tasks
("CoT induces overconfidence in VLMs", "highly confident yet incorrect
outputs"). For our YES/NO p_yes probe, thinking adds latency without a
reliable AUROC benefit.

The user can keep thinking enabled for the chat path; GSA always disables
it for the probe specifically.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from autodidact.llm_client import ChatResponseWithLogprobs, LLMClient
from autodidact.signals.grounded_self_assessment import SelfAssessment


@pytest.fixture
def mock_client():
    client = MagicMock(spec=LLMClient)
    client.chat_with_logprobs.return_value = ChatResponseWithLogprobs(
        content="YES",
        model="qwen3:14b",
        avg_logprob=-0.1,
        logprobs=[-0.1],
        top_logprobs_by_position=[{" YES": -0.1, " NO": -2.0}],
    )
    return client


class TestGsaPassesThinkFalse:
    """SelfAssessment.compute() must include think=False in its LLM call."""

    def test_think_false_passed_in_kwargs(self, mock_client):
        sa = SelfAssessment(mock_client)
        sa.compute("any question")

        # Inspect the call args.
        call = mock_client.chat_with_logprobs.call_args
        kwargs = call.kwargs
        assert kwargs.get("think") is False, (
            f"GSA must pass think=False; got kwargs={kwargs}"
        )

    def test_think_false_passed_in_v4_too(self, mock_client):
        """Same behavior on v4 prompt — thinking is disabled regardless of prompt version."""
        sa = SelfAssessment(mock_client, prompt_version="v4")
        sa.compute("any question")
        kwargs = mock_client.chat_with_logprobs.call_args.kwargs
        assert kwargs.get("think") is False

    def test_max_tokens_increased_to_capture_yes_no(self, mock_client):
        """With think=false the model can emit YES/NO directly. Keep max_tokens=1."""
        sa = SelfAssessment(mock_client)
        sa.compute("any question")
        kwargs = mock_client.chat_with_logprobs.call_args.kwargs
        assert kwargs.get("max_tokens") == 1

"""Tests for GSA prompt templates.

Two concerns covered here:

1. The current (v3 default) prompts must actually render with whitespace
   between concatenated string literals. Earlier versions had
   'this question?' 'Answering NO' with no space, producing
   'this question?Answering NO' — harder for the model to parse.

2. A v4 variant is opt-in via prompt_version="v4". It uses an adversarial
   framing ("if you say YES and your answer is wrong, the user loses trust.
   Say NO unless you are confident.") to fight sycophancy-driven YES bias.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from autodidact.llm_client import ChatResponseWithLogprobs, LLMClient
from autodidact.signals.grounded_self_assessment import (
    BARE_PROMPT_TEMPLATE,
    PROMPT_VERSION,
    PROMPT_VERSION_V2,
    PROMPT_VERSION_V4,
    WITH_RETRIEVAL_PROMPT_TEMPLATE,
    SelfAssessment,
)


# ── Fix: v3 prompts must have sentence separators ──────────────────


class TestV3PromptWellFormed:
    """The shipped v3 templates must be human-readable (whitespace between sentences)."""

    def test_v3_bare_separates_sentences(self):
        """No 'question?Answering' run-on in the rendered prompt."""
        rendered = BARE_PROMPT_TEMPLATE.format(query="x")
        assert "?Answering" not in rendered
        assert "choice.The" not in rendered
        assert "one.Respond" not in rendered

    def test_v3_bare_readable_structure(self):
        """Sentences end with period+space or newline, not jammed together."""
        rendered = BARE_PROMPT_TEMPLATE.format(query="x")
        # The three sentences that live at the end should each be separable.
        assert rendered.count("YES or NO") >= 1
        # Post-fix, we expect spaces between consecutive sentences.
        assert "question? Answering" in rendered or "question?\nAnswering" in rendered

    def test_v3_with_retrieval_separates_sentences(self):
        rendered = WITH_RETRIEVAL_PROMPT_TEMPLATE.format(query="x", hits_block="h")
        assert "?Answering" not in rendered
        assert "choice.The" not in rendered
        assert "one.Respond" not in rendered


# ── Version dispatch: v4 is opt-in ─────────────────────────────────


@pytest.fixture
def mock_client():
    client = MagicMock(spec=LLMClient)
    client.chat_with_logprobs.return_value = ChatResponseWithLogprobs(
        content="YES",
        model="qwen2.5:7b",
        avg_logprob=-0.1,
        logprobs=[-0.1],
        top_logprobs_by_position=[{" YES": -0.1, " NO": -2.0}],
    )
    return client


class TestPromptVersionDispatch:
    """prompt_version parameter routes to the right template."""

    def test_default_is_v3(self, mock_client):
        sa = SelfAssessment(mock_client)
        assert sa.prompt_version == PROMPT_VERSION  # gsa-v3-retrieval-conditional

    def test_v4_opt_in(self, mock_client):
        sa = SelfAssessment(mock_client, prompt_version="v4")
        assert sa.prompt_version == PROMPT_VERSION_V4

    def test_v2_legacy_flag_still_works(self, mock_client):
        """Backwards compat with the older use_v2_legacy boolean."""
        sa = SelfAssessment(mock_client, use_v2_legacy=True)
        assert sa.prompt_version == PROMPT_VERSION_V2

    def test_v2_via_version_string(self, mock_client):
        sa = SelfAssessment(mock_client, prompt_version="v2")
        assert sa.prompt_version == PROMPT_VERSION_V2

    def test_invalid_version_raises(self, mock_client):
        with pytest.raises(ValueError):
            SelfAssessment(mock_client, prompt_version="v99")

    def test_conflicting_flags_raises(self, mock_client):
        """Passing both is ambiguous — refuse rather than silently pick one."""
        with pytest.raises(ValueError):
            SelfAssessment(mock_client, prompt_version="v4", use_v2_legacy=True)


# ── v4 bare prompt content ─────────────────────────────────────────


class TestV4BarePrompt:
    """v4 adversarial framing — trust cost + permission to say NO."""

    def test_v4_bare_has_trust_cost(self, mock_client):
        """Spells out the cost of a wrong YES: user loses trust."""
        sa = SelfAssessment(mock_client, prompt_version="v4")
        sa.compute("what is openclaw?")
        prompt = mock_client.chat_with_logprobs.call_args[0][0][0].content.lower()
        assert "trust" in prompt

    def test_v4_bare_instructs_no_unless_confident(self, mock_client):
        sa = SelfAssessment(mock_client, prompt_version="v4")
        sa.compute("what is openclaw?")
        prompt = mock_client.chat_with_logprobs.call_args[0][0][0].content.lower()
        assert "say no unless you are confident" in prompt

    def test_v4_bare_still_asks_yes_no(self, mock_client):
        sa = SelfAssessment(mock_client, prompt_version="v4")
        sa.compute("what is openclaw?")
        prompt = mock_client.chat_with_logprobs.call_args[0][0][0].content
        assert "YES" in prompt and "NO" in prompt

    def test_v4_bare_includes_user_query(self, mock_client):
        sa = SelfAssessment(mock_client, prompt_version="v4")
        sa.compute("what is openclaw?")
        prompt = mock_client.chat_with_logprobs.call_args[0][0][0].content
        assert "what is openclaw?" in prompt

    def test_v4_bare_no_whitespace_issues(self, mock_client):
        """v4 must not have the same run-on-sentence bug as the old v3 did."""
        sa = SelfAssessment(mock_client, prompt_version="v4")
        sa.compute("x")
        prompt = mock_client.chat_with_logprobs.call_args[0][0][0].content
        assert "?Say" not in prompt
        assert "wrong.Say" not in prompt
        assert "trust.Say" not in prompt


# ── v4 with-retrieval prompt ───────────────────────────────────────


class MockScoredEntry:
    """Minimal stand-in for ScoredKnowledgeEntry."""

    def __init__(self, content: str, question: str, score: float):
        self.score = score

        class _E:
            pass
        self.entry = _E()
        self.entry.content = content
        self.entry.question = question


class TestV4WithRetrievalPrompt:

    def test_v4_retrieval_includes_hits(self, mock_client):
        sa = SelfAssessment(mock_client, prompt_version="v4")
        hits = [MockScoredEntry(
            content="Paris is the capital of France.",
            question="What is the capital of France?",
            score=0.9,
        )]
        sa.compute("capital?", retrieved_hits=hits)
        prompt = mock_client.chat_with_logprobs.call_args[0][0][0].content
        assert "Paris is the capital of France." in prompt

    def test_v4_retrieval_still_has_trust_cost(self, mock_client):
        sa = SelfAssessment(mock_client, prompt_version="v4")
        hits = [MockScoredEntry(
            content="Paris.",
            question="capital?",
            score=0.9,
        )]
        sa.compute("capital?", retrieved_hits=hits)
        prompt = mock_client.chat_with_logprobs.call_args[0][0][0].content.lower()
        assert "trust" in prompt
        assert "say no unless you are confident" in prompt

    def test_v4_low_score_hit_falls_back_to_bare(self, mock_client):
        """Below min_similarity -> bare v4 prompt (identical to 'no retrieval run')."""
        sa = SelfAssessment(mock_client, prompt_version="v4", min_similarity=0.70)
        weak = [MockScoredEntry(content="Paris.", question="q?", score=0.5)]
        sa.compute("q?", retrieved_hits=weak)
        prompt = mock_client.chat_with_logprobs.call_args[0][0][0].content
        assert "Paris." not in prompt


# ── Integration: signal still extracts ─────────────────────────────


class TestV4SignalExtraction:

    def test_v4_extracts_p_yes_via_logprobs(self, mock_client):
        sa = SelfAssessment(mock_client, prompt_version="v4")
        result = sa.compute("x")
        assert result.extraction_mode == "logprob_softmax"
        assert result.p_yes > 0.8

    def test_v4_result_reports_v4_version(self, mock_client):
        sa = SelfAssessment(mock_client, prompt_version="v4")
        assert sa.prompt_version == PROMPT_VERSION_V4

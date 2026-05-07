"""Tests for the SelfAssessment signal, with focus on the v3 retrieval-conditional
prompt behavior promoted after EXP-005.

EXP-005 (n=931 qwen2.5:7b) found that showing strong retrieved hits improves GSA
AUROC by +0.037 over the always-bare v2 prompt, but only when the bare fallback
is INDISTINGUISHABLE from a no-retrieval run (no "no knowledge retrieved"
language that primes the model toward NO).

These tests verify:
  - v3 with strong hits produces the with-retrieval prompt
  - v3 without strong hits produces the bare prompt with NO priming language
  - v2-legacy mode always produces the bare prompt
  - min_similarity threshold is respected
  - extraction logic produces sensible p_yes from mocked logprobs
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock

import pytest

from autodidact.llm_client import ChatResponseWithLogprobs
from autodidact.signals.grounded_self_assessment import (
    BARE_PROMPT_TEMPLATE,
    DEFAULT_MIN_SIMILARITY_V3,
    PROMPT_VERSION,
    PROMPT_VERSION_V2,
    WITH_RETRIEVAL_PROMPT_TEMPLATE,
    SelfAssessment,
)


@dataclass
class _FakeEntry:
    content: str = "Water boils at 100C at 1 atm."
    question: str = "At what temperature does water boil?"


@dataclass
class _FakeScoredHit:
    """Minimal duck-typed stand-in for ScoredKnowledgeEntry."""
    score: float
    entry: _FakeEntry = field(default_factory=_FakeEntry)


def _mock_logprob_response(content: str, yes_lp: float, no_lp: float) -> ChatResponseWithLogprobs:
    """Build a fake ChatResponseWithLogprobs with the given YES/NO logprobs at position 0."""
    return ChatResponseWithLogprobs(
        content=content,
        model="test-model",
        input_tokens=10,
        output_tokens=1,
        latency_ms=10,
        logprobs=[yes_lp],
        avg_logprob=yes_lp,
        top_logprobs_by_position=[{"YES": yes_lp, "NO": no_lp}],
    )


class TestV3PromptSelection:
    """The v3 class picks BARE or WITH-RETRIEVAL based on hit scores."""

    def test_no_hits_uses_bare_prompt(self):
        client = MagicMock()
        client.chat_with_logprobs.return_value = _mock_logprob_response("YES", -0.1, -3.0)

        sa = SelfAssessment(client)  # default v3, min_similarity=0.70
        result = sa.compute("What is 2+2?", retrieved_hits=None)

        prompt = client.chat_with_logprobs.call_args[0][0][0].content
        assert "Here is what you recall" not in prompt, "bare prompt must not mention retrieval"
        assert "no relevant knowledge" not in prompt.lower(), "bare prompt must not prime for absence"
        assert "knowledge base" not in prompt.lower(), "bare prompt must not hint at memory"
        assert result.had_retrieval is False
        assert result.n_hits_used == 0

    def test_weak_hits_below_threshold_fall_back_to_bare(self):
        """Hits with score < min_similarity don't trigger the retrieval prompt."""
        client = MagicMock()
        client.chat_with_logprobs.return_value = _mock_logprob_response("YES", -0.1, -3.0)

        sa = SelfAssessment(client, min_similarity=0.70)
        weak_hits = [_FakeScoredHit(score=0.60), _FakeScoredHit(score=0.65)]
        result = sa.compute("What is 2+2?", retrieved_hits=weak_hits)

        prompt = client.chat_with_logprobs.call_args[0][0][0].content
        assert "Here is what you recall" not in prompt
        assert result.had_retrieval is False
        assert result.n_hits_used == 0

    def test_strong_hits_produce_retrieval_prompt(self):
        client = MagicMock()
        client.chat_with_logprobs.return_value = _mock_logprob_response("YES", -0.1, -3.0)

        sa = SelfAssessment(client, min_similarity=0.70)
        strong_hits = [
            _FakeScoredHit(score=0.85),
            _FakeScoredHit(score=0.72),
            _FakeScoredHit(score=0.50),  # this one below threshold; should be filtered
        ]
        result = sa.compute("What is 2+2?", retrieved_hits=strong_hits)

        prompt = client.chat_with_logprobs.call_args[0][0][0].content
        assert "Here is what you recall from your knowledge base" in prompt
        assert result.had_retrieval is True
        assert result.n_hits_used == 2  # the 0.50 hit was filtered out

    def test_bare_fallback_is_byte_identical_to_v2(self):
        """The bare fallback must be the same string v2 would produce so the
        model cannot distinguish 'retrieval ran and returned nothing strong'
        from 'retrieval was never attempted'."""
        client_v3 = MagicMock()
        client_v3.chat_with_logprobs.return_value = _mock_logprob_response("YES", -0.1, -3.0)
        client_v2 = MagicMock()
        client_v2.chat_with_logprobs.return_value = _mock_logprob_response("YES", -0.1, -3.0)

        sa_v3 = SelfAssessment(client_v3)
        sa_v2 = SelfAssessment(client_v2, use_v2_legacy=True)

        sa_v3.compute("Same question.", retrieved_hits=None)
        sa_v2.compute("Same question.", retrieved_hits=None)

        prompt_v3 = client_v3.chat_with_logprobs.call_args[0][0][0].content
        prompt_v2 = client_v2.chat_with_logprobs.call_args[0][0][0].content
        assert prompt_v3 == prompt_v2


class TestV2Legacy:
    """use_v2_legacy=True reproduces the pre-v3 behavior."""

    def test_v2_legacy_ignores_hits(self):
        client = MagicMock()
        client.chat_with_logprobs.return_value = _mock_logprob_response("YES", -0.1, -3.0)

        sa = SelfAssessment(client, use_v2_legacy=True)
        strong_hits = [_FakeScoredHit(score=0.95)]
        result = sa.compute("What is 2+2?", retrieved_hits=strong_hits)

        prompt = client.chat_with_logprobs.call_args[0][0][0].content
        assert "Here is what you recall" not in prompt
        assert result.had_retrieval is False

    def test_v2_legacy_uses_v2_prompt_version(self):
        sa = SelfAssessment(MagicMock(), use_v2_legacy=True)
        assert sa.prompt_version == PROMPT_VERSION_V2

    def test_v3_default_uses_v3_prompt_version(self):
        sa = SelfAssessment(MagicMock())
        assert sa.prompt_version == PROMPT_VERSION


class TestExtraction:
    """Three-tier extraction: logprob_softmax → text_hard → neutral."""

    def test_logprob_softmax_dominant_yes(self):
        client = MagicMock()
        client.chat_with_logprobs.return_value = _mock_logprob_response("YES", -0.01, -10.0)

        sa = SelfAssessment(client)
        result = sa.compute("Q", retrieved_hits=None)
        assert result.extraction_mode == "logprob_softmax"
        assert result.p_yes > 0.95

    def test_logprob_softmax_dominant_no(self):
        client = MagicMock()
        client.chat_with_logprobs.return_value = _mock_logprob_response("NO", -10.0, -0.01)

        sa = SelfAssessment(client)
        result = sa.compute("Q", retrieved_hits=None)
        assert result.extraction_mode == "logprob_softmax"
        assert result.p_yes < 0.05

    def test_text_hard_fallback_when_no_logprobs(self):
        client = MagicMock()
        client.chat_with_logprobs.return_value = ChatResponseWithLogprobs(
            content="YES",
            model="test-model",
            input_tokens=10,
            output_tokens=1,
            latency_ms=10,
            logprobs=[],
            avg_logprob=None,
            top_logprobs_by_position=[],
        )

        sa = SelfAssessment(client)
        result = sa.compute("Q", retrieved_hits=None)
        assert result.extraction_mode == "text_hard"
        assert result.p_yes == 1.0

    def test_neutral_fallback_for_garbage_response(self):
        client = MagicMock()
        client.chat_with_logprobs.return_value = ChatResponseWithLogprobs(
            content="maybe",
            model="test-model",
            input_tokens=10,
            output_tokens=1,
            latency_ms=10,
            logprobs=[],
            avg_logprob=None,
            top_logprobs_by_position=[],
        )

        sa = SelfAssessment(client)
        result = sa.compute("Q", retrieved_hits=None)
        assert result.extraction_mode == "neutral"
        assert result.p_yes == 0.5


class TestBackCompat:
    """The v1/v2 class name alias still works."""

    def test_grounded_self_assessment_alias_still_exists(self):
        from autodidact.signals.grounded_self_assessment import GroundedSelfAssessment
        assert GroundedSelfAssessment is SelfAssessment

    def test_default_min_similarity_is_070(self):
        assert DEFAULT_MIN_SIMILARITY_V3 == 0.70

    def test_bare_prompt_template_has_no_retrieval_language(self):
        """Defensive: any future prompt edit that adds retrieval-hint language
        to the bare template would defeat v3's whole purpose."""
        assert "knowledge base" not in BARE_PROMPT_TEMPLATE.lower()
        assert "retrieved" not in BARE_PROMPT_TEMPLATE.lower()
        assert "recall" not in BARE_PROMPT_TEMPLATE.lower()
        assert "memory" not in BARE_PROMPT_TEMPLATE.lower()

    def test_with_retrieval_template_expects_hits_block(self):
        assert "{hits_block}" in WITH_RETRIEVAL_PROMPT_TEMPLATE
        assert "{query}" in WITH_RETRIEVAL_PROMPT_TEMPLATE

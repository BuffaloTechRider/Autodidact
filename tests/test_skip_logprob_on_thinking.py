"""Tests for skipping the logprob confidence check on thinking-model responses.

Background: logprob-based confidence (avg log-probability of generated tokens)
is calibrated against the *answer*. On thinking models, `avg_logprob` is
diluted by thinking-token logprobs that are inherently noisy as the model
explores options. A perfectly correct answer with extensive thinking can
have avg_logprob below the routing threshold, causing false escalations.

Fix: ChatResponseWithLogprobs tracks whether the response included thinking
tokens. When it did, _compute_confidence returns max-confidence (1.0) so
the local-vs-cloud gate doesn't penalize thinking responses. The refusal
detector and GSA gate still fire and are unaffected.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from autodidact.agent import Agent, SavingsReport
from autodidact.database import init_database
from autodidact.knowledge_store import KnowledgeStore
from autodidact.llm_client import (
    ChatMessage,
    ChatResponseWithLogprobs,
    LLMClient,
    LLMConfig,
)
from autodidact.types import AutodidactConfig


# ── had_thinking field on the response ────────────────────────────


class TestHadThinkingField:
    """ChatResponseWithLogprobs has a had_thinking bool, defaulting to False."""

    def test_default_false(self):
        resp = ChatResponseWithLogprobs(content="hi", model="x")
        assert resp.had_thinking is False

    def test_explicit_true(self):
        resp = ChatResponseWithLogprobs(content="hi", model="x", had_thinking=True)
        assert resp.had_thinking is True


# ── _compute_confidence skips logprob gate when thinking present ──


def _make_agent_with_threshold(threshold: float = 0.7) -> Agent:
    a = Agent.__new__(Agent)
    a.confidence_threshold = threshold
    a.staleness_days = 7
    a.gsa_enabled = False
    a._db_path = ":memory:"
    a._conn = init_database(":memory:")
    a._config = AutodidactConfig(db_path=":memory:", embedding_dim=32)
    a.memory = KnowledgeStore(a._conn, a._config)
    a._local_client = MagicMock(spec=LLMClient)
    a._local_client.config = LLMConfig(provider="ollama", model="qwen3:8b")
    a._cloud_client = MagicMock(spec=LLMClient)
    a._embed_client = a._local_client
    a._local_model_name = "ollama/qwen3:8b"
    a._cloud_model_name = "openai/gpt-4o"
    a._session_stats = SavingsReport()
    a._history = []
    a.documents = None
    a._gsa = None
    return a


class TestComputeConfidence:

    def test_non_thinking_uses_logprob_normally(self):
        """When had_thinking is False, the existing sigmoid mapping applies."""
        agent = _make_agent_with_threshold()
        resp = ChatResponseWithLogprobs(
            content="hi", model="x",
            avg_logprob=-0.13,  # high confidence after sigmoid
            had_thinking=False,
        )
        # sigmoid(2*(-0.13) + 3) = sigmoid(2.74) ≈ 0.94
        confidence = agent._compute_confidence(resp)
        assert confidence > 0.7

    def test_thinking_response_returns_max_confidence(self):
        """When had_thinking is True, return 1.0 — the gate is skipped.

        The thinking-token logprobs would otherwise drag the average below
        threshold even on perfect answers.
        """
        agent = _make_agent_with_threshold()
        resp = ChatResponseWithLogprobs(
            content="The capital of France is Paris.",
            model="qwen3:8b",
            avg_logprob=-1.5,  # would normally yield confidence ~0.5
            had_thinking=True,
        )
        confidence = agent._compute_confidence(resp)
        assert confidence == 1.0, (
            "Thinking responses must bypass the logprob gate by returning "
            "max confidence; got %r" % confidence
        )

    def test_no_logprobs_still_neutral(self):
        """Neutral fallback for missing logprobs is unchanged (0.5)."""
        agent = _make_agent_with_threshold()
        resp = ChatResponseWithLogprobs(
            content="hi", model="x", avg_logprob=None, had_thinking=False,
        )
        assert agent._compute_confidence(resp) == 0.5


# ── Streaming parser sets had_thinking ────────────────────────────


class TestStreamingSetsHadThinking:
    """The Ollama streaming consumer flags had_thinking when it sees thinking deltas."""

    def test_thinking_chunks_set_flag(self):
        from autodidact.llm_client import _consume_ollama_stream
        import json
        import time

        chunks = [
            {"message": {"thinking": "let me think...", "content": ""}, "done": False},
            {"message": {"thinking": "", "content": "Paris."}, "done": False},
            {"message": {"content": ""}, "done": True,
             "prompt_eval_count": 5, "eval_count": 2},
        ]
        resp = MagicMock(status_code=200)
        resp.iter_lines.return_value = [json.dumps(c).encode() for c in chunks]

        result = _consume_ollama_stream(
            resp, on_token=lambda _: None,
            fallback_model="qwen3:8b", started=time.perf_counter(),
        )
        assert result.had_thinking is True

    def test_no_thinking_chunks_leaves_flag_false(self):
        from autodidact.llm_client import _consume_ollama_stream
        import json
        import time

        chunks = [
            {"message": {"content": "Paris."}, "done": False},
            {"message": {"content": ""}, "done": True,
             "prompt_eval_count": 5, "eval_count": 1},
        ]
        resp = MagicMock(status_code=200)
        resp.iter_lines.return_value = [json.dumps(c).encode() for c in chunks]

        result = _consume_ollama_stream(
            resp, on_token=lambda _: None,
            fallback_model="qwen2.5:7b", started=time.perf_counter(),
        )
        assert result.had_thinking is False


# ── Non-streaming Ollama parser sets had_thinking ─────────────────


class TestNonStreamingSetsHadThinking:
    """The non-streaming _chat_ollama_with_logprobs flags had_thinking too."""

    def test_thinking_field_in_response_sets_flag(self):
        # We test by mocking the HTTP response inside an LLMClient.
        from unittest.mock import patch
        client = LLMClient(LLMConfig(provider="ollama", model="qwen3:8b"))

        fake_data = {
            "message": {
                "content": "Paris.",
                "thinking": "Let me consider...",
            },
            "model": "qwen3:8b",
            "prompt_eval_count": 5,
            "eval_count": 1,
            "logprobs": [{"token": "Paris", "logprob": -0.1, "top_logprobs": []}],
        }
        with patch.object(client, "_ollama_post", return_value=fake_data):
            result = client.chat_with_logprobs(
                [ChatMessage(role="user", content="capital?")],
            )
        assert result.had_thinking is True

    def test_no_thinking_field_keeps_flag_false(self):
        from unittest.mock import patch
        client = LLMClient(LLMConfig(provider="ollama", model="qwen2.5:7b"))

        fake_data = {
            "message": {"content": "Paris."},
            "model": "qwen2.5:7b",
            "prompt_eval_count": 5,
            "eval_count": 1,
        }
        with patch.object(client, "_ollama_post", return_value=fake_data):
            result = client.chat_with_logprobs(
                [ChatMessage(role="user", content="capital?")],
            )
        assert result.had_thinking is False

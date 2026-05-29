"""Tests for the typed ProgressEvent protocol.

The agent's on_progress callback used to receive untyped dicts:
``{"type": "thinking", "memory_hits": N, ...}``. Each consumer parsed
the magic ``"type"`` key and pulled fields by string. Adding a new
event meant editing every consumer with no static-analysis safety net.

Replaced with a tagged union of small dataclasses. Each event has its
own typed shape; consumers pattern-match (``isinstance``) and read
attributes. Static analysis catches "consumer doesn't handle the new
event" and "agent emits an unknown field" — both previously silent
bugs.
"""

from __future__ import annotations

from dataclasses import is_dataclass

import pytest


# ── Event shapes ────────────────────────────────────────────────


class TestEventShapes:
    """Each event type is a frozen dataclass with documented fields."""

    def test_thinking_event(self):
        from autodidact.events import ThinkingEvent

        e = ThinkingEvent(memory_hits=3, best_similarity=0.42)
        assert e.memory_hits == 3
        assert e.best_similarity == pytest.approx(0.42)
        assert is_dataclass(e)

    def test_memory_hit_event(self):
        from autodidact.events import MemoryHitEvent

        e = MemoryHitEvent(
            similarity=0.9, memory_source="What's the capital?", age_days=2.5,
        )
        assert e.similarity == pytest.approx(0.9)
        assert e.memory_source == "What's the capital?"
        assert e.age_days == pytest.approx(2.5)

    def test_gsa_check_event(self):
        from autodidact.events import GsaCheckEvent

        e = GsaCheckEvent()
        assert is_dataclass(e)

    def test_local_done_event(self):
        from autodidact.events import LocalDoneEvent

        e = LocalDoneEvent(confidence=1.0)
        assert e.confidence == pytest.approx(1.0)

    def test_cloud_call_event(self):
        from autodidact.events import CloudCallEvent

        e = CloudCallEvent(model="gpt-4o")
        assert e.model == "gpt-4o"

    def test_cloud_done_event(self):
        from autodidact.events import CloudDoneEvent

        e = CloudDoneEvent(model="gpt-4o", cost=0.025, latency_ms=400)
        assert e.model == "gpt-4o"
        assert e.cost == pytest.approx(0.025)
        assert e.latency_ms == 400

    def test_token_event_local_content(self):
        from autodidact.events import TokenEvent

        e = TokenEvent(source="local", phase="content", text="Paris")
        assert e.source == "local"
        assert e.phase == "content"
        assert e.text == "Paris"

    def test_token_event_cloud_thinking(self):
        from autodidact.events import TokenEvent

        e = TokenEvent(source="cloud", phase="thinking", text="reasoning...")
        assert e.source == "cloud"
        assert e.phase == "thinking"

    def test_progress_event_union(self):
        """All event types should be members of the ProgressEvent union."""
        from autodidact.events import (
            CloudCallEvent,
            CloudDoneEvent,
            GsaCheckEvent,
            LocalDoneEvent,
            MemoryHitEvent,
            ProgressEvent,
            ThinkingEvent,
            TokenEvent,
        )

        events: list[ProgressEvent] = [
            ThinkingEvent(memory_hits=0, best_similarity=0.0),
            MemoryHitEvent(similarity=0.9, memory_source="x", age_days=0.0),
            GsaCheckEvent(),
            LocalDoneEvent(confidence=1.0),
            CloudCallEvent(model="x"),
            CloudDoneEvent(model="x", cost=0.0, latency_ms=0),
            TokenEvent(source="local", phase="content", text="x"),
        ]
        # Just constructing the list is the test — types are proven by mypy
        # if we run it; at runtime we verify each is_dataclass.
        for ev in events:
            assert is_dataclass(ev)


# ── Pattern matching ────────────────────────────────────────────


class TestEventPatternMatching:
    """Consumers should branch on isinstance(), not string keys."""

    def test_isinstance_branching_works(self):
        from autodidact.events import (
            CloudCallEvent,
            ThinkingEvent,
            TokenEvent,
        )

        events = [
            ThinkingEvent(memory_hits=2, best_similarity=0.5),
            TokenEvent(source="local", phase="content", text="hi"),
            CloudCallEvent(model="gpt-4o"),
        ]

        thinking_count = sum(1 for e in events if isinstance(e, ThinkingEvent))
        token_count = sum(1 for e in events if isinstance(e, TokenEvent))
        cloud_call_count = sum(1 for e in events if isinstance(e, CloudCallEvent))

        assert thinking_count == 1
        assert token_count == 1
        assert cloud_call_count == 1


# ── Frozen / immutable ──────────────────────────────────────────


class TestEventsAreFrozen:
    """Events are value objects; mutation is a code smell."""

    def test_thinking_event_is_frozen(self):
        from autodidact.events import ThinkingEvent
        from dataclasses import FrozenInstanceError

        e = ThinkingEvent(memory_hits=1, best_similarity=0.5)
        with pytest.raises(FrozenInstanceError):
            e.memory_hits = 999  # type: ignore[misc]


# ── Agent emits typed events ────────────────────────────────────


class TestAgentEmitsTypedEvents:
    """End-to-end: Agent.query()'s on_progress callback receives ProgressEvent
    instances, not dicts.

    This covers the key API contract for #4. The big behavioural coverage of
    Agent routing lives elsewhere (test_agent.py, test_progress_callbacks.py,
    test_streaming_agent.py); those will be migrated to assert on typed
    events as part of this refactor.
    """

    def _make_agent_with_local_only(self, tmp_path):
        from unittest.mock import MagicMock
        from autodidact.agent import Agent
        from autodidact.llm_client import (
            ChatResponse, ChatResponseWithLogprobs, LLMClient, LLMConfig,
        )

        a = Agent.__new__(Agent)
        a._db_path = ":memory:"
        from autodidact.database import init_database
        from autodidact.knowledge_store import KnowledgeStore
        from autodidact.types import AutodidactConfig
        a._conn = init_database(":memory:")
        a._config = AutodidactConfig(db_path=":memory:", embedding_dim=32)
        a.memory = KnowledgeStore(a._conn, a._config)
        a.staleness_days = 7
        a.gsa_enabled = False  # keep it simple
        a.gsa_threshold = 0.55
        a.confidence_threshold = 0.7

        local = MagicMock(spec=LLMClient)
        local.config = LLMConfig(provider="ollama", model="qwen3:8b")
        import numpy as np
        local.embed.return_value = np.zeros(32, dtype=np.float32)

        def fake_local_stream(messages, *, on_token, **opts):
            on_token({"phase": "content", "text": "Paris."})
            return ChatResponse(content="Paris.", model="qwen3:8b")
        local.chat_stream_ollama_no_logprobs = MagicMock(side_effect=fake_local_stream)

        a._local_client = local
        a._cloud_client = None
        a._embed_client = local
        a._local_model_name = "ollama/qwen3:8b"
        a._cloud_model_name = None

        from autodidact.agent import SavingsReport
        a._session_stats = SavingsReport()
        a._history = []
        a.documents = None
        a._gsa = None
        a._query_stages = None
        a._correct_stages = None
        return a

    def test_query_emits_thinking_event(self, tmp_path):
        from autodidact.events import ThinkingEvent

        agent = self._make_agent_with_local_only(tmp_path)
        events = []
        agent.query("hello", on_progress=events.append)

        thinking = [e for e in events if isinstance(e, ThinkingEvent)]
        assert len(thinking) == 1

    def test_query_emits_local_done_event(self, tmp_path):
        from autodidact.events import LocalDoneEvent

        agent = self._make_agent_with_local_only(tmp_path)
        events = []
        agent.query("hello", on_progress=events.append)

        local_done = [e for e in events if isinstance(e, LocalDoneEvent)]
        assert len(local_done) == 1
        assert local_done[0].confidence == pytest.approx(1.0)

    def test_query_emits_token_events(self, tmp_path):
        from autodidact.events import TokenEvent

        agent = self._make_agent_with_local_only(tmp_path)
        events = []
        agent.query("hello", on_progress=events.append)

        tokens = [e for e in events if isinstance(e, TokenEvent)]
        assert len(tokens) >= 1
        assert tokens[0].source == "local"
        assert tokens[0].phase == "content"
        assert tokens[0].text == "Paris."

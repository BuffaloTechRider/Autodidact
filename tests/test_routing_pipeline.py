"""Tests for the RoutingStage pipeline.

The Agent's query() and correct() methods are now composers over a list of
stages. This test suite verifies stages in isolation (cheap to set up) and
the composer (which hand-rolled stage lists exercise the pipeline plumbing).

What the existing test_agent.py / test_progress_callbacks.py / etc. cover:
end-to-end Agent.query() behavior. Those tests stay green and are the
regression guard. This file exercises the *new* pieces — the stage
interface, RoutingState, and the composer — directly.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Optional
from unittest.mock import MagicMock

import numpy as np
import pytest


# ── RoutingState basics ──────────────────────────────────────────


class TestRoutingState:
    """RoutingState carries shared mutable state between stages."""

    def test_construct_with_required_fields(self):
        from autodidact.routing import RoutingState

        emitted: list = []
        state = RoutingState(
            question="hi",
            context=None,
            started=time.perf_counter(),
            emit=emitted.append,
        )
        assert state.question == "hi"
        assert state.memory_hits == []
        assert state.q_emb is None
        assert state.best_hit is None
        assert state.gsa_p_yes is None
        assert state.refused is False
        assert state.has_doc_context is False
        assert state.local_response is None

    def test_emit_is_callable(self):
        from autodidact.routing import RoutingState

        events: list = []
        state = RoutingState(
            question="hi", context=None, started=0.0, emit=events.append,
        )
        state.emit({"type": "test", "val": 1})
        assert events == [{"type": "test", "val": 1}]


# ── Outcome types ────────────────────────────────────────────────


class TestStageOutcomes:
    def test_resolved_carries_response(self):
        from autodidact.agent import QueryResponse
        from autodidact.routing import Resolved

        resp = QueryResponse(
            answer="x", routed_to="local", confidence=1.0, cost_usd=0.0,
            learned=False, latency_ms=10,
        )
        out = Resolved(response=resp)
        assert out.response is resp

    def test_continue_is_constructible(self):
        from autodidact.routing import Continue

        out = Continue()
        # Continues should be cheap singletons / sentinels — no required state.
        assert out is not None


# ── Pipeline composer ───────────────────────────────────────────


class TestPipelineComposer:
    """run_pipeline walks stages until one returns Resolved."""

    def test_first_resolving_stage_wins(self):
        from autodidact.agent import QueryResponse
        from autodidact.routing import Continue, Resolved, RoutingState, run_pipeline

        resp = QueryResponse(
            answer="from-stage-1", routed_to="local", confidence=1.0,
            cost_usd=0.0, learned=False, latency_ms=10,
        )

        def stage1(state):
            return Resolved(response=resp)

        def stage2(state):
            pytest.fail("stage2 must not run after stage1 resolves")

        state = RoutingState(question="q", context=None, started=0.0, emit=lambda e: None)
        out = run_pipeline([stage1, stage2], state)
        assert out is resp

    def test_continue_advances_to_next_stage(self):
        from autodidact.agent import QueryResponse
        from autodidact.routing import Continue, Resolved, RoutingState, run_pipeline

        calls: list[str] = []
        resp = QueryResponse(
            answer="from-stage-2", routed_to="local", confidence=1.0,
            cost_usd=0.0, learned=False, latency_ms=10,
        )

        def stage1(state):
            calls.append("s1")
            return Continue()

        def stage2(state):
            calls.append("s2")
            return Resolved(response=resp)

        state = RoutingState(question="q", context=None, started=0.0, emit=lambda e: None)
        out = run_pipeline([stage1, stage2], state)
        assert calls == ["s1", "s2"]
        assert out is resp

    def test_pipeline_with_no_resolution_raises(self):
        from autodidact.routing import Continue, RoutingState, run_pipeline

        def stage(state):
            return Continue()

        state = RoutingState(question="q", context=None, started=0.0, emit=lambda e: None)
        with pytest.raises(RuntimeError, match="did not produce a response"):
            run_pipeline([stage, stage], state)

    def test_state_is_threaded_through_stages(self):
        """Mutations made by stage N are visible to stage N+1."""
        from autodidact.agent import QueryResponse
        from autodidact.routing import Continue, Resolved, RoutingState, run_pipeline

        def stage1(state):
            state.gsa_p_yes = 0.42
            return Continue()

        def stage2(state):
            assert state.gsa_p_yes == 0.42
            return Resolved(response=QueryResponse(
                answer="ok", routed_to="local", confidence=1.0, cost_usd=0.0,
                learned=False, latency_ms=0,
            ))

        state = RoutingState(question="q", context=None, started=0.0, emit=lambda e: None)
        out = run_pipeline([stage1, stage2], state)
        assert out.answer == "ok"


# ── MemoryStage ─────────────────────────────────────────────────


@dataclass
class _FakeMemoryDeps:
    """A minimal stand-in for what MemoryStage actually needs from Agent.

    Threading dependencies explicitly per Q1's decision — no Agent reference.
    """
    check_memory_fn: Callable
    knowledge_store_access: Callable
    entry_age_fn: Callable
    staleness_days: float
    has_local_client: bool
    build_messages_fn: Callable
    call_local_fn: Callable
    record_query_fn: Callable
    append_history_fn: Callable


class TestMemoryStage:
    """MemoryStage handles the Stage 1 memory-check logic."""

    def _make_state(self):
        from autodidact.routing import RoutingState
        events: list = []
        state = RoutingState(
            question="What is the capital of France?",
            context=None,
            started=time.perf_counter(),
            emit=events.append,
        )
        # Hand the events list back via emit closure for assertions.
        state._captured_events = events  # type: ignore[attr-defined]
        return state

    def _make_hit(self, *, score: float, entry_id: str = "e1", question: str = "old?",
                  content: str = "old answer", age_days: float = 1.0):
        entry = MagicMock()
        entry.id = entry_id
        entry.question = question
        entry.content = content
        entry.created_at = "2026-01-01T00:00:00Z"
        hit = MagicMock()
        hit.entry = entry
        hit.score = score
        return hit, age_days

    def test_emits_thinking_event(self):
        from autodidact.routing.stages import MemoryStage

        hit, age = self._make_hit(score=0.5)
        deps = _FakeMemoryDeps(
            check_memory_fn=lambda q: ([hit], np.zeros(4)),
            knowledge_store_access=lambda eid: None,
            entry_age_fn=lambda e: age,
            staleness_days=30,
            has_local_client=True,
            build_messages_fn=lambda *a, **k: ([], []),
            call_local_fn=lambda *a, **k: None,
            record_query_fn=lambda *a, **k: None,
            append_history_fn=lambda *a, **k: None,
        )
        stage = MemoryStage(deps)
        state = self._make_state()
        stage(state)
        events = state._captured_events  # type: ignore[attr-defined]
        thinking = [e for e in events if e["type"] == "thinking"]
        assert len(thinking) == 1
        assert thinking[0]["memory_hits"] == 1
        assert thinking[0]["best_similarity"] == 0.5

    def test_populates_state_with_hits(self):
        from autodidact.routing.stages import MemoryStage

        hit, age = self._make_hit(score=0.5)
        q_emb = np.array([1.0, 2.0, 3.0])
        deps = _FakeMemoryDeps(
            check_memory_fn=lambda q: ([hit], q_emb),
            knowledge_store_access=lambda eid: None,
            entry_age_fn=lambda e: age,
            staleness_days=30,
            has_local_client=True,
            build_messages_fn=lambda *a, **k: ([], []),
            call_local_fn=lambda *a, **k: None,
            record_query_fn=lambda *a, **k: None,
            append_history_fn=lambda *a, **k: None,
        )
        state = self._make_state()
        MemoryStage(deps)(state)
        assert state.memory_hits == [hit]
        assert state.best_hit is hit
        assert state.q_emb is q_emb

    def test_resolves_on_fresh_direct_hit(self):
        """score >= 0.80 + age <= staleness + has local => Resolved with routed_to='memory'."""
        from autodidact.routing import Resolved
        from autodidact.routing.stages import MemoryStage

        hit, age = self._make_hit(score=0.95, age_days=1.0)
        local_resp = MagicMock()
        local_resp.content = "Paris."

        accessed: list = []
        recorded: list = []
        appended: list = []
        deps = _FakeMemoryDeps(
            check_memory_fn=lambda q: ([hit], np.zeros(4)),
            knowledge_store_access=lambda eid: accessed.append(eid),
            entry_age_fn=lambda e: age,
            staleness_days=30,
            has_local_client=True,
            build_messages_fn=lambda q, ctx, hits: (["msg"], ["src"]),
            call_local_fn=lambda messages, emit: local_resp,
            record_query_fn=lambda *a, **k: recorded.append(a),
            append_history_fn=lambda q, a: appended.append((q, a)),
        )
        state = self._make_state()
        out = MemoryStage(deps)(state)

        assert isinstance(out, Resolved)
        assert out.response.routed_to == "memory"
        assert out.response.answer == "Paris."
        assert out.response.confidence == pytest.approx(0.95)
        assert out.response.cost_usd == 0.0
        assert out.response.learned is False
        assert out.response.memory_source == "old?"
        assert out.response.context_sources == ["src"]
        assert accessed == ["e1"]
        # Side effects on the question's lifecycle
        assert appended and appended[0][0] == state.question
        # memory_hit event emitted
        events = state._captured_events  # type: ignore[attr-defined]
        assert any(e["type"] == "memory_hit" for e in events)

    def test_continues_on_no_hits(self):
        from autodidact.routing import Continue
        from autodidact.routing.stages import MemoryStage

        deps = _FakeMemoryDeps(
            check_memory_fn=lambda q: ([], None),
            knowledge_store_access=lambda eid: None,
            entry_age_fn=lambda e: 0.0,
            staleness_days=30,
            has_local_client=True,
            build_messages_fn=lambda *a, **k: ([], []),
            call_local_fn=lambda *a, **k: None,
            record_query_fn=lambda *a, **k: None,
            append_history_fn=lambda *a, **k: None,
        )
        state = self._make_state()
        out = MemoryStage(deps)(state)
        assert isinstance(out, Continue)

    def test_continues_on_low_score_hit(self):
        from autodidact.routing import Continue
        from autodidact.routing.stages import MemoryStage

        hit, _ = self._make_hit(score=0.5)
        deps = _FakeMemoryDeps(
            check_memory_fn=lambda q: ([hit], np.zeros(4)),
            knowledge_store_access=lambda eid: None,
            entry_age_fn=lambda e: 0.0,
            staleness_days=30,
            has_local_client=True,
            build_messages_fn=lambda *a, **k: ([], []),
            call_local_fn=lambda *a, **k: None,
            record_query_fn=lambda *a, **k: None,
            append_history_fn=lambda *a, **k: None,
        )
        state = self._make_state()
        out = MemoryStage(deps)(state)
        assert isinstance(out, Continue)
        # Best hit is preserved on state for downstream stages.
        assert state.best_hit is hit

    def test_continues_on_stale_hit(self):
        """Direct hit but too old → fall through, no memory_hit event."""
        from autodidact.routing import Continue
        from autodidact.routing.stages import MemoryStage

        hit, _ = self._make_hit(score=0.95)
        deps = _FakeMemoryDeps(
            check_memory_fn=lambda q: ([hit], np.zeros(4)),
            knowledge_store_access=lambda eid: None,
            entry_age_fn=lambda e: 100.0,  # very old
            staleness_days=30,
            has_local_client=True,
            build_messages_fn=lambda *a, **k: ([], []),
            call_local_fn=lambda *a, **k: None,
            record_query_fn=lambda *a, **k: None,
            append_history_fn=lambda *a, **k: None,
        )
        state = self._make_state()
        out = MemoryStage(deps)(state)
        assert isinstance(out, Continue)
        events = state._captured_events  # type: ignore[attr-defined]
        assert not any(e["type"] == "memory_hit" for e in events)

    def test_continues_when_no_local_client_even_on_direct_hit(self):
        """If there's no local client to render the answer, fall through.

        This preserves the quirk in the v1.0.6 code at agent.py:286: the
        memory_hit event still fires (already happened), but the early
        return is gated on _local_client.
        """
        from autodidact.routing import Continue
        from autodidact.routing.stages import MemoryStage

        hit, age = self._make_hit(score=0.95)
        deps = _FakeMemoryDeps(
            check_memory_fn=lambda q: ([hit], np.zeros(4)),
            knowledge_store_access=lambda eid: None,
            entry_age_fn=lambda e: age,
            staleness_days=30,
            has_local_client=False,  # ← key
            build_messages_fn=lambda *a, **k: ([], []),
            call_local_fn=lambda *a, **k: None,
            record_query_fn=lambda *a, **k: None,
            append_history_fn=lambda *a, **k: None,
        )
        state = self._make_state()
        out = MemoryStage(deps)(state)
        assert isinstance(out, Continue)


# ── GsaPreGateStage ─────────────────────────────────────────────


@dataclass
class _FakeGsaDeps:
    gsa_enabled: bool
    gsa_threshold: float
    has_local_client: bool
    has_cloud_client: bool
    document_search_fn: Optional[Callable]  # None when no doc store
    build_gsa_fn: Callable                   # () -> SelfAssessment-like
    escalate_fn: Callable                    # for the early-return path


class TestGsaPreGateStage:
    """GsaPreGateStage gates local generation on a YES/NO probe."""

    def _make_state_with_memory(self, *, best_score: float = 0.0):
        from autodidact.routing import RoutingState
        events: list = []
        state = RoutingState(
            question="q?", context=None, started=time.perf_counter(),
            emit=events.append,
        )
        state._captured_events = events  # type: ignore[attr-defined]
        if best_score > 0:
            hit = MagicMock()
            hit.score = best_score
            state.memory_hits = [hit]
            state.best_hit = hit
        return state

    def _gsa_returning(self, p_yes: float):
        gsa = MagicMock()
        result = MagicMock()
        result.p_yes = p_yes
        gsa.compute.return_value = result
        return gsa

    def test_skips_when_disabled(self):
        from autodidact.routing import Continue
        from autodidact.routing.stages import GsaPreGateStage

        deps = _FakeGsaDeps(
            gsa_enabled=False, gsa_threshold=0.55,
            has_local_client=True, has_cloud_client=True,
            document_search_fn=None,
            build_gsa_fn=lambda: pytest.fail("must not build GSA when disabled"),
            escalate_fn=lambda *a, **k: pytest.fail("must not escalate"),
        )
        state = self._make_state_with_memory()
        out = GsaPreGateStage(deps)(state)
        assert isinstance(out, Continue)
        assert state.gsa_p_yes is None

    def test_skips_without_cloud_client(self):
        from autodidact.routing import Continue
        from autodidact.routing.stages import GsaPreGateStage

        deps = _FakeGsaDeps(
            gsa_enabled=True, gsa_threshold=0.55,
            has_local_client=True, has_cloud_client=False,
            document_search_fn=None,
            build_gsa_fn=lambda: pytest.fail("must not build GSA without cloud target"),
            escalate_fn=lambda *a, **k: pytest.fail("must not escalate"),
        )
        state = self._make_state_with_memory()
        out = GsaPreGateStage(deps)(state)
        assert isinstance(out, Continue)

    def test_skips_when_strong_doc_context_exists(self):
        """If docs have a hit ≥ 0.75 the local model has what it needs — skip the probe."""
        from autodidact.routing import Continue
        from autodidact.routing.stages import GsaPreGateStage

        doc_hit = MagicMock()
        doc_hit.score = 0.85
        deps = _FakeGsaDeps(
            gsa_enabled=True, gsa_threshold=0.55,
            has_local_client=True, has_cloud_client=True,
            document_search_fn=lambda q, q_emb: [doc_hit],
            build_gsa_fn=lambda: pytest.fail("must not build GSA when docs cover it"),
            escalate_fn=lambda *a, **k: pytest.fail("must not escalate"),
        )
        state = self._make_state_with_memory()
        out = GsaPreGateStage(deps)(state)
        assert isinstance(out, Continue)
        assert state.has_doc_context is True

    def test_continues_on_high_p_yes(self):
        from autodidact.routing import Continue
        from autodidact.routing.stages import GsaPreGateStage

        gsa = self._gsa_returning(p_yes=0.92)
        deps = _FakeGsaDeps(
            gsa_enabled=True, gsa_threshold=0.55,
            has_local_client=True, has_cloud_client=True,
            document_search_fn=None,
            build_gsa_fn=lambda: gsa,
            escalate_fn=lambda *a, **k: pytest.fail("must not escalate on high p_yes"),
        )
        state = self._make_state_with_memory()
        out = GsaPreGateStage(deps)(state)
        assert isinstance(out, Continue)
        assert state.gsa_p_yes == 0.92
        events = state._captured_events  # type: ignore[attr-defined]
        assert any(e["type"] == "gsa_check" for e in events)

    def test_resolves_on_low_p_yes(self):
        """Low p_yes → escalate to cloud, response gets escalated_on_gsa=True."""
        from autodidact.agent import QueryResponse
        from autodidact.routing import Resolved
        from autodidact.routing.stages import GsaPreGateStage

        gsa = self._gsa_returning(p_yes=0.10)
        cloud_resp = QueryResponse(
            answer="cloud says", routed_to="cloud", confidence=0.0,
            cost_usd=0.01, learned=True, latency_ms=200,
        )
        deps = _FakeGsaDeps(
            gsa_enabled=True, gsa_threshold=0.55,
            has_local_client=True, has_cloud_client=True,
            document_search_fn=None,
            build_gsa_fn=lambda: gsa,
            escalate_fn=lambda state: cloud_resp,
        )
        state = self._make_state_with_memory()
        out = GsaPreGateStage(deps)(state)
        assert isinstance(out, Resolved)
        assert out.response.routed_to == "cloud"
        assert out.response.escalated_on_gsa is True
        assert out.response.gsa_p_yes == pytest.approx(0.10)
        assert state.gsa_p_yes == pytest.approx(0.10)

    def test_continues_when_probe_raises(self):
        """A flaky GSA must not block the query."""
        from autodidact.routing import Continue
        from autodidact.routing.stages import GsaPreGateStage

        gsa = MagicMock()
        gsa.compute.side_effect = RuntimeError("probe died")
        deps = _FakeGsaDeps(
            gsa_enabled=True, gsa_threshold=0.55,
            has_local_client=True, has_cloud_client=True,
            document_search_fn=None,
            build_gsa_fn=lambda: gsa,
            escalate_fn=lambda *a, **k: pytest.fail("must not escalate after probe failure"),
        )
        state = self._make_state_with_memory()
        out = GsaPreGateStage(deps)(state)
        assert isinstance(out, Continue)
        assert state.gsa_p_yes is None

    def test_skips_when_memory_was_strong(self):
        """If Stage 1 saw ≥ MEMORY_DIRECT_THRESHOLD but fell through (stale), skip GSA."""
        from autodidact.routing import Continue
        from autodidact.routing.stages import GsaPreGateStage

        deps = _FakeGsaDeps(
            gsa_enabled=True, gsa_threshold=0.55,
            has_local_client=True, has_cloud_client=True,
            document_search_fn=None,
            build_gsa_fn=lambda: pytest.fail("must not run GSA after strong memory"),
            escalate_fn=lambda *a, **k: pytest.fail("must not escalate"),
        )
        state = self._make_state_with_memory(best_score=0.95)  # strong hit fell through (stale)
        out = GsaPreGateStage(deps)(state)
        assert isinstance(out, Continue)


# ── LocalGenerationStage ────────────────────────────────────────


@dataclass
class _FakeLocalDeps:
    has_local_client: bool
    has_cloud_client: bool
    build_messages_fn: Callable
    call_local_fn: Callable
    refusal_detector: Callable
    record_query_fn: Callable
    append_history_fn: Callable


class TestLocalGenerationStage:
    def _state(self):
        from autodidact.routing import RoutingState
        events: list = []
        state = RoutingState(
            question="q?", context=None, started=time.perf_counter(),
            emit=events.append,
        )
        state._captured_events = events  # type: ignore[attr-defined]
        return state

    def test_resolves_on_successful_local(self):
        from autodidact.routing import Resolved
        from autodidact.routing.stages import LocalGenerationStage

        local_resp = MagicMock(content="Paris.")
        recorded: list = []
        appended: list = []

        deps = _FakeLocalDeps(
            has_local_client=True, has_cloud_client=True,
            build_messages_fn=lambda q, ctx, hits: (["msg"], ["src1"]),
            call_local_fn=lambda messages, emit: local_resp,
            refusal_detector=lambda text: False,
            record_query_fn=lambda *a, **k: recorded.append(a),
            append_history_fn=lambda q, a: appended.append((q, a)),
        )
        state = self._state()
        out = LocalGenerationStage(deps)(state)
        assert isinstance(out, Resolved)
        assert out.response.routed_to == "local"
        assert out.response.confidence == 1.0
        assert out.response.cost_usd == 0.0
        assert out.response.learned is False
        assert out.response.context_sources == ["src1"]
        assert appended == [("q?", "Paris.")]
        events = state._captured_events  # type: ignore[attr-defined]
        assert any(e["type"] == "local_done" and e["confidence"] == 1.0 for e in events)

    def test_continues_on_refusal_so_cloud_can_handle(self):
        from autodidact.routing import Continue
        from autodidact.routing.stages import LocalGenerationStage

        local_resp = MagicMock(content="I don't have real-time access.")
        deps = _FakeLocalDeps(
            has_local_client=True, has_cloud_client=True,
            build_messages_fn=lambda q, ctx, hits: ([], []),
            call_local_fn=lambda messages, emit: local_resp,
            refusal_detector=lambda text: True,
            record_query_fn=lambda *a, **k: None,
            append_history_fn=lambda *a, **k: None,
        )
        state = self._state()
        out = LocalGenerationStage(deps)(state)
        assert isinstance(out, Continue)
        assert state.refused is True
        assert state.local_response is local_resp

    def test_resolves_no_model_canned_response_when_neither_client(self):
        """No local + no cloud → return the canned 'No model configured' response."""
        from autodidact.routing import Resolved
        from autodidact.routing.stages import LocalGenerationStage

        deps = _FakeLocalDeps(
            has_local_client=False, has_cloud_client=False,
            build_messages_fn=lambda *a, **k: ([], []),
            call_local_fn=lambda *a, **k: pytest.fail("must not call local without client"),
            refusal_detector=lambda *a: False,
            record_query_fn=lambda *a, **k: None,
            append_history_fn=lambda *a, **k: None,
        )
        state = self._state()
        out = LocalGenerationStage(deps)(state)
        assert isinstance(out, Resolved)
        assert "No model configured" in out.response.answer
        assert out.response.routed_to == "local"

    def test_continues_when_no_local_but_cloud_available(self):
        """No local but has cloud → continue so CloudEscalationStage handles it."""
        from autodidact.routing import Continue
        from autodidact.routing.stages import LocalGenerationStage

        deps = _FakeLocalDeps(
            has_local_client=False, has_cloud_client=True,
            build_messages_fn=lambda *a, **k: ([], []),
            call_local_fn=lambda *a, **k: pytest.fail("must not call local"),
            refusal_detector=lambda *a: False,
            record_query_fn=lambda *a, **k: None,
            append_history_fn=lambda *a, **k: None,
        )
        state = self._state()
        out = LocalGenerationStage(deps)(state)
        assert isinstance(out, Continue)


# ── CloudEscalationStage ────────────────────────────────────────


@dataclass
class _FakeCloudDeps:
    has_cloud_client: bool
    escalate_fn: Callable                  # builds and returns a QueryResponse
    record_query_fn: Callable              # used by no-cloud fallback
    append_history_fn: Callable            # used by no-cloud fallback


class TestCloudEscalationStage:
    """Cloud escalation handles the full-cloud path AND the no-cloud refusal fallback."""

    def _state(self, *, refused: bool = False, gsa_p_yes: Optional[float] = None,
               local_content: Optional[str] = None):
        from autodidact.routing import RoutingState
        events: list = []
        state = RoutingState(
            question="q?", context=None, started=time.perf_counter(),
            emit=events.append,
        )
        state._captured_events = events  # type: ignore[attr-defined]
        state.refused = refused
        state.gsa_p_yes = gsa_p_yes
        if local_content is not None:
            state.local_response = MagicMock(content=local_content)
        return state

    def test_resolves_via_escalate_in_normal_path(self):
        from autodidact.agent import QueryResponse
        from autodidact.routing import Resolved
        from autodidact.routing.stages import CloudEscalationStage

        captured = {}

        def escalate(state):
            captured["state"] = state
            return QueryResponse(
                answer="cloud answer", routed_to="cloud", confidence=0.0,
                cost_usd=0.02, learned=True, latency_ms=300,
                escalated_on_refusal=state.refused,
            )

        deps = _FakeCloudDeps(
            has_cloud_client=True,
            escalate_fn=escalate,
            record_query_fn=lambda *a, **k: None,
            append_history_fn=lambda *a, **k: None,
        )
        state = self._state(refused=True, gsa_p_yes=0.4)
        out = CloudEscalationStage(deps, force=False)(state)
        assert isinstance(out, Resolved)
        assert out.response.escalated_on_refusal is True
        assert out.response.gsa_p_yes == 0.4

    def test_resolves_with_no_cloud_fallback_returning_local_text(self):
        """No cloud + Stage 2 produced a refused local answer → return the local text."""
        from autodidact.routing import Resolved
        from autodidact.routing.stages import CloudEscalationStage

        recorded: list = []
        appended: list = []

        deps = _FakeCloudDeps(
            has_cloud_client=False,
            escalate_fn=lambda *a, **k: pytest.fail("must not call escalate without cloud"),
            record_query_fn=lambda *a, **k: recorded.append(a),
            append_history_fn=lambda q, a: appended.append((q, a)),
        )
        state = self._state(refused=True, gsa_p_yes=0.30,
                            local_content="I'm not sure, but Paris maybe.")
        out = CloudEscalationStage(deps, force=False)(state)
        assert isinstance(out, Resolved)
        assert out.response.routed_to == "local"
        assert out.response.confidence == 0.5
        assert out.response.gsa_p_yes == 0.30
        assert "Paris" in out.response.answer
        # Event protocol: emits local_done with confidence=0.5 on the fallback.
        events = state._captured_events  # type: ignore[attr-defined]
        local_done = [e for e in events if e["type"] == "local_done"]
        assert local_done and local_done[0]["confidence"] == 0.5

    def test_force_mode_escalates_unconditionally(self):
        """In force=True (correct() path) the stage escalates without the refusal flag."""
        from autodidact.agent import QueryResponse
        from autodidact.routing import Resolved
        from autodidact.routing.stages import CloudEscalationStage

        seen_refused: list = []

        def escalate(state):
            seen_refused.append(state.refused)
            return QueryResponse(
                answer="x", routed_to="cloud", confidence=0.0,
                cost_usd=0.0, learned=False, latency_ms=0,
            )

        deps = _FakeCloudDeps(
            has_cloud_client=True,
            escalate_fn=escalate,
            record_query_fn=lambda *a, **k: None,
            append_history_fn=lambda *a, **k: None,
        )
        # state.refused starts True (e.g. would have been refused) but force=True
        # means the stage escalates regardless. Crucially, the response's
        # escalated_on_refusal is built from state.refused inside escalate_fn —
        # we just check the path is taken.
        state = self._state(refused=True)
        out = CloudEscalationStage(deps, force=True)(state)
        assert isinstance(out, Resolved)
        assert seen_refused == [True]

    def test_force_mode_with_no_cloud_returns_canned_response(self):
        """correct() with no cloud → a 'cannot re-verify' canned answer, NOT a fallback to local."""
        from autodidact.routing import Resolved
        from autodidact.routing.stages import CloudEscalationStage

        deps = _FakeCloudDeps(
            has_cloud_client=False,
            escalate_fn=lambda *a, **k: pytest.fail("must not escalate without cloud"),
            record_query_fn=lambda *a, **k: None,
            append_history_fn=lambda *a, **k: None,
        )
        state = self._state()  # no local_response — correct() doesn't run local
        out = CloudEscalationStage(deps, force=True)(state)
        assert isinstance(out, Resolved)
        assert "No cloud model" in out.response.answer
        assert out.response.routed_to == "local"


# ── CorrectionInvalidationStage ─────────────────────────────────


@dataclass
class _FakeCorrectionDeps:
    has_embed_client: bool
    embed_fn: Callable                     # text -> np.ndarray
    memory_search_fn: Callable             # (q_emb, limit, min_similarity) -> hits
    memory_invalidate_fn: Callable         # entry_id -> None


class TestCorrectionInvalidationStage:
    """Only used by Agent.correct() — invalidates the closest memory entry."""

    def _state(self):
        from autodidact.routing import RoutingState
        events: list = []
        return RoutingState(
            question="forget that", context=None, started=time.perf_counter(),
            emit=events.append,
        )

    def test_invalidates_closest_match(self):
        from autodidact.routing import Continue
        from autodidact.routing.stages import CorrectionInvalidationStage

        invalidated: list = []
        hit = MagicMock()
        hit.entry = MagicMock(id="entry-42")
        deps = _FakeCorrectionDeps(
            has_embed_client=True,
            embed_fn=lambda text: np.zeros(4),
            memory_search_fn=lambda emb, limit, min_similarity: [hit],
            memory_invalidate_fn=lambda eid: invalidated.append(eid),
        )
        state = self._state()
        out = CorrectionInvalidationStage(deps)(state)
        assert isinstance(out, Continue)
        assert invalidated == ["entry-42"]

    def test_skips_when_no_embed_client(self):
        from autodidact.routing import Continue
        from autodidact.routing.stages import CorrectionInvalidationStage

        deps = _FakeCorrectionDeps(
            has_embed_client=False,
            embed_fn=lambda text: pytest.fail("must not embed without client"),
            memory_search_fn=lambda *a, **k: [],
            memory_invalidate_fn=lambda *a, **k: None,
        )
        state = self._state()
        out = CorrectionInvalidationStage(deps)(state)
        assert isinstance(out, Continue)

    def test_no_match_is_a_noop_continue(self):
        from autodidact.routing import Continue
        from autodidact.routing.stages import CorrectionInvalidationStage

        invalidated: list = []
        deps = _FakeCorrectionDeps(
            has_embed_client=True,
            embed_fn=lambda text: np.zeros(4),
            memory_search_fn=lambda emb, limit, min_similarity: [],
            memory_invalidate_fn=lambda eid: invalidated.append(eid),
        )
        state = self._state()
        out = CorrectionInvalidationStage(deps)(state)
        assert isinstance(out, Continue)
        assert invalidated == []


# ── Agent integration: pipelines wired in ──────────────────────


class TestAgentUsesPipelines:
    """Agent.query() and Agent.correct() must run their respective pipelines.

    These tests don't reproduce the full Agent fixture — they verify the Agent
    delegates to a configurable list of stages via attributes
    `_query_stages` / `_correct_stages`. The big end-to-end coverage stays in
    test_agent.py (which uses the real default stages).
    """

    def _make_agent(self):
        """Build an Agent without running __init__, like other tests do."""
        from autodidact.agent import Agent
        return Agent.__new__(Agent)

    def test_query_calls_pipeline_in_order(self):
        from autodidact.agent import QueryResponse
        from autodidact.routing import Resolved, Continue

        a = self._make_agent()

        calls: list[str] = []

        def s1(state):
            calls.append("s1")
            return Continue()

        def s2(state):
            calls.append("s2")
            return Resolved(response=QueryResponse(
                answer="ok", routed_to="local", confidence=1.0,
                cost_usd=0.0, learned=False, latency_ms=0,
            ))

        a._query_stages = [s1, s2]
        resp = a.query("hello")
        assert resp.answer == "ok"
        assert calls == ["s1", "s2"]

    def test_correct_calls_dedicated_pipeline(self):
        from autodidact.agent import QueryResponse
        from autodidact.routing import Resolved

        a = self._make_agent()
        called: list[str] = []

        def cstage(state):
            called.append("c")
            return Resolved(response=QueryResponse(
                answer="re-asked", routed_to="cloud", confidence=0.0,
                cost_usd=0.05, learned=True, latency_ms=200,
            ))

        a._correct_stages = [cstage]
        # Even with _query_stages defined, correct() must use _correct_stages.
        a._query_stages = [lambda state: pytest.fail("query stages must not run on correct()")]
        resp = a.correct("forget that")
        assert resp.answer == "re-asked"
        assert called == ["c"]

    def test_default_query_stages_after_init(self):
        """After Agent.__init__, _query_stages is populated with the four real stages in order."""
        from autodidact.agent import Agent
        from autodidact.routing.stages import (
            CloudEscalationStage, GsaPreGateStage, LocalGenerationStage, MemoryStage,
        )

        # Build a minimal Agent. db_path uses a tmpfile per call.
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db = f.name
        agent = Agent(db_path=db)
        kinds = [type(s) for s in agent._query_stages]
        assert kinds == [
            MemoryStage, GsaPreGateStage, LocalGenerationStage, CloudEscalationStage,
        ]
        cinds = [type(s) for s in agent._correct_stages]
        # Correct: invalidate then escalate-with-force.
        from autodidact.routing.stages import CorrectionInvalidationStage
        assert cinds == [CorrectionInvalidationStage, CloudEscalationStage]
        # The cloud escalation in the correct pipeline is the force=True variant.
        assert agent._correct_stages[-1].force is True
        assert agent._query_stages[-1].force is False

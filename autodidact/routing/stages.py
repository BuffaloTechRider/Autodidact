"""Concrete RoutingStage implementations.

Each stage is a class with explicit dependencies (per the Q1 decision —
no Agent reference inside stages). The Agent wires them up at __init__
by passing the dependency dataclasses below.

Stages preserve the exact behaviour of the inline pipeline that used to
live in Agent.query() / Agent.correct(). The progress event protocol,
QueryResponse field set, and side-effect order are unchanged.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, TYPE_CHECKING

import numpy as np

from autodidact.routing import Continue, Resolved, RoutingState, StageOutcome

if TYPE_CHECKING:
    from autodidact.agent import QueryResponse
    from autodidact.knowledge_store import ScoredKnowledgeEntry
    from autodidact.llm_client import ChatResponse

logger = logging.getLogger(__name__)


# ── Memory direct-hit threshold ─────────────────────────────────
# Mirrors agent.MEMORY_DIRECT_THRESHOLD. Imported lazily inside stages
# instead of at module scope because the Agent module imports from here
# and we want to avoid a circular import.
def _memory_direct_threshold() -> float:
    from autodidact.agent import MEMORY_DIRECT_THRESHOLD
    return MEMORY_DIRECT_THRESHOLD


def _query_response():
    """Lazy import for the same circular-import reason."""
    from autodidact.agent import QueryResponse
    return QueryResponse


def _elapsed_ms_for(state: RoutingState) -> int:
    from autodidact.agent import _elapsed_ms
    return _elapsed_ms(state.started)


# ── MemoryStage ─────────────────────────────────────────────────


@dataclass
class MemoryStageDeps:
    """Dependencies threaded into MemoryStage (Q1 decision: no Agent ref)."""

    check_memory_fn: Callable[[str], tuple[list, Optional[np.ndarray]]]
    knowledge_store_access: Callable[[str], None]
    entry_age_fn: Callable[[Any], float]
    staleness_days: float
    has_local_client: bool
    build_messages_fn: Callable[..., tuple[list, list[str]]]
    call_local_fn: Callable[..., "ChatResponse"]
    record_query_fn: Callable[..., None]
    append_history_fn: Callable[[str, str], None]


class MemoryStage:
    """Stage 1: check the knowledge store for a fresh, high-similarity hit.

    Resolves with routed_to='memory' when the best hit is fresh, scores at
    or above MEMORY_DIRECT_THRESHOLD, AND a local client is available to
    render the answer using the hit as context.

    Continues otherwise (no hits, low score, stale, or no local client).
    Populates state.memory_hits / state.q_emb / state.best_hit so
    downstream stages can reuse them.
    """

    def __init__(self, deps: MemoryStageDeps) -> None:
        self.deps = deps

    def __call__(self, state: RoutingState) -> StageOutcome:
        memory_hits, q_emb = self.deps.check_memory_fn(state.question)
        state.memory_hits = memory_hits
        state.q_emb = q_emb
        state.best_hit = memory_hits[0] if memory_hits else None

        state.emit({
            "type": "thinking",
            "memory_hits": len(memory_hits),
            "best_similarity": state.best_hit.score if state.best_hit else 0.0,
        })

        if state.best_hit is None or state.best_hit.score < _memory_direct_threshold():
            return Continue()

        entry = state.best_hit.entry
        self.deps.knowledge_store_access(entry.id)
        age_days = self.deps.entry_age_fn(entry)
        is_stale = age_days > self.deps.staleness_days

        if is_stale:
            logger.info(
                "Memory hit is stale (%.1f days old); falling through to local",
                age_days,
            )
            return Continue()

        state.emit({
            "type": "memory_hit",
            "similarity": state.best_hit.score,
            "memory_source": entry.question,
            "age_days": age_days,
        })

        if not self.deps.has_local_client:
            # Quirk preserved from agent.py:286 — direct hit but no local
            # client to render it. memory_hit event has fired, but we
            # fall through to let downstream stages take over.
            return Continue()

        # Generate the answer using memory as context (streaming through emit).
        messages, ctx_sources = self.deps.build_messages_fn(
            state.question, state.context, memory_hits,
        )
        local_resp = self.deps.call_local_fn(messages, state.emit)
        answer = local_resp.content

        latency = _elapsed_ms_for(state)
        self.deps.record_query_fn(
            "memory", 0.0, state.best_hit.score, latency, question=state.question,
        )
        self.deps.append_history_fn(state.question, answer)

        QueryResponse = _query_response()
        return Resolved(response=QueryResponse(
            answer=answer,
            routed_to="memory",
            confidence=state.best_hit.score,
            cost_usd=0.0,
            learned=False,
            latency_ms=latency,
            context_sources=ctx_sources,
            memory_source=entry.question,
            memory_age_days=age_days,
            stale=False,
        ))


# ── GsaPreGateStage ─────────────────────────────────────────────


@dataclass
class GsaPreGateDeps:
    """Dependencies for the Stage 1.5 GSA pre-gate."""

    gsa_enabled: bool
    gsa_threshold: float
    has_local_client: bool
    has_cloud_client: bool
    document_search_fn: Optional[Callable[[str, Optional[np.ndarray]], list]]
    build_gsa_fn: Callable[[], Any]            # () -> SelfAssessment
    escalate_fn: Callable[[RoutingState], "QueryResponse"]


class GsaPreGateStage:
    """Stage 1.5: ask the local model 'can you answer this?' before generating.

    Resolves (escalating to cloud) when the probe says "no" with confidence
    below gsa_threshold. Continues otherwise.

    Skips itself entirely (Continue with state.gsa_p_yes=None) when:
    - GSA is disabled
    - No local client (no probe target)
    - No cloud client (no escalation target — gate is moot)
    - The memory stage saw a strong hit but fell through (stale): no point
      asking if the local can do it; just let it try
    - Strong document context exists (≥ 0.75): local will have what it needs
    - The probe itself raises (best-effort signal)
    """

    def __init__(self, deps: GsaPreGateDeps) -> None:
        self.deps = deps

    def __call__(self, state: RoutingState) -> StageOutcome:
        if not self.deps.gsa_enabled:
            return Continue()
        if not self.deps.has_local_client or not self.deps.has_cloud_client:
            return Continue()

        best_similarity = state.best_hit.score if state.best_hit else 0.0
        if best_similarity >= _memory_direct_threshold():
            # Strong memory match that fell through (stale). Don't gate.
            return Continue()

        # Skip GSA when the document store has a strong hit — local will
        # have the context to answer without escalation.
        if self.deps.document_search_fn is not None:
            try:
                doc_hits = self.deps.document_search_fn(state.question, state.q_emb)
                if doc_hits and doc_hits[0].score >= 0.75:
                    state.has_doc_context = True
                    return Continue()
            except Exception:
                pass

        state.emit({"type": "gsa_check"})
        try:
            probe = self.deps.build_gsa_fn()
            result = probe.compute(state.question, retrieved_hits=state.memory_hits)
            state.gsa_p_yes = result.p_yes
        except Exception as e:
            logger.warning("GSA probe failed, skipping gate: %s", e)
            state.gsa_p_yes = None
            return Continue()

        if state.gsa_p_yes < self.deps.gsa_threshold:
            response = self.deps.escalate_fn(state)
            response.escalated_on_gsa = True
            response.gsa_p_yes = state.gsa_p_yes
            return Resolved(response=response)

        return Continue()


# ── LocalGenerationStage ────────────────────────────────────────


@dataclass
class LocalGenerationDeps:
    """Dependencies for Stage 2 local generation."""

    has_local_client: bool
    has_cloud_client: bool
    build_messages_fn: Callable[..., tuple[list, list[str]]]
    call_local_fn: Callable[..., "ChatResponse"]
    refusal_detector: Callable[[str], bool]
    record_query_fn: Callable[..., None]
    append_history_fn: Callable[[str, str], None]


class LocalGenerationStage:
    """Stage 2: generate locally; resolve on success, continue on refusal.

    Three exit paths:
    - No local AND no cloud → Resolved with the canned 'No model configured'
      response. This is a terminal failure for callers that have nothing
      configured at all.
    - No local but has cloud → Continue (CloudEscalationStage will handle it).
    - Has local: run it. If the response doesn't look like a refusal, resolve
      with routed_to='local'. If it does look like a refusal, set
      state.refused=True / state.local_response=resp and Continue so the
      cloud stage picks it up.
    """

    def __init__(self, deps: LocalGenerationDeps) -> None:
        self.deps = deps

    def __call__(self, state: RoutingState) -> StageOutcome:
        if not self.deps.has_local_client:
            if not self.deps.has_cloud_client:
                # Nothing configured at all.
                QueryResponse = _query_response()
                return Resolved(response=QueryResponse(
                    answer="No model configured. Run `autodidact init` to set up.",
                    routed_to="local",
                    confidence=0.0,
                    cost_usd=0.0,
                    learned=False,
                    latency_ms=_elapsed_ms_for(state),
                ))
            # Has cloud — let CloudEscalationStage handle it.
            return Continue()

        messages, ctx_sources = self.deps.build_messages_fn(
            state.question, state.context, state.memory_hits,
        )
        local_resp = self.deps.call_local_fn(messages, state.emit)
        state.local_response = local_resp

        if self.deps.refusal_detector(local_resp.content):
            state.refused = True
            return Continue()

        confidence = 1.0
        state.emit({"type": "local_done", "confidence": confidence})
        latency = _elapsed_ms_for(state)
        self.deps.record_query_fn(
            "local", 0.0, confidence, latency, question=state.question,
        )
        self.deps.append_history_fn(state.question, local_resp.content)

        QueryResponse = _query_response()
        return Resolved(response=QueryResponse(
            answer=local_resp.content,
            routed_to="local",
            confidence=confidence,
            cost_usd=0.0,
            learned=False,
            latency_ms=latency,
            context_sources=ctx_sources,
            memory_similarity=state.best_hit.score if state.best_hit else None,
            gsa_p_yes=state.gsa_p_yes,
        ))


# ── CloudEscalationStage ────────────────────────────────────────


@dataclass
class CloudEscalationDeps:
    """Dependencies for cloud escalation. Used by both query() and correct()."""

    has_cloud_client: bool
    escalate_fn: Callable[[RoutingState], "QueryResponse"]
    record_query_fn: Callable[..., None]
    append_history_fn: Callable[[str, str], None]


class CloudEscalationStage:
    """Stage 3: escalate to cloud OR fall back to a stale local answer.

    Two modes via the `force` constructor flag:

    - force=False (query() pipeline): when there's no cloud, the stage
      returns the local model's last response (stored on state) with a
      neutral confidence of 0.5. This preserves the v1.0.6 quirk where a
      refused-local + no-cloud still returns the local text.

    - force=True (correct() pipeline): when there's no cloud, the stage
      returns a 'No cloud model configured — cannot re-verify' canned
      response. correct() doesn't run local, so no local_response is
      available to fall back on.

    With cloud available, both modes call escalate_fn(state) and stamp
    gsa_p_yes from state onto the result.
    """

    def __init__(self, deps: CloudEscalationDeps, *, force: bool) -> None:
        self.deps = deps
        self.force = force

    def __call__(self, state: RoutingState) -> StageOutcome:
        QueryResponse = _query_response()

        if not self.deps.has_cloud_client:
            if self.force:
                # correct() path with no cloud — bail out cleanly.
                return Resolved(response=QueryResponse(
                    answer="No cloud model configured — cannot re-verify.",
                    routed_to="local",
                    confidence=0.0,
                    cost_usd=0.0,
                    learned=False,
                    latency_ms=_elapsed_ms_for(state),
                ))

            # query() path with no cloud: return local's last (refused) answer.
            assert state.local_response is not None, (
                "CloudEscalationStage(force=False) without cloud expects "
                "LocalGenerationStage to have populated state.local_response"
            )
            state.emit({"type": "local_done", "confidence": 0.5})
            latency = _elapsed_ms_for(state)
            self.deps.record_query_fn(
                "local", 0.0, 0.5, latency, question=state.question,
            )
            self.deps.append_history_fn(state.question, state.local_response.content)
            return Resolved(response=QueryResponse(
                answer=state.local_response.content,
                routed_to="local",
                confidence=0.5,
                cost_usd=0.0,
                learned=False,
                latency_ms=latency,
                gsa_p_yes=state.gsa_p_yes,
            ))

        response = self.deps.escalate_fn(state)
        # Stamp GSA p_yes onto whatever escalate produced. Whether the response
        # is escalated_on_refusal is set by escalate_fn based on state.refused.
        response.gsa_p_yes = state.gsa_p_yes
        return Resolved(response=response)


# ── CorrectionInvalidationStage ─────────────────────────────────


@dataclass
class CorrectionInvalidationDeps:
    """Dependencies for the correct() pipeline's invalidation step."""

    has_embed_client: bool
    embed_fn: Callable[[str], np.ndarray]
    memory_search_fn: Callable[..., list]
    memory_invalidate_fn: Callable[[str], None]


class CorrectionInvalidationStage:
    """Invalidate the closest memory entry to the corrected question.

    Always returns Continue — the pipeline always escalates to cloud after.
    """

    def __init__(self, deps: CorrectionInvalidationDeps) -> None:
        self.deps = deps

    def __call__(self, state: RoutingState) -> StageOutcome:
        if not self.deps.has_embed_client:
            return Continue()

        try:
            q_emb = self.deps.embed_fn(state.question)
        except Exception as e:
            logger.warning("Failed to embed question for invalidation: %s", e)
            return Continue()

        try:
            hits = self.deps.memory_search_fn(q_emb, limit=1, min_similarity=0.80)
        except Exception as e:
            logger.warning("Memory search failed during correction: %s", e)
            return Continue()

        for hit in hits:
            try:
                self.deps.memory_invalidate_fn(hit.entry.id)
                logger.info("Invalidated memory entry %s for correction", hit.entry.id)
            except Exception as e:
                logger.warning("Failed to invalidate %s: %s", hit.entry.id, e)

        return Continue()


__all__ = [
    "CloudEscalationDeps",
    "CloudEscalationStage",
    "CorrectionInvalidationDeps",
    "CorrectionInvalidationStage",
    "GsaPreGateDeps",
    "GsaPreGateStage",
    "LocalGenerationDeps",
    "LocalGenerationStage",
    "MemoryStage",
    "MemoryStageDeps",
]

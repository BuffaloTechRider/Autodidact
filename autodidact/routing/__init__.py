"""Routing pipeline — composable stages for Agent.query() / Agent.correct().

The Agent's routing logic used to live as a 200-line method on Agent.query().
It's now a list of RoutingStage callables that share a RoutingState. Each
stage returns Resolved (pipeline halts with this response) or Continue
(advance to next stage).

This is a structural refactor — the *behaviour* is identical to the inline
version. Stages call back into Agent helpers via dependency dataclasses
threaded explicitly at construction (no Agent reference inside stages).
See docs/RAG-PIPELINE.md and CONTEXT.md for the routing concepts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, TYPE_CHECKING, Union

import numpy as np

if TYPE_CHECKING:
    from autodidact.agent import QueryResponse
    from autodidact.events import ProgressEvent
    from autodidact.knowledge_store import ScoredKnowledgeEntry
    from autodidact.llm_client import ChatResponse


@dataclass
class RoutingState:
    """Shared mutable state threaded through the pipeline.

    Stages may read and write fields on this object. A field starts at its
    default until a stage populates it; downstream stages can rely on the
    value or fall back to the default if the upstream stage was skipped.
    """

    question: str
    context: Optional[str]
    started: float                                  # time.perf_counter() at query start
    emit: Callable[["ProgressEvent"], None]         # typed event callback

    # Populated by MemoryStage
    memory_hits: list["ScoredKnowledgeEntry"] = field(default_factory=list)
    q_emb: Optional[np.ndarray] = None
    best_hit: Optional["ScoredKnowledgeEntry"] = None

    # Populated by GsaPreGateStage
    gsa_p_yes: Optional[float] = None
    has_doc_context: bool = False

    # Populated by LocalGenerationStage
    refused: bool = False
    local_response: Optional["ChatResponse"] = None


@dataclass(frozen=True)
class Resolved:
    """A stage produced a final response. The pipeline halts."""

    response: "QueryResponse"


@dataclass(frozen=True)
class Continue:
    """A stage didn't resolve. Advance to the next stage."""


StageOutcome = Union[Resolved, Continue]


# A stage is anything callable as `stage(state) -> StageOutcome`. Concrete
# stages are classes (so they can hold their dependencies) but the pipeline
# composer doesn't care about that — plain functions work too.
RoutingStage = Callable[[RoutingState], StageOutcome]


def run_pipeline(stages: list[RoutingStage], state: RoutingState) -> "QueryResponse":
    """Run stages in order until one returns Resolved.

    Raises RuntimeError if all stages return Continue. In production this
    can't happen — CloudEscalationStage always resolves — but the guard
    catches misconfigured pipelines in tests.
    """
    for stage in stages:
        outcome = stage(state)
        if isinstance(outcome, Resolved):
            return outcome.response
    raise RuntimeError("routing pipeline did not produce a response")


__all__ = [
    "Continue",
    "Resolved",
    "RoutingStage",
    "RoutingState",
    "StageOutcome",
    "run_pipeline",
]

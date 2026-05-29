"""Typed progress events for ``Agent.query`` / ``Agent.correct``.

Replaces the dict-typed protocol where consumers had to read ``event["type"]``
and pull fields by string. Each event is a frozen dataclass with documented
fields. Consumers branch via ``isinstance``:

    def on_progress(event: ProgressEvent) -> None:
        if isinstance(event, ThinkingEvent):
            print(f"checking memory ({event.memory_hits} hits)")
        elif isinstance(event, TokenEvent):
            sys.stdout.write(event.text)
        elif isinstance(event, CloudCallEvent):
            print(f"escalating to {event.model}")
        ...

Static analysis catches "consumer doesn't handle the new event" and "agent
emits an unknown field" — both silent bugs in the dict-typed era.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Union


@dataclass(frozen=True)
class ThinkingEvent:
    """Stage 1 (memory check) started.

    Always the first event Agent.query() emits. ``memory_hits`` is the
    count of similar past Q&A retrieved (0 means cold-start). The
    ``best_similarity`` is the top hit's score (or 0.0 if no hits).
    """

    memory_hits: int
    best_similarity: float


@dataclass(frozen=True)
class MemoryHitEvent:
    """Stage 1 found a fresh, high-similarity hit and is using it as context.

    Emitted only when the best hit's score >= MEMORY_DIRECT_THRESHOLD AND
    the entry is fresh (newer than ``staleness_days``). ``memory_source``
    is the original question that produced this stored answer.
    """

    similarity: float
    memory_source: str
    age_days: float


@dataclass(frozen=True)
class GsaCheckEvent:
    """Stage 1.5 (GSA pre-gate probe) is about to run.

    No fields. The probe's result feeds into the routing decision; if the
    probe fires this event, the local model is being asked "can you
    answer this?" before full generation.
    """


@dataclass(frozen=True)
class LocalDoneEvent:
    """Stage 2 (local generation) finished.

    ``confidence`` is the logprob-based confidence score in [0, 1] when
    available. Today the agent always emits ``1.0`` on success and
    ``0.5`` on the no-cloud refusal-fallback path; the field is kept for
    forward compatibility with future per-token-confidence work.
    """

    confidence: float


@dataclass(frozen=True)
class CloudCallEvent:
    """Stage 3 (cloud escalation) is about to call the cloud provider.

    ``model`` is the canonical model identifier the agent will invoke
    (e.g. ``us.anthropic.claude-sonnet-4-5-20250929-v1:0``).
    """

    model: str


@dataclass(frozen=True)
class CloudDoneEvent:
    """Stage 3 finished. Cloud answer received.

    ``cost`` is the estimated USD cost for this call. ``latency_ms`` is
    the wall-time of the cloud round-trip (including streaming).
    """

    model: str
    cost: float
    latency_ms: int


@dataclass(frozen=True)
class TokenEvent:
    """One streaming chunk of model output.

    ``source`` is which model produced it (the local slot or the cloud
    slot, regardless of underlying provider). ``phase`` distinguishes
    user-facing ``content`` from reasoning/thinking output that callers
    typically render dimmed or hidden.
    """

    source: Literal["local", "cloud"]
    phase: Literal["content", "thinking"]
    text: str


# ── The public union ─────────────────────────────────────────────


# Order matters for runtime isinstance checks against this union: more
# specific types first, but since these are all leaf dataclasses the
# order is purely cosmetic for isinstance(). Used for type annotations
# at call boundaries.
ProgressEvent = Union[
    ThinkingEvent,
    MemoryHitEvent,
    GsaCheckEvent,
    LocalDoneEvent,
    CloudCallEvent,
    CloudDoneEvent,
    TokenEvent,
]


__all__ = [
    "CloudCallEvent",
    "CloudDoneEvent",
    "GsaCheckEvent",
    "LocalDoneEvent",
    "MemoryHitEvent",
    "ProgressEvent",
    "ThinkingEvent",
    "TokenEvent",
]

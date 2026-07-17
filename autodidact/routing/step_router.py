"""Step-level routing thresholds for the executor's ReAct loop.

Q&A routing (``stages.py``) decides local-vs-cloud once per *query*. The
executor decides once per *tool call* — the moat carried into the agent loop.
Each iteration the local model proposes a tool call with an ``avg_logprob``;
the router turns that confidence into a tier:

    avg_logprob > threshold.high   → Tier 1: trust the local call
    avg_logprob < threshold.low    → Tier 3: escalate to cloud immediately
    otherwise                      → Tier 2: verify (self-consistency)

Logprobs are ``<= 0`` (log of a probability); values closer to 0 mean the
model was more confident, so ``high`` sits above ``low`` on the number line
(e.g. high=-0.3, low=-1.0).

``StepRouter`` is the seam. ``FixedThresholdRouter`` is the trivial
implementation used to bring the executor up; a Thompson-sampled router that
learns per-category bands from ``record_outcome`` (reading/writing the
``thompson_params`` table) plugs in behind this same interface later — the
executor never changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol, runtime_checkable


class Tier(str, Enum):
    """Which path a step took, for outcome recording and trajectory logging."""

    LOCAL = "LOCAL"          # Tier 1: local model, trusted outright
    VERIFY = "VERIFY"        # Tier 2: local model, confirmed by self-consistency
    CLOUD = "CLOUD"          # Tier 3: escalated to cloud


@dataclass(frozen=True)
class Threshold:
    """Confidence band edges for one category. ``high >= low`` always holds.

    Both are ``avg_logprob`` cutoffs (``<= 0``). A step is Tier 1 when its
    logprob is above ``high``, Tier 3 when below ``low``, Tier 2 in between.
    """

    high: float
    low: float

    def __post_init__(self) -> None:
        if self.high < self.low:
            raise ValueError(
                f"Threshold.high ({self.high}) must be >= low ({self.low})"
            )

    def tier_for(self, avg_logprob: float | None) -> Tier:
        """Classify a step's confidence into a tier.

        A ``None`` logprob (backend without logprob support, e.g. Bedrock)
        is treated as unknown-confidence and routed to Tier 2 for verification
        rather than blindly trusted or escalated.
        """
        if avg_logprob is None:
            return Tier.VERIFY
        if avg_logprob > self.high:
            return Tier.LOCAL
        if avg_logprob < self.low:
            return Tier.CLOUD
        return Tier.VERIFY


@runtime_checkable
class StepRouter(Protocol):
    """Turns per-step confidence into a routing threshold, and learns from
    outcomes. The executor depends only on these two methods.
    """

    def get_threshold(self, category: str) -> Threshold:
        """Return the confidence band for ``category`` (e.g. "code", "terminal")."""
        ...

    def record_outcome(self, category: str, tier: Tier, *, success: bool) -> None:
        """Report how a routed step turned out, so an adaptive router can update
        its posteriors. Fixed routers ignore this."""
        ...


# Conservative defaults for the fixed router. Calibrated loosely: a step needs
# solid confidence (> -0.3 avg logprob) to run locally unverified, and only a
# clearly low-confidence step (< -1.0) escalates outright; the wide middle band
# gets a cheap self-consistency check before spending a cloud call. These are
# placeholders — the Thompson router will learn per-category bands from data.
_DEFAULT_THRESHOLD = Threshold(high=-0.3, low=-1.0)


class FixedThresholdRouter:
    """Static, non-learning ``StepRouter``.

    Returns the same band for every category (overridable per-category via the
    constructor) and drops outcome reports. Enough to run the executor's tiered
    loop end-to-end before the adaptive router exists.
    """

    def __init__(
        self,
        default: Threshold = _DEFAULT_THRESHOLD,
        *,
        per_category: dict[str, Threshold] | None = None,
    ) -> None:
        self._default = default
        self._per_category = dict(per_category or {})

    def get_threshold(self, category: str) -> Threshold:
        return self._per_category.get(category, self._default)

    def record_outcome(self, category: str, tier: Tier, *, success: bool) -> None:
        # No-op: this router does not learn. The adaptive router overrides this.
        return None


__all__ = [
    "FixedThresholdRouter",
    "StepRouter",
    "Threshold",
    "Tier",
]

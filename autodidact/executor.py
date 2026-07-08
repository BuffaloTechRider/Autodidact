"""ReAct execution loop with step-level tiered routing.

This is the central new component of v2: a bounded tool-calling loop where
*each step's* local-vs-cloud decision runs through Autodidact's confidence
routing — the moat, spliced into an otherwise commodity agent loop (see
docs/HERMES-LEARNINGS.md §3).

The loop, per iteration:
    1. Local model generates a response *with tools + logprobs* in one call.
    2. If it has no tool call → the task is done; return its text.
    3. Otherwise route the proposed tool call by its avg_logprob:
         Tier 1 (high confidence)  → run the local call as-is
         Tier 3 (low confidence)   → regenerate on cloud, run that
         Tier 2 (uncertain)        → regenerate locally @ temp 0.3;
                                      if it agrees with itself, run local;
                                      else escalate to cloud.
    4. Dispatch the chosen tool call, append the ``role="tool"`` result.
    5. Loop until done or the iteration budget is spent.

Only the pieces needed to run end-to-end live here. Cache-aware message
construction, trajectory persistence/resume, and trajectory compression are
layered on in later steps (tasks #4–#6). Kept deliberately small per CLAUDE.md
§2 — the dependencies are narrow protocols, not the whole Agent, so the loop is
unit-testable without a live model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Protocol

from autodidact.llm_client import ChatMessage, ChatResponseWithLogprobs, ToolCall
from autodidact.routing.step_router import StepRouter, Threshold, Tier
from autodidact.tools.registry import ToolRegistry


# ── Collaborator protocols ───────────────────────────────────────
#
# The executor depends on capabilities, not concrete classes. In production
# these are backed by the Agent's LLMClients; in tests, by fakes.


class ToolLLM(Protocol):
    """A model that can generate a tool call and score its confidence."""

    def chat_with_logprobs(
        self, messages: list[ChatMessage], **opts: Any
    ) -> ChatResponseWithLogprobs:
        ...


ProgressCallback = Optional[Callable[[dict], None]]


# ── Result type ──────────────────────────────────────────────────


@dataclass
class ExecutionResult:
    """Outcome of an executor run. Mirrors the DESIGN-V2 contract."""

    answer: str
    steps_taken: int
    escalations: int
    tools_used: list[str] = field(default_factory=list)
    cost_usd: float = 0.0
    stop_reason: str = "done"  # "done" | "budget_exhausted" | "no_local_model"


# ── Executor ─────────────────────────────────────────────────────


class Executor:
    """Runs a task to completion via the tiered ReAct loop.

    Parameters
    ----------
    local
        Model used for every step's first attempt (must support logprobs).
    cloud
        Model used on escalation. May be ``None`` — then Tier 3 / disagreement
        degrade to running the local call rather than crashing (logged via the
        result's escalation count staying flat).
    tools
        Registry supplying the function schemas and dispatch.
    router
        Supplies per-category confidence bands and receives outcomes.
    max_iterations
        Hard cap on loop turns (prevents runaway tool loops).
    estimate_cost
        Optional ``(input_tokens, output_tokens) -> usd`` for cloud calls,
        so the executor doesn't duplicate the Agent's rate table.
    """

    def __init__(
        self,
        *,
        local: Optional[ToolLLM],
        cloud: Optional[ToolLLM],
        tools: ToolRegistry,
        router: StepRouter,
        max_iterations: int = 20,
        estimate_cost: Optional[Callable[[int, int], float]] = None,
    ) -> None:
        self._local = local
        self._cloud = cloud
        self._tools = tools
        self._router = router
        self._max_iterations = max(1, int(max_iterations))
        self._estimate_cost = estimate_cost or (lambda _i, _o: 0.0)

    def execute(
        self,
        task: str,
        *,
        system: Optional[str] = None,
        context: Optional[str] = None,
        on_progress: ProgressCallback = None,
    ) -> ExecutionResult:
        emit = on_progress or (lambda _e: None)

        if self._local is None:
            return ExecutionResult(
                answer="No local model configured.",
                steps_taken=0,
                escalations=0,
                stop_reason="no_local_model",
            )

        schemas = self._tools.get_schemas()
        messages = self._build_initial_messages(task, system, context)

        steps = 0
        escalations = 0
        cost = 0.0
        tools_used: list[str] = []

        while steps < self._max_iterations:
            steps += 1

            # 1. Local generation with tools + logprobs (one call, both signals).
            resp = self._local.chat_with_logprobs(
                messages, tools=schemas, temperature=0.0,
            )

            # 2. No tool call → the model answered in text. Task complete.
            if not resp.tool_calls:
                return ExecutionResult(
                    answer=resp.content,
                    steps_taken=steps,
                    escalations=escalations,
                    tools_used=tools_used,
                    cost_usd=cost,
                    stop_reason="done",
                )

            proposed = resp.tool_calls[0]
            category = self._tools.toolset_of(proposed.name) or "default"
            threshold = self._router.get_threshold(category)
            tier = threshold.tier_for(resp.avg_logprob)

            # 3. Resolve the tier into the tool call we actually run.
            chosen, chosen_resp, escalated = self._route_step(
                tier, proposed, resp, messages, schemas, emit,
            )
            if escalated:
                escalations += 1
                cost += self._estimate_cost(
                    chosen_resp.input_tokens, chosen_resp.output_tokens,
                )
                actual_tier = Tier.CLOUD
            else:
                actual_tier = Tier.LOCAL if tier is Tier.LOCAL else Tier.VERIFY

            # 4. Dispatch the chosen tool call and append its result.
            result_json = self._tools.dispatch(chosen.name, chosen.arguments)
            tools_used.append(chosen.name)
            emit({
                "type": "tool_result",
                "tool": chosen.name,
                "tier": actual_tier.value,
                "result": result_json,
            })

            messages.append(ChatMessage(
                role="assistant", content=chosen_resp.content, tool_calls=[chosen],
            ))
            messages.append(ChatMessage(
                role="tool",
                content=result_json,
                tool_call_id=chosen.id,
                name=chosen.name,
            ))

            # 5. Record the routing outcome. A dispatched tool that didn't
            #    error is treated as a success for the router's posteriors.
            self._router.record_outcome(
                category, actual_tier, success=_dispatch_ok(result_json),
            )

        # Budget exhausted without a text answer.
        return ExecutionResult(
            answer="",
            steps_taken=steps,
            escalations=escalations,
            tools_used=tools_used,
            cost_usd=cost,
            stop_reason="budget_exhausted",
        )

    # ── Tier resolution ──────────────────────────────────────────

    def _route_step(
        self,
        tier: Tier,
        proposed: ToolCall,
        proposed_resp: ChatResponseWithLogprobs,
        messages: list[ChatMessage],
        schemas: list[dict],
        emit: Callable[[dict], None],
    ) -> tuple[ToolCall, ChatResponseWithLogprobs, bool]:
        """Return (tool_call_to_run, its_response, escalated_to_cloud)."""
        if tier is Tier.LOCAL:
            return proposed, proposed_resp, False

        if tier is Tier.CLOUD:
            return self._escalate(proposed, proposed_resp, messages, schemas, emit)

        # Tier 2 — verify via self-consistency at a small temperature.
        verify = self._local.chat_with_logprobs(  # type: ignore[union-attr]
            messages, tools=schemas, temperature=0.3,
        )
        if verify.tool_calls and _same_call(verify.tool_calls[0], proposed):
            emit({"type": "verify", "result": "consistent", "tool": proposed.name})
            return proposed, proposed_resp, False

        emit({"type": "verify", "result": "disagreement", "tool": proposed.name})
        return self._escalate(proposed, proposed_resp, messages, schemas, emit)

    def _escalate(
        self,
        proposed: ToolCall,
        proposed_resp: ChatResponseWithLogprobs,
        messages: list[ChatMessage],
        schemas: list[dict],
        emit: Callable[[dict], None],
    ) -> tuple[ToolCall, ChatResponseWithLogprobs, bool]:
        """Regenerate the step on the cloud model. Falls back to the local call
        when no cloud model is configured (escalated=False so cost/count stay
        honest)."""
        if self._cloud is None:
            return proposed, proposed_resp, False

        emit({"type": "cloud_call", "tool": proposed.name})
        cloud_resp = self._cloud.chat_with_logprobs(
            messages, tools=schemas, temperature=0.0,
        )
        if not cloud_resp.tool_calls:
            # Cloud declined to call a tool. Trust the local proposal rather
            # than stalling the loop; the text answer is not actionable here.
            return proposed, proposed_resp, False
        return cloud_resp.tool_calls[0], cloud_resp, True

    # ── Message construction ─────────────────────────────────────

    def _build_initial_messages(
        self, task: str, system: Optional[str], context: Optional[str],
    ) -> list[ChatMessage]:
        messages: list[ChatMessage] = []
        if system:
            messages.append(ChatMessage(role="system", content=system))
        user = task if not context else f"{context}\n\n{task}"
        messages.append(ChatMessage(role="user", content=user))
        return messages


# ── Helpers ──────────────────────────────────────────────────────


def _same_call(a: ToolCall, b: ToolCall) -> bool:
    """Two tool calls are 'the same' if name and arguments match (ids differ
    per generation, so they're ignored)."""
    return a.name == b.name and a.arguments == b.arguments


def _dispatch_ok(result_json: str) -> bool:
    """Whether a dispatch envelope reported success. Best-effort — a
    non-JSON or unexpected payload counts as failure."""
    import json

    try:
        return bool(json.loads(result_json).get("ok"))
    except Exception:
        return False


__all__ = ["Executor", "ExecutionResult", "ToolLLM"]

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
from autodidact.trajectory_compress import Summarizer, compress_if_needed
from autodidact.trajectory_store import StepRecord, TrajectoryStore


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


# Task-entry memory tier (FR-1): given the task text, return the best stored
# answer if there's a fresh, high-similarity hit — else None. Backed by the
# Agent's knowledge store; the executor stays ignorant of storage details.
@dataclass
class MemoryHit:
    """A fresh, high-similarity task-level answer from the knowledge store."""

    answer: str
    similarity: float
    source_question: str


MemoryProbe = Callable[[str], Optional[MemoryHit]]


# Per-step GSA pre-gate (FR-1): given the running messages, return p_yes — the
# local model's own estimate that it can handle the next action. Below the
# threshold, the step escalates before wasting a local generation.
GsaProbe = Callable[[list[ChatMessage]], Optional[float]]


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
    stop_reason: str = "done"  # "done" | "memory" | "budget_exhausted" | "no_local_model"


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
        max_escalations: int = 5,
        estimate_cost: Optional[Callable[[int, int], float]] = None,
        cloud_cache_ttl: Optional[str] = None,
        store: Optional[TrajectoryStore] = None,
        compress_token_budget: Optional[int] = None,
        summarize: Optional[Summarizer] = None,
        memory: Optional[MemoryProbe] = None,
        gsa: Optional[GsaProbe] = None,
        gsa_threshold: float = 0.55,
    ) -> None:
        self._local = local
        self._cloud = cloud
        self._tools = tools
        self._router = router
        self._max_iterations = max(1, int(max_iterations))
        # Cap cloud escalations per task to bound cost; once spent, uncertain
        # steps stay local instead of escalating.
        self._max_escalations = max(0, int(max_escalations))
        self._estimate_cost = estimate_cost or (lambda _i, _o: 0.0)
        # Optional trajectory compression: when the message list exceeds the
        # budget, summarize the stale middle (head + tail preserved).
        self._compress_token_budget = compress_token_budget
        self._summarize = summarize
        # When set ("5m"/"1h"), cloud calls request Anthropic prompt caching on
        # the stable prefix. Leave None for OpenAI/other strict endpoints.
        self._cloud_cache_ttl = cloud_cache_ttl
        # Optional: persist each iteration for resume + learning. None = the
        # loop runs in memory only (unit tests, ephemeral runs).
        self._store = store
        # FR-1 memory tier: task-entry short-circuit against the knowledge
        # store. None = no memory tier (executor is then 3-tier, as before).
        self._memory = memory
        # FR-1 GSA pre-gate: per-step "can local handle this?" probe. Escalates
        # hopeless steps before paying for a local generation. None = disabled.
        self._gsa = gsa
        self._gsa_threshold = gsa_threshold

    def execute(
        self,
        task: str,
        *,
        system: Optional[str] = None,
        context: Optional[str] = None,
        on_progress: ProgressCallback = None,
        resume_from: Optional[str] = None,
    ) -> ExecutionResult:
        emit = on_progress or (lambda _e: None)

        if self._local is None:
            return ExecutionResult(
                answer="No local model configured.",
                steps_taken=0,
                escalations=0,
                stop_reason="no_local_model",
            )

        # Tier 0 — memory (FR-1): a fresh, high-similarity hit for the whole
        # task short-circuits the loop with the stored answer, at $0. Only on a
        # fresh start (a resume already has committed work to continue).
        if self._memory is not None and resume_from is None:
            hit = self._memory(task)
            if hit is not None:
                emit({
                    "type": "memory_hit",
                    "similarity": hit.similarity,
                    "source": hit.source_question,
                })
                return ExecutionResult(
                    answer=hit.answer, steps_taken=0, escalations=0,
                    stop_reason="memory",
                )

        schemas = self._tools.get_schemas()

        # Resume an interrupted run, or start fresh. On resume we replay the
        # persisted tool turns after the rebuilt prefix and continue where the
        # committed steps left off.
        steps = 0
        messages: list[ChatMessage]
        trajectory_id: Optional[str] = None
        if resume_from is not None and self._store is not None:
            head = self._store.header(resume_from)
            if head is None:
                raise ValueError(f"unknown trajectory to resume: {resume_from!r}")
            trajectory_id = resume_from
            task = head["task"]
            system = head["system"]
            context = head["context"]
            prior = self._store.load_steps(resume_from)
            steps = prior[-1].step_index if prior else 0
            messages = self._build_initial_messages(task, system, context)
            messages.extend(self._store.replay_messages(resume_from))
        else:
            messages = self._build_initial_messages(task, system, context)
            if self._store is not None:
                trajectory_id = self._store.start(task, system=system, context=context)

        escalations = 0
        cost = 0.0
        tools_used: list[str] = []

        while steps < self._max_iterations:
            steps += 1

            # 0. Compress the trajectory if it has outgrown the context budget
            #    (protects head + tail, summarizes the stale middle).
            if self._compress_token_budget is not None and self._summarize is not None:
                messages = compress_if_needed(
                    messages,
                    token_budget=self._compress_token_budget,
                    summarize=self._summarize,
                )

            # 1. GSA pre-gate (FR-1): probe "can local handle this step?" before
            #    generating. A veto escalates the step straight to cloud, so we
            #    don't pay for a local generation the model predicts will fail.
            gsa_escalated = False
            resp: Optional[ChatResponseWithLogprobs] = None
            if (self._gsa is not None and self._cloud is not None
                    and escalations < self._max_escalations):
                p_yes = self._gsa(messages)
                if p_yes is not None and p_yes < self._gsa_threshold:
                    emit({"type": "gsa_check", "p_yes": p_yes, "escalate": True})
                    resp = self._cloud.chat_with_logprobs(
                        messages, **self._cloud_opts(schemas),
                    )
                    gsa_escalated = True

            # 2. Local generation with tools + logprobs (one call, both signals).
            #    Skipped when the GSA gate already escalated this step to cloud.
            if resp is None:
                resp = self._local.chat_with_logprobs(
                    messages, tools=schemas, temperature=0.0,
                )

            # 3. No tool call → the model answered in text. Task complete.
            if not resp.tool_calls:
                return self._finish(
                    trajectory_id, answer=resp.content, steps=steps,
                    escalations=escalations, tools_used=tools_used, cost=cost,
                    stop_reason="done",
                )

            proposed = resp.tool_calls[0]
            category = self._tools.toolset_of(proposed.name) or "default"

            # 4. Resolve the tier into the tool call we actually run. A GSA
            #    escalation is already a cloud call; otherwise route on the
            #    logprob tier (once the budget is spent, uncertain steps stay
            #    local).
            if gsa_escalated:
                tier = Tier.CLOUD
                chosen, chosen_resp, escalated = proposed, resp, True
            else:
                tier = self._router.get_threshold(category).tier_for(resp.avg_logprob)
                allow_escalation = escalations < self._max_escalations
                chosen, chosen_resp, escalated = self._route_step(
                    tier, proposed, resp, messages, schemas, emit,
                    allow_escalation=allow_escalation,
                )
            if escalated:
                escalations += 1
                cost += self._estimate_cost(
                    chosen_resp.input_tokens, chosen_resp.output_tokens,
                )
                actual_tier = Tier.CLOUD
            else:
                actual_tier = Tier.LOCAL if tier is Tier.LOCAL else Tier.VERIFY

            # 5. Dispatch the chosen tool call and append its result.
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

            # 6. Checkpoint this iteration (committed immediately, so a crash on
            #    the next step resumes from here).
            if self._store is not None and trajectory_id is not None:
                self._store.record_step(trajectory_id, StepRecord(
                    step_index=steps, category=category, tier=actual_tier.value,
                    avg_logprob=resp.avg_logprob, tool_name=chosen.name,
                    tool_arguments=chosen.arguments, tool_result=result_json,
                    assistant_content=chosen_resp.content,
                ))

            # 7. Record the routing outcome. A dispatched tool that didn't
            #    error is treated as a success for the router's posteriors.
            self._router.record_outcome(
                category, actual_tier, success=_dispatch_ok(result_json),
            )

        # Budget exhausted without a text answer.
        return self._finish(
            trajectory_id, answer="", steps=steps, escalations=escalations,
            tools_used=tools_used, cost=cost, stop_reason="budget_exhausted",
        )

    def _finish(
        self, trajectory_id: Optional[str], *, answer: str, steps: int,
        escalations: int, tools_used: list[str], cost: float, stop_reason: str,
    ) -> ExecutionResult:
        if self._store is not None and trajectory_id is not None:
            self._store.finish(
                trajectory_id, status=stop_reason, answer=answer,
                escalations=escalations, tools_used=tools_used, cost_usd=cost,
            )
        return ExecutionResult(
            answer=answer, steps_taken=steps, escalations=escalations,
            tools_used=tools_used, cost_usd=cost, stop_reason=stop_reason,
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
        *,
        allow_escalation: bool,
    ) -> tuple[ToolCall, ChatResponseWithLogprobs, bool]:
        """Return (tool_call_to_run, its_response, escalated_to_cloud).

        When ``allow_escalation`` is False (escalation budget spent), a Tier 3
        or a Tier 2 disagreement runs the local proposal instead of escalating.
        """
        if tier is Tier.LOCAL:
            return proposed, proposed_resp, False

        if tier is Tier.CLOUD:
            if not allow_escalation:
                return proposed, proposed_resp, False
            return self._escalate(proposed, proposed_resp, messages, schemas, emit)

        # Tier 2 — verify via self-consistency at a small temperature.
        verify = self._local.chat_with_logprobs(  # type: ignore[union-attr]
            messages, tools=schemas, temperature=0.3,
        )
        if verify.tool_calls and _same_call(verify.tool_calls[0], proposed):
            emit({"type": "verify", "result": "consistent", "tool": proposed.name})
            return proposed, proposed_resp, False

        emit({"type": "verify", "result": "disagreement", "tool": proposed.name})
        if not allow_escalation:
            return proposed, proposed_resp, False
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
        cloud_resp = self._cloud.chat_with_logprobs(messages, **self._cloud_opts(schemas))
        if not cloud_resp.tool_calls:
            # Cloud declined to call a tool. Trust the local proposal rather
            # than stalling the loop; the text answer is not actionable here.
            return proposed, proposed_resp, False
        return cloud_resp.tool_calls[0], cloud_resp, True

    def _cloud_opts(self, schemas: list[dict]) -> dict[str, Any]:
        """Cloud call options, with prompt caching on the stable prefix when
        configured (Anthropic). Shared by the GSA gate and _escalate."""
        opts: dict[str, Any] = {"tools": schemas, "temperature": 0.0}
        if self._cloud_cache_ttl:
            opts["cache_ttl"] = self._cloud_cache_ttl
        return opts

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

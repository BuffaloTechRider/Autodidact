"""End-to-end tests for the tiered ReAct executor.

Uses a scripted fake LLM (no live model) and an isolated ToolRegistry so the
loop's routing, dispatch, and termination are exercised deterministically.
"""

from __future__ import annotations

import json

from autodidact.executor import Executor, ExecutionResult, MemoryHit
from autodidact.llm_client import ChatResponseWithLogprobs, ToolCall
from autodidact.routing.step_router import FixedThresholdRouter, Threshold, Tier
from autodidact.tools.registry import ToolRegistry


# ── Fakes ────────────────────────────────────────────────────────


class ScriptedLLM:
    """Returns a queued list of ChatResponseWithLogprobs, one per call."""

    def __init__(self, responses: list[ChatResponseWithLogprobs]) -> None:
        self._responses = list(responses)
        self.calls: list[dict] = []

    def chat_with_logprobs(self, messages, **opts) -> ChatResponseWithLogprobs:
        self.calls.append({"messages": list(messages), "opts": opts})
        return self._responses.pop(0)


def _tool_resp(name: str, args: dict, *, avg_logprob: float, content: str = "") -> ChatResponseWithLogprobs:
    return ChatResponseWithLogprobs(
        content=content,
        model="fake",
        tool_calls=[ToolCall(id="call_0", name=name, arguments=args)],
        avg_logprob=avg_logprob,
    )


def _text_resp(text: str, *, avg_logprob: float = -0.1) -> ChatResponseWithLogprobs:
    return ChatResponseWithLogprobs(
        content=text, model="fake", tool_calls=[], avg_logprob=avg_logprob,
    )


def _registry() -> ToolRegistry:
    reg = ToolRegistry()
    calls: list[dict] = []

    def echo(args: dict) -> dict:
        calls.append(args)
        return {"echoed": args.get("value")}

    reg.register(
        "echo",
        description="echo a value",
        parameters={
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
        },
        handler=echo,
        toolset="terminal",
    )
    reg._test_calls = calls  # type: ignore[attr-defined]
    return reg


# ── Tests ────────────────────────────────────────────────────────


def test_tier1_high_confidence_runs_local_then_completes():
    reg = _registry()
    local = ScriptedLLM([
        _tool_resp("echo", {"value": "hi"}, avg_logprob=-0.05),  # Tier 1
        _text_resp("done"),                                       # completes
    ])
    ex = Executor(local=local, cloud=None, tools=reg, router=FixedThresholdRouter())

    res = ex.execute("do the thing")

    assert isinstance(res, ExecutionResult)
    assert res.answer == "done"
    assert res.steps_taken == 2
    assert res.escalations == 0
    assert res.tools_used == ["echo"]
    assert reg._test_calls == [{"value": "hi"}]  # tool actually ran


def test_tier3_low_confidence_escalates_to_cloud():
    reg = _registry()
    local = ScriptedLLM([
        _tool_resp("echo", {"value": "local"}, avg_logprob=-3.0),  # Tier 3
        _text_resp("done"),
    ])
    cloud = ScriptedLLM([
        _tool_resp("echo", {"value": "cloud"}, avg_logprob=-0.01),
    ])
    ex = Executor(local=local, cloud=cloud, tools=reg,
                  router=FixedThresholdRouter(), estimate_cost=lambda i, o: 0.01)

    res = ex.execute("do the thing")

    assert res.answer == "done"
    assert res.escalations == 1
    # The cloud's tool call is the one that ran.
    assert reg._test_calls == [{"value": "cloud"}]


def test_tier2_self_consistent_runs_local_no_escalation():
    reg = _registry()
    local = ScriptedLLM([
        _tool_resp("echo", {"value": "x"}, avg_logprob=-0.6),  # Tier 2
        _tool_resp("echo", {"value": "x"}, avg_logprob=-0.5),  # verify: agrees
        _text_resp("done"),
    ])
    cloud = ScriptedLLM([])  # must not be consulted
    ex = Executor(local=local, cloud=cloud, tools=reg, router=FixedThresholdRouter())

    res = ex.execute("do the thing")

    assert res.answer == "done"
    assert res.escalations == 0
    assert reg._test_calls == [{"value": "x"}]


def test_tier2_disagreement_escalates():
    reg = _registry()
    local = ScriptedLLM([
        _tool_resp("echo", {"value": "a"}, avg_logprob=-0.6),  # Tier 2
        _tool_resp("echo", {"value": "b"}, avg_logprob=-0.6),  # verify: differs
        _text_resp("done"),
    ])
    cloud = ScriptedLLM([
        _tool_resp("echo", {"value": "cloud"}, avg_logprob=-0.01),
    ])
    ex = Executor(local=local, cloud=cloud, tools=reg, router=FixedThresholdRouter())

    res = ex.execute("do the thing")

    assert res.escalations == 1
    assert reg._test_calls == [{"value": "cloud"}]


def test_immediate_text_answer_no_tools():
    reg = _registry()
    local = ScriptedLLM([_text_resp("just an answer")])
    ex = Executor(local=local, cloud=None, tools=reg, router=FixedThresholdRouter())

    res = ex.execute("what is 2+2?")

    assert res.answer == "just an answer"
    assert res.steps_taken == 1
    assert res.tools_used == []


def test_budget_exhausted_when_model_never_stops():
    reg = _registry()
    # Always proposes a tool, never a text answer.
    local = ScriptedLLM([
        _tool_resp("echo", {"value": str(i)}, avg_logprob=-0.05) for i in range(10)
    ])
    ex = Executor(local=local, cloud=None, tools=reg,
                  router=FixedThresholdRouter(), max_iterations=3)

    res = ex.execute("loop forever")

    assert res.stop_reason == "budget_exhausted"
    assert res.steps_taken == 3


def test_no_local_model_returns_gracefully():
    reg = _registry()
    ex = Executor(local=None, cloud=None, tools=reg, router=FixedThresholdRouter())
    res = ex.execute("anything")
    assert res.stop_reason == "no_local_model"
    assert res.steps_taken == 0


def test_tier3_without_cloud_falls_back_to_local():
    reg = _registry()
    local = ScriptedLLM([
        _tool_resp("echo", {"value": "local"}, avg_logprob=-3.0),  # Tier 3
        _text_resp("done"),
    ])
    # No cloud configured — must not crash; runs local, no escalation counted.
    ex = Executor(local=local, cloud=None, tools=reg, router=FixedThresholdRouter())

    res = ex.execute("do the thing")

    assert res.escalations == 0
    assert reg._test_calls == [{"value": "local"}]


# ── Tier 0: task-entry memory short-circuit (FR-1) ───────────────


def test_memory_hit_short_circuits_before_loop():
    reg = _registry()
    local = ScriptedLLM([])  # must never be called on a memory hit
    events: list[dict] = []
    hit = MemoryHit(answer="cached answer", similarity=0.91, source_question="prior task")

    ex = Executor(
        local=local, cloud=None, tools=reg, router=FixedThresholdRouter(),
        memory=lambda task: hit,
    )
    res = ex.execute("do the thing", on_progress=events.append)

    assert res.answer == "cached answer"
    assert res.steps_taken == 0
    assert res.stop_reason == "memory"
    assert local.calls == []  # loop never ran
    assert any(e["type"] == "memory_hit" for e in events)


def test_memory_miss_falls_through_to_loop():
    reg = _registry()
    local = ScriptedLLM([
        _tool_resp("echo", {"value": "hi"}, avg_logprob=-0.05),
        _text_resp("done"),
    ])
    ex = Executor(
        local=local, cloud=None, tools=reg, router=FixedThresholdRouter(),
        memory=lambda task: None,  # no hit
    )
    res = ex.execute("do the thing")

    assert res.stop_reason == "done"
    assert res.steps_taken == 2


def test_memory_not_consulted_on_resume():
    # A resume has committed work to continue; re-answering from memory would
    # discard it. The probe must not fire.
    reg = _registry()
    local = ScriptedLLM([_text_resp("finished")])
    probe_calls: list[str] = []

    class _OneStepStore:
        def header(self, tid):
            return {"task": "t", "system": None, "context": None}
        def load_steps(self, tid):
            return []
        def replay_messages(self, tid):
            return []
        def finish(self, *a, **k):
            pass

    ex = Executor(
        local=local, cloud=None, tools=reg, router=FixedThresholdRouter(),
        store=_OneStepStore(), memory=lambda task: probe_calls.append(task) or None,
    )
    ex.execute("ignored", resume_from="traj-1")

    assert probe_calls == []  # memory tier skipped on resume


# ── GSA pre-gate per step (FR-1) ─────────────────────────────────


def test_gsa_veto_escalates_step_without_local_generation():
    reg = _registry()
    # Local would be Tier 1 if it ran — but the GSA gate vetoes first, so the
    # step escalates to cloud and the local model is never called this turn.
    local = ScriptedLLM([_text_resp("done")])  # only the completion turn
    cloud = ScriptedLLM([
        _tool_resp("echo", {"value": "cloud"}, avg_logprob=-0.01),
    ])
    events: list[dict] = []

    # Veto the first step (escalates to cloud), then pass so the completion
    # turn runs locally against the one scripted local response.
    gsa_scores = iter([0.10, 0.90])

    ex = Executor(
        local=local, cloud=cloud, tools=reg, router=FixedThresholdRouter(),
        gsa=lambda messages: next(gsa_scores), gsa_threshold=0.55,
    )
    res = ex.execute("hard task", on_progress=events.append)

    assert res.escalations == 1
    assert reg._test_calls == [{"value": "cloud"}]
    # Local ran once (the completion turn), not for the vetoed step.
    assert len(local.calls) == 1
    assert any(e["type"] == "gsa_check" and e["escalate"] for e in events)


def test_gsa_pass_routes_on_logprob_tier():
    reg = _registry()
    local = ScriptedLLM([
        _tool_resp("echo", {"value": "local"}, avg_logprob=-0.05),  # Tier 1
        _text_resp("done"),
    ])
    cloud = ScriptedLLM([])  # must not be consulted when GSA passes + Tier 1
    ex = Executor(
        local=local, cloud=cloud, tools=reg, router=FixedThresholdRouter(),
        gsa=lambda messages: 0.90, gsa_threshold=0.55,  # p_yes >= threshold → pass
    )
    res = ex.execute("easy task")

    assert res.escalations == 0
    assert reg._test_calls == [{"value": "local"}]


def test_gsa_gate_inert_without_cloud():
    # No cloud target → gate is moot; the local proposal runs as normal.
    reg = _registry()
    local = ScriptedLLM([
        _tool_resp("echo", {"value": "local"}, avg_logprob=-0.05),
        _text_resp("done"),
    ])
    ex = Executor(
        local=local, cloud=None, tools=reg, router=FixedThresholdRouter(),
        gsa=lambda messages: 0.01,  # would veto, but no cloud to escalate to
    )
    res = ex.execute("task")

    assert res.escalations == 0
    assert reg._test_calls == [{"value": "local"}]

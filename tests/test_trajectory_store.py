"""Tests for executor trajectory persistence + resume."""

from __future__ import annotations

import json

from autodidact.database import init_database
from autodidact.executor import Executor
from autodidact.llm_client import ChatResponseWithLogprobs, ToolCall
from autodidact.routing.step_router import FixedThresholdRouter
from autodidact.tools.registry import ToolRegistry
from autodidact.trajectory_store import StepRecord, TrajectoryStore


class ScriptedLLM:
    def __init__(self, responses):
        self._responses = list(responses)
        self.call_count = 0

    def chat_with_logprobs(self, messages, **opts):
        self.call_count += 1
        return self._responses.pop(0)


def _tool(name, args, lp=-0.05):
    return ChatResponseWithLogprobs(
        content="", model="f", avg_logprob=lp,
        tool_calls=[ToolCall(id="c", name=name, arguments=args)],
    )


def _text(t):
    return ChatResponseWithLogprobs(content=t, model="f", avg_logprob=-0.05, tool_calls=[])


def _reg():
    reg = ToolRegistry()
    reg.register(
        "echo", description="echo",
        parameters={"type": "object", "properties": {"value": {"type": "string"}}},
        handler=lambda a: {"echoed": a.get("value")}, toolset="terminal",
    )
    return reg


def _conn():
    return init_database(":memory:")


# ── Persistence ──────────────────────────────────────────────────


def test_completed_run_persists_trajectory_and_steps():
    conn = _conn()
    store = TrajectoryStore(conn)
    local = ScriptedLLM([_tool("echo", {"value": "a"}), _text("done")])
    ex = Executor(local=local, cloud=None, tools=_reg(),
                  router=FixedThresholdRouter(), store=store)

    res = ex.execute("mytask", system="SYS")

    assert res.stop_reason == "done"
    row = conn.execute(
        "SELECT task, status, answer, steps_taken FROM execution_trajectories"
    ).fetchone()
    assert (row["task"], row["status"], row["answer"], row["steps_taken"]) == (
        "mytask", "done", "done", 1,
    )
    steps = conn.execute(
        "SELECT tool_name, tool_arguments, tier FROM trajectory_steps"
    ).fetchall()
    assert len(steps) == 1
    assert steps[0]["tool_name"] == "echo"
    assert json.loads(steps[0]["tool_arguments"]) == {"value": "a"}


def test_step_committed_before_next_iteration():
    # A store that raises on the 2nd generation simulates a mid-run crash;
    # the first step must already be committed.
    conn = _conn()
    store = TrajectoryStore(conn)

    class CrashLLM:
        def __init__(self):
            self.n = 0

        def chat_with_logprobs(self, messages, **opts):
            self.n += 1
            if self.n == 1:
                return _tool("echo", {"value": "first"})
            raise RuntimeError("boom (simulated crash)")

    ex = Executor(local=CrashLLM(), cloud=None, tools=_reg(),
                  router=FixedThresholdRouter(), store=store)
    try:
        ex.execute("crashy")
    except RuntimeError:
        pass

    # First step survived the crash; trajectory left 'running' (resumable).
    steps = conn.execute("SELECT step_index, tool_name FROM trajectory_steps").fetchall()
    assert [(s["step_index"], s["tool_name"]) for s in steps] == [(1, "echo")]
    status = conn.execute("SELECT status FROM execution_trajectories").fetchone()[0]
    assert status == "running"


# ── Resume ───────────────────────────────────────────────────────


def test_resume_continues_from_last_step():
    conn = _conn()
    store = TrajectoryStore(conn)

    # First run: one tool step, then "crash" (budget=1 stops it mid-task).
    local1 = ScriptedLLM([_tool("echo", {"value": "step1"})])
    ex1 = Executor(local=local1, cloud=None, tools=_reg(),
                   router=FixedThresholdRouter(), store=store, max_iterations=1)
    res1 = ex1.execute("resumable task", system="SYS")
    assert res1.stop_reason == "budget_exhausted"

    tid = conn.execute("SELECT id FROM execution_trajectories").fetchone()[0]

    # Resume: the second-step model should see the replayed step1 tool result
    # in its message list, then finish.
    seen_messages = {}
    local2 = ScriptedLLM([_text("finished after resume")])

    def spy(messages, **opts):
        seen_messages["roles"] = [m.role for m in messages]
        return local2._responses.pop(0)

    local2.chat_with_logprobs = spy  # type: ignore[method-assign]
    ex2 = Executor(local=local2, cloud=None, tools=_reg(),
                   router=FixedThresholdRouter(), store=store, max_iterations=5)
    res2 = ex2.execute("ignored-on-resume", resume_from=tid)

    assert res2.stop_reason == "done"
    assert res2.answer == "finished after resume"
    # Replay reconstructed the prefix + the prior assistant/tool turns.
    assert seen_messages["roles"] == ["system", "user", "assistant", "tool"]
    # Trajectory now marked done.
    status = conn.execute(
        "SELECT status FROM execution_trajectories WHERE id = ?", (tid,)
    ).fetchone()[0]
    assert status == "done"


def test_replay_messages_roundtrip():
    conn = _conn()
    store = TrajectoryStore(conn)
    tid = store.start("t", system="s", context=None)
    store.record_step(tid, StepRecord(
        step_index=1, category="terminal", tier="LOCAL", avg_logprob=-0.1,
        tool_name="echo", tool_arguments={"value": "x"},
        tool_result='{"ok": true, "result": {"echoed": "x"}}',
        assistant_content="",
    ))
    msgs = store.replay_messages(tid)
    assert [m.role for m in msgs] == ["assistant", "tool"]
    assert msgs[0].tool_calls[0].name == "echo"
    assert msgs[1].tool_call_id == msgs[0].tool_calls[0].id

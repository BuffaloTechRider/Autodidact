"""Tests for trajectory compression + executor budgets (task #6)."""

from __future__ import annotations

from autodidact.executor import Executor
from autodidact.llm_client import ChatMessage, ChatResponseWithLogprobs, ToolCall
from autodidact.routing.step_router import FixedThresholdRouter
from autodidact.tools.registry import ToolRegistry
from autodidact.trajectory_compress import (
    compress_if_needed,
    estimate_tokens,
    make_local_summarizer,
)


def _msg(role, content="", tool_name=None, args=None):
    tcs = [ToolCall(id="c", name=tool_name, arguments=args or {})] if tool_name else None
    return ChatMessage(role=role, content=content, tool_calls=tcs)


# ── compress_if_needed ───────────────────────────────────────────


def test_under_budget_is_unchanged():
    msgs = [_msg("system", "s"), _msg("user", "task"), _msg("assistant", "short")]
    out = compress_if_needed(msgs, token_budget=10_000, summarize=lambda m: "SUM")
    assert out is msgs


def test_over_budget_compresses_middle_preserving_head_and_tail():
    big = "x" * 4000  # ~1000 tokens each
    msgs = [
        _msg("system", "SYS"),
        _msg("user", "TASK"),
        _msg("assistant", big),   # middle
        _msg("tool", big),        # middle
        _msg("assistant", big),   # middle
        _msg("tool", "recent-1"),
        _msg("assistant", "recent-2"),
        _msg("tool", "recent-3"),
        _msg("assistant", "recent-4"),
    ]
    out = compress_if_needed(
        msgs, token_budget=1000, summarize=lambda m: "MIDDLE SUMMARY",
        head_keep=2, tail_keep=4,
    )
    # head (2) + summary (1) + tail (5, nudged from 4 to keep the assistant
    # turn that owns the first tail tool result — never orphan a tool msg).
    assert out[0].content == "SYS"
    assert out[1].content == "TASK"
    assert "MIDDLE SUMMARY" in out[2].content
    assert out[2].role == "user"
    assert out[-1].content == "recent-4"
    # Invariant: the compressed list is shorter than the original, and every
    # role="tool" message is immediately preceded by an assistant turn.
    assert len(out) < len(msgs)
    for i, m in enumerate(out):
        if m.role == "tool":
            assert i > 0 and out[i - 1].role == "assistant"


def test_too_short_to_compress_is_unchanged():
    msgs = [_msg("system", "s" * 8000), _msg("user", "t" * 8000)]
    out = compress_if_needed(msgs, token_budget=1, summarize=lambda m: "SUM")
    assert out is msgs  # nothing between head and tail to summarize


def test_estimate_tokens_counts_content_and_tool_calls():
    msgs = [_msg("assistant", "abcd", tool_name="echo", args={"v": "x"})]
    assert estimate_tokens(msgs) > 0


def test_make_local_summarizer_uses_local_chat():
    class FakeLocal:
        def __init__(self):
            self.called_with = None

        def chat(self, messages, **opts):
            self.called_with = messages
            return ChatResponseWithLogprobs(content="RECAP", model="f")

    fake = FakeLocal()
    summ = make_local_summarizer(fake)
    out = summ([_msg("assistant", "did a thing", tool_name="echo", args={"v": 1})])
    assert out == "RECAP"
    # It summarized via the local model, not the cloud.
    assert fake.called_with is not None


# ── Executor budgets ─────────────────────────────────────────────


def _reg():
    reg = ToolRegistry()
    reg.register(
        "echo", description="echo",
        parameters={"type": "object", "properties": {"value": {"type": "string"}}},
        handler=lambda a: {"echoed": a.get("value")}, toolset="terminal",
    )
    return reg


class ScriptedLLM:
    def __init__(self, factory):
        self._factory = factory
        self.n = 0

    def chat_with_logprobs(self, messages, **opts):
        self.n += 1
        return self._factory(self.n)


def test_max_escalations_caps_cloud_calls():
    reg = _reg()
    # Local always low-confidence (Tier 3 → wants to escalate every step).
    local = ScriptedLLM(lambda n: ChatResponseWithLogprobs(
        content="", model="f", avg_logprob=-3.0,
        tool_calls=[ToolCall(id="c", name="echo", arguments={"value": str(n)})],
    ))
    cloud_calls = {"n": 0}

    class Cloud:
        def chat_with_logprobs(self, messages, **opts):
            cloud_calls["n"] += 1
            return ChatResponseWithLogprobs(
                content="", model="cloud", avg_logprob=-0.01,
                tool_calls=[ToolCall(id="cc", name="echo", arguments={"value": "cloud"})],
            )

    ex = Executor(local=local, cloud=Cloud(), tools=reg,
                  router=FixedThresholdRouter(), max_iterations=10, max_escalations=2)
    res = ex.execute("task")

    # Cloud consulted at most the escalation cap, even over 10 iterations.
    assert cloud_calls["n"] == 2
    assert res.escalations == 2


def test_compression_invoked_when_budget_exceeded():
    reg = _reg()
    # Model keeps calling tools with huge content so the trajectory grows.
    big = "y" * 4000
    local = ScriptedLLM(lambda n: (
        ChatResponseWithLogprobs(
            content=big, model="f", avg_logprob=-0.05,
            tool_calls=[ToolCall(id="c", name="echo", arguments={"value": str(n)})],
        ) if n < 4 else
        ChatResponseWithLogprobs(content="done", model="f", avg_logprob=-0.05, tool_calls=[])
    ))
    summarize_calls = {"n": 0}

    def summarize(middle):
        summarize_calls["n"] += 1
        return "COMPRESSED"

    ex = Executor(local=local, cloud=None, tools=reg, router=FixedThresholdRouter(),
                  max_iterations=10, compress_token_budget=500, summarize=summarize)
    res = ex.execute("task", system="SYS")

    assert res.stop_reason == "done"
    # The growing trajectory tripped the compressor at least once.
    assert summarize_calls["n"] >= 1

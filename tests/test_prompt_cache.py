"""Tests for Anthropic prompt-cache breakpoint injection and the executor's
byte-stable prefix invariant."""

from __future__ import annotations

import copy

from autodidact.executor import Executor
from autodidact.llm_client import ChatResponseWithLogprobs, ToolCall
from autodidact.prompt_cache import apply_anthropic_cache_control
from autodidact.routing.step_router import FixedThresholdRouter
from autodidact.tools.registry import ToolRegistry


# ── apply_anthropic_cache_control ────────────────────────────────


def test_system_message_gets_cache_marker():
    msgs = [{"role": "system", "content": "prefix"}, {"role": "user", "content": "hi"}]
    out = apply_anthropic_cache_control(msgs)
    block = out[0]["content"][0]
    assert block["cache_control"] == {"type": "ephemeral"}
    assert block["text"] == "prefix"


def test_does_not_mutate_input():
    msgs = [{"role": "system", "content": "prefix"}]
    snapshot = copy.deepcopy(msgs)
    apply_anthropic_cache_control(msgs)
    assert msgs == snapshot


def test_at_most_four_breakpoints():
    msgs = [{"role": "system", "content": "s"}] + [
        {"role": "user", "content": f"m{i}"} for i in range(10)
    ]
    out = apply_anthropic_cache_control(msgs)
    marked = sum(
        1
        for m in out
        if "cache_control" in m
        or (isinstance(m.get("content"), list) and any("cache_control" in b for b in m["content"]))
    )
    assert marked == 4  # system + last 3


def test_ttl_1h_carried():
    msgs = [{"role": "system", "content": "s"}]
    out = apply_anthropic_cache_control(msgs, cache_ttl="1h")
    assert out[0]["content"][0]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}


def test_tool_result_message_marked_directly():
    msgs = [{"role": "tool", "content": '{"ok": true}', "tool_call_id": "c0"}]
    out = apply_anthropic_cache_control(msgs)
    assert out[0]["cache_control"] == {"type": "ephemeral"}


# ── Executor byte-stable prefix invariant ────────────────────────


class RecordingLLM:
    """Records the messages passed on each call, returns a scripted queue."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.seen_prefixes: list[tuple] = []

    def chat_with_logprobs(self, messages, **opts):
        # Capture the system + first user message (the stable prefix) verbatim.
        self.seen_prefixes.append(
            tuple((m.role, m.content) for m in messages[:2])
        )
        return self._responses.pop(0)


def _reg():
    reg = ToolRegistry()
    reg.register(
        "echo",
        description="echo",
        parameters={"type": "object", "properties": {"value": {"type": "string"}}},
        handler=lambda a: {"echoed": a.get("value")},
        toolset="terminal",
    )
    return reg


def test_prefix_is_byte_stable_across_iterations():
    reg = _reg()
    local = RecordingLLM([
        ChatResponseWithLogprobs(
            content="", model="f", avg_logprob=-0.05,
            tool_calls=[ToolCall(id="c0", name="echo", arguments={"value": "a"})],
        ),
        ChatResponseWithLogprobs(
            content="", model="f", avg_logprob=-0.05,
            tool_calls=[ToolCall(id="c1", name="echo", arguments={"value": "b"})],
        ),
        ChatResponseWithLogprobs(content="done", model="f", avg_logprob=-0.05, tool_calls=[]),
    ])
    ex = Executor(local=local, cloud=None, tools=reg,
                  router=FixedThresholdRouter(), max_iterations=5)
    ex.execute("task", system="STABLE SYSTEM PROMPT")

    # The (system, first-user) prefix must be identical on every iteration —
    # tool results append after it, never rewrite it.
    assert len(set(local.seen_prefixes)) == 1
    assert local.seen_prefixes[0][0] == ("system", "STABLE SYSTEM PROMPT")


def test_context_goes_into_user_message_not_system():
    reg = _reg()
    local = RecordingLLM([
        ChatResponseWithLogprobs(content="done", model="f", avg_logprob=-0.05, tool_calls=[]),
    ])
    ex = Executor(local=local, cloud=None, tools=reg, router=FixedThresholdRouter())
    ex.execute("the task", system="SYS", context="EPHEMERAL CONTEXT")

    roles = dict(local.seen_prefixes[0])
    assert roles["system"] == "SYS"  # system stays clean
    assert "EPHEMERAL CONTEXT" in roles["user"]  # context rode in the user turn


# ── OpenAI backend cache_ttl wiring ──────────────────────────────


def test_openai_backend_injects_cache_control_when_ttl_set():
    from unittest.mock import MagicMock, patch

    from autodidact.llm import OpenAICompatBackend
    from autodidact.llm_client import ChatMessage, LLMConfig

    backend = OpenAICompatBackend(
        LLMConfig(provider="openai", model="claude-via-proxy", api_key="t")
    )
    message = MagicMock(content="ok", tool_calls=None)
    choice = MagicMock(message=message)
    resp = MagicMock(choices=[choice], model="m", usage=MagicMock(prompt_tokens=1, completion_tokens=1))
    client = MagicMock()
    client.chat.completions.create.return_value = resp
    with patch.object(backend, "_get_client", return_value=client):
        backend.chat([ChatMessage(role="system", content="prefix")], cache_ttl="5m")
    sent = client.chat.completions.create.call_args.kwargs["messages"]
    assert sent[0]["content"][0]["cache_control"] == {"type": "ephemeral"}


def test_openai_backend_no_cache_control_by_default():
    from unittest.mock import MagicMock, patch

    from autodidact.llm import OpenAICompatBackend
    from autodidact.llm_client import ChatMessage, LLMConfig

    backend = OpenAICompatBackend(
        LLMConfig(provider="openai", model="gpt-4o", api_key="t")
    )
    message = MagicMock(content="ok", tool_calls=None)
    choice = MagicMock(message=message)
    resp = MagicMock(choices=[choice], model="m", usage=MagicMock(prompt_tokens=1, completion_tokens=1))
    client = MagicMock()
    client.chat.completions.create.return_value = resp
    with patch.object(backend, "_get_client", return_value=client):
        backend.chat([ChatMessage(role="system", content="prefix")])
    sent = client.chat.completions.create.call_args.kwargs["messages"]
    # Plain string content, no cache_control — safe for strict OpenAI.
    assert sent[0]["content"] == "prefix"

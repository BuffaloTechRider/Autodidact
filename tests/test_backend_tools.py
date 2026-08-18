"""Tests for the tool-calling parameter on the LLM backends.

Covers the Ollama and OpenAI-compatible backends: outgoing tool schemas and
tool-result messages are serialized correctly, and incoming tool_calls are
parsed into ToolCall objects. Bedrock rejects tools until its follow-up lands.

The HTTP/SDK layer is mocked — no live model is contacted.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from autodidact.llm import OllamaBackend, OpenAICompatBackend
from autodidact.llm.backend import _message_to_tool_dict
from autodidact.llm_client import (
    ChatMessage,
    LLMClient,
    LLMClientError,
    LLMConfig,
    ToolCall,
)


_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "terminal",
            "description": "run a command",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
    }
]


# ── Shared message serialization ─────────────────────────────────


class TestMessageSerialization:
    def test_plain_message(self):
        m = ChatMessage(role="user", content="hi")
        assert _message_to_tool_dict(m) == {"role": "user", "content": "hi"}

    def test_assistant_with_tool_calls(self):
        m = ChatMessage(
            role="assistant",
            content="",
            tool_calls=[ToolCall(id="call_0", name="terminal", arguments={"command": "ls"})],
        )
        out = _message_to_tool_dict(m)
        assert out["role"] == "assistant"
        tc = out["tool_calls"][0]
        assert tc["id"] == "call_0"
        assert tc["function"]["name"] == "terminal"
        # Arguments are serialized to a JSON string (OpenAI/Ollama wire format).
        assert json.loads(tc["function"]["arguments"]) == {"command": "ls"}

    def test_tool_result_message(self):
        m = ChatMessage(role="tool", content='{"ok": true}', tool_call_id="call_0")
        assert _message_to_tool_dict(m) == {
            "role": "tool",
            "content": '{"ok": true}',
            "tool_call_id": "call_0",
        }


# ── Ollama ───────────────────────────────────────────────────────


class TestOllamaTools:
    def _backend(self):
        return OllamaBackend(LLMConfig(provider="ollama", model="qwen3:8b"))

    def _resp(self, payload: dict):
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = payload
        return resp

    def test_tools_passed_in_request_body(self):
        backend = self._backend()
        resp = self._resp({"message": {"content": "ok"}, "model": "qwen3:8b"})
        with patch("autodidact.llm.ollama.requests.post", return_value=resp) as mock_post:
            backend.chat([ChatMessage(role="user", content="hi")], tools=_TOOLS)
        body = mock_post.call_args.kwargs["json"]
        assert body["tools"] == _TOOLS

    def test_no_tools_key_when_absent(self):
        backend = self._backend()
        resp = self._resp({"message": {"content": "ok"}})
        with patch("autodidact.llm.ollama.requests.post", return_value=resp) as mock_post:
            backend.chat([ChatMessage(role="user", content="hi")])
        assert "tools" not in mock_post.call_args.kwargs["json"]

    def test_parses_tool_calls_from_response(self):
        backend = self._backend()
        resp = self._resp(
            {
                "message": {
                    "content": "",
                    "tool_calls": [
                        {"function": {"name": "terminal", "arguments": {"command": "ls"}}}
                    ],
                }
            }
        )
        with patch("autodidact.llm.ollama.requests.post", return_value=resp):
            out = backend.chat([ChatMessage(role="user", content="hi")], tools=_TOOLS)
        assert len(out.tool_calls) == 1
        assert out.tool_calls[0].name == "terminal"
        assert out.tool_calls[0].arguments == {"command": "ls"}
        # Ollama gives no id → one is synthesized.
        assert out.tool_calls[0].id == "call_0"

    def test_no_tool_calls_is_empty_list(self):
        backend = self._backend()
        resp = self._resp({"message": {"content": "just text"}})
        with patch("autodidact.llm.ollama.requests.post", return_value=resp):
            out = backend.chat([ChatMessage(role="user", content="hi")])
        assert out.tool_calls == []


# ── OpenAI-compatible ────────────────────────────────────────────


class TestOpenAITools:
    def _backend(self):
        return OpenAICompatBackend(
            LLMConfig(provider="openai", model="gpt-4o-mini", api_key="test")
        )

    def _mock_client(self, *, tool_calls=None, content="ok"):
        message = MagicMock()
        message.content = content
        message.tool_calls = tool_calls
        choice = MagicMock()
        choice.message = message
        resp = MagicMock()
        resp.choices = [choice]
        resp.model = "gpt-4o-mini"
        resp.usage = MagicMock(prompt_tokens=1, completion_tokens=1)
        client = MagicMock()
        client.chat.completions.create.return_value = resp
        return client

    def test_tools_passed_to_create(self):
        backend = self._backend()
        client = self._mock_client()
        with patch.object(backend, "_get_client", return_value=client):
            backend.chat([ChatMessage(role="user", content="hi")], tools=_TOOLS)
        kwargs = client.chat.completions.create.call_args.kwargs
        assert kwargs["tools"] == _TOOLS

    def test_tool_choice_forwarded(self):
        backend = self._backend()
        client = self._mock_client()
        with patch.object(backend, "_get_client", return_value=client):
            backend.chat(
                [ChatMessage(role="user", content="hi")],
                tools=_TOOLS,
                tool_choice="auto",
            )
        kwargs = client.chat.completions.create.call_args.kwargs
        assert kwargs["tool_choice"] == "auto"

    def test_no_tools_key_when_absent(self):
        backend = self._backend()
        client = self._mock_client()
        with patch.object(backend, "_get_client", return_value=client):
            backend.chat([ChatMessage(role="user", content="hi")])
        assert "tools" not in client.chat.completions.create.call_args.kwargs

    def test_parses_tool_calls_with_json_string_args(self):
        backend = self._backend()
        fn = MagicMock()
        fn.name = "terminal"
        fn.arguments = '{"command": "ls"}'  # OpenAI sends args as a JSON string
        tc = MagicMock()
        tc.id = "call_abc"
        tc.function = fn
        client = self._mock_client(tool_calls=[tc], content=None)
        with patch.object(backend, "_get_client", return_value=client):
            out = backend.chat([ChatMessage(role="user", content="hi")], tools=_TOOLS)
        assert len(out.tool_calls) == 1
        assert out.tool_calls[0].id == "call_abc"
        assert out.tool_calls[0].name == "terminal"
        assert out.tool_calls[0].arguments == {"command": "ls"}

    def test_malformed_json_args_yields_empty_dict(self):
        backend = self._backend()
        fn = MagicMock()
        fn.name = "terminal"
        fn.arguments = "{not valid json"
        tc = MagicMock()
        tc.id = "call_bad"
        tc.function = fn
        client = self._mock_client(tool_calls=[tc], content=None)
        with patch.object(backend, "_get_client", return_value=client):
            out = backend.chat([ChatMessage(role="user", content="hi")], tools=_TOOLS)
        # Name preserved, arguments default to empty rather than dropping the call.
        assert out.tool_calls[0].name == "terminal"
        assert out.tool_calls[0].arguments == {}


# ── Tools + logprobs together (step-level routing needs both) ────


class TestToolsWithLogprobs:
    """The tiered executor generates a tool call and scores its confidence in a
    single call, so chat_with_logprobs must return BOTH tool_calls and logprobs.
    """

    def test_ollama_logprobs_path_passes_tools_and_parses_calls(self):
        backend = OllamaBackend(LLMConfig(provider="ollama", model="qwen3:8b"))
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {
            "model": "qwen3:8b",
            "message": {
                "content": "",
                "tool_calls": [
                    {"function": {"name": "terminal", "arguments": {"command": "ls"}}}
                ],
            },
            "logprobs": [{"logprob": -0.1}, {"logprob": -0.3}],
        }
        with patch("autodidact.llm.ollama.requests.post", return_value=resp) as mock_post:
            out = backend.chat_with_logprobs(
                [ChatMessage(role="user", content="hi")], tools=_TOOLS
            )
        # tools reached the request body
        assert mock_post.call_args.kwargs["json"]["tools"] == _TOOLS
        # both signals present on the response
        assert out.avg_logprob is not None
        assert len(out.tool_calls) == 1
        assert out.tool_calls[0].name == "terminal"
        assert out.tool_calls[0].arguments == {"command": "ls"}

    def test_openai_logprobs_path_passes_tools_and_parses_calls(self):
        backend = OpenAICompatBackend(
            LLMConfig(provider="openai", model="gpt-4o-mini", api_key="test")
        )
        fn = MagicMock()
        fn.name = "terminal"
        fn.arguments = '{"command": "ls"}'
        tc = MagicMock()
        tc.id = "call_abc"
        tc.function = fn
        message = MagicMock()
        message.content = None
        message.tool_calls = [tc]
        lp_item = MagicMock()
        lp_item.logprob = -0.2
        lp_item.top_logprobs = []
        choice = MagicMock()
        choice.message = message
        choice.logprobs = MagicMock(content=[lp_item])
        resp = MagicMock()
        resp.choices = [choice]
        resp.model = "gpt-4o-mini"
        resp.usage = MagicMock(prompt_tokens=1, completion_tokens=1)
        client = MagicMock()
        client.chat.completions.create.return_value = resp
        with patch.object(backend, "_get_client", return_value=client):
            out = backend.chat_with_logprobs(
                [ChatMessage(role="user", content="hi")], tools=_TOOLS
            )
        kwargs = client.chat.completions.create.call_args.kwargs
        assert kwargs["tools"] == _TOOLS
        assert kwargs["logprobs"] is True
        assert out.avg_logprob is not None
        assert len(out.tool_calls) == 1
        assert out.tool_calls[0].name == "terminal"
        assert out.tool_calls[0].arguments == {"command": "ls"}


# ── Bedrock ──────────────────────────────────────────────────────
# Bedrock Converse tool-calling is implemented; its coverage (schema
# conversion, toolUse/toolResult mapping, round trip) lives in
# tests/test_bedrock_tools.py.

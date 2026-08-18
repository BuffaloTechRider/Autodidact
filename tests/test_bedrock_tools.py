"""Bedrock Converse tool-calling: schema conversion, message mapping, parsing.

Uses a mocked boto3 client (injected via backend._client) so no AWS calls fire.
Covers the round trip the executor's cloud-escalation path needs: attach tool
schemas → parse toolUse from the response → feed toolResult back.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from autodidact.llm import BedrockBackend
from autodidact.llm.bedrock import _parse_converse_tool_uses, _to_tool_config
from autodidact.llm_client import ChatMessage, LLMConfig, ToolCall
from autodidact.tools import REGISTRY


def _backend() -> BedrockBackend:
    b = BedrockBackend(LLMConfig(provider="bedrock", model="anthropic.claude"))
    b._client = MagicMock()
    return b


class TestToolConfigConversion:
    def test_openai_schema_becomes_converse_toolspec(self):
        cfg = _to_tool_config(REGISTRY.get_schemas())
        assert cfg is not None and "tools" in cfg
        spec = cfg["tools"][0]["toolSpec"]
        assert "name" in spec and "inputSchema" in spec
        assert "json" in spec["inputSchema"]  # Converse wraps the JSON schema

    def test_none_and_empty_return_none(self):
        assert _to_tool_config(None) is None
        assert _to_tool_config([]) is None


class TestParseToolUse:
    def test_parses_tooluse_blocks(self):
        parts = [
            {"text": "let me look"},
            {"toolUse": {"toolUseId": "tu_1", "name": "read_file", "input": {"path": "a.txt"}}},
        ]
        calls = _parse_converse_tool_uses(parts)
        assert len(calls) == 1
        assert calls[0].id == "tu_1"
        assert calls[0].name == "read_file"
        assert calls[0].arguments == {"path": "a.txt"}

    def test_no_tooluse_returns_empty(self):
        assert _parse_converse_tool_uses([{"text": "just prose"}]) == []


class TestMessageMapping:
    def test_assistant_tool_call_maps_to_tooluse_block(self):
        b = _backend()
        msgs = [ChatMessage(
            role="assistant", content="",
            tool_calls=[ToolCall(id="tu_1", name="terminal", arguments={"command": "ls"})],
        )]
        _system, converse = b._to_messages(msgs)
        block = converse[0]["content"][0]
        assert block["toolUse"]["toolUseId"] == "tu_1"
        assert block["toolUse"]["name"] == "terminal"
        assert block["toolUse"]["input"] == {"command": "ls"}

    def test_tool_result_maps_to_user_toolresult(self):
        b = _backend()
        msgs = [ChatMessage(role="tool", content='{"ok": true}', tool_call_id="tu_1")]
        _system, converse = b._to_messages(msgs)
        # Converse requires toolResult inside a user message.
        assert converse[0]["role"] == "user"
        tr = converse[0]["content"][0]["toolResult"]
        assert tr["toolUseId"] == "tu_1"
        assert tr["content"][0]["text"] == '{"ok": true}'

    def test_parallel_tool_results_merge_into_one_user_turn(self):
        b = _backend()
        msgs = [
            ChatMessage(role="tool", content="r1", tool_call_id="tu_1"),
            ChatMessage(role="tool", content="r2", tool_call_id="tu_2"),
        ]
        _system, converse = b._to_messages(msgs)
        assert len(converse) == 1
        assert len(converse[0]["content"]) == 2


class TestChatToolRoundTrip:
    def test_chat_attaches_toolconfig_and_parses_tooluse(self):
        b = _backend()
        b._client.converse.return_value = {
            "output": {"message": {"content": [
                {"toolUse": {"toolUseId": "tu_9", "name": "list_directory", "input": {"path": "."}}},
            ]}},
            "usage": {"inputTokens": 10, "outputTokens": 5},
            "stopReason": "tool_use",
        }
        resp = b.chat(
            [ChatMessage(role="user", content="list the dir")],
            tools=REGISTRY.get_schemas(),
        )
        # toolConfig was attached to the API call.
        _args, kwargs = b._client.converse.call_args
        assert "toolConfig" in kwargs
        # The toolUse came back as a ToolCall.
        assert len(resp.tool_calls) == 1
        assert resp.tool_calls[0].name == "list_directory"

    def test_chat_without_tools_omits_toolconfig(self):
        b = _backend()
        b._client.converse.return_value = {
            "output": {"message": {"content": [{"text": "hi"}]}},
            "usage": {"inputTokens": 3, "outputTokens": 1},
        }
        resp = b.chat([ChatMessage(role="user", content="hi")])
        _args, kwargs = b._client.converse.call_args
        assert "toolConfig" not in kwargs
        assert resp.content == "hi"
        assert resp.tool_calls == []

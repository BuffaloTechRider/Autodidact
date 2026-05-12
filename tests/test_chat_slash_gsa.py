"""Tests for the /gsa chat slash command.

User types /gsa v4 in the chat REPL -> GSA probe is rebuilt with v4 prompt
for the rest of the session. /gsa v3 flips back. /gsa alone reports current
version.

Session-only: no config write, no agent rebuild, just swap the probe.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from autodidact.cli import _dispatch_slash, _handle_gsa_command
from autodidact.signals.grounded_self_assessment import (
    PROMPT_VERSION,
    PROMPT_VERSION_V2,
    PROMPT_VERSION_V4,
    SelfAssessment,
)


class _FakeAgent:
    """Minimal stand-in — only the attrs the slash handler reads/writes."""

    def __init__(self, gsa=None):
        self._local_client = MagicMock()
        self._gsa = gsa  # may be None (not yet built) or a SelfAssessment


class TestHandleGsaCommand:
    """_handle_gsa_command rebuilds agent._gsa with the requested version."""

    def test_set_v4(self):
        agent = _FakeAgent()
        _handle_gsa_command(agent, "/gsa v4")
        assert isinstance(agent._gsa, SelfAssessment)
        assert agent._gsa.prompt_version == PROMPT_VERSION_V4

    def test_set_v3(self):
        agent = _FakeAgent()
        _handle_gsa_command(agent, "/gsa v3")
        assert agent._gsa.prompt_version == PROMPT_VERSION

    def test_set_v2(self):
        agent = _FakeAgent()
        _handle_gsa_command(agent, "/gsa v2")
        assert agent._gsa.prompt_version == PROMPT_VERSION_V2

    def test_switch_versions(self):
        """v3 -> v4 -> v3 works without leaking state."""
        agent = _FakeAgent()
        _handle_gsa_command(agent, "/gsa v4")
        assert agent._gsa.prompt_version == PROMPT_VERSION_V4
        _handle_gsa_command(agent, "/gsa v3")
        assert agent._gsa.prompt_version == PROMPT_VERSION

    def test_invalid_version_leaves_unchanged(self):
        """/gsa v99 prints an error and keeps the existing probe."""
        existing = MagicMock(spec=SelfAssessment)
        existing.prompt_version = PROMPT_VERSION
        agent = _FakeAgent(gsa=existing)
        _handle_gsa_command(agent, "/gsa v99")
        assert agent._gsa is existing  # unchanged

    def test_status_with_no_arg_when_probe_built(self, capsys):
        """/gsa alone reports current version."""
        probe = MagicMock(spec=SelfAssessment)
        probe.prompt_version = PROMPT_VERSION_V4
        agent = _FakeAgent(gsa=probe)
        _handle_gsa_command(agent, "/gsa")
        out = capsys.readouterr().out
        assert "v4" in out or PROMPT_VERSION_V4 in out

    def test_status_with_no_probe_built_yet(self, capsys):
        """/gsa before any query reports the default (v3)."""
        agent = _FakeAgent(gsa=None)
        _handle_gsa_command(agent, "/gsa")
        out = capsys.readouterr().out
        assert "v3" in out.lower() or "default" in out.lower()

    def test_help_lists_versions(self, capsys):
        """/gsa help prints the available versions."""
        agent = _FakeAgent()
        _handle_gsa_command(agent, "/gsa help")
        out = capsys.readouterr().out
        assert "v2" in out and "v3" in out and "v4" in out


class TestDispatchSlash:
    """_dispatch_slash returns True iff it handled a slash command."""

    def test_gsa_routed(self):
        agent = _FakeAgent()
        handled = _dispatch_slash(agent, "/gsa v4", renderer=None)
        assert handled is True
        assert agent._gsa.prompt_version == PROMPT_VERSION_V4

    def test_plain_text_not_handled(self):
        agent = _FakeAgent()
        handled = _dispatch_slash(agent, "what is 2+2?", renderer=None)
        assert handled is False

    def test_wrong_still_routed(self):
        """/wrong predates this PR — must still work."""
        agent = _FakeAgent()
        agent._history = []  # no previous query
        agent.correct = MagicMock()
        handled = _dispatch_slash(agent, "/wrong", renderer=MagicMock())
        assert handled is True  # handled (even though it prints a no-history warning)

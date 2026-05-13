"""Tests for the /cloud chat slash command.

Two forms:
  /cloud            -> alias of /wrong: re-route last question to cloud,
                       invalidate the matching memory entry if any
  /cloud <text>     -> force-escalate a NEW question to cloud, skipping
                       memory/GSA/local

Both paths reuse Agent.correct() — the invalidation behavior is correct
whether or not a memory entry exists (no-op if the query was answered
locally, so nothing was ever stored).
"""

from __future__ import annotations

from unittest.mock import MagicMock

from autodidact.cli import _dispatch_slash, _handle_cloud_command


class _FakeAgent:
    def __init__(self):
        self._history = []
        self.correct = MagicMock(return_value=MagicMock(answer="cloud answer"))


class TestHandleCloudCommand:

    def test_cloud_alone_reroutes_last_question(self):
        """/cloud with no arg re-sends the most recent user turn via correct()."""
        agent = _FakeAgent()
        agent._history = [
            {"role": "user", "content": "what is the capital of france?"},
            {"role": "assistant", "content": "Paris (probably)"},
        ]
        _handle_cloud_command(agent, "/cloud", renderer=MagicMock())
        agent.correct.assert_called_once(); assert agent.correct.call_args.args == ("what is the capital of france?",)

    def test_cloud_alone_finds_most_recent_user_turn(self):
        """Even with multiple past turns, pick the latest user one."""
        agent = _FakeAgent()
        agent._history = [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "q2"},
            {"role": "assistant", "content": "a2"},
        ]
        _handle_cloud_command(agent, "/cloud", renderer=MagicMock())
        agent.correct.assert_called_once(); assert agent.correct.call_args.args == ("q2",)

    def test_cloud_with_new_question(self):
        """/cloud <text> sends <text> via correct()."""
        agent = _FakeAgent()
        _handle_cloud_command(agent, "/cloud what is 2+2?", renderer=MagicMock())
        agent.correct.assert_called_once(); assert agent.correct.call_args.args == ("what is 2+2?",)

    def test_cloud_with_multiword_question(self):
        agent = _FakeAgent()
        _handle_cloud_command(
            agent,
            "/cloud who invented the transistor?",
            renderer=MagicMock(),
        )
        agent.correct.assert_called_once(); assert agent.correct.call_args.args == ("who invented the transistor?",)

    def test_cloud_alone_with_no_history(self, capsys):
        """/cloud with no history prints a helpful message, no cloud call."""
        agent = _FakeAgent()
        agent._history = []
        _handle_cloud_command(agent, "/cloud", renderer=MagicMock())
        agent.correct.assert_not_called()
        out = capsys.readouterr().out
        assert "no" in out.lower()

    def test_cloud_strips_whitespace(self):
        """Leading/trailing whitespace in the arg is stripped."""
        agent = _FakeAgent()
        _handle_cloud_command(agent, "/cloud   hello world   ", renderer=MagicMock())
        agent.correct.assert_called_once(); assert agent.correct.call_args.args == ("hello world",)


class TestDispatchCloud:
    """_dispatch_slash routes /cloud correctly and doesn't break existing commands."""

    def test_cloud_alone_routed(self):
        agent = _FakeAgent()
        agent._history = [
            {"role": "user", "content": "q?"},
            {"role": "assistant", "content": "a"},
        ]
        handled = _dispatch_slash(agent, "/cloud", renderer=MagicMock())
        assert handled is True
        agent.correct.assert_called_once(); assert agent.correct.call_args.args == ("q?",)

    def test_cloud_with_text_routed(self):
        agent = _FakeAgent()
        handled = _dispatch_slash(agent, "/cloud what is X?", renderer=MagicMock())
        assert handled is True
        agent.correct.assert_called_once(); assert agent.correct.call_args.args == ("what is X?",)

    def test_plain_text_not_handled(self):
        agent = _FakeAgent()
        handled = _dispatch_slash(agent, "what is X?", renderer=MagicMock())
        assert handled is False
        agent.correct.assert_not_called()

    def test_wrong_still_works(self):
        """/wrong is unchanged — it also calls correct(), no surprises."""
        agent = _FakeAgent()
        agent._history = [
            {"role": "user", "content": "q?"},
            {"role": "assistant", "content": "wrong answer"},
        ]
        handled = _dispatch_slash(agent, "/wrong", renderer=MagicMock())
        assert handled is True
        agent.correct.assert_called_once(); assert agent.correct.call_args.args == ("q?",)

    def test_gsa_still_works(self):
        """/gsa still routes to its own handler (regression check)."""
        from autodidact.signals.grounded_self_assessment import SelfAssessment
        agent = _FakeAgent()
        agent._local_client = MagicMock()
        agent._gsa = None
        handled = _dispatch_slash(agent, "/gsa v4", renderer=MagicMock())
        assert handled is True
        assert isinstance(agent._gsa, SelfAssessment)

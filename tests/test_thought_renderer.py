"""Tests for ThoughtRenderer — visible learning UX.

The renderer formats the agent's internal steps for terminal output.
It takes QueryResponse / SavingsReport dataclasses and produces
human-readable output with tags like [THINKING], [MEMORY], [LOCAL], etc.
"""

from __future__ import annotations

from io import StringIO

import pytest

from autodidact.agent import QueryResponse, SavingsReport
from autodidact.thought_renderer import ThoughtRenderer


@pytest.fixture
def renderer():
    """A ThoughtRenderer that writes to a captured StringIO instead of stdout."""
    buf = StringIO()
    return ThoughtRenderer(file=buf), buf


class TestRenderThinking:
    """2.1 — Thought process tags."""

    def test_render_thinking_outputs_tag(self, renderer):
        r, buf = renderer
        r.render_thinking("Checking memory... found 3 similar entries")
        output = buf.getvalue()
        assert "[THINKING]" in output
        assert "Checking memory" in output

    def test_render_thinking_multiple_steps(self, renderer):
        r, buf = renderer
        r.render_thinking("Step 1")
        r.render_thinking("Step 2")
        output = buf.getvalue()
        assert output.count("[THINKING]") == 2


class TestRenderResponse:
    """2.1 + 2.2 — Full response rendering with route tag, cost, confidence."""

    def test_local_route_shows_local_tag(self, renderer):
        r, buf = renderer
        resp = QueryResponse(
            answer="Paris is the capital of France.",
            routed_to="local",
            confidence=0.87,
            cost_usd=0.0,
            learned=False,
            latency_ms=500,
        )
        r.render_response(resp)
        output = buf.getvalue()
        assert "[LOCAL]" in output
        assert "Paris is the capital of France." in output

    def test_cloud_route_shows_cloud_tag_and_learned(self, renderer):
        r, buf = renderer
        resp = QueryResponse(
            answer="The GDP of France is $2.78 trillion.",
            routed_to="cloud",
            confidence=0.34,
            cost_usd=0.003,
            learned=True,
            latency_ms=1200,
        )
        r.render_response(resp)
        output = buf.getvalue()
        assert "[CLOUD]" in output
        assert "[LEARNED]" in output

    def test_memory_route_shows_memory_tag_and_source(self, renderer):
        r, buf = renderer
        resp = QueryResponse(
            answer="Paris is the capital of France.",
            routed_to="memory",
            confidence=0.91,
            cost_usd=0.0,
            learned=False,
            latency_ms=50,
            memory_source="What is the capital of France?",
            memory_age_days=2.0,
        )
        r.render_response(resp)
        output = buf.getvalue()
        assert "[MEMORY]" in output
        assert "2" in output  # age in days
        assert "What is the capital of France?" in output

    def test_cost_display_zero_for_local(self, renderer):
        """R2 AC3: Cost shown per response — $0.00 for local."""
        r, buf = renderer
        resp = QueryResponse(
            answer="Answer", routed_to="local", confidence=0.9,
            cost_usd=0.0, learned=False, latency_ms=100,
        )
        r.render_response(resp)
        output = buf.getvalue()
        assert "$0.00" in output

    def test_cost_display_nonzero_for_cloud(self, renderer):
        """R2 AC3: Cost shown per response — actual cost for cloud."""
        r, buf = renderer
        resp = QueryResponse(
            answer="Answer", routed_to="cloud", confidence=0.3,
            cost_usd=0.0042, learned=True, latency_ms=1000,
        )
        r.render_response(resp)
        output = buf.getvalue()
        assert "$0.004" in output  # should show actual cost

    def test_route_displayed(self, renderer):
        """Route shown per response."""
        r, buf = renderer
        resp = QueryResponse(
            answer="Answer", routed_to="local", confidence=0.87,
            cost_usd=0.0, learned=False, latency_ms=100,
        )
        r.render_response(resp)
        output = buf.getvalue()
        assert "local" in output


class TestRenderSessionSummary:
    """2.3 — Session summary on exit."""

    def test_session_summary_shows_totals(self, renderer):
        """R2 AC5: queries total, local, local+memory, cloud, cost, knowledge learned."""
        r, buf = renderer
        report = SavingsReport(
            total_queries=10,
            local_queries=6,
            cloud_queries=3,
            memory_queries=1,
            total_cost_usd=0.012,
            estimated_all_cloud_cost_usd=0.10,
            saved_usd=0.088,
            saved_pct=88.0,
            facts_learned=3,
        )
        r.render_session_summary(report)
        output = buf.getvalue()
        assert "10" in output  # total queries
        assert "Local: 6" in output  # local only
        assert "Local + Memory: 1" in output  # memory-assisted
        assert "Cloud: 3" in output  # cloud
        assert "$0.012" in output or "$0.01" in output  # total cost
        assert "Knowledge learned: 3" in output

    def test_session_summary_zero_queries(self, renderer):
        """Edge case: no queries in session."""
        r, buf = renderer
        report = SavingsReport()
        r.render_session_summary(report)
        output = buf.getvalue()
        assert "0" in output  # total queries = 0

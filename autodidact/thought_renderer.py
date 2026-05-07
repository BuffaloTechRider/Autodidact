"""Visible Learning UX — ThoughtRenderer.

Formats the agent's internal reasoning steps for terminal output.
Tags: [THINKING], [MEMORY], [LOCAL], [CLOUD], [LEARNED].

Usage:
    from autodidact.thought_renderer import ThoughtRenderer
    renderer = ThoughtRenderer()
    renderer.render_thinking("Checking memory...")
    renderer.render_response(query_response)
    renderer.render_session_summary(savings_report)
"""

from __future__ import annotations

import sys
from typing import IO, Optional

from rich.console import Console
from rich.text import Text

from autodidact.agent import QueryResponse, SavingsReport

# ── Tag styles ─────────────────────────────────────────────────────
_TAG_STYLES = {
    "THINKING": "dim cyan",
    "MEMORY": "bold magenta",
    "LOCAL": "bold green",
    "CLOUD": "bold yellow",
    "LEARNED": "bold blue",
}


class ThoughtRenderer:
    """Formats agent reasoning for terminal display using rich."""

    def __init__(self, *, file: Optional[IO[str]] = None) -> None:
        self._console = Console(file=file or sys.stderr, highlight=False)

    # ── Individual thought steps ───────────────────────────────────

    def render_thinking(self, message: str) -> None:
        """Show a [THINKING] step."""
        self._tag("THINKING", message)

    # ── Full response rendering ────────────────────────────────────

    def render_response(self, resp: QueryResponse) -> None:
        """Render a complete query response with route tag, answer, cost, and confidence."""
        route = resp.routed_to.upper()  # "local" → "LOCAL"

        # Route tag + answer.
        self._tag(route, resp.answer)

        # Memory source attribution (R2 AC4).
        if resp.routed_to == "memory" and resp.memory_source:
            age = int(resp.memory_age_days) if resp.memory_age_days else "?"
            self._console.print(
                f"  ↳ Recalled from: \"{resp.memory_source}\" (learned {age} days ago)",
                style="dim",
            )

        # Learned tag.
        if resp.learned:
            self._tag("LEARNED", "✅ Stored for future reference")

        # Cost + confidence line (R2 AC2, AC3).
        cost_str = f"${resp.cost_usd:.3f}" if resp.cost_usd > 0 else "$0.00"
        parts = [f"💰 {cost_str}", f"Confidence: {resp.confidence:.2f}", f"Route: {resp.routed_to}"]
        if resp.learned:
            parts.append("✅ Learned")
        self._console.print("  " + " | ".join(parts), style="dim")

    # ── Session summary ────────────────────────────────────────────

    def render_session_summary(self, report: SavingsReport) -> None:
        """Render end-of-session summary (R2 AC5)."""
        total = report.total_queries
        self._console.print()
        self._console.print("─── Session Summary ───", style="bold")

        if total == 0:
            self._console.print("  0 queries this session.")
            return

        local_pct = round(100 * report.local_queries / total)
        cloud_pct = round(100 * report.cloud_queries / total)
        memory_pct = round(100 * report.memory_queries / total)

        self._console.print(f"  Queries: {total}")
        self._console.print(f"  Local: {report.local_queries} ({local_pct}%)")
        self._console.print(f"  Cloud: {report.cloud_queries} ({cloud_pct}%)")
        self._console.print(f"  Memory: {report.memory_queries} ({memory_pct}%)")
        self._console.print(f"  Cost: ${report.total_cost_usd:.3f}")
        if report.saved_usd > 0:
            self._console.print(f"  Saved: ${report.saved_usd:.3f} ({report.saved_pct:.0f}%)")
        self._console.print(f"  Facts learned: {report.facts_learned}")

    # ── Helpers ─────────────────────────────────────────────────────

    def _tag(self, tag: str, message: str) -> None:
        """Print a tagged line like '[THINKING] message'."""
        style = _TAG_STYLES.get(tag, "")
        self._console.print(f"[{tag}] {message}", style=style)

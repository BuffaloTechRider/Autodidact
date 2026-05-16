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

    def __init__(self, *, file: Optional[IO[str]] = None, verbose: bool = False) -> None:
        self._console = Console(file=file or sys.stderr, highlight=False)
        self.verbose = verbose

    # ── Individual thought steps ───────────────────────────────────

    def render_thinking(self, message: str) -> None:
        """Show a [THINKING] step."""
        self._tag("THINKING", message)

    # ── Full response rendering ────────────────────────────────────

    def render_response(self, resp: QueryResponse) -> None:
        """Render a complete query response with route tag, answer, cost, and confidence."""
        route = resp.routed_to.upper()  # "local" → "LOCAL"

        # If the spinner already streamed the body live, skip re-printing it
        # (would double the answer). The spinner sets _already_streamed on
        # the response when it's done so. Footer still prints below.
        already_streamed = bool(getattr(resp, "_already_streamed", False))

        # Route tag + answer.
        if not already_streamed:
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

        # Context sources (shows WHERE the answer came from).
        if resp.context_sources:
            ctx_str = ", ".join(resp.context_sources)
            self._console.print(f"  ↳ Context: {ctx_str}", style="dim")

        # Verbose debug info (routing signals).
        if self.verbose:
            debug_parts: list[str] = []
            if resp.memory_similarity is not None:
                debug_parts.append(f"mem_sim={resp.memory_similarity:.2f}")
            if resp.gsa_p_yes is not None:
                debug_parts.append(f"gsa={resp.gsa_p_yes:.2f}")
            if resp.escalated_on_refusal:
                debug_parts.append("refusal_detected")
            if resp.escalated_on_gsa:
                debug_parts.append("gsa_vetoed")
            if debug_parts:
                self._console.print(f"  [debug] {' | '.join(debug_parts)}", style="dim yellow")

        # Cost + route line.
        cost_str = f"${resp.cost_usd:.3f}" if resp.cost_usd > 0 else "$0.00"
        parts = [f"💰 {cost_str}", f"Route: {resp.routed_to}"]
        if resp.learned:
            parts.append("✅ Learned")
        if resp.routed_to in ("local", "memory") and not resp.learned:
            parts.append("Type /cloud if wrong")
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

        local_only = report.local_queries
        local_memory = report.memory_queries
        local_total = local_only + local_memory
        local_pct = round(100 * local_total / total)
        memory_pct = round(100 * local_memory / total)
        cloud_pct = round(100 * report.cloud_queries / total)

        self._console.print(f"  Queries: {total}")
        self._console.print(f"  Local: {local_only} ({round(100 * local_only / total)}%)")
        if local_memory > 0:
            self._console.print(f"  Local + Memory: {local_memory} ({memory_pct}%)")
        self._console.print(f"  Cloud: {report.cloud_queries} ({cloud_pct}%)")
        self._console.print(f"  Cost: ${report.total_cost_usd:.3f}")
        if report.saved_usd > 0:
            self._console.print(f"  Saved: ${report.saved_usd:.3f} ({report.saved_pct:.0f}%)")
        self._console.print(f"  Knowledge learned: {report.facts_learned}")

    # ── Helpers ─────────────────────────────────────────────────────

    def _tag(self, tag: str, message: str) -> None:
        """Print a tagged line like '[THINKING] message'."""
        style = _TAG_STYLES.get(tag, "")
        self._console.print(f"[{tag}] {message}", style=style)

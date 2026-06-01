"""Smoke test run after ``autodidact init`` writes the config.

Checks that the configured agent can be constructed and answer a tiny
canned question. Errors are categorized into actionable hints
(Ollama down, model not pulled, missing API key, Bedrock validation,
AWS credentials) rather than raw tracebacks.
"""

from __future__ import annotations

from autodidact.setup_wizard._console import console


def _agent_from_config(config: dict):
    """Lazy import — agent module pulls in the routing pipeline + LLM clients."""
    from autodidact.cli import _agent_from_config as _impl
    return _impl(config)


def _run_smoke_test(config: dict) -> None:
    """Run a quick smoke test to verify the configured models are reachable.

    Categorizes common errors and gives actionable next steps — better than
    surfacing raw tracebacks.
    """
    try:
        agent = _agent_from_config(config)
        console.print("\nRunning smoke test...", style="dim")
        resp = agent.query("What is 2+2?")
        console.print(f"  ✓ Smoke test: routed to [cyan]{resp.routed_to}[/cyan]", style="dim")
    except Exception as e:
        _render_smoke_test_error(e, config)


def _render_smoke_test_error(exc: Exception, config: dict) -> None:
    """Print a human-friendly diagnostic for a smoke-test failure."""
    message = str(exc)
    lower = message.lower()

    console.print()
    console.print("[yellow]⚠ Smoke test failed.[/yellow]", style="bold")
    console.print(f"  Error: {message[:300]}", style="dim")
    console.print()

    hints: list[str] = []

    if "ollama" in lower and ("connection" in lower or "refused" in lower or "timeout" in lower):
        hints.append("Ollama doesn't seem to be running. Start it with: [cyan]ollama serve[/cyan]")
    if "model" in lower and ("not found" in lower or "404" in lower):
        local_model = config.get("local", {}).get("model", "")
        if local_model:
            hints.append(
                f"The model [cyan]{local_model}[/cyan] isn't pulled. "
                f"Run: [cyan]ollama pull {local_model}[/cyan]"
            )
    if "api_key" in lower or "unauthorized" in lower or "401" in lower or "403" in lower:
        hints.append(
            "API key may be invalid or missing. Re-check the key in your config, "
            "or set the provider's env var (OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.)."
        )
    if "credential" in lower or "nocredentialserror" in lower:
        hints.append(
            "AWS credentials not found. Set AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY, "
            "configure [cyan]aws configure[/cyan], or re-run init and pick IAM User auth mode."
        )
    if "validationexception" in lower:
        hints.append(
            "Bedrock rejected the model ID. Check your model name matches a real Bedrock "
            "model ID (e.g. [cyan]anthropic.claude-sonnet-4-5-20250929-v1:0[/cyan])."
        )

    if hints:
        console.print("  Likely cause:", style="bold")
        for hint in hints:
            console.print(f"    • {hint}")
    else:
        console.print(
            "  Your config was written but the agent could not reach any model. "
            "Check your settings and run [cyan]autodidact query \"hello\"[/cyan] to retry.",
            style="dim",
        )
    console.print()


__all__ = [
    "_render_smoke_test_error",
    "_run_smoke_test",
]

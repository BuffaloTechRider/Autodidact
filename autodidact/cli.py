"""Autodidact CLI — the primary user experience.

Commands:
    autodidact init          Interactive config generation
    autodidact chat          Interactive chat with visible thought process
    autodidact query "q"     Single query mode
    autodidact savings       Cumulative cost savings and learning stats
    autodidact memory stats  Knowledge store info
    autodidact memory search Search learned knowledge
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import typer
import yaml
from rich.console import Console

from autodidact.agent import Agent, QueryResponse, SavingsReport
from autodidact.thought_renderer import ThoughtRenderer

app = typer.Typer(help="Autodidact — self-learning AI agent")
memory_app = typer.Typer(help="Knowledge store commands")
app.add_typer(memory_app, name="memory")

console = Console()

# ── Config loading ─────────────────────────────────────────────────

_DEFAULT_CONFIG_PATH = Path("~/.autodidact/config.yaml").expanduser()


def _load_config(path: Path) -> dict:
    """Load config YAML, with env var overrides."""
    if not path.exists():
        return {}
    with open(path) as f:
        config = yaml.safe_load(f) or {}

    # Env var overrides (R8 AC3).
    import os

    if os.environ.get("OPENAI_API_KEY"):
        config.setdefault("cloud", {})["api_key"] = os.environ["OPENAI_API_KEY"]
    if os.environ.get("AUTODIDACT_MODEL"):
        config.setdefault("local", {})["model"] = os.environ["AUTODIDACT_MODEL"]

    return config


def _agent_from_config(config: dict) -> Agent:
    """Create an Agent from a config dict."""
    local_model = config.get("local", {}).get("model")
    if local_model:
        local_model = f"ollama/{local_model}"

    cloud_model = None
    cloud_provider = config.get("cloud", {}).get("provider", "openai")
    cloud_api_key = config.get("cloud", {}).get("api_key")
    if config.get("cloud", {}).get("model"):
        cloud_model = f"{cloud_provider}/{config['cloud']['model']}"

    embedding_model = config.get("local", {}).get("embedding_model")
    db_path = config.get("memory", {}).get("path", "~/.autodidact/memory.db")
    threshold = config.get("routing", {}).get("confidence_threshold", 0.7)

    kwargs: dict = dict(
        local_model=local_model,
        cloud_model=cloud_model,
        cloud_provider=cloud_provider,
        db_path=db_path,
        confidence_threshold=threshold,
    )
    if embedding_model:
        kwargs["embedding_model"] = embedding_model
    if cloud_api_key:
        import os
        os.environ.setdefault("OPENAI_API_KEY", cloud_api_key)

    return Agent(**kwargs)


def _get_agent(config_path: Optional[Path] = None) -> Agent:
    """Load config and create agent."""
    path = config_path or _DEFAULT_CONFIG_PATH
    config = _load_config(path)
    return _agent_from_config(config)


# ── Commands ───────────────────────────────────────────────────────


@app.command()
def init(
    config_path: Optional[str] = typer.Option(
        None, "--config-path", help="Path to write config file"
    ),
) -> None:
    """Interactive config generation (R8)."""
    out_path = Path(config_path) if config_path else _DEFAULT_CONFIG_PATH

    # Ask: local model name.
    local_model = typer.prompt("Local model name", default="qwen2.5:7b")

    # Ask: cloud provider (or skip).
    cloud_input = typer.prompt(
        "Cloud provider (openai/bedrock/skip)", default="skip"
    )

    cloud_config: dict = {}
    if cloud_input.lower() != "skip":
        api_key = typer.prompt("API key")
        cloud_config = {
            "provider": cloud_input.lower(),
            "model": "gpt-4o" if cloud_input.lower() == "openai" else "claude-sonnet-4-5",
            "api_key": api_key,
        }

    # Ask: memory DB path.
    db_path = typer.prompt("Memory DB path", default="~/.autodidact/memory.db")

    # Build config.
    config: dict = {
        "local": {
            "model": local_model,
            "embedding_model": "qllama/bge-large-en-v1.5",
        },
        "routing": {
            "confidence_threshold": 0.7,
        },
        "memory": {
            "path": db_path,
        },
    }
    if cloud_config:
        config["cloud"] = cloud_config

    # Write config YAML.
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    console.print(f"\nConfig written to {out_path}", style="green")

    # Smoke test.
    _run_smoke_test(config)

    console.print("\n✅ Ready! Run `autodidact chat` to start.", style="bold green")


def _run_smoke_test(config: dict) -> None:
    """Run a quick smoke test to verify models are reachable."""
    try:
        agent = _agent_from_config(config)
        console.print("\nRunning smoke test...", style="dim")
        resp = agent.query("What is 2+2?")
        console.print(f"  Local test: {resp.routed_to} — OK", style="dim")
    except Exception as e:
        console.print(f"  Smoke test warning: {e}", style="yellow")


@app.command()
def chat(
    config_path: Optional[str] = typer.Option(None, "--config-path"),
) -> None:
    """Interactive chat with visible thought process."""
    path = Path(config_path) if config_path else None
    agent = _get_agent(path)
    renderer = ThoughtRenderer()

    console.print("Autodidact chat — type 'quit' or 'exit' to stop.\n", style="bold")

    while True:
        try:
            line = typer.prompt("you", prompt_suffix="> ")
        except (KeyboardInterrupt, EOFError):
            break

        if line.strip().lower() in ("quit", "exit", "q"):
            break

        if not line.strip():
            continue

        # User correction flow.
        if line.strip().lower() in ("/wrong", "/correct", "that's wrong"):
            # Re-escalate the last question.
            if agent._history:
                last_q = agent._history[-2]["content"] if len(agent._history) >= 2 else line
                resp = agent.correct(last_q)
                renderer.render_response(resp)
            else:
                console.print("No previous question to correct.", style="yellow")
            continue

        resp = agent.query(line.strip())
        renderer.render_response(resp)

    # Session summary on exit.
    report = agent.savings()
    renderer.render_session_summary(report)


@app.command()
def query(
    question: str = typer.Argument(..., help="Question to ask"),
    config_path: Optional[str] = typer.Option(None, "--config-path"),
) -> None:
    """Single query mode."""
    path = Path(config_path) if config_path else None
    agent = _get_agent(path)
    renderer = ThoughtRenderer()

    resp = agent.query(question)
    renderer.render_response(resp)


@app.command()
def savings(
    config_path: Optional[str] = typer.Option(None, "--config-path"),
) -> None:
    """Cumulative cost savings and learning stats."""
    path = Path(config_path) if config_path else None
    agent = _get_agent(path)
    report = agent.savings()

    console.print("─── Savings Report ───", style="bold")
    console.print(f"  Total queries:  {report.total_queries}")
    console.print(f"  Local:          {report.local_queries}")
    console.print(f"  Cloud:          {report.cloud_queries}")
    console.print(f"  Memory:         {report.memory_queries}")
    console.print(f"  Total cost:     ${report.total_cost_usd:.3f}")
    if report.estimated_all_cloud_cost_usd > 0:
        console.print(f"  All-cloud est:  ${report.estimated_all_cloud_cost_usd:.3f}")
        console.print(f"  Saved:          ${report.saved_usd:.3f} ({report.saved_pct:.0f}%)")
    console.print(f"  Facts learned:  {report.facts_learned}")


# ── Memory sub-commands ────────────────────────────────────────────


@memory_app.command("stats")
def memory_stats(
    config_path: Optional[str] = typer.Option(None, "--config-path"),
) -> None:
    """Knowledge store size, recent entries, domain breakdown."""
    path = Path(config_path) if config_path else None
    agent = _get_agent(path)

    total = agent.memory.count()
    stats = agent.memory.get_stats()
    domains = agent.memory.list_domains()

    console.print("─── Memory Stats ───", style="bold")
    console.print(f"  Total entries:  {total}")
    console.print(f"  STM:            {stats.get('stm', 0)}")
    console.print(f"  LTM:            {stats.get('ltm', 0)}")
    if domains:
        console.print(f"  Domains:        {', '.join(domains)}")


@memory_app.command("search")
def memory_search(
    query: str = typer.Argument(..., help="Search query"),
    config_path: Optional[str] = typer.Option(None, "--config-path"),
) -> None:
    """Search what the agent has learned."""
    path = Path(config_path) if config_path else None
    agent = _get_agent(path)

    if agent._embed_client is None:
        console.print("No embedding model configured. Run `autodidact init`.", style="yellow")
        raise typer.Exit(1)

    q_emb = agent._embed_client.embed(query)
    results = agent.memory.search(q_emb, limit=10, min_similarity=0.3)

    if not results:
        console.print("No matching knowledge found.", style="dim")
        return

    console.print(f"Found {len(results)} result(s):\n", style="bold")
    for i, hit in enumerate(results, 1):
        entry = hit.entry
        q = entry.question or "—"
        a = (entry.content or "")[:200]
        console.print(f"  {i}. [{hit.score:.2f}] Q: {q}")
        console.print(f"     A: {a}", style="dim")

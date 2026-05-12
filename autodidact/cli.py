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
from autodidact.setup_wizard import (
    build_config,
    detect_ollama,
    get_cloud_preset,
    get_ollama_install_command,
    is_model_available,
    list_cloud_providers,
    pull_ollama_model,
)
from autodidact.thought_renderer import ThoughtRenderer

console = Console()

_DEFAULT_CONFIG_PATH = Path("~/.autodidact/config.yaml").expanduser()

app = typer.Typer(
    help="Autodidact — self-learning AI agent",
    invoke_without_command=True,
    no_args_is_help=False,  # we'll handle the no-args case ourselves
)
memory_app = typer.Typer(help="Knowledge store commands")

app.add_typer(memory_app, name="memory")


@app.callback()
def _main(ctx: typer.Context) -> None:
    """Show a quickstart hint on bare `autodidact` invocations."""
    if ctx.invoked_subcommand is not None:
        return
    # User typed `autodidact` with no subcommand — welcome them.
    console.print("[bold]Autodidact[/bold] — a self-evolving AI agent that learns like a new employee.")
    console.print()
    if _DEFAULT_CONFIG_PATH.exists():
        console.print("Quick reference:")
        console.print()
        console.print("  [cyan]autodidact chat[/cyan]              Interactive chat")
        console.print("  [cyan]autodidact learn <path>[/cyan]      Ingest docs / code")
        console.print("  [cyan]autodidact savings[/cyan]           Cost savings report")
        console.print("  [cyan]autodidact memory stats[/cyan]      Knowledge store summary")
        console.print()
        console.print("  [cyan]autodidact --help[/cyan]            Full command list")
    else:
        console.print("Get started:")
        console.print()
        console.print("  [cyan]autodidact init[/cyan]              Zero-friction setup wizard")
        console.print()
        console.print("Already set up elsewhere? Point at your config with [cyan]--config-path[/cyan].")

# ── Config loading ─────────────────────────────────────────────────


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
    """Create an Agent from a config dict.

    Handles three modes:
    - local+cloud: local.model is bare (e.g. 'qwen2.5:7b') → Ollama.
    - cloud+cloud: local.provider is set (e.g. 'openai') → cheap cloud in local slot.
    - local-only: no 'cloud' section.
    """
    import os

    local_cfg = config.get("local", {})
    cloud_cfg = config.get("cloud", {})

    # ── "Local" slot (may be Ollama or a cheap cloud in cloud+cloud mode) ─
    local_provider = local_cfg.get("provider")  # set only in cloud+cloud mode
    local_model_name = local_cfg.get("model")
    local_model: Optional[str] = None
    local_base_url: Optional[str] = None
    local_api_key_env: Optional[str] = None
    local_bedrock: Optional[dict] = None

    if local_model_name:
        if local_provider and local_provider != "ollama":
            # Cloud+cloud: cheap cloud model in the local slot.
            local_model = f"{local_provider}/{local_model_name}"
            local_base_url = local_cfg.get("base_url")
            preset = get_cloud_preset(local_provider)
            local_api_key_env = preset.get("api_key_env") or "OPENAI_API_KEY"
            # If config embeds an API key, export it into the env var the
            # LLMClient reads from.
            local_api_key = local_cfg.get("api_key")
            if local_api_key and local_api_key_env:
                os.environ.setdefault(local_api_key_env, local_api_key)
            # Bedrock uses its own auth config, not a generic API key.
            if local_provider == "bedrock":
                local_bedrock = local_cfg.get("bedrock")
        else:
            # Local+cloud or local-only: Ollama.
            local_model = f"ollama/{local_model_name}"

    # ── Cloud slot ─────────────────────────────────────────────────
    cloud_provider = cloud_cfg.get("provider", "openai")
    cloud_model_name = cloud_cfg.get("model")
    cloud_model: Optional[str] = None
    cloud_base_url: Optional[str] = None
    cloud_api_key_env: Optional[str] = None
    cloud_bedrock: Optional[dict] = None

    if cloud_model_name:
        cloud_model = f"{cloud_provider}/{cloud_model_name}"
        cloud_base_url = cloud_cfg.get("base_url")
        preset = get_cloud_preset(cloud_provider)
        cloud_api_key_env = preset.get("api_key_env") or "OPENAI_API_KEY"
        cloud_api_key = cloud_cfg.get("api_key")
        if cloud_api_key and cloud_api_key_env:
            os.environ.setdefault(cloud_api_key_env, cloud_api_key)
        if cloud_provider == "bedrock":
            cloud_bedrock = cloud_cfg.get("bedrock")

    # ── Common ─────────────────────────────────────────────────────
    embedding_model = local_cfg.get("embedding_model")
    db_path = config.get("memory", {}).get("path", "~/.autodidact/memory.db")
    routing_cfg = config.get("routing", {})
    threshold = routing_cfg.get("confidence_threshold", 0.7)
    gsa_prompt_version = routing_cfg.get("gsa_prompt_version", "v3")

    kwargs: dict = dict(
        local_model=local_model,
        cloud_model=cloud_model,
        cloud_provider=cloud_provider,
        db_path=db_path,
        confidence_threshold=threshold,
        gsa_prompt_version=gsa_prompt_version,
    )
    if embedding_model:
        kwargs["embedding_model"] = embedding_model
    if local_base_url:
        kwargs["local_base_url"] = local_base_url
    if local_api_key_env:
        kwargs["local_api_key_env"] = local_api_key_env
    if local_bedrock:
        kwargs["local_bedrock"] = local_bedrock
    if cloud_base_url:
        kwargs["cloud_base_url"] = cloud_base_url
    if cloud_api_key_env:
        kwargs["cloud_api_key_env"] = cloud_api_key_env
    if cloud_bedrock:
        kwargs["cloud_bedrock"] = cloud_bedrock

    agent = Agent(**kwargs)

    # Attach a DocumentStore so ingested docs are retrieved alongside memory (R9).
    if agent._embed_client is not None:
        from autodidact.document_store import DocumentStore

        agent.attach_document_store(DocumentStore(
            agent._conn,
            agent._embed_client,
            embedding_dim=agent._config.embedding_dim,
        ))

    return agent


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
    """Zero-friction setup wizard (R8).

    Three modes:
      1. Local + Cloud — Ollama local + cloud API for escalation (best savings)
      2. Cloud + Cloud — cheap cloud + expensive cloud (no Ollama needed)
      3. Local only — Ollama only, no cloud (free, no escalation)

    For Ollama modes: auto-detects Ollama, offers install if missing, auto-pulls
    models. For cloud modes: uses presets for OpenAI, OpenRouter, DeepSeek, Bedrock.
    """
    out_path = Path(config_path) if config_path else _DEFAULT_CONFIG_PATH

    console.print("[bold]Autodidact — Setup Wizard[/bold]")
    console.print()
    console.print("[bold]Pick a setup mode:[/bold]")
    console.print("  1. Local + Cloud   — Ollama local + cloud for escalation (best savings)")
    console.print("  2. Cloud + Cloud   — cheap cloud + expensive cloud (no GPU needed)")
    console.print("  3. Local only      — Ollama only, no cloud (free, no learning escalations)")
    mode_input = typer.prompt("Mode", default="1")
    mode_map = {"1": "local_cloud", "2": "cloud_cloud", "3": "local_only"}
    mode = mode_map.get(mode_input.strip(), "local_cloud")

    if mode in ("local_cloud", "local_only"):
        config = _init_with_ollama(mode)
    else:
        config = _init_cloud_to_cloud()

    db_path = typer.prompt("Memory DB path", default="~/.autodidact/memory.db")
    config.setdefault("memory", {})["path"] = db_path

    # Write config YAML.
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    console.print(f"\nConfig written to [green]{out_path}[/green]")

    # Smoke test.
    _run_smoke_test(config)

    console.print()
    console.print("✅ [bold green]Ready![/bold green] Here's what to do next:")
    console.print()
    console.print("  [cyan]autodidact learn <path>[/cyan]   Seed the agent with your docs or codebase")
    console.print("  [cyan]autodidact chat[/cyan]           Start an interactive chat with the agent")
    console.print()
    console.print("  Run [cyan]autodidact --help[/cyan] for the full command list.")


def _init_with_ollama(mode: str) -> dict:
    """Run the Ollama-based init flow. Returns a config dict."""
    # Detect Ollama.
    status = detect_ollama()
    if not status.installed:
        console.print(
            "\n[yellow]Ollama is not installed on your system.[/yellow]",
        )
        cmd = get_ollama_install_command()
        console.print(f"Install it with:  [cyan]{cmd}[/cyan]")
        keep = typer.prompt("Continue anyway? (y/n)", default="n")
        if keep.strip().lower() != "y":
            console.print("Aborted. Install Ollama and re-run `autodidact init`.", style="yellow")
            raise typer.Exit(0)

    # Pick local model.
    local_model = typer.prompt("Local chat model", default="qwen2.5:7b")
    embedding_model = "qllama/bge-large-en-v1.5"

    # Auto-pull missing models.
    if status.installed:
        if not is_model_available(local_model):
            console.print(f"Model [cyan]{local_model}[/cyan] not found. Pulling...", style="dim")
            pull_ollama_model(local_model)
        if not is_model_available(embedding_model):
            console.print(f"Embedding model [cyan]{embedding_model}[/cyan] not found. Pulling...", style="dim")
            pull_ollama_model(embedding_model)

    # Cloud setup (only for local_cloud mode).
    if mode == "local_cloud":
        cloud_cfg = _prompt_single_cloud_provider(slot="cloud")
        return build_config(
            mode="local_cloud",
            local_model=local_model,
            embedding_model=embedding_model,
            cloud_provider=cloud_cfg["provider"],
            cloud_model=cloud_cfg["model"],
            cloud_api_key=cloud_cfg["api_key"],
            cloud_base_url=cloud_cfg.get("base_url"),
            cloud_bedrock=cloud_cfg.get("bedrock"),
        )

    return build_config(
        mode="local_only",
        local_model=local_model,
        embedding_model=embedding_model,
    )


def _init_cloud_to_cloud() -> dict:
    """Run the cloud+cloud init flow. Returns a config dict."""
    console.print("\n[bold]Cheap cloud model[/bold] (used for most queries):")
    cheap = _prompt_single_cloud_provider(slot="cheap")

    console.print("\n[bold]Expensive cloud model[/bold] (escalation target):")
    expensive = _prompt_single_cloud_provider(slot="expensive")

    return build_config(
        mode="cloud_cloud",
        cheap_cloud_provider=cheap["provider"],
        cheap_cloud_model=cheap["model"],
        cheap_cloud_api_key=cheap["api_key"],
        cheap_cloud_base_url=cheap.get("base_url"),
        cheap_cloud_bedrock=cheap.get("bedrock"),
        expensive_cloud_provider=expensive["provider"],
        expensive_cloud_model=expensive["model"],
        expensive_cloud_api_key=expensive["api_key"],
        expensive_cloud_base_url=expensive.get("base_url"),
        expensive_cloud_bedrock=expensive.get("bedrock"),
    )


def _prompt_single_cloud_provider(*, slot: str) -> dict:
    """Interactively configure one cloud provider.

    slot: 'cloud' / 'cheap' / 'expensive' — used only for prompt labeling
    and to decide which default model (cheap vs expensive) to pick.

    Bedrock is handled separately — it uses AWS credentials, not a
    generic API key, and supports multiple auth modes.
    """
    providers = list_cloud_providers()
    console.print("  Providers: " + ", ".join(providers))

    # Validate provider — offer closest match for typos.
    while True:
        provider = typer.prompt("  Provider", default="openai").strip().lower()
        if provider in providers:
            break
        import difflib
        suggestions = difflib.get_close_matches(provider, providers, n=3, cutoff=0.4)
        if suggestions:
            suggestion_str = ", ".join(f"[cyan]{s}[/cyan]" for s in suggestions)
            console.print(
                f"  [yellow]Unknown provider '[cyan]{provider}[/cyan]'.[/yellow] "
                f"Did you mean: {suggestion_str}?"
            )
        else:
            console.print(
                f"  [yellow]Unknown provider '[cyan]{provider}[/cyan]'.[/yellow] "
                f"Pick one of: {', '.join(providers)}."
            )
        if typer.confirm(f"  Use '{provider}' anyway as a custom provider?", default=False):
            break
        # Otherwise: loop and re-prompt.

    preset = get_cloud_preset(provider)

    if provider == "bedrock":
        return _prompt_bedrock_config(preset, slot)

    return _prompt_openai_compat_config(provider, preset, slot)


def _prompt_model_name(preset: dict, *, slot: str) -> str:
    """Prompt for a model name, warning if it's not in the preset list.

    Allows users to enter custom models (fine-tunes, newer models the preset
    doesn't know about) while catching typos in known model names.
    """
    default_model = preset.get("default_cheap", "")
    if slot in ("cloud", "expensive"):
        default_model = preset.get("default_expensive", "") or default_model
    models = preset.get("models", [])
    if models:
        console.print("  Available models: " + ", ".join(models))
    model = typer.prompt("  Model", default=default_model).strip()

    if models and model not in models:
        import difflib
        suggestions = difflib.get_close_matches(model, models, n=3, cutoff=0.5)
        if suggestions:
            suggestion_str = ", ".join(f"[cyan]{s}[/cyan]" for s in suggestions)
            console.print(
                f"  [yellow]'{model}' is not in the known model list. "
                f"Did you mean: {suggestion_str}?[/yellow]"
            )
            if not typer.confirm(f"  Use '{model}' anyway?", default=False):
                # Re-prompt recursively so the user can pick again.
                return _prompt_model_name(preset, slot=slot)
        else:
            console.print(
                f"  [yellow]'{model}' is not in the known model list — "
                f"using as a custom model name.[/yellow]"
            )

    return model


def _prompt_openai_compat_config(provider: str, preset: dict, slot: str) -> dict:
    """Prompt for an OpenAI-compatible provider: API key + model."""
    api_key = typer.prompt("  API key")
    model = _prompt_model_name(preset, slot=slot)

    return {
        "provider": provider,
        "model": model,
        "api_key": api_key,
        "base_url": preset.get("base_url") or None,
    }


def _prompt_bedrock_config(preset: dict, slot: str) -> dict:
    """Prompt for Bedrock: auth mode + region + model. No generic API key."""
    console.print("  Bedrock auth mode:")
    console.print("    1. IAM Role / default credential chain  (env vars, ~/.aws/credentials, SSO, IMDS)")
    console.print("    2. IAM User  (paste aws_access_key_id and aws_secret_access_key)")
    console.print("    3. Bedrock API key  (short-lived bearer token from AWS Console)")
    mode_input = typer.prompt("  Mode", default="1").strip()
    mode_map = {"1": "default", "2": "iam_user", "3": "api_key"}
    auth_mode = mode_map.get(mode_input, "default")

    bedrock_cfg: dict = {"auth_mode": auth_mode}

    if auth_mode == "iam_user":
        bedrock_cfg["access_key_id"] = typer.prompt("  aws_access_key_id")
        bedrock_cfg["secret_access_key"] = typer.prompt("  aws_secret_access_key", hide_input=True)
        session_token = typer.prompt(
            "  aws_session_token (optional, leave blank if not using temporary credentials)",
            default="",
            show_default=False,
        )
        if session_token.strip():
            bedrock_cfg["session_token"] = session_token.strip()
    elif auth_mode == "api_key":
        bedrock_cfg["api_key"] = typer.prompt("  Bedrock API key", hide_input=True)
    # default mode: nothing to collect — boto3 picks up credentials from env/config.

    region = typer.prompt("  AWS region", default="us-west-2")
    bedrock_cfg["region"] = region

    model = _prompt_model_name(preset, slot=slot)

    return {
        "provider": "bedrock",
        "model": model,
        "api_key": None,  # not applicable to bedrock
        "base_url": None,
        "bedrock": bedrock_cfg,
    }


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
                with console.status("[dim]Re-verifying with cloud...", spinner="dots"):
                    resp = agent.correct(last_q)
                renderer.render_response(resp)
            else:
                console.print("No previous question to correct.", style="yellow")
            continue

        resp = _query_with_spinner(agent, line.strip())
        renderer.render_response(resp)

    # Session summary on exit.
    report = agent.savings()
    renderer.render_session_summary(report)


def _query_with_spinner(agent: Agent, question: str) -> QueryResponse:
    """Run agent.query() while showing a 'thinking' spinner that updates per progress event.

    The spinner text changes as the agent moves through its stages so the user
    knows whether the latency is from memory search, local generation, or a
    cloud round-trip.
    """
    with console.status("[dim]Thinking...", spinner="dots") as status:
        def on_progress(event: dict) -> None:
            et = event.get("type")
            if et == "thinking":
                hits = event.get("memory_hits", 0)
                if hits:
                    status.update(f"[dim]Checking memory... found {hits} similar entries")
                else:
                    status.update("[dim]Checking memory...")
            elif et == "memory_hit":
                status.update("[dim]Recalling from memory...")
            elif et == "local_done":
                conf = event.get("confidence", 0.0)
                status.update(f"[dim]Local answer (confidence {conf:.2f})...")
            elif et == "cloud_call":
                model = event.get("model", "cloud")
                status.update(f"[dim]Asking {model}...")
            elif et == "cloud_done":
                status.update("[dim]Got cloud answer, learning from it...")
            elif et == "learning":
                status.update("[dim]Storing new knowledge...")

        return agent.query(question, on_progress=on_progress)


@app.command()
def query(
    question: str = typer.Argument(..., help="Question to ask"),
    config_path: Optional[str] = typer.Option(None, "--config-path"),
) -> None:
    """Single query mode."""
    path = Path(config_path) if config_path else None
    agent = _get_agent(path)
    renderer = ThoughtRenderer()

    resp = _query_with_spinner(agent, question)
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


# ── autodidact learn ───────────────────────────────────────────────


@app.command()
def learn(
    path: Optional[str] = typer.Argument(
        None, help="File or directory to ingest"
    ),
    stats: bool = typer.Option(
        False, "--stats", help="Show ingestion stats instead of ingesting"
    ),
    config_path: Optional[str] = typer.Option(None, "--config-path"),
) -> None:
    """Ingest documents to solve cold start (R9).

    Points the agent at existing files so it has knowledge from day one,
    before any cloud escalations.

        autodidact learn ~/docs/policies/     # ingest a folder
        autodidact learn ./README.md          # ingest a file
        autodidact learn --stats              # show totals
    """
    cfg_path = Path(config_path) if config_path else None
    agent = _get_agent(cfg_path)

    if agent.documents is None:
        console.print(
            "No document store available. Check your config — an embedding "
            "client is required for `autodidact learn`.",
            style="red",
        )
        raise typer.Exit(1)

    if stats:
        s = agent.documents.get_stats()
        console.print("─── Document Store Stats ───", style="bold")
        console.print(f"  Total files:   {s.get('total_files', 0)}")
        console.print(f"  Total chunks:  {s.get('total_chunks', 0)}")
        sources = s.get("sources", {})
        if sources:
            console.print("  Top sources:")
            for src, n in list(sources.items())[:5]:
                short = Path(src).name
                console.print(f"    {short:40} {n} chunks")
        return

    if path is None:
        console.print(
            "Provide a file or directory to ingest (or use --stats).",
            style="yellow",
        )
        raise typer.Exit(1)

    target = Path(path).expanduser()
    if not target.exists():
        console.print(f"Path does not exist: {target}", style="red")
        raise typer.Exit(1)

    console.print(f"Ingesting {target}...", style="dim")

    def _progress(evt: dict) -> None:
        if evt.get("type") == "file_ingested":
            f = Path(evt.get("file", "")).name
            chunks = evt.get("chunks", 0)
            total = evt.get("total_files", 0)
            console.print(f"  [{total}] {f} → {chunks} chunks", style="dim")

    result = agent.documents.ingest(target, on_progress=_progress)

    console.print("─── Ingestion Complete ───", style="bold green")
    console.print(f"  Files ingested:  {result.files_ingested}")
    console.print(f"  Chunks created:  {result.chunks_created}")
    if result.files_skipped > 0:
        console.print(f"  Files skipped:   {result.files_skipped}", style="yellow")

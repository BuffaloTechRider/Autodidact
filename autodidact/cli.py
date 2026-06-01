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
from typing import Callable, Optional
import subprocess
import time

import typer
import yaml
from rich.console import Console

from autodidact.agent import Agent, QueryResponse, SavingsReport
from autodidact.hardware import detect_hardware, recommended_local_model
from autodidact.setup_wizard import (
    BedrockDiscoveryError,
    OpenRouterDiscoveryError,
    OpenRouterModel,
    build_config,
    detect_ollama,
    discover_bedrock_models,
    discover_openrouter_models,
    get_cloud_preset,
    get_ollama_install_command,
    install_ollama,
    is_model_available,
    is_ollama_running,
    list_cloud_providers,
    pull_ollama_model,
    start_ollama_daemon,
    verify_model_loadable,
)
# Wizard prompt / flow / smoke moved to setup_wizard sub-modules. Re-imported
# here so existing test patches against `autodidact.cli.<helper>` continue
# to mutate cli.py's bindings (which is what cli.py's `init` command
# references via local lookup).
from autodidact.setup_wizard.flow import (
    _ensure_ollama_running,
    _init_cloud_to_cloud,
    _init_custom_server,
    _init_with_ollama,
    _offer_to_install_ollama,
    _offer_to_start_ollama,
    _pull_and_verify,
    _restart_ollama,
)
from autodidact.setup_wizard.picker import (
    _OTHER_CHOICE,
    _pick_from_list,
    _questionary_available,
)
from autodidact.setup_wizard.prompts import (
    _BROWSE_OPENROUTER_CHOICE,
    _LOCAL_MODEL_CHOICES,
    _PROVIDER_LABELS,
    _browse_openrouter_models,
    _pick_cloud_model,
    _pick_cloud_provider,
    _pick_local_model,
    _pick_openrouter_model,
    _pick_setup_mode,
    _prompt_bedrock_config,
    _prompt_model_name,
    _prompt_openai_compat_config,
    _prompt_single_cloud_provider,
)
from autodidact.setup_wizard.smoke import (
    _render_smoke_test_error,
    _run_smoke_test,
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
    """Load config YAML, with env var overrides.

    Kept as a thin shim over ``AgentConfig.from_yaml`` for tests that work
    with raw config dicts. Production code should use
    ``AgentConfig.from_yaml(path).build_agent()`` directly.
    """
    if not path.exists():
        return {}
    with open(path) as f:
        config = yaml.safe_load(f) or {}

    # Env var overrides (R8 AC3). Mirrors AgentConfig._resolve_api_keys
    # for the dict-shaped path.
    import os

    if os.environ.get("OPENAI_API_KEY"):
        config.setdefault("cloud", {})["api_key"] = os.environ["OPENAI_API_KEY"]
    if os.environ.get("AUTODIDACT_MODEL"):
        config.setdefault("local", {})["model"] = os.environ["AUTODIDACT_MODEL"]

    return config


def _agent_from_config(config: dict) -> Agent:
    """Build an Agent from a YAML-shaped dict.

    Pure shim over ``AgentConfig`` for tests and back-compat. New code
    should call ``AgentConfig.from_yaml(path).build_agent()``.

    Empty dict ⇒ no-op Agent (no models configured). Used by the CLI
    fallback when no config file exists; downstream logic prints a helpful
    "run autodidact init" message.
    """
    from autodidact.config import AgentConfig

    if not config or not config.get("local"):
        # Old behaviour: return a no-op Agent so the CLI can render its
        # own "no model configured" guidance instead of a Pydantic stack.
        return Agent()

    # Empty cloud section ⇒ local-only.
    if isinstance(config.get("cloud"), dict) and not config["cloud"]:
        config = {**config}
        del config["cloud"]

    cfg = AgentConfig.model_validate(config)
    cfg._resolve_api_keys()
    cfg._validate_api_keys_present()
    agent = cfg.build_agent()

    # Attach a DocumentStore so ingested docs are retrieved alongside
    # memory (R9). Also wire in KnowledgeStore + LLM client for document
    # synthesis.
    if agent._embed_client is not None:
        from autodidact.document_store import DocumentStore

        extractor_client = agent._local_client or agent._cloud_client
        agent.attach_document_store(DocumentStore(
            agent._conn,
            agent._embed_client,
            embedding_dim=agent._config.embedding_dim,
            knowledge_store=agent.memory,
            extractor_client=extractor_client,
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
    mode = _pick_setup_mode()

    if mode in ("local_cloud", "local_only", "local_local"):
        config = _init_with_ollama(mode)
    elif mode == "custom_server":
        config = _init_custom_server()
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


@app.command()
def chat(
    config_path: Optional[str] = typer.Option(None, "--config-path"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show debug info: memory similarity, GSA scores, routing signals"),
) -> None:
    """Interactive chat with visible thought process."""
    path = Path(config_path) if config_path else None
    agent = _get_agent(path)
    renderer = ThoughtRenderer(verbose=verbose)

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

        # Slash commands: /wrong, /gsa v4, etc. Return True if handled.
        if _dispatch_slash(agent, line.strip(), renderer):
            continue

        resp = _query_with_spinner(agent, line.strip())
        renderer.render_response(resp)

    # Session summary on exit (current session only, not all-time DB stats).
    report = getattr(agent, "_session_stats", None)
    if not isinstance(report, SavingsReport):
        report = agent.savings()
    else:
        all_cloud = report.estimated_all_cloud_cost_usd or 0.0
        report.saved_usd = all_cloud - report.total_cost_usd
        report.saved_pct = (report.saved_usd / all_cloud * 100) if all_cloud > 0 else 0.0
    renderer.render_session_summary(report)


def _dispatch_slash(agent: Agent, line: str, renderer) -> bool:
    """Route a user input line to a slash-command handler. Returns True iff handled.

    Known commands:
      /wrong, /correct, "that's wrong"  — re-escalate the last question to cloud
      /cloud [text]                     — same as /wrong (no arg) or force a new
                                          question to cloud (/cloud <text>)
      /gsa [v2|v3|v4|help]              — show or switch the GSA prompt version
      /learn <path>                     — ingest a file/folder into the document store
                                          (/learn . for the current directory)
    """
    lower = line.lower().strip()

    if lower in ("/wrong", "/correct", "that's wrong"):
        _handle_wrong_command(agent, renderer)
        return True

    if lower == "/cloud" or lower.startswith("/cloud "):
        _handle_cloud_command(agent, line, renderer)
        return True

    if lower == "/gsa" or lower.startswith("/gsa "):
        _handle_gsa_command(agent, line)
        return True

    if lower == "/learn" or lower.startswith("/learn "):
        _handle_learn_command(agent, line, renderer)
        return True

    return False


def _handle_learn_command(agent: Agent, line: str, renderer) -> None:
    """Ingest a file or directory into the agent's document store.

    Usage:
      /learn <path>   — ingest the given file or directory
      /learn .        — shortcut for the current working directory
      /learn          — print usage hint, do nothing

    Mirrors the ``autodidact learn`` CLI command but works mid-chat so users
    can drop docs into context without leaving the REPL.
    """
    parts = line.split(maxsplit=1)
    arg = parts[1].strip() if len(parts) > 1 else ""

    if not arg:
        console.print(
            "Usage: [cyan]/learn <path>[/cyan]   (or [cyan]/learn .[/cyan] for the current directory)",
            style="yellow",
        )
        return

    if agent.documents is None:
        console.print(
            "[red]No document store available.[/red] "
            "Check your config — an embedding client is required for [cyan]/learn[/cyan].",
        )
        return

    target = Path(arg).expanduser()
    # `.` resolves against the current working directory.
    if not target.is_absolute():
        target = (Path.cwd() / target).resolve() if str(target) == "." else target
    if not target.exists():
        console.print(f"[red]Path does not exist:[/red] {target}")
        return

    console.print(f"Ingesting [cyan]{target}[/cyan]...", style="dim")

    def _progress(evt: dict) -> None:
        if evt.get("type") == "file_ingested":
            f = Path(evt.get("file", "")).name
            chunks = evt.get("chunks", 0)
            total = evt.get("total_files", 0)
            console.print(f"  [{total}] {f} → {chunks} chunks", style="dim")

    try:
        result = agent.documents.ingest(target, on_progress=_progress)
    except Exception as e:
        console.print(f"[red]Ingest failed:[/red] {e}")
        return

    console.print("─── Ingestion Complete ───", style="bold green")
    console.print(f"  Files ingested:  {result.files_ingested}")
    console.print(f"  Chunks created:  {result.chunks_created}")


def _handle_cloud_command(agent: Agent, line: str, renderer) -> None:
    """Force cloud escalation, either re-routing the last question or a new one.

    Usage:
      /cloud          — alias of /wrong: re-route the last user turn to cloud
      /cloud <text>   — send <text> directly to cloud, skipping memory/GSA/local

    Both forms call Agent.correct() under the hood. If the last question was
    answered locally, there is no stored memory entry to invalidate (that's
    a no-op). If it was answered by cloud, invalidating the entry is the
    right thing to do — we're asking cloud to re-answer.
    """
    parts = line.split(maxsplit=1)
    arg = parts[1].strip() if len(parts) > 1 else ""

    if not arg:
        # /cloud alone — re-route the last user turn.
        if not agent._history:
            console.print("No previous question to re-route to cloud.", style="yellow")
            return
        last_q = ""
        for turn in reversed(agent._history):
            if turn["role"] == "user":
                last_q = turn["content"]
                break
        if not last_q:
            console.print("No previous question to re-route to cloud.", style="yellow")
            return
        question = last_q
    else:
        question = arg

    resp = _correct_with_spinner(agent, question)
    if renderer is not None:
        renderer.render_response(resp)


def _handle_wrong_command(agent: Agent, renderer) -> None:
    """Re-escalate the last question to cloud and replace the stored answer."""
    if not agent._history:
        console.print("No previous question to correct.", style="yellow")
        return
    last_q = agent._history[-2]["content"] if len(agent._history) >= 2 else ""
    if not last_q:
        console.print("No previous question to correct.", style="yellow")
        return
    resp = _correct_with_spinner(agent, last_q)
    if renderer is not None:
        renderer.render_response(resp)


def _handle_gsa_command(agent: Agent, line: str) -> None:
    """Show or change the GSA prompt version for the rest of this session.

    Usage:
      /gsa            — print current version
      /gsa help       — list available versions
      /gsa v4         — switch to v4 (opt-in adversarial-trust prompt)
      /gsa v3         — switch back to the default
      /gsa v2         — legacy bare prompt, no retrieval

    This is session-only. Persisting the choice requires editing ~/.autodidact/config.yaml.
    """
    from autodidact.signals.grounded_self_assessment import SelfAssessment

    parts = line.split(maxsplit=1)
    arg = parts[1].strip().lower() if len(parts) > 1 else ""

    valid = ("v2", "v3", "v4")

    if arg == "" or arg in ("status", "show"):
        current = _gsa_current_version(agent)
        console.print(f"GSA prompt version: [cyan]{current}[/cyan]")
        return

    if arg in ("help", "-h", "--help", "?"):
        console.print("Usage: [cyan]/gsa [v2|v3|v4|help][/cyan]")
        console.print("  v2 — legacy bare prompt (no retrieval)")
        console.print("  v3 — default: retrieval-conditional, specific-knowledge framing")
        console.print("  v4 — opt-in: adversarial trust framing")
        return

    if arg not in valid:
        console.print(
            f"[yellow]Unknown version '{arg}'. Valid: {', '.join(valid)}. "
            f"Try [cyan]/gsa help[/cyan].[/yellow]"
        )
        return

    # Rebuild the probe with the new version. Next query picks it up.
    agent._gsa = SelfAssessment(agent._local_client, prompt_version=arg)
    console.print(f"GSA prompt version set to [cyan]{agent._gsa.prompt_version}[/cyan].")


def _gsa_current_version(agent: Agent) -> str:
    """Return a human-readable current GSA prompt version."""
    probe = getattr(agent, "_gsa", None)
    if probe is None:
        return "v3 (default, no probe built yet)"
    return probe.prompt_version


def _query_with_spinner(agent: Agent, question: str) -> QueryResponse:
    """Run agent.query() with live streaming output. Wrapper over _run_with_spinner."""
    return _run_with_spinner(lambda cb: agent.query(question, on_progress=cb))


def _correct_with_spinner(agent: Agent, question: str) -> QueryResponse:
    """Run agent.correct() with live streaming output. Wrapper over _run_with_spinner."""
    return _run_with_spinner(lambda cb: agent.correct(question, on_progress=cb))


def _run_with_spinner(call: Callable[[Callable[[dict], None]], QueryResponse]) -> QueryResponse:
    """Run an agent operation that takes an on_progress callback, rendering live progress.

    Two phases the user sees:
      Spinner phase  — memory check, GSA probe, possibly thinking-token reasoning
      Streaming phase — content tokens arrive live; we drop the spinner and
                        print tokens directly so the user reads as it generates.

    Tokens carry source='local' or 'cloud' so we tag them appropriately.
    """
    state = {
        "phase": None,             # "thinking" | "content" | None
        "source": None,            # "local" | "cloud" | None
        "thinking_buf": [],
        "content_buf": [],
        "rendering_live": False,    # True once we've left the spinner
    }

    def _start_streaming(source: str) -> None:
        """Stop the spinner and print the route prefix for streaming output."""
        if source == "cloud":
            tag = "[bold blue][CLOUD][/bold blue] "
        else:
            tag = "[bold green][LOCAL][/bold green] "
        # The status object is closed-over from the outer scope; tracked
        # via state to keep the closure simple.
        state["status"].stop()
        console.print()  # whitespace under the spinner row
        console.print(tag, end="")
        state["rendering_live"] = True
        state["phase"] = "content"
        state["source"] = source

    with console.status("[dim]Thinking...", spinner="dots") as status:
        state["status"] = status

        def on_progress(event) -> None:
            from autodidact.events import (
                CloudCallEvent,
                CloudDoneEvent,
                GsaCheckEvent,
                LocalDoneEvent,
                MemoryHitEvent,
                ThinkingEvent,
                TokenEvent,
            )

            if isinstance(event, ThinkingEvent):
                if event.memory_hits:
                    status.update(
                        f"[dim]Checking memory... found {event.memory_hits} similar entries"
                    )
                else:
                    status.update("[dim]Checking memory...")

            elif isinstance(event, GsaCheckEvent):
                status.update("[dim]Confirming with local brain...")

            elif isinstance(event, MemoryHitEvent):
                status.update("[dim]Recalling from memory...")

            elif isinstance(event, TokenEvent):
                if not event.text:
                    return

                if event.phase == "thinking":
                    if state["phase"] != "thinking":
                        if event.source == "local":
                            status.update(
                                "[dim]Local brain working...\n"
                                "  If I fumble this one, type /cloud to ask my sensei"
                            )
                        else:
                            status.update("[dim]Thinking...")
                        state["phase"] = "thinking"
                    state["thinking_buf"].append(event.text)

                elif event.phase == "content":
                    # First content token from this source — drop spinner and
                    # start printing tokens directly.
                    if not state["rendering_live"] or state["source"] != event.source:
                        if state["rendering_live"]:
                            # Source switched (rare: local, then cloud during one query).
                            console.print()
                            state["rendering_live"] = False
                            status.start()
                        _start_streaming(event.source)
                    console.print(event.text, end="", soft_wrap=True, highlight=False)
                    state["content_buf"].append(event.text)

            elif isinstance(event, LocalDoneEvent):
                # Non-streaming path (e.g. test mock or non-Ollama local).
                if not state["rendering_live"]:
                    status.update(f"[dim]Local answer (confidence {event.confidence:.2f})...")

            elif isinstance(event, CloudCallEvent):
                # If we already streamed local content, finish that line.
                if state["rendering_live"]:
                    console.print()
                    state["rendering_live"] = False
                    state["source"] = None
                    status.start()
                status.update(f"[dim]Asking {event.model}...")

            elif isinstance(event, CloudDoneEvent):
                # Token-level streaming has already shown the answer; this
                # event still fires after the stream ends. If we never
                # streamed cloud (test mock), update the spinner.
                if not state["rendering_live"]:
                    status.update("[dim]Got cloud answer, learning from it...")

        resp = call(on_progress)

    # If we streamed any content live, the body is already on screen. Mark
    # the response so the renderer prints only the footer (cost/route),
    # not a duplicate of the body.
    already_streamed = bool(state["content_buf"])
    if already_streamed:
        console.print()  # newline so the footer lands on its own row
    setattr(resp, "_already_streamed", already_streamed)
    return resp


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
        elif evt.get("type") == "synthesized":
            f = Path(evt.get("file", "")).name
            facts = evt.get("facts", 0)
            console.print(f"  ✦ {f} → {facts} facts learned", style="cyan")

    result = agent.documents.ingest(target, on_progress=_progress)

    console.print("─── Ingestion Complete ───", style="bold green")
    console.print(f"  Files ingested:  {result.files_ingested}")
    console.print(f"  Chunks created:  {result.chunks_created}")
    if agent._embed_client and agent._local_client and result.files_ingested > 0:
        console.print("  Synthesizing knowledge in background...", style="cyan")
    if result.files_skipped > 0:
        console.print(f"  Files skipped:   {result.files_skipped}", style="yellow")

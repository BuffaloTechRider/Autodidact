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
    mode = _pick_setup_mode()

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


def _offer_to_install_ollama() -> bool:
    """Show the install command and ask the user to confirm. Returns True if installed.

    On Windows we cannot auto-install in v1.0; print manual instructions and
    return False so the wizard exits cleanly.
    """
    import sys

    console.print()
    console.print("[yellow]Ollama is not installed on your system.[/yellow]")

    if sys.platform == "win32":
        console.print(
            "  v1.0 doesn't auto-install Ollama on Windows. "
            "Download the installer from [cyan]https://ollama.com/download/windows[/cyan], "
            "run it, then re-run [cyan]autodidact init[/cyan]."
        )
        return False

    cmd = get_ollama_install_command()
    console.print(f"  Install command: [cyan]{cmd}[/cyan]")
    console.print(
        "  This downloads and runs the official installer from [cyan]ollama.com[/cyan].",
        style="dim",
    )

    if not typer.confirm("Install Ollama now?", default=True):
        return False

    console.print("Installing Ollama...", style="dim")
    if install_ollama():
        console.print("✓ Ollama installed.", style="green")
        return True

    console.print(
        "[red]Install failed.[/red] You can run the command manually:\n"
        f"  [cyan]{cmd}[/cyan]"
    )
    return False


def _offer_to_start_ollama() -> bool:
    """Detect that Ollama isn't running and ask to start it. Returns True iff up."""
    console.print()
    console.print("[yellow]Ollama is installed but the daemon isn't running.[/yellow]")
    if not typer.confirm("Start the Ollama daemon now?", default=True):
        return False

    console.print("Starting Ollama daemon...", style="dim")
    if start_ollama_daemon(wait_timeout_s=20.0):
        console.print("✓ Ollama daemon is running.", style="green")
        return True

    import sys
    if sys.platform == "darwin":
        console.print(
            "[red]Could not start the daemon automatically.[/red] "
            "macOS may have shown a Gatekeeper prompt for the Ollama app, "
            "or asked to approve a background login item.\n"
            "  • Approve any prompts in System Settings → Privacy & Security "
            "and General → Login Items, then\n"
            "  • Open the Ollama app from Applications, or run "
            "[cyan]ollama serve[/cyan] in another terminal.\n"
            "Then re-run [cyan]autodidact init[/cyan]."
        )
    else:
        console.print(
            "[red]Could not start the daemon automatically.[/red] "
            "Try running [cyan]ollama serve[/cyan] in another terminal, "
            "then re-run [cyan]autodidact init[/cyan]."
        )
    return False


def _init_with_ollama(mode: str) -> dict:

    # Detect Ollama.
    status = detect_ollama()
    if not status.installed:
        if not _offer_to_install_ollama():
            console.print(
                "Aborted. Install Ollama and re-run [cyan]autodidact init[/cyan].",
                style="yellow",
            )
            raise typer.Exit(0)
        # Re-detect after install.
        status = detect_ollama()
        if not status.installed:
            console.print(
                "[red]Install ran but Ollama still isn't on PATH.[/red] "
                "You may need to restart your shell, then re-run "
                "[cyan]autodidact init[/cyan].",
            )
            raise typer.Exit(1)

    # Daemon needs to be running for pulls and embedding/chat calls.
    if not is_ollama_running():
        if not _offer_to_start_ollama():
            console.print(
                "Aborted. Start the Ollama daemon and re-run [cyan]autodidact init[/cyan].",
                style="yellow",
            )
            raise typer.Exit(0)

    # Hardware-aware default.
    profile = detect_hardware()
    recommended = recommended_local_model(profile)
    if profile.tier != "unknown":
        ram_str = f"{profile.ram_gb:.0f}GB RAM"
        apple_str = " Apple Silicon," if profile.is_apple_silicon else ""
        gpu_str = f" {profile.vram_gb:.0f}GB NVIDIA VRAM," if profile.vram_gb else ""
        console.print(
            f"[dim]Detected:{apple_str}{gpu_str} {ram_str} → tier [bold]{profile.tier}[/bold][/dim]"
        )

    # Pick local model from curated list.
    local_model = _pick_local_model(recommended=recommended)
    embedding_model = "qllama/bge-large-en-v1.5"

    # Auto-pull missing models, then verify.
    if status.installed:
        _pull_and_verify(local_model, label="Chat model")
        _pull_and_verify(embedding_model, label="Embedding model")

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


def _pull_and_verify(model_name: str, *, label: str) -> None:
    """Pull a model if needed, then verify Ollama can actually serve it.

    Catches the 'cloud-only tag' case where pull returns success but the
    model never appears in ollama list. If verify fails, prints a helpful
    error and raises typer.Exit so the wizard stops before writing a
    broken config.
    """
    if is_model_available(model_name):
        return

    console.print(f"{label} [cyan]{model_name}[/cyan] not pulled yet. Downloading...", style="dim")
    pull_ollama_model(model_name)

    if not verify_model_loadable(model_name):
        console.print(
            f"\n[red]{label} [cyan]{model_name}[/cyan] pulled but cannot be loaded locally.[/red]"
        )
        console.print(
            "  Likely cause: this tag points to cloud-only inference "
            "(e.g. qwen3.5:9b, *:cloud). Pick a tag with real weights.",
            style="dim",
        )
        console.print(
            f"  Try: [cyan]ollama run {model_name}[/cyan] to confirm, "
            f"then re-run [cyan]autodidact init[/cyan] with a different model.",
            style="dim",
        )
        raise typer.Exit(1)


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


def _questionary_available() -> bool:
    """True if questionary can be imported AND stdin is a TTY.

    Falls back to typer.prompt in non-interactive shells (CI, piped input)
    so the wizard works in both modes.
    """
    try:
        import questionary  # noqa: F401
    except ImportError:
        return False
    import sys
    return sys.stdin.isatty()


def _pick_from_list(title: str, choices: list[str], default: str) -> str:
    """Show a picker for ``choices`` with ``default`` pre-selected.

    Uses questionary.select (arrow-key navigation) when available,
    otherwise prints a numbered list and reads a number from typer.prompt.
    Pressing Enter alone accepts the default; invalid input also returns
    the default rather than looping.
    """
    if _questionary_available():
        import questionary
        # questionary raises on interrupt; keep behavior consistent with typer.
        answer = questionary.select(title, choices=choices, default=default).ask()
        if answer is None:
            # User pressed Ctrl+C; re-raise as KeyboardInterrupt so typer handles it.
            raise KeyboardInterrupt
        return answer

    # Fallback: numbered list.
    console.print(f"[bold]{title}[/bold]")
    for i, choice in enumerate(choices, start=1):
        marker = " (default)" if choice == default else ""
        console.print(f"  {i}. {choice}{marker}")
    default_idx = str(choices.index(default) + 1) if default in choices else "1"
    raw = typer.prompt("Choice", default=default_idx).strip()
    if not raw:
        return default
    # Accept a 1-based index…
    try:
        idx = int(raw) - 1
        if 0 <= idx < len(choices):
            return choices[idx]
    except ValueError:
        pass
    # …or a direct label / substring match against the choice list.
    # This keeps existing integration tests working (they feed provider
    # names like 'openrouter' rather than numbers).
    raw_lower = raw.lower()
    for choice in choices:
        if choice.lower() == raw_lower:
            return choice
    for choice in choices:
        if raw_lower in choice.lower():
            return choice
    return default


# Curated list of Ollama models. Qwen 3 first (current generation, best
# benchmarks), Qwen 2.5 kept for users who specifically want it, plus a few
# alternatives. Routing signals are designed to be model-agnostic, so we
# default to the newer generation even though our experiments ran on
# qwen2.5:7b.
_LOCAL_MODEL_CHOICES = [
    ("qwen3:32b",          "largest dense Qwen 3 (20GB, needs 32GB+ RAM)"),
    ("qwen3-coder:30b",    "code-specialized MoE (18GB, 32GB+ RAM)"),
    ("qwen3:14b",          "bigger — 9GB, needs 16GB+ RAM"),
    ("qwen3:8b",           "balanced default (5.2GB)"),
    ("qwen3:4b",           "lightweight (2.5GB, 8GB machines)"),
    ("qwen3:0.6b",         "minimal (523MB — quality is meh)"),
    ("qwen2.5:14b",        "Qwen 2.5 generation, larger (9GB)"),
    ("qwen2.5:7b",         "Qwen 2.5 generation, balanced (4.7GB)"),
    ("llama3.2:3b",        "Meta small, 2GB"),
    ("llama3.1:8b",        "Meta general, 4.9GB"),
    ("mistral:7b-instruct", "Mistral instruct, 4.4GB"),
]
_OTHER_CHOICE = "Other (type a model name)"


def _pick_local_model(*, recommended: str) -> str:
    """Show the curated local-model list with ``recommended`` highlighted."""
    labeled: list[str] = []
    default_label = ""
    for name, desc in _LOCAL_MODEL_CHOICES:
        label = f"{name} — {desc}"
        if name == recommended:
            label = f"{name} — {desc} (recommended for this machine)"
            default_label = label
        labeled.append(label)
    labeled.append(_OTHER_CHOICE)
    if not default_label:
        # recommended wasn't in the curated list; put it at the top.
        custom_rec_label = f"{recommended} (recommended for this machine)"
        labeled.insert(0, custom_rec_label)
        default_label = custom_rec_label

    chosen = _pick_from_list("Local chat model", labeled, default_label)
    if chosen == _OTHER_CHOICE:
        return typer.prompt("Model name").strip()
    # Label format is "name — description". Pull the name off.
    return chosen.split(" ", 1)[0].strip()


def _pick_cloud_provider() -> str:
    """Show the cloud provider list with 'other' fallback."""
    providers = list_cloud_providers()
    choices = list(providers) + [_OTHER_CHOICE]
    chosen = _pick_from_list("Cloud provider", choices, "openai")
    if chosen == _OTHER_CHOICE:
        return typer.prompt("Provider name").strip().lower()
    return chosen


def _pick_cloud_model(preset: dict, *, slot: str) -> str:
    """Show the cloud provider's model list with 'other' fallback."""
    models = preset.get("models") or []
    default_model = preset.get("default_cheap", "")
    if slot in ("cloud", "expensive"):
        default_model = preset.get("default_expensive") or default_model

    if not models:
        # No curated list — just prompt freely.
        return typer.prompt("Model", default=default_model).strip()

    choices = list(models) + [_OTHER_CHOICE]
    default = default_model if default_model in models else models[0]
    chosen = _pick_from_list("Model", choices, default)
    if chosen == _OTHER_CHOICE:
        return typer.prompt("Model name").strip()
    return chosen


def _pick_setup_mode() -> str:
    """Show the 3 setup modes as a list; return the canonical key."""
    labels = [
        "Local + Cloud   — Ollama local + cloud for escalation (best savings)",
        "Cloud + Cloud   — cheap cloud + expensive cloud (no GPU needed)",
        "Local only      — Ollama only, no cloud (free, no learning escalations)",
    ]
    chosen = _pick_from_list("Pick a setup mode", labels, labels[0])
    if "Local + Cloud" in chosen:
        return "local_cloud"
    if "Cloud + Cloud" in chosen:
        return "cloud_cloud"
    return "local_only"


def _prompt_single_cloud_provider(*, slot: str) -> dict:
    """Interactively configure one cloud provider.

    slot: 'cloud' / 'cheap' / 'expensive' — used only for prompt labeling
    and to decide which default model (cheap vs expensive) to pick.

    Bedrock is handled separately — it uses AWS credentials, not a
    generic API key, and supports multiple auth modes.
    """
    provider = _pick_cloud_provider()
    preset = get_cloud_preset(provider)

    if provider == "bedrock":
        return _prompt_bedrock_config(preset, slot)

    return _prompt_openai_compat_config(provider, preset, slot)


def _prompt_model_name(preset: dict, *, slot: str) -> str:
    """Prompt for a model name using the curated-list picker.

    Delegates to ``_pick_cloud_model`` which handles the 'Other' escape for
    custom fine-tunes and new models the preset doesn't know about.
    """
    return _pick_cloud_model(preset, slot=slot)


_BROWSE_OPENROUTER_CHOICE = "↪ Browse all OpenRouter models (live)"


def _prompt_openai_compat_config(provider: str, preset: dict, slot: str) -> dict:
    """Prompt for an OpenAI-compatible provider: API key + model.

    OpenRouter gets a special ``Browse all`` entry in the picker that hits
    the public ``/v1/models`` endpoint. The catalogue is hundreds of models
    long and changes weekly; the curated preset can't keep up, and slug
    typos lose users at chat time.
    """
    api_key = typer.prompt("  API key")

    if provider == "openrouter":
        model = _pick_openrouter_model(preset, slot=slot)
    else:
        model = _prompt_model_name(preset, slot=slot)

    return {
        "provider": provider,
        "model": model,
        "api_key": api_key,
        "base_url": preset.get("base_url") or None,
    }


def _pick_openrouter_model(preset: dict, *, slot: str) -> str:
    """Picker for OpenRouter: curated preset + 'Browse all' + 'Other'."""
    models = list(preset.get("models") or [])
    default = preset.get("default_cheap", "")
    if slot in ("cloud", "expensive"):
        default = preset.get("default_expensive") or default
    if not default and models:
        default = models[0]

    choices = list(models) + [_BROWSE_OPENROUTER_CHOICE, _OTHER_CHOICE]
    chosen = _pick_from_list("Model", choices, default if default in models else choices[0])

    if chosen == _BROWSE_OPENROUTER_CHOICE:
        return _browse_openrouter_models()
    if chosen == _OTHER_CHOICE:
        return typer.prompt("Model name").strip()
    return chosen


def _browse_openrouter_models() -> str:
    """Fetch the live OpenRouter catalogue and let the user pick.

    Falls back to a free-form prompt with the original error if discovery
    fails (network down, 5xx, parse error).
    """
    try:
        models = discover_openrouter_models()
    except OpenRouterDiscoveryError as e:
        console.print(f"  [yellow]Could not list OpenRouter models:[/yellow] {e}")
        console.print(
            "  [dim]Falling back to manual entry. "
            "Browse the catalogue at https://openrouter.ai/models[/dim]",
        )
        return typer.prompt("  Model ID").strip()

    if not models:
        console.print("  [yellow]OpenRouter returned no usable models.[/yellow]")
        return typer.prompt("  Model ID").strip()

    # Build labeled rows: "id  ($X.XX / $Y.YY per 1M)". The picker still
    # returns the raw choice string so we keep id + label correspondence
    # via a parallel map.
    rows: list[str] = []
    label_to_id: dict[str, str] = {}
    for m in models:
        label = (
            f"{m.id}  "
            f"(${m.prompt_per_million:.2f} in / ${m.completion_per_million:.2f} out per 1M)"
        )
        rows.append(label)
        label_to_id[label] = m.id

    rows.append(_OTHER_CHOICE)
    label_to_id[_OTHER_CHOICE] = ""  # sentinel — handled below

    console.print(f"  [dim]{len(models)} models, sorted cheapest first.[/dim]")
    chosen = _pick_from_list("OpenRouter model", rows, rows[0])
    if chosen == _OTHER_CHOICE:
        return typer.prompt("  Model ID").strip()
    return label_to_id[chosen]


def _prompt_bedrock_config(preset: dict, slot: str) -> dict:
    """Prompt for Bedrock: auth mode + region + model. No generic API key.

    The model list is *discovered* at this point by querying Bedrock with
    the supplied region + auth. If discovery fails (no creds, no perms,
    network), we fall back to a free-form prompt and surface the error.
    """
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

    # ── Live model discovery ─────────────────────────────────────
    discovered: list[str] = []
    discovery_error: Optional[BedrockDiscoveryError] = None
    try:
        discovered = discover_bedrock_models(
            region=region,
            auth_mode=auth_mode,
            access_key_id=bedrock_cfg.get("access_key_id"),
            secret_access_key=bedrock_cfg.get("secret_access_key"),
            session_token=bedrock_cfg.get("session_token"),
            api_key=bedrock_cfg.get("api_key"),
        )
    except BedrockDiscoveryError as e:
        discovery_error = e

    if discovered:
        # Suggest a sensible default per slot if available.
        prefer_cheap = ("haiku", "nova-micro", "nova-lite")
        prefer_expensive = ("sonnet", "opus", "nova-pro", "nova-premier")
        keywords = prefer_cheap if slot == "cheap" else prefer_expensive
        default = next(
            (m for kw in keywords for m in discovered if kw in m.lower()),
            discovered[0],
        )
        choices = list(discovered) + [_OTHER_CHOICE]
        chosen = _pick_from_list("Bedrock model", choices, default)
        if chosen == _OTHER_CHOICE:
            model = typer.prompt("  Model ID").strip()
        else:
            model = chosen
    else:
        if discovery_error is not None:
            console.print(
                f"  [yellow]Could not list Bedrock models:[/yellow] {discovery_error}",
            )
            console.print(
                "  [dim]Falling back to manual entry. "
                "Find IDs at https://docs.aws.amazon.com/bedrock/latest/userguide/models-supported.html[/dim]",
            )
        model = typer.prompt("  Model ID").strip()

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

        # Slash commands: /wrong, /gsa v4, etc. Return True if handled.
        if _dispatch_slash(agent, line.strip(), renderer):
            continue

        resp = _query_with_spinner(agent, line.strip())
        renderer.render_response(resp)

    # Session summary on exit.
    report = agent.savings()
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

            elif et == "token":
                phase = event.get("phase", "content")
                source = event.get("source", "local")
                text = event.get("text", "")
                if not text:
                    return

                if phase == "thinking":
                    if state["phase"] != "thinking":
                        status.update("[dim]Thinking...")
                        state["phase"] = "thinking"
                    state["thinking_buf"].append(text)

                elif phase == "content":
                    # First content token from this source — drop spinner and
                    # start printing tokens directly.
                    if not state["rendering_live"] or state["source"] != source:
                        if state["rendering_live"]:
                            # Source switched (rare: local, then cloud during one query).
                            # Close the previous line cleanly.
                            console.print()
                            state["rendering_live"] = False
                            status.start()
                        _start_streaming(source)
                    console.print(text, end="", soft_wrap=True, highlight=False)
                    state["content_buf"].append(text)

            elif et == "local_done":
                # Non-streaming path (e.g. test mock or non-Ollama local).
                if not state["rendering_live"]:
                    conf = event.get("confidence", 0.0)
                    status.update(f"[dim]Local answer (confidence {conf:.2f})...")

            elif et == "cloud_call":
                # If we already streamed local content, finish that line.
                if state["rendering_live"]:
                    console.print()
                    state["rendering_live"] = False
                    state["source"] = None
                    status.start()
                model = event.get("model", "cloud")
                status.update(f"[dim]Asking {model}...")

            elif et == "cloud_done":
                # Token-level streaming has already shown the answer; this
                # event still fires after the stream ends. If we never
                # streamed cloud (test mock), update the spinner.
                if not state["rendering_live"]:
                    status.update("[dim]Got cloud answer, learning from it...")

            elif et == "learning":
                if state["rendering_live"]:
                    # Don't clobber the streamed answer; just print a hint.
                    pass
                else:
                    status.update("[dim]Storing new knowledge...")

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

    result = agent.documents.ingest(target, on_progress=_progress)

    console.print("─── Ingestion Complete ───", style="bold green")
    console.print(f"  Files ingested:  {result.files_ingested}")
    console.print(f"  Chunks created:  {result.chunks_created}")
    if result.files_skipped > 0:
        console.print(f"  Files skipped:   {result.files_skipped}", style="yellow")

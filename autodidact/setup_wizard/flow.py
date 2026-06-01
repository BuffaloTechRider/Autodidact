"""Top-level wizard flows for ``autodidact init``.

The init Typer command in cli.py orchestrates by:
  1. picking a setup mode via prompts._pick_setup_mode
  2. dispatching to one of the _init_* functions here
  3. writing the YAML
  4. running the smoke test (smoke._run_smoke_test)

Each _init_* function returns a config dict that the caller writes
to disk. They handle their own user interaction (Ollama install, model
pull, cloud-provider config) and can ``raise typer.Exit(1)`` on
unrecoverable failure.
"""

from __future__ import annotations

import subprocess
import sys
import time

import typer

from autodidact.hardware import detect_hardware, recommended_local_model
from autodidact.setup_wizard._console import console
from autodidact.setup_wizard.builder import build_config
from autodidact.setup_wizard.ollama import (
    _has_homebrew,
    detect_ollama,
    get_ollama_install_command,
    install_ollama,
    is_model_available,
    is_ollama_running,
    pull_ollama_model,
    start_ollama_daemon,
    verify_model_loadable,
)
from autodidact.setup_wizard.prompts import (
    _pick_local_model,
    _prompt_single_cloud_provider,
)


# ── Ollama install / daemon orchestration ───────────────────────


def _offer_to_install_ollama() -> bool:
    """Install Ollama: auto-install → retry → manual with wait. Returns True if installed.

    Flow:
    1. Try automatic install (with retry on transient failures)
    2. If auto fails → show manual command, wait for user to confirm done
    3. Re-detect Ollama on PATH
    4. If installed and not running → start daemon automatically
    """
    console.print()
    console.print("[yellow]Ollama is not installed on your system.[/yellow]")

    if sys.platform == "win32":
        console.print(
            "  Download the installer from [cyan]https://ollama.com/download/windows[/cyan], "
            "run it, then press Enter to continue."
        )
        typer.prompt("Press Enter when Ollama is installed", default="", show_default=False)
        return detect_ollama().installed

    if sys.platform == "darwin" and _has_homebrew():
        cmd = "brew install ollama"
    else:
        cmd = get_ollama_install_command()
    console.print(f"  Install command: [cyan]{cmd}[/cyan]")

    if typer.confirm("Install Ollama automatically?", default=True):
        console.print("Installing Ollama...", style="dim")
        if install_ollama():
            console.print("✓ Ollama installed.", style="green")
            return _ensure_ollama_running()

    # Auto-install failed or user declined — manual install flow with retry loop.
    console.print()
    console.print(
        "[yellow]Please install Ollama manually:[/yellow]\n"
        f"  [cyan]{cmd}[/cyan]\n"
    )

    for _ in range(3):
        typer.prompt("Press Enter when done", default="", show_default=False)
        if detect_ollama().installed:
            console.print("✓ Ollama detected.", style="green")
            return _ensure_ollama_running()
        console.print(
            "[yellow]Ollama still not found.[/yellow] "
            "Make sure the install completed and Ollama is on your PATH.\n"
            "  (You may need to open a new terminal for PATH changes to take effect.)\n"
            f"  Install command: [cyan]{cmd}[/cyan]\n"
        )
        if not typer.confirm("Try again?", default=True):
            break

    console.print("Aborted. Re-run [cyan]autodidact init[/cyan] after installing Ollama.")
    return False


def _restart_ollama() -> None:
    """Kill and restart the Ollama daemon to pick up a new binary version."""
    # Kill all ollama processes — old daemon from /Applications AND any serve processes.
    # Use -9 to force-kill since the old daemon may resist SIGTERM.
    for cmd in [["pkill", "-9", "-f", "Ollama"], ["pkill", "-9", "-f", "ollama serve"]]:
        try:
            subprocess.run(cmd, capture_output=True, timeout=5)
        except (OSError, subprocess.TimeoutExpired):
            pass
    time.sleep(3)
    start_ollama_daemon(wait_timeout_s=20.0)


def _ensure_ollama_running() -> bool:
    """If Ollama is installed but daemon isn't running, start it. Returns True if ready."""
    if is_ollama_running():
        return True
    console.print("Starting Ollama daemon...", style="dim")
    if start_ollama_daemon(wait_timeout_s=20.0):
        console.print("✓ Ollama daemon is running.", style="green")
        return True
    console.print(
        "[yellow]Ollama installed but daemon didn't start.[/yellow]\n"
        "  Start it manually: [cyan]ollama serve[/cyan] (in another terminal)\n"
        "  Then press Enter to continue."
    )
    typer.prompt("Press Enter when Ollama is running", default="", show_default=False)
    return is_ollama_running()


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


# ── Per-mode orchestrators ──────────────────────────────────────


def _init_with_ollama(mode: str) -> dict:
    """Three Ollama-using modes: local_cloud, local_only, local_local."""
    # Detect Ollama — install if missing, start daemon if not running.
    status = detect_ollama()
    if not status.installed:
        if not _offer_to_install_ollama():
            console.print(
                "Aborted. Re-run [cyan]autodidact init[/cyan] after installing Ollama.",
                style="yellow",
            )
            raise typer.Exit(0)
    elif not is_ollama_running():
        if not _ensure_ollama_running():
            console.print(
                "Aborted. Start Ollama and re-run [cyan]autodidact init[/cyan].",
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
    # Re-check status since Ollama may have been installed above.
    if status.installed or is_ollama_running():
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

    # Local+Local: small model already chosen above; pick a bigger one for escalation.
    if mode == "local_local":
        console.print("\n[bold]Big model[/bold] (escalation target — slower but smarter):")
        big_model = _pick_local_model(recommended="qwen2.5:14b")
        _pull_and_verify(big_model, label="Big model")
        return build_config(
            mode="local_local",
            local_model=local_model,
            embedding_model=embedding_model,
            cloud_provider="ollama",
            cloud_model=big_model,
        )

    return build_config(
        mode="local_only",
        local_model=local_model,
        embedding_model=embedding_model,
    )


def _pull_and_verify(model_name: str, *, label: str) -> None:
    """Pull a model if needed, then verify Ollama can actually serve it.

    Distinguishes three failure modes:
    1. Pull failed (network/TLS error) → suggest retry or different network
    2. Pull succeeded but model is cloud-only → suggest a different tag
    3. Pull succeeded and model loads → success
    """
    if is_model_available(model_name):
        return

    console.print(f"{label} [cyan]{model_name}[/cyan] not pulled yet. Downloading...", style="dim")
    pull_ok, pull_error = pull_ollama_model(model_name)

    if not pull_ok and ("newer version" in pull_error.lower() or "412" in pull_error):
        # Step 1: Restart in case a newer binary is installed but old daemon is running.
        console.print("  [dim]Ollama needs a newer version. Restarting daemon...[/dim]")
        _restart_ollama()
        pull_ok, pull_error = pull_ollama_model(model_name)

    if not pull_ok and ("newer version" in pull_error.lower() or "412" in pull_error):
        # Step 2: Homebrew may lag behind — try the official curl installer for the latest.
        console.print("  [dim]Updating Ollama via official installer (this may take a minute)...[/dim]")
        install_ollama_result = subprocess.run(
            ["bash", "-c", "curl -fsSL https://ollama.com/install.sh | sh"],
            timeout=300,
        )
        if install_ollama_result.returncode == 0:
            console.print("  [dim]Restarting Ollama...[/dim]")
            _restart_ollama()
            console.print(f"  [dim]Retrying pull of {model_name}...[/dim]")
            pull_ok, pull_error = pull_ollama_model(model_name)

    if not pull_ok:
        console.print(
            f"\n[red]{label} [cyan]{model_name}[/cyan] download failed.[/red]"
        )
        if "newer version" in pull_error.lower() or "412" in pull_error:
            console.print(
                "  [bold]Your Ollama version is outdated.[/bold]\n"
                "  Update Ollama: [cyan]https://ollama.com/download[/cyan]\n"
                f"  Then re-run [cyan]autodidact init[/cyan]",
                style="dim",
            )
        else:
            console.print(
                "  Likely cause: network issue (TLS handshake failure, proxy, or firewall).\n"
                "  Options:\n"
                f"    1. [cyan]ollama pull {model_name}[/cyan] manually to see the full error\n"
                "    2. Check your internet connection / VPN / proxy settings\n"
                "    3. Try on a different network (e.g. personal hotspot)\n"
                f"    4. Once pulled, re-run [cyan]autodidact init[/cyan]\n"
                "\n"
                "  [bold]On a corporate network?[/bold] Try disabling VPN and run\n"
                "  [cyan]autodidact init[/cyan] again. Or choose mode 2 (Cloud + Cloud)\n"
                "  — no Ollama needed, just an API key.",
                style="dim",
            )
        raise typer.Exit(1)

    if not verify_model_loadable(model_name):
        console.print(
            f"\n[red]{label} [cyan]{model_name}[/cyan] pulled but cannot be loaded locally.[/red]"
        )
        console.print(
            "  Likely cause: this tag points to cloud-only inference "
            "(e.g. qwen3.5:9b, *:cloud). Pick a tag with real weights.\n"
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


def _init_custom_server() -> dict:
    """Run the custom local server init flow. Any OpenAI-compatible server.

    Works with: llama.cpp server, LM Studio, vLLM, text-generation-inference,
    LocalAI, or any server that speaks the OpenAI chat completions API.
    """
    console.print("\n[bold]Custom local server[/bold]")
    console.print("  Any server that speaks the OpenAI chat completions API.")
    console.print()
    console.print("  Popular options:", style="dim")
    console.print("    • [cyan]LM Studio[/cyan]     — GUI app, download from lmstudio.ai", style="dim")
    console.print("    • [cyan]llama.cpp[/cyan]     — CLI: brew install llama.cpp && llama-server -m model.gguf", style="dim")
    console.print("    • [cyan]vLLM[/cyan]          — pip install vllm && vllm serve model-name", style="dim")
    console.print("    • [cyan]LocalAI[/cyan]       — docker run -p 8080:8080 localai/localai", style="dim")
    console.print()

    base_url = typer.prompt(
        "  Server URL",
        default="http://localhost:8080/v1",
    ).strip().rstrip("/")

    model = typer.prompt(
        "  Model name (as the server knows it)",
        default="default",
    ).strip()

    console.print()
    has_cloud = typer.confirm("Add a cloud model for escalation (learning)?", default=True)

    if has_cloud:
        console.print("\n[bold]Cloud model[/bold] (escalation target):")
        cloud_cfg = _prompt_single_cloud_provider(slot="cloud")
        return build_config(
            mode="cloud_cloud",
            cheap_cloud_provider="openai",
            cheap_cloud_model=model,
            cheap_cloud_api_key=None,
            cheap_cloud_base_url=base_url,
            expensive_cloud_provider=cloud_cfg["provider"],
            expensive_cloud_model=cloud_cfg["model"],
            expensive_cloud_api_key=cloud_cfg["api_key"],
            expensive_cloud_base_url=cloud_cfg.get("base_url"),
            expensive_cloud_bedrock=cloud_cfg.get("bedrock"),
        )

    # No cloud — local only via custom server.
    return {
        "local": {
            "provider": "openai",
            "model": model,
            "base_url": base_url,
        },
        "routing": {"confidence_threshold": 0.7},
    }


__all__ = [
    "_ensure_ollama_running",
    "_init_cloud_to_cloud",
    "_init_custom_server",
    "_init_with_ollama",
    "_offer_to_install_ollama",
    "_offer_to_start_ollama",
    "_pull_and_verify",
    "_restart_ollama",
]

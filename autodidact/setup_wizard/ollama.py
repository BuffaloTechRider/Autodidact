"""Ollama detection, install, daemon, and model lifecycle.

Library-level helpers — no interactive prompting. The wizard's interactive
flow (asking the user before installing, etc.) lives in flow.py and prompts.py.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Optional

import requests


# ── Detection ────────────────────────────────────────────────────


@dataclass
class OllamaStatus:
    installed: bool
    path: Optional[str]


def detect_ollama() -> OllamaStatus:
    """Check if Ollama is installed and return its path."""
    path = shutil.which("ollama")
    return OllamaStatus(installed=path is not None, path=path)


def list_ollama_models() -> list[str]:
    """List models currently pulled in Ollama."""
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return []
        models = []
        for line in result.stdout.strip().split("\n")[1:]:  # skip header
            if line.strip():
                name = line.split()[0]
                models.append(name)
        return models
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return []


def is_model_available(model_name: str) -> bool:
    """Check if Ollama can serve this model locally.

    Thin alias for ``verify_model_loadable`` — kept under the historical
    name because callers across the codebase use both. The two functions
    are now identical.
    """
    return verify_model_loadable(model_name)


def verify_model_loadable(model_name: str) -> bool:
    """Check that Ollama can actually serve this model locally.

    Asks Ollama directly via ``POST /api/show``. This handles three things
    that subprocess-based ``ollama list`` parsing got wrong:

    1. **Tag normalization.** ``foo`` and ``foo:latest`` both resolve via
       Ollama itself — no string-matching heuristics needed.
    2. **Cloud-only manifests.** Some tags ('qwen3-coder:480b-cloud',
       certain Qwen 3.5 sizes on some days) ``pull`` a tiny manifest that
       points at remote inference, not local weights. Ollama's
       ``/api/show`` returns 200 for these but ``details.format`` is empty.
       We treat empty format as "not loadable locally."
    3. **Fewer subprocess calls.** One HTTP call vs spawning ``ollama list``.

    Returns False on any error (daemon down, timeout, malformed response).
    """
    try:
        resp = requests.post(
            "http://localhost:11434/api/show",
            json={"name": model_name},
            timeout=5.0,
        )
    except (requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
            requests.exceptions.RequestException):
        return False

    if resp.status_code != 200:
        return False

    try:
        body = resp.json()
    except (ValueError, requests.exceptions.JSONDecodeError):
        return False

    # A real local model has details.format like 'gguf' or 'safetensors'.
    # Cloud-only manifests have format='' (empty string).
    fmt = (body.get("details") or {}).get("format", "")
    return bool(fmt)


def pull_ollama_model(model_name: str) -> tuple[bool, str]:
    """Pull a model via Ollama. Returns (success, error_output)."""
    try:
        result = subprocess.run(
            ["ollama", "pull", model_name],
            timeout=600,  # 10 min timeout for large models
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            return True, ""
        return False, result.stderr or result.stdout or ""
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as e:
        return False, str(e)


# ── Installation ─────────────────────────────────────────────────


def get_ollama_install_command() -> str:
    """Return the install command for Ollama on the current platform.

    macOS and Linux both use the official curl-piped installer — it works on
    both without Homebrew. Windows isn't supported by the auto-installer in
    v1.0, so we return the manual download URL instead.
    """
    if sys.platform == "darwin":
        return "curl -fsSL https://ollama.com/install.sh | sh"
    elif sys.platform.startswith("linux"):
        return "curl -fsSL https://ollama.com/install.sh | sh"
    else:
        # Windows or other.
        return "Download from https://ollama.com/download/windows"


def _has_homebrew() -> bool:
    """Check if Homebrew is available."""
    try:
        result = subprocess.run(["brew", "--version"], capture_output=True, timeout=5)
        return result.returncode == 0
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return False


def install_ollama(retries: int = 2) -> bool:
    """Run the Ollama installer for the current platform.

    On macOS, prefers Homebrew if available (works better on corporate
    networks/VPNs). Falls back to the official curl installer.

    Retries on failure (transient 403s from Ollama's CDN are common).
    Returns True on success, False otherwise. Does NOT confirm with the user
    — the caller is responsible for getting consent before invoking this.

    Windows is not supported; returns False without attempting anything.
    """
    if sys.platform not in ("darwin",) and not sys.platform.startswith("linux"):
        return False

    # On macOS, try Homebrew first (better for corporate networks/VPNs).
    if sys.platform == "darwin" and _has_homebrew():
        try:
            result = subprocess.run(["brew", "install", "ollama"], timeout=600)
            if result.returncode == 0:
                return True
        except (OSError, subprocess.TimeoutExpired):
            pass

    # Fall back to the official curl installer.
    for attempt in range(retries):
        try:
            result = subprocess.run(
                ["bash", "-c", "set -o pipefail; curl -fsSL https://ollama.com/install.sh | sh"],
                timeout=600,
            )
            if result.returncode == 0:
                return True
        except (OSError, subprocess.TimeoutExpired):
            pass
        if attempt < retries - 1:
            time.sleep(3)
    return False


# ── Daemon lifecycle ─────────────────────────────────────────────


def is_ollama_running() -> bool:
    """Check whether the Ollama daemon is responding on localhost:11434.

    Connection errors, timeouts, and non-200 responses all return False.
    """
    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=2.0)
        return resp.status_code == 200
    except (requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
            requests.exceptions.RequestException):
        return False


def wait_for_ollama_daemon(timeout_s: float = 30.0, poll_interval_s: float = 0.5) -> bool:
    """Poll is_ollama_running until True or timeout. Returns True iff up."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if is_ollama_running():
            return True
        time.sleep(poll_interval_s)
    # One last check, in case the loop exited just after a sleep.
    return is_ollama_running()


def start_ollama_daemon(wait_timeout_s: float = 30.0) -> bool:
    """Best-effort start of the Ollama daemon.

    macOS: tries `open -a Ollama` (GUI app from official installer), falls
    back to `ollama serve` (Homebrew install). Linux: `ollama serve`.

    Returns True iff is_ollama_running becomes True within wait_timeout_s.
    """
    if sys.platform == "darwin":
        cmds = [["open", "-a", "Ollama"], ["ollama", "serve"]]
    elif sys.platform.startswith("linux"):
        cmds = [["ollama", "serve"]]
    else:
        return False

    for cmd in cmds:
        try:
            subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
            )
            if wait_for_ollama_daemon(timeout_s=wait_timeout_s):
                return True
        except (OSError, FileNotFoundError):
            continue
    return False


__all__ = [
    "OllamaStatus",
    "detect_ollama",
    "get_ollama_install_command",
    "install_ollama",
    "is_model_available",
    "is_ollama_running",
    "list_ollama_models",
    "pull_ollama_model",
    "start_ollama_daemon",
    "verify_model_loadable",
    "wait_for_ollama_daemon",
]

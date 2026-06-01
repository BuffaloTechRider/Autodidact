"""Setup wizard — zero-friction Ollama detection, model pulling, and config generation.

Handles four setup modes:
- local_cloud: Ollama local model + cloud escalation (default)
- cloud_cloud: cheap cloud model + expensive cloud model (no local)
- local_local: small Ollama model + bigger Ollama model
- local_only: Ollama local model, no cloud

The package is organized by concern:
  - ollama.py    — Ollama detection, install, daemon, model pull/verify
  - presets.py   — _CLOUD_PRESETS registry + accessors
  - discovery.py — Bedrock + OpenRouter live model discovery
  - builder.py   — build_config (the YAML emit)
  - picker.py    — _pick_from_list, questionary detection
  - prompts.py   — interactive prompt flows for cloud providers
  - flow.py      — _init_with_ollama / _init_cloud_to_cloud / _init_custom_server
  - smoke.py     — _run_smoke_test, _render_smoke_test_error

Public re-exports below preserve every name that callers used when this
was a single ``setup_wizard.py`` file. ``from autodidact.setup_wizard
import X`` keeps working for every X.

The module-level imports (``requests``, ``subprocess``, ``sys``, ``time``)
are re-exported on purpose: existing tests patch
``autodidact.setup_wizard.requests.post`` etc. and rely on those paths
resolving to the live module objects.
"""

from __future__ import annotations

# Module-level imports re-exported for backwards-compatible test patches.
import os         # noqa: F401  # patched by some tests
import shutil     # noqa: F401
import subprocess  # noqa: F401  # patched by tests
import sys         # noqa: F401  # patched by tests
import time        # noqa: F401  # patched by tests

import requests    # noqa: F401  # patched by tests via autodidact.setup_wizard.requests

# ── Library helpers (pure functions, no UI) ─────────────────────

from autodidact.setup_wizard.ollama import (
    OllamaStatus,
    detect_ollama,
    get_ollama_install_command,
    install_ollama,
    is_model_available,
    is_ollama_running,
    list_ollama_models,
    pull_ollama_model,
    start_ollama_daemon,
    verify_model_loadable,
    wait_for_ollama_daemon,
    _has_homebrew,  # patched by tests
)
from autodidact.setup_wizard.presets import (
    get_cloud_preset,
    list_cloud_providers,
    _CLOUD_PRESETS,
)
from autodidact.setup_wizard.discovery import (
    BedrockDiscoveryError,
    OpenRouterDiscoveryError,
    OpenRouterModel,
    discover_bedrock_models,
    discover_openrouter_models,
    _import_boto3,  # patched by tests
    _REGION_TO_PROFILE_PREFIX,  # patched by tests
)
from autodidact.setup_wizard.builder import build_config


__all__ = [
    "BedrockDiscoveryError",
    "OllamaStatus",
    "OpenRouterDiscoveryError",
    "OpenRouterModel",
    "build_config",
    "detect_ollama",
    "discover_bedrock_models",
    "discover_openrouter_models",
    "get_cloud_preset",
    "get_ollama_install_command",
    "install_ollama",
    "is_model_available",
    "is_ollama_running",
    "list_cloud_providers",
    "list_ollama_models",
    "pull_ollama_model",
    "start_ollama_daemon",
    "verify_model_loadable",
    "wait_for_ollama_daemon",
]

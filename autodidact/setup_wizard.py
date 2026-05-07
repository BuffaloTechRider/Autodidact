"""Setup wizard — zero-friction Ollama detection, model pulling, and config generation.

Handles three setup modes:
- local_cloud: Ollama local model + cloud escalation (default)
- cloud_cloud: cheap cloud model + expensive cloud model (no local)
- local_only: Ollama local model, no cloud

Auto-detects Ollama installation and pulled models. Provides install
commands per platform and cloud provider presets for common APIs.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from dataclasses import dataclass
from typing import Optional


# ── Ollama detection ─────────────────────────────────────────────

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
    """Check if a specific model is pulled in Ollama."""
    models = list_ollama_models()
    return model_name in models


def pull_ollama_model(model_name: str) -> bool:
    """Pull a model via Ollama. Returns True on success."""
    try:
        result = subprocess.run(
            ["ollama", "pull", model_name],
            timeout=600,  # 10 min timeout for large models
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return False


# ── Platform-specific install commands ───────────────────────────

def get_ollama_install_command() -> str:
    """Return the install command for Ollama on the current platform."""
    if sys.platform == "darwin":
        return "brew install ollama"
    elif sys.platform.startswith("linux"):
        return "curl -fsSL https://ollama.com/install.sh | sh"
    else:
        return "curl -fsSL https://ollama.com/install.sh | sh"


# ── Cloud provider presets ───────────────────────────────────────

_CLOUD_PRESETS: dict[str, dict] = {
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
        "models": ["gpt-4o", "gpt-4o-mini", "gpt-4.1-mini"],
        "default_cheap": "gpt-4o-mini",
        "default_expensive": "gpt-4o",
        "embedding_model": "text-embedding-3-small",
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
        "models": ["openai/gpt-4o", "anthropic/claude-sonnet-4-5", "google/gemini-2.5-flash"],
        "default_cheap": "google/gemini-2.5-flash",
        "default_expensive": "anthropic/claude-sonnet-4-5",
        "embedding_model": "openai/text-embedding-3-small",
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com/v1",
        "api_key_env": "DEEPSEEK_API_KEY",
        "models": ["deepseek-chat", "deepseek-reasoner"],
        "default_cheap": "deepseek-chat",
        "default_expensive": "deepseek-reasoner",
        "embedding_model": None,
    },
    "bedrock": {
        "base_url": "",
        "api_key_env": "",
        "models": ["claude-sonnet-4-5", "claude-haiku"],
        "default_cheap": "claude-haiku",
        "default_expensive": "claude-sonnet-4-5",
        "embedding_model": None,
    },
}


def get_cloud_preset(provider: str) -> dict:
    """Get preset config for a cloud provider."""
    if provider in _CLOUD_PRESETS:
        return _CLOUD_PRESETS[provider]
    return {
        "base_url": "",
        "api_key_env": "",
        "models": [],
        "default_cheap": "",
        "default_expensive": "",
        "embedding_model": None,
    }


def list_cloud_providers() -> list[str]:
    """List available cloud provider presets."""
    return list(_CLOUD_PRESETS.keys())


# ── Config builder ───────────────────────────────────────────────

def build_config(
    mode: str = "local_cloud",
    *,
    # local_cloud and local_only
    local_model: Optional[str] = None,
    embedding_model: Optional[str] = None,
    # local_cloud
    cloud_provider: Optional[str] = None,
    cloud_model: Optional[str] = None,
    cloud_api_key: Optional[str] = None,
    cloud_base_url: Optional[str] = None,
    # cloud_cloud
    cheap_cloud_provider: Optional[str] = None,
    cheap_cloud_model: Optional[str] = None,
    cheap_cloud_api_key: Optional[str] = None,
    cheap_cloud_base_url: Optional[str] = None,
    expensive_cloud_provider: Optional[str] = None,
    expensive_cloud_model: Optional[str] = None,
    expensive_cloud_api_key: Optional[str] = None,
    expensive_cloud_base_url: Optional[str] = None,
    # common
    db_path: str = "~/.autodidact/memory.db",
    confidence_threshold: float = 0.7,
) -> dict:
    """Build a config dict for the given setup mode."""
    config: dict = {
        "routing": {"confidence_threshold": confidence_threshold},
        "memory": {"path": db_path},
    }

    if mode == "local_cloud":
        config["local"] = {
            "model": local_model or "qwen2.5:7b",
            "embedding_model": embedding_model or "qllama/bge-large-en-v1.5",
        }
        if cloud_provider and cloud_model:
            cloud_cfg: dict = {
                "provider": cloud_provider,
                "model": cloud_model,
            }
            if cloud_api_key:
                cloud_cfg["api_key"] = cloud_api_key
            if cloud_base_url:
                cloud_cfg["base_url"] = cloud_base_url
            config["cloud"] = cloud_cfg

    elif mode == "cloud_cloud":
        # "Local" slot is the cheap cloud model.
        cheap_preset = get_cloud_preset(cheap_cloud_provider or "openai")
        expensive_preset = get_cloud_preset(expensive_cloud_provider or "openai")

        config["local"] = {
            "provider": cheap_cloud_provider or "openai",
            "model": cheap_cloud_model or cheap_preset.get("default_cheap", ""),
            "base_url": cheap_cloud_base_url or cheap_preset.get("base_url", ""),
            "embedding_model": cheap_preset.get("embedding_model") or "text-embedding-3-small",
        }
        if cheap_cloud_api_key:
            config["local"]["api_key"] = cheap_cloud_api_key

        config["cloud"] = {
            "provider": expensive_cloud_provider or "openai",
            "model": expensive_cloud_model or expensive_preset.get("default_expensive", ""),
        }
        if expensive_cloud_base_url:
            config["cloud"]["base_url"] = expensive_cloud_base_url
        if expensive_cloud_api_key:
            config["cloud"]["api_key"] = expensive_cloud_api_key

    elif mode == "local_only":
        config["local"] = {
            "model": local_model or "qwen2.5:7b",
            "embedding_model": embedding_model or "qllama/bge-large-en-v1.5",
        }

    return config

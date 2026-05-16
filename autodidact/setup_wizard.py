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
import time
from dataclasses import dataclass
from typing import Optional

import requests


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


def install_ollama(retries: int = 2) -> bool:
    """Run the Ollama installer for the current platform.

    Retries on failure (transient 403s from Ollama's CDN are common).
    Returns True on success, False otherwise. Does NOT confirm with the user
    — the caller is responsible for getting consent before invoking this.

    Windows is not supported; returns False without attempting anything.
    """
    if sys.platform not in ("darwin",) and not sys.platform.startswith("linux"):
        return False

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

    macOS: opens the Ollama.app via `open -a Ollama` (the daemon ships as a
    GUI app there). Linux: spawns `ollama serve` in the background.

    Returns True iff is_ollama_running becomes True within wait_timeout_s.
    """
    if sys.platform == "darwin":
        cmd = ["open", "-a", "Ollama"]
    elif sys.platform.startswith("linux"):
        cmd = ["ollama", "serve"]
    else:
        return False

    try:
        # Detach so we don't keep the wizard waiting on the daemon process.
        subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    except (OSError, FileNotFoundError):
        return False

    return wait_for_ollama_daemon(timeout_s=wait_timeout_s)


# ── Cloud provider presets ───────────────────────────────────────

_CLOUD_PRESETS: dict[str, dict] = {
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
        "models": [
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4.1",
            "gpt-4.1-mini",
            "gpt-4.1-nano",
            "o3-mini",
            "o1",
        ],
        "default_cheap": "gpt-4o-mini",
        "default_expensive": "gpt-4o",
        "embedding_model": "text-embedding-3-small",
    },
    "anthropic": {
        # Anthropic's OpenAI-compat shim — works with our openai-provider client.
        # Direct API has quirks; OpenRouter route is also supported for Claude.
        "base_url": "https://api.anthropic.com/v1",
        "api_key_env": "ANTHROPIC_API_KEY",
        "models": [
            "claude-sonnet-4-5",
            "claude-opus-4",
            "claude-haiku-4",
            "claude-3-5-sonnet-20241022",
            "claude-3-5-haiku-20241022",
            "claude-3-opus-20240229",
        ],
        "default_cheap": "claude-haiku-4",
        "default_expensive": "claude-sonnet-4-5",
        "embedding_model": None,
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
        "models": [
            "openai/gpt-4o",
            "openai/gpt-4o-mini",
            "anthropic/claude-sonnet-4-5",
            "anthropic/claude-haiku-4",
            "google/gemini-2.5-pro",
            "google/gemini-2.5-flash",
            "meta-llama/llama-3.3-70b-instruct",
            "deepseek/deepseek-chat",
        ],
        "default_cheap": "google/gemini-2.5-flash",
        "default_expensive": "anthropic/claude-sonnet-4-5",
        "embedding_model": "openai/text-embedding-3-small",
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com/v1",
        "api_key_env": "DEEPSEEK_API_KEY",
        "models": [
            "deepseek-chat",
            "deepseek-reasoner",
            "deepseek-coder",
        ],
        "default_cheap": "deepseek-chat",
        "default_expensive": "deepseek-reasoner",
        "embedding_model": None,
    },
    "mistral": {
        "base_url": "https://api.mistral.ai/v1",
        "api_key_env": "MISTRAL_API_KEY",
        "models": [
            "mistral-large-latest",
            "mistral-medium-latest",
            "mistral-small-latest",
            "pixtral-large-latest",
            "codestral-latest",
        ],
        "default_cheap": "mistral-small-latest",
        "default_expensive": "mistral-large-latest",
        "embedding_model": "mistral-embed",
    },
    "groq": {
        # Fastest OpenAI-compat inference; great for the cheap slot in cloud+cloud.
        "base_url": "https://api.groq.com/openai/v1",
        "api_key_env": "GROQ_API_KEY",
        "models": [
            "llama-3.3-70b-versatile",
            "llama-3.1-70b-versatile",
            "llama-3.1-8b-instant",
            "mixtral-8x7b-32768",
            "gemma2-9b-it",
        ],
        "default_cheap": "llama-3.1-8b-instant",
        "default_expensive": "llama-3.3-70b-versatile",
        "embedding_model": None,
    },
    "together": {
        "base_url": "https://api.together.xyz/v1",
        "api_key_env": "TOGETHER_API_KEY",
        "models": [
            "meta-llama/Llama-3.3-70B-Instruct-Turbo",
            "meta-llama/Llama-3.1-8B-Instruct-Turbo",
            "Qwen/Qwen2.5-72B-Instruct-Turbo",
            "Qwen/Qwen2.5-7B-Instruct-Turbo",
            "deepseek-ai/DeepSeek-V3",
            "mistralai/Mixtral-8x7B-Instruct-v0.1",
        ],
        "default_cheap": "meta-llama/Llama-3.1-8B-Instruct-Turbo",
        "default_expensive": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        "embedding_model": "togethercomputer/m2-bert-80M-8k-retrieval",
    },
    "fireworks": {
        "base_url": "https://api.fireworks.ai/inference/v1",
        "api_key_env": "FIREWORKS_API_KEY",
        "models": [
            "accounts/fireworks/models/llama-v3p3-70b-instruct",
            "accounts/fireworks/models/llama-v3p1-8b-instruct",
            "accounts/fireworks/models/qwen2p5-72b-instruct",
            "accounts/fireworks/models/deepseek-v3",
            "accounts/fireworks/models/mixtral-8x22b-instruct",
        ],
        "default_cheap": "accounts/fireworks/models/llama-v3p1-8b-instruct",
        "default_expensive": "accounts/fireworks/models/llama-v3p3-70b-instruct",
        "embedding_model": None,
    },
    "xai": {
        # xAI Grok — OpenAI-compat.
        "base_url": "https://api.x.ai/v1",
        "api_key_env": "XAI_API_KEY",
        "models": [
            "grok-4",
            "grok-3",
            "grok-3-mini",
            "grok-2-latest",
        ],
        "default_cheap": "grok-3-mini",
        "default_expensive": "grok-4",
        "embedding_model": None,
    },
    "bedrock": {
        "base_url": "",
        "api_key_env": "",
        # The Bedrock model list is *discovered at wizard time* via
        # `discover_bedrock_models()` because (a) Bedrock evolves rapidly,
        # (b) availability differs per region, and (c) some models are
        # inference-profile-only with region-specific prefixes. The static
        # entries below are only a last-resort hint shown if discovery fails
        # AND the user wants to pick from a list rather than type freely.
        "models": [],
        "default_cheap": "",
        "default_expensive": "",
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


# ── Bedrock model discovery ──────────────────────────────────────


class BedrockDiscoveryError(Exception):
    """Raised when we can't enumerate Bedrock models at wizard time.

    Wraps boto/botocore errors with their original message so the wizard
    can show the user *why* discovery failed (auth, network, perms).
    """


def _import_boto3():
    """Imported via a small helper so tests can patch it cleanly."""
    import boto3  # type: ignore
    return boto3


# Map AWS region prefixes to the inference-profile ID prefixes that work
# from that region. `global.*` profiles are usable from anywhere.
_REGION_TO_PROFILE_PREFIX = {
    "us-": "us.",
    "eu-": "eu.",
    "ap-": "apac.",
}


def _profile_prefix_for_region(region: str) -> Optional[str]:
    for region_prefix, profile_prefix in _REGION_TO_PROFILE_PREFIX.items():
        if region.startswith(region_prefix):
            return profile_prefix
    return None


def discover_bedrock_models(
    *,
    region: str,
    auth_mode: str = "default",
    access_key_id: Optional[str] = None,
    secret_access_key: Optional[str] = None,
    session_token: Optional[str] = None,
    api_key: Optional[str] = None,
) -> list[str]:
    """Return Bedrock model IDs the user can actually invoke from ``region``.

    Two API calls:
      1. ``list_foundation_models`` — keep entries with TEXT output, ON_DEMAND
         inference, and ACTIVE lifecycle. These IDs are used as-is.
      2. ``list_inference_profiles`` (SYSTEM_DEFINED) — keep ACTIVE profiles
         whose ID prefix matches the region (us-* → us., eu-* → eu., ap-* →
         apac.) plus ``global.*`` profiles which work from any region.

    Merged, deduped, sorted. Raises :class:`BedrockDiscoveryError` if either
    boto3 is missing or the API calls fail; the wizard catches this and
    falls back to free-form input.
    """
    try:
        boto3 = _import_boto3()
    except ImportError as e:
        raise BedrockDiscoveryError(
            "boto3 is not installed. Install with `pip install autodidact[bedrock]`."
        ) from e

    client_kwargs: dict = {"service_name": "bedrock", "region_name": region}
    if auth_mode == "iam_user":
        if not (access_key_id and secret_access_key):
            raise BedrockDiscoveryError(
                "iam_user auth mode requires access_key_id and secret_access_key."
            )
        client_kwargs["aws_access_key_id"] = access_key_id
        client_kwargs["aws_secret_access_key"] = secret_access_key
        if session_token:
            client_kwargs["aws_session_token"] = session_token
    elif auth_mode == "api_key":
        if not api_key:
            raise BedrockDiscoveryError("api_key auth mode requires api_key.")
        # Bedrock API keys go through AWS_BEARER_TOKEN_BEDROCK; boto3 picks
        # them up from the env (same as the runtime client).
        import os
        os.environ["AWS_BEARER_TOKEN_BEDROCK"] = api_key

    try:
        client = boto3.client(**client_kwargs)
    except Exception as e:
        raise BedrockDiscoveryError(f"Could not create Bedrock client: {e}") from e

    try:
        fm_resp = client.list_foundation_models(byOutputModality="TEXT")
    except TypeError:
        # Fakes in tests may not accept kwargs; retry without filter.
        fm_resp = client.list_foundation_models()
    except Exception as e:
        raise BedrockDiscoveryError(f"list_foundation_models failed: {e}") from e

    on_demand_ids: set[str] = set()
    for entry in fm_resp.get("modelSummaries", []) or []:
        if "ON_DEMAND" not in (entry.get("inferenceTypesSupported") or []):
            continue
        if (entry.get("modelLifecycle") or {}).get("status") != "ACTIVE":
            continue
        modalities = entry.get("outputModalities") or []
        if modalities and "TEXT" not in modalities:
            continue
        on_demand_ids.add(entry["modelId"])

    try:
        ip_resp = client.list_inference_profiles(typeEquals="SYSTEM_DEFINED")
    except TypeError:
        ip_resp = client.list_inference_profiles()
    except Exception as e:
        raise BedrockDiscoveryError(f"list_inference_profiles failed: {e}") from e

    region_prefix = _profile_prefix_for_region(region)
    profile_ids: set[str] = set()
    for entry in ip_resp.get("inferenceProfileSummaries", []) or []:
        if entry.get("status") != "ACTIVE":
            continue
        pid = entry.get("inferenceProfileId") or ""
        if not pid:
            continue
        if pid.startswith("global."):
            profile_ids.add(pid)
        elif region_prefix and pid.startswith(region_prefix):
            profile_ids.add(pid)

    return sorted(on_demand_ids | profile_ids)


# ── OpenRouter model discovery ───────────────────────────────────


class OpenRouterDiscoveryError(Exception):
    """Raised when the OpenRouter /v1/models endpoint can't be reached or parsed."""


@dataclass
class OpenRouterModel:
    """A single OpenRouter model surfaced by discovery.

    Pricing is normalized to USD per 1M tokens (the unit users read in
    OpenRouter's docs / pricing pages). The raw API returns USD per token.
    """
    id: str
    prompt_per_million: float
    completion_per_million: float
    context_length: int


_OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"


def discover_openrouter_models() -> list[OpenRouterModel]:
    """Query OpenRouter's public /v1/models endpoint.

    No API key is needed — the catalog is public. Filters to text-output
    models with valid pricing, sorts cheapest first by prompt+completion
    cost. Returns an empty list only if the API returns zero usable models;
    raises :class:`OpenRouterDiscoveryError` for any network or HTTP error.
    """
    try:
        resp = requests.get(_OPENROUTER_MODELS_URL, timeout=10.0)
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise OpenRouterDiscoveryError(str(e)) from e

    try:
        payload = resp.json()
    except ValueError as e:
        raise OpenRouterDiscoveryError(f"non-JSON response: {e}") from e

    out: list[OpenRouterModel] = []
    for entry in payload.get("data", []) or []:
        model_id = entry.get("id")
        if not model_id:
            continue
        modalities = (entry.get("architecture") or {}).get("output_modalities") or []
        if modalities and "text" not in modalities:
            continue

        pricing = entry.get("pricing") or {}
        prompt_str = pricing.get("prompt")
        completion_str = pricing.get("completion")
        if prompt_str is None or completion_str is None:
            continue
        try:
            prompt = float(prompt_str)
            completion = float(completion_str)
        except (TypeError, ValueError):
            continue

        # OpenRouter encodes "dynamic / unknown pricing" as -1 (used by their
        # auto-router meta-models like openrouter/auto). Skip these — they
        # aren't user-pickable models, just routing shortcuts.
        if prompt < 0 or completion < 0:
            continue

        out.append(OpenRouterModel(
            id=model_id,
            prompt_per_million=prompt * 1_000_000,
            completion_per_million=completion * 1_000_000,
            context_length=int(entry.get("context_length") or 0),
        ))

    out.sort(key=lambda m: m.prompt_per_million + m.completion_per_million)
    return out


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
    cloud_bedrock: Optional[dict] = None,
    # cloud_cloud
    cheap_cloud_provider: Optional[str] = None,
    cheap_cloud_model: Optional[str] = None,
    cheap_cloud_api_key: Optional[str] = None,
    cheap_cloud_base_url: Optional[str] = None,
    cheap_cloud_bedrock: Optional[dict] = None,
    expensive_cloud_provider: Optional[str] = None,
    expensive_cloud_model: Optional[str] = None,
    expensive_cloud_api_key: Optional[str] = None,
    expensive_cloud_base_url: Optional[str] = None,
    expensive_cloud_bedrock: Optional[dict] = None,
    # common
    db_path: str = "~/.autodidact/memory.db",
    confidence_threshold: float = 0.7,
) -> dict:
    """Build a config dict for the given setup mode.

    Bedrock-specific auth settings (auth_mode, access_key_id, api_key, region, ...)
    are passed via the *_cloud_bedrock dicts and stored under the 'bedrock' key
    on the cloud/local section.
    """
    config: dict = {
        "routing": {"confidence_threshold": confidence_threshold},
        "memory": {"path": db_path},
    }

    if mode == "local_cloud":
        config["local"] = {
            "model": local_model or "qwen3:4b",
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
            if cloud_bedrock:
                cloud_cfg["bedrock"] = cloud_bedrock
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
        if cheap_cloud_bedrock:
            config["local"]["bedrock"] = cheap_cloud_bedrock

        config["cloud"] = {
            "provider": expensive_cloud_provider or "openai",
            "model": expensive_cloud_model or expensive_preset.get("default_expensive", ""),
        }
        if expensive_cloud_base_url:
            config["cloud"]["base_url"] = expensive_cloud_base_url
        if expensive_cloud_api_key:
            config["cloud"]["api_key"] = expensive_cloud_api_key
        if expensive_cloud_bedrock:
            config["cloud"]["bedrock"] = expensive_cloud_bedrock

    elif mode == "local_local":
        config["local"] = {
            "model": local_model or "qwen3:4b",
            "embedding_model": embedding_model or "qllama/bge-large-en-v1.5",
        }
        if cloud_provider == "ollama" and cloud_model:
            config["cloud"] = {
                "provider": "ollama",
                "model": cloud_model,
            }

    elif mode == "local_only":
        config["local"] = {
            "model": local_model or "qwen3:4b",
            "embedding_model": embedding_model or "qllama/bge-large-en-v1.5",
        }

    return config

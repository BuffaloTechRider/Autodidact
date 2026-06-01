"""Interactive prompts for the init wizard.

Build on top of ``picker._pick_from_list``. Each function here corresponds
to one interactive question (or coherent group of questions) the wizard
asks the user during ``autodidact init``.

Owns:
  - The curated local-model list (_LOCAL_MODEL_CHOICES)
  - The provider-label registry (_PROVIDER_LABELS)
  - The "Browse all OpenRouter models" sentinel
  - All the _pick_* / _prompt_* helpers
"""

from __future__ import annotations

from typing import Optional

import typer

from autodidact.setup_wizard._console import console
from autodidact.setup_wizard.discovery import (
    BedrockDiscoveryError,
    OpenRouterDiscoveryError,
    discover_bedrock_models,
    discover_openrouter_models,
)
from autodidact.setup_wizard.picker import (
    _OTHER_CHOICE,
    _pick_from_list,
)
from autodidact.setup_wizard.presets import (
    get_cloud_preset,
    list_cloud_providers,
)


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

_PROVIDER_LABELS: dict[str, str] = {
    "google": "google (free tier available, no credit card needed)",
    "openai": "openai (requires API key, not ChatGPT subscription)",
    "anthropic": "anthropic (requires API key, not Claude subscription)",
    "openrouter": "openrouter (pay-per-token, all models, from $0)",
    "groq": "groq (free tier available, fast inference)",
}

_BROWSE_OPENROUTER_CHOICE = "↪ Browse all OpenRouter models (live)"


# ── Local model picker ──────────────────────────────────────────


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


# ── Cloud provider + cloud model pickers ────────────────────────


def _pick_cloud_provider() -> str:
    """Show the cloud provider list with 'other' fallback."""
    console.print(
        "\n  [dim]Note: ChatGPT/Claude subscriptions do NOT include API access.[/dim]"
        "\n  [dim]You need a separate API key — platform credits start at $5.[/dim]"
        "\n  [dim]  Google:    https://aistudio.google.com/apikey (free)[/dim]"
        "\n  [dim]  OpenAI:    https://platform.openai.com/api-keys[/dim]"
        "\n  [dim]  Anthropic: https://console.anthropic.com/settings/keys[/dim]\n"
    )
    providers = list_cloud_providers()
    choices = [_PROVIDER_LABELS.get(p, p) for p in providers] + [_OTHER_CHOICE]
    default_label = _PROVIDER_LABELS.get("google", "google")
    chosen = _pick_from_list("Cloud provider", choices, default_label)
    if chosen == _OTHER_CHOICE:
        return typer.prompt("Provider name").strip().lower()
    # Strip label decoration to get the raw provider key.
    for p in providers:
        if chosen == _PROVIDER_LABELS.get(p, p):
            return p
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


# ── Top-level mode picker ───────────────────────────────────────


def _pick_setup_mode() -> str:
    """Show the 5 setup modes as a list; return the canonical key."""
    labels = [
        "Local + Cloud   — Ollama local + cloud for escalation (best savings)",
        "Cloud + Cloud   — cheap cloud + expensive cloud (no GPU needed)",
        "Local + Local   — small Ollama + big Ollama (free, fully offline, still learns)",
        "Custom local    — any OpenAI-compatible local server (llama.cpp, LM Studio, vLLM)",
        "Local only      — Ollama only, no cloud (free, no learning escalations)",
    ]
    chosen = _pick_from_list("Pick a setup mode", labels, labels[0])
    if "Local + Cloud" in chosen:
        return "local_cloud"
    if "Cloud + Cloud" in chosen:
        return "cloud_cloud"
    if "Local + Local" in chosen:
        return "local_local"
    if "Custom local" in chosen:
        return "custom_server"
    return "local_only"


# ── Provider-config flow ────────────────────────────────────────


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


def _prompt_openai_compat_config(provider: str, preset: dict, slot: str) -> dict:
    """Prompt for an OpenAI-compatible provider: API key + model.

    OpenRouter gets a special ``Browse all`` entry in the picker that hits
    the public ``/v1/models`` endpoint. The catalogue is hundreds of models
    long and changes weekly; the curated preset can't keep up, and slug
    typos lose users at chat time.
    """
    if provider == "google":
        console.print("  [dim]Get a free key at: https://aistudio.google.com/apikey[/dim]")
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


# ── OpenRouter live-browse picker ───────────────────────────────


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


# ── Bedrock-specific flow ───────────────────────────────────────


def _prompt_bedrock_config(preset: dict, slot: str) -> dict:
    """Prompt for Bedrock: auth mode + region + model. No generic API key.

    The model list is *discovered* at this point by querying Bedrock with
    the supplied region + auth. If discovery fails (no creds, no perms,
    network), we fall back to a free-form prompt and surface the error.
    """
    auth_choices = [
        "IAM Role / default credential chain (env vars, ~/.aws/credentials, SSO, IMDS)",
        "IAM User (paste aws_access_key_id and aws_secret_access_key)",
        "Bedrock API key (short-lived bearer token from AWS Console)",
    ]
    auth_chosen = _pick_from_list("Bedrock auth mode", auth_choices, auth_choices[0])
    auth_mode_map = {
        auth_choices[0]: "default",
        auth_choices[1]: "iam_user",
        auth_choices[2]: "api_key",
    }
    auth_mode = auth_mode_map.get(auth_chosen, "default")

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
        # Offer common Bedrock models as defaults
        fallback_models = [
            "us.anthropic.claude-sonnet-4-5-20250514-v1:0",
            "us.anthropic.claude-haiku-4-20250514-v1:0",
            "us.anthropic.claude-opus-4-20250514-v1:0",
            "us.amazon.nova-pro-v1:0",
            "us.amazon.nova-lite-v1:0",
        ]
        keywords = ("haiku", "nova-lite") if slot == "cheap" else ("sonnet", "opus")
        default = next(
            (m for kw in keywords for m in fallback_models if kw in m),
            fallback_models[0],
        )
        choices = fallback_models + [_OTHER_CHOICE]
        chosen = _pick_from_list("Bedrock model", choices, default)
        if chosen == _OTHER_CHOICE:
            model = typer.prompt("  Model ID").strip()
        else:
            model = chosen

    return {
        "provider": "bedrock",
        "model": model,
        "api_key": None,  # not applicable to bedrock
        "base_url": None,
        "bedrock": bedrock_cfg,
    }


__all__ = [
    "_BROWSE_OPENROUTER_CHOICE",
    "_LOCAL_MODEL_CHOICES",
    "_PROVIDER_LABELS",
    "_browse_openrouter_models",
    "_pick_cloud_model",
    "_pick_cloud_provider",
    "_pick_local_model",
    "_pick_openrouter_model",
    "_pick_setup_mode",
    "_prompt_bedrock_config",
    "_prompt_model_name",
    "_prompt_openai_compat_config",
    "_prompt_single_cloud_provider",
]

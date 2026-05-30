"""Live model discovery for Bedrock and OpenRouter.

Bedrock: lists ON_DEMAND foundation models + region-matched system-defined
inference profiles. OpenRouter: hits the public /v1/models catalogue and
returns a sorted-by-price list.

Used by the wizard's interactive Bedrock and OpenRouter prompts to populate
the model picker. Both raise typed exceptions so the prompt path can fall
back to free-form input.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import requests


# ── Bedrock ──────────────────────────────────────────────────────


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


# ── OpenRouter ───────────────────────────────────────────────────


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


__all__ = [
    "BedrockDiscoveryError",
    "OpenRouterDiscoveryError",
    "OpenRouterModel",
    "discover_bedrock_models",
    "discover_openrouter_models",
]

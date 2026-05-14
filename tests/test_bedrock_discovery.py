"""Tests for runtime discovery of Bedrock model IDs.

The hardcoded `_CLOUD_PRESETS["bedrock"]["models"]` list rots fast — Bedrock
adds models, retires others, and renames inference profiles. Worse, the same
ID may be on-demand in one region and inference-profile-only in another, and
*new* Anthropic models are inference-profile-only entirely. A static list
guarantees ValidationException for some users.

The fix: at wizard time, after the user has supplied region + auth, query
Bedrock and merge the actual on-demand foundation-model IDs with the
inference-profile IDs that match the user's region. Fall back to free-form
input if discovery fails.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


class _FakeBoto3:
    """Minimal stand-in for the boto3 module that hands out a configured client."""

    def __init__(self, client):
        self._client = client

    def client(self, *args, **kwargs):
        return self._client


# ── discover_bedrock_models ──────────────────────────────────────


class TestDiscoverBedrockModelsHappyPath:
    """Discovery merges on-demand foundation models with region-matched inference profiles."""

    def _client_with(self, *, foundation_models, inference_profiles):
        client = MagicMock()
        client.list_foundation_models.return_value = {
            "modelSummaries": foundation_models,
        }
        client.list_inference_profiles.return_value = {
            "inferenceProfileSummaries": inference_profiles,
        }
        return client

    def test_merges_on_demand_models_and_us_inference_profiles(self):
        from autodidact.setup_wizard import discover_bedrock_models

        client = self._client_with(
            foundation_models=[
                {
                    "modelId": "anthropic.claude-3-5-haiku-20241022-v1:0",
                    "inferenceTypesSupported": ["ON_DEMAND"],
                    "modelLifecycle": {"status": "ACTIVE"},
                    "outputModalities": ["TEXT"],
                },
                {
                    "modelId": "mistral.mistral-large-2407-v1:0",
                    "inferenceTypesSupported": ["ON_DEMAND"],
                    "modelLifecycle": {"status": "ACTIVE"},
                    "outputModalities": ["TEXT"],
                },
                # Inference-profile-only — should be dropped from FM list, picked up via profile list.
                {
                    "modelId": "anthropic.claude-sonnet-4-5-20250929-v1:0",
                    "inferenceTypesSupported": ["INFERENCE_PROFILE"],
                    "modelLifecycle": {"status": "ACTIVE"},
                    "outputModalities": ["TEXT"],
                },
            ],
            inference_profiles=[
                {
                    "inferenceProfileId": "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
                    "status": "ACTIVE",
                },
                {
                    "inferenceProfileId": "us.anthropic.claude-3-5-haiku-20241022-v1:0",
                    "status": "ACTIVE",
                },
                {
                    "inferenceProfileId": "global.anthropic.claude-opus-4-5-20251101-v1:0",
                    "status": "ACTIVE",
                },
            ],
        )

        with patch("autodidact.setup_wizard._import_boto3", return_value=_FakeBoto3(client)):
            ids = discover_bedrock_models(region="us-west-2", auth_mode="default")

        # On-demand models surface as bare IDs.
        assert "anthropic.claude-3-5-haiku-20241022-v1:0" in ids
        assert "mistral.mistral-large-2407-v1:0" in ids
        # Region-matched inference profiles surface with their prefix.
        assert "us.anthropic.claude-sonnet-4-5-20250929-v1:0" in ids
        assert "us.anthropic.claude-3-5-haiku-20241022-v1:0" in ids
        # global.* profiles are available from any region.
        assert "global.anthropic.claude-opus-4-5-20251101-v1:0" in ids

    def test_eu_region_keeps_eu_profiles_only(self):
        from autodidact.setup_wizard import discover_bedrock_models

        client = self._client_with(
            foundation_models=[],
            inference_profiles=[
                {"inferenceProfileId": "us.anthropic.claude-3-5-haiku-20241022-v1:0",
                 "status": "ACTIVE"},
                {"inferenceProfileId": "eu.anthropic.claude-3-5-sonnet-20240620-v1:0",
                 "status": "ACTIVE"},
                {"inferenceProfileId": "apac.anthropic.claude-3-5-sonnet-20240620-v1:0",
                 "status": "ACTIVE"},
                {"inferenceProfileId": "global.anthropic.claude-opus-4-5-20251101-v1:0",
                 "status": "ACTIVE"},
            ],
        )

        with patch("autodidact.setup_wizard._import_boto3", return_value=_FakeBoto3(client)):
            ids = discover_bedrock_models(region="eu-west-1", auth_mode="default")

        assert "eu.anthropic.claude-3-5-sonnet-20240620-v1:0" in ids
        assert "global.anthropic.claude-opus-4-5-20251101-v1:0" in ids
        # us.* / apac.* profiles must not show up for an eu user.
        assert not any(i.startswith("us.") for i in ids)
        assert not any(i.startswith("apac.") for i in ids)

    def test_apac_region_keeps_apac_profiles(self):
        from autodidact.setup_wizard import discover_bedrock_models

        client = self._client_with(
            foundation_models=[],
            inference_profiles=[
                {"inferenceProfileId": "us.anthropic.claude-3-5-haiku-20241022-v1:0",
                 "status": "ACTIVE"},
                {"inferenceProfileId": "apac.anthropic.claude-3-5-sonnet-20240620-v1:0",
                 "status": "ACTIVE"},
            ],
        )

        with patch("autodidact.setup_wizard._import_boto3", return_value=_FakeBoto3(client)):
            ids = discover_bedrock_models(region="ap-northeast-1", auth_mode="default")

        assert "apac.anthropic.claude-3-5-sonnet-20240620-v1:0" in ids
        assert not any(i.startswith("us.") for i in ids)

    def test_drops_inactive_models(self):
        from autodidact.setup_wizard import discover_bedrock_models

        client = self._client_with(
            foundation_models=[
                {
                    "modelId": "anthropic.claude-instant-v1",
                    "inferenceTypesSupported": ["ON_DEMAND"],
                    "modelLifecycle": {"status": "LEGACY"},
                    "outputModalities": ["TEXT"],
                },
                {
                    "modelId": "anthropic.claude-3-5-haiku-20241022-v1:0",
                    "inferenceTypesSupported": ["ON_DEMAND"],
                    "modelLifecycle": {"status": "ACTIVE"},
                    "outputModalities": ["TEXT"],
                },
            ],
            inference_profiles=[
                {"inferenceProfileId": "us.anthropic.claude-3-5-haiku-20241022-v1:0",
                 "status": "INACTIVE"},
            ],
        )

        with patch("autodidact.setup_wizard._import_boto3", return_value=_FakeBoto3(client)):
            ids = discover_bedrock_models(region="us-west-2", auth_mode="default")

        assert "anthropic.claude-instant-v1" not in ids
        assert "us.anthropic.claude-3-5-haiku-20241022-v1:0" not in ids
        assert "anthropic.claude-3-5-haiku-20241022-v1:0" in ids

    def test_drops_non_text_modalities(self):
        from autodidact.setup_wizard import discover_bedrock_models

        client = self._client_with(
            foundation_models=[
                {
                    "modelId": "amazon.titan-image-generator-v2:0",
                    "inferenceTypesSupported": ["ON_DEMAND"],
                    "modelLifecycle": {"status": "ACTIVE"},
                    "outputModalities": ["IMAGE"],
                },
                {
                    "modelId": "anthropic.claude-3-5-haiku-20241022-v1:0",
                    "inferenceTypesSupported": ["ON_DEMAND"],
                    "modelLifecycle": {"status": "ACTIVE"},
                    "outputModalities": ["TEXT"],
                },
            ],
            inference_profiles=[],
        )

        with patch("autodidact.setup_wizard._import_boto3", return_value=_FakeBoto3(client)):
            ids = discover_bedrock_models(region="us-west-2", auth_mode="default")

        assert "amazon.titan-image-generator-v2:0" not in ids
        assert "anthropic.claude-3-5-haiku-20241022-v1:0" in ids

    def test_result_is_sorted_and_deduped(self):
        from autodidact.setup_wizard import discover_bedrock_models

        client = self._client_with(
            foundation_models=[
                {
                    "modelId": "anthropic.claude-3-5-haiku-20241022-v1:0",
                    "inferenceTypesSupported": ["ON_DEMAND"],
                    "modelLifecycle": {"status": "ACTIVE"},
                    "outputModalities": ["TEXT"],
                },
                {
                    "modelId": "anthropic.claude-3-5-haiku-20241022-v1:0",
                    "inferenceTypesSupported": ["ON_DEMAND"],
                    "modelLifecycle": {"status": "ACTIVE"},
                    "outputModalities": ["TEXT"],
                },
            ],
            inference_profiles=[
                {"inferenceProfileId": "us.meta.llama3-3-70b-instruct-v1:0",
                 "status": "ACTIVE"},
            ],
        )

        with patch("autodidact.setup_wizard._import_boto3", return_value=_FakeBoto3(client)):
            ids = discover_bedrock_models(region="us-west-2", auth_mode="default")

        assert ids == sorted(set(ids))


# ── Failure modes ────────────────────────────────────────────────


class TestDiscoverBedrockModelsErrors:
    """Discovery surfaces typed errors so the wizard can fall back gracefully."""

    def test_missing_boto3_raises_typed_error(self):
        from autodidact.setup_wizard import (
            BedrockDiscoveryError,
            discover_bedrock_models,
        )

        with patch("autodidact.setup_wizard._import_boto3",
                   side_effect=ImportError("boto3 not installed")):
            with pytest.raises(BedrockDiscoveryError) as exc:
                discover_bedrock_models(region="us-west-2", auth_mode="default")
        assert "boto3" in str(exc.value).lower()

    def test_access_denied_raises_typed_error(self):
        from autodidact.setup_wizard import (
            BedrockDiscoveryError,
            discover_bedrock_models,
        )

        client = MagicMock()
        client.list_foundation_models.side_effect = Exception(
            "AccessDeniedException: not authorized to call ListFoundationModels"
        )

        with patch("autodidact.setup_wizard._import_boto3",
                   return_value=_FakeBoto3(client)):
            with pytest.raises(BedrockDiscoveryError) as exc:
                discover_bedrock_models(region="us-west-2", auth_mode="default")
        # Original error must be preserved in the message for the user.
        assert "AccessDenied" in str(exc.value) or "not authorized" in str(exc.value)


# ── Wizard wiring: discovery → picker, with fallback ─────────────


class TestPromptBedrockUsesDiscovery:
    """The Bedrock branch of _prompt_bedrock_config calls discovery and uses the result."""

    def test_picker_uses_discovered_ids_when_discovery_succeeds(self, monkeypatch):
        from autodidact import cli

        prompts = iter([
            "1",                    # auth mode 1 = default
            "us-west-2",            # region
        ])
        monkeypatch.setattr(cli.typer, "prompt", lambda *a, **k: next(prompts))

        captured: dict = {}

        def fake_pick(title, choices, default):
            captured["choices"] = choices
            captured["default"] = default
            return choices[0]

        monkeypatch.setattr(cli, "_pick_from_list", fake_pick)
        monkeypatch.setattr(
            cli,
            "discover_bedrock_models",
            lambda **kwargs: [
                "anthropic.claude-3-5-haiku-20241022-v1:0",
                "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
            ],
        )

        result = cli._prompt_bedrock_config(preset={"models": [], "default_cheap": ""}, slot="cheap")

        # The picker received discovered IDs, not the stale preset list.
        assert "anthropic.claude-3-5-haiku-20241022-v1:0" in captured["choices"]
        assert "us.anthropic.claude-sonnet-4-5-20250929-v1:0" in captured["choices"]
        assert result["model"] == "anthropic.claude-3-5-haiku-20241022-v1:0"
        assert result["bedrock"]["region"] == "us-west-2"

    def test_falls_back_to_free_form_when_discovery_fails(self, monkeypatch):
        from autodidact import cli
        from autodidact.setup_wizard import BedrockDiscoveryError

        prompts = iter([
            "1",                                                  # auth mode 1 = default
            "us-west-2",                                          # region
            "anthropic.claude-3-5-haiku-20241022-v1:0",           # free-form model
        ])
        monkeypatch.setattr(cli.typer, "prompt", lambda *a, **k: next(prompts))

        # _pick_from_list must NOT be called in the failure path.
        monkeypatch.setattr(
            cli,
            "_pick_from_list",
            lambda *a, **k: pytest.fail(
                "picker should not be invoked when discovery fails"
            ),
        )

        def boom(**kwargs):
            raise BedrockDiscoveryError("AccessDeniedException")

        monkeypatch.setattr(cli, "discover_bedrock_models", boom)

        result = cli._prompt_bedrock_config(preset={"models": [], "default_cheap": ""}, slot="cheap")
        assert result["model"] == "anthropic.claude-3-5-haiku-20241022-v1:0"

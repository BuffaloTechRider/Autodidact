"""Tests for Bedrock auth modes (default / IAM user / API key).

Bedrock supports three ways to authenticate. The CLI wizard prompts for
which one, writes it into the config under `cloud.bedrock`, and the Agent
constructor translates it into LLMConfig fields. _get_bedrock_client()
then constructs the boto3 client differently per mode.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from autodidact.agent import _apply_bedrock_auth
from autodidact.llm_client import LLMClient, LLMConfig


# ── Config translation ──────────────────────────────────────────


class TestApplyBedrockAuthHelper:
    """_apply_bedrock_auth translates the YAML dict shape into LLMConfig kwargs."""

    def test_default_mode_minimal(self):
        kwargs: dict = {"provider": "bedrock", "model": "claude"}
        _apply_bedrock_auth(kwargs, {"auth_mode": "default", "region": "us-east-1"})
        assert kwargs["bedrock_auth_mode"] == "default"
        assert kwargs["region"] == "us-east-1"
        # No access keys in default mode.
        assert "bedrock_access_key_id" not in kwargs
        assert "bedrock_api_key" not in kwargs

    def test_iam_user_mode_carries_access_keys(self):
        kwargs: dict = {"provider": "bedrock", "model": "claude"}
        _apply_bedrock_auth(kwargs, {
            "auth_mode": "iam_user",
            "access_key_id": "AKIAEXAMPLE",
            "secret_access_key": "secret123",
            "region": "us-west-2",
        })
        assert kwargs["bedrock_auth_mode"] == "iam_user"
        assert kwargs["bedrock_access_key_id"] == "AKIAEXAMPLE"
        assert kwargs["bedrock_secret_access_key"] == "secret123"
        # Session token optional.
        assert "bedrock_session_token" not in kwargs

    def test_iam_user_mode_with_session_token(self):
        kwargs: dict = {"provider": "bedrock", "model": "claude"}
        _apply_bedrock_auth(kwargs, {
            "auth_mode": "iam_user",
            "access_key_id": "AKIA",
            "secret_access_key": "s",
            "session_token": "tok",
            "region": "us-west-2",
        })
        assert kwargs["bedrock_session_token"] == "tok"

    def test_api_key_mode_carries_key(self):
        kwargs: dict = {"provider": "bedrock", "model": "claude"}
        _apply_bedrock_auth(kwargs, {
            "auth_mode": "api_key",
            "api_key": "bedrock-abc123",
            "region": "us-west-2",
        })
        assert kwargs["bedrock_auth_mode"] == "api_key"
        assert kwargs["bedrock_api_key"] == "bedrock-abc123"


# ── LLMClient bedrock client construction ───────────────────────


class TestBedrockClientConstruction:
    """_get_bedrock_client branches on auth mode to call boto3.client() correctly."""

    def _build_client(self, **overrides) -> tuple[LLMClient, MagicMock]:
        """Build an LLMClient in bedrock mode, patching boto3.client."""
        config_kwargs = {
            "provider": "bedrock",
            "model": "anthropic.claude-haiku-4-20250514-v1:0",
            **overrides,
        }
        client = LLMClient(LLMConfig(**config_kwargs))
        boto3_mock = MagicMock()
        with patch.dict("sys.modules", {"boto3": boto3_mock, "botocore.config": MagicMock()}):
            client._get_bedrock_client()
        return client, boto3_mock

    def test_default_mode_no_explicit_credentials(self):
        """Default mode: boto3.client() gets no credentials — chain picks them up."""
        _, boto3_mock = self._build_client(bedrock_auth_mode="default")
        call_kwargs = boto3_mock.client.call_args.kwargs
        assert call_kwargs["service_name"] == "bedrock-runtime"
        assert call_kwargs["region_name"] == "us-west-2"
        assert "aws_access_key_id" not in call_kwargs
        assert "aws_secret_access_key" not in call_kwargs

    def test_iam_user_mode_passes_credentials(self):
        """IAM user mode: access_key_id/secret_access_key passed to boto3.client()."""
        _, boto3_mock = self._build_client(
            bedrock_auth_mode="iam_user",
            bedrock_access_key_id="AKIA",
            bedrock_secret_access_key="secret",
        )
        call_kwargs = boto3_mock.client.call_args.kwargs
        assert call_kwargs["aws_access_key_id"] == "AKIA"
        assert call_kwargs["aws_secret_access_key"] == "secret"

    def test_iam_user_mode_missing_creds_raises(self):
        """IAM user mode without creds in the config is a clear error, not silent."""
        from autodidact.llm_client import LLMClientError

        client = LLMClient(LLMConfig(
            provider="bedrock",
            model="x",
            bedrock_auth_mode="iam_user",
            # Both keys omitted — should raise.
        ))
        with pytest.raises(LLMClientError, match="iam_user"):
            with patch.dict("sys.modules", {"boto3": MagicMock(), "botocore.config": MagicMock()}):
                client._get_bedrock_client()

    def test_api_key_mode_sets_bearer_token_env(self):
        """API key mode: AWS_BEARER_TOKEN_BEDROCK gets set from config."""
        original = os.environ.pop("AWS_BEARER_TOKEN_BEDROCK", None)
        try:
            self._build_client(
                bedrock_auth_mode="api_key",
                bedrock_api_key="bedrock-xyz",
            )
            assert os.environ.get("AWS_BEARER_TOKEN_BEDROCK") == "bedrock-xyz"
        finally:
            if original is not None:
                os.environ["AWS_BEARER_TOKEN_BEDROCK"] = original
            else:
                os.environ.pop("AWS_BEARER_TOKEN_BEDROCK", None)

    def test_api_key_mode_missing_key_raises(self):
        """API key mode without a key in the config is a clear error."""
        from autodidact.llm_client import LLMClientError

        client = LLMClient(LLMConfig(
            provider="bedrock",
            model="x",
            bedrock_auth_mode="api_key",
            # api_key omitted — should raise.
        ))
        with pytest.raises(LLMClientError, match="api_key"):
            with patch.dict("sys.modules", {"boto3": MagicMock(), "botocore.config": MagicMock()}):
                client._get_bedrock_client()

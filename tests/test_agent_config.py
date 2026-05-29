"""Tests for AgentConfig — the typed YAML→Agent translation.

The old flow had two layers of mechanical translation:
  YAML dict → cli._agent_from_config kwargs (95 lines)
        → Agent.__init__ kwarg→LLMConfig translation (80 lines)

Now:
  YAML → AgentConfig (Pydantic, validated)
        → AgentConfig.build_agent() → Agent

Behaviour change worth knowing: YAML loading is now STRICT. Missing
required fields, ranges out of [0, 1], API keys without explicit config —
all fail at load time with helpful errors. Programmatic
``Agent(local_model=..., cloud_model=..., ...)`` construction stays loose
(legacy shim). This file covers both paths.
"""

from __future__ import annotations

import os
from pathlib import Path
from textwrap import dedent
from unittest.mock import patch

import pytest
import yaml


# ── Loading from YAML ───────────────────────────────────────────


class TestFromYamlHappyPath:
    """The wizard writes one of three valid shapes; AgentConfig accepts all three."""

    def _write(self, tmp_path: Path, body: str) -> Path:
        p = tmp_path / "config.yaml"
        p.write_text(dedent(body))
        return p

    def test_local_cloud_mode(self, tmp_path):
        from autodidact.config import AgentConfig

        p = self._write(tmp_path, """
            local:
              model: qwen3:8b
              embedding_model: qllama/bge-large-en-v1.5
            cloud:
              provider: openai
              model: gpt-4o-mini
              api_key: sk-test-123
            memory:
              path: ~/.autodidact/memory.db
            routing:
              confidence_threshold: 0.7
        """)
        cfg = AgentConfig.from_yaml(p)
        assert cfg.local.model == "qwen3:8b"
        assert cfg.local.embedding_model == "qllama/bge-large-en-v1.5"
        assert cfg.cloud is not None
        assert cfg.cloud.provider == "openai"
        assert cfg.cloud.model == "gpt-4o-mini"
        assert cfg.cloud.api_key == "sk-test-123"
        assert cfg.routing.confidence_threshold == pytest.approx(0.7)

    def test_local_only_mode_with_no_cloud_section(self, tmp_path):
        from autodidact.config import AgentConfig

        p = self._write(tmp_path, """
            local:
              model: qwen3:8b
        """)
        cfg = AgentConfig.from_yaml(p)
        assert cfg.local.model == "qwen3:8b"
        assert cfg.cloud is None  # no cloud section ⇒ local-only

    def test_local_only_mode_with_empty_cloud_section(self, tmp_path):
        """Empty cloud section is treated as local-only, not as a config error.

        Per the design decision: ``cloud: {}`` → local-only.
        """
        from autodidact.config import AgentConfig

        p = self._write(tmp_path, """
            local:
              model: qwen3:8b
            cloud: {}
        """)
        cfg = AgentConfig.from_yaml(p)
        assert cfg.cloud is None

    def test_cloud_cloud_mode(self, tmp_path):
        """Cheap cloud in the local slot, expensive cloud in the cloud slot."""
        from autodidact.config import AgentConfig

        p = self._write(tmp_path, """
            local:
              provider: openai
              model: gpt-4o-mini
              api_key: sk-cheap
              embedding_model: text-embedding-3-small
            cloud:
              provider: openai
              model: gpt-4o
              api_key: sk-expensive
        """)
        cfg = AgentConfig.from_yaml(p)
        assert cfg.local.provider == "openai"
        assert cfg.local.model == "gpt-4o-mini"
        assert cfg.local.api_key == "sk-cheap"
        assert cfg.cloud.provider == "openai"
        assert cfg.cloud.model == "gpt-4o"

    def test_bedrock_with_default_auth(self, tmp_path):
        from autodidact.config import AgentConfig

        p = self._write(tmp_path, """
            local:
              model: qwen3:8b
            cloud:
              provider: bedrock
              model: us.anthropic.claude-sonnet-4-5-20250929-v1:0
              bedrock:
                auth_mode: default
                region: us-west-2
        """)
        cfg = AgentConfig.from_yaml(p)
        assert cfg.cloud.bedrock is not None
        assert cfg.cloud.bedrock.auth_mode == "default"
        assert cfg.cloud.bedrock.region == "us-west-2"


# ── Tightened validation ────────────────────────────────────────


class TestStrictValidation:
    """Invalid configs fail at load with human-readable errors."""

    def _write(self, tmp_path: Path, body: str) -> Path:
        p = tmp_path / "config.yaml"
        p.write_text(dedent(body))
        return p

    def test_missing_local_model_is_error(self, tmp_path):
        from autodidact.config import AgentConfig, ConfigError

        p = self._write(tmp_path, """
            local: {}
            cloud:
              provider: openai
              model: gpt-4o
              api_key: sk-x
        """)
        with pytest.raises(ConfigError) as exc:
            AgentConfig.from_yaml(p)
        assert "local.model" in str(exc.value)

    def test_missing_cloud_provider_is_error(self, tmp_path):
        """Today this silently defaulted to 'openai' — footgun for Bedrock users."""
        from autodidact.config import AgentConfig, ConfigError

        p = self._write(tmp_path, """
            local:
              model: qwen3:8b
            cloud:
              model: claude-haiku-4
              api_key: sk-x
        """)
        with pytest.raises(ConfigError) as exc:
            AgentConfig.from_yaml(p)
        assert "cloud.provider" in str(exc.value)

    def test_missing_cloud_model_is_error(self, tmp_path):
        from autodidact.config import AgentConfig, ConfigError

        p = self._write(tmp_path, """
            local:
              model: qwen3:8b
            cloud:
              provider: openai
              api_key: sk-x
        """)
        with pytest.raises(ConfigError) as exc:
            AgentConfig.from_yaml(p)
        assert "cloud.model" in str(exc.value)

    def test_openai_cloud_without_api_key_or_env_is_error(self, tmp_path, monkeypatch):
        """OpenAI-compat cloud without explicit api_key AND no matching env var → fail load."""
        from autodidact.config import AgentConfig, ConfigError

        # Make sure no OPENAI_API_KEY in env so the test is deterministic.
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        p = self._write(tmp_path, """
            local:
              model: qwen3:8b
            cloud:
              provider: openai
              model: gpt-4o
        """)
        with pytest.raises(ConfigError) as exc:
            AgentConfig.from_yaml(p)
        assert "api_key" in str(exc.value)

    def test_openai_cloud_with_env_var_set_is_accepted(self, tmp_path, monkeypatch):
        """If api_key is omitted but the matching env var is set, accept."""
        from autodidact.config import AgentConfig

        monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")

        p = self._write(tmp_path, """
            local:
              model: qwen3:8b
            cloud:
              provider: openai
              model: gpt-4o
        """)
        cfg = AgentConfig.from_yaml(p)
        assert cfg.cloud.provider == "openai"
        # api_key may be None when env var is the source of truth — config
        # loader validates the env var is set, doesn't copy the value.
        # The downstream llm_client picks it up via api_key_env.

    def test_openai_cloud_with_explicit_api_key_env_is_accepted(self, tmp_path, monkeypatch):
        from autodidact.config import AgentConfig

        monkeypatch.setenv("CUSTOM_KEY", "sk-custom")
        p = self._write(tmp_path, """
            local:
              model: qwen3:8b
            cloud:
              provider: openai
              model: gpt-4o
              api_key_env: CUSTOM_KEY
        """)
        cfg = AgentConfig.from_yaml(p)
        assert cfg.cloud.api_key_env == "CUSTOM_KEY"

    def test_bedrock_iam_user_without_secret_is_error(self, tmp_path):
        from autodidact.config import AgentConfig, ConfigError

        p = self._write(tmp_path, """
            local:
              model: qwen3:8b
            cloud:
              provider: bedrock
              model: us.anthropic.claude-sonnet-4-5-20250929-v1:0
              bedrock:
                auth_mode: iam_user
                access_key_id: AKIA-test
        """)
        with pytest.raises(ConfigError) as exc:
            AgentConfig.from_yaml(p)
        assert "secret_access_key" in str(exc.value)

    def test_bedrock_api_key_mode_without_key_is_error(self, tmp_path):
        from autodidact.config import AgentConfig, ConfigError

        p = self._write(tmp_path, """
            local:
              model: qwen3:8b
            cloud:
              provider: bedrock
              model: us.anthropic.claude-sonnet-4-5-20250929-v1:0
              bedrock:
                auth_mode: api_key
        """)
        with pytest.raises(ConfigError) as exc:
            AgentConfig.from_yaml(p)
        assert "api_key" in str(exc.value)

    def test_confidence_threshold_out_of_range(self, tmp_path):
        from autodidact.config import AgentConfig, ConfigError

        p = self._write(tmp_path, """
            local:
              model: qwen3:8b
            routing:
              confidence_threshold: 1.5
        """)
        with pytest.raises(ConfigError) as exc:
            AgentConfig.from_yaml(p)
        assert "confidence_threshold" in str(exc.value)

    def test_gsa_threshold_out_of_range(self, tmp_path):
        from autodidact.config import AgentConfig, ConfigError

        p = self._write(tmp_path, """
            local:
              model: qwen3:8b
            gsa:
              threshold: -0.1
        """)
        with pytest.raises(ConfigError) as exc:
            AgentConfig.from_yaml(p)
        assert "threshold" in str(exc.value)

    def test_unknown_fields_are_ignored(self, tmp_path):
        """Forward-compat: configs from a newer wizard with extra fields don't break."""
        from autodidact.config import AgentConfig

        p = self._write(tmp_path, """
            local:
              model: qwen3:8b
              future_field: 42
            cloud:
              provider: openai
              model: gpt-4o
              api_key: sk-x
              another_future_field: hello
            unknown_top_level: ignored
        """)
        cfg = AgentConfig.from_yaml(p)
        assert cfg.local.model == "qwen3:8b"


# ── Env var overrides ───────────────────────────────────────────


class TestEnvOverrides:
    """AUTODIDACT_MODEL and OPENAI_API_KEY override config values, like today."""

    def _write(self, tmp_path: Path, body: str) -> Path:
        p = tmp_path / "config.yaml"
        p.write_text(dedent(body))
        return p

    def test_autodidact_model_overrides_local(self, tmp_path, monkeypatch):
        from autodidact.config import AgentConfig

        monkeypatch.setenv("AUTODIDACT_MODEL", "qwen3:14b")
        p = self._write(tmp_path, """
            local:
              model: qwen3:8b
        """)
        cfg = AgentConfig.from_yaml(p)
        assert cfg.local.model == "qwen3:14b"

    def test_openai_key_env_var_supplies_cloud_key(self, tmp_path, monkeypatch):
        from autodidact.config import AgentConfig

        monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")
        p = self._write(tmp_path, """
            local:
              model: qwen3:8b
            cloud:
              provider: openai
              model: gpt-4o
        """)
        # No api_key in YAML — env var fills the gap.
        cfg = AgentConfig.from_yaml(p)
        # When env var supplied the key, api_key_env is set so downstream
        # LLMClient knows where to read it from.
        assert cfg.cloud.api_key_env == "OPENAI_API_KEY"


# ── build_agent() — translation to LLMConfig ────────────────────


class TestBuildAgent:
    """AgentConfig.build_agent() returns a fully wired Agent.

    The big behavioural coverage of Agent.query() lives in test_agent.py.
    These tests focus on the construction translation: do the right
    LLMClients get built, with the right LLMConfigs?
    """

    def test_local_cloud_build(self, tmp_path):
        from autodidact.config import AgentConfig
        from autodidact.llm import OllamaBackend, OpenAICompatBackend

        cfg = AgentConfig.model_validate({
            "local": {"model": "qwen3:8b", "embedding_model": "qllama/bge-large-en-v1.5"},
            "cloud": {"provider": "openai", "model": "gpt-4o", "api_key": "sk-x"},
            "memory": {"path": str(tmp_path / "mem.db")},
        })
        agent = cfg.build_agent()
        assert agent._local_client is not None
        assert isinstance(agent._local_client._backend, OllamaBackend)
        assert agent._cloud_client is not None
        assert isinstance(agent._cloud_client._backend, OpenAICompatBackend)
        assert agent._embed_client is agent._local_client

    def test_local_only_build(self, tmp_path):
        from autodidact.config import AgentConfig

        cfg = AgentConfig.model_validate({
            "local": {"model": "qwen3:8b"},
            "memory": {"path": str(tmp_path / "mem.db")},
        })
        agent = cfg.build_agent()
        assert agent._local_client is not None
        assert agent._cloud_client is None

    def test_cloud_cloud_build(self, tmp_path):
        from autodidact.config import AgentConfig
        from autodidact.llm import OpenAICompatBackend

        cfg = AgentConfig.model_validate({
            "local": {
                "provider": "openai",
                "model": "gpt-4o-mini",
                "api_key": "sk-cheap",
                "embedding_model": "text-embedding-3-small",
            },
            "cloud": {"provider": "openai", "model": "gpt-4o", "api_key": "sk-expensive"},
            "memory": {"path": str(tmp_path / "mem.db")},
        })
        agent = cfg.build_agent()
        assert isinstance(agent._local_client._backend, OpenAICompatBackend)
        assert isinstance(agent._cloud_client._backend, OpenAICompatBackend)

    def test_bedrock_auth_default_mode_build(self, tmp_path):
        from autodidact.config import AgentConfig
        from autodidact.llm import BedrockBackend

        cfg = AgentConfig.model_validate({
            "local": {"model": "qwen3:8b"},
            "cloud": {
                "provider": "bedrock",
                "model": "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
                "bedrock": {"auth_mode": "default", "region": "us-west-2"},
            },
            "memory": {"path": str(tmp_path / "mem.db")},
        })
        agent = cfg.build_agent()
        assert isinstance(agent._cloud_client._backend, BedrockBackend)
        assert agent._cloud_client.config.bedrock_auth_mode == "default"
        assert agent._cloud_client.config.region == "us-west-2"

    def test_bedrock_iam_user_propagates_to_llmconfig(self, tmp_path):
        from autodidact.config import AgentConfig

        cfg = AgentConfig.model_validate({
            "local": {"model": "qwen3:8b"},
            "cloud": {
                "provider": "bedrock",
                "model": "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
                "bedrock": {
                    "auth_mode": "iam_user",
                    "region": "us-east-1",
                    "access_key_id": "AKIA-x",
                    "secret_access_key": "secret-x",
                    "session_token": "token-x",
                },
            },
            "memory": {"path": str(tmp_path / "mem.db")},
        })
        agent = cfg.build_agent()
        llm_cfg = agent._cloud_client.config
        assert llm_cfg.bedrock_auth_mode == "iam_user"
        assert llm_cfg.bedrock_access_key_id == "AKIA-x"
        assert llm_cfg.bedrock_secret_access_key == "secret-x"
        assert llm_cfg.bedrock_session_token == "token-x"
        assert llm_cfg.region == "us-east-1"

    def test_routing_threshold_propagates(self, tmp_path):
        from autodidact.config import AgentConfig

        cfg = AgentConfig.model_validate({
            "local": {"model": "qwen3:8b"},
            "memory": {"path": str(tmp_path / "mem.db")},
            "routing": {"confidence_threshold": 0.42},
            "gsa": {"enabled": False, "threshold": 0.65},
        })
        agent = cfg.build_agent()
        assert agent.confidence_threshold == pytest.approx(0.42)
        assert agent.gsa_enabled is False
        assert agent.gsa_threshold == pytest.approx(0.65)

    def test_embedding_model_default_when_omitted(self, tmp_path):
        from autodidact.config import AgentConfig

        cfg = AgentConfig.model_validate({
            "local": {"model": "qwen3:8b"},
            "memory": {"path": str(tmp_path / "mem.db")},
        })
        agent = cfg.build_agent()
        # qllama/ prefix preserved (it's the model namespace, not a provider tag).
        assert agent._local_client.config.embedding_model == "qllama/bge-large-en-v1.5"


# ── Legacy kwarg-style Agent construction (back-compat) ─────────


class TestLegacyKwargConstruction:
    """Agent(local_model=..., cloud_model=...) keeps working unchanged.

    The kwarg path is a shim that builds an AgentConfig internally. Strictness
    rules apply only to YAML loading; programmatic callers have full control
    and are trusted (per the decision in this PR's plan).
    """

    def test_local_cloud_kwargs(self, tmp_path):
        from autodidact.agent import Agent

        agent = Agent(
            local_model="qwen3:8b",
            cloud_model="gpt-4o",
            cloud_provider="openai",
            cloud_api_key_env="OPENAI_API_KEY",
            db_path=str(tmp_path / "mem.db"),
        )
        assert agent._local_client is not None
        assert agent._cloud_client is not None

    def test_no_models_kwargs_still_works(self, tmp_path):
        """Programmatic construction with no models — used by some test fixtures."""
        from autodidact.agent import Agent

        agent = Agent(db_path=str(tmp_path / "mem.db"))
        assert agent._local_client is None
        assert agent._cloud_client is None

    def test_threshold_kwargs_propagate(self, tmp_path):
        from autodidact.agent import Agent

        agent = Agent(
            db_path=str(tmp_path / "mem.db"),
            confidence_threshold=0.5,
            gsa_threshold=0.6,
            gsa_enabled=False,
            staleness_days=14,
        )
        assert agent.confidence_threshold == pytest.approx(0.5)
        assert agent.gsa_threshold == pytest.approx(0.6)
        assert agent.gsa_enabled is False
        assert agent.staleness_days == 14

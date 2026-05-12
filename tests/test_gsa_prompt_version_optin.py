"""Tests for opting into GSA prompt v4 at the Agent and config levels.

Three opt-in paths:
  1. Python API: Agent(gsa_prompt_version="v4")
  2. Config YAML: routing.gsa_prompt_version or gsa.prompt_version
  3. (future) env var — intentionally not implemented here

Defaults remain v3 everywhere. v4 must be explicitly asked for.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import yaml

from autodidact.agent import Agent
from autodidact.cli import _agent_from_config
from autodidact.signals.grounded_self_assessment import (
    PROMPT_VERSION,
    PROMPT_VERSION_V4,
)


# ── Python API: Agent(gsa_prompt_version=...) ─────────────────────


class TestAgentGsaPromptVersionKwarg:
    """Agent accepts gsa_prompt_version and passes it to SelfAssessment."""

    def test_default_uses_v3(self, tmp_path):
        agent = Agent(
            local_model="ollama/qwen2.5:7b",
            db_path=f"{tmp_path}/t.db",
        )
        assert agent.gsa_prompt_version == "v3"

    def test_v4_opt_in(self, tmp_path):
        agent = Agent(
            local_model="ollama/qwen2.5:7b",
            db_path=f"{tmp_path}/t.db",
            gsa_prompt_version="v4",
        )
        assert agent.gsa_prompt_version == "v4"

    def test_invalid_version_raises(self, tmp_path):
        with pytest.raises(ValueError):
            Agent(
                local_model="ollama/qwen2.5:7b",
                db_path=f"{tmp_path}/t.db",
                gsa_prompt_version="v99",
            )

    def test_gsa_instance_uses_configured_version(self, tmp_path):
        """When the agent lazily creates its SelfAssessment, it must pass the version."""
        agent = Agent(
            local_model="ollama/qwen2.5:7b",
            db_path=f"{tmp_path}/t.db",
            gsa_prompt_version="v4",
        )
        # Trigger lazy construction by calling the getter.
        gsa = agent._get_gsa()
        assert gsa.prompt_version == PROMPT_VERSION_V4

    def test_default_gsa_instance_uses_v3(self, tmp_path):
        agent = Agent(
            local_model="ollama/qwen2.5:7b",
            db_path=f"{tmp_path}/t.db",
        )
        gsa = agent._get_gsa()
        assert gsa.prompt_version == PROMPT_VERSION


# ── Config YAML: routing.gsa_prompt_version ────────────────────────


class TestConfigGsaPromptVersion:
    """_agent_from_config reads routing.gsa_prompt_version from YAML."""

    def _write_config(self, tmp_path: Path, routing: dict) -> dict:
        cfg = {
            "local": {"model": "qwen2.5:7b"},
            "memory": {"path": f"{tmp_path}/m.db"},
            "routing": routing,
        }
        return cfg

    def test_config_default_is_v3(self, tmp_path):
        cfg = self._write_config(tmp_path, {"confidence_threshold": 0.7})
        agent = _agent_from_config(cfg)
        assert agent.gsa_prompt_version == "v3"

    def test_config_opt_in_v4(self, tmp_path):
        cfg = self._write_config(tmp_path, {
            "confidence_threshold": 0.7,
            "gsa_prompt_version": "v4",
        })
        agent = _agent_from_config(cfg)
        assert agent.gsa_prompt_version == "v4"

    def test_config_invalid_version_raises(self, tmp_path):
        cfg = self._write_config(tmp_path, {
            "confidence_threshold": 0.7,
            "gsa_prompt_version": "v42",
        })
        with pytest.raises(ValueError):
            _agent_from_config(cfg)


# ── Smoke test: real YAML round-trip ──────────────────────────────


class TestConfigYamlRoundtrip:
    """A user edits their config.yaml and re-runs chat — opt-in persists."""

    def test_yaml_file_v4_opts_in(self, tmp_path):
        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml.safe_dump({
            "local": {"model": "qwen2.5:7b"},
            "memory": {"path": f"{tmp_path}/m.db"},
            "routing": {
                "confidence_threshold": 0.7,
                "gsa_prompt_version": "v4",
            },
        }))
        loaded = yaml.safe_load(config_path.read_text())
        agent = _agent_from_config(loaded)
        assert agent.gsa_prompt_version == "v4"

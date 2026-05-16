"""Tests for the zero-friction `autodidact init` flow (Task 5.2, R8).

TDD: tests written first, then implementation.

The init command uses the setup wizard to:
- Detect Ollama, offer install if missing, auto-pull models
- Present three setup modes: local+cloud, cloud+cloud, local-only
- Use cloud provider presets for pre-filled URLs and models
- Write a YAML config
- Run a smoke test at the end

We also test the Agent constructor's cloud-to-cloud support (the
'local' slot holding an OpenAI-compatible cheap model) and the CLI
config loader's handling of that mode.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml
from typer.testing import CliRunner

from autodidact.cli import app

runner = CliRunner()


# ── Helpers ──────────────────────────────────────────────────────


def _read_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


# ── Init mode selection ──────────────────────────────────────────


class TestInitModeSelection:
    """R8 AC1: three modes — local+cloud, cloud+cloud, local-only."""

    @patch("autodidact.cli._run_smoke_test")
    @patch("autodidact.cli.detect_ollama")
    @patch("autodidact.cli.is_model_available", return_value=True)
    def test_local_cloud_mode_produces_local_and_cloud_sections(
        self, mock_model, mock_detect, mock_smoke, tmp_path
    ):
        """Mode 1 (local+cloud) writes both 'local' and 'cloud' sections."""
        from autodidact.setup_wizard import OllamaStatus
        mock_detect.return_value = OllamaStatus(installed=True, path="/usr/local/bin/ollama")
        cfg = tmp_path / "config.yaml"

        # Inputs: mode=1, local model default, cloud provider openai, api key, model default, db default
        inputs = "1\n\nopenai\nsk-test123\n\n\n"
        result = runner.invoke(app, ["init", "--config-path", str(cfg)], input=inputs)

        assert result.exit_code == 0, result.output
        config = _read_config(cfg)
        assert "local" in config
        assert "cloud" in config
        assert config["cloud"]["provider"] == "openai"

    @patch("autodidact.cli._run_smoke_test")
    def test_cloud_cloud_mode_produces_cloud_local_slot(self, mock_smoke, tmp_path):
        """Mode 2 (cloud+cloud): the 'local' slot holds a cloud provider, not Ollama."""
        cfg = tmp_path / "config.yaml"

        # Inputs: mode=2, cheap provider openai, cheap key, cheap model default,
        # expensive provider openai, expensive key, expensive model default, db default
        inputs = "2\nopenai\nsk-cheap\n\nopenai\nsk-expensive\n\n\n"
        result = runner.invoke(app, ["init", "--config-path", str(cfg)], input=inputs)

        assert result.exit_code == 0, result.output
        config = _read_config(cfg)
        assert config["local"]["provider"] == "openai"
        assert "base_url" in config["local"]  # cloud-to-cloud has a base_url
        assert config["cloud"]["provider"] == "openai"

    @patch("autodidact.cli._run_smoke_test")
    @patch("autodidact.cli.detect_ollama")
    @patch("autodidact.cli.is_model_available", return_value=True)
    def test_local_only_mode_omits_cloud_section(
        self, mock_model, mock_detect, mock_smoke, tmp_path
    ):
        """Mode 3 (local only): no 'cloud' section in config."""
        from autodidact.setup_wizard import OllamaStatus
        mock_detect.return_value = OllamaStatus(installed=True, path="/usr/local/bin/ollama")
        cfg = tmp_path / "config.yaml"

        # Inputs: mode=3, local model default, db default
        inputs = "5\n\n\n"
        result = runner.invoke(app, ["init", "--config-path", str(cfg)], input=inputs)

        assert result.exit_code == 0, result.output
        config = _read_config(cfg)
        assert "local" in config
        assert "cloud" not in config


# ── Ollama auto-detection ────────────────────────────────────────


class TestOllamaAutoDetection:
    """R8 AC2: detect Ollama; offer install if missing; auto-pull models."""

    @patch("autodidact.cli._run_smoke_test")
    @patch("autodidact.cli.detect_ollama")
    @patch("autodidact.cli.is_model_available", return_value=True)
    def test_detects_installed_ollama(
        self, mock_model, mock_detect, mock_smoke, tmp_path
    ):
        """When Ollama is installed, init proceeds without prompting to install."""
        from autodidact.setup_wizard import OllamaStatus
        mock_detect.return_value = OllamaStatus(installed=True, path="/usr/local/bin/ollama")
        cfg = tmp_path / "config.yaml"

        inputs = "5\n\n\n"  # local-only mode, defaults
        result = runner.invoke(app, ["init", "--config-path", str(cfg)], input=inputs)

        assert result.exit_code == 0
        # Should NOT prompt about install command.
        assert "brew install" not in result.output
        assert "curl -fsSL" not in result.output

    @patch("autodidact.cli._run_smoke_test")
    @patch("autodidact.cli.detect_ollama")
    @patch("autodidact.cli.get_ollama_install_command", return_value="brew install ollama")
    def test_missing_ollama_shows_install_command(
        self, mock_cmd, mock_detect, mock_smoke, tmp_path
    ):
        """When Ollama is missing, init shows the install command."""
        from autodidact.setup_wizard import OllamaStatus
        mock_detect.return_value = OllamaStatus(installed=False, path=None)
        cfg = tmp_path / "config.yaml"

        # Mode=4 (local-only), then user says 'no' to continuing without ollama.
        inputs = "5\nn\n"
        result = runner.invoke(app, ["init", "--config-path", str(cfg)], input=inputs)

        # Should show the install command.
        assert "brew install ollama" in result.output

    @patch("autodidact.cli._run_smoke_test")
    @patch("autodidact.cli.detect_ollama")
    @patch("autodidact.cli.verify_model_loadable", return_value=True)
    @patch("autodidact.cli.is_model_available", return_value=False)
    @patch("autodidact.cli.pull_ollama_model", return_value=True)
    def test_missing_model_gets_pulled(
        self, mock_pull, mock_available, mock_verify, mock_detect, mock_smoke, tmp_path
    ):
        """When the selected model isn't pulled, init pulls it automatically.

        is_model_available=False triggers the pull; verify_model_loadable=True
        simulates a successful download (the model appears in ollama list after).
        """
        from autodidact.setup_wizard import OllamaStatus
        mock_detect.return_value = OllamaStatus(installed=True, path="/usr/local/bin/ollama")
        cfg = tmp_path / "config.yaml"

        inputs = "5\n\n\n"  # local-only mode, defaults
        result = runner.invoke(app, ["init", "--config-path", str(cfg)], input=inputs)

        assert result.exit_code == 0
        # pull_ollama_model should have been called (for chat model, embedding model, or both).
        assert mock_pull.call_count >= 1

    @patch("autodidact.cli._run_smoke_test")
    @patch("autodidact.cli.detect_ollama")
    @patch("autodidact.cli.verify_model_loadable", return_value=False)
    @patch("autodidact.cli.is_model_available", return_value=False)
    @patch("autodidact.cli.pull_ollama_model", return_value=True)
    def test_cloud_only_pull_fails_verification_and_exits(
        self, mock_pull, mock_available, mock_verify, mock_detect, mock_smoke, tmp_path
    ):
        """Ollama 'cloud-only' tags 'pull' successfully but aren't loadable.

        The wizard must detect this (verify_model_loadable=False after pull)
        and stop before writing a broken config.
        """
        from autodidact.setup_wizard import OllamaStatus
        mock_detect.return_value = OllamaStatus(installed=True, path="/usr/local/bin/ollama")
        cfg = tmp_path / "config.yaml"

        inputs = "5\n\n\n"  # local-only mode, defaults
        result = runner.invoke(app, ["init", "--config-path", str(cfg)], input=inputs)

        assert result.exit_code != 0, "Wizard must fail when pulled model isn't loadable"
        assert not cfg.exists(), "Broken config must not be written"
        # Error message should be specific about cloud-only tags.
        assert ("cloud-only" in result.output.lower()
                or "cannot be loaded" in result.output.lower())


# ── Cloud provider presets ───────────────────────────────────────


class TestCloudPresets:
    """R8 AC3: cloud provider presets (OpenAI, OpenRouter, DeepSeek, Bedrock)."""

    @patch("autodidact.cli._run_smoke_test")
    def test_openrouter_preset_fills_base_url(self, mock_smoke, tmp_path):
        """Choosing 'openrouter' in cloud+cloud mode fills in the right base_url."""
        cfg = tmp_path / "config.yaml"
        # Mode 2, cheap=openrouter, key, default model, expensive=openrouter, key, default model, db default
        inputs = "2\nopenrouter\nor-test\n\nopenrouter\nor-test\n\n\n"
        result = runner.invoke(app, ["init", "--config-path", str(cfg)], input=inputs)

        assert result.exit_code == 0
        config = _read_config(cfg)
        assert "openrouter" in config["local"]["base_url"]

    @patch("autodidact.cli._run_smoke_test")
    def test_deepseek_preset_fills_models(self, mock_smoke, tmp_path):
        """DeepSeek preset provides models in the 'local' slot."""
        cfg = tmp_path / "config.yaml"
        inputs = "2\ndeepseek\nds-test\n\ndeepseek\nds-test\n\n\n"
        result = runner.invoke(app, ["init", "--config-path", str(cfg)], input=inputs)

        assert result.exit_code == 0
        config = _read_config(cfg)
        assert "deepseek" in config["local"]["base_url"]


# ── Smoke test ────────────────────────────────────────────────────


class TestSmokeTest:
    """R8 AC7: smoke test on init."""

    @patch("autodidact.cli._run_smoke_test")
    @patch("autodidact.cli.detect_ollama")
    @patch("autodidact.cli.is_model_available", return_value=True)
    def test_smoke_test_is_called(
        self, mock_model, mock_detect, mock_smoke, tmp_path
    ):
        """After writing config, init runs a smoke test."""
        from autodidact.setup_wizard import OllamaStatus
        mock_detect.return_value = OllamaStatus(installed=True, path="/usr/local/bin/ollama")
        cfg = tmp_path / "config.yaml"

        inputs = "5\n\n\n"
        result = runner.invoke(app, ["init", "--config-path", str(cfg)], input=inputs)

        assert result.exit_code == 0
        mock_smoke.assert_called_once()


# ── Agent cloud-to-cloud support ─────────────────────────────────


class TestAgentCloudToCloud:
    """The Agent accepts a cloud-backed 'local' slot for cloud-to-cloud routing."""

    def test_agent_accepts_local_base_url_and_api_key_env(self):
        """Agent(local_model='openai/gpt-4o-mini', local_base_url=..., local_api_key_env=...)
        builds an OpenAI-compat client for the 'local' slot.
        """
        from autodidact.agent import Agent

        # Don't actually call the API — just verify the client is constructed with
        # the right config.
        with patch("autodidact.agent.LLMClient") as MockClient:
            mock_instance = MagicMock()
            mock_instance.embed.return_value = [0.0] * 32
            MockClient.return_value = mock_instance

            agent = Agent(
                local_model="openai/gpt-4o-mini",
                local_base_url="https://api.openai.com/v1",
                local_api_key_env="OPENAI_API_KEY",
                cloud_model="openai/gpt-4o",
                db_path=":memory:",
            )

            # Find the LLMConfig used to build the "local" client (first construction).
            calls = MockClient.call_args_list
            # There should be at least one call for local and one for cloud.
            assert len(calls) >= 1
            # The first call's LLMConfig should have base_url set for openai provider.
            first_config = calls[0][0][0]
            assert first_config.provider == "openai"
            assert first_config.model == "gpt-4o-mini"
            assert first_config.base_url == "https://api.openai.com/v1"


# ── Config loader cloud-to-cloud ─────────────────────────────────


class TestConfigLoaderCloudToCloud:
    """The CLI config loader builds a cloud-to-cloud Agent correctly."""

    def test_cloud_cloud_config_creates_agent_with_cloud_local_slot(self, tmp_path):
        """Loading a cloud+cloud config produces an Agent whose 'local' is a cloud client."""
        from autodidact.cli import _agent_from_config

        cfg = {
            "local": {
                "provider": "openai",
                "model": "gpt-4o-mini",
                "base_url": "https://api.openai.com/v1",
                "api_key": "sk-test",
                "embedding_model": "text-embedding-3-small",
            },
            "cloud": {
                "provider": "openai",
                "model": "gpt-4o",
                "api_key": "sk-test",
            },
            "memory": {"path": str(tmp_path / "mem.db")},
            "routing": {"confidence_threshold": 0.7},
        }

        with patch("autodidact.cli.Agent") as MockAgent:
            mock_agent = MagicMock()
            mock_agent._embed_client = MagicMock()
            mock_agent._conn = MagicMock()
            mock_agent._config = MagicMock(embedding_dim=1024)
            MockAgent.return_value = mock_agent

            _agent_from_config(cfg)

            # Inspect the Agent(...) call.
            call_kwargs = MockAgent.call_args.kwargs
            assert call_kwargs.get("local_model") == "openai/gpt-4o-mini"
            assert call_kwargs.get("local_base_url") == "https://api.openai.com/v1"
            assert call_kwargs.get("local_api_key_env") is not None

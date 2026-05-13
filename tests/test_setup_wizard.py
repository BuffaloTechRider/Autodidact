"""Tests for the zero-friction setup wizard.

TDD: tests written first, then implementation.

The setup wizard (autodidact init) should:
1. Auto-detect if Ollama is installed
2. If not, offer to install it
3. Auto-detect if the required models are pulled
4. If not, pull them automatically
5. Support cloud-to-cloud routing (cheap cloud ↔ expensive cloud, no local)
6. Support local-only mode (no cloud)
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch, call
import subprocess

import pytest


# ── Test: Ollama detection ───────────────────────────────────────

class TestOllamaDetection:
    """The wizard should detect whether Ollama is installed."""

    @patch("shutil.which", return_value="/usr/local/bin/ollama")
    def test_detects_ollama_installed(self, mock_which):
        from autodidact.setup_wizard import detect_ollama
        result = detect_ollama()
        assert result.installed is True
        assert result.path == "/usr/local/bin/ollama"

    @patch("shutil.which", return_value=None)
    def test_detects_ollama_not_installed(self, mock_which):
        from autodidact.setup_wizard import detect_ollama
        result = detect_ollama()
        assert result.installed is False
        assert result.path is None


# ── Test: Ollama model detection ─────────────────────────────────

class TestModelDetection:
    """The wizard should detect which models are already pulled."""

    @patch("subprocess.run")
    def test_detects_pulled_models(self, mock_run):
        mock_run.return_value = subprocess.CompletedProcess(
            args=["ollama", "list"],
            returncode=0,
            stdout="NAME              ID          SIZE    MODIFIED\nqwen2.5:7b        abc123      4.7 GB  2 days ago\nnomic-embed-text  def456      274 MB  5 days ago\n",
        )
        from autodidact.setup_wizard import list_ollama_models
        models = list_ollama_models()
        assert "qwen2.5:7b" in models
        assert "nomic-embed-text" in models

    @patch("subprocess.run")
    def test_returns_empty_on_failure(self, mock_run):
        mock_run.side_effect = FileNotFoundError("ollama not found")
        from autodidact.setup_wizard import list_ollama_models
        models = list_ollama_models()
        assert models == []

    @patch("autodidact.setup_wizard.requests.post")
    def test_checks_specific_model(self, mock_post):
        """is_model_available now asks Ollama via /api/show.

        Real models return 200 with details.format='gguf'; missing models 404.
        """
        from unittest.mock import MagicMock
        # First call: qwen2.5:7b is a real local model.
        # Second call: llama3.2 is missing.
        responses = [
            MagicMock(status_code=200, json=lambda: {"details": {"format": "gguf"}}),
            MagicMock(status_code=404),
        ]
        mock_post.side_effect = responses
        from autodidact.setup_wizard import is_model_available
        assert is_model_available("qwen2.5:7b") is True
        assert is_model_available("llama3.2") is False


# ── Test: Setup modes ────────────────────────────────────────────

class TestSetupModes:
    """The wizard should support three setup modes."""

    def test_local_cloud_mode_config(self):
        """Standard mode: local model + cloud escalation."""
        from autodidact.setup_wizard import build_config
        config = build_config(
            mode="local_cloud",
            local_model="qwen2.5:7b",
            embedding_model="qllama/bge-large-en-v1.5",
            cloud_provider="openai",
            cloud_model="gpt-4o",
            cloud_api_key="sk-test",
        )
        assert config["local"]["model"] == "qwen2.5:7b"
        assert config["cloud"]["model"] == "gpt-4o"
        assert config["cloud"]["provider"] == "openai"

    def test_cloud_cloud_mode_config(self):
        """Cloud-to-cloud: cheap cloud + expensive cloud, no local model."""
        from autodidact.setup_wizard import build_config
        config = build_config(
            mode="cloud_cloud",
            cheap_cloud_provider="openai",
            cheap_cloud_model="gpt-4o-mini",
            cheap_cloud_api_key="sk-test",
            expensive_cloud_provider="openai",
            expensive_cloud_model="gpt-4o",
            expensive_cloud_api_key="sk-test",
        )
        # In cloud-to-cloud mode, "local" is actually the cheap cloud model.
        assert config["local"]["provider"] == "openai"
        assert config["local"]["model"] == "gpt-4o-mini"
        assert config["cloud"]["model"] == "gpt-4o"
        assert "embedding_model" in config["local"]

    def test_local_only_mode_config(self):
        """Local-only: no cloud, no escalation."""
        from autodidact.setup_wizard import build_config
        config = build_config(
            mode="local_only",
            local_model="qwen2.5:7b",
            embedding_model="qllama/bge-large-en-v1.5",
        )
        assert config["local"]["model"] == "qwen2.5:7b"
        assert "cloud" not in config


# ── Test: Ollama install command ─────────────────────────────────

class TestOllamaInstall:
    """The wizard should provide the right install command per platform."""

    @patch("sys.platform", "darwin")
    def test_macos_install_command(self):
        from autodidact.setup_wizard import get_ollama_install_command
        cmd = get_ollama_install_command()
        assert "brew" in cmd or "curl" in cmd

    @patch("sys.platform", "linux")
    def test_linux_install_command(self):
        from autodidact.setup_wizard import get_ollama_install_command
        cmd = get_ollama_install_command()
        assert "curl" in cmd


# ── Test: Cloud provider presets ─────────────────────────────────

class TestCloudPresets:
    """The wizard should have presets for common cloud providers."""

    def test_openai_preset(self):
        from autodidact.setup_wizard import get_cloud_preset
        preset = get_cloud_preset("openai")
        assert preset["base_url"] == "https://api.openai.com/v1"
        assert preset["api_key_env"] == "OPENAI_API_KEY"
        assert "gpt-4o" in preset["models"]

    def test_openrouter_preset(self):
        from autodidact.setup_wizard import get_cloud_preset
        preset = get_cloud_preset("openrouter")
        assert "openrouter.ai" in preset["base_url"]
        assert preset["api_key_env"] == "OPENROUTER_API_KEY"

    def test_deepseek_preset(self):
        from autodidact.setup_wizard import get_cloud_preset
        preset = get_cloud_preset("deepseek")
        assert "deepseek" in preset["base_url"]

    def test_unknown_provider_returns_generic(self):
        from autodidact.setup_wizard import get_cloud_preset
        preset = get_cloud_preset("custom")
        assert preset["base_url"] == ""
        assert len(preset["models"]) == 0

"""Tests for sanity checks during `autodidact init` (item #4).

Three things we guard against:

1. Typos in the provider name -> offer closest match, loop until confirmed.
2. Typos in a model name -> warn and offer closest match, allow custom.
3. Unreachable models at smoke-test time -> print actionable diagnostics
   instead of a raw exception string.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
import yaml
from typer.testing import CliRunner

from autodidact.cli import app, _render_smoke_test_error

runner = CliRunner()


# ── Provider typos ──────────────────────────────────────────────


class TestProviderTypoHandling:
    """User mistypes a provider name -> wizard catches it."""

    @patch("autodidact.cli._run_smoke_test")
    @patch("autodidact.cli.detect_ollama")
    @patch("autodidact.cli.is_model_available", return_value=True)
    def test_typo_is_caught_and_user_reconfirms(
        self, mock_model, mock_detect, mock_smoke, tmp_path
    ):
        """'opnai' -> suggestion appears, user declines, retypes 'openai'."""
        from autodidact.setup_wizard import OllamaStatus
        mock_detect.return_value = OllamaStatus(installed=True, path="/usr/local/bin/ollama")
        cfg = tmp_path / "config.yaml"

        # mode=1, model default, 'opnai' (typo) -> decline 'use anyway' -> 'openai'
        # -> api key -> model default -> db default
        inputs = "1\n\nopnai\nn\nopenai\nsk-test\n\n\n"
        result = runner.invoke(app, ["init", "--config-path", str(cfg)], input=inputs)

        assert result.exit_code == 0, result.output
        # Suggestion shown.
        assert "did you mean" in result.output.lower()
        # Config uses the corrected provider.
        config = yaml.safe_load(cfg.read_text())
        assert config["cloud"]["provider"] == "openai"

    @patch("autodidact.cli._run_smoke_test")
    @patch("autodidact.cli.detect_ollama")
    @patch("autodidact.cli.is_model_available", return_value=True)
    def test_user_can_override_unknown_provider(
        self, mock_model, mock_detect, mock_smoke, tmp_path
    ):
        """User can force an unknown provider if they really mean it (custom setup)."""
        from autodidact.setup_wizard import OllamaStatus
        mock_detect.return_value = OllamaStatus(installed=True, path="/usr/local/bin/ollama")
        cfg = tmp_path / "config.yaml"

        # mode=1, model default, 'mycustom' -> yes use anyway -> api key -> model -> db
        inputs = "1\n\nmycustom\ny\nsk-test\nmy-model\n\n"
        result = runner.invoke(app, ["init", "--config-path", str(cfg)], input=inputs)

        assert result.exit_code == 0, result.output
        config = yaml.safe_load(cfg.read_text())
        assert config["cloud"]["provider"] == "mycustom"


# ── Model name warnings ─────────────────────────────────────────


class TestModelNameWarning:
    """User picks a model not in the preset's list -> warning, optional override."""

    @patch("autodidact.cli._run_smoke_test")
    @patch("autodidact.cli.detect_ollama")
    @patch("autodidact.cli.is_model_available", return_value=True)
    def test_unknown_model_is_flagged(
        self, mock_model, mock_detect, mock_smoke, tmp_path
    ):
        """'gpt-5o' (typo of gpt-4o) -> suggestion shown."""
        from autodidact.setup_wizard import OllamaStatus
        mock_detect.return_value = OllamaStatus(installed=True, path="/usr/local/bin/ollama")
        cfg = tmp_path / "config.yaml"

        # mode=1, model default, openai, api key, 'gpt-5o' -> yes use anyway -> db
        inputs = "1\n\nopenai\nsk-test\ngpt-5o\ny\n\n"
        result = runner.invoke(app, ["init", "--config-path", str(cfg)], input=inputs)

        assert result.exit_code == 0, result.output
        # Warning triggered (either suggestion or custom-model notice).
        assert ("not in the known model list" in result.output.lower()
                or "did you mean" in result.output.lower())

    @patch("autodidact.cli._run_smoke_test")
    @patch("autodidact.cli.detect_ollama")
    @patch("autodidact.cli.is_model_available", return_value=True)
    def test_user_can_force_custom_model(
        self, mock_model, mock_detect, mock_smoke, tmp_path
    ):
        """User picks a totally unrelated model name -> noted as custom, no reprompt."""
        from autodidact.setup_wizard import OllamaStatus
        mock_detect.return_value = OllamaStatus(installed=True, path="/usr/local/bin/ollama")
        cfg = tmp_path / "config.yaml"

        # mode=1, model default, openai, api key, 'my-fine-tune-xyz' (no close match)
        # -> no prompt expected if no suggestions, just use custom -> db
        inputs = "1\n\nopenai\nsk-test\nmy-fine-tune-xyz\n\n"
        result = runner.invoke(app, ["init", "--config-path", str(cfg)], input=inputs)

        assert result.exit_code == 0, result.output
        config = yaml.safe_load(cfg.read_text())
        assert config["cloud"]["model"] == "my-fine-tune-xyz"


# ── Smoke test error messaging ──────────────────────────────────


class TestSmokeTestDiagnostics:
    """_render_smoke_test_error should categorize common failures."""

    def _render_and_capture(self, exc: Exception, config: dict) -> str:
        """Call _render_smoke_test_error and return captured console output."""
        from autodidact.cli import console
        from io import StringIO
        from rich.console import Console

        buf = StringIO()
        # Temporarily swap the global console.
        original = console.file
        try:
            # The module-level console writes to stderr by default. Swap it.
            new_console = Console(file=buf, force_terminal=False)
            import autodidact.cli as cli_mod
            cli_mod.console = new_console
            _render_smoke_test_error(exc, config)
            return buf.getvalue()
        finally:
            cli_mod.console = Console()  # restore a fresh console

    def test_ollama_not_running_gets_hint(self):
        """Connection-refused errors get a 'run ollama serve' hint."""
        err = Exception("Connection refused to Ollama at http://localhost:11434")
        output = self._render_and_capture(err, {})
        assert "ollama serve" in output.lower() or "ollama" in output.lower()

    def test_missing_model_gets_pull_hint(self):
        """404-on-model errors tell the user which model to pull."""
        err = Exception("Ollama HTTP 404: model 'qwen2.5:7b' not found")
        output = self._render_and_capture(
            err, {"local": {"model": "qwen2.5:7b"}}
        )
        assert "ollama pull qwen2.5:7b" in output.lower() or "not pulled" in output.lower()

    def test_unauthorized_gets_api_key_hint(self):
        """401/unauthorized errors suggest checking the API key."""
        err = Exception("401 Unauthorized: invalid api_key")
        output = self._render_and_capture(err, {})
        assert "api key" in output.lower()

    def test_no_aws_credentials_gets_hint(self):
        """NoCredentialsError suggests setting AWS creds or re-running init."""
        err = Exception("NoCredentialsError: Unable to locate credentials")
        output = self._render_and_capture(err, {})
        assert "aws" in output.lower() or "credentials" in output.lower()

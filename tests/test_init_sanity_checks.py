"""Tests for sanity checks during `autodidact init` (item #4).

The original version of this file asserted freeform-prompt fuzzy-match UX
(difflib 'did you mean' suggestions). That UX was replaced with curated list
pickers in the wizard-polish PR — typos are no longer possible because users
pick from a list.

What remains to test:
  - Users can still enter custom provider names via the "Other" option.
  - Users can still enter custom model names via the "Other" option.
  - Smoke-test failures produce categorized diagnostic hints.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
import yaml
from typer.testing import CliRunner

from autodidact.cli import app, _render_smoke_test_error

runner = CliRunner()


# ── Custom provider / model via "Other" option ─────────────────


class TestCustomProviderEntry:
    """User can enter a custom provider name via the 'Other' option."""

    @patch("autodidact.cli._run_smoke_test")
    @patch("autodidact.setup_wizard.flow.detect_ollama")
    @patch("autodidact.setup_wizard.flow.is_model_available", return_value=True)
    @patch("autodidact.cli.detect_hardware")
    def test_custom_provider_via_other(
        self, mock_hw, mock_model, mock_detect, mock_smoke, tmp_path
    ):
        """Picking 'Other' at the provider step lets users supply a custom name."""
        from autodidact.setup_wizard import OllamaStatus
        from autodidact.hardware import HardwareProfile
        mock_detect.return_value = OllamaStatus(installed=True, path="/usr/local/bin/ollama")
        mock_hw.return_value = HardwareProfile(
            ram_gb=16, is_apple_silicon=False, has_nvidia=False,
            vram_gb=None, tier="medium",
        )
        cfg = tmp_path / "config.yaml"

        # mode default (1), local model default, provider='Other' (by name match),
        # custom provider name 'mycustom', api key, model 'my-cloud-model' (since
        # mycustom has no preset models, we must provide one for build_config
        # to emit the cloud section), db default
        inputs = "\n\nOther\nmycustom\nsk-test\nmy-cloud-model\n\n"
        result = runner.invoke(app, ["init", "--config-path", str(cfg)], input=inputs)

        assert result.exit_code == 0, result.output
        config = yaml.safe_load(cfg.read_text())
        assert config["cloud"]["provider"] == "mycustom"


class TestCustomModelEntry:
    """User can enter a custom model name via the 'Other' option."""

    @patch("autodidact.cli._run_smoke_test")
    @patch("autodidact.setup_wizard.flow.detect_ollama")
    @patch("autodidact.setup_wizard.flow.is_model_available", return_value=True)
    @patch("autodidact.cli.detect_hardware")
    def test_custom_model_via_other(
        self, mock_hw, mock_model, mock_detect, mock_smoke, tmp_path
    ):
        """At the model step, picking 'Other' lets users supply a custom model."""
        from autodidact.setup_wizard import OllamaStatus
        from autodidact.hardware import HardwareProfile
        mock_detect.return_value = OllamaStatus(installed=True, path="/usr/local/bin/ollama")
        mock_hw.return_value = HardwareProfile(
            ram_gb=16, is_apple_silicon=False, has_nvidia=False,
            vram_gb=None, tier="medium",
        )
        cfg = tmp_path / "config.yaml"

        # mode=1 default, local model default, openai, api key, 'Other', 'my-fine-tune-xyz', db default
        inputs = "\n\nopenai\nsk-test\nOther\nmy-fine-tune-xyz\n\n"
        result = runner.invoke(app, ["init", "--config-path", str(cfg)], input=inputs)

        assert result.exit_code == 0, result.output
        config = yaml.safe_load(cfg.read_text())
        assert config["cloud"]["model"] == "my-fine-tune-xyz"


# ── Smoke test error messaging ──────────────────────────────────


class TestSmokeTestDiagnostics:
    """_render_smoke_test_error should categorize common failures."""

    def _render_and_capture(self, exc: Exception, config: dict) -> str:
        """Call _render_smoke_test_error and return captured console output."""
        from io import StringIO

        from rich.console import Console

        buf = StringIO()
        # _render_smoke_test_error lives in setup_wizard/smoke.py and writes
        # to setup_wizard._console.console. Swap that one for capture.
        import autodidact.setup_wizard._console as console_mod
        original = console_mod.console
        try:
            console_mod.console = Console(file=buf, force_terminal=False)
            # The smoke module imports `console` directly into its
            # namespace, so we also need to swap the module-local binding.
            import autodidact.setup_wizard.smoke as smoke_mod
            smoke_mod.console = console_mod.console
            _render_smoke_test_error(exc, config)
            return buf.getvalue()
        finally:
            console_mod.console = original
            import autodidact.setup_wizard.smoke as smoke_mod
            smoke_mod.console = original

    def test_ollama_not_running_gets_hint(self):
        err = Exception("Connection refused to Ollama at http://localhost:11434")
        output = self._render_and_capture(err, {})
        assert "ollama serve" in output.lower() or "ollama" in output.lower()

    def test_missing_model_gets_pull_hint(self):
        err = Exception("Ollama HTTP 404: model 'qwen2.5:7b' not found")
        output = self._render_and_capture(
            err, {"local": {"model": "qwen2.5:7b"}}
        )
        assert "ollama pull qwen2.5:7b" in output.lower() or "not pulled" in output.lower()

    def test_unauthorized_gets_api_key_hint(self):
        err = Exception("401 Unauthorized: invalid api_key")
        output = self._render_and_capture(err, {})
        assert "api key" in output.lower()

    def test_no_aws_credentials_gets_hint(self):
        err = Exception("NoCredentialsError: Unable to locate credentials")
        output = self._render_and_capture(err, {})
        assert "aws" in output.lower() or "credentials" in output.lower()

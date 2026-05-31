"""Tests for the Ollama install + daemon-start flow in `autodidact init`.

Three things to cover:
  1. is_ollama_running() — pings localhost:11434/api/tags, gracefully handles
     connection errors.
  2. install_ollama() — runs the platform-specific install command, returns
     a success bool. Tests use mocks; we never actually run the installer
     in CI.
  3. start_ollama_daemon() — best-effort start. Returns True if it can
     verify the daemon is up afterwards, False otherwise.

The wizard integration (when to prompt the user) is covered in
test_init_wizard_integration.py.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest
import requests

from autodidact.setup_wizard import (
    OllamaStatus,
    get_ollama_install_command,
    install_ollama,
    is_ollama_running,
    start_ollama_daemon,
    wait_for_ollama_daemon,
)


# ── is_ollama_running ─────────────────────────────────────────────


class TestIsOllamaRunning:
    """Detects whether the daemon is up by pinging /api/tags."""

    @patch("autodidact.setup_wizard.ollama.requests.get")
    def test_daemon_up_returns_true(self, mock_get):
        mock_get.return_value = MagicMock(status_code=200)
        assert is_ollama_running() is True

    @patch("autodidact.setup_wizard.ollama.requests.get")
    def test_daemon_down_returns_false(self, mock_get):
        mock_get.side_effect = requests.exceptions.ConnectionError("connection refused")
        assert is_ollama_running() is False

    @patch("autodidact.setup_wizard.ollama.requests.get")
    def test_timeout_returns_false(self, mock_get):
        mock_get.side_effect = requests.exceptions.Timeout("timed out")
        assert is_ollama_running() is False

    @patch("autodidact.setup_wizard.ollama.requests.get")
    def test_unexpected_status_returns_false(self, mock_get):
        """Non-200 (5xx, 401 etc.) is treated as 'not really ready'."""
        mock_get.return_value = MagicMock(status_code=500)
        assert is_ollama_running() is False


# ── get_ollama_install_command ────────────────────────────────────


class TestInstallCommand:
    """The recommended install command depends on platform."""

    @patch("autodidact.setup_wizard.ollama.sys")
    def test_macos_uses_official_installer(self, mock_sys):
        """macOS gets the official curl-based installer (works without Homebrew)."""
        mock_sys.platform = "darwin"
        cmd = get_ollama_install_command()
        assert "ollama.com/install.sh" in cmd or "ollama.com/download" in cmd

    @patch("autodidact.setup_wizard.ollama.sys")
    def test_linux_uses_official_installer(self, mock_sys):
        mock_sys.platform = "linux"
        cmd = get_ollama_install_command()
        assert "ollama.com/install.sh" in cmd

    @patch("autodidact.setup_wizard.ollama.sys")
    def test_windows_returns_manual_instructions(self, mock_sys):
        """Windows isn't auto-installed in v1.0; the helper points at the download page."""
        mock_sys.platform = "win32"
        cmd = get_ollama_install_command()
        assert "ollama.com/download" in cmd or "windows" in cmd.lower()


# ── install_ollama ────────────────────────────────────────────────


class TestInstallOllama:
    """Runs the install command. Returns True on a successful subprocess run."""

    @patch("autodidact.setup_wizard.ollama._has_homebrew", return_value=False)
    @patch("autodidact.setup_wizard.ollama.subprocess.run")
    @patch("autodidact.setup_wizard.ollama.sys")
    def test_macos_install_success(self, mock_sys, mock_run, _mock_brew):
        mock_sys.platform = "darwin"
        mock_run.return_value = MagicMock(returncode=0)
        assert install_ollama() is True
        # Must invoke a curl-pipe-sh command.
        args = mock_run.call_args.args[0]
        assert any("install.sh" in str(a) for a in args)

    @patch("autodidact.setup_wizard.ollama.subprocess.run")
    @patch("autodidact.setup_wizard.ollama.sys")
    def test_install_failure(self, mock_sys, mock_run):
        mock_sys.platform = "darwin"
        mock_run.return_value = MagicMock(returncode=1, stderr="error")
        assert install_ollama() is False

    @patch("autodidact.setup_wizard.ollama.subprocess.run", side_effect=OSError("not found"))
    @patch("autodidact.setup_wizard.ollama.sys")
    def test_install_subprocess_oserror(self, mock_sys, mock_run):
        mock_sys.platform = "darwin"
        assert install_ollama() is False

    @patch("autodidact.setup_wizard.ollama.sys")
    def test_windows_install_returns_false(self, mock_sys):
        """Windows install isn't supported in v1.0; the function refuses."""
        mock_sys.platform = "win32"
        assert install_ollama() is False


# ── start_ollama_daemon + wait_for_ollama_daemon ──────────────────


class TestStartDaemon:
    """Best-effort: spawn the daemon, then poll until /api/tags responds."""

    @patch("autodidact.setup_wizard.ollama.subprocess.Popen")
    @patch("autodidact.setup_wizard.ollama.is_ollama_running", return_value=True)
    @patch("autodidact.setup_wizard.ollama.sys")
    def test_macos_open_app(self, mock_sys, mock_running, mock_popen):
        """On macOS, the daemon ships as an .app — `open -a Ollama` launches it."""
        mock_sys.platform = "darwin"
        assert start_ollama_daemon() is True
        # Must call `open -a Ollama` or invoke the bare ollama binary.
        cmd = mock_popen.call_args.args[0]
        assert "open" in cmd or "ollama" in cmd

    @patch("autodidact.setup_wizard.ollama.subprocess.Popen")
    @patch("autodidact.setup_wizard.ollama.is_ollama_running", return_value=True)
    @patch("autodidact.setup_wizard.ollama.sys")
    def test_linux_runs_serve(self, mock_sys, mock_running, mock_popen):
        """On Linux, `ollama serve` in background brings up the daemon."""
        mock_sys.platform = "linux"
        assert start_ollama_daemon() is True
        cmd = mock_popen.call_args.args[0]
        assert "ollama" in cmd and "serve" in cmd

    @patch("autodidact.setup_wizard.ollama.subprocess.Popen")
    @patch("autodidact.setup_wizard.ollama.is_ollama_running", return_value=False)
    @patch("autodidact.setup_wizard.ollama.sys")
    def test_failure_to_come_up_returns_false(self, mock_sys, mock_running, mock_popen):
        """If polling for the daemon times out, return False."""
        mock_sys.platform = "linux"
        assert start_ollama_daemon(wait_timeout_s=0.5) is False


class TestWaitForDaemon:
    """Polls is_ollama_running until True or timeout."""

    @patch("autodidact.setup_wizard.ollama.is_ollama_running")
    @patch("autodidact.setup_wizard.ollama.time.sleep")
    def test_returns_true_when_daemon_comes_up(self, _sleep, mock_running):
        # Simulates the daemon coming up on the second check.
        mock_running.side_effect = [False, True]
        assert wait_for_ollama_daemon(timeout_s=2.0) is True

    @patch("autodidact.setup_wizard.ollama.is_ollama_running", return_value=False)
    @patch("autodidact.setup_wizard.ollama.time.sleep")
    def test_returns_false_on_timeout(self, _sleep, _running):
        assert wait_for_ollama_daemon(timeout_s=0.1) is False

    @patch("autodidact.setup_wizard.ollama.is_ollama_running", return_value=True)
    @patch("autodidact.setup_wizard.ollama.time.sleep")
    def test_returns_immediately_when_already_running(self, _sleep, _running):
        assert wait_for_ollama_daemon(timeout_s=2.0) is True


# ── Safety net: conftest.py autouse fixture must be in effect ─────


class TestSafetyFixtureBypassedHere:
    """The autouse fixture in conftest.py skips this file so we can test the real impl.

    But other tests must NOT see the real impl. Verify by importing from a
    non-test module and confirming it's the real callable, not a stub.
    """

    def test_real_install_ollama_imported_here(self):
        """In this file specifically, install_ollama is the real one (not stubbed)."""
        from autodidact.setup_wizard import install_ollama
        # Real function defined in setup_wizard.py — not a lambda from conftest.
        assert install_ollama.__module__ == "autodidact.setup_wizard.ollama"
        assert install_ollama.__name__ == "install_ollama"

    def test_real_start_daemon_imported_here(self):
        from autodidact.setup_wizard import start_ollama_daemon
        assert start_ollama_daemon.__module__ == "autodidact.setup_wizard.ollama"

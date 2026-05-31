"""Verifies the conftest.py autouse safety fixture is actually in effect.

These tests are NOT in test_ollama_install_flow.py, so the autouse fixture
should stub out the system-effect helpers. We assert the stubs are present
by checking that calling them through the modules where they're used
(autodidact.cli) returns the stubbed values, not what the real
implementations would do.

Background: an early version of the install-on-init flow ran the real Ollama
installer on a developer's machine. The autouse fixture in conftest.py
blocks that. These tests are a regression guard.

Note: ``from autodidact.setup_wizard import install_ollama`` would bind the
*real* function at import time and bypass the monkeypatch. We test through
``autodidact.cli`` (the consumer) since that's the path the wizard actually
takes at runtime.
"""

from __future__ import annotations

import autodidact.cli as cli


def test_install_ollama_is_stubbed_to_false():
    """The conftest fixture stubs install_ollama where the wizard calls it."""
    # If this returned True, the real installer ran. That's the bug we guard against.
    assert cli.install_ollama() is False


def test_start_ollama_daemon_is_stubbed_to_false():
    assert cli.start_ollama_daemon() is False
    assert cli.start_ollama_daemon(wait_timeout_s=99) is False


def test_pull_ollama_model_is_stubbed_to_true():
    """Pulls return True (success) without invoking the network."""
    assert cli.pull_ollama_model("anything") is True


def test_is_ollama_running_is_stubbed_to_true():
    """The default stub assumes the daemon is up so wizard tests skip the offer-to-start path."""
    assert cli.is_ollama_running() is True



# ── End-to-end safety: running the wizard never invokes real installers ──


def test_wizard_with_no_ollama_installed_does_not_invoke_real_installer(
    tmp_path, monkeypatch
):
    """The full init wizard, run with no Ollama and a 'yes' answer to the
    install prompt, must NOT actually run a real installer.

    This is the strongest regression guard: it simulates the exact failure
    condition from the original incident.
    """
    from typer.testing import CliRunner

    from autodidact.cli import app
    from autodidact.setup_wizard import OllamaStatus

    # Ollama is "not installed."
    monkeypatch.setattr(
        "autodidact.cli.detect_ollama",
        lambda: OllamaStatus(installed=False, path=None),
    )

    # Patch subprocess.run with a tripwire — if the real installer ran, this
    # would have been called. The autouse fixture's stub for install_ollama
    # should mean we never reach subprocess.run for the install.
    real_subprocess_run_called = []

    def tripwire(*args, **kwargs):
        # Capture the call so the test assertion can see it.
        real_subprocess_run_called.append((args, kwargs))
        # Pretend success so the wizard continues.
        from unittest.mock import MagicMock
        return MagicMock(returncode=0)

    monkeypatch.setattr("autodidact.setup_wizard.ollama.subprocess.run", tripwire)

    runner = CliRunner()
    cfg = tmp_path / "config.yaml"
    # Inputs: confirm=Y to install, mode default (1), local model default,
    # provider default, api key, model default, db default. Plenty of \n
    # to avoid stdin exhaustion.
    inputs = "y\n\n\nopenai\nsk-test\n\n\n\n\n"
    runner.invoke(app, ["init", "--config-path", str(cfg)], input=inputs)

    # The conftest fixture stubs install_ollama with a no-op. So
    # subprocess.run('curl ... | sh ...') must NOT have been invoked.
    install_calls = [
        call for call in real_subprocess_run_called
        if any("install.sh" in str(arg) for arg in call[0])
    ]
    assert install_calls == [], (
        f"Wizard invoked the real installer {len(install_calls)} time(s)! "
        "The autouse safety fixture in conftest.py is not working."
    )

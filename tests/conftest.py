"""Shared pytest fixtures and safety nets for the test suite.

Importantly: this file installs an autouse fixture that prevents the test
suite from EVER running real system installers, daemon starters, or model
pulls. Tests that legitimately need to verify those code paths must mock the
helpers explicitly with their own ``patch()`` calls (which take precedence
over these autouse stubs because pytest applies decorators bottom-up).

Why this is here: an early version of the install-on-init flow ran the real
Ollama installer on a developer's machine because an old test exercised the
code path without mocking. That should never be possible from CI or a local
test run.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def _block_real_system_side_effects(request, monkeypatch):
    """Prevent any test from running real installers, daemon starters, or pulls.

    Every test runs with these helpers stubbed out by default. Tests that
    want to assert against specific behaviors of these functions can override
    the fixture's stubs with ``@patch(...)`` decorators (which apply later
    and so win over monkeypatch).

    Skipped for tests in ``test_ollama_install_flow.py`` — that file
    explicitly tests these helpers and uses fully-mocked subprocess /
    requests calls instead of relying on this guard.
    """
    if request.node.path.name == "test_ollama_install_flow.py":
        # Those tests set up their own mocks and should run end-to-end.
        return

    # Block real system side effects across the rest of the suite. Returning
    # False / True here matches how the real functions signal failure / success.
    #
    # The wizard interactive prompts in autodidact.setup_wizard.flow import
    # these helpers from autodidact.setup_wizard.ollama at module load time
    # — patches must target the bindings IN flow.py (where they're called)
    # AND in ollama.py (where the real implementation lives) for full
    # safety coverage. Patching only the package-level alias would leave the
    # real installer reachable via the flow.<name> binding.
    monkeypatch.setattr(
        "autodidact.setup_wizard.ollama.install_ollama",
        lambda: False,
        raising=False,
    )
    monkeypatch.setattr(
        "autodidact.setup_wizard.flow.install_ollama",
        lambda: False,
        raising=False,
    )
    monkeypatch.setattr(
        "autodidact.setup_wizard.ollama.start_ollama_daemon",
        lambda *a, **k: False,
        raising=False,
    )
    monkeypatch.setattr(
        "autodidact.setup_wizard.flow.start_ollama_daemon",
        lambda *a, **k: False,
        raising=False,
    )
    monkeypatch.setattr(
        "autodidact.setup_wizard.ollama.pull_ollama_model",
        lambda *a, **k: (True, ""),
        raising=False,
    )
    monkeypatch.setattr(
        "autodidact.setup_wizard.flow.pull_ollama_model",
        lambda *a, **k: (True, ""),
        raising=False,
    )
    monkeypatch.setattr(
        "autodidact.setup_wizard.ollama.is_ollama_running",
        lambda: True,
        raising=False,
    )
    monkeypatch.setattr(
        "autodidact.setup_wizard.flow.is_ollama_running",
        lambda: True,
        raising=False,
    )
    # Also patch the package-level re-exports so direct
    # ``from autodidact.setup_wizard import install_ollama`` callers are
    # equally safe.
    monkeypatch.setattr(
        "autodidact.setup_wizard.install_ollama",
        lambda: False,
        raising=False,
    )
    monkeypatch.setattr(
        "autodidact.setup_wizard.start_ollama_daemon",
        lambda *a, **k: False,
        raising=False,
    )
    monkeypatch.setattr(
        "autodidact.setup_wizard.pull_ollama_model",
        lambda *a, **k: (True, ""),
        raising=False,
    )
    monkeypatch.setattr(
        "autodidact.setup_wizard.is_ollama_running",
        lambda: True,
        raising=False,
    )
    # Also patch the symbols re-exported into autodidact.cli, since the wizard
    # imports them directly and patches via the cli module wouldn't see
    # changes to setup_wizard alone.
    monkeypatch.setattr(
        "autodidact.cli.install_ollama",
        lambda: False,
        raising=False,
    )
    monkeypatch.setattr(
        "autodidact.cli.start_ollama_daemon",
        lambda *a, **k: False,
        raising=False,
    )
    monkeypatch.setattr(
        "autodidact.cli.is_ollama_running",
        lambda: True,
        raising=False,
    )
    monkeypatch.setattr(
        "autodidact.cli.pull_ollama_model",
        lambda *a, **k: True,
        raising=False,
    )

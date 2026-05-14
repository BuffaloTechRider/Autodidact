"""Tests for OpenRouter model discovery.

OpenRouter has 300+ models and the slugs are easy to mistype. Static preset
covers ~8. Discovery via the public /v1/models endpoint (no API key needed)
gives the user a way to browse the full catalogue without leaving the wizard.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest


# ── Helpers ──────────────────────────────────────────────────────


def _model(
    *,
    id: str,
    prompt: str = "0.000001",
    completion: str = "0.000003",
    output_modalities: list[str] | None = None,
    context_length: int = 8192,
):
    return {
        "id": id,
        "pricing": {"prompt": prompt, "completion": completion},
        "architecture": {"output_modalities": output_modalities or ["text"]},
        "context_length": context_length,
    }


class _FakeResponse:
    def __init__(self, payload, *, status_code: int = 200):
        self._payload = payload
        self.status_code = status_code
        self.ok = 200 <= status_code < 300

    def json(self):
        return self._payload

    def raise_for_status(self):
        if not self.ok:
            from requests.exceptions import HTTPError
            raise HTTPError(f"{self.status_code}")


# ── discover_openrouter_models ───────────────────────────────────


class TestDiscoverOpenRouterModels:
    def test_returns_id_and_pricing_sorted_cheapest_first(self):
        from autodidact.setup_wizard import discover_openrouter_models

        payload = {
            "data": [
                _model(id="expensive/model", prompt="0.0001", completion="0.0003"),
                _model(id="cheap/model", prompt="0.0000001", completion="0.0000003"),
                _model(id="medium/model", prompt="0.00001", completion="0.00003"),
            ],
        }

        with patch(
            "autodidact.setup_wizard.requests.get",
            return_value=_FakeResponse(payload),
        ):
            entries = discover_openrouter_models()

        ids = [e.id for e in entries]
        assert ids == ["cheap/model", "medium/model", "expensive/model"]
        # Each entry exposes per-1M-token pricing for the picker label.
        cheap = entries[0]
        assert cheap.prompt_per_million == pytest.approx(0.1)         # 0.0000001 * 1e6
        assert cheap.completion_per_million == pytest.approx(0.3)     # 0.0000003 * 1e6

    def test_filters_non_text_output_models(self):
        from autodidact.setup_wizard import discover_openrouter_models

        payload = {
            "data": [
                _model(id="text/only", output_modalities=["text"]),
                _model(id="image/only", output_modalities=["image"]),
                _model(id="audio/only", output_modalities=["audio"]),
                # Multimodal-out models are kept as long as text is one of the outputs.
                _model(id="text+image/out", output_modalities=["text", "image"]),
            ],
        }

        with patch(
            "autodidact.setup_wizard.requests.get",
            return_value=_FakeResponse(payload),
        ):
            ids = [e.id for e in discover_openrouter_models()]

        assert "text/only" in ids
        assert "text+image/out" in ids
        assert "image/only" not in ids
        assert "audio/only" not in ids

    def test_models_with_missing_pricing_are_dropped(self):
        from autodidact.setup_wizard import discover_openrouter_models

        payload = {
            "data": [
                _model(id="ok/model"),
                {"id": "no-pricing/model", "architecture": {"output_modalities": ["text"]}},
                {"id": "partial/model", "pricing": {"prompt": "0.0001"},
                 "architecture": {"output_modalities": ["text"]}},
            ],
        }

        with patch(
            "autodidact.setup_wizard.requests.get",
            return_value=_FakeResponse(payload),
        ):
            ids = [e.id for e in discover_openrouter_models()]

        assert ids == ["ok/model"]

    def test_negative_pricing_is_dropped(self):
        """OpenRouter encodes dynamic-pricing meta-models as -1. They are
        routing shortcuts (openrouter/auto), not real model picks."""
        from autodidact.setup_wizard import discover_openrouter_models

        payload = {
            "data": [
                _model(id="real/model"),
                _model(id="openrouter/auto", prompt="-1", completion="-1"),
                _model(id="partial-negative/model", prompt="0.001", completion="-1"),
            ],
        }

        with patch(
            "autodidact.setup_wizard.requests.get",
            return_value=_FakeResponse(payload),
        ):
            ids = [e.id for e in discover_openrouter_models()]

        assert "openrouter/auto" not in ids
        assert "partial-negative/model" not in ids
        assert "real/model" in ids

    def test_http_failure_raises_typed_error(self):
        from autodidact.setup_wizard import (
            OpenRouterDiscoveryError,
            discover_openrouter_models,
        )
        from requests.exceptions import RequestException

        with patch(
            "autodidact.setup_wizard.requests.get",
            side_effect=RequestException("connection refused"),
        ):
            with pytest.raises(OpenRouterDiscoveryError) as exc:
                discover_openrouter_models()
        assert "connection refused" in str(exc.value)

    def test_non_200_response_raises_typed_error(self):
        from autodidact.setup_wizard import (
            OpenRouterDiscoveryError,
            discover_openrouter_models,
        )

        with patch(
            "autodidact.setup_wizard.requests.get",
            return_value=_FakeResponse({"error": "nope"}, status_code=503),
        ):
            with pytest.raises(OpenRouterDiscoveryError):
                discover_openrouter_models()


# ── Wizard wiring: "Browse all" entry triggers discovery ─────────


class TestPromptOpenRouterUsesBrowse:
    """The OpenRouter branch of _prompt_openai_compat_config exposes a
    `Browse all OpenRouter models` choice that calls discovery."""

    def test_preset_picker_includes_browse_choice(self, monkeypatch):
        from autodidact import cli

        captured: dict = {}

        def fake_pick(title, choices, default):
            captured.setdefault("calls", []).append({"choices": choices, "default": default})
            return choices[0]  # pick first preset entry

        monkeypatch.setattr(cli, "_pick_from_list", fake_pick)
        monkeypatch.setattr(cli.typer, "prompt", lambda *a, **k: "sk-or-v1-test")

        preset = {
            "models": ["openai/gpt-4o", "anthropic/claude-sonnet-4-5"],
            "default_cheap": "openai/gpt-4o",
            "default_expensive": "anthropic/claude-sonnet-4-5",
            "base_url": "https://openrouter.ai/api/v1",
            "api_key_env": "OPENROUTER_API_KEY",
        }

        result = cli._prompt_openai_compat_config("openrouter", preset, slot="cheap")

        first_call = captured["calls"][0]
        assert "openai/gpt-4o" in first_call["choices"]
        assert any("Browse" in c for c in first_call["choices"]), (
            f"Browse-all choice missing from picker; got {first_call['choices']!r}"
        )
        assert result["model"] == "openai/gpt-4o"

    def test_browse_choice_invokes_discovery_and_picks_from_full_list(self, monkeypatch):
        from autodidact import cli
        from autodidact.setup_wizard import OpenRouterModel

        captured_choices: list[list[str]] = []

        def fake_pick(title, choices, default):
            captured_choices.append(list(choices))
            if "Browse" in choices[0] or any("Browse" in c for c in choices[:3]):
                # First call: preset picker. Pick the Browse entry.
                for c in choices:
                    if "Browse" in c:
                        return c
            # Second call: full discovered catalogue with labeled rows.
            # Pick the row whose label contains the qwen id.
            for c in choices:
                if "qwen/qwen-2.5-coder-32b" in c:
                    return c
            return choices[0]

        monkeypatch.setattr(cli, "_pick_from_list", fake_pick)
        monkeypatch.setattr(cli.typer, "prompt", lambda *a, **k: "sk-or-v1-test")

        monkeypatch.setattr(
            cli,
            "discover_openrouter_models",
            lambda: [
                OpenRouterModel(
                    id="qwen/qwen-2.5-coder-32b",
                    prompt_per_million=0.07,
                    completion_per_million=0.07,
                    context_length=131072,
                ),
                OpenRouterModel(
                    id="anthropic/claude-opus-4-5",
                    prompt_per_million=15.0,
                    completion_per_million=75.0,
                    context_length=200000,
                ),
            ],
        )

        preset = {
            "models": ["openai/gpt-4o"],
            "default_cheap": "openai/gpt-4o",
            "default_expensive": "openai/gpt-4o",
            "base_url": "https://openrouter.ai/api/v1",
            "api_key_env": "OPENROUTER_API_KEY",
        }

        result = cli._prompt_openai_compat_config("openrouter", preset, slot="cheap")

        # Second picker call (the browse picker) should have received the
        # discovered IDs as choices, even if labeled with pricing.
        browse_choices = captured_choices[1]
        assert any("qwen/qwen-2.5-coder-32b" in c for c in browse_choices)
        assert any("anthropic/claude-opus-4-5" in c for c in browse_choices)
        # The picker returns a labeled row; the implementation must extract
        # the bare id for the agent config.
        assert result["model"] == "qwen/qwen-2.5-coder-32b"

    def test_browse_falls_back_to_free_form_when_discovery_fails(self, monkeypatch):
        from autodidact import cli
        from autodidact.setup_wizard import OpenRouterDiscoveryError

        # Picker is called once (preset), then user picks Browse → discovery
        # fails → free-form prompt collects the model.
        picker_calls = iter(["↪ Browse all OpenRouter models (live)"])
        text_prompts = iter([
            "sk-or-v1-test",
            "qwen/qwen-2.5-coder-32b",  # free-form fallback
        ])

        monkeypatch.setattr(
            cli,
            "_pick_from_list",
            lambda *a, **k: next(picker_calls),
        )
        monkeypatch.setattr(cli.typer, "prompt", lambda *a, **k: next(text_prompts))

        def boom():
            raise OpenRouterDiscoveryError("503 from /v1/models")

        monkeypatch.setattr(cli, "discover_openrouter_models", boom)

        preset = {
            "models": ["openai/gpt-4o"],
            "default_cheap": "openai/gpt-4o",
            "default_expensive": "openai/gpt-4o",
            "base_url": "https://openrouter.ai/api/v1",
            "api_key_env": "OPENROUTER_API_KEY",
        }

        result = cli._prompt_openai_compat_config("openrouter", preset, slot="cheap")
        assert result["model"] == "qwen/qwen-2.5-coder-32b"

    def test_non_openrouter_provider_unchanged(self, monkeypatch):
        """Other OpenAI-compat providers must not get the Browse choice or call discovery."""
        from autodidact import cli

        captured: list[list[str]] = []

        def fake_pick(title, choices, default):
            captured.append(list(choices))
            return choices[0]

        monkeypatch.setattr(cli, "_pick_from_list", fake_pick)
        monkeypatch.setattr(cli.typer, "prompt", lambda *a, **k: "sk-test")

        # If discovery is reached for non-openrouter, fail loudly.
        monkeypatch.setattr(
            cli,
            "discover_openrouter_models",
            lambda: pytest.fail("discover_openrouter_models must not be called for non-openrouter providers"),
        )

        preset = {
            "models": ["gpt-4o", "gpt-4o-mini"],
            "default_cheap": "gpt-4o-mini",
            "default_expensive": "gpt-4o",
            "base_url": "https://api.openai.com/v1",
            "api_key_env": "OPENAI_API_KEY",
        }

        cli._prompt_openai_compat_config("openai", preset, slot="cheap")

        # No Browse choice in any picker call.
        for choices in captured:
            assert not any("Browse" in c for c in choices)

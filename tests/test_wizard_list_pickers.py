"""Tests for the curated-list pickers used in the wizard.

We pick the list libs through a thin abstraction (_pick_from_list) so tests
don't need to pump questionary's internal state. The abstraction takes a
title, choices, and a default, and returns the selection string. In tests
we mock _pick_from_list; in prod it delegates to questionary.select when
available, falling back to a numbered prompt when questionary is missing
or stdin isn't a TTY.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from autodidact.cli import (
    _pick_from_list,
    _pick_local_model,
    _pick_cloud_provider,
    _pick_setup_mode,
)


# ── _pick_from_list fallback ──────────────────────────────────────


class TestPickFromList:
    """_pick_from_list gracefully falls back when questionary isn't usable."""

    @patch("autodidact.setup_wizard.picker._questionary_available", return_value=False)
    @patch("autodidact.setup_wizard.picker.typer.prompt", return_value="2")
    def test_falls_back_to_numbered_prompt(self, _tp, _qa):
        result = _pick_from_list("Pick one", ["a", "b", "c"], default="a")
        assert result == "b"

    @patch("autodidact.setup_wizard.picker._questionary_available", return_value=False)
    @patch("autodidact.setup_wizard.picker.typer.prompt", return_value="")
    def test_empty_input_returns_default(self, _tp, _qa):
        """Pressing Enter alone picks the default."""
        result = _pick_from_list("Pick one", ["a", "b", "c"], default="b")
        assert result == "b"

    @patch("autodidact.setup_wizard.picker._questionary_available", return_value=False)
    @patch("autodidact.setup_wizard.picker.typer.prompt", return_value="99")
    def test_invalid_index_returns_default(self, _tp, _qa):
        result = _pick_from_list("Pick one", ["a", "b", "c"], default="c")
        assert result == "c"


# ── Setup mode picker ────────────────────────────────────────────


class TestPickSetupMode:
    """_pick_setup_mode offers 3 modes and returns the canonical key."""

    @patch("autodidact.setup_wizard.prompts._pick_from_list")
    def test_local_cloud_is_default(self, mock_pick):
        mock_pick.return_value = "Local + Cloud   — Ollama local + cloud for escalation (best savings)"
        mode = _pick_setup_mode()
        assert mode == "local_cloud"

    @patch("autodidact.setup_wizard.prompts._pick_from_list")
    def test_cloud_cloud(self, mock_pick):
        mock_pick.return_value = "Cloud + Cloud   — cheap cloud + expensive cloud (no GPU needed)"
        mode = _pick_setup_mode()
        assert mode == "cloud_cloud"

    @patch("autodidact.setup_wizard.prompts._pick_from_list")
    def test_local_only(self, mock_pick):
        mock_pick.return_value = "Local only      — Ollama only, no cloud (free, no learning escalations)"
        mode = _pick_setup_mode()
        assert mode == "local_only"


# ── Local model picker ───────────────────────────────────────────


class TestPickLocalModel:
    """_pick_local_model lists curated options + 'other'; returns a model name."""

    @patch("autodidact.setup_wizard.prompts._pick_from_list")
    def test_picks_from_curated_list(self, mock_pick):
        # Any curated option returns its model name.
        mock_pick.return_value = "qwen2.5:7b — balanced default (4.7GB)"
        result = _pick_local_model(recommended="qwen2.5:7b")
        assert result == "qwen2.5:7b"

    @patch("autodidact.setup_wizard.prompts._pick_from_list")
    @patch("autodidact.setup_wizard.prompts.typer.prompt", return_value="my-custom-model:7b")
    def test_other_prompts_for_custom_name(self, _tp, mock_pick):
        mock_pick.return_value = "Other (type a model name)"
        result = _pick_local_model(recommended="qwen2.5:7b")
        assert result == "my-custom-model:7b"

    @patch("autodidact.setup_wizard.prompts._pick_from_list")
    def test_recommended_model_is_the_default_choice(self, mock_pick):
        """The picker must pass the recommended model as the default string."""
        _pick_local_model(recommended="qwen2.5:14b")
        # Inspect the default= arg given to _pick_from_list.
        default_arg = mock_pick.call_args.kwargs.get("default") or mock_pick.call_args.args[2]
        assert "qwen2.5:14b" in default_arg


# ── Cloud provider picker ────────────────────────────────────────


class TestPickCloudProvider:
    """_pick_cloud_provider offers the known providers + 'other'."""

    @patch("autodidact.setup_wizard.prompts._pick_from_list")
    def test_picks_openai(self, mock_pick):
        mock_pick.return_value = "openai"
        assert _pick_cloud_provider() == "openai"

    @patch("autodidact.setup_wizard.prompts._pick_from_list")
    def test_picks_bedrock(self, mock_pick):
        mock_pick.return_value = "bedrock"
        assert _pick_cloud_provider() == "bedrock"

    @patch("autodidact.setup_wizard.prompts._pick_from_list")
    @patch("autodidact.setup_wizard.prompts.typer.prompt", return_value="my-custom-provider")
    def test_other_allows_custom_provider(self, _tp, mock_pick):
        mock_pick.return_value = "Other (type a model name)"
        assert _pick_cloud_provider() == "my-custom-provider"

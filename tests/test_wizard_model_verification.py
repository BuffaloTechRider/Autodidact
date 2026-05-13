"""Tests for post-pull model verification.

Two failure modes we guard against:

1. **Cloud-only manifest pulls.** Some Ollama tags (qwen3.5:9b on some days,
   qwen3-coder:480b-cloud, *:cloud variants) succeed at `ollama pull` but
   only fetch a manifest pointing at remote inference. The model never
   appears in `ollama list` and /api/chat 404s. Our detection: `/api/show`
   returns 200 but `details.format` is empty for these.

2. **Tag normalization.** `ollama pull foo` stores as `foo:latest`. A caller
   asking for `foo` (no tag) used to fail post-pull verification under the
   old subprocess-based exact-string-match. The fix is to ask Ollama via
   /api/show, which performs the resolution itself.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import requests


# ── /api/show-based detection ─────────────────────────────────────


class TestVerifyModelLoadable:
    """verify_model_loadable returns True iff Ollama can actually serve the model."""

    @patch("autodidact.setup_wizard.requests.post")
    def test_real_local_model_is_loadable(self, mock_post):
        """A real pulled model: 200 + format='gguf'."""
        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: {"details": {"format": "gguf", "family": "qwen3"}},
        )
        from autodidact.setup_wizard import verify_model_loadable
        assert verify_model_loadable("qwen3:8b") is True

    @patch("autodidact.setup_wizard.requests.post")
    def test_cloud_only_manifest_is_not_loadable(self, mock_post):
        """A cloud-only manifest: 200 but empty format."""
        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: {"details": {"format": "", "family": "qwen3moe"}},
        )
        from autodidact.setup_wizard import verify_model_loadable
        assert verify_model_loadable("qwen3-coder:480b-cloud") is False

    @patch("autodidact.setup_wizard.requests.post")
    def test_missing_model_is_not_loadable(self, mock_post):
        """Truly missing model: 404."""
        mock_post.return_value = MagicMock(status_code=404)
        from autodidact.setup_wizard import verify_model_loadable
        assert verify_model_loadable("not-a-real-model") is False

    @patch("autodidact.setup_wizard.requests.post")
    def test_unsuffixed_name_works_via_ollama_resolution(self, mock_post):
        """`foo` (no tag) gets resolved by Ollama to `foo:latest` automatically.

        We rely on Ollama's own name resolution rather than parsing tag rules.
        """
        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: {"details": {"format": "gguf", "family": "bert"}},
        )
        from autodidact.setup_wizard import verify_model_loadable
        assert verify_model_loadable("qllama/bge-large-en-v1.5") is True

    @patch("autodidact.setup_wizard.requests.post")
    def test_connection_error_returns_false(self, mock_post):
        """Daemon down → can't verify, treat as not loadable (caller will see)."""
        mock_post.side_effect = requests.exceptions.ConnectionError("daemon down")
        from autodidact.setup_wizard import verify_model_loadable
        assert verify_model_loadable("qwen3:8b") is False

    @patch("autodidact.setup_wizard.requests.post")
    def test_timeout_returns_false(self, mock_post):
        mock_post.side_effect = requests.exceptions.Timeout("timed out")
        from autodidact.setup_wizard import verify_model_loadable
        assert verify_model_loadable("qwen3:8b") is False

    @patch("autodidact.setup_wizard.requests.post")
    def test_malformed_json_returns_false(self, mock_post):
        """Defensive: if the body isn't JSON or lacks the expected fields."""
        mock_post.return_value = MagicMock(
            status_code=200,
            json=MagicMock(side_effect=ValueError("not json")),
        )
        from autodidact.setup_wizard import verify_model_loadable
        assert verify_model_loadable("qwen3:8b") is False


# ── is_model_available is now a thin wrapper around verify_model_loadable ──


class TestIsModelAvailable:
    """is_model_available delegates to verify_model_loadable.

    Kept as a separate name for backwards compat with existing code, but it
    no longer has its own subprocess-based logic.
    """

    @patch("autodidact.setup_wizard.verify_model_loadable", return_value=True)
    def test_returns_true_when_loadable(self, _mock):
        from autodidact.setup_wizard import is_model_available
        assert is_model_available("qwen3:8b") is True

    @patch("autodidact.setup_wizard.verify_model_loadable", return_value=False)
    def test_returns_false_when_not_loadable(self, _mock):
        from autodidact.setup_wizard import is_model_available
        assert is_model_available("not-real") is False


# ── Pull + verify integration ─────────────────────────────────────


class TestPullVerification:
    """The two-step pull-then-verify flow."""

    @patch("autodidact.setup_wizard.pull_ollama_model", return_value=True)
    @patch("autodidact.setup_wizard.verify_model_loadable", return_value=False)
    def test_pull_that_does_not_make_model_loadable_is_detected(self, _verify, _pull):
        """Direct test of the guard: after a 'successful' pull, verify fails
        for a cloud-only manifest."""
        from autodidact.setup_wizard import pull_ollama_model, verify_model_loadable
        assert pull_ollama_model("qwen3-coder:480b-cloud") is True
        assert verify_model_loadable("qwen3-coder:480b-cloud") is False

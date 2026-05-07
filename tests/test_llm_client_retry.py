"""Tests for the Bedrock throttle retry logic added after EXP-003 post-mortem.

EXP-003 observed 69 consecutive `ServiceUnavailableException` failures in a
single ~12-second burst at the tail of a main experiment. The old client only
retried `ConnectionError` / `Timeout`, so throttle-class errors propagated and
the affected eval rows got recorded as failure placeholders.

These tests verify that the updated retry path:
  1. detects throttle-class ClientErrors via their Error.Code,
  2. retries them up to `max_retries` times,
  3. does NOT retry permanent auth / validation errors.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from autodidact.llm_client import (
    LLMClient,
    LLMClientError,
    LLMConfig,
)


def _mk_client_error(code: str) -> Exception:
    """Build a botocore ClientError-lookalike with Error.Code == `code`."""
    try:
        from botocore.exceptions import ClientError
    except ImportError:
        pytest.skip("botocore not installed; bedrock retry tests skipped")
    return ClientError(
        error_response={"Error": {"Code": code, "Message": f"mocked {code}"}},
        operation_name="Converse",
    )


@pytest.fixture
def bedrock_client() -> LLMClient:
    return LLMClient(LLMConfig(provider="bedrock", model="anthropic.claude-dummy", region="us-west-2", max_retries=3))


class TestBedrockThrottleRetry:
    def test_service_unavailable_is_retryable(self, bedrock_client, monkeypatch):
        """The exact error class that caused 69 EXP-003 failures must now retry."""
        call_count = {"n": 0}

        def fake_converse(**kwargs):
            call_count["n"] += 1
            if call_count["n"] < 3:
                raise _mk_client_error("ServiceUnavailableException")
            return {
                "output": {"message": {"content": [{"text": "ok"}]}},
                "usage": {"inputTokens": 10, "outputTokens": 2},
            }

        fake_client = MagicMock()
        fake_client.converse.side_effect = fake_converse
        monkeypatch.setattr(bedrock_client, "_get_bedrock_client", lambda: fake_client)

        # Also stub out time.sleep so the test doesn't actually wait 1+2 seconds.
        monkeypatch.setattr("autodidact.llm_client.time.sleep", lambda _s: None)

        from autodidact.llm_client import ChatMessage
        resp = bedrock_client.chat([ChatMessage(role="user", content="hi")])
        assert resp.content == "ok"
        assert call_count["n"] == 3  # failed twice, succeeded on third

    def test_throttling_exception_is_retryable(self, bedrock_client, monkeypatch):
        """ThrottlingException is the more common throttle code; also retryable."""
        call_count = {"n": 0}

        def fake_converse(**kwargs):
            call_count["n"] += 1
            if call_count["n"] < 2:
                raise _mk_client_error("ThrottlingException")
            return {
                "output": {"message": {"content": [{"text": "throttle-recovered"}]}},
                "usage": {},
            }

        fake_client = MagicMock()
        fake_client.converse.side_effect = fake_converse
        monkeypatch.setattr(bedrock_client, "_get_bedrock_client", lambda: fake_client)
        monkeypatch.setattr("autodidact.llm_client.time.sleep", lambda _s: None)

        from autodidact.llm_client import ChatMessage
        resp = bedrock_client.chat([ChatMessage(role="user", content="hi")])
        assert resp.content == "throttle-recovered"
        assert call_count["n"] == 2

    def test_validation_exception_is_not_retryable(self, bedrock_client, monkeypatch):
        """Permanent errors must fail fast, not burn 6 retries."""
        call_count = {"n": 0}

        def fake_converse(**kwargs):
            call_count["n"] += 1
            raise _mk_client_error("ValidationException")

        fake_client = MagicMock()
        fake_client.converse.side_effect = fake_converse
        monkeypatch.setattr(bedrock_client, "_get_bedrock_client", lambda: fake_client)
        monkeypatch.setattr("autodidact.llm_client.time.sleep", lambda _s: None)

        from autodidact.llm_client import ChatMessage
        with pytest.raises(LLMClientError, match="ValidationException"):
            bedrock_client.chat([ChatMessage(role="user", content="hi")])
        assert call_count["n"] == 1  # no retry

    def test_persistent_throttle_gives_up_after_max_retries(self, bedrock_client, monkeypatch):
        """If throttling never clears, we eventually raise LLMClientError."""
        call_count = {"n": 0}

        def fake_converse(**kwargs):
            call_count["n"] += 1
            raise _mk_client_error("ServiceUnavailableException")

        fake_client = MagicMock()
        fake_client.converse.side_effect = fake_converse
        monkeypatch.setattr(bedrock_client, "_get_bedrock_client", lambda: fake_client)
        monkeypatch.setattr("autodidact.llm_client.time.sleep", lambda _s: None)

        from autodidact.llm_client import ChatMessage
        with pytest.raises(LLMClientError, match="Transient failure"):
            bedrock_client.chat([ChatMessage(role="user", content="hi")])
        assert call_count["n"] == 3  # fixture's max_retries

"""Tests for step-level routing thresholds (executor tiered routing)."""

from __future__ import annotations

import pytest

from autodidact.routing.step_router import (
    FixedThresholdRouter,
    StepRouter,
    Threshold,
    Tier,
)


class TestThreshold:
    def test_tier_high_confidence_is_local(self):
        t = Threshold(high=-0.3, low=-1.0)
        assert t.tier_for(-0.1) is Tier.LOCAL

    def test_tier_low_confidence_is_cloud(self):
        t = Threshold(high=-0.3, low=-1.0)
        assert t.tier_for(-2.0) is Tier.CLOUD

    def test_tier_middle_band_is_verify(self):
        t = Threshold(high=-0.3, low=-1.0)
        assert t.tier_for(-0.6) is Tier.VERIFY

    def test_boundaries_are_exclusive_at_high_inclusive_of_band(self):
        t = Threshold(high=-0.3, low=-1.0)
        # exactly on an edge falls in the middle band (not strictly above/below)
        assert t.tier_for(-0.3) is Tier.VERIFY
        assert t.tier_for(-1.0) is Tier.VERIFY

    def test_none_logprob_routes_to_verify(self):
        # Backends without logprob support (Bedrock) return None; don't blindly
        # trust or escalate — verify.
        t = Threshold(high=-0.3, low=-1.0)
        assert t.tier_for(None) is Tier.VERIFY

    def test_inverted_band_rejected(self):
        with pytest.raises(ValueError, match="must be >= low"):
            Threshold(high=-1.0, low=-0.3)


class TestFixedThresholdRouter:
    def test_satisfies_protocol(self):
        assert isinstance(FixedThresholdRouter(), StepRouter)

    def test_default_threshold_for_any_category(self):
        r = FixedThresholdRouter()
        assert r.get_threshold("code") == r.get_threshold("terminal")

    def test_per_category_override(self):
        strict = Threshold(high=-0.1, low=-0.5)
        r = FixedThresholdRouter(per_category={"terminal": strict})
        assert r.get_threshold("terminal") == strict
        # unspecified category falls back to default
        assert r.get_threshold("code") != strict

    def test_record_outcome_is_noop(self):
        r = FixedThresholdRouter()
        # Must not raise; fixed router simply ignores outcomes.
        assert r.record_outcome("code", Tier.LOCAL, success=True) is None

"""Tests for hardware detection used to pick a sensible default local model.

The heuristic:
  - Apple Silicon + 32GB+ unified memory -> 'high'   (qwen3:8b)
  - NVIDIA GPU + 16GB+ VRAM             -> 'high'
  - Any system, 16GB+ RAM               -> 'medium'  (qwen3:4b)
  - 8-16GB RAM                          -> 'low'     (qwen3:1.7b)
  - <8GB                                -> 'minimal' (qwen3:0.6b)

Detection must degrade gracefully when psutil or subprocess calls fail —
we never want hardware probing to block the wizard.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from autodidact.hardware import (
    HardwareProfile,
    detect_hardware,
    recommended_local_model,
)


# ── HardwareProfile data class ────────────────────────────────────


class TestHardwareProfile:

    def test_has_required_fields(self):
        p = HardwareProfile(
            ram_gb=16.0,
            is_apple_silicon=True,
            has_nvidia=False,
            vram_gb=None,
            tier="medium",
        )
        assert p.ram_gb == 16.0
        assert p.is_apple_silicon is True
        assert p.tier == "medium"


# ── Tier classification ──────────────────────────────────────────


class TestTierClassification:
    """detect_hardware returns the right tier for typical systems."""

    @patch("autodidact.hardware._apple_silicon", return_value=True)
    @patch("autodidact.hardware._system_ram_gb", return_value=64.0)
    @patch("autodidact.hardware._nvidia_vram_gb", return_value=None)
    def test_apple_silicon_64gb_is_high(self, _nv, _ram, _as):
        p = detect_hardware()
        assert p.tier == "high"
        assert p.is_apple_silicon is True

    @patch("autodidact.hardware._apple_silicon", return_value=True)
    @patch("autodidact.hardware._system_ram_gb", return_value=16.0)
    @patch("autodidact.hardware._nvidia_vram_gb", return_value=None)
    def test_apple_silicon_16gb_is_medium(self, _nv, _ram, _as):
        p = detect_hardware()
        assert p.tier == "medium"

    @patch("autodidact.hardware._apple_silicon", return_value=False)
    @patch("autodidact.hardware._system_ram_gb", return_value=32.0)
    @patch("autodidact.hardware._nvidia_vram_gb", return_value=24.0)
    def test_nvidia_24gb_vram_is_high(self, _nv, _ram, _as):
        p = detect_hardware()
        assert p.tier == "high"
        assert p.has_nvidia is True

    @patch("autodidact.hardware._apple_silicon", return_value=False)
    @patch("autodidact.hardware._system_ram_gb", return_value=12.0)
    @patch("autodidact.hardware._nvidia_vram_gb", return_value=None)
    def test_12gb_ram_no_gpu_is_low(self, _nv, _ram, _as):
        p = detect_hardware()
        assert p.tier == "low"

    @patch("autodidact.hardware._apple_silicon", return_value=False)
    @patch("autodidact.hardware._system_ram_gb", return_value=4.0)
    @patch("autodidact.hardware._nvidia_vram_gb", return_value=None)
    def test_4gb_ram_is_minimal(self, _nv, _ram, _as):
        p = detect_hardware()
        assert p.tier == "minimal"


# ── Graceful degradation ─────────────────────────────────────────


class TestDegradation:
    """Never raise from detect_hardware(); fall back to a safe default."""

    @patch("autodidact.hardware._apple_silicon", side_effect=RuntimeError("boom"))
    @patch("autodidact.hardware._system_ram_gb", return_value=16.0)
    @patch("autodidact.hardware._nvidia_vram_gb", return_value=None)
    def test_platform_probe_failure_does_not_raise(self, _nv, _ram, _as):
        p = detect_hardware()
        assert p.tier in ("low", "medium", "high", "minimal", "unknown")

    @patch("autodidact.hardware._apple_silicon", return_value=False)
    @patch("autodidact.hardware._system_ram_gb", side_effect=RuntimeError("no psutil"))
    @patch("autodidact.hardware._nvidia_vram_gb", return_value=None)
    def test_ram_probe_failure_returns_unknown_tier(self, _nv, _ram, _as):
        p = detect_hardware()
        assert p.tier == "unknown"


# ── Model recommendation ─────────────────────────────────────────


class TestRecommendedModel:
    """recommended_local_model maps a profile to a concrete model name."""

    def test_high_tier(self):
        p = HardwareProfile(ram_gb=32, is_apple_silicon=True, has_nvidia=False, vram_gb=None, tier="high")
        assert recommended_local_model(p) == "qwen3:8b"

    def test_medium_tier(self):
        p = HardwareProfile(ram_gb=16, is_apple_silicon=False, has_nvidia=False, vram_gb=None, tier="medium")
        assert recommended_local_model(p) == "qwen3:4b"

    def test_low_tier(self):
        p = HardwareProfile(ram_gb=10, is_apple_silicon=False, has_nvidia=False, vram_gb=None, tier="low")
        assert recommended_local_model(p) == "qwen3:1.7b"

    def test_minimal_tier(self):
        p = HardwareProfile(ram_gb=4, is_apple_silicon=False, has_nvidia=False, vram_gb=None, tier="minimal")
        assert recommended_local_model(p) == "qwen3:0.6b"

    def test_unknown_tier_falls_back_to_medium(self):
        """When detection fails, pick the conservative mid-tier default (qwen3:4b)."""
        p = HardwareProfile(ram_gb=0, is_apple_silicon=False, has_nvidia=False, vram_gb=None, tier="unknown")
        assert recommended_local_model(p) == "qwen3:4b"

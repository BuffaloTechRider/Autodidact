"""Hardware detection used to pick a sensible default local model.

Heuristic only — we read RAM, detect Apple Silicon, and try nvidia-smi. None
of these probes are allowed to raise; every failure falls back to the
``unknown`` tier, which maps to the current conservative default
(``qwen2.5:7b``).

Why tiers instead of raw numbers: model choice is quantized (you either fit
or you don't), and hiding the math behind a tier makes the recommendation
self-documenting in the wizard output.
"""

from __future__ import annotations

import logging
import platform
import shutil
import subprocess
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class HardwareProfile:
    """What we managed to learn about the host."""

    ram_gb: float
    is_apple_silicon: bool
    has_nvidia: bool
    vram_gb: Optional[float]
    tier: str  # "high" | "medium" | "low" | "minimal" | "unknown"


# ── Probes ─────────────────────────────────────────────────────────


def _system_ram_gb() -> float:
    """Total physical RAM in GB. Uses psutil if available, else raises."""
    import psutil  # type: ignore

    return psutil.virtual_memory().total / 1e9


def _apple_silicon() -> bool:
    """True if running on Apple Silicon (M1/M2/M3/...)."""
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def _nvidia_vram_gb() -> Optional[float]:
    """Total NVIDIA VRAM in GB (sum across GPUs), or None if no NVIDIA GPU."""
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode != 0:
            return None
        total_mib = sum(int(line.strip()) for line in result.stdout.strip().splitlines() if line.strip())
        return total_mib / 1024.0
    except (subprocess.TimeoutExpired, ValueError, OSError):
        return None


# ── Public API ─────────────────────────────────────────────────────


def detect_hardware() -> HardwareProfile:
    """Probe the host and classify into a tier. Never raises.

    Any single probe failure degrades gracefully. If RAM probing fails
    outright (no psutil, etc.) the tier is 'unknown' and we default to
    a conservative mid-tier model in recommended_local_model.
    """
    try:
        is_apple = _apple_silicon()
    except Exception as e:
        logger.debug("Apple-silicon probe failed: %s", e)
        is_apple = False

    try:
        ram_gb = _system_ram_gb()
    except Exception as e:
        logger.debug("RAM probe failed: %s", e)
        return HardwareProfile(
            ram_gb=0.0,
            is_apple_silicon=is_apple,
            has_nvidia=False,
            vram_gb=None,
            tier="unknown",
        )

    vram_gb = None
    try:
        vram_gb = _nvidia_vram_gb()
    except Exception as e:
        logger.debug("NVIDIA probe failed: %s", e)

    has_nvidia = vram_gb is not None

    # Tier rules. Apple Silicon has unified memory so system RAM == GPU RAM.
    if has_nvidia and vram_gb and vram_gb >= 16:
        tier = "high"
    elif is_apple and ram_gb >= 32:
        tier = "high"
    elif ram_gb >= 16:
        tier = "medium"
    elif ram_gb >= 8:
        tier = "low"
    else:
        tier = "minimal"

    return HardwareProfile(
        ram_gb=ram_gb,
        is_apple_silicon=is_apple,
        has_nvidia=has_nvidia,
        vram_gb=vram_gb,
        tier=tier,
    )


# ── Recommendation ─────────────────────────────────────────────────

# Conservative choices across tiers. Qwen 3 is the current generation with
# better benchmark performance than Qwen 2.5; our routing signals are designed
# to be model-agnostic so using the newer default is the right call.
#
# Sizing was bumped DOWN one tier (May 2026) after live testing showed that
# qwen3:14b takes 30+ seconds to think on M-series Apple Silicon for casual
# queries. Smaller models give a far better default chat UX. Users who want
# the bigger model are one pick away in the curated list (cli._LOCAL_MODEL_CHOICES).
_TIER_MODELS = {
    "high":    "qwen3:8b",     # was qwen3:14b — 14b too slow on M-series
    "medium":  "qwen3:4b",     # was qwen3:8b
    "low":     "qwen3:1.7b",   # was qwen3:4b
    "minimal": "qwen3:0.6b",   # unchanged
    "unknown": "qwen3:4b",     # conservative mid-tier fallback (was qwen3:8b)
}


def recommended_local_model(profile: HardwareProfile) -> str:
    """Return a default local model name for the given profile."""
    return _TIER_MODELS.get(profile.tier, "qwen3:4b")

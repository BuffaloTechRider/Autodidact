"""Thompson Sampling Calibration Benchmark.

Measures how well Thompson Sampling learns to route correctly compared to
fixed-threshold and random baselines.
"""

from __future__ import annotations

import json
import os
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta as beta_dist

from autodidact.database import init_database
from autodidact.types import AutodidactConfig


# ── Difficulty simulation ────────────────────────────────────────────

DIFFICULTY_PROFILES = {
    "easy": {"local_success_rate": 0.9, "signal_mean": 0.85},
    "medium": {"local_success_rate": 0.5, "signal_mean": 0.55},
    "hard": {"local_success_rate": 0.1, "signal_mean": 0.25},
}


def simulate_query(difficulty: str, rng: np.random.RandomState) -> dict:
    """Simulate a query with known difficulty and signal values."""
    profile = DIFFICULTY_PROFILES[difficulty]
    signals = {
        "knowledge_similarity": np.clip(rng.normal(profile["signal_mean"], 0.15), 0, 1),
        "logprob_uncertainty": np.clip(rng.normal(profile["signal_mean"], 0.1), 0, 1),
        "self_consistency": np.clip(rng.normal(profile["signal_mean"], 0.12), 0, 1),
        "query_classification": np.clip(rng.normal(0.5, 0.1), 0, 1),
    }
    local_succeeds = rng.random() < profile["local_success_rate"]
    return {"difficulty": difficulty, "signals": signals, "local_succeeds": local_succeeds}


# ── Routing strategies ───────────────────────────────────────────────

class ThompsonRouter:
    """Thompson Sampling router that learns from outcomes."""

    def __init__(self) -> None:
        self.params = {
            name: {"alpha": 1.0, "beta_param": 1.0}
            for name in ["knowledge_similarity", "logprob_uncertainty",
                         "self_consistency", "query_classification"]
        }

    def route(self, signals: dict[str, float], rng: np.random.RandomState) -> tuple[str, float]:
        weighted_sum = 0.0
        weight_total = 0.0
        for name, value in signals.items():
            p = self.params[name]
            theta = float(beta_dist.rvs(p["alpha"], p["beta_param"], random_state=rng))
            weighted_sum += theta * value
            weight_total += theta
        fused = weighted_sum / weight_total if weight_total > 0 else 0.5
        route = "LOCAL" if fused >= 0.7 else "CLOUD"
        return route, fused

    def update(self, signals: dict[str, float], success: bool) -> None:
        for name in signals:
            if success:
                self.params[name]["alpha"] += 1
            else:
                self.params[name]["beta_param"] += 1


class FixedThresholdRouter:
    """Fixed threshold on average signal value."""

    def route(self, signals: dict[str, float]) -> tuple[str, float]:
        avg = sum(signals.values()) / len(signals)
        route = "LOCAL" if avg >= 0.7 else "CLOUD"
        return route, avg


class RandomRouter:
    """Random routing baseline."""

    def route(self, rng: np.random.RandomState) -> tuple[str, float]:
        score = float(rng.random())
        route = "LOCAL" if score >= 0.5 else "CLOUD"
        return route, score


# ── Benchmark runner ─────────────────────────────────────────────────

def run_thompson_calibration(
    output_dir: str = "results",
    n_queries: int = 500,
    seed: int = 42,
) -> dict:
    """Run the Thompson Sampling calibration benchmark."""
    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.RandomState(seed)
    random.seed(seed)

    thompson = ThompsonRouter()
    fixed = FixedThresholdRouter()
    rand = RandomRouter()

    difficulties = ["easy", "medium", "hard"]
    results = {"thompson": [], "fixed": [], "random": []}

    # Tracking for calibration
    thompson_correct = 0
    fixed_correct = 0
    random_correct = 0

    for i in range(n_queries):
        difficulty = random.choice(difficulties)
        q = simulate_query(difficulty, rng)

        # Thompson Sampling
        ts_route, ts_score = thompson.route(q["signals"], rng)
        ts_correct_decision = (
            (ts_route == "LOCAL" and q["local_succeeds"])
            or (ts_route == "CLOUD" and not q["local_succeeds"])
        )
        if ts_correct_decision:
            thompson_correct += 1
        # Update Thompson params based on whether LOCAL would have succeeded
        if ts_route == "LOCAL":
            thompson.update(q["signals"], q["local_succeeds"])

        # Fixed threshold
        ft_route, ft_score = fixed.route(q["signals"])
        ft_correct = (
            (ft_route == "LOCAL" and q["local_succeeds"])
            or (ft_route == "CLOUD" and not q["local_succeeds"])
        )
        if ft_correct:
            fixed_correct += 1

        # Random
        r_route, r_score = rand.route(rng)
        r_correct = (
            (r_route == "LOCAL" and q["local_succeeds"])
            or (r_route == "CLOUD" and not q["local_succeeds"])
        )
        if r_correct:
            random_correct += 1

        results["thompson"].append({
            "query_index": i,
            "difficulty": difficulty,
            "route": ts_route,
            "fused_score": round(ts_score, 4),
            "correct": ts_correct_decision,
            "accuracy": round(thompson_correct / (i + 1), 4),
        })
        results["fixed"].append({
            "query_index": i,
            "difficulty": difficulty,
            "route": ft_route,
            "fused_score": round(ft_score, 4),
            "correct": ft_correct,
            "accuracy": round(fixed_correct / (i + 1), 4),
        })
        results["random"].append({
            "query_index": i,
            "difficulty": difficulty,
            "route": r_route,
            "fused_score": round(r_score, 4),
            "correct": r_correct,
            "accuracy": round(random_correct / (i + 1), 4),
        })

    output = {
        "benchmark": "thompson_calibration",
        "n_queries": n_queries,
        "final_accuracy": {
            "thompson": results["thompson"][-1]["accuracy"],
            "fixed": results["fixed"][-1]["accuracy"],
            "random": results["random"][-1]["accuracy"],
        },
        "results": results,
    }

    json_path = os.path.join(output_dir, "thompson_calibration.json")
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)

    plot_path = os.path.join(output_dir, "thompson_calibration.png")
    _plot_calibration(results, plot_path)

    print(f"Thompson calibration: {n_queries} queries")
    for method, acc in output["final_accuracy"].items():
        print(f"  {method}: {acc:.2%}")
    print(f"  Results: {json_path}")
    print(f"  Plot: {plot_path}")

    return output


def _plot_calibration(results: dict, path: str) -> None:
    """Plot routing accuracy over time for all three methods."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for method, color, label in [
        ("thompson", "#4CAF50", "Thompson Sampling"),
        ("fixed", "#FF9800", "Fixed Threshold"),
        ("random", "#9E9E9E", "Random"),
    ]:
        indices = [r["query_index"] for r in results[method]]
        accuracies = [r["accuracy"] for r in results[method]]
        ax.plot(indices, accuracies, color=color, linewidth=2, label=label)

    ax.set_xlabel("Query Index", fontsize=12)
    ax.set_ylabel("Routing Accuracy", fontsize=12)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.suptitle("Thompson Sampling Calibration", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    run_thompson_calibration()

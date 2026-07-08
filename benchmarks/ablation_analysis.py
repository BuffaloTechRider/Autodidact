"""Ablation analysis for the v0.1 experiment.

Reads rows from `experiment_results` for a given run_id, computes:
- Per-signal AUROC against `local_correct` with 1000-sample bootstrap CIs
- Per-combo AUROC via simple mean fusion AND Thompson Sampling fusion
- Calibration (reliability) diagrams for the best combo and two worst signals
- ROC curve overlay comparing every approach
- A summary.json with all numbers plus the RouteLLM baseline gap

Then fills the memo template (Task 9) with concrete numbers.

Run:
    python -m benchmarks.ablation_analysis --run-id <run_id>

Requirement 6 (R6) from the v0.1 spec is satisfied here.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sqlite3
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # no GUI needed
import matplotlib.pyplot as plt
import numpy as np

from autodidact.database import init_database

logger = logging.getLogger(__name__)

# ── Combos we evaluate ─────────────────────────────────────────────

SIGNAL_COLUMNS = [
    "knowledge_similarity",
    "query_classification",
    "energy_scorer",
    "grounded_self_assessment",
    "logprob_uncertainty",
    "self_consistency",
]

ALL_SIX = [
    "knowledge_similarity",
    "query_classification",
    "energy_scorer",
    "grounded_self_assessment",
    "logprob_uncertainty",
    "self_consistency",
]

COMBOS: dict[str, list[str]] = {
    "energy_only": ["energy_scorer"],
    "knowledge_similarity_only": ["knowledge_similarity"],
    "grounded_self_assessment_only": ["grounded_self_assessment"],
    "logprob_uncertainty_only": ["logprob_uncertainty"],
    "logprob_plus_gsa": ["logprob_uncertainty", "grounded_self_assessment"],
    "logprob_plus_knowledge": ["logprob_uncertainty", "knowledge_similarity"],
    "logprob_plus_gsa_plus_knowledge": [
        "logprob_uncertainty", "grounded_self_assessment", "knowledge_similarity",
    ],
    "energy_plus_knowledge": ["energy_scorer", "knowledge_similarity"],
    "energy_plus_knowledge_plus_gsa": [
        "energy_scorer", "knowledge_similarity", "grounded_self_assessment",
    ],
    "all_six_mean": ALL_SIX,
    "all_six_thompson": ALL_SIX,  # special-cased below
    # EXP-005: GSA variants grounded on retrieved memory (threshold-gated prompt).
    # These combos are evaluated only when a sidecar (--gsa-sidecar) is passed;
    # otherwise the signals are NaN and the combos report as unavailable.
    "gsa_v3_070_only": ["gsa_v3_070"],
    "gsa_v3_060_only": ["gsa_v3_060"],
    "logprob_plus_gsa_v3_070": ["logprob_uncertainty", "gsa_v3_070"],
    "logprob_plus_gsa_v3_060": ["logprob_uncertainty", "gsa_v3_060"],
}

BASELINE_COMBOS = ("routellm_no_memory", "routellm_plus_ks")


# ── Data loading ────────────────────────────────────────────────────

@dataclass
class ExperimentRows:
    """A run's per-query signals and labels, as aligned numpy arrays."""

    run_id: str
    n: int
    # Signal arrays in the order of SIGNAL_COLUMNS. NaN for missing (e.g. energy_scorer disabled).
    signals: dict[str, np.ndarray] = field(default_factory=dict)
    routellm_no_memory: np.ndarray = field(default_factory=lambda: np.zeros(0))
    routellm_plus_ks: np.ndarray = field(default_factory=lambda: np.zeros(0))
    local_correct: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))
    retrieval_recall_at_5: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))
    had_error: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))


def load_rows(
    conn: sqlite3.Connection,
    run_id: str,
    drop_errors: bool = True,
    gsa_sidecar_path: Optional[str] = None,
) -> ExperimentRows:
    rows = conn.execute(
        """SELECT query_index, query_id, knowledge_similarity, query_classification, energy_scorer,
                  grounded_self_assessment, logprob_uncertainty, self_consistency,
                  routellm_no_memory, routellm_plus_ks, retrieval_recall_at_5,
                  local_correct, error_info
           FROM experiment_results WHERE run_id = ?
           ORDER BY query_index""",
        (run_id,),
    ).fetchall()

    if not rows:
        raise RuntimeError(f"No experiment_results rows for run_id={run_id}")

    er = ExperimentRows(run_id=run_id, n=len(rows))
    er.had_error = np.array([r["error_info"] is not None for r in rows], dtype=bool)
    mask = ~er.had_error if drop_errors else np.ones(len(rows), dtype=bool)

    for col in SIGNAL_COLUMNS:
        vals = [r[col] for r in rows]
        arr = np.array([np.nan if v is None else float(v) for v in vals], dtype=np.float64)
        er.signals[col] = arr[mask]

    # Sidecar signals (EXP-005 style): gsa_v3_070, gsa_v3_060 from a rerun JSONL.
    # Initialize to all-NaN by default so combos that reference them never
    # KeyError; NaN will mask them out of combo scoring. If a sidecar JSONL is
    # provided below, we overwrite with real values per query_id.
    n_kept = int(mask.sum())
    for name in ("gsa_v3_070", "gsa_v3_060"):
        er.signals[name] = np.full(n_kept, np.nan, dtype=np.float64)

    sidecar_by_qid: dict[str, dict] = {}
    if gsa_sidecar_path:
        import json as _json
        with open(gsa_sidecar_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = _json.loads(line)
                sidecar_by_qid[d["query_id"]] = d
        logger.info(
            "Loaded %d GSA sidecar rows from %s",
            len(sidecar_by_qid), gsa_sidecar_path,
        )
        # Populate gsa_v3_* arrays. Threshold keys are strings like "0.70".
        for t_key, col_name in [("0.70", "gsa_v3_070"), ("0.60", "gsa_v3_060")]:
            vals = []
            for r in rows:
                sc = sidecar_by_qid.get(r["query_id"])
                if sc and t_key in sc.get("per_threshold", {}):
                    vals.append(float(sc["per_threshold"][t_key]["p_yes"]))
                else:
                    vals.append(np.nan)
            er.signals[col_name] = np.array(vals, dtype=np.float64)[mask]

    er.routellm_no_memory = np.array([float(r["routellm_no_memory"]) for r in rows])[mask]
    er.routellm_plus_ks = np.array([float(r["routellm_plus_ks"]) for r in rows])[mask]
    er.local_correct = np.array([int(r["local_correct"]) for r in rows])[mask]
    er.retrieval_recall_at_5 = np.array([int(r["retrieval_recall_at_5"]) for r in rows])[mask]

    # Drop rows from the masked-kept set where every REQUIRED signal is NaN or the label is invalid.
    # Sidecar signals (gsa_v3_*) are optional and NOT required.
    keep = np.ones(len(er.local_correct), dtype=bool)
    for col in SIGNAL_COLUMNS:
        if col == "energy_scorer":
            continue  # may legitimately be NaN
        keep &= ~np.isnan(er.signals[col])
    keep &= np.isin(er.local_correct, [0, 1])
    er.n = int(keep.sum())
    for col in list(er.signals.keys()):
        er.signals[col] = er.signals[col][keep]
    er.routellm_no_memory = er.routellm_no_memory[keep]
    er.routellm_plus_ks = er.routellm_plus_ks[keep]
    er.local_correct = er.local_correct[keep]
    er.retrieval_recall_at_5 = er.retrieval_recall_at_5[keep]
    return er


# ── AUROC + bootstrap ───────────────────────────────────────────────

def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Compute AUROC via Mann-Whitney U statistic. Handles ties correctly."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int32)
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    # Rank-based computation (handles ties correctly).
    all_scores = np.concatenate([pos, neg])
    # argsort gives ranks; use average rank for ties
    order = np.argsort(all_scores, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(all_scores) + 1)
    # Handle ties: average ranks within equal groups.
    sorted_scores = all_scores[order]
    i = 0
    while i < len(sorted_scores):
        j = i
        while j + 1 < len(sorted_scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        if j > i:
            avg = np.mean(ranks[order[i:j + 1]])
            ranks[order[i:j + 1]] = avg
        i = j + 1
    rank_sum_pos = ranks[: len(pos)].sum()
    n_pos, n_neg = len(pos), len(neg)
    u = rank_sum_pos - n_pos * (n_pos + 1) / 2
    return float(u / (n_pos * n_neg))


def bootstrap_auroc_ci(
    scores: np.ndarray,
    labels: np.ndarray,
    n_bootstrap: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Return (point, lo, hi) with (1-alpha) bootstrap CI on AUROC."""
    rng = np.random.default_rng(seed)
    n = len(scores)
    point = auroc(scores, labels)
    if np.isnan(point) or n == 0:
        return point, float("nan"), float("nan")
    samples = np.empty(n_bootstrap, dtype=np.float64)
    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        samples[b] = auroc(scores[idx], labels[idx])
    lo = float(np.nanpercentile(samples, 100 * alpha / 2))
    hi = float(np.nanpercentile(samples, 100 * (1 - alpha / 2)))
    return point, lo, hi


def paired_bootstrap_delta_ci(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    labels: np.ndarray,
    n_bootstrap: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict:
    """Return a paired bootstrap CI on (AUROC_a - AUROC_b).

    Paired means the same bootstrap resample index is applied to both score
    arrays. This is the statistically right way to compare two scorers on the
    same dataset — it removes the shared sampling noise and produces tighter
    CIs than comparing two independent bootstrap CIs.

    Returns a dict with keys:
        point: AUROC_a - AUROC_b (observed difference)
        lo, hi: (1-alpha) bootstrap CI on the difference
        p_value_approx: approximate one-sided p-value (fraction of bootstrap
                        resamples where delta <= 0). Useful but not rigorous.
        significant: True if 0 is not in the (1-alpha) CI.
    """
    scores_a = np.asarray(scores_a, dtype=np.float64)
    scores_b = np.asarray(scores_b, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int32)
    n = len(labels)
    if len(scores_a) != n or len(scores_b) != n:
        raise ValueError("scores_a, scores_b, labels must all have the same length")

    point = auroc(scores_a, labels) - auroc(scores_b, labels)
    if np.isnan(point):
        return {"point": point, "lo": float("nan"), "hi": float("nan"),
                "p_value_approx": float("nan"), "significant": False}

    rng = np.random.default_rng(seed)
    deltas = np.empty(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        deltas[i] = auroc(scores_a[idx], labels[idx]) - auroc(scores_b[idx], labels[idx])
    lo = float(np.nanpercentile(deltas, 100 * alpha / 2))
    hi = float(np.nanpercentile(deltas, 100 * (1 - alpha / 2)))
    # One-sided p-value: how often did B beat A in the bootstrap?
    p = float(np.mean(deltas <= 0))
    return {
        "point": float(point),
        "lo": lo,
        "hi": hi,
        "p_value_approx": p,
        "significant": lo > 0 or hi < 0,  # CI excludes 0
    }


# ── Fusion ─────────────────────────────────────────────────────────

def fuse_mean(signal_arrays: list[np.ndarray]) -> np.ndarray:
    """Simple arithmetic mean of signals. NaNs ignored per-position."""
    stacked = np.stack(signal_arrays, axis=0)  # (k, n)
    with np.errstate(invalid="ignore"):
        return np.nanmean(stacked, axis=0)


def fuse_thompson(
    conn: sqlite3.Connection,
    signal_names: list[str],
    signal_arrays: list[np.ndarray],
    seed: int = 0,
) -> np.ndarray:
    """Thompson-sampling-weighted fusion.

    Uses the current Beta(alpha, beta) parameters from `thompson_params` for each
    signal. For each query we sample a theta per signal and compute a weighted
    average. We sample once per query (per the existing ConfidenceEvaluator math)
    but use a fixed numpy Generator seed so analysis is reproducible.
    """
    params: dict[str, tuple[float, float]] = {}
    rows = conn.execute(
        "SELECT signal_name, alpha, beta_param FROM thompson_params"
    ).fetchall()
    for r in rows:
        params[r["signal_name"]] = (float(r["alpha"]), float(r["beta_param"]))

    rng = np.random.default_rng(seed)
    n = len(signal_arrays[0])
    fused = np.zeros(n, dtype=np.float64)
    for i in range(n):
        weighted_sum = 0.0
        weight_total = 0.0
        for name, arr in zip(signal_names, signal_arrays):
            value = arr[i]
            if np.isnan(value):
                continue
            a, b = params.get(name, (1.0, 1.0))
            theta = float(rng.beta(a, b))
            weighted_sum += theta * value
            weight_total += theta
        fused[i] = weighted_sum / weight_total if weight_total > 0 else 0.0
    return fused


# ── Analysis pipeline ──────────────────────────────────────────────

@dataclass
class CombinedResult:
    name: str
    auroc: float
    auroc_lo: float
    auroc_hi: float
    n: int


def compute_all(
    rows: ExperimentRows,
    conn: sqlite3.Connection,
    bootstrap_seed: int = 0,
) -> dict:
    """Compute per-signal and per-combo AUROC + CIs, return a dict."""
    out: dict = {
        "run_id": rows.run_id,
        "n_total": rows.n,
        "n_local_correct": int(rows.local_correct.sum()),
        "per_signal": {},
        "per_combo": {},
        "baselines": {},
        "retrieval": {},
    }

    labels = rows.local_correct

    # Per-signal AUROC
    for col in SIGNAL_COLUMNS:
        arr = rows.signals[col]
        valid = ~np.isnan(arr)
        if valid.sum() < 10:
            out["per_signal"][col] = {"auroc": None, "ci": None, "n": int(valid.sum())}
            continue
        point, lo, hi = bootstrap_auroc_ci(arr[valid], labels[valid], seed=bootstrap_seed)
        out["per_signal"][col] = {"auroc": point, "ci": [lo, hi], "n": int(valid.sum())}

    # Baselines
    for bname in BASELINE_COMBOS:
        arr = getattr(rows, bname)
        point, lo, hi = bootstrap_auroc_ci(arr, labels, seed=bootstrap_seed)
        out["baselines"][bname] = {"auroc": point, "ci": [lo, hi], "n": int(len(arr))}

    # Combos
    for combo_name, signals in COMBOS.items():
        signal_arrays = [rows.signals[s] for s in signals]
        # Drop rows where ANY required signal is NaN (except energy_scorer, which nanmean handles).
        mask = np.ones(len(labels), dtype=bool)
        for s, arr in zip(signals, signal_arrays):
            if s == "energy_scorer":
                continue
            mask &= ~np.isnan(arr)
        if mask.sum() < 10:
            out["per_combo"][combo_name] = {
                "mean": {"auroc": None, "ci": None, "n": int(mask.sum())},
                "thompson": {"auroc": None, "ci": None, "n": int(mask.sum())},
            }
            continue
        # If the combo only contains energy_scorer and it's entirely NaN, skip.
        if all(s == "energy_scorer" and np.isnan(arr).all() for s, arr in zip(signals, signal_arrays)):
            out["per_combo"][combo_name] = {
                "mean": {"auroc": None, "ci": None, "n": int(mask.sum()), "reason": "energy_scorer disabled"},
                "thompson": {"auroc": None, "ci": None, "n": int(mask.sum()), "reason": "energy_scorer disabled"},
            }
            continue
        signal_arrays_masked = [arr[mask] for arr in signal_arrays]
        labels_masked = labels[mask]

        fused_mean = fuse_mean(signal_arrays_masked)
        fused_thompson = fuse_thompson(conn, signals, signal_arrays_masked, seed=bootstrap_seed)

        mean_point, mean_lo, mean_hi = bootstrap_auroc_ci(fused_mean, labels_masked, seed=bootstrap_seed)
        th_point, th_lo, th_hi = bootstrap_auroc_ci(fused_thompson, labels_masked, seed=bootstrap_seed)
        out["per_combo"][combo_name] = {
            "mean": {"auroc": mean_point, "ci": [mean_lo, mean_hi], "n": int(mask.sum())},
            "thompson": {"auroc": th_point, "ci": [th_lo, th_hi], "n": int(mask.sum())},
        }

    # Retrieval stratification
    if rows.retrieval_recall_at_5.sum() > 0:
        good_mask = rows.retrieval_recall_at_5 == 1
        bad_mask = rows.retrieval_recall_at_5 == 0
        best_combo_name, best_auroc = _best_combo_from_out(out)
        if best_combo_name is not None and " (" in best_combo_name:
            # best_combo_name looks like "all_six_thompson (mean)"; split on " (" gives ["all_six_thompson", "mean)"]
            combo_key, variant = best_combo_name.split(" (")
            variant = variant.rstrip(")")
            combo_signals = COMBOS[combo_key]
            signal_arrays = [rows.signals[s] for s in combo_signals]
            for submask, label in [(good_mask, "good_retrieval"), (bad_mask, "bad_retrieval")]:
                if submask.sum() < 10:
                    continue
                sub_arrays = [arr[submask] for arr in signal_arrays]
                if variant == "thompson":
                    fused = fuse_thompson(conn, combo_signals, sub_arrays, seed=bootstrap_seed)
                else:
                    fused = fuse_mean(sub_arrays)
                point, lo, hi = bootstrap_auroc_ci(fused, labels[submask], seed=bootstrap_seed)
                out["retrieval"][label] = {
                    "combo": best_combo_name,
                    "auroc": point, "ci": [lo, hi],
                    "n": int(submask.sum()),
                }

    # Summary: headline numbers
    best_combo_name, best_auroc = _best_combo_from_out(out)
    out["headline"] = {
        "best_combo": best_combo_name,
        "best_auroc": best_auroc,
        "routellm_no_memory_auroc": out["baselines"]["routellm_no_memory"]["auroc"],
        "routellm_plus_ks_auroc": out["baselines"]["routellm_plus_ks"]["auroc"],
        "gap_vs_routellm_plus_ks": (
            (best_auroc - out["baselines"]["routellm_plus_ks"]["auroc"])
            if best_auroc is not None and out["baselines"]["routellm_plus_ks"]["auroc"] is not None
            else None
        ),
    }

    # ── Paired bootstrap deltas (statistically proper comparisons) ──
    out["paired_deltas"] = {}
    if best_combo_name is not None:
        # Reconstruct the fused scores for the best combo so we can pair-bootstrap against baselines.
        best_name, variant = best_combo_name.split(" (")
        variant = variant.rstrip(")")
        best_signals = COMBOS[best_name]
        best_signal_arrays = [rows.signals[s] for s in best_signals]
        best_mask = np.ones(len(labels), dtype=bool)
        for s, arr in zip(best_signals, best_signal_arrays):
            if s == "energy_scorer":
                continue
            best_mask &= ~np.isnan(arr)
        if best_mask.sum() >= 10:
            best_arrays_masked = [arr[best_mask] for arr in best_signal_arrays]
            if variant == "thompson":
                best_fused = fuse_thompson(conn, best_signals, best_arrays_masked, seed=bootstrap_seed)
            else:
                best_fused = fuse_mean(best_arrays_masked)

            # Best combo vs routellm_plus_ks (the key paper-worthiness check)
            out["paired_deltas"]["best_vs_routellm_plus_ks"] = paired_bootstrap_delta_ci(
                best_fused, rows.routellm_plus_ks[best_mask], labels[best_mask],
                seed=bootstrap_seed,
            )
            # Best combo vs routellm_no_memory (sanity check)
            out["paired_deltas"]["best_vs_routellm_no_memory"] = paired_bootstrap_delta_ci(
                best_fused, rows.routellm_no_memory[best_mask], labels[best_mask],
                seed=bootstrap_seed,
            )

        # Simplicity check: energy_plus_knowledge_plus_gsa vs all_six_thompson
        simple_name = "energy_plus_knowledge_plus_gsa"
        full_name = "all_six_thompson"
        simple_signals = COMBOS[simple_name]
        full_signals = COMBOS[full_name]
        simple_arrays = [rows.signals[s] for s in simple_signals]
        full_arrays = [rows.signals[s] for s in full_signals]
        # Rows where both combos are computable (union of non-NaN requirements)
        both_mask = np.ones(len(labels), dtype=bool)
        for s, arr in zip(simple_signals + full_signals, simple_arrays + full_arrays):
            if s == "energy_scorer":
                continue
            both_mask &= ~np.isnan(arr)
        if both_mask.sum() >= 10:
            simple_fused = fuse_thompson(
                conn, simple_signals, [arr[both_mask] for arr in simple_arrays], seed=bootstrap_seed,
            )
            full_fused = fuse_thompson(
                conn, full_signals, [arr[both_mask] for arr in full_arrays], seed=bootstrap_seed,
            )
            # Negative delta means full beats simple (we subtract full - simple)
            out["paired_deltas"]["full_vs_simple"] = paired_bootstrap_delta_ci(
                full_fused, simple_fused, labels[both_mask],
                seed=bootstrap_seed,
            )

    return out


def _best_combo_from_out(out: dict) -> tuple[Optional[str], Optional[float]]:
    """Pick the best (non-baseline) combo AUROC across mean and thompson variants."""
    best_name, best_val = None, -1.0
    for name, info in out["per_combo"].items():
        for variant_name, variant_info in info.items():
            v = variant_info["auroc"]
            if v is None:
                continue
            if v > best_val:
                best_val = v
                best_name = f"{name} ({variant_name})"
    return (best_name, best_val) if best_name is not None else (None, None)


# ── Plotting ────────────────────────────────────────────────────────

def plot_roc_overlay(
    rows: ExperimentRows, conn: sqlite3.Connection, out_path: str, bootstrap_seed: int
) -> None:
    """Overlay ROC curves for every combo and both baselines."""
    labels = rows.local_correct

    curves: list[tuple[str, np.ndarray, np.ndarray, float]] = []

    # Combos
    for combo_name, signals in COMBOS.items():
        signal_arrays = [rows.signals[s] for s in signals]
        mask = np.ones(len(labels), dtype=bool)
        for s, arr in zip(signals, signal_arrays):
            if s == "energy_scorer":
                continue
            mask &= ~np.isnan(arr)
        if mask.sum() < 10:
            continue
        arrs_m = [arr[mask] for arr in signal_arrays]
        fused = (
            fuse_thompson(conn, signals, arrs_m, seed=bootstrap_seed)
            if "thompson" in combo_name else fuse_mean(arrs_m)
        )
        fpr, tpr = _roc_curve(fused, labels[mask])
        curves.append((combo_name, fpr, tpr, auroc(fused, labels[mask])))

    # Baselines
    for bname in BASELINE_COMBOS:
        arr = getattr(rows, bname)
        fpr, tpr = _roc_curve(arr, labels)
        curves.append((bname, fpr, tpr, auroc(arr, labels)))

    plt.figure(figsize=(8, 7))
    for name, fpr, tpr, a in sorted(curves, key=lambda c: -c[3]):
        plt.plot(fpr, tpr, label=f"{name} ({a:.3f})")
    plt.plot([0, 1], [0, 1], "k--", linewidth=0.5, alpha=0.5)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC overlay: confidence signals vs. RouteLLM baselines (run_id={rows.run_id})")
    plt.legend(loc="lower right", fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_calibration(
    scores: np.ndarray, labels: np.ndarray, out_path: str, title: str, n_bins: int = 10
) -> None:
    """Reliability diagram with per-bin empirical correct rate."""
    bins = np.linspace(0, 1, n_bins + 1)
    midpoints = (bins[:-1] + bins[1:]) / 2
    bin_idx = np.clip(np.digitize(scores, bins) - 1, 0, n_bins - 1)
    empirical = np.array([
        labels[bin_idx == i].mean() if (bin_idx == i).any() else np.nan
        for i in range(n_bins)
    ])
    counts = np.array([(bin_idx == i).sum() for i in range(n_bins)])

    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="perfect calibration")
    # Bar widths scaled to bin population
    for i in range(n_bins):
        if counts[i] == 0 or np.isnan(empirical[i]):
            continue
        plt.bar(midpoints[i], empirical[i], width=0.8 / n_bins,
                edgecolor="black", alpha=0.6)
    plt.plot(midpoints, empirical, "o-", label="observed")
    plt.xlabel("Predicted probability")
    plt.ylabel("Empirical correct rate")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _roc_curve(scores: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(-scores, kind="mergesort")
    labels_sorted = labels[order]
    tp = np.cumsum(labels_sorted == 1)
    fp = np.cumsum(labels_sorted == 0)
    total_pos = max(1, (labels == 1).sum())
    total_neg = max(1, (labels == 0).sum())
    tpr = tp / total_pos
    fpr = fp / total_neg
    # Prepend (0,0)
    return np.concatenate([[0.0], fpr]), np.concatenate([[0.0], tpr])


# ── Main ────────────────────────────────────────────────────────────

def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Autodidact v0.1 ablation analysis")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--db-path", default="autodidact_experiment.db")
    parser.add_argument("--output-dir", default="results/experiment",
                        help="Parent directory; artifacts land in <output-dir>/<run_id>/")
    parser.add_argument("--flat-output", action="store_true",
                        help="Write artifacts directly to --output-dir without a per-run subdir "
                             "(overwrites previous memo; use only when intentional).")
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--gsa-sidecar", default=None,
                        help="Path to a GSA rerun JSONL sidecar (produced by "
                             "benchmarks.gsa_retrieval_rerun). Joins by query_id "
                             "and enables gsa_v3_* combos. Optional.")
    args = parser.parse_args()

    # Per-run output directory unless --flat-output was passed.
    parent_dir = Path(args.output_dir)
    parent_dir.mkdir(parents=True, exist_ok=True)
    if args.flat_output:
        out_dir = parent_dir
    else:
        out_dir = parent_dir / args.run_id
        out_dir.mkdir(parents=True, exist_ok=True)
        # Update/refresh the "latest" symlink so `cat results/experiment/latest/MEMO.md` always works.
        latest_link = parent_dir / "latest"
        try:
            if latest_link.is_symlink() or latest_link.exists():
                latest_link.unlink()
            latest_link.symlink_to(args.run_id, target_is_directory=True)
        except OSError as e:
            logger.warning("Could not update 'latest' symlink: %s", e)

    conn = init_database(args.db_path)
    rows = load_rows(conn, args.run_id, gsa_sidecar_path=args.gsa_sidecar)
    if rows.n == 0:
        logger.error("No rows for run_id=%s", args.run_id)
        return 2

    result = compute_all(rows, conn, bootstrap_seed=args.bootstrap_seed)
    summary_path = out_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info("Wrote %s", summary_path)

    # Plots
    plot_roc_overlay(rows, conn, str(out_dir / "roc_overlay.png"), bootstrap_seed=args.bootstrap_seed)
    logger.info("Wrote %s/roc_overlay.png", out_dir)

    # Calibration for best combo
    best_name, _ = _best_combo_from_out(result)
    if best_name is not None:
        combo_key, variant = best_name.split(" (")
        variant = variant.rstrip(")")
        signals = COMBOS[combo_key]
        signal_arrays = [rows.signals[s] for s in signals]
        mask = np.ones(rows.n, dtype=bool)
        for s, arr in zip(signals, signal_arrays):
            if s == "energy_scorer":
                continue
            mask &= ~np.isnan(arr)
        arrs_m = [arr[mask] for arr in signal_arrays]
        fused = (
            fuse_thompson(conn, signals, arrs_m, seed=args.bootstrap_seed)
            if variant == "thompson" else fuse_mean(arrs_m)
        )
        plot_calibration(
            fused, rows.local_correct[mask],
            str(out_dir / f"calibration_best_{combo_key}_{variant}.png"),
            title=f"Calibration: {combo_key} ({variant})",
        )
        logger.info("Wrote calibration plot for best combo %s (%s)", combo_key, variant)

    # Calibration for the two weakest individual signals
    per_signal = [
        (name, info["auroc"]) for name, info in result["per_signal"].items()
        if info["auroc"] is not None
    ]
    per_signal.sort(key=lambda x: x[1])
    for name, _a in per_signal[:2]:
        arr = rows.signals[name]
        valid = ~np.isnan(arr)
        plot_calibration(
            arr[valid], rows.local_correct[valid],
            str(out_dir / f"calibration_weak_{name}.png"),
            title=f"Calibration: {name}",
        )

    # Headline banner
    hl = result["headline"]
    logger.info("=" * 60)
    logger.info("HEADLINE")
    logger.info("  Best combo: %s (AUROC %.3f)", hl["best_combo"], hl["best_auroc"] or float("nan"))
    logger.info("  routellm_no_memory AUROC: %.3f", hl["routellm_no_memory_auroc"] or float("nan"))
    logger.info("  routellm_plus_ks AUROC:   %.3f", hl["routellm_plus_ks_auroc"] or float("nan"))
    logger.info("  Gap vs routellm_plus_ks:  %+0.3f", hl["gap_vs_routellm_plus_ks"] or float("nan"))
    logger.info("=" * 60)

    # Render the memo
    try:
        from benchmarks.memo import fill_memo, TEMPLATE_FILENAME, OUTPUT_FILENAME
        template_path = os.path.join(os.path.dirname(__file__), TEMPLATE_FILENAME)
        calib_best = next(
            (p.name for p in out_dir.glob("calibration_best_*.png")),
            "calibration_best.png",
        )
        fill_memo(
            summary_path=str(summary_path),
            template_path=template_path,
            output_path=str(out_dir / OUTPUT_FILENAME),
            cfg_meta={
                "local_model": "(see --local-model used at harness time)",
                "cloud_model": "(see --cloud-model used at harness time)",
                "embedding_model": "qllama/bge-large-en-v1.5",
                "eval_seed": 42,
                "train_seed": 43,
                "n_training_rows": 1000,
                "calibration_best_path": calib_best,
            },
        )
    except Exception as e:
        logger.warning("Memo filler failed: %s. summary.json is available regardless.", e)

    return 0


if __name__ == "__main__":
    sys.exit(main())

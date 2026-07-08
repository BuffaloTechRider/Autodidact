"""Does content-aware aggregation of FULL-response logprobs beat the naive mean
as a routing/confidence signal?

Unlike the streaming study (streaming_viability.py), this uses the ENTIRE
response — no latency constraint — so the "signal only matures late" problem
that killed streaming early-exit does NOT apply. The only question is whether
we're reading the full generation optimally.

Production today uses `avg_logprob` = unweighted mean over ALL tokens. On long
generations that mean is dominated by near-p=1 function words / punctuation,
diluting the content tokens that actually encode correctness. This script
compares several full-sequence aggregations against that baseline with a PAIRED
bootstrap (same queries), which is the right test for "is scorer A better than
scorer B on the same data".

Aggregations compared (all: higher score = more likely correct):
  mean            — unweighted mean logprob (PRODUCTION BASELINE)
  content_mean    — mean over content tokens only (stopwords/punct filtered)
  min             — single least-confident token
  bottomk_mean    — mean of the k least-confident tokens
  content_bottomk — bottomk over content tokens only
  neg_mean_entropy— -mean predictive entropy over content tokens (needs top-k)
  perplexity      — -exp(-mean logprob) (monotone transform of mean; sanity)

Run:
    python -m benchmarks.full_response_aggregation \\
        --rows results/experiment/streaming_viability/<...>/rows.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

from benchmarks.ablation_analysis import (
    auroc, bootstrap_auroc_ci, paired_bootstrap_delta_ci,
)
from benchmarks.streaming_viability import _is_content, _entropy

logger = logging.getLogger(__name__)


def _agg_scores(row: dict, bottom_k: int) -> dict[str, float]:
    """All full-response aggregation scores for one row (higher = confident)."""
    lp = np.asarray(row.get("token_logprobs") or [], dtype=np.float64)
    toks = row.get("tokens") or []
    n = len(lp)
    if n == 0:
        return {}
    content_mask = (np.array([_is_content(t) for t in toks], dtype=bool)
                    if len(toks) == n else np.ones(n, dtype=bool))
    clp = lp[content_mask]
    if clp.size == 0:
        clp = lp  # degenerate: no content tokens, fall back to all

    def bk(a: np.ndarray) -> float:
        k = min(bottom_k, a.size)
        return float(np.sort(a)[:k].mean())

    out = {
        "mean": float(lp.mean()),
        "content_mean": float(clp.mean()),
        "min": float(lp.min()),
        f"bottom{bottom_k}_mean": bk(lp),
        f"content_bottom{bottom_k}": bk(clp),
        "perplexity": -float(np.exp(-lp.mean())),
    }

    tops = row.get("top_logprobs_by_position") or []
    if len(tops) == n:
        ents = [_entropy(tops[j]) for j in range(n) if content_mask[j]]
        ents = [e for e in ents if not np.isnan(e)]
        if ents:
            out["neg_mean_entropy"] = -float(np.mean(ents))
    return out


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rows", nargs="+", required=True)
    p.add_argument("--bottom-k", type=int, default=10)
    p.add_argument("--baseline", default="mean", help="method to compare others against")
    args = p.parse_args()

    rows: list[dict] = []
    for rp in args.rows:
        with open(rp) as f:
            rows.extend(json.loads(line) for line in f if line.strip())
    rows = [r for r in rows if r.get("n_tokens", 0) > 0]
    if not rows:
        logger.error("no usable rows")
        return 2

    labels = np.array([r["correct"] for r in rows], dtype=np.int32)
    n_pos, n_neg = int(labels.sum()), int((labels == 0).sum())

    # Build score matrix per method, keeping only methods present in every row.
    per_row = [_agg_scores(r, args.bottom_k) for r in rows]
    methods = set(per_row[0])
    for pr in per_row:
        methods &= set(pr)
    methods = sorted(methods)
    scores = {m: np.array([pr[m] for pr in per_row], dtype=np.float64) for m in methods}

    # AUROC + CI per method.
    results = {}
    for m in methods:
        pt, lo, hi = bootstrap_auroc_ci(scores[m], labels, seed=0)
        results[m] = (pt, lo, hi)

    base = args.baseline
    dataset = rows[0].get("dataset", "?")
    model = rows[0].get("local_model", "?")

    print(f"\n{'='*78}")
    print(" Full-response aggregation vs naive-mean baseline")
    print(f"{'='*78}")
    print(f"  Model: {model}   Dataset: {dataset}   "
          f"n={len(rows)} (correct={n_pos}, incorrect={n_neg})")
    print(f"  Baseline (production): {base}\n")
    print(f"  {'method':>20}  {'AUROC':>6}  {'95% CI':>16}  {'Δ vs base':>10}  {'p':>6}  sig")
    # Sort by AUROC desc for readability.
    order = sorted(methods, key=lambda m: results[m][0], reverse=True)
    for m in order:
        pt, lo, hi = results[m]
        if m == base:
            print(f"  {m:>20}  {pt:>6.3f}  [{lo:.3f},{hi:.3f}]  {'—':>10}  {'—':>6}   (baseline)")
        else:
            d = paired_bootstrap_delta_ci(scores[m], scores[base], labels, seed=0)
            sig = "***" if d["significant"] else ""
            print(f"  {m:>20}  {pt:>6.3f}  [{lo:.3f},{hi:.3f}]  "
                  f"{d['point']:>+10.3f}  {d['p_value_approx']:>6.3f}  {sig}")
    print(f"{'='*78}")
    print("  Δ>0 means the method beats the naive-mean baseline. *** = 0 outside")
    print("  the paired 95% CI (significant on THIS n).")
    return 0


if __name__ == "__main__":
    sys.exit(main())

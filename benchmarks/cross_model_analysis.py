"""Cross-model analysis: aggregate GSA + answer-quality results across models.

Reads all recent GSA prompt-study runs and answer-quality-study runs from
results/experiment/, groups by (local_model), and produces a unified
comparison table suitable for the memo.

Used to answer:
- Does the same GSA prompt variant win across all tested models?
- Does retrieval injection have the same effect on all tested models?
- Which signals have consistent rankings vs. model-specific rankings?

Run:
    python -m benchmarks.cross_model_analysis

Output: results/experiment/cross_model_<timestamp>/
  - gsa_comparison.md
  - answer_quality_comparison.md
  - summary.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


def scan_gsa_studies(root: Path) -> list[dict[str, Any]]:
    """Walk root/gsa_prompt_study/<timestamp>/ and load each summary."""
    out = []
    root = root / "gsa_prompt_study"
    if not root.exists():
        return out
    for run_dir in sorted(root.iterdir()):
        if not run_dir.is_dir():
            continue
        summary_path = run_dir / "summary.json"
        rows_path = run_dir / "rows.jsonl"
        if not summary_path.exists() or not rows_path.exists():
            continue
        summary = json.loads(summary_path.read_text())
        # Parse the first line of rows.jsonl to get metadata we need.
        model = None
        use_kb = None
        kb_threshold = None
        eval_seed = None
        with open(rows_path) as f:
            first = f.readline()
            if first:
                try:
                    _ = json.loads(first)
                except Exception:
                    pass
        # Fall back to parsing the table.md header, which has the structured metadata.
        table_md = run_dir / "table.md"
        if table_md.exists():
            for line in table_md.read_text().splitlines():
                if line.startswith("**Local model:**"):
                    model = line.split("**Local model:**", 1)[1].strip()
                elif line.startswith("**Use KB:**"):
                    use_kb = line.split("**Use KB:**", 1)[1].strip() == "True"
                elif line.startswith("**KB threshold:**"):
                    try:
                        kb_threshold = float(line.split("**KB threshold:**", 1)[1].strip())
                    except ValueError:
                        kb_threshold = None
                elif line.startswith("**Eval seed:**"):
                    try:
                        eval_seed = int(line.split("**Eval seed:**", 1)[1].strip())
                    except ValueError:
                        eval_seed = None
        out.append({
            "run_dir": str(run_dir),
            "timestamp": run_dir.name,
            "local_model": model,
            "use_kb": use_kb,
            "kb_threshold": kb_threshold,
            "eval_seed": eval_seed,
            "summary": summary,
        })
    return out


def scan_answer_quality_studies(root: Path) -> list[dict[str, Any]]:
    """Walk root/answer_quality_study/<timestamp>/ and load each summary."""
    out = []
    root = root / "answer_quality_study"
    if not root.exists():
        return out
    for run_dir in sorted(root.iterdir()):
        if not run_dir.is_dir():
            continue
        summary_path = run_dir / "summary.json"
        if not summary_path.exists():
            continue
        summary = json.loads(summary_path.read_text())
        model = None
        table_md = run_dir / "table.md"
        if table_md.exists():
            for line in table_md.read_text().splitlines():
                if line.startswith("**Local model:**"):
                    model = line.split("**Local model:**", 1)[1].strip()
                    break
        out.append({
            "run_dir": str(run_dir),
            "timestamp": run_dir.name,
            "local_model": model,
            "summary": summary,
        })
    return out


def build_gsa_comparison(studies: list[dict]) -> str:
    """Build a per-(model, seed) × per-variant AUROC table.

    Only uses no-KB studies (use_kb=False) since that's the v2 setting.
    Shows every distinct (model, eval_seed) combination so replications are visible.
    """
    # Filter: no-KB studies only.
    studies = [s for s in studies if s["use_kb"] is False]
    if not studies:
        return "No no-KB GSA studies found."

    # Group by (model, eval_seed). Keep the largest-n within each group.
    by_key: dict[tuple, dict] = {}
    for s in studies:
        model = s["local_model"] or "unknown"
        seed = s.get("eval_seed")
        key = (model, seed)
        summary = s["summary"]
        n = next(iter(summary.values())).get("n", 0) if summary else 0
        if key not in by_key or n > next(iter(by_key[key]["summary"].values())).get("n", 0):
            by_key[key] = s

    # Variants from any study.
    variants = []
    for s in by_key.values():
        for v in s["summary"].keys():
            if v not in variants:
                variants.append(v)

    lines = ["# Cross-Model GSA Comparison (no-KB, v2 prompt study)", ""]
    lines.append("| Model | seed | " + " | ".join(f"`{v}`" for v in variants) + " | winner | n |")
    lines.append("|---|---|" + "|".join(["---"] * (len(variants) + 2)) + "|")
    # Sort for stable output: by model then by seed.
    for key in sorted(by_key.keys(), key=lambda k: (k[0], k[1] or 0)):
        model, seed = key
        s = by_key[key]
        summary = s["summary"]
        n = next(iter(summary.values())).get("n", "?") if summary else "?"
        best = None
        best_auroc = -1
        cells = []
        for v in variants:
            info = summary.get(v, {})
            auroc = info.get("signed_auroc")
            if auroc is not None and not (isinstance(auroc, float) and (auroc != auroc)):
                cells.append(f"{auroc:.3f}")
                if auroc > best_auroc:
                    best_auroc = auroc
                    best = v
            else:
                cells.append("n/a")
        seed_str = str(seed) if seed is not None else "?"
        lines.append(f"| `{model}` | {seed_str} | " + " | ".join(cells) + f" | **`{best}`** | {n} |")

    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    # Count winners per (model, seed).
    winners = defaultdict(int)
    for s in by_key.values():
        summary = s["summary"]
        best = None
        best_a = -1
        for v, info in summary.items():
            a = info.get("signed_auroc")
            if a is not None and a > best_a:
                best_a = a
                best = v
        if best:
            winners[best] += 1
    lines.append("Winning variant count across (model, seed) pairs: " + ", ".join(
        f"`{v}`={c}" for v, c in sorted(winners.items(), key=lambda kv: -kv[1])
    ))
    # Replication check: for each model, are multiple seeds consistent?
    models_with_multi = defaultdict(list)
    for (model, seed), s in by_key.items():
        summary = s["summary"]
        best = None
        best_a = -1
        for v, info in summary.items():
            a = info.get("signed_auroc")
            if a is not None and a > best_a:
                best_a = a
                best = v
        if best:
            models_with_multi[model].append((seed, best, best_a))
    for model, entries in models_with_multi.items():
        if len(entries) >= 2:
            winners_set = {e[1] for e in entries}
            if len(winners_set) == 1:
                lines.append(f"- `{model}` (n={len(entries)} seeds): same winner `{list(winners_set)[0]}` — stable across seeds.")
            else:
                lines.append(f"- `{model}` (n={len(entries)} seeds): different winners {winners_set} — unstable; single-seed result was a fluke.")
    if len(winners) == 1:
        lines.append("")
        lines.append("**One variant wins across all tested (model, seed) pairs.** Strong evidence for prompt-framing generalization.")
    elif len(by_key) >= 2 and len(winners) > 1:
        lines.append("")
        lines.append("**Different variants win on different models.** The prompt is model-specific; v0.1 either ships with per-model prompt selection or documents the scope.")
    lines.append("")
    return "\n".join(lines)


def build_answer_quality_comparison(studies: list[dict]) -> str:
    """Build a per-model retrieval-effect-on-answer table."""
    if not studies:
        return "No answer-quality studies found."
    # Keep largest-n study per model.
    by_model: dict[str, dict] = {}
    for s in studies:
        model = s["local_model"] or "unknown"
        summary = s["summary"]
        n = summary.get("n", 0)
        if model not in by_model or n > by_model[model]["summary"].get("n", 0):
            by_model[model] = s

    lines = [
        "# Cross-Model Answer Quality Comparison (retrieval injection effect)",
        "",
        "| Model | Acc WITH KB | Acc WITHOUT KB | Delta | p-value | n_with_hits/n | Verdict |",
        "|---|---|---|---|---|---|---|",
    ]
    for model, s in by_model.items():
        summary = s["summary"]
        acc_with = summary.get("accuracy_with_kb", float("nan"))
        acc_wo = summary.get("accuracy_without_kb", float("nan"))
        delta = summary.get("delta", float("nan"))
        pval = summary.get("mcnemar_pvalue", float("nan"))
        n = summary.get("n", "?")
        n_hits = summary.get("n_with_retrieved_hits", "?")

        if delta > 0.05 and pval < 0.05:
            verdict = "retrieval HELPS"
        elif delta < -0.05 and pval < 0.05:
            verdict = "retrieval HURTS"
        elif abs(delta) < 0.03:
            verdict = "wash"
        else:
            verdict = "underpowered"

        lines.append(
            f"| `{model}` | {acc_with:.3f} | {acc_wo:.3f} | {delta:+.3f} | "
            f"{pval:.3f} | {n_hits}/{n} | {verdict} |"
        )

    lines.append("")
    # Verdict consistency.
    verdicts = []
    for model, s in by_model.items():
        summary = s["summary"]
        delta = summary.get("delta", 0)
        pval = summary.get("mcnemar_pvalue", 1)
        if delta > 0.05 and pval < 0.05:
            verdicts.append("helps")
        elif delta < -0.05 and pval < 0.05:
            verdicts.append("hurts")
        else:
            verdicts.append("wash")

    lines.append("## Interpretation")
    lines.append("")
    uniq = set(verdicts)
    if uniq == {"wash"}:
        lines.append("**Retrieval injection is a wash across all tested models.** Safe to drop for latency savings.")
    elif uniq == {"helps"}:
        lines.append("**Retrieval injection helps on all tested models.** Keep it in the main answer prompt.")
    elif uniq == {"hurts"}:
        lines.append("**Retrieval injection hurts on all tested models.** Remove it from the main answer prompt.")
    else:
        lines.append(
            "**Verdict varies by model.** Consider per-model configuration or investigate whether "
            "the retrieval pipeline quality differs across models."
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="Cross-model analysis across GSA and answer-quality studies")
    p.add_argument("--results-root", default="results/experiment")
    p.add_argument("--output-dir", default=None,
                   help="Output directory; defaults to results/experiment/cross_model_<timestamp>/")
    args = p.parse_args()

    root = Path(args.results_root)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) if args.output_dir else root / f"cross_model_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    gsa = scan_gsa_studies(root)
    logger.info("Found %d GSA studies", len(gsa))
    aq = scan_answer_quality_studies(root)
    logger.info("Found %d answer-quality studies", len(aq))

    gsa_md = build_gsa_comparison(gsa)
    aq_md = build_answer_quality_comparison(aq)

    (out_dir / "gsa_comparison.md").write_text(gsa_md)
    (out_dir / "answer_quality_comparison.md").write_text(aq_md)

    summary = {
        "gsa_studies_found": len(gsa),
        "answer_quality_studies_found": len(aq),
        "gsa_models": sorted({s["local_model"] for s in gsa if s["local_model"]}),
        "answer_quality_models": sorted({s["local_model"] for s in aq if s["local_model"]}),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print("=" * 70)
    print(gsa_md)
    print()
    print(aq_md)
    print("=" * 70)
    print(f"\nArtifacts: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

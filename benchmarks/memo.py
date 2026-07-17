"""Fill the memo template with real numbers from summary.json.

Applies the decision rules from the v0.1 design:

Best-combo AUROC:
  >= 0.80 → build the product
  0.70-0.79 → build with caveats, conservative threshold
  0.60-0.69 → improve mechanism before building
  < 0.60 → pivot mechanism

Signal-complexity rule:
  energy_plus_knowledge_plus_gsa within 2 AUROC points of all_six_thompson
      → ship the 3-signal version

RouteLLM gap rule:
  best combo beats routellm_plus_ks by >= 0.05 AUROC with non-overlapping CIs
      → paper is viable

Output: results/experiment/MEMO.md

Requirement 7 (R7) from the v0.1 spec is satisfied here.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

TEMPLATE_FILENAME = "memo_template.md"
OUTPUT_FILENAME = "MEMO.md"


def fmt_auroc(auroc: Optional[float]) -> str:
    return f"{auroc:.3f}" if auroc is not None else "n/a"


def fmt_ci(ci: Optional[list]) -> str:
    if not ci or len(ci) != 2 or ci[0] is None:
        return "n/a"
    return f"[{ci[0]:.3f}, {ci[1]:.3f}]"


def fmt_gap(gap: Optional[float]) -> str:
    if gap is None:
        return "n/a"
    return f"{gap:+0.3f}"


def build_summary_paragraph(summary: dict) -> str:
    hl = summary["headline"]
    n = summary["n_total"]
    best_combo = hl["best_combo"] or "n/a"
    best_auroc = hl["best_auroc"]
    gap = hl["gap_vs_routellm_plus_ks"]
    if best_auroc is None:
        return f"Analysis on {n} queries produced no complete ablation combo. Investigate missing signals before drawing conclusions."
    recommendation = _build_recommendation(summary)["headline"]
    return (
        f"Across {n} evaluation queries, the best signal combination `{best_combo}` "
        f"achieved AUROC {fmt_auroc(best_auroc)} against the RouteLLM-plus-KS baseline's {fmt_auroc(hl['routellm_plus_ks_auroc'])} "
        f"(gap {fmt_gap(gap)}). {recommendation}"
    )


def build_per_signal_table(summary: dict) -> str:
    lines = []
    for name, info in summary["per_signal"].items():
        a = fmt_auroc(info["auroc"])
        ci = fmt_ci(info["ci"])
        lines.append(f"| `{name}` | {a} | {ci} | {info['n']} |")
    return "\n".join(lines)


def build_per_combo_table(summary: dict) -> str:
    lines = []
    for name, info in summary["per_combo"].items():
        m_a = fmt_auroc(info["mean"]["auroc"])
        m_ci = fmt_ci(info["mean"].get("ci"))
        t_a = fmt_auroc(info["thompson"]["auroc"])
        t_ci = fmt_ci(info["thompson"].get("ci"))
        n = info["mean"]["n"]
        lines.append(f"| `{name}` | {m_a} | {m_ci} | {t_a} | {t_ci} | {n} |")
    return "\n".join(lines)


def _build_recommendation(summary: dict) -> dict:
    """Apply the decision rules from the design. Returns dict with sections."""
    hl = summary["headline"]
    auroc = hl["best_auroc"]
    gap = hl["gap_vs_routellm_plus_ks"]

    # AUROC-based recommendation
    if auroc is None:
        headline = "No complete combo produced; mechanism unusable as-is."
        action = "Debug the signal pipeline. Verify `logprobs` are reaching the local backend."
    elif auroc >= 0.80:
        headline = (
            "The confidence evaluator works well. Best combo AUROC >= 0.80 — "
            "this is strong enough to build a user-facing product on top of."
        )
        action = "Proceed to v0.2 product work. Pick a product shape (Aider proxy, tool-learning demo, etc.)."
    elif auroc >= 0.70:
        headline = (
            "The confidence evaluator is usable but not strong. Best combo AUROC is 0.70-0.79 — "
            "build with caveats. Set the local/cloud threshold conservatively so the product prefers "
            "cloud escalation when signals are ambiguous."
        )
        action = "Proceed to v0.2 with a conservative threshold. Consider improving retrieval quality first."
    elif auroc >= 0.60:
        headline = (
            "The confidence evaluator is marginal. Best combo AUROC is 0.60-0.69 — probably not enough "
            "for a user-facing product without more work."
        )
        action = "Improve the mechanism before shipping. Try a bigger local model, fine-tune a classifier on the training data, or add a new signal."
    else:
        headline = (
            "The confidence evaluator does not work at this model size. Best combo AUROC < 0.60. "
            "The local-first-plus-memory approach as mechanized here is not viable."
        )
        action = "Pivot the mechanism. Step back before committing to a product direction."

    # Signal-complexity recommendation — uses paired delta if available
    simplicity_note = ""
    if auroc is not None:
        paired_full = (summary.get("paired_deltas") or {}).get("full_vs_simple")
        combos = summary["per_combo"]
        simple_name = "energy_plus_knowledge_plus_gsa"
        full_name = "all_six_thompson"
        simple_info = combos.get(simple_name, {}).get("thompson", {})
        full_info = combos.get(full_name, {}).get("thompson", {})
        simple_a = simple_info.get("auroc")
        full_a = full_info.get("auroc")
        if paired_full is not None and simple_a is not None and full_a is not None:
            pt = paired_full.get("point")  # full - simple; positive means full is better
            lo = paired_full.get("lo")
            hi = paired_full.get("hi")
            sig = bool(paired_full.get("significant"))
            if not sig:
                simplicity_note = (
                    f"The 3-signal combo `{simple_name}` (AUROC {simple_a:.3f}) is statistically "
                    f"indistinguishable from the full 6-signal combo (AUROC {full_a:.3f}) — "
                    f"paired ΔAUROC {pt:+0.3f} (95% CI [{lo:+0.3f}, {hi:+0.3f}]) includes 0. "
                    "**Drop self_consistency and logprob_uncertainty** — they don't earn their latency cost."
                )
            elif pt > 0:
                simplicity_note = (
                    f"The full 6-signal combo (AUROC {full_a:.3f}) significantly beats the 3-signal "
                    f"combo (AUROC {simple_a:.3f}) — paired ΔAUROC {pt:+0.3f} (95% CI [{lo:+0.3f}, {hi:+0.3f}]). "
                    "The expensive signals earn their keep. Keep them."
                )
            else:
                simplicity_note = (
                    f"The 3-signal combo (AUROC {simple_a:.3f}) significantly beats the full "
                    f"6-signal combo (AUROC {full_a:.3f}) — paired ΔAUROC {pt:+0.3f} "
                    f"(95% CI [{lo:+0.3f}, {hi:+0.3f}]). Drop the extra signals; they're adding noise."
                )
        elif simple_a is not None and full_a is not None:
            # Fallback: point-estimate heuristic
            diff = full_a - simple_a
            if diff <= 0.02:
                simplicity_note = (
                    f"The 3-signal combo `{simple_name}` (AUROC {simple_a:.3f}) is within "
                    f"2 AUROC points of the full 6-signal combo (AUROC {full_a:.3f}). "
                    "**Drop self_consistency and logprob_uncertainty** — they don't earn their latency cost."
                )
            else:
                simplicity_note = (
                    f"The full 6-signal Thompson combo (AUROC {full_a:.3f}) beats the 3-signal "
                    f"combo (AUROC {simple_a:.3f}) by {diff:.3f}. The expensive signals earn their keep. "
                    "Keep them."
                )

    # Paper viability — uses the paired bootstrap delta CI if available (statistically
    # proper), and falls back to the rougher non-overlapping-CI heuristic if not.
    paper_note = ""
    paired = (summary.get("paired_deltas") or {}).get("best_vs_routellm_plus_ks")
    if gap is not None and auroc is not None and auroc >= 0.70:
        if paired is not None:
            pt = paired.get("point")
            lo = paired.get("lo")
            hi = paired.get("hi")
            significant = bool(paired.get("significant"))
            if pt is not None and lo is not None and hi is not None:
                if significant and pt >= 0.05:
                    paper_note = (
                        f"**Paper is viable.** Paired-bootstrap ΔAUROC vs `routellm_plus_ks` is "
                        f"{pt:+0.3f} (95% CI [{lo:+0.3f}, {hi:+0.3f}]); CI excludes 0, "
                        "so the lift is statistically significant at alpha=0.05. A workshop paper "
                        "or ArXiv preprint framed as \"memory-aware confidence gating for lifelong "
                        "local LLM agents\" would be defensible. Product remains the priority; "
                        "paper is a side-output."
                    )
                elif significant and pt > 0:
                    paper_note = (
                        f"**Paper is marginal.** Paired-bootstrap ΔAUROC is {pt:+0.3f} "
                        f"(95% CI [{lo:+0.3f}, {hi:+0.3f}]); CI excludes 0 but the effect size "
                        "is below the +0.050 threshold. Publishable as a short empirical note; "
                        "not a strong paper."
                    )
                elif pt > 0:
                    paper_note = (
                        f"**Paper not viable.** Paired-bootstrap ΔAUROC is {pt:+0.3f} "
                        f"(95% CI [{lo:+0.3f}, {hi:+0.3f}]) — the CI includes 0. Our combo "
                        "is not significantly better than `routellm_plus_ks` at n={n}. Ship the "
                        "product only; skip the paper."
                    ).format(n=summary.get("n_total", "?"))
                else:
                    paper_note = (
                        f"**Negative result.** Paired-bootstrap ΔAUROC is {pt:+0.3f} "
                        f"(95% CI [{lo:+0.3f}, {hi:+0.3f}]); `routellm_plus_ks` beats our combo. "
                        "Consider using supervised routing with retrieval features directly; "
                        "our multi-signal fusion adds no statistical value at this scale."
                    )
        else:
            # Fallback heuristic (no paired delta available for some reason)
            rllm_ci = summary["baselines"]["routellm_plus_ks"].get("ci") or [None, None]
            best_ci = _find_best_combo_ci(summary)
            non_overlapping = False
            if best_ci and rllm_ci[0] is not None and best_ci[0] is not None:
                non_overlapping = best_ci[0] > rllm_ci[1]
            if gap >= 0.05 and non_overlapping:
                paper_note = (
                    f"**Paper is viable.** Best combo beats `routellm_plus_ks` by {gap:+0.3f} AUROC "
                    "with non-overlapping 95% CIs. Paired-delta bootstrap was unavailable; interpretation "
                    "is slightly conservative."
                )
            else:
                paper_note = (
                    f"**Paper is not viable yet.** Gap over `routellm_plus_ks` is {fmt_gap(gap)} "
                    "(need >= +0.050 with non-overlapping CIs). Ship the product only; skip the paper."
                )
    elif gap is not None:
        paper_note = f"**Paper not viable** at current AUROC. Focus on product."

    return {
        "headline": headline,
        "action": action,
        "simplicity": simplicity_note,
        "paper": paper_note,
    }


def _find_best_combo_ci(summary: dict) -> Optional[list]:
    """Return the CI of whatever combo is named as best_combo in the headline."""
    best_combo = summary["headline"].get("best_combo")
    if not best_combo:
        return None
    # Format: "combo_name (variant)"
    try:
        name, variant = best_combo.split(" (")
        variant = variant.rstrip(")")
    except ValueError:
        return None
    combo_info = summary["per_combo"].get(name)
    if not combo_info:
        return None
    return combo_info.get(variant, {}).get("ci")


def build_recommendation_block(summary: dict) -> str:
    r = _build_recommendation(summary)
    parts = [r["headline"], "", f"**Next action:** {r['action']}"]
    if r["simplicity"]:
        parts += ["", r["simplicity"]]
    if r["paper"]:
        parts += ["", r["paper"]]
    return "\n".join(parts)


def build_prior_art_interpretation(summary: dict) -> str:
    hl = summary["headline"]
    best = hl["best_auroc"]
    rllm_ks = hl["routellm_plus_ks_auroc"]
    if best is None or rllm_ks is None:
        return ""
    gap = best - rllm_ks
    if gap >= 0.05:
        return (
            "Interpretation: our memory-aware combo substantively beats a strong supervised baseline "
            "that also has access to the knowledge-similarity feature. This supports the claim that "
            "dynamic, memory-aware confidence gating is meaningfully different from static supervised routing."
        )
    if gap >= 0.0:
        return (
            "Interpretation: our memory-aware combo is close to a strong supervised baseline but does not "
            "clearly beat it. The architecture is validated but the algorithmic win is marginal. Product viable; "
            "paper narrower than hoped."
        )
    return (
        "Interpretation: a static supervised classifier with the knowledge-similarity feature beats our "
        "dynamic combo. This is an important negative result — it means memory-aware fusion as designed "
        "does not add value over a single learned signal. Consider using the supervised baseline directly "
        "and dropping the multi-signal Thompson fusion."
    )


def build_retrieval_section(summary: dict) -> str:
    retrieval = summary.get("retrieval") or {}
    if not retrieval:
        return "Not enough retrieval-recall split to stratify."
    lines = ["| Condition | Best combo AUROC | 95% CI | n |", "|---|---|---|---|"]
    for label, info in retrieval.items():
        a = fmt_auroc(info.get("auroc"))
        ci = fmt_ci(info.get("ci"))
        lines.append(f"| `{label}` | {a} | {ci} | {info.get('n', 0)} |")
    return "\n".join(lines)


def build_calibration_notes(summary: dict) -> str:
    hl = summary["headline"]
    best = hl.get("best_auroc")
    if best is None:
        return "Calibration not computed — no best combo."
    # Look at calibration qualitatively based on ECE-like intuition:
    # ideally sloped at 1:1, observed close to predicted.
    return (
        "A well-calibrated signal has its observed curve on the diagonal. Systematic bias above the "
        "diagonal means under-confidence (the signal says 0.6 but the real correct rate is higher); "
        "below the diagonal means over-confidence. Temperature scaling can fix mis-calibration cheaply "
        "if needed at product time."
    )


def fill_memo(summary_path: str, template_path: str, output_path: str, cfg_meta: dict) -> None:
    summary = json.loads(Path(summary_path).read_text())
    template = Path(template_path).read_text()

    substitutions = {
        "run_id": summary["run_id"],
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "local_model": cfg_meta.get("local_model", "(unknown)"),
        "cloud_model": cfg_meta.get("cloud_model", "(unknown)"),
        "embedding_model": cfg_meta.get("embedding_model", "(unknown)"),
        "n_total": str(summary.get("n_total", "?")),
        "eval_seed": str(cfg_meta.get("eval_seed", "?")),
        "train_seed": str(cfg_meta.get("train_seed", "?")),
        "n_training_rows": str(cfg_meta.get("n_training_rows", "?")),
        "summary_paragraph": build_summary_paragraph(summary),
        "per_signal_table": build_per_signal_table(summary),
        "per_combo_table": build_per_combo_table(summary),
        "routellm_no_memory_auroc": fmt_auroc(summary["baselines"]["routellm_no_memory"]["auroc"]),
        "routellm_no_memory_ci": fmt_ci(summary["baselines"]["routellm_no_memory"].get("ci")),
        "routellm_plus_ks_auroc": fmt_auroc(summary["baselines"]["routellm_plus_ks"]["auroc"]),
        "routellm_plus_ks_ci": fmt_ci(summary["baselines"]["routellm_plus_ks"].get("ci")),
        "best_combo": summary["headline"].get("best_combo") or "n/a",
        "best_auroc": fmt_auroc(summary["headline"].get("best_auroc")),
        "best_ci": fmt_ci(_find_best_combo_ci(summary)),
        "gap_vs_routellm_plus_ks": fmt_gap(summary["headline"].get("gap_vs_routellm_plus_ks")),
        "prior_art_interpretation": build_prior_art_interpretation(summary),
        "calibration_best_path": cfg_meta.get("calibration_best_path", "calibration_best.png"),
        "calibration_notes": build_calibration_notes(summary),
        "retrieval_section": build_retrieval_section(summary),
        "recommendation": build_recommendation_block(summary),
    }

    out = template
    for key, val in substitutions.items():
        out = out.replace("{{" + key + "}}", str(val))

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(out)
    logger.info("Wrote memo to %s", output_path)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Fill the Autodidact v0.1 memo template")
    parser.add_argument("--output-dir", default="results/experiment")
    parser.add_argument("--local-model", default="qwen2.5:7b")
    parser.add_argument("--cloud-model", default="anthropic.claude-3-haiku-20240307-v1:0")
    parser.add_argument("--embedding-model", default="qllama/bge-large-en-v1.5")
    parser.add_argument("--eval-seed", type=int, default=42)
    parser.add_argument("--train-seed", type=int, default=43)
    parser.add_argument("--n-training-rows", type=int, default=1000)
    parser.add_argument("--calibration-best", default="calibration_best.png",
                        help="Filename (relative to output_dir) of the best calibration plot to embed")
    args = parser.parse_args()

    output_dir = args.output_dir
    summary_path = os.path.join(output_dir, "summary.json")
    template_path = os.path.join(os.path.dirname(__file__), TEMPLATE_FILENAME)
    output_path = os.path.join(output_dir, OUTPUT_FILENAME)

    if not os.path.exists(summary_path):
        logger.error("summary.json not found at %s. Run ablation_analysis first.", summary_path)
        return 2

    cfg_meta = {
        "local_model": args.local_model,
        "cloud_model": args.cloud_model,
        "embedding_model": args.embedding_model,
        "eval_seed": args.eval_seed,
        "train_seed": args.train_seed,
        "n_training_rows": args.n_training_rows,
        "calibration_best_path": args.calibration_best,
    }

    fill_memo(summary_path, template_path, output_path, cfg_meta)
    return 0


if __name__ == "__main__":
    sys.exit(main())

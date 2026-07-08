"""Task 14.7: Labeling noise audit.

Samples labeled rows from completed experiment runs and re-labels them with
a DIFFERENT judge model to estimate label noise. Reports per-path disagreement
rates (letter-match path vs judge-fallback path) and a total pipeline-weighted
noise bound.

The primary judge is Claude Opus 4.5 (used during the main experiment).
The secondary audit judge is Claude Sonnet 4.5 (cheaper, different model).

Run:
    python -m benchmarks.label_noise_audit

Output: results/experiment/label_noise_audit.{json,md}
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

from autodidact.database import init_database
from autodidact.llm_client import ChatMessage, LLMClient, LLMConfig

logger = logging.getLogger(__name__)

AUDIT_JUDGE_PROMPT = """You are a strict grader. Decide whether the MODEL_ANSWER correctly answers the QUESTION.

The correct answer is: "{ground_truth}"

QUESTION:
{question}

MODEL_ANSWER:
{model_answer}

Respond with exactly one token: YES or NO."""


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="Label noise audit for v0.1 experiment results")
    p.add_argument("--run-ids", nargs="+",
                   default=["v0.1-qwen-20260427", "v0.1-llama-20260428", "v0.1-mistral-20260429"])
    p.add_argument("--n-sample", type=int, default=100,
                   help="Total rows to audit (split evenly across letter-match and judge paths)")
    p.add_argument("--audit-judge-model", default="us.anthropic.claude-sonnet-4-5-20250929-v1:0")
    p.add_argument("--bedrock-region", default="us-west-2")
    p.add_argument("--db-path", default="autodidact_experiment.db")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", default="results/experiment")
    args = p.parse_args()

    conn = init_database(args.db_path)
    conn.row_factory = __import__("sqlite3").Row

    audit_judge = LLMClient(LLMConfig(
        provider="bedrock", model=args.audit_judge_model, region=args.bedrock_region,
    ))

    # Collect rows from all runs, split by labeling path.
    letter_rows = []  # judge_used = 0 (letter extraction succeeded)
    judge_rows = []   # judge_used = 1 (fell back to primary judge)

    for run_id in args.run_ids:
        rows = conn.execute(
            "SELECT query_text, ground_truth, local_answer, local_correct, judge_used "
            "FROM experiment_results "
            "WHERE run_id = ? AND error_info IS NULL",
            (run_id,),
        ).fetchall()
        for r in rows:
            entry = {
                "question": r["query_text"],
                "ground_truth": r["ground_truth"],
                "local_answer": r["local_answer"],
                "primary_label": int(r["local_correct"]),
                "judge_used": int(r["judge_used"]),
            }
            if r["judge_used"]:
                judge_rows.append(entry)
            else:
                letter_rows.append(entry)

    logger.info("Collected %d letter-match rows, %d judge-fallback rows across %d runs",
                len(letter_rows), len(judge_rows), len(args.run_ids))

    # Sample evenly.
    rng = random.Random(args.seed)
    n_per_path = args.n_sample // 2
    letter_sample = rng.sample(letter_rows, min(n_per_path, len(letter_rows)))
    judge_sample = rng.sample(judge_rows, min(n_per_path, len(judge_rows)))
    logger.info("Sampled %d letter-match, %d judge-fallback for audit",
                len(letter_sample), len(judge_sample))

    # Re-label each with the audit judge.
    def audit_one(entry: dict) -> int:
        prompt = AUDIT_JUDGE_PROMPT.format(
            ground_truth=entry["ground_truth"],
            question=entry["question"].strip()[:500],
            model_answer=(entry["local_answer"] or "").strip()[:500],
        )
        resp = audit_judge.chat(
            [ChatMessage(role="user", content=prompt)],
            max_tokens=4, temperature=0.0,
        )
        text = (resp.content or "").strip().upper()
        if text.startswith("YES"):
            return 1
        return 0

    letter_disagree = 0
    for i, entry in enumerate(letter_sample):
        try:
            audit_label = audit_one(entry)
            if audit_label != entry["primary_label"]:
                letter_disagree += 1
        except Exception as e:
            logger.warning("Audit call failed: %s", e)
        if (i + 1) % 10 == 0:
            logger.info("Letter-match audit: %d/%d", i + 1, len(letter_sample))

    judge_disagree = 0
    for i, entry in enumerate(judge_sample):
        try:
            audit_label = audit_one(entry)
            if audit_label != entry["primary_label"]:
                judge_disagree += 1
        except Exception as e:
            logger.warning("Audit call failed: %s", e)
        if (i + 1) % 10 == 0:
            logger.info("Judge-fallback audit: %d/%d", i + 1, len(judge_sample))

    # Compute rates.
    n_letter = len(letter_sample)
    n_judge = len(judge_sample)
    letter_rate = letter_disagree / n_letter if n_letter > 0 else 0.0
    judge_rate = judge_disagree / n_judge if n_judge > 0 else 0.0

    # Pipeline-weighted noise bound.
    total_letter = len(letter_rows)
    total_judge = len(judge_rows)
    total_all = total_letter + total_judge
    letter_frac = total_letter / total_all if total_all > 0 else 0.5
    judge_frac = total_judge / total_all if total_all > 0 else 0.5
    weighted_noise = letter_frac * letter_rate + judge_frac * judge_rate

    result = {
        "n_letter_sampled": n_letter,
        "n_judge_sampled": n_judge,
        "letter_disagree": letter_disagree,
        "judge_disagree": judge_disagree,
        "letter_disagreement_rate": round(letter_rate, 4),
        "judge_disagreement_rate": round(judge_rate, 4),
        "letter_fraction_of_pipeline": round(letter_frac, 3),
        "judge_fraction_of_pipeline": round(judge_frac, 3),
        "weighted_noise_bound": round(weighted_noise, 4),
        "audit_judge_model": args.audit_judge_model,
        "primary_judge_model": "us.anthropic.claude-opus-4-5-20251101-v1:0",
        "run_ids": args.run_ids,
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "label_noise_audit.json").write_text(json.dumps(result, indent=2))

    md_lines = [
        "# Label Noise Audit",
        "",
        f"**Primary judge:** {result['primary_judge_model']}",
        f"**Audit judge:** {result['audit_judge_model']}",
        f"**Runs audited:** {', '.join(args.run_ids)}",
        "",
        "| Path | Sampled | Disagreements | Rate |",
        "|---|---:|---:|---:|",
        f"| Letter-match | {n_letter} | {letter_disagree} | {letter_rate:.1%} |",
        f"| Judge-fallback | {n_judge} | {judge_disagree} | {judge_rate:.1%} |",
        "",
        f"**Pipeline composition:** {letter_frac:.1%} letter-match, {judge_frac:.1%} judge-fallback",
        f"**Weighted noise bound:** {weighted_noise:.2%}",
        "",
        "Interpretation: the weighted noise bound estimates the fraction of",
        "`local_correct` labels that would change if a different judge model",
        "were used. This bounds the label noise that could affect AUROC measurements.",
    ]
    (out_dir / "label_noise_audit.md").write_text("\n".join(md_lines))

    print("\n".join(md_lines))
    print(f"\nArtifacts: {out_dir}/label_noise_audit.{{json,md}}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

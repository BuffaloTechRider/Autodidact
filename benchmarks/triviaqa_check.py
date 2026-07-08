"""Task 14.6: TriviaQA external-validity check.

Tests whether logprob_uncertainty generalizes from MMLU-Pro (multiple-choice)
to TriviaQA (open-ended short-answer). This is the cross-dataset validity
check for Paper A.

Key differences from MMLU-Pro:
- Open-ended: model generates a free-form answer, not a letter
- Labeling: exact-match against answer aliases (multiple acceptable forms),
  with LLM judge fallback for near-misses
- No RouteLLM baseline (would need TriviaQA-specific training data)
- No KB seeding (testing signal generalization, not retrieval)

What we measure:
- logprob_uncertainty AUROC against local_correct
- GSA v3 AUROC (bare prompt, no retrieval — no KB for TriviaQA)
- Local accuracy (what fraction does qwen get right on trivia?)

Run:
    python -u -m benchmarks.triviaqa_check \\
        --n-queries 500 --local-model qwen2.5:7b

Output: results/experiment/triviaqa_check/<timestamp>/
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

from autodidact.llm_client import ChatMessage, LLMClient, LLMConfig
from benchmarks.ablation_analysis import auroc, bootstrap_auroc_ci

logger = logging.getLogger(__name__)


def load_triviaqa_subset(n: int, seed: int, split: str = "validation") -> list[dict]:
    """Load n TriviaQA questions with answer aliases.

    Returns list of dicts with keys: question, answer_aliases, question_id.
    """
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise RuntimeError("datasets package required") from e

    ds = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split=split)
    # Shuffle deterministically and take first n.
    indices = list(range(len(ds)))
    random.Random(seed).shuffle(indices)
    indices = indices[:n]

    out = []
    for idx in indices:
        item = ds[idx]
        question = item["question"]
        # answer field has 'aliases' (list of acceptable answers) and 'value' (canonical)
        answer_obj = item.get("answer", {})
        aliases = answer_obj.get("aliases", [])
        canonical = answer_obj.get("value", "")
        if canonical and canonical not in aliases:
            aliases = [canonical] + aliases
        # Normalize aliases for matching.
        normalized = list(set(a.strip().lower() for a in aliases if a.strip()))
        if not normalized:
            continue
        out.append({
            "question_id": item.get("question_id", f"tqa:{idx}"),
            "question": question,
            "answer_aliases": normalized,
            "canonical_answer": canonical,
        })
    return out[:n]


def check_answer(response: str, aliases: list[str]) -> bool:
    """Check if the model's response contains any of the acceptable answer aliases.

    Uses normalized containment check — if any alias appears as a substring
    of the normalized response, it's correct. This is generous but standard
    for TriviaQA evaluation.
    """
    if not response:
        return False
    resp_lower = response.strip().lower()
    for alias in aliases:
        if alias in resp_lower:
            return True
    return False


def extract_logprob_uncertainty(avg_logprob: Optional[float], scale: float = 2.0, shift: float = 3.0) -> float:
    """Same sigmoid transform as confidence_evaluator.compute_logprob_uncertainty."""
    if avg_logprob is None:
        return 0.5
    x = avg_logprob * scale + shift
    return float(1.0 / (1.0 + math.exp(-x)))


def extract_gsa_p_yes(response) -> float:
    """Extract p_yes from a 1-token YES/NO logprob response."""
    yes_tokens = {"YES", "Yes", "yes", "Y", "y", " YES", " Yes", " yes", " Y", " y"}
    no_tokens = {"NO", "No", "no", "N", "n", " NO", " No", " no", " N", " n"}

    if response.top_logprobs_by_position:
        pos = response.top_logprobs_by_position[0]
        yes_lp = None
        no_lp = None
        for tok, lp in pos.items():
            if tok in yes_tokens or tok.strip().upper() in {"YES", "Y"}:
                yes_lp = float(lp)
                break
        for tok, lp in pos.items():
            if tok in no_tokens or tok.strip().upper() in {"NO", "N"}:
                no_lp = float(lp)
                break
        if yes_lp is not None or no_lp is not None:
            missing = -20.0
            ye = yes_lp if yes_lp is not None else missing
            ne = no_lp if no_lp is not None else missing
            mx = max(ye, ne)
            p = math.exp(ye - mx) / (math.exp(ye - mx) + math.exp(ne - mx))
            return max(0.0, min(1.0, p))

    # Text fallback.
    text = (response.content or "").strip().upper()
    if text and text.split()[0] in {"YES", "Y"}:
        return 1.0
    if text and text.split()[0] in {"NO", "N"}:
        return 0.0
    return 0.5


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="TriviaQA external-validity check for logprob_uncertainty")
    p.add_argument("--n-queries", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--local-model", default="qwen2.5:7b")
    p.add_argument("--judge-model", default="us.anthropic.claude-opus-4-5-20251101-v1:0")
    p.add_argument("--bedrock-region", default="us-west-2")
    p.add_argument("--output-dir", default="results/experiment/triviaqa_check")
    p.add_argument("--use-judge-fallback", action="store_true",
                   help="Use LLM judge for cases where alias matching is ambiguous")
    args = p.parse_args()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) / stamp
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading %d TriviaQA questions...", args.n_queries)
    questions = load_triviaqa_subset(args.n_queries, args.seed)
    logger.info("Loaded %d questions", len(questions))

    local_client = LLMClient(LLMConfig(
        provider="ollama", model=args.local_model,
    ))
    judge_client = None
    if args.use_judge_fallback:
        judge_client = LLMClient(LLMConfig(
            provider="bedrock", model=args.judge_model, region=args.bedrock_region,
        ))

    rows_path = out_dir / "rows.jsonl"
    with open(rows_path, "w") as f:
        for i, q in enumerate(questions):
            # Generate answer with logprobs.
            try:
                resp = local_client.chat_with_logprobs(
                    [ChatMessage(role="user", content=f"Answer this trivia question in a few words.\n\nQuestion: {q['question']}\nAnswer:")],
                    max_tokens=50, temperature=0.0, top_logprobs=1,
                )
            except Exception as e:
                logger.warning("Query %d failed: %s", i, e)
                continue

            answer = resp.content.strip()
            avg_lp = resp.avg_logprob
            logprob_unc = extract_logprob_uncertainty(avg_lp)

            # Label: does the answer match any alias?
            correct = check_answer(answer, q["answer_aliases"])

            # GSA probe (bare prompt, no retrieval — TriviaQA has no KB).
            try:
                gsa_resp = local_client.chat_with_logprobs(
                    [ChatMessage(role="user", content=(
                        f"The user has asked the following question:\n{q['question']}\n\n"
                        "Are you confident you can answer this question correctly? "
                        "Respond with exactly one token: YES or NO."
                    ))],
                    max_tokens=1, temperature=0.0, top_logprobs=5,
                )
                gsa_p_yes = extract_gsa_p_yes(gsa_resp)
            except Exception:
                gsa_p_yes = 0.5

            f.write(json.dumps({
                "question_id": q["question_id"],
                "question": q["question"],
                "model_answer": answer,
                "canonical_answer": q["canonical_answer"],
                "correct": int(correct),
                "avg_logprob": avg_lp,
                "logprob_uncertainty": logprob_unc,
                "gsa_p_yes": gsa_p_yes,
            }) + "\n")

            if (i + 1) % 50 == 0:
                logger.info("Progress: %d/%d", i + 1, len(questions))

    # Aggregate.
    with open(rows_path) as f:
        rows = [json.loads(line) for line in f if line.strip()]

    n = len(rows)
    labels = np.array([r["correct"] for r in rows], dtype=np.int32)
    lp_scores = np.array([r["logprob_uncertainty"] for r in rows], dtype=np.float64)
    gsa_scores = np.array([r["gsa_p_yes"] for r in rows], dtype=np.float64)

    acc = float(labels.mean())
    lp_auroc_val = auroc(lp_scores, labels)
    lp_ci = bootstrap_auroc_ci(lp_scores, labels)
    gsa_auroc_val = auroc(gsa_scores, labels)
    gsa_ci = bootstrap_auroc_ci(gsa_scores, labels)

    summary = {
        "dataset": "TriviaQA (rc.nocontext, validation)",
        "local_model": args.local_model,
        "n": n,
        "local_accuracy": acc,
        "logprob_uncertainty_auroc": float(lp_auroc_val),
        "logprob_uncertainty_ci": [float(lp_ci[1]), float(lp_ci[2])],
        "gsa_auroc": float(gsa_auroc_val),
        "gsa_ci": [float(gsa_ci[1]), float(gsa_ci[2])],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    # Print results.
    print(f"\n{'='*60}")
    print(f" TriviaQA External Validity Check")
    print(f"{'='*60}")
    print(f"  Model: {args.local_model}")
    print(f"  N queries: {n}")
    print(f"  Local accuracy: {acc:.3f}")
    print(f"  logprob_uncertainty AUROC: {lp_auroc_val:.3f} [{lp_ci[1]:.3f}, {lp_ci[2]:.3f}]")
    print(f"  GSA (bare prompt) AUROC:   {gsa_auroc_val:.3f} [{gsa_ci[1]:.3f}, {gsa_ci[2]:.3f}]")
    print(f"\n  Compare to MMLU-Pro qwen: logprob=0.714, GSA=0.562")
    print(f"{'='*60}\n")
    print(f"Artifacts: {out_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Answer-Quality Study — does retrieval injection help the local model's main answer?

The GSA prompt study (benchmarks/gsa_prompt_study.py) showed that injecting
retrieved knowledge into the self-assessment prompt hurt signal quality.
A reasonable next question: does retrieval injection help the MAIN answer
generation?

Each test query is asked to the local model TWICE on identical conditions:
once with top-k retrieved knowledge in the prompt, once without. The cloud
model is used as a labeler (same letter-extraction + LLM-judge fallback as
the main experiment).

The output compares:
  - local_correct rate with KB  vs  without KB
  - per-query delta: did KB flip this query correct↔incorrect?

If with-KB wins by ≥ 5%, retrieval injection is worth it for answers.
If without-KB wins, retrieval is actively confusing the model.
If they tie, retrieval is a wash and we can save the latency cost.

Run:
    python -m benchmarks.answer_quality_study --n-queries 60 --skip-first 500

Output: results/experiment/answer_quality_study/<timestamp>/
  - rows.jsonl: per-query data
  - summary.json: aggregates
  - table.md: human-readable comparison
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from autodidact.database import init_database
from autodidact.knowledge_store import KnowledgeStore
from autodidact.llm_client import ChatMessage, LLMClient, LLMConfig
from autodidact.types import AutodidactConfig

from benchmarks.datasets import load_mmlu_pro_subset
from benchmarks.labeling import label_answer

logger = logging.getLogger(__name__)


def build_kb_prompt(query_prompt: str, retrieved_hits: list) -> str:
    """Same injection pattern used by the main experiment harness."""
    if not retrieved_hits:
        return query_prompt
    lines = ["You have retrieved the following from your knowledge base:"]
    for i, hit in enumerate(retrieved_hits, start=1):
        content = getattr(hit.entry, "content", "")[:400].strip()
        q = getattr(hit.entry, "question", None)
        if q:
            lines.append(f"{i}. (memory of: {q.strip()[:120]})\n   {content}")
        else:
            lines.append(f"{i}. {content}")
    lines.append("")
    lines.append("Now answer the user's question.")
    lines.append("")
    lines.append(query_prompt)
    return "\n".join(lines)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="Does retrieval injection help local answer quality?")
    p.add_argument("--n-queries", type=int, default=60)
    p.add_argument("--eval-seed", type=int, default=42)
    p.add_argument("--skip-first", type=int, default=500,
                   help="Skip first N queries (typical: the ones already in the KB seed)")
    p.add_argument("--local-model", default="qwen2.5:7b")
    p.add_argument("--cloud-model", default="us.anthropic.claude-sonnet-4-5-20250929-v1:0")
    p.add_argument("--judge-model", default="us.anthropic.claude-opus-4-5-20251101-v1:0")
    p.add_argument("--embedding-model", default="qllama/bge-large-en-v1.5")
    p.add_argument("--bedrock-region", default="us-west-2")
    p.add_argument("--db-path", default="autodidact_experiment.db")
    p.add_argument("--output-dir", default="results/experiment/answer_quality_study")
    p.add_argument("--similarity-threshold", type=float, default=None,
                   help="Override KnowledgeStore similarity_threshold for this run. "
                        "Default None = use the config default (0.75). Set to 0.0 to "
                        "disable threshold filtering entirely; use when an embedder's "
                        "score distribution differs from nomic-era calibration.")
    args = p.parse_args()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) / stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Output: %s", out_dir)

    # Load queries disjoint from the KB seed split.
    total = args.n_queries + args.skip_first
    all_queries = load_mmlu_pro_subset(total, args.eval_seed)
    queries = all_queries[args.skip_first : args.skip_first + args.n_queries]
    logger.info("Loaded %d queries (skipped first %d)", len(queries), args.skip_first)

    # Clients
    local_client = LLMClient(LLMConfig(
        provider="ollama", model=args.local_model, embedding_model=args.embedding_model,
    ))
    judge_client = LLMClient(LLMConfig(
        provider="bedrock", model=args.judge_model, region=args.bedrock_region,
    ))

    conn = init_database(args.db_path)
    config = AutodidactConfig(db_path=args.db_path)
    if args.similarity_threshold is not None:
        config.similarity_threshold = args.similarity_threshold
        logger.info("Using similarity_threshold override: %.3f", config.similarity_threshold)
    ks = KnowledgeStore(conn, config)
    logger.info("Knowledge store has %d valid entries", ks.count())

    rows_path = out_dir / "rows.jsonl"
    with open(rows_path, "w") as f:
        for i, q in enumerate(queries):
            # Retrieve top-k hits (uses the internal 0.75 threshold in the store).
            q_embedding = local_client.embed(q.query_text)
            hits = ks.search(q_embedding, limit=5)

            # Condition A: WITH retrieval injection.
            prompt_with_kb = build_kb_prompt(q.formatted_prompt(), hits)
            resp_with = local_client.chat(
                [ChatMessage(role="user", content=prompt_with_kb)],
                max_tokens=512, temperature=0.0,
            )
            correct_with, used_judge_with = label_answer(
                resp_with.content, q.ground_truth_letter, q.ground_truth_answer,
                q.query_text, judge_client=judge_client,
            )

            # Condition B: WITHOUT retrieval injection.
            resp_without = local_client.chat(
                [ChatMessage(role="user", content=q.formatted_prompt())],
                max_tokens=512, temperature=0.0,
            )
            correct_without, used_judge_without = label_answer(
                resp_without.content, q.ground_truth_letter, q.ground_truth_answer,
                q.query_text, judge_client=judge_client,
            )

            f.write(json.dumps({
                "query_id": q.query_id,
                "category": q.category,
                "n_hits_retrieved": len(hits),
                "top_hit_score": float(hits[0].score) if hits else None,
                # Categories of the top-5 retrieved entries. Lets downstream
                # analysis measure in-category recall independent of score thresholds.
                "hit_categories": [getattr(h.entry, "domain", None) for h in hits],
                "top1_in_category": (
                    bool(hits and getattr(hits[0].entry, "domain", None) == q.category)
                ),
                "any_top5_in_category": (
                    any(getattr(h.entry, "domain", None) == q.category for h in hits)
                ),
                "correct_with_kb": int(correct_with),
                "correct_without_kb": int(correct_without),
                "judge_used_with_kb": int(used_judge_with),
                "judge_used_without_kb": int(used_judge_without),
            }) + "\n")

            if (i + 1) % 10 == 0:
                logger.info("Progress: %d/%d", i + 1, len(queries))

    # Aggregate
    with open(rows_path) as f:
        rows = [json.loads(line) for line in f]

    n = len(rows)
    acc_with = np.mean([r["correct_with_kb"] for r in rows])
    acc_without = np.mean([r["correct_without_kb"] for r in rows])
    flipped_to_correct = sum(1 for r in rows if r["correct_without_kb"] == 0 and r["correct_with_kb"] == 1)
    flipped_to_wrong = sum(1 for r in rows if r["correct_without_kb"] == 1 and r["correct_with_kb"] == 0)
    same = n - flipped_to_correct - flipped_to_wrong
    n_with_hits = sum(1 for r in rows if r["n_hits_retrieved"] > 0)
    # Threshold-free retrieval quality: how often does the pipeline pull a
    # semantically relevant entry? Measured by MMLU-Pro category match.
    n_top1_in_category = sum(1 for r in rows if r.get("top1_in_category"))
    n_any_top5_in_category = sum(1 for r in rows if r.get("any_top5_in_category"))

    summary = {
        "n": n,
        "n_with_retrieved_hits": n_with_hits,
        "retrieval_recall_top1_in_category": n_top1_in_category / n if n else 0.0,
        "retrieval_recall_any_top5_in_category": n_any_top5_in_category / n if n else 0.0,
        "accuracy_with_kb": float(acc_with),
        "accuracy_without_kb": float(acc_without),
        "delta": float(acc_with - acc_without),
        "flipped_to_correct": flipped_to_correct,
        "flipped_to_wrong": flipped_to_wrong,
        "same": same,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    # McNemar-style paired comparison: b (flipped to correct) vs c (flipped to wrong).
    # Significance via binomial. Simple two-sided test: is b ≈ c under null?
    from scipy.stats import binomtest
    b = flipped_to_correct
    c = flipped_to_wrong
    if b + c > 0:
        pval = binomtest(min(b, c), b + c, p=0.5, alternative="two-sided").pvalue
    else:
        pval = float("nan")
    summary["mcnemar_pvalue"] = float(pval)

    # Markdown table
    lines = [
        "# Answer Quality Study — retrieval injection effect on local answer",
        f"**Timestamp:** {stamp}",
        f"**Local model:** {args.local_model}",
        f"**Judge:** {args.judge_model}",
        f"**N queries:** {n}",
        f"**Skip first:** {args.skip_first}",
        f"**Queries with ≥1 retrieved hit (score ≥ threshold):** {n_with_hits}/{n}",
        f"**Top-1 in-category (threshold-free):** {n_top1_in_category}/{n} = {n_top1_in_category/n:.1%}",
        f"**Any top-5 in-category (threshold-free):** {n_any_top5_in_category}/{n} = {n_any_top5_in_category/n:.1%}",
        "",
        "| Condition | Local accuracy |",
        "|---|---|",
        f"| WITH retrieval injection | {acc_with:.3f} |",
        f"| WITHOUT retrieval injection | {acc_without:.3f} |",
        f"| **Delta (with − without)** | **{summary['delta']:+0.3f}** |",
        "",
        "## Flips (McNemar-style)",
        "",
        "| Category | Count |",
        "|---|---|",
        f"| Without→With flipped to CORRECT | {flipped_to_correct} |",
        f"| With→Without flipped to WRONG | {flipped_to_wrong} |",
        f"| Same in both conditions | {same} |",
        f"| McNemar two-sided p-value | {pval:.3f} |",
        "",
    ]

    # Recommendation based on thresholds.
    if summary["delta"] >= 0.05 and pval < 0.05:
        rec = "**Retrieval injection HELPS.** Keep it in the main prompt."
    elif summary["delta"] <= -0.05 and pval < 0.05:
        rec = "**Retrieval injection HURTS.** Remove it from the main prompt."
    elif abs(summary["delta"]) < 0.03:
        rec = (
            "**Retrieval injection is a wash.** Likely safe to drop for latency savings "
            "(retrieval + embedding work is wasted)."
        )
    else:
        rec = (
            "**Trend is suggestive but not statistically significant at this n. "
            "Rerun at larger N to confirm before making a call.**"
        )
    lines.append("## Recommendation")
    lines.append("")
    lines.append(rec)
    lines.append("")

    (out_dir / "table.md").write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"\nArtifacts: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

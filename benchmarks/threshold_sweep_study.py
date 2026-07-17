"""EXP-002: Similarity threshold sweep.

Measures how the `AutodidactConfig.similarity_threshold` affects downstream
signal quality on qwen2.5:7b with the current 498-entry bge-large KB.

Per-threshold we measure:
  - knowledge_similarity signal AUROC against local_correct.
    Computed as raw max-cosine-similarity from the top-5 retrieved hits at
    that threshold (NO zero-clamp, regardless of whether Change B has
    landed — we want the threshold's effect in isolation).
  - Four GSA variants' AUROC (current, direct, confidence, prediction)
    against local_correct. GSA prompts see the hits returned at the current
    threshold, so this is a downstream measurement of the retrieval
    threshold's effect.
  - retrieval_recall_any_top5_in_category (threshold-FREE reference).
    Should be constant across thresholds.

Why bypass compute_knowledge_similarity: the current production code has a
"return 0.0 if max_sim < threshold" clamp in `confidence_evaluator.py` that
would confound this sweep with Change B's behavior. We measure the raw
signal here and let Change B be a separate decision.

Run:
    python -m benchmarks.threshold_sweep_study \\
        --n-queries 60 --skip-first 500 \\
        --local-model qwen2.5:7b \\
        --thresholds 0.50,0.55,0.60,0.65,0.70,0.75

Output: results/experiment/threshold_sweep_study/<timestamp>/
  - rows.jsonl — per-(query, threshold) data
  - summary.json — per-threshold AUROC table
  - table.md — human-readable comparison
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

from autodidact.database import init_database
from autodidact.knowledge_store import KnowledgeStore
from autodidact.llm_client import ChatMessage, LLMClient, LLMConfig
from autodidact.types import AutodidactConfig

from benchmarks.datasets import load_mmlu_pro_subset
from benchmarks.gsa_prompt_study import (
    PROMPT_VARIANTS,
    _build_prompt,
    _extract_p_yes,
    signed_auroc,
)
from benchmarks.labeling import label_answer

logger = logging.getLogger(__name__)


@dataclass
class PerQueryRecord:
    """Signal values for one query at one threshold."""
    query_id: str
    category: str
    threshold: float
    n_hits: int
    max_sim: float  # Raw; no zero-clamp.
    top1_in_category: bool
    any_top5_in_category: bool
    local_correct: int
    gsa_p_yes: dict[str, float]  # variant -> p_yes


def run_at_threshold(
    threshold: float,
    queries: list,
    local_client: LLMClient,
    ks: KnowledgeStore,
    gsa_cache: dict[tuple[str, str], float],
    local_correct_cache: dict[str, int],
) -> list[PerQueryRecord]:
    """Run the signal pipeline at a given threshold.

    The `gsa_cache` key is (query_id, variant) → p_yes. We don't cache across
    thresholds because the retrieved hits change; only within a single
    threshold's runs.

    `local_correct_cache` is query_id → 0/1, shared across thresholds because
    local answer generation is independent of retrieval.
    """
    # Swap in the threshold for this run.
    old_threshold = ks.config.similarity_threshold
    ks.config.similarity_threshold = threshold
    # Note: the FAISS index contents don't depend on the threshold (threshold
    # is applied in `_faiss_search` after FAISS returns candidates), so we
    # don't need to rebuild. Leave _faiss_dirty alone.

    records: list[PerQueryRecord] = []
    for q in queries:
        q_emb = local_client.embed(q.query_text)
        hits = ks.search(q_emb, limit=5)

        # Raw max_sim. Zero if no hits above threshold.
        max_sim = max((h.score for h in hits), default=0.0)

        # Threshold-FREE category-match metrics: do a separate unfiltered
        # search so the "did the pipeline return a semantically relevant
        # entry at all" question is answered independently of the threshold
        # being swept. The threshold is applied after FAISS retrieval, so
        # swapping config.similarity_threshold temporarily doesn't invalidate
        # the index.
        ks.config.similarity_threshold = 0.0
        unfiltered_hits = ks.search(q_emb, limit=5)
        ks.config.similarity_threshold = threshold

        top1_in_cat = (
            bool(unfiltered_hits) and unfiltered_hits[0].entry.domain == q.category
        )
        any_top5_in_cat = any(h.entry.domain == q.category for h in unfiltered_hits)

        # Run each GSA variant, cache to avoid recomputation if called again.
        gsa_p_yes: dict[str, float] = {}
        for variant, template in PROMPT_VARIANTS.items():
            cache_key = (q.query_id, variant, f"{threshold:.2f}")
            if cache_key in gsa_cache:
                gsa_p_yes[variant] = gsa_cache[cache_key]
                continue
            prompt = _build_prompt(template, q.query_text, hits)
            response = local_client.chat_with_logprobs(
                [ChatMessage(role="user", content=prompt)],
                max_tokens=1, temperature=0.0, top_logprobs=5,
            )
            p_yes, _mode = _extract_p_yes(response)
            gsa_cache[cache_key] = p_yes
            gsa_p_yes[variant] = p_yes

        records.append(PerQueryRecord(
            query_id=q.query_id,
            category=q.category,
            threshold=threshold,
            n_hits=len(hits),
            max_sim=float(max_sim),
            top1_in_category=top1_in_cat,
            any_top5_in_category=any_top5_in_cat,
            local_correct=local_correct_cache[q.query_id],
            gsa_p_yes=gsa_p_yes,
        ))

    ks.config.similarity_threshold = old_threshold
    return records


def _compute_auroc_bundle(records: list[PerQueryRecord]) -> dict:
    """Compute knowledge_similarity and per-variant GSA AUROC for this threshold."""
    ks_scores = np.array([r.max_sim for r in records], dtype=np.float64)
    labels = np.array([r.local_correct for r in records], dtype=np.int32)
    ks_auroc = signed_auroc(ks_scores, labels)

    gsa_aurocs: dict[str, float] = {}
    for variant in PROMPT_VARIANTS:
        scores = np.array([r.gsa_p_yes[variant] for r in records], dtype=np.float64)
        gsa_aurocs[variant] = float(signed_auroc(scores, labels))

    n = len(records)
    top1_rate = sum(1 for r in records if r.top1_in_category) / n if n else 0.0
    any5_rate = sum(1 for r in records if r.any_top5_in_category) / n if n else 0.0
    mean_max_sim = float(np.mean(ks_scores))
    n_with_hits = sum(1 for r in records if r.n_hits > 0)

    return {
        "n": n,
        "n_with_hits": n_with_hits,
        "mean_max_sim": mean_max_sim,
        "retrieval_recall_top1_in_category": top1_rate,
        "retrieval_recall_any_top5_in_category": any5_rate,
        "knowledge_similarity_auroc": float(ks_auroc) if not np.isnan(ks_auroc) else None,
        "gsa_auroc": gsa_aurocs,
    }


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="Sweep the similarity threshold on downstream AUROC")
    p.add_argument("--n-queries", type=int, default=60)
    p.add_argument("--eval-seed", type=int, default=42)
    p.add_argument("--skip-first", type=int, default=500,
                   help="Skip first N queries (typical: the KB-seeded ones). "
                        "At skip_first=500 with a 498-entry KB we probe held-out queries.")
    p.add_argument("--local-model", default="qwen2.5:7b")
    p.add_argument("--cloud-model", default="us.anthropic.claude-sonnet-4-5-20250929-v1:0")
    p.add_argument("--judge-model", default="us.anthropic.claude-opus-4-5-20251101-v1:0")
    p.add_argument("--embedding-model", default="qllama/bge-large-en-v1.5")
    p.add_argument("--bedrock-region", default="us-west-2")
    p.add_argument("--db-path", default="autodidact_experiment.db")
    p.add_argument("--output-dir", default="results/experiment/threshold_sweep_study")
    p.add_argument("--thresholds", default="0.50,0.55,0.60,0.65,0.70,0.75",
                   help="Comma-separated thresholds to sweep")
    args = p.parse_args()

    thresholds = [float(s) for s in args.thresholds.split(",")]
    logger.info("Sweeping thresholds: %s", thresholds)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) / stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Output dir: %s", out_dir)

    # Load queries
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

    # KB (single shared instance; we mutate config.similarity_threshold per run)
    conn = init_database(args.db_path)
    config = AutodidactConfig(db_path=args.db_path)
    ks = KnowledgeStore(conn, config)
    logger.info("Knowledge store has %d valid entries", ks.count())

    # Stage 1: compute local_correct labels once, cached across all thresholds.
    # This is the only expensive cloud call we make (judge fallbacks).
    local_correct_cache: dict[str, int] = {}
    logger.info("Stage 1: labeling %d local answers (shared across all thresholds)", len(queries))
    for i, q in enumerate(queries):
        resp = local_client.chat(
            [ChatMessage(role="user", content=q.formatted_prompt())],
            max_tokens=512, temperature=0.0,
        )
        correct, _judge_used = label_answer(
            resp.content, q.ground_truth_letter, q.ground_truth_answer,
            q.query_text, judge_client=judge_client,
        )
        local_correct_cache[q.query_id] = int(correct)
        if (i + 1) % 10 == 0:
            logger.info("  labeled %d/%d", i + 1, len(queries))
    correct_rate = sum(local_correct_cache.values()) / len(local_correct_cache)
    logger.info("Local correct rate: %.2f", correct_rate)

    # Stage 2: for each threshold, run retrieval + GSA probes.
    gsa_cache: dict = {}
    rows_path = out_dir / "rows.jsonl"
    all_summaries: dict[str, dict] = {}

    with open(rows_path, "w") as f:
        for t in thresholds:
            logger.info("Sweep point: threshold = %.2f", t)
            records = run_at_threshold(
                threshold=t,
                queries=queries,
                local_client=local_client,
                ks=ks,
                gsa_cache=gsa_cache,
                local_correct_cache=local_correct_cache,
            )
            for r in records:
                f.write(json.dumps({
                    "threshold": r.threshold,
                    "query_id": r.query_id,
                    "category": r.category,
                    "n_hits": r.n_hits,
                    "max_sim": r.max_sim,
                    "top1_in_category": r.top1_in_category,
                    "any_top5_in_category": r.any_top5_in_category,
                    "local_correct": r.local_correct,
                    "gsa_p_yes": r.gsa_p_yes,
                }) + "\n")

            bundle = _compute_auroc_bundle(records)
            all_summaries[f"{t:.2f}"] = bundle

    # Stage 3: summary and table
    (out_dir / "summary.json").write_text(
        json.dumps({
            "local_model": args.local_model,
            "n_queries": len(queries),
            "correct_rate": correct_rate,
            "per_threshold": all_summaries,
        }, indent=2)
    )

    # Markdown table
    lines = [
        "# Threshold Sweep Study (EXP-002)",
        f"**Timestamp:** {stamp}",
        f"**Local model:** {args.local_model}",
        f"**N queries:** {len(queries)}",
        f"**Eval seed:** {args.eval_seed}, skip_first={args.skip_first}",
        f"**Correct rate:** {correct_rate:.2f}",
        "",
        "## Per-threshold AUROC",
        "",
        "| Threshold | n_hits | mean max_sim | top1_in_cat | any5_in_cat | "
        "KS AUROC | GSA current | GSA direct | GSA confidence | GSA prediction |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for t_key, s in all_summaries.items():
        ks_a = s["knowledge_similarity_auroc"]
        ks_str = f"{ks_a:.3f}" if ks_a is not None else "n/a"
        gsa = s["gsa_auroc"]
        lines.append(
            f"| {t_key} | {s['n_with_hits']}/{s['n']} | {s['mean_max_sim']:.3f} | "
            f"{s['retrieval_recall_top1_in_category']:.1%} | "
            f"{s['retrieval_recall_any_top5_in_category']:.1%} | "
            f"{ks_str} | "
            f"{gsa['current']:.3f} | {gsa['direct']:.3f} | "
            f"{gsa['confidence']:.3f} | {gsa['prediction']:.3f} |"
        )
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")

    # Find peaks and report.
    ks_by_t = {t: all_summaries[t]["knowledge_similarity_auroc"] for t in all_summaries
               if all_summaries[t]["knowledge_similarity_auroc"] is not None}
    if ks_by_t:
        best_t = max(ks_by_t, key=ks_by_t.get)
        lines.append(f"- **Best threshold for knowledge_similarity AUROC:** {best_t} "
                     f"(AUROC {ks_by_t[best_t]:.3f})")
        worst_t = min(ks_by_t, key=ks_by_t.get)
        lines.append(f"- **Worst threshold for knowledge_similarity AUROC:** {worst_t} "
                     f"(AUROC {ks_by_t[worst_t]:.3f})")
        gap = ks_by_t[best_t] - ks_by_t[worst_t]
        lines.append(f"- **Best-vs-worst AUROC gap:** {gap:.3f}")

    recalls = {t: all_summaries[t]["retrieval_recall_any_top5_in_category"]
               for t in all_summaries}
    recall_range = max(recalls.values()) - min(recalls.values())
    lines.append(f"- **Threshold-free recall variation across thresholds:** "
                 f"{recall_range:.3f} (should be ~0; sanity check)")

    (out_dir / "table.md").write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"\nArtifacts: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""EXP-009: Does RouteLLM transfer across models?

Train RouteLLM on model A's labels, evaluate on model B's eval queries with
model B's correctness labels. Tests whether the supervised classifier learns
query-difficulty (model-agnostic) or model-specific behavior.

If it transfers: RouteLLM's advantage is learning query difficulty, not model behavior.
If it fails: RouteLLM is model-specific, confirming per-model retraining is mandatory.

Cost: ~$0 (reuses cached data + one embedding pass). Time: ~3 min.

Run:
    python -m benchmarks.routellm_cross_model_transfer
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import sys
from pathlib import Path

import numpy as np

from autodidact.database import init_database
from autodidact.llm_client import LLMClient, LLMConfig
from benchmarks.ablation_analysis import auroc, bootstrap_auroc_ci
from benchmarks.datasets import load_mmlu_pro_subset

logger = logging.getLogger(__name__)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--source-model-dir", default="results/experiment/v0.1-qwen-20260427",
                        help="Directory containing the RouteLLM pickles to transfer FROM")
    parser.add_argument("--target-run-ids", nargs="+",
                        default=["v0.1-llama-20260428", "v0.1-mistral-20260429"],
                        help="Run IDs to evaluate the transferred classifier ON")
    parser.add_argument("--embedding-model", default="qllama/bge-large-en-v1.5")
    parser.add_argument("--db-path", default="autodidact_experiment.db")
    parser.add_argument("--eval-seed", type=int, default=42)
    parser.add_argument("--n-seed", type=int, default=1000)
    parser.add_argument("--n-eval", type=int, default=1000)
    parser.add_argument("--output-dir", default="results/experiment")
    args = parser.parse_args()

    # Load the source model's trained classifiers.
    nm_path = os.path.join(args.source_model_dir, "routellm_no_memory.pkl")
    pks_path = os.path.join(args.source_model_dir, "routellm_plus_ks.pkl")
    if not os.path.exists(nm_path) or not os.path.exists(pks_path):
        logger.error("Source pickles not found at %s", args.source_model_dir)
        return 2
    with open(nm_path, "rb") as f:
        nm_model = pickle.load(f)
    with open(pks_path, "rb") as f:
        pks_model = pickle.load(f)
    logger.info("Loaded source RouteLLM from %s", args.source_model_dir)

    # Load eval queries (same for all models — indices n_seed to n_seed+n_eval).
    total = args.n_seed + args.n_eval
    all_queries = load_mmlu_pro_subset(total, args.eval_seed)
    eval_queries = all_queries[args.n_seed:]
    logger.info("Loaded %d eval queries", len(eval_queries))

    # Embed all eval queries once (embeddings are model-agnostic via bge-large).
    embed_client = LLMClient(LLMConfig(
        provider="ollama", model="qwen2.5:7b",  # model doesn't matter for embed
        embedding_model=args.embedding_model,
    ))
    logger.info("Embedding %d eval queries...", len(eval_queries))
    embeddings: dict[str, np.ndarray] = {}
    for i, q in enumerate(eval_queries):
        try:
            embeddings[q.query_id] = embed_client.embed(q.query_text)
        except Exception as e:
            logger.warning("Embed failed for %s: %s", q.query_id, e)
        if (i + 1) % 100 == 0:
            logger.info("  embedded %d/%d", i + 1, len(eval_queries))
    logger.info("Embedded %d queries", len(embeddings))

    # For each target run, score with the source classifier and compute AUROC.
    conn = init_database(args.db_path)
    conn.row_factory = __import__("sqlite3").Row

    results: dict[str, dict] = {}
    for run_id in args.target_run_ids:
        rows = conn.execute(
            "SELECT query_id, knowledge_similarity, local_correct "
            "FROM experiment_results WHERE run_id = ? AND error_info IS NULL "
            "ORDER BY query_index",
            (run_id,),
        ).fetchall()
        logger.info("Run %s: %d clean eval rows", run_id, len(rows))

        nm_scores = []
        pks_scores = []
        labels = []
        for r in rows:
            qid = r["query_id"]
            if qid not in embeddings:
                continue
            emb = embeddings[qid]
            ks = float(r["knowledge_similarity"])
            # Score with SOURCE model's classifier.
            x_nm = emb.reshape(1, -1).astype(np.float32)
            nm_proba = nm_model.predict_proba(x_nm)[0]
            nm_idx = list(nm_model.classes_).index(1)
            nm_scores.append(float(nm_proba[nm_idx]))

            x_pks = np.concatenate([emb, [ks]]).reshape(1, -1).astype(np.float32)
            pks_proba = pks_model.predict_proba(x_pks)[0]
            pks_idx = list(pks_model.classes_).index(1)
            pks_scores.append(float(pks_proba[pks_idx]))

            labels.append(int(r["local_correct"]))

        nm_arr = np.array(nm_scores)
        pks_arr = np.array(pks_scores)
        lab_arr = np.array(labels)

        nm_auroc = auroc(nm_arr, lab_arr)
        pks_auroc = auroc(pks_arr, lab_arr)
        nm_ci = bootstrap_auroc_ci(nm_arr, lab_arr)
        pks_ci = bootstrap_auroc_ci(pks_arr, lab_arr)

        results[run_id] = {
            "n": len(labels),
            "nm_auroc": float(nm_auroc),
            "nm_ci": [float(nm_ci[1]), float(nm_ci[2])],
            "pks_auroc": float(pks_auroc),
            "pks_ci": [float(pks_ci[1]), float(pks_ci[2])],
        }
        logger.info(
            "  %s: nm_auroc=%.3f [%.3f, %.3f]  pks_auroc=%.3f [%.3f, %.3f]",
            run_id, nm_auroc, nm_ci[1], nm_ci[2], pks_auroc, pks_ci[1], pks_ci[2],
        )

    # Save and print.
    out_path = Path(args.output_dir) / "routellm_cross_model_transfer.json"
    out_path.write_text(json.dumps({
        "source": args.source_model_dir,
        "results": results,
    }, indent=2))
    logger.info("Wrote %s", out_path)

    print("\n=== RouteLLM Cross-Model Transfer (trained on qwen, evaluated on others) ===\n")
    print(f"{'target':>30}  {'nm AUROC':>10}  {'pks AUROC':>10}  {'n':>6}")
    print("-" * 65)
    for run_id, r in results.items():
        model = run_id.split("-")[1]  # e.g. "llama" from "v0.1-llama-20260428"
        print(f"{model:>30}  {r['nm_auroc']:>10.3f}  {r['pks_auroc']:>10.3f}  {r['n']:>6}")

    # Also print the native (per-model trained) baselines for comparison.
    print("\n=== For comparison: natively-trained RouteLLM (per-model) ===\n")
    for run_id in args.target_run_ids:
        summary_path = Path(args.output_dir) / run_id / "summary.json"
        if summary_path.exists():
            s = json.loads(summary_path.read_text())
            nm = s["baselines"]["routellm_no_memory"]["auroc"]
            pks = s["baselines"]["routellm_plus_ks"]["auroc"]
            model = run_id.split("-")[1]
            print(f"  {model:>28}  nm={nm:.3f}  pks={pks:.3f}  (trained on own labels)")

    return 0


if __name__ == "__main__":
    sys.exit(main())

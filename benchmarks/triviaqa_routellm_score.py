"""Score TriviaQA queries with RouteLLM trained on MMLU-Pro.

Tests cross-DATASET transfer of the supervised baseline. The classifier
was trained on MMLU-Pro query embeddings; here we score TriviaQA query
embeddings and measure AUROC against each model's correctness labels.

Run after triviaqa_check.py has produced rows for each model.

    python -m benchmarks.triviaqa_routellm_score
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

from autodidact.llm_client import LLMClient, LLMConfig
from benchmarks.ablation_analysis import auroc, bootstrap_auroc_ci
from benchmarks.triviaqa_check import load_triviaqa_subset

logger = logging.getLogger(__name__)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="Score TriviaQA with MMLU-Pro-trained RouteLLM")
    p.add_argument("--source-model-dir", default="results/experiment/v0.1-qwen-20260427",
                    help="Dir with RouteLLM pickles (trained on MMLU-Pro)")
    p.add_argument("--triviaqa-dirs", nargs="+", default=[
        "results/experiment/triviaqa_check",
        "results/experiment/triviaqa_check_llama",
        "results/experiment/triviaqa_check_mistral",
    ])
    p.add_argument("--embedding-model", default="qllama/bge-large-en-v1.5")
    p.add_argument("--n-queries", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    # Load classifier.
    nm_path = os.path.join(args.source_model_dir, "routellm_no_memory.pkl")
    with open(nm_path, "rb") as f:
        nm_model = pickle.load(f)
    logger.info("Loaded RouteLLM from %s", args.source_model_dir)

    # Load and embed TriviaQA questions (same set used by all models).
    questions = load_triviaqa_subset(args.n_queries, args.seed)
    embed_client = LLMClient(LLMConfig(
        provider="ollama", model="qwen2.5:7b", embedding_model=args.embedding_model,
    ))
    logger.info("Embedding %d TriviaQA queries...", len(questions))
    embeddings = {}
    for i, q in enumerate(questions):
        try:
            embeddings[q["question_id"]] = embed_client.embed(q["question"])
        except Exception:
            pass
        if (i + 1) % 100 == 0:
            logger.info("  %d/%d", i + 1, len(questions))
    logger.info("Embedded %d", len(embeddings))

    # For each TriviaQA result dir, load rows, score with RouteLLM, compute AUROC.
    print(f"\n{'model':>20}  {'RouteLLM AUROC':>15}  {'95% CI':>20}  {'n':>5}")
    print("-" * 70)

    for tqa_dir_base in args.triviaqa_dirs:
        tqa_dir = Path(tqa_dir_base)
        if not tqa_dir.is_dir():
            continue
        # Find the most recent timestamp subdir.
        subdirs = sorted([d for d in tqa_dir.iterdir() if d.is_dir()], reverse=True)
        if not subdirs:
            continue
        rows_path = subdirs[0] / "rows.jsonl"
        summary_path = subdirs[0] / "summary.json"
        if not rows_path.exists():
            continue

        with open(rows_path) as f:
            rows = [json.loads(line) for line in f if line.strip()]

        # Read model name from summary.
        model_name = "unknown"
        if summary_path.exists():
            s = json.loads(summary_path.read_text())
            model_name = s.get("local_model", "unknown")

        scores = []
        labels = []
        for r in rows:
            qid = r["question_id"]
            if qid not in embeddings:
                continue
            emb = embeddings[qid]
            x = emb.reshape(1, -1).astype(np.float32)
            proba = nm_model.predict_proba(x)[0]
            idx = list(nm_model.classes_).index(1)
            scores.append(float(proba[idx]))
            labels.append(int(r["correct"]))

        if len(scores) < 10:
            print(f"  {model_name:>18}  {'n/a':>15}  {'':>20}  {len(scores):>5}")
            continue

        s_arr = np.array(scores)
        l_arr = np.array(labels)
        a = auroc(s_arr, l_arr)
        ci = bootstrap_auroc_ci(s_arr, l_arr)
        print(f"  {model_name:>18}  {a:>15.3f}  [{ci[1]:.3f}, {ci[2]:.3f}]  {len(scores):>5}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

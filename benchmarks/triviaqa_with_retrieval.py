"""EXP-011: TriviaQA with retrieval-conditional GSA.

Seeds a TriviaQA-specific KB (200 questions, separate from the 500 eval set),
then re-runs the TriviaQA eval with GSA v3 retrieval-conditional prompting
and logprob+GSA fusion.

Tests whether claim 4 (GSA v3 with retrieval > bare GSA) and claim 6 (naive
fusion hurts) hold on a different dataset.

Uses a SEPARATE SQLite DB (triviaqa_experiment.db) so TriviaQA KB entries
don't mix with MMLU-Pro entries.

Run:
    python -u -m benchmarks.triviaqa_with_retrieval \\
        --local-model qwen2.5:7b

Does all 3 models sequentially if --all-models is passed.
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

from autodidact.database import init_database
from autodidact.knowledge_store import KnowledgeStore
from autodidact.llm_client import ChatMessage, LLMClient, LLMConfig
from autodidact.types import AutodidactConfig, NewKnowledgeEntry
from benchmarks.ablation_analysis import auroc, bootstrap_auroc_ci
from benchmarks.triviaqa_check import (
    check_answer,
    extract_gsa_p_yes,
    extract_logprob_uncertainty,
    load_triviaqa_subset,
)

logger = logging.getLogger(__name__)

DB_PATH = "triviaqa_experiment.db"
N_SEED = 200
N_EVAL = 500
SEED = 42
GSA_THRESHOLD = 0.70

# Prompts — same as gsa_retrieval_rerun.py
BARE_PROMPT = (
    "The user has asked the following question:\n"
    "{query}\n\n"
    "Are you confident you can answer this question correctly? "
    "Respond with exactly one token: YES or NO."
)

WITH_RETRIEVAL_PROMPT = (
    "The user has asked the following question:\n"
    "{query}\n\n"
    "Here is what you recall from your knowledge base:\n"
    "{hits_block}\n\n"
    "Are you confident you can answer this question correctly? "
    "Respond with exactly one token: YES or NO."
)


def _render_hits(hits: list) -> str:
    lines = []
    for i, h in enumerate(hits, 1):
        entry = h.entry
        content = (getattr(entry, "content", "") or "")[:400].strip()
        q = getattr(entry, "question", None)
        if q:
            lines.append(f"{i}. (memory of: {q.strip()[:120]})\n   {content}")
        else:
            lines.append(f"{i}. {content}")
    return "\n".join(lines)


def seed_triviaqa_kb(
    cloud_client: LLMClient,
    embed_client: LLMClient,
    ks: KnowledgeStore,
    n_seed: int,
    seed: int,
) -> dict:
    """Seed the TriviaQA KB with cloud answers to the first n_seed questions."""
    if ks.count() >= n_seed:
        logger.info("TriviaQA KB already has %d entries; skipping", ks.count())
        return {"inserted": 0, "skipped": n_seed}

    # Load seed questions — DISJOINT from eval set.
    # Eval uses seed=42 with skip_first=N_SEED. Seed KB uses the FIRST N_SEED.
    all_q = load_triviaqa_subset(n_seed + N_EVAL, seed)
    seed_questions = all_q[:n_seed]

    inserted = 0
    failed = 0
    for i, q in enumerate(seed_questions):
        try:
            answer = cloud_client.chat(
                [ChatMessage(role="user", content=f"Answer this trivia question.\n\nQuestion: {q['question']}\nAnswer:")],
                max_tokens=100, temperature=0.0,
            )
            q_emb = embed_client.embed(q["question"]).tolist()
            a_emb = embed_client.embed(answer.content).tolist()
            ks.insert(NewKnowledgeEntry(
                content=answer.content,
                question=q["question"],
                source="cloud_escalation",
                confidence=0.7,
                tags=["trivia"],
                embedding=q_emb,
                answer_embedding=a_emb,
                domain="trivia",
                topic="triviaqa",
                metadata={"question_id": q["question_id"]},
                verbatim_response=answer.content,
            ))
            inserted += 1
        except Exception as e:
            logger.warning("Seed failed for %s: %s", q["question_id"], e)
            failed += 1
        if (i + 1) % 25 == 0:
            logger.info("Seeding: %d/%d (inserted %d, failed %d)", i + 1, n_seed, inserted, failed)

    logger.info("Seeding done: inserted=%d, failed=%d", inserted, failed)
    return {"inserted": inserted, "failed": failed}


def run_eval_with_retrieval(
    local_client: LLMClient,
    embed_client: LLMClient,
    ks: KnowledgeStore,
    model_name: str,
    out_dir: Path,
) -> dict:
    """Run TriviaQA eval with retrieval-conditional GSA + logprob."""
    # Load eval questions (skip the seed set).
    all_q = load_triviaqa_subset(N_SEED + N_EVAL, SEED)
    eval_questions = all_q[N_SEED:N_SEED + N_EVAL]
    logger.info("Eval: %d questions for %s", len(eval_questions), model_name)

    rows_path = out_dir / "rows.jsonl"
    with open(rows_path, "w") as f:
        for i, q in enumerate(eval_questions):
            try:
                # Generate answer with logprobs.
                resp = local_client.chat_with_logprobs(
                    [ChatMessage(role="user", content=f"Answer this trivia question in a few words.\n\nQuestion: {q['question']}\nAnswer:")],
                    max_tokens=50, temperature=0.0, top_logprobs=1,
                )
                answer = resp.content.strip()
                logprob_unc = extract_logprob_uncertainty(resp.avg_logprob)
                correct = check_answer(answer, q["answer_aliases"])

                # Retrieve from TriviaQA KB.
                q_emb = embed_client.embed(q["question"])
                hits_all = ks.search(q_emb, limit=5, min_similarity=0.0)
                hits_strong = [h for h in hits_all if h.score >= GSA_THRESHOLD]

                # GSA v3: retrieval-conditional.
                if hits_strong:
                    gsa_prompt = WITH_RETRIEVAL_PROMPT.format(
                        query=q["question"].strip(),
                        hits_block=_render_hits(hits_strong),
                    )
                    had_retrieval = True
                else:
                    gsa_prompt = BARE_PROMPT.format(query=q["question"].strip())
                    had_retrieval = False

                gsa_resp = local_client.chat_with_logprobs(
                    [ChatMessage(role="user", content=gsa_prompt)],
                    max_tokens=1, temperature=0.0, top_logprobs=5,
                )
                gsa_v3_p_yes = extract_gsa_p_yes(gsa_resp)

                # Also compute bare GSA for comparison.
                bare_prompt = BARE_PROMPT.format(query=q["question"].strip())
                bare_resp = local_client.chat_with_logprobs(
                    [ChatMessage(role="user", content=bare_prompt)],
                    max_tokens=1, temperature=0.0, top_logprobs=5,
                )
                gsa_bare_p_yes = extract_gsa_p_yes(bare_resp)

                # Knowledge similarity (raw max_sim).
                max_sim = max((h.score for h in hits_all), default=0.0)

                f.write(json.dumps({
                    "question_id": q["question_id"],
                    "correct": int(correct),
                    "logprob_uncertainty": logprob_unc,
                    "gsa_v3_p_yes": gsa_v3_p_yes,
                    "gsa_bare_p_yes": gsa_bare_p_yes,
                    "had_retrieval": had_retrieval,
                    "n_strong_hits": len(hits_strong),
                    "max_sim": float(max_sim),
                }) + "\n")
            except Exception as e:
                logger.warning("Query %d failed: %s", i, e)

            if (i + 1) % 50 == 0:
                logger.info("  %s: %d/%d", model_name, i + 1, len(eval_questions))

    # Aggregate.
    with open(rows_path) as f:
        rows = [json.loads(line) for line in f if line.strip()]

    n = len(rows)
    labels = np.array([r["correct"] for r in rows], dtype=np.int32)
    lp = np.array([r["logprob_uncertainty"] for r in rows], dtype=np.float64)
    gsa_v3 = np.array([r["gsa_v3_p_yes"] for r in rows], dtype=np.float64)
    gsa_bare = np.array([r["gsa_bare_p_yes"] for r in rows], dtype=np.float64)

    # Fusion: simple mean of logprob + gsa_v3.
    fusion = (lp + gsa_v3) / 2.0

    lp_a = auroc(lp, labels)
    gsa_v3_a = auroc(gsa_v3, labels)
    gsa_bare_a = auroc(gsa_bare, labels)
    fusion_a = auroc(fusion, labels)

    lp_ci = bootstrap_auroc_ci(lp, labels)
    gsa_v3_ci = bootstrap_auroc_ci(gsa_v3, labels)
    gsa_bare_ci = bootstrap_auroc_ci(gsa_bare, labels)
    fusion_ci = bootstrap_auroc_ci(fusion, labels)

    n_with_retrieval = sum(1 for r in rows if r["had_retrieval"])
    acc = float(labels.mean())

    summary = {
        "model": model_name,
        "n": n,
        "accuracy": acc,
        "n_with_retrieval": n_with_retrieval,
        "logprob_uncertainty": {"auroc": float(lp_a), "ci": [float(lp_ci[1]), float(lp_ci[2])]},
        "gsa_v3_retrieval": {"auroc": float(gsa_v3_a), "ci": [float(gsa_v3_ci[1]), float(gsa_v3_ci[2])]},
        "gsa_bare": {"auroc": float(gsa_bare_a), "ci": [float(gsa_bare_ci[1]), float(gsa_bare_ci[2])]},
        "logprob_plus_gsa_v3_fusion": {"auroc": float(fusion_a), "ci": [float(fusion_ci[1]), float(fusion_ci[2])]},
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\n  {model_name}: n={n}, acc={acc:.3f}, retrieval={n_with_retrieval}/{n}")
    print(f"    logprob_uncertainty:      {lp_a:.3f} [{lp_ci[1]:.3f}, {lp_ci[2]:.3f}]")
    print(f"    GSA v3 (w/ retrieval):    {gsa_v3_a:.3f} [{gsa_v3_ci[1]:.3f}, {gsa_v3_ci[2]:.3f}]")
    print(f"    GSA bare:                 {gsa_bare_a:.3f} [{gsa_bare_ci[1]:.3f}, {gsa_bare_ci[2]:.3f}]")
    print(f"    logprob + GSA v3 fusion:  {fusion_a:.3f} [{fusion_ci[1]:.3f}, {fusion_ci[2]:.3f}]")

    return summary


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="TriviaQA with retrieval-conditional GSA")
    p.add_argument("--local-model", default="qwen2.5:7b")
    p.add_argument("--all-models", action="store_true",
                   help="Run qwen, llama, mistral sequentially")
    p.add_argument("--cloud-model", default="us.anthropic.claude-sonnet-4-5-20250929-v1:0")
    p.add_argument("--embedding-model", default="qllama/bge-large-en-v1.5")
    p.add_argument("--bedrock-region", default="us-west-2")
    p.add_argument("--output-dir", default="results/experiment/triviaqa_with_retrieval")
    args = p.parse_args()

    models = ["qwen2.5:7b", "llama3.1:8b", "mistral:7b-instruct"] if args.all_models else [args.local_model]

    # Seed the TriviaQA KB (shared across all models — question embeddings are model-agnostic).
    conn = init_database(DB_PATH)
    ks = KnowledgeStore(conn, AutodidactConfig(db_path=DB_PATH))

    cloud_client = LLMClient(LLMConfig(
        provider="bedrock", model=args.cloud_model, region=args.bedrock_region,
    ))
    embed_client = LLMClient(LLMConfig(
        provider="ollama", model="qwen2.5:7b", embedding_model=args.embedding_model,
    ))

    logger.info("Seeding TriviaQA KB (%d entries)...", N_SEED)
    seed_result = seed_triviaqa_kb(cloud_client, embed_client, ks, N_SEED, SEED)
    logger.info("KB has %d entries after seeding", ks.count())

    # Run eval for each model.
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    base_dir = Path(args.output_dir) / stamp
    base_dir.mkdir(parents=True, exist_ok=True)

    all_summaries = {}
    for model in models:
        logger.info("Running eval for %s...", model)
        local_client = LLMClient(LLMConfig(
            provider="ollama", model=model, embedding_model=args.embedding_model,
        ))
        model_dir = base_dir / model.replace(":", "_").replace("/", "_")
        model_dir.mkdir(parents=True, exist_ok=True)
        summary = run_eval_with_retrieval(local_client, embed_client, ks, model, model_dir)
        all_summaries[model] = summary

    # Print comparison table.
    print(f"\n{'='*70}")
    print(f" TriviaQA with Retrieval — Cross-Model Comparison")
    print(f"{'='*70}")
    print(f"{'':>25}  {'logprob':>8}  {'GSA v3':>8}  {'GSA bare':>8}  {'fusion':>8}")
    print("-" * 70)
    for model, s in all_summaries.items():
        short = model.split(":")[0]
        print(f"  {short:>23}  {s['logprob_uncertainty']['auroc']:>8.3f}  "
              f"{s['gsa_v3_retrieval']['auroc']:>8.3f}  "
              f"{s['gsa_bare']['auroc']:>8.3f}  "
              f"{s['logprob_plus_gsa_v3_fusion']['auroc']:>8.3f}")
    print(f"{'='*70}\n")
    print(f"Artifacts: {base_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

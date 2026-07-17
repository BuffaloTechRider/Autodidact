"""v0.1 ablation experiment harness.

Orchestrates the full run:
1. Load disjoint evaluation (seed 42, 500 queries) and training (seed 43, 1000) corpora.
2. Seed the knowledge store with the first 100 evaluation queries (cloud answers,
   question embeddings).
3. For each of the remaining 400 evaluation queries, compute all 6 confidence signals
   plus both frozen RouteLLM baseline scores, generate local + cloud answers, label
   correctness, and write one row to `experiment_results`.

Resumable: skips query indices that already have a row for the given run_id.

Requirement 3 (R3) from the v0.1 spec is satisfied here.

Run:
    python -m benchmarks.ablation_experiment \\
        --run-id v0.1-$(date +%Y%m%d) \\
        --local-model qwen2.5:7b

The Bedrock credentials are picked up from the standard AWS chain; the Ollama
host from OLLAMA_HOST (defaults to http://localhost:11434).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import numpy as np

from autodidact.confidence_evaluator import ConfidenceEvaluator
from autodidact.database import init_database
from autodidact.knowledge_store import KnowledgeStore
from autodidact.llm_client import ChatMessage, LLMClient, LLMConfig
from autodidact.signals.grounded_self_assessment import GroundedSelfAssessment
from autodidact.types import AutodidactConfig

from benchmarks.datasets import MMLUProQuery, load_two_disjoint_subsets
from benchmarks.labeling import label_answer
from benchmarks.routellm_baseline import RouteLLMModels, load_routellm_models, train_routellm_baselines
from benchmarks.seeding import seed_knowledge_store

logger = logging.getLogger(__name__)

# Rough cost estimates for MMLU-Pro queries, in USD per million tokens.
# Adjust if you use different models.
COST_RATES = {
    # ON_DEMAND models (call by plain model ID)
    "anthropic.claude-3-haiku-20240307-v1:0": {"input": 0.25, "output": 1.25},
    "anthropic.claude-3-sonnet-20240229-v1:0": {"input": 3.0, "output": 15.0},
    "anthropic.claude-3-5-haiku-20241022-v1:0": {"input": 0.80, "output": 4.0},

    # INFERENCE_PROFILE models (must call via us.* profile ID)
    "us.anthropic.claude-3-haiku-20240307-v1:0": {"input": 0.25, "output": 1.25},
    "us.anthropic.claude-3-sonnet-20240229-v1:0": {"input": 3.0, "output": 15.0},
    "us.anthropic.claude-3-5-haiku-20241022-v1:0": {"input": 0.80, "output": 4.0},
    "us.anthropic.claude-3-5-sonnet-20241022-v2:0": {"input": 3.0, "output": 15.0},
    "us.anthropic.claude-3-7-sonnet-20250219-v1:0": {"input": 3.0, "output": 15.0},
    "us.anthropic.claude-sonnet-4-20250514-v1:0": {"input": 3.0, "output": 15.0},
    "us.anthropic.claude-sonnet-4-5-20250929-v1:0": {"input": 3.0, "output": 15.0},
    "us.anthropic.claude-sonnet-4-6": {"input": 3.0, "output": 15.0},
    "us.anthropic.claude-opus-4-20250514-v1:0": {"input": 15.0, "output": 75.0},
    "us.anthropic.claude-opus-4-1-20250805-v1:0": {"input": 15.0, "output": 75.0},
    "us.anthropic.claude-opus-4-5-20251101-v1:0": {"input": 5.0, "output": 25.0},
    "us.anthropic.claude-opus-4-7": {"input": 5.0, "output": 25.0},
}


@dataclass
class HarnessConfig:
    run_id: str
    db_path: str
    output_dir: str
    n_seed: int
    n_eval: int
    n_training: int
    eval_seed: int
    train_seed: int
    local_model: str
    cloud_model: str
    judge_model: str
    embedding_model: str
    ollama_host: Optional[str]
    bedrock_region: str
    confirm_cost: bool
    cost_threshold_usd: float


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Autodidact v0.1 ablation experiment harness")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--db-path", default="autodidact_experiment.db")
    parser.add_argument("--output-dir", default="results/experiment")
    parser.add_argument("--n-seed", type=int, default=1000)
    parser.add_argument("--n-eval", type=int, default=1000)
    parser.add_argument("--n-training", type=int, default=1000,
                        help="RouteLLM baseline training corpus size. "
                             "Keep at 1000 for the real run; lower for dry-runs.")
    parser.add_argument("--eval-seed", type=int, default=42)
    parser.add_argument("--train-seed", type=int, default=43)
    parser.add_argument("--local-model", default="qwen2.5:7b")
    parser.add_argument("--cloud-model", default="us.anthropic.claude-sonnet-4-5-20250929-v1:0")
    parser.add_argument("--judge-model", default="us.anthropic.claude-opus-4-5-20251101-v1:0")
    parser.add_argument("--embedding-model", default="qllama/bge-large-en-v1.5")
    parser.add_argument("--bedrock-region", default="us-west-2")
    parser.add_argument("--confirm-cost", action="store_true",
                        help="Pass to acknowledge cost estimates over threshold")
    parser.add_argument("--cost-threshold-usd", type=float, default=5.0)
    parser.add_argument("--train-baselines", action="store_true",
                        help="Train the RouteLLM baselines if they're missing")
    args = parser.parse_args()

    cfg = HarnessConfig(
        run_id=args.run_id,
        db_path=args.db_path,
        output_dir=args.output_dir,
        n_seed=args.n_seed,
        n_eval=args.n_eval,
        n_training=args.n_training,
        eval_seed=args.eval_seed,
        train_seed=args.train_seed,
        local_model=args.local_model,
        cloud_model=args.cloud_model,
        judge_model=args.judge_model,
        embedding_model=args.embedding_model,
        ollama_host=os.environ.get("OLLAMA_HOST"),
        bedrock_region=args.bedrock_region,
        confirm_cost=args.confirm_cost,
        cost_threshold_usd=args.cost_threshold_usd,
    )
    return run_experiment(cfg, train_baselines_if_missing=args.train_baselines)


def run_experiment(cfg: HarnessConfig, *, train_baselines_if_missing: bool = False) -> int:
    os.makedirs(cfg.output_dir, exist_ok=True)
    conn = init_database(cfg.db_path)

    # Build LLM clients
    local_client = LLMClient(LLMConfig(
        provider="ollama", model=cfg.local_model,
        embedding_model=cfg.embedding_model,
    ))
    embedding_client = local_client  # same Ollama client handles embeddings
    cloud_client = LLMClient(LLMConfig(
        provider="bedrock", model=cfg.cloud_model, region=cfg.bedrock_region,
    ))
    judge_client = LLMClient(LLMConfig(
        provider="bedrock", model=cfg.judge_model, region=cfg.bedrock_region,
    ))

    # Load corpora
    eval_queries, training_queries = load_two_disjoint_subsets(
        n_eval=cfg.n_seed + cfg.n_eval,
        n_train=cfg.n_training,
        eval_seed=cfg.eval_seed,
        train_seed=cfg.train_seed,
    )
    logger.info(
        "Loaded %d eval queries and %d training queries",
        len(eval_queries), len(training_queries),
    )
    seeding_queries = eval_queries[: cfg.n_seed]
    evaluation_queries = eval_queries[cfg.n_seed :]

    # Cost estimate
    est = _estimate_cost(cfg, n_eval=len(evaluation_queries), n_seed=len(seeding_queries))
    logger.info("Estimated cloud cost: $%.2f (threshold $%.2f)", est, cfg.cost_threshold_usd)
    if est > cfg.cost_threshold_usd and not cfg.confirm_cost:
        logger.error(
            "Estimated cost $%.2f exceeds threshold $%.2f. "
            "Rerun with --confirm-cost to proceed.",
            est, cfg.cost_threshold_usd,
        )
        return 2

    # Knowledge store (seeded with question embeddings)
    kb_config = AutodidactConfig(db_path=cfg.db_path)
    ks = KnowledgeStore(conn, kb_config)
    logger.info("Seeding knowledge store (skip if already populated)")
    seed_result = seed_knowledge_store(
        queries=seeding_queries,
        cloud_client=cloud_client,
        embedding_client=embedding_client,
        knowledge_store=ks,
    )
    logger.info("Seeding result: %s", seed_result)

    # Data-leakage guard: the KB entries are drawn from the first `n_seed` eval
    # queries and the evaluation loop iterates `eval_queries[n_seed:]`. We
    # need KB count ≤ n_seed. If KB > n_seed, some queries in the eval set
    # would already be in memory (leakage). If KB < n_seed by a small margin,
    # that's a seeding failure (e.g., query too long for the embedder's context
    # window) which we tolerate. We warn on missing entries but don't abort.
    kb_actual = ks.count()
    if kb_actual > cfg.n_seed:
        logger.error(
            "Data-leakage guard tripped: n_seed=%d but knowledge_store.count()=%d "
            "(KB is LARGER than expected). Eval queries may already be in the KB. "
            "Fix: drop the KB and reseed, OR set --n-seed to match the current "
            "KB size (%d). Refusing to proceed.",
            cfg.n_seed, kb_actual, kb_actual,
        )
        return 2
    if kb_actual < cfg.n_seed:
        missing = cfg.n_seed - kb_actual
        tolerance = max(10, cfg.n_seed // 100)  # 1% or at least 10 entries
        if missing > tolerance:
            logger.error(
                "Seeding deficit too large: n_seed=%d but KB has %d entries "
                "(missing %d > tolerance %d). Check seeding logs for systematic "
                "failures. Refusing to proceed.",
                cfg.n_seed, kb_actual, missing, tolerance,
            )
            return 2
        logger.warning(
            "Seeding deficit tolerated: n_seed=%d, KB has %d entries "
            "(missing %d, tolerance %d). Proceeding.",
            cfg.n_seed, kb_actual, missing, tolerance,
        )

    # RouteLLM baselines
    try:
        models = load_routellm_models(
            cfg.output_dir, conn,
            training_seed=cfg.train_seed,
            local_model=cfg.local_model,
        )
        logger.info(
            "Loaded RouteLLM baselines (trained on %d rows, dim=%d)",
            models.n_training_rows, models.embedding_dim,
        )
    except FileNotFoundError:
        if not train_baselines_if_missing:
            logger.error(
                "RouteLLM baselines not found in %s. "
                "Run with --train-baselines, or run `python -m benchmarks.routellm_baseline` first.",
                cfg.output_dir,
            )
            return 2
        logger.info("RouteLLM baselines missing; training them now (expensive)")
        models = train_routellm_baselines(
            training_queries=training_queries,
            local_client=local_client,
            cloud_client=cloud_client,
            judge_client=judge_client,
            embedding_client=embedding_client,
            knowledge_store=ks,
            conn=conn,
            training_seed=cfg.train_seed,
            output_dir=cfg.output_dir,
            local_model=cfg.local_model,
        )

    # Confidence evaluator (reuses the existing signal math) and GSA signal
    evaluator = ConfidenceEvaluator(conn, kb_config)
    gsa = GroundedSelfAssessment(local_client)

    # Evaluation loop
    logger.info("Starting evaluation loop over %d queries", len(evaluation_queries))
    total = len(evaluation_queries)
    wrote = 0
    skipped = 0
    failed = 0

    for idx, q in enumerate(evaluation_queries, start=cfg.n_seed):
        if _row_exists(conn, cfg.run_id, idx):
            skipped += 1
            continue
        try:
            _process_query(
                conn=conn, cfg=cfg, run_id=cfg.run_id,
                query=q, query_index=idx,
                local_client=local_client, cloud_client=cloud_client,
                judge_client=judge_client, embedding_client=embedding_client,
                knowledge_store=ks, evaluator=evaluator, gsa=gsa,
                models=models,
            )
            wrote += 1
        except Exception as e:
            failed += 1
            logger.warning("Query %d (%s) failed: %s", idx, q.query_id, e)
            _record_failure_row(conn, cfg.run_id, idx, q, str(e))

        if (idx + 1) % 20 == 0:
            logger.info(
                "Progress: %d/%d (wrote %d, skipped %d, failed %d)",
                idx + 1 - cfg.n_seed, total, wrote, skipped, failed,
            )

    logger.info(
        "Experiment complete. wrote=%d skipped=%d failed=%d (total=%d)",
        wrote, skipped, failed, total,
    )
    return 0 if failed == 0 else 1


# ── Per-query processing ───────────────────────────────────────────

def _process_query(
    *,
    conn,
    cfg: HarnessConfig,
    run_id: str,
    query: MMLUProQuery,
    query_index: int,
    local_client: LLMClient,
    cloud_client: LLMClient,
    judge_client: LLMClient,
    embedding_client: LLMClient,
    knowledge_store: KnowledgeStore,
    evaluator: ConfidenceEvaluator,
    gsa: GroundedSelfAssessment,
    models: RouteLLMModels,
) -> None:
    """Run the full signal pipeline on one evaluation query and persist one row."""

    # ── Pre-gen signals ──
    # One FAISS search at threshold=0.0; each consumer post-filters to its own threshold.
    # Per EXP-002, different consumers want different floors:
    #   - knowledge_similarity feature:  no floor (0.0) — raw gradient for ML
    #   - GSA prompt:                    0.70 — strong hits or honest absence
    #   - answer-injection prompt:       0.60 — enough context to help the answer
    t0 = time.perf_counter()
    q_embedding = embedding_client.embed(query.query_text)
    hits_unfiltered = knowledge_store.search(q_embedding, limit=5, min_similarity=0.0)
    hit_embeddings = [
        np.asarray(h.entry.embedding, dtype=np.float32)
        for h in hits_unfiltered
        if h.entry.embedding is not None
    ]
    knowledge_similarity = evaluator.compute_knowledge_similarity(q_embedding, hit_embeddings)
    t_ks = _elapsed_ms(t0)

    # (The answer-injection consumer at threshold 0.60 is deliberately NOT
    #  wired into the main harness. The v0.1 ablation measures raw signals;
    #  retrieval injection is a separate product choice tested in
    #  answer_quality_study. Wiring both changes what each signal measures.)

    t0 = time.perf_counter()
    query_classification = evaluator.compute_query_classification(query.query_text)
    t_qc = _elapsed_ms(t0)

    t0 = time.perf_counter()
    energy_score = evaluator.compute_energy_score(q_embedding)
    t_es = _elapsed_ms(t0)

    # ── GSA (one small generation) ──
    # v3 retrieval-conditional: SelfAssessment applies its own `min_similarity`
    # threshold (default 0.70). We pass the unfiltered top-5 scored hits; the
    # signal decides per-query whether to show retrieval or fall back to the
    # bare prompt (indistinguishable from never-searched). EXP-005 validated
    # this design: +0.037 AUROC over v2 baseline on qwen2.5:7b n=931.
    t0 = time.perf_counter()
    gsa_result = gsa.compute(query.query_text, hits_unfiltered)
    grounded_self_assessment = gsa_result.p_yes
    gsa_extraction_mode = gsa_result.extraction_mode
    t_gsa = _elapsed_ms(t0)

    # ── Full local generation (used for logprob_uncertainty AND as the local answer) ──
    t0 = time.perf_counter()
    local_with_lp = local_client.chat_with_logprobs(
        [ChatMessage(role="user", content=query.formatted_prompt())],
        max_tokens=512, temperature=0.0, top_logprobs=1,
    )
    t_lp = _elapsed_ms(t0)
    local_answer = local_with_lp.content
    avg_logprob = local_with_lp.avg_logprob if local_with_lp.avg_logprob is not None else -1.5
    logprob_uncertainty = evaluator.compute_logprob_uncertainty(avg_logprob)

    # ── Second local generation for self-consistency ──
    t0 = time.perf_counter()
    # Use a slightly different temperature and seed to actually get a different sample.
    local_sc = local_client.chat(
        [ChatMessage(role="user", content=query.formatted_prompt())],
        max_tokens=512, temperature=0.7, seed=7,
    )
    t_sc = _elapsed_ms(t0)
    self_consistency = evaluator.compute_self_consistency(local_answer, local_sc.content)

    # ── RouteLLM baseline scores ──
    routellm_no_memory = models.score_no_memory(q_embedding)
    routellm_plus_ks = models.score_plus_ks(q_embedding, knowledge_similarity)

    # ── Retrieval quality proxy (category match among top-5) ──
    # Measured on the UNFILTERED top-5 so this is a threshold-free retrieval-
    # quality metric, not a reflection of the per-consumer thresholds.
    retrieval_recall_at_5 = 0
    for h in hits_unfiltered:
        if h.entry.domain == query.category:
            retrieval_recall_at_5 = 1
            break

    # ── Cloud answer for ground-truth comparison ──
    cloud_resp = cloud_client.chat(
        [ChatMessage(role="user", content=query.formatted_prompt())],
        max_tokens=512, temperature=0.0,
    )
    cloud_answer = cloud_resp.content

    # ── Labels ──
    local_correct, local_used_judge = label_answer(
        local_answer, query.ground_truth_letter, query.ground_truth_answer,
        query.query_text, judge_client=judge_client,
    )
    cloud_correct, cloud_used_judge = label_answer(
        cloud_answer, query.ground_truth_letter, query.ground_truth_answer,
        query.query_text, judge_client=judge_client,
    )
    judge_used = int(local_used_judge or cloud_used_judge)

    # ── Cost accounting (rough) ──
    cloud_cost = _cost_of(cfg.cloud_model, cloud_resp.input_tokens, cloud_resp.output_tokens)
    judge_cost = 0.0
    if local_used_judge:
        judge_cost += _cost_of(cfg.judge_model, 300, 4)
    if cloud_used_judge:
        judge_cost += _cost_of(cfg.judge_model, 300, 4)

    # ── Insert row ──
    conn.execute(
        """INSERT INTO experiment_results (
            id, run_id, query_index, query_id, query_text, ground_truth, category,
            knowledge_similarity, query_classification, energy_scorer,
            grounded_self_assessment, logprob_uncertainty, self_consistency,
            latency_knowledge_similarity, latency_query_classification, latency_energy_scorer,
            latency_grounded_self_assessment, latency_logprob_uncertainty, latency_self_consistency,
            local_answer, local_avg_logprob, cloud_answer,
            routellm_no_memory, routellm_plus_ks, retrieval_recall_at_5,
            gsa_extraction_mode,
            local_correct, cloud_correct, judge_used,
            cost_cloud_answer_usd, cost_judge_usd, cost_local_usd,
            error_info, created_at
        ) VALUES (?,?,?,?,?,?,?, ?,?,?, ?,?,?, ?,?,?, ?,?,?, ?,?,?, ?,?,?, ?, ?,?,?, ?,?,?, ?,?)""",
        (
            str(uuid.uuid4()), run_id, query_index, query.query_id,
            query.query_text, query.ground_truth_letter, query.category,
            knowledge_similarity, query_classification, energy_score,
            grounded_self_assessment, logprob_uncertainty, self_consistency,
            t_ks, t_qc, t_es,
            t_gsa, t_lp, t_sc,
            local_answer, local_with_lp.avg_logprob, cloud_answer,
            routellm_no_memory, routellm_plus_ks, retrieval_recall_at_5,
            gsa_extraction_mode,
            int(local_correct), int(cloud_correct), judge_used,
            cloud_cost, judge_cost, 0.0,
            None, datetime.now(timezone.utc).isoformat(),
        ),
    )
    conn.commit()


# ── Helpers ────────────────────────────────────────────────────────

def _row_exists(conn, run_id: str, query_index: int) -> bool:
    row = conn.execute(
        "SELECT 1 FROM experiment_results WHERE run_id = ? AND query_index = ?",
        (run_id, query_index),
    ).fetchone()
    return row is not None


def _record_failure_row(conn, run_id: str, query_index: int, q: MMLUProQuery, error: str) -> None:
    """Insert a placeholder row for a failed query so it's not retried silently."""
    try:
        conn.execute(
            """INSERT INTO experiment_results (
                id, run_id, query_index, query_id, query_text, ground_truth, category,
                knowledge_similarity, query_classification, energy_scorer,
                grounded_self_assessment, logprob_uncertainty, self_consistency,
                latency_knowledge_similarity, latency_query_classification, latency_energy_scorer,
                latency_grounded_self_assessment, latency_logprob_uncertainty, latency_self_consistency,
                local_answer, local_avg_logprob, cloud_answer,
                routellm_no_memory, routellm_plus_ks, retrieval_recall_at_5,
                gsa_extraction_mode,
                local_correct, cloud_correct, judge_used,
                cost_cloud_answer_usd, cost_judge_usd, cost_local_usd,
                error_info, created_at
            ) VALUES (?,?,?,?,?,?,?, ?,?,?, ?,?,?, ?,?,?, ?,?,?, ?,?,?, ?,?,?, ?, ?,?,?, ?,?,?, ?,?)""",
            (
                str(uuid.uuid4()), run_id, query_index, q.query_id,
                q.query_text, q.ground_truth_letter, q.category,
                0.0, 0.5, None, 0.5, 0.5, 0.0,
                0, 0, None, 0, 0, 0,
                "", None, "",
                0.5, 0.5, 0,
                "neutral",
                0, 0, 0,
                0.0, 0.0, 0.0,
                error[:500], datetime.now(timezone.utc).isoformat(),
            ),
        )
        conn.commit()
    except Exception as e:
        logger.error("Failed to record failure row for index %d: %s", query_index, e)


def _elapsed_ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)


def _cost_of(model: str, in_tokens: int, out_tokens: int) -> float:
    rates = COST_RATES.get(model)
    if rates is None:
        return 0.0
    return (in_tokens * rates["input"] + out_tokens * rates["output"]) / 1_000_000.0


def _estimate_cost(cfg: HarnessConfig, n_eval: int, n_seed: int) -> float:
    """Rough pre-flight cost estimate. Assumes ~1500 output tokens per cloud answer,
    30% judge-call rate, ~100 output tokens per judge call."""
    # Seeding: n_seed cloud answers
    c_seed = n_seed * _cost_of(cfg.cloud_model, 500, 1500)
    # Eval: n_eval cloud answers + ~30% judge calls (both local and cloud response)
    c_eval = n_eval * _cost_of(cfg.cloud_model, 500, 1500)
    c_judge = n_eval * 0.6 * _cost_of(cfg.judge_model, 300, 4)
    return c_seed + c_eval + c_judge


if __name__ == "__main__":
    sys.exit(main())

"""RouteLLM-style baseline trainers for the v0.1 ablation.

Trains two logistic regression classifiers on a disjoint 1000-query training
corpus, frozen at inference. These are the fair comparison point for the
memo: what a static supervised router would do given the same data.

- routellm_no_memory: query_embedding -> P(local_is_sufficient)
- routellm_plus_ks:  [query_embedding, knowledge_similarity] -> P(local_is_sufficient)

Both pickled to disk so the main experiment harness can load and score them
per evaluation query.

Requirement 4 (R4) from the v0.1 spec is satisfied here.
"""

from __future__ import annotations

import logging
import os
import pickle
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import numpy as np
from sklearn.linear_model import LogisticRegressionCV

from autodidact.knowledge_store import KnowledgeStore
from autodidact.llm_client import ChatMessage, LLMClient
from benchmarks.datasets import MMLUProQuery
from benchmarks.labeling import label_answer

logger = logging.getLogger(__name__)

NO_MEMORY_FILENAME = "routellm_no_memory.pkl"
PLUS_KS_FILENAME = "routellm_plus_ks.pkl"


@dataclass
class RouteLLMModels:
    """The two frozen classifiers plus metadata for scoring at inference."""

    no_memory: LogisticRegressionCV
    plus_ks: LogisticRegressionCV
    embedding_dim: int
    training_seed: int
    n_training_rows: int

    def score_no_memory(self, query_embedding: np.ndarray) -> float:
        """P(local_is_sufficient) given the query embedding alone."""
        x = query_embedding.reshape(1, -1).astype(np.float32)
        proba = self.no_memory.predict_proba(x)[0]
        # Class index for label 1 ("local_is_sufficient" true)
        idx = list(self.no_memory.classes_).index(1)
        return float(proba[idx])

    def score_plus_ks(self, query_embedding: np.ndarray, knowledge_similarity: float) -> float:
        """P(local_is_sufficient) given embedding + knowledge similarity feature."""
        x = np.concatenate([query_embedding, [float(knowledge_similarity)]]).reshape(1, -1).astype(np.float32)
        proba = self.plus_ks.predict_proba(x)[0]
        idx = list(self.plus_ks.classes_).index(1)
        return float(proba[idx])


def train_routellm_baselines(
    training_queries: list[MMLUProQuery],
    local_client: LLMClient,
    cloud_client: LLMClient,
    judge_client: LLMClient,
    embedding_client: LLMClient,
    knowledge_store: KnowledgeStore,
    conn: sqlite3.Connection,
    training_seed: int = 43,
    output_dir: str = "results/experiment",
    force_retrain: bool = False,
    local_model: str = "",
) -> RouteLLMModels:
    """Train and persist both RouteLLM-style baselines.

    Uses the `routellm_training_rows` table to cache labeled rows so reruns
    are cheap. If both pickle files exist and the cached-row count matches
    `len(training_queries)` for the seed AND local_model, skips training
    entirely (unless `force_retrain=True`).

    The `local_model` parameter scopes the cache by the local model that
    generated the labels. Different local models produce different
    `local_is_sufficient` labels on the same training queries; scoping by
    model prevents cross-model label contamination when multiple models
    share a DB (as in the cross-model main experiment).
    """
    os.makedirs(output_dir, exist_ok=True)
    nm_path = os.path.join(output_dir, NO_MEMORY_FILENAME)
    pks_path = os.path.join(output_dir, PLUS_KS_FILENAME)

    if (
        not force_retrain
        and os.path.exists(nm_path)
        and os.path.exists(pks_path)
        and _count_cached_rows(conn, training_seed, local_model) >= len(training_queries)
    ):
        logger.info(
            "RouteLLM baselines already trained for seed %d, model %r; loading from disk",
            training_seed, local_model,
        )
        return _load_models(nm_path, pks_path, training_seed, conn, local_model)

    # Step 1. Produce (or reuse cached) labeled training rows.
    _ensure_all_training_rows(
        training_queries=training_queries,
        local_client=local_client,
        cloud_client=cloud_client,
        judge_client=judge_client,
        embedding_client=embedding_client,
        knowledge_store=knowledge_store,
        conn=conn,
        training_seed=training_seed,
        local_model=local_model,
    )

    # Step 2. Load all cached rows and build training matrices.
    X_nm, X_pks, y = _load_training_matrices(conn, training_seed, local_model)
    if X_nm.shape[0] < 50:
        raise RuntimeError(
            f"Only {X_nm.shape[0]} training rows available; need at least 50 for CV"
        )
    if len(set(y.tolist())) < 2:
        raise RuntimeError(
            "Training labels are all one class; cannot train a binary classifier. "
            "This usually means the local model answered every query correctly "
            "(or every one wrong) for the training corpus."
        )

    # Step 3. Train both classifiers with 5-fold CV for regularization.
    logger.info("Training routellm_no_memory on %d rows", X_nm.shape[0])
    no_memory = LogisticRegressionCV(
        cv=5, max_iter=2000, random_state=training_seed, n_jobs=-1, scoring="roc_auc"
    ).fit(X_nm, y)
    logger.info("Training routellm_plus_ks on %d rows", X_pks.shape[0])
    plus_ks = LogisticRegressionCV(
        cv=5, max_iter=2000, random_state=training_seed, n_jobs=-1, scoring="roc_auc"
    ).fit(X_pks, y)

    # Step 4. Persist.
    with open(nm_path, "wb") as f:
        pickle.dump(no_memory, f)
    with open(pks_path, "wb") as f:
        pickle.dump(plus_ks, f)
    logger.info("Pickled RouteLLM models to %s and %s", nm_path, pks_path)

    return RouteLLMModels(
        no_memory=no_memory,
        plus_ks=plus_ks,
        embedding_dim=X_nm.shape[1],
        training_seed=training_seed,
        n_training_rows=X_nm.shape[0],
    )


def load_routellm_models(
    output_dir: str,
    conn: sqlite3.Connection,
    training_seed: int = 43,
    local_model: str = "",
) -> RouteLLMModels:
    """Load pre-trained models from disk. Raises if either file is missing."""
    nm_path = os.path.join(output_dir, NO_MEMORY_FILENAME)
    pks_path = os.path.join(output_dir, PLUS_KS_FILENAME)
    if not os.path.exists(nm_path) or not os.path.exists(pks_path):
        raise FileNotFoundError(
            f"RouteLLM baselines not found at {output_dir}. "
            "Run `python -m benchmarks.routellm_baseline` first."
        )
    return _load_models(nm_path, pks_path, training_seed, conn, local_model)


# ── Internals ──────────────────────────────────────────────────────

def _count_cached_rows(conn: sqlite3.Connection, training_seed: int, local_model: str) -> int:
    row = conn.execute(
        "SELECT COUNT(*) AS cnt FROM routellm_training_rows "
        "WHERE training_seed = ? AND local_model = ?",
        (training_seed, local_model),
    ).fetchone()
    return int(row["cnt"]) if row is not None else 0


def _ensure_all_training_rows(
    *,
    training_queries: list[MMLUProQuery],
    local_client: LLMClient,
    cloud_client: LLMClient,
    judge_client: LLMClient,
    embedding_client: LLMClient,
    knowledge_store: KnowledgeStore,
    conn: sqlite3.Connection,
    training_seed: int,
    local_model: str,
) -> None:
    """For each training query, ensure a labeled row exists in routellm_training_rows."""
    existing_ids = {
        row["query_id"]
        for row in conn.execute(
            "SELECT query_id FROM routellm_training_rows "
            "WHERE training_seed = ? AND local_model = ?",
            (training_seed, local_model),
        ).fetchall()
    }
    logger.info(
        "RouteLLM training: %d/%d cached for model %r, %d to label",
        len(existing_ids), len(training_queries), local_model,
        len(training_queries) - len(existing_ids),
    )

    for i, q in enumerate(training_queries):
        if q.query_id in existing_ids:
            continue
        try:
            # Local answer
            local_resp = local_client.chat(
                [ChatMessage(role="user", content=q.formatted_prompt())],
                max_tokens=512, temperature=0.0,
            )
            # Cloud answer
            cloud_resp = cloud_client.chat(
                [ChatMessage(role="user", content=q.formatted_prompt())],
                max_tokens=512, temperature=0.0,
            )
            # Labels
            local_correct, local_used_judge = label_answer(
                local_resp.content, q.ground_truth_letter, q.ground_truth_answer,
                q.query_text, judge_client=judge_client,
            )
            cloud_correct, cloud_used_judge = label_answer(
                cloud_resp.content, q.ground_truth_letter, q.ground_truth_answer,
                q.query_text, judge_client=judge_client,
            )
            # Features
            q_embedding = embedding_client.embed(q.query_text)
            hits = knowledge_store.search(q_embedding, limit=5)
            ks = max((h.score for h in hits), default=0.0)

            conn.execute(
                """INSERT INTO routellm_training_rows
                (id, training_seed, local_model, query_id, query_text, query_embedding,
                 knowledge_similarity, local_answer, cloud_answer,
                 local_correct, cloud_correct, local_is_sufficient,
                 cost_cloud_usd, cost_judge_usd, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    str(uuid.uuid4()),
                    training_seed,
                    local_model,
                    q.query_id,
                    q.query_text,
                    q_embedding.astype(np.float32).tobytes(),
                    float(ks),
                    local_resp.content,
                    cloud_resp.content,
                    int(local_correct),
                    int(cloud_correct),
                    int(local_correct),  # local_is_sufficient == local_correct for MMLU-Pro
                    0.0,  # cost accounting deferred to the full experiment; baseline is cheap enough
                    0.0,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            conn.commit()

            if (i + 1) % 50 == 0:
                logger.info(
                    "RouteLLM training labels: %d/%d done",
                    i + 1, len(training_queries),
                )
        except Exception as e:
            logger.warning("Failed to label training row for %s: %s", q.query_id, e)


def _load_training_matrices(
    conn: sqlite3.Connection, training_seed: int, local_model: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load cached rows into X_no_memory, X_plus_ks, y arrays."""
    rows = conn.execute(
        """SELECT query_embedding, knowledge_similarity, local_is_sufficient
           FROM routellm_training_rows
           WHERE training_seed = ? AND local_model = ?
           ORDER BY created_at""",
        (training_seed, local_model),
    ).fetchall()
    if not rows:
        raise RuntimeError(
            f"No cached training rows for seed {training_seed}, model {local_model!r}"
        )

    first_emb = np.frombuffer(rows[0]["query_embedding"], dtype=np.float32)
    dim = len(first_emb)

    X_nm = np.zeros((len(rows), dim), dtype=np.float32)
    X_pks = np.zeros((len(rows), dim + 1), dtype=np.float32)
    y = np.zeros(len(rows), dtype=np.int32)

    for i, row in enumerate(rows):
        emb = np.frombuffer(row["query_embedding"], dtype=np.float32)
        X_nm[i] = emb
        X_pks[i, :dim] = emb
        X_pks[i, dim] = float(row["knowledge_similarity"])
        y[i] = int(row["local_is_sufficient"])

    return X_nm, X_pks, y


def _load_models(
    nm_path: str, pks_path: str, training_seed: int, conn: sqlite3.Connection,
    local_model: str,
) -> RouteLLMModels:
    with open(nm_path, "rb") as f:
        no_memory = pickle.load(f)
    with open(pks_path, "rb") as f:
        plus_ks = pickle.load(f)
    n_rows = _count_cached_rows(conn, training_seed, local_model)
    dim = int(no_memory.coef_.shape[1])
    return RouteLLMModels(
        no_memory=no_memory,
        plus_ks=plus_ks,
        embedding_dim=dim,
        training_seed=training_seed,
        n_training_rows=n_rows,
    )


# ── CLI entry point ─────────────────────────────────────────────────

def _cli_main() -> int:
    """Standalone entry point: train both RouteLLM baselines and persist them.

    Reuses the same seeding function as the main experiment so the knowledge
    store (needed to compute the `knowledge_similarity` feature for
    `routellm_plus_ks`) is populated from the same 100-query seed the main
    experiment will use.
    """
    import argparse
    import sys
    from autodidact.database import init_database
    from autodidact.knowledge_store import KnowledgeStore
    from autodidact.llm_client import LLMClient, LLMConfig
    from autodidact.types import AutodidactConfig
    from benchmarks.datasets import load_two_disjoint_subsets
    from benchmarks.seeding import seed_knowledge_store

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(
        description="Train RouteLLM baseline classifiers for the v0.1 ablation"
    )
    parser.add_argument("--db-path", default="autodidact_experiment.db")
    parser.add_argument("--output-dir", default="results/experiment")
    parser.add_argument("--n-seed", type=int, default=1000)
    parser.add_argument("--n-eval", type=int, default=1000,
                        help="Eval corpus size (only used to sample a disjoint training set)")
    parser.add_argument("--n-training", type=int, default=1000)
    parser.add_argument("--eval-seed", type=int, default=42)
    parser.add_argument("--train-seed", type=int, default=43)
    parser.add_argument("--local-model", default="qwen2.5:7b")
    parser.add_argument("--cloud-model", default="us.anthropic.claude-sonnet-4-5-20250929-v1:0")
    parser.add_argument("--judge-model", default="us.anthropic.claude-opus-4-5-20251101-v1:0")
    parser.add_argument("--embedding-model", default="qllama/bge-large-en-v1.5")
    parser.add_argument("--bedrock-region", default="us-west-2")
    parser.add_argument("--force-retrain", action="store_true",
                        help="Re-train even if pickled classifiers already exist")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    conn = init_database(args.db_path)

    local_client = LLMClient(LLMConfig(
        provider="ollama", model=args.local_model,
        embedding_model=args.embedding_model,
    ))
    cloud_client = LLMClient(LLMConfig(
        provider="bedrock", model=args.cloud_model, region=args.bedrock_region,
    ))
    judge_client = LLMClient(LLMConfig(
        provider="bedrock", model=args.judge_model, region=args.bedrock_region,
    ))
    embedding_client = local_client  # same Ollama client handles embeddings

    # Load disjoint corpora so the training set doesn't overlap with the eval set
    eval_queries, training_queries = load_two_disjoint_subsets(
        n_eval=args.n_seed + args.n_eval,
        n_train=args.n_training,
        eval_seed=args.eval_seed,
        train_seed=args.train_seed,
    )
    seeding_queries = eval_queries[: args.n_seed]
    logger.info(
        "Loaded %d seeding queries and %d training queries",
        len(seeding_queries), len(training_queries),
    )

    # Ensure the knowledge store is seeded (shared state with the main experiment)
    kb_config = AutodidactConfig(db_path=args.db_path)
    ks = KnowledgeStore(conn, kb_config)
    seed_result = seed_knowledge_store(
        queries=seeding_queries,
        cloud_client=cloud_client,
        embedding_client=embedding_client,
        knowledge_store=ks,
    )
    logger.info("Seeding result: %s", seed_result)

    # Train
    models = train_routellm_baselines(
        training_queries=training_queries,
        local_client=local_client,
        cloud_client=cloud_client,
        judge_client=judge_client,
        embedding_client=embedding_client,
        knowledge_store=ks,
        conn=conn,
        training_seed=args.train_seed,
        output_dir=args.output_dir,
        force_retrain=args.force_retrain,
        local_model=args.local_model,
    )
    logger.info(
        "Done. routellm_no_memory and routellm_plus_ks pickled to %s. "
        "n_training_rows=%d, embedding_dim=%d.",
        args.output_dir, models.n_training_rows, models.embedding_dim,
    )
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(_cli_main())

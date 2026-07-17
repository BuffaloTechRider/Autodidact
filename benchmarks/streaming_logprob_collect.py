"""Collect per-token logprobs by replaying v0.1 MMLU-Pro queries.

Purpose: Paper A (streaming log-probability for early-exit routing) retrospective
analysis requires per-token logprobs, but v0.1 only stored the final averaged
`local_avg_logprob`. This script re-runs local generation (temperature=0.0,
deterministic) to capture the per-token list and persist it to a sidecar table
joined on query_id.

Cost: $0 (local-only, no cloud calls). Labels (`local_correct`) are reused from
the v0.1 `experiment_results` table — no judge calls required.

Run (proof-of-concept, 500 queries per model):
    python -m benchmarks.streaming_logprob_collect \\
        --run-id v0.1-qwen-20260427 --n 500

Scale to full v0.1 eval set by setting --n 1000 (or omitting).
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys
import time
from pathlib import Path

from autodidact.database import init_database
from autodidact.llm_client import LLMClient, LLMConfig, ChatMessage
from benchmarks.datasets import load_two_disjoint_subsets

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def ensure_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS streaming_logprobs (
            run_id TEXT NOT NULL,
            query_id TEXT NOT NULL,
            local_answer TEXT,
            avg_logprob REAL,
            token_logprobs_json TEXT,
            n_tokens INTEGER,
            collected_at TEXT DEFAULT (datetime('now')),
            PRIMARY KEY (run_id, query_id)
        )
        """
    )
    conn.commit()


def already_collected(conn: sqlite3.Connection, run_id: str) -> set[str]:
    cur = conn.execute(
        "SELECT query_id FROM streaming_logprobs WHERE run_id = ?", (run_id,)
    )
    return {r[0] for r in cur.fetchall()}


def load_v0_1_queries(conn: sqlite3.Connection, run_id: str, n: int) -> list[tuple[str, str]]:
    """Return [(query_id, query_text)] for the v0.1 eval set in order."""
    cur = conn.execute(
        """
        SELECT query_id, query_text
        FROM experiment_results
        WHERE run_id = ?
        ORDER BY query_index ASC
        LIMIT ?
        """,
        (run_id, n),
    )
    return cur.fetchall()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run-id", required=True, help="v0.1 run_id to replay (e.g., v0.1-qwen-20260427)")
    p.add_argument("--local-model", help="Override Ollama model (default: infer from run_id)")
    p.add_argument("--n", type=int, default=500, help="Number of queries to replay")
    p.add_argument("--db-path", default="autodidact_experiment.db")
    p.add_argument("--ollama-host", default="http://localhost:11434")
    args = p.parse_args()

    # Infer model from run_id if not overridden.
    model_map = {
        "qwen": "qwen2.5:7b",
        "llama": "llama3.1:8b",
        "mistral": "mistral:7b-instruct",
    }
    if args.local_model:
        local_model = args.local_model
    else:
        for key, name in model_map.items():
            if key in args.run_id:
                local_model = name
                break
        else:
            logger.error("Could not infer local model from run-id; pass --local-model")
            return 2

    logger.info("Run: %s  Model: %s  N: %d", args.run_id, local_model, args.n)

    conn = init_database(args.db_path)
    ensure_table(conn)

    done = already_collected(conn, args.run_id)
    logger.info("Already collected: %d queries", len(done))

    queries = load_v0_1_queries(conn, args.run_id, args.n)
    pending = [(qid, qt) for qid, qt in queries if qid not in done]
    logger.info("Pending: %d queries", len(pending))

    if not pending:
        logger.info("Nothing to do.")
        return 0

    # Need MMLU-Pro dataset to get formatted_prompt() — query_text in DB is the raw
    # question without the option choices. Reload with same params as the original
    # v0.1 runs (n_seed=1000 + n_eval=1000 combined, eval_seed=42) and index by query_id.
    logger.info("Loading MMLU-Pro for prompt formatting...")
    eval_split, _ = load_two_disjoint_subsets(
        n_eval=2000,  # 1000 seed + 1000 eval, as used by ablation_experiment
        n_train=1000,
        eval_seed=42,
        train_seed=43,
    )
    by_id = {q.query_id: q for q in eval_split}

    missing = [qid for qid, _ in pending if qid not in by_id]
    if missing:
        logger.warning("%d query_ids in DB not found in dataset — skipping", len(missing))

    client = LLMClient(
        LLMConfig(provider="ollama", model=local_model, ollama_host=args.ollama_host)
    )

    t_start = time.time()
    collected = 0
    for i, (qid, _) in enumerate(pending):
        q = by_id.get(qid)
        if q is None:
            continue
        try:
            resp = client.chat_with_logprobs(
                [ChatMessage(role="user", content=q.formatted_prompt())],
                max_tokens=512, temperature=0.0, top_logprobs=1,
            )
        except Exception as e:
            logger.warning("Query %s failed: %s", qid, e)
            continue

        conn.execute(
            """
            INSERT OR REPLACE INTO streaming_logprobs
                (run_id, query_id, local_answer, avg_logprob, token_logprobs_json, n_tokens)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                args.run_id,
                qid,
                resp.content,
                resp.avg_logprob,
                json.dumps(resp.logprobs),
                len(resp.logprobs),
            ),
        )
        collected += 1
        if collected % 25 == 0:
            conn.commit()
            elapsed = time.time() - t_start
            rate = collected / elapsed
            eta = (len(pending) - collected) / rate if rate > 0 else 0
            logger.info(
                "Progress: %d/%d  rate=%.1f/s  ETA=%.1f min",
                collected, len(pending), rate, eta / 60,
            )

    conn.commit()
    conn.close()

    elapsed = time.time() - t_start
    logger.info("Done. Collected %d queries in %.1f min", collected, elapsed / 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Knowledge store seeding for the v0.1 experiment.

Shared by the RouteLLM baseline trainer (Task 6) and the main experiment
harness (Task 7). Seeding is deterministic: given the same seed, the same
100 queries land in the store with the same content and embeddings.

Key invariant: the embedding stored on each entry is the embedding of the
ORIGINAL QUESTION, not the cloud answer. The answer lives in `content` and
`verbatim_response`. This makes retrieval a question-to-question similarity
search, which is what downstream signals actually want.
"""

from __future__ import annotations

import logging
from typing import Optional

from autodidact.knowledge_store import KnowledgeStore
from autodidact.llm_client import ChatMessage, LLMClient
from autodidact.types import NewKnowledgeEntry

from benchmarks.datasets import MMLUProQuery

logger = logging.getLogger(__name__)


def seed_knowledge_store(
    queries: list[MMLUProQuery],
    cloud_client: LLMClient,
    embedding_client: LLMClient,
    knowledge_store: KnowledgeStore,
    skip_if_populated: bool = True,
) -> dict[str, int]:
    """Seed the knowledge store from a list of MMLU-Pro queries.

    For each query, call the cloud model, store the answer as `content` and
    `verbatim_response`, embed the QUESTION for retrieval similarity, and
    also store the ANSWER embedding as analysis-only data for v0.2.

    Returns a dict with counts: {'inserted': N, 'skipped': M, 'failed': K}.

    If `skip_if_populated` is True and the store already has at least as many
    currently-valid entries as the seeding set, seeding is skipped entirely.
    If the store is partially populated (some queries already seeded, others
    not), we delegate to the resumable variant so we don't create duplicates
    on top of existing rows.
    """
    current_count = knowledge_store.count()
    if skip_if_populated and current_count >= len(queries):
        logger.info(
            "Knowledge store already has %d entries (>= %d needed); skipping seeding",
            current_count, len(queries),
        )
        return {"inserted": 0, "skipped": len(queries), "failed": 0}

    # Partially populated: delegate to the resumable path, which skips any
    # query_id already present in the store.
    if current_count > 0:
        logger.info(
            "Knowledge store has %d/%d entries; using resumable seeder to top up",
            current_count, len(queries),
        )
        return seed_knowledge_store_resumable(
            queries=queries,
            cloud_client=cloud_client,
            embedding_client=embedding_client,
            knowledge_store=knowledge_store,
        )

    inserted = 0
    failed = 0
    for q in queries:
        try:
            answer = cloud_client.chat(
                [ChatMessage(role="user", content=q.formatted_prompt())],
                max_tokens=512,
                temperature=0.0,
            )
            # Embed the ORIGINAL QUESTION for retrieval (question-to-question
            # similarity is what the v0.1 pipeline expects). Also embed the
            # ANSWER as analysis-only data so v0.2 Level-1 retrieval
            # experiments can reuse the KB without re-seeding.
            q_embedding = embedding_client.embed(q.query_text).tolist()
            a_embedding = embedding_client.embed(answer.content).tolist()
            knowledge_store.insert(
                NewKnowledgeEntry(
                    content=answer.content,
                    question=q.query_text,
                    source="cloud_escalation",
                    confidence=0.7,
                    tags=[q.category],
                    embedding=q_embedding,
                    answer_embedding=a_embedding,
                    domain=q.category,
                    topic="mmlu_pro",
                    metadata={
                        "query_id": q.query_id,
                        "ground_truth_letter": q.ground_truth_letter,
                    },
                    verbatim_response=answer.content,
                )
            )
            inserted += 1
        except Exception as e:
            logger.warning("Seeding failed for query %s: %s", q.query_id, e)
            failed += 1
    return {"inserted": inserted, "skipped": 0, "failed": failed}


# ── CLI entry point ────────────────────────────────────────────────

def seed_knowledge_store_resumable(
    queries: list,
    cloud_client,
    embedding_client,
    knowledge_store,
) -> dict:
    """Same as seed_knowledge_store but resumable per-query.

    Checks which query_ids are already in the store via metadata['query_id'],
    and only seeds the missing ones. Safe to re-run after a crash.
    """
    from autodidact.types import NewKnowledgeEntry
    from autodidact.llm_client import ChatMessage
    import json

    # Determine which query_ids are already seeded.
    existing: set[str] = set()
    rows = knowledge_store.conn.execute(
        "SELECT metadata FROM knowledge_entries WHERE valid_to IS NULL"
    ).fetchall()
    for r in rows:
        try:
            meta = json.loads(r["metadata"])
            qid = meta.get("query_id")
            if qid:
                existing.add(qid)
        except Exception:
            pass

    logger.info(
        "Resumable seed: %d/%d already present, %d to seed",
        len(existing), len(queries), len(queries) - len(existing),
    )

    inserted = 0
    skipped = 0
    failed = 0
    for i, q in enumerate(queries):
        if q.query_id in existing:
            skipped += 1
            continue
        try:
            answer = cloud_client.chat(
                [ChatMessage(role="user", content=q.formatted_prompt())],
                max_tokens=512,
                temperature=0.0,
            )
            q_embedding = embedding_client.embed(q.query_text).tolist()
            a_embedding = embedding_client.embed(answer.content).tolist()
            knowledge_store.insert(
                NewKnowledgeEntry(
                    content=answer.content,
                    question=q.query_text,
                    source="cloud_escalation",
                    confidence=0.7,
                    tags=[q.category],
                    embedding=q_embedding,
                    answer_embedding=a_embedding,
                    domain=q.category,
                    topic="mmlu_pro",
                    metadata={
                        "query_id": q.query_id,
                        "ground_truth_letter": q.ground_truth_letter,
                    },
                    verbatim_response=answer.content,
                )
            )
            inserted += 1
            if (i + 1) % 25 == 0:
                logger.info(
                    "Seeding progress: %d/%d (inserted %d, skipped %d, failed %d)",
                    i + 1, len(queries), inserted, skipped, failed,
                )
        except Exception as e:
            logger.warning("Seeding failed for %s: %s", q.query_id, e)
            failed += 1
    return {"inserted": inserted, "skipped": skipped, "failed": failed}


def _cli_main() -> int:
    """Standalone CLI: seed the knowledge store to a target size.

    Typical use:
        python -m benchmarks.seeding --n-seed 500
    """
    import argparse
    import sys
    from autodidact.database import init_database
    from autodidact.knowledge_store import KnowledgeStore
    from autodidact.llm_client import LLMClient, LLMConfig
    from autodidact.types import AutodidactConfig
    from benchmarks.datasets import load_mmlu_pro_subset

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="Seed the knowledge store from MMLU-Pro via cloud answers")
    p.add_argument("--n-seed", type=int, default=1000, help="Target number of KB entries")
    p.add_argument("--eval-seed", type=int, default=42,
                   help="Dataset seed — must match what the main experiment will use")
    p.add_argument("--local-model", default="qwen2.5:7b")
    p.add_argument("--cloud-model", default="us.anthropic.claude-sonnet-4-5-20250929-v1:0")
    p.add_argument("--embedding-model", default="qllama/bge-large-en-v1.5")
    p.add_argument("--bedrock-region", default="us-west-2")
    p.add_argument("--db-path", default="autodidact_experiment.db")
    args = p.parse_args()

    conn = init_database(args.db_path)
    ks = KnowledgeStore(conn, AutodidactConfig(db_path=args.db_path))

    current = ks.count()
    logger.info("Knowledge store currently has %d valid entries; target %d", current, args.n_seed)

    # Load the first n_seed queries from the canonical eval subset.
    # These match what the main experiment uses as its seeding split.
    queries = load_mmlu_pro_subset(args.n_seed, args.eval_seed)

    local_client = LLMClient(LLMConfig(
        provider="ollama", model=args.local_model, embedding_model=args.embedding_model,
    ))
    cloud_client = LLMClient(LLMConfig(
        provider="bedrock", model=args.cloud_model, region=args.bedrock_region,
    ))

    result = seed_knowledge_store_resumable(
        queries=queries,
        cloud_client=cloud_client,
        embedding_client=local_client,
        knowledge_store=ks,
    )
    logger.info("Seeding complete: %s", result)
    logger.info("Knowledge store now has %d valid entries", ks.count())
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(_cli_main())

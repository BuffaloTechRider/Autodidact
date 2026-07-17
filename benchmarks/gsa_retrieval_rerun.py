"""EXP-005: GSA with retrieval-conditional prompting, rerun over existing eval rows.

For each completed eval row in `experiment_results`, re-compute the GSA signal
under two new variants:

  - gsa-v3-070: if ≥1 retrieved hit has score ≥ 0.70, show retrieved content.
                Otherwise, use the bare (v2-confidence-identical) prompt — NO
                mention of "no knowledge retrieved" to avoid priming the model
                to say NO.
  - gsa-v3-060: same as above but threshold 0.60.

Both use the same underlying "confidence" framing as v2 (what EXP-002 found
was the least-bad single-prompt winner on qwen no-retrieval).

The script does NOT touch `experiment_results`. It writes p_yes values to a
sidecar JSONL file that `ablation_analysis.py` reads as new signal columns.

Per-query cost: ~$0 (local-only GSA probes). Per-query time: ~500ms (one-token
generation with logprobs) × 2 variants + one embedding = ~1.1s.

Run:
    python -u -m benchmarks.gsa_retrieval_rerun \\
        --run-id v0.1-qwen-20260427 \\
        --local-model qwen2.5:7b
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from autodidact.database import init_database
from autodidact.knowledge_store import KnowledgeStore
from autodidact.llm_client import ChatMessage, LLMClient, LLMConfig
from autodidact.types import AutodidactConfig

logger = logging.getLogger(__name__)

PROMPT_VERSION = "gsa-v3-retrieval-conditional"

# Identical to v2-confidence when there are no hits.
BARE_PROMPT_TEMPLATE = (
    "The user has asked the following question:\n"
    "{query}\n\n"
    "Are you confident you can answer this question correctly? "
    "Respond with exactly one token: YES or NO."
)

# When ≥1 hit clears threshold, we show the retrieved memory.
WITH_RETRIEVAL_PROMPT_TEMPLATE = (
    "The user has asked the following question:\n"
    "{query}\n\n"
    "Here is what you recall from your knowledge base:\n"
    "{hits_block}\n\n"
    "Are you confident you can answer this question correctly? "
    "Respond with exactly one token: YES or NO."
)

_YES_TOKENS = {"YES", "Yes", "yes", "Y", "y", " YES", " Yes", " yes", " Y", " y"}
_NO_TOKENS = {"NO", "No", "no", "N", "n", " NO", " No", " no", " N", " n"}


def _build_hits_block(hits: list) -> str:
    """Format top hits into a numbered block for prompt injection."""
    lines: list[str] = []
    for i, hit in enumerate(hits, start=1):
        content = (getattr(hit.entry, "content", "") or "")[:400].strip()
        q = getattr(hit.entry, "question", None)
        if q:
            lines.append(f"{i}. (memory of: {q.strip()[:120]})\n   {content}")
        else:
            lines.append(f"{i}. {content}")
    return "\n".join(lines)


def _extract_p_yes(response) -> tuple[float, str]:
    """Same three-tier extraction logic as SelfAssessment v2."""
    if response.top_logprobs_by_position:
        first_pos = response.top_logprobs_by_position[0]
        yes_lp = _first_matching_logprob(first_pos, _YES_TOKENS)
        no_lp = _first_matching_logprob(first_pos, _NO_TOKENS)
        if yes_lp is not None or no_lp is not None:
            missing_lp = -20.0
            yes_eff = yes_lp if yes_lp is not None else missing_lp
            no_eff = no_lp if no_lp is not None else missing_lp
            max_lp = max(yes_eff, no_eff)
            yes_exp = math.exp(yes_eff - max_lp)
            no_exp = math.exp(no_eff - max_lp)
            p_yes = yes_exp / (yes_exp + no_exp)
            return (max(0.0, min(1.0, p_yes)), "logprob_softmax")

    text = (response.content or "").strip().strip(".,;:!?)('\"`").upper()
    if text:
        first_word = text.split()[0] if text.split() else text
        if first_word in {"YES", "Y"}:
            return (1.0, "text_hard")
        if first_word in {"NO", "N"}:
            return (0.0, "text_hard")
    return (0.5, "neutral")


def _first_matching_logprob(logprobs_at_pos: dict, candidates: set) -> Optional[float]:
    for token, lp in logprobs_at_pos.items():
        if token in candidates:
            return float(lp)
    for token, lp in logprobs_at_pos.items():
        stripped = token.strip().upper()
        if stripped in {"YES", "Y"} and candidates is _YES_TOKENS:
            return float(lp)
        if stripped in {"NO", "N"} and candidates is _NO_TOKENS:
            return float(lp)
    return None


def _probe(
    local_client: LLMClient, prompt: str
) -> tuple[float, str]:
    """One-token YES/NO probe returning (p_yes, extraction_mode)."""
    response = local_client.chat_with_logprobs(
        [ChatMessage(role="user", content=prompt)],
        max_tokens=1, temperature=0.0, top_logprobs=5,
    )
    return _extract_p_yes(response)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run-id", required=True,
                        help="Which experiment run to rerun GSA for (reads eval rows from experiment_results)")
    parser.add_argument("--local-model", required=True,
                        help="The local model whose responses were measured in the run (must match the original run's model)")
    parser.add_argument("--embedding-model", default="qllama/bge-large-en-v1.5")
    parser.add_argument("--db-path", default="autodidact_experiment.db")
    parser.add_argument("--output-dir", default="results/experiment")
    parser.add_argument("--thresholds", default="0.70,0.60",
                        help="Comma-separated thresholds to test. Each produces a gsa-v3-<t> sidecar column.")
    args = parser.parse_args()

    thresholds = [float(s) for s in args.thresholds.split(",")]
    logger.info("Thresholds to sweep: %s", thresholds)

    conn = init_database(args.db_path)
    conn.row_factory = __import__("sqlite3").Row
    kb_config = AutodidactConfig(db_path=args.db_path)
    ks = KnowledgeStore(conn, kb_config)
    logger.info("KB has %d valid entries", ks.count())

    local_client = LLMClient(LLMConfig(
        provider="ollama", model=args.local_model,
        embedding_model=args.embedding_model,
    ))

    rows = conn.execute(
        "SELECT query_index, query_id, query_text, category, "
        "knowledge_similarity, grounded_self_assessment, local_correct "
        "FROM experiment_results "
        "WHERE run_id = ? AND error_info IS NULL "
        "ORDER BY query_index",
        (args.run_id,),
    ).fetchall()
    logger.info("Found %d clean eval rows for run %r", len(rows), args.run_id)
    if not rows:
        logger.error("No rows to process. Did the run complete?")
        return 2

    out_dir = Path(args.output_dir) / args.run_id / "gsa_retrieval_rerun"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    sidecar_path = out_dir / f"{stamp}.jsonl"
    logger.info("Writing sidecar to %s", sidecar_path)

    # Bookkeeping: how many queries actually saw retrieval at each threshold.
    with_retrieval_counts = {t: 0 for t in thresholds}

    with open(sidecar_path, "w") as f:
        for i, row in enumerate(rows):
            query_text = row["query_text"]

            # Re-embed the query. Cheap; ~100ms. We don't persist embeddings
            # so we have to recompute.
            try:
                q_emb = local_client.embed(query_text)
            except Exception as e:
                logger.warning("Embedding failed for %s: %s", row["query_id"], e)
                continue

            # One unfiltered search; we post-filter per threshold to avoid
            # multiple FAISS roundtrips.
            hits_all = ks.search(q_emb, limit=5, min_similarity=0.0)

            per_threshold: dict[str, dict] = {}
            for t in thresholds:
                hits_t = [h for h in hits_all if h.score >= t]
                had_retrieval = len(hits_t) > 0
                if had_retrieval:
                    with_retrieval_counts[t] += 1
                    prompt = WITH_RETRIEVAL_PROMPT_TEMPLATE.format(
                        query=query_text.strip(),
                        hits_block=_build_hits_block(hits_t),
                    )
                else:
                    # CRITICAL: identical to bare v2-confidence. No "(no relevant
                    # knowledge retrieved)" mention — that was found to prime
                    # the model toward NO.
                    prompt = BARE_PROMPT_TEMPLATE.format(query=query_text.strip())

                try:
                    p_yes, mode = _probe(local_client, prompt)
                except Exception as e:
                    logger.warning(
                        "Probe failed at threshold %.2f for %s: %s",
                        t, row["query_id"], e,
                    )
                    p_yes, mode = 0.5, "neutral"

                per_threshold[f"{t:.2f}"] = {
                    "p_yes": p_yes,
                    "mode": mode,
                    "had_retrieval": had_retrieval,
                    "n_hits": len(hits_t),
                    "top_hit_score": float(hits_t[0].score) if hits_t else None,
                }

            f.write(json.dumps({
                "query_index": row["query_index"],
                "query_id": row["query_id"],
                "category": row["category"],
                "local_correct": row["local_correct"],
                "gsa_v2_p_yes": row["grounded_self_assessment"],
                "knowledge_similarity": row["knowledge_similarity"],
                "prompt_version": PROMPT_VERSION,
                "per_threshold": per_threshold,
            }) + "\n")

            if (i + 1) % 50 == 0:
                logger.info(
                    "Progress: %d/%d  (retrieval counts: %s)",
                    i + 1, len(rows),
                    {f"{t:.2f}": with_retrieval_counts[t] for t in thresholds},
                )

    logger.info(
        "Done. %d rows processed. Final retrieval counts: %s",
        len(rows),
        {f"{t:.2f}": with_retrieval_counts[t] for t in thresholds},
    )
    logger.info("Sidecar: %s", sidecar_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())

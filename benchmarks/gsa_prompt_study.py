"""Mini prompt study for the Grounded Self-Assessment signal.

Tests multiple prompt variants on a small MMLU-Pro subset against ground-truth
`local_correct` labels. Produces a per-variant table with AUROC (signed — so we
can see when the signal is inverted), p_yes distribution, and calibration
quality. The winning prompt is the one with the highest POSITIVE AUROC.

Rationale for existing: initial dry-run (see LAB_NOTES P7) found the current
prompt made qwen2.5:7b say "NO" to 99% of queries even when it could answer
them correctly, producing AUROC 0.183 (informative but inverted). Rather than
auto-flipping post-hoc, we ask a different question.

Run:
    python -m benchmarks.gsa_prompt_study \
        --n-queries 30 --local-model qwen2.5:7b

Output: results/experiment/gsa_prompt_study/<timestamp>/
  - table.md — markdown table of per-variant metrics
  - rows.jsonl — per-(query, variant) raw data
  - summary.json — aggregated metrics per variant
  - chosen.txt — the winning variant name, consumable by later steps
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

from autodidact.knowledge_store import KnowledgeStore
from autodidact.database import init_database
from autodidact.llm_client import ChatMessage, LLMClient, LLMConfig
from autodidact.types import AutodidactConfig
from benchmarks.datasets import MMLUProQuery, load_mmlu_pro_subset
from benchmarks.labeling import label_answer

logger = logging.getLogger(__name__)

# ── Prompt variants ────────────────────────────────────────────────
# PROMPT_VERSION is stored on each persisted row so historical studies stay
# interpretable even as we add or rename variants.

PROMPT_VARIANTS: dict[str, str] = {
    "current": (
        "You are about to answer a user's question. Before answering, honestly assess whether "
        "you have enough information to answer CORRECTLY.\n\n"
        "Here is what you recall from your knowledge base (relevance score shown):\n"
        "{hits_block}\n\n"
        "The user's question is:\n{query}\n\n"
        "Do you have enough information to answer this question correctly? "
        "Respond with exactly one token: YES or NO."
    ),
    "direct": (
        "You will be asked a question. Here is what you recall from your knowledge base:\n"
        "{hits_block}\n\n"
        "The user's question is:\n{query}\n\n"
        "Can you answer this question correctly? Respond with exactly one token: YES or NO."
    ),
    "confidence": (
        "You will be asked a question. Here is what you recall from your knowledge base:\n"
        "{hits_block}\n\n"
        "The user's question is:\n{query}\n\n"
        "Are you confident you can answer this question correctly? "
        "Respond with exactly one token: YES or NO."
    ),
    "prediction": (
        "You will be asked a question. Here is what you recall from your knowledge base:\n"
        "{hits_block}\n\n"
        "The user's question is:\n{query}\n\n"
        "If you attempt this question, will your answer be correct? "
        "Respond with exactly one token: YES or NO."
    ),
}


@dataclass
class PromptResult:
    variant: str
    query_id: str
    p_yes: float
    extraction_mode: str
    local_answer: str
    local_correct: int


# ── Prompt helpers ─────────────────────────────────────────────────

def _build_prompt(template: str, query: str, hits: list) -> str:
    if not hits:
        hits_block = "(no relevant knowledge retrieved)"
    else:
        lines = []
        for i, hit in enumerate(hits, start=1):
            content = getattr(hit.entry, "content", "")[:400].strip()
            q = getattr(hit.entry, "question", None)
            if q:
                lines.append(f"{i}. (memory of: {q.strip()[:120]})\n   {content}")
            else:
                lines.append(f"{i}. {content}")
        hits_block = "\n".join(lines)
    return template.format(hits_block=hits_block, query=query.strip())


def _extract_p_yes(response) -> tuple[float, str]:
    """Same logic as GroundedSelfAssessment, inlined so prompt study has no dep on signal changes."""
    import math
    yes_tokens = {"YES", "Yes", "yes", "Y", "y", " YES", " Yes", " yes", " Y", " y"}
    no_tokens = {"NO", "No", "no", "N", "n", " NO", " No", " no", " N", " n"}

    def first_match(pos: dict, cands: set) -> Optional[float]:
        for t, lp in pos.items():
            if t in cands:
                return float(lp)
        for t, lp in pos.items():
            stripped = t.strip().upper()
            if stripped in {"YES", "Y"} and cands is yes_tokens:
                return float(lp)
            if stripped in {"NO", "N"} and cands is no_tokens:
                return float(lp)
        return None

    if response.top_logprobs_by_position:
        first_pos = response.top_logprobs_by_position[0]
        yes_lp = first_match(first_pos, yes_tokens)
        no_lp = first_match(first_pos, no_tokens)
        if yes_lp is not None or no_lp is not None:
            missing = -20.0
            yes_eff = yes_lp if yes_lp is not None else missing
            no_eff = no_lp if no_lp is not None else missing
            max_lp = max(yes_eff, no_eff)
            py = math.exp(yes_eff - max_lp) / (math.exp(yes_eff - max_lp) + math.exp(no_eff - max_lp))
            return (max(0.0, min(1.0, py)), "logprob_softmax")

    text = (response.content or "").strip().strip(".,;:!?)('\"`").upper()
    if text:
        first_word = text.split()[0] if text.split() else text
        if first_word in {"YES", "Y"}:
            return (1.0, "text_hard")
        if first_word in {"NO", "N"}:
            return (0.0, "text_hard")
    return (0.5, "neutral")


# ── AUROC ──────────────────────────────────────────────────────────

def signed_auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """AUROC that preserves its sign: returns values in [0, 1] where
    < 0.5 means the signal is inverted (high score → low label)."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int32)
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    all_scores = np.concatenate([pos, neg])
    order = np.argsort(all_scores, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(all_scores) + 1)
    # Tie correction
    i = 0
    sorted_scores = all_scores[order]
    while i < len(sorted_scores):
        j = i
        while j + 1 < len(sorted_scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        if j > i:
            avg = np.mean(ranks[order[i:j + 1]])
            ranks[order[i:j + 1]] = avg
        i = j + 1
    rank_sum_pos = ranks[: len(pos)].sum()
    u = rank_sum_pos - len(pos) * (len(pos) + 1) / 2
    return float(u / (len(pos) * len(neg)))


# ── Main study ─────────────────────────────────────────────────────

def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="GSA prompt study")
    parser.add_argument("--n-queries", type=int, default=30)
    parser.add_argument("--eval-seed", type=int, default=42)
    parser.add_argument("--skip-first", type=int, default=0,
                        help="Skip the first N queries from the stratified sample. Use this "
                             "when the first N are already in the KB (to avoid trivial retrieval). "
                             "E.g. --skip-first 500 --n-queries 30 after seeding 500.")
    parser.add_argument("--local-model", default="qwen2.5:7b")
    parser.add_argument("--cloud-model", default="us.anthropic.claude-sonnet-4-5-20250929-v1:0")
    parser.add_argument("--judge-model", default="us.anthropic.claude-opus-4-5-20251101-v1:0")
    parser.add_argument("--embedding-model", default="qllama/bge-large-en-v1.5")
    parser.add_argument("--bedrock-region", default="us-west-2")
    parser.add_argument("--db-path", default="autodidact_experiment.db")
    parser.add_argument("--output-dir", default="results/experiment/gsa_prompt_study")
    parser.add_argument("--use-kb", action="store_true",
                        help="Include retrieved knowledge from the existing knowledge_store. "
                             "Default: prompt shows '(no relevant knowledge retrieved)' for all "
                             "so we isolate the prompt-framing effect.")
    parser.add_argument("--kb-threshold", type=float, default=0.0,
                        help="When --use-kb is on, only include retrieved entries with "
                             "similarity >= this threshold. Entries below threshold are "
                             "dropped silently. Set to 0.7 to match the knowledge_similarity "
                             "signal's floor. Separate from --similarity-threshold.")
    parser.add_argument("--similarity-threshold", type=float, default=None,
                        help="Override KnowledgeStore.similarity_threshold for this run. "
                             "Default None = use config default (0.75). Set to 0.0 to "
                             "disable FAISS-level threshold filtering entirely.")
    args = parser.parse_args()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) / stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Output dir: %s", out_dir)

    # Clients
    local_client = LLMClient(LLMConfig(
        provider="ollama", model=args.local_model, embedding_model=args.embedding_model,
    ))
    cloud_client = LLMClient(LLMConfig(
        provider="bedrock", model=args.cloud_model, region=args.bedrock_region,
    ))
    judge_client = LLMClient(LLMConfig(
        provider="bedrock", model=args.judge_model, region=args.bedrock_region,
    ))

    # Load a small fixed subset
    # If --skip-first > 0, load N+skip queries and slice off the first N. Used
    # to get queries disjoint from the KB seed when KB was seeded with the
    # same eval_seed.
    total_needed = args.n_queries + args.skip_first
    all_queries = load_mmlu_pro_subset(total_needed, args.eval_seed)
    queries = all_queries[args.skip_first : args.skip_first + args.n_queries]
    logger.info("Loaded %d queries (skipped first %d)", len(queries), args.skip_first)

    # Optionally use the existing seeded knowledge store
    ks = None
    if args.use_kb:
        conn = init_database(args.db_path)
        config = AutodidactConfig(db_path=args.db_path)
        if args.similarity_threshold is not None:
            config.similarity_threshold = args.similarity_threshold
            logger.info("Using similarity_threshold override: %.3f", config.similarity_threshold)
        ks = KnowledgeStore(conn, config)
        logger.info("Using existing knowledge store with %d entries", ks.count())

    # For each query: generate one local answer, label it. Reuse that answer across
    # variants so the ground-truth `local_correct` is constant per query.
    rows_path = out_dir / "rows.jsonl"
    per_variant: dict[str, list[PromptResult]] = {v: [] for v in PROMPT_VARIANTS}
    local_correct_by_qid: dict[str, int] = {}
    local_answer_by_qid: dict[str, str] = {}

    with open(rows_path, "w") as f:
        for qi, q in enumerate(queries):
            # 1. Full local answer → label → reuse across variants.
            full_resp = local_client.chat(
                [ChatMessage(role="user", content=q.formatted_prompt())],
                max_tokens=512, temperature=0.0,
            )
            correct, _judge_used = label_answer(
                full_resp.content, q.ground_truth_letter, q.ground_truth_answer,
                q.query_text, judge_client=judge_client,
            )
            local_correct_by_qid[q.query_id] = int(correct)
            local_answer_by_qid[q.query_id] = full_resp.content

            # 2. Retrieved hits (only if --use-kb). Empty otherwise.
            # Thresholding: drop entries below --kb-threshold similarity.
            hits = []
            if ks is not None:
                q_emb = local_client.embed(q.query_text)
                all_hits = ks.search(q_emb, limit=5)
                if args.kb_threshold > 0:
                    hits = [h for h in all_hits if h.score >= args.kb_threshold]
                else:
                    hits = all_hits

            # 3. For each variant, run a 1-token GSA probe.
            for variant, template in PROMPT_VARIANTS.items():
                prompt = _build_prompt(template, q.query_text, hits)
                response = local_client.chat_with_logprobs(
                    [ChatMessage(role="user", content=prompt)],
                    max_tokens=1, temperature=0.0, top_logprobs=5,
                )
                p_yes, mode = _extract_p_yes(response)
                pr = PromptResult(
                    variant=variant, query_id=q.query_id,
                    p_yes=p_yes, extraction_mode=mode,
                    local_answer=full_resp.content, local_correct=int(correct),
                )
                per_variant[variant].append(pr)
                f.write(json.dumps({
                    "variant": variant,
                    "query_id": q.query_id,
                    "query_text": q.query_text,
                    "category": q.category,
                    "p_yes": p_yes,
                    "extraction_mode": mode,
                    "local_correct": int(correct),
                    "n_hits_shown": len(hits),
                    "top_hit_score": float(hits[0].score) if hits else None,
                    # Threshold-free retrieval quality signals. Populated whenever
                    # --use-kb is on; absent on no-KB runs.
                    "hit_categories": [getattr(h.entry, "domain", None) for h in hits],
                    "top1_in_category": (
                        bool(hits and getattr(hits[0].entry, "domain", None) == q.category)
                    ),
                    "any_top5_in_category": (
                        any(getattr(h.entry, "domain", None) == q.category for h in hits)
                    ),
                }) + "\n")
            if (qi + 1) % 5 == 0:
                logger.info("Done %d/%d queries", qi + 1, len(queries))

    # Aggregate
    summary = {}
    for variant, results in per_variant.items():
        scores = np.array([r.p_yes for r in results], dtype=np.float64)
        labels = np.array([r.local_correct for r in results], dtype=np.int32)
        auroc = signed_auroc(scores, labels)
        summary[variant] = {
            "n": len(results),
            "signed_auroc": auroc,
            "abs_auroc": float(abs(auroc - 0.5) + 0.5) if not np.isnan(auroc) else float("nan"),
            "inverted": (not np.isnan(auroc)) and auroc < 0.5,
            "avg_p_yes": float(np.mean(scores)),
            "p_yes_std": float(np.std(scores)),
            "yes_rate": float(np.mean(scores > 0.5)),
            "correct_rate": float(np.mean(labels)),
            "extraction_modes": {
                mode: int(sum(1 for r in results if r.extraction_mode == mode))
                for mode in ("logprob_softmax", "text_hard", "neutral")
            },
        }

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    # Pick the winner: highest signed_auroc (positive direction wins ties at same |auroc|).
    scored = sorted(
        [(v, s) for v, s in summary.items() if not np.isnan(s["signed_auroc"])],
        key=lambda kv: kv[1]["signed_auroc"],
        reverse=True,
    )
    winner = scored[0][0] if scored else None
    if winner:
        (out_dir / "chosen.txt").write_text(winner + "\n")

    # Markdown table
    # Retrieval summary — how often did at least one hit clear the threshold?
    retrieval_summary = ""
    if args.use_kb:
        # Re-scan rows.jsonl for n_hits_shown aggregates.
        n_hits_shown_list = []
        top_scores = []
        with open(rows_path) as rf:
            for line in rf:
                d = json.loads(line)
                # Count once per query (rows.jsonl has 1 row per (query, variant))
                # Skip dupes by filtering to first variant
                if d["variant"] != list(PROMPT_VARIANTS.keys())[0]:
                    continue
                n_hits_shown_list.append(d.get("n_hits_shown", 0))
                if d.get("top_hit_score") is not None:
                    top_scores.append(d["top_hit_score"])
        nonzero = sum(1 for n in n_hits_shown_list if n > 0)
        retrieval_summary = (
            f"**Retrieval quality at threshold {args.kb_threshold}:** "
            f"{nonzero}/{len(n_hits_shown_list)} queries had at least one hit above threshold. "
            + (f"Top-hit scores: min {min(top_scores):.3f}, "
               f"mean {np.mean(top_scores):.3f}, max {max(top_scores):.3f}. "
               if top_scores else "No queries had any hits above threshold. ")
        )

    lines = [
        "# GSA Prompt Study",
        f"**Timestamp:** {stamp}",
        f"**Local model:** {args.local_model}",
        f"**N queries:** {args.n_queries}",
        f"**Use KB:** {args.use_kb}",
        f"**KB threshold:** {args.kb_threshold}",
        f"**Eval seed:** {args.eval_seed}",
        f"**Skip first:** {args.skip_first}",
        "",
    ]
    if retrieval_summary:
        lines.extend([retrieval_summary, ""])
    lines.extend([
        "| Variant | AUROC (signed) | inverted? | avg p_yes | yes_rate | n | extraction |",
        "|---|---|---|---|---|---|---|",
    ])
    for variant, s in summary.items():
        em = s["extraction_modes"]
        em_str = "/".join(f"{k[:2]}={em[k]}" for k in ("logprob_softmax", "text_hard", "neutral"))
        lines.append(
            f"| `{variant}` | {s['signed_auroc']:.3f} | "
            f"{'yes' if s['inverted'] else 'no'} | "
            f"{s['avg_p_yes']:.3f} | {s['yes_rate']:.3f} | "
            f"{s['n']} | {em_str} |"
        )
    lines.append("")
    if winner:
        lines.append(f"## Winner: `{winner}`")
        lines.append(f"Signed AUROC {summary[winner]['signed_auroc']:.3f}, "
                     f"avg_p_yes={summary[winner]['avg_p_yes']:.3f}.")
        if summary[winner]["inverted"]:
            lines.append(
                "**NOTE:** all variants are inverted; the best one is least-inverted. "
                "Consider running with --use-kb, or exploring more prompts."
            )
    else:
        lines.append("## No winner — all variants produced NaN AUROC.")

    (out_dir / "table.md").write_text("\n".join(lines))

    print("\n".join(lines))
    print(f"\nArtifacts: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

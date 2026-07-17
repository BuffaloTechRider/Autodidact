"""Paper A viability gate: streaming-logprob AUROC over token position (TriviaQA).

Why TriviaQA and not the v0.1 MMLU-Pro data: streaming-logprob early-exit only
has anything to stream when local generation is multi-token. On MMLU-Pro the
answer is a single letter, so there is no token-position curve to measure. Free
-form TriviaQA answers give a real running-logprob signal AND free exact-match
labels (no judge, $0).

Two phases in one script:
  1. collect  — generate free-form answers with per-token logprobs, label by
                exact-match against answer aliases, write rows to JSONL.
  2. analyze  — for each token position K, compute the running-mean (and
                running-min) logprob over the first K tokens and its AUROC vs
                local_correct. Compare against the full-answer avg_logprob AUROC
                (the ceiling). Emit a curve + summary.

Run collect (per model, small n first to see the curve shape):
    python -u -m benchmarks.streaming_viability collect \\
        --n-queries 200 --local-model qwen2.5:7b

Then analyze (reads all rows.jsonl under the run dir):
    python -u -m benchmarks.streaming_viability analyze \\
        --rows results/experiment/streaming_viability/<stamp>/rows.jsonl

Viability decision (from paper/upcoming_papers_plan.md):
  - AUROC at token 10-15 within 0.05 of full avg_logprob AUROC -> Paper A viable.
  - Needs 50+ tokens to stabilize -> modest contribution, reconsider scope.
  - Never stabilizes before full generation -> streaming does not work; negative
    finding, move on.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from autodidact.llm_client import ChatMessage, LLMClient, LLMConfig
from benchmarks.ablation_analysis import auroc, bootstrap_auroc_ci
from benchmarks.triviaqa_check import check_answer, load_triviaqa_subset

logger = logging.getLogger(__name__)


# ── Datasets ───────────────────────────────────────────────────────
#
# Each dataset returns items with a uniform shape and supplies (prompt, label)
# helpers. TriviaQA = short free-form answers (tests signal-stabilization but
# not latency). GSM8K = long chain-of-thought (tests the latency claim: can we
# abort a 100-300 token generation early?).


def _load_gsm8k(n: int, seed: int) -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="test")
    idx = list(range(len(ds)))
    random.Random(seed).shuffle(idx)
    out = []
    for i in idx[:n]:
        item = ds[i]
        gold = item["answer"].split("####")[-1].strip().replace(",", "")
        out.append({
            "question_id": f"gsm8k:{i}",
            "question": item["question"],
            "gold_number": gold,
        })
    return out


_GSM8K_PROMPT = (
    "Solve this math problem. Think step by step, then give the final answer "
    "on a new line as 'The answer is <number>'.\n\nQuestion: {q}"
)

_NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def _gsm8k_correct(answer: str, gold: str) -> bool:
    """Exact-match on the last number in the response vs the gold number."""
    nums = _NUM_RE.findall(answer or "")
    if not nums:
        return False
    pred = nums[-1].replace(",", "").rstrip(".")
    try:
        return abs(float(pred) - float(gold)) < 1e-6
    except ValueError:
        return pred == gold


def load_items(dataset: str, n: int, seed: int) -> list[dict]:
    if dataset == "triviaqa":
        return load_triviaqa_subset(n, seed)
    if dataset == "gsm8k":
        return _load_gsm8k(n, seed)
    raise ValueError(f"unknown dataset: {dataset}")


def build_prompt(dataset: str, item: dict) -> str:
    if dataset == "triviaqa":
        return (f"Answer this trivia question in a few words.\n\n"
                f"Question: {item['question']}\nAnswer:")
    return _GSM8K_PROMPT.format(q=item["question"])


def label_correct(dataset: str, answer: str, item: dict) -> bool:
    if dataset == "triviaqa":
        return check_answer(answer, item["answer_aliases"])
    return _gsm8k_correct(answer, item["gold_number"])


# ── Phase 1: collect ───────────────────────────────────────────────


def collect(args: argparse.Namespace) -> int:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    # Include dataset + model in the dir name so parallel collects (which can
    # share a start-second) never clobber each other's rows.jsonl.
    model_tag = args.local_model.replace(":", "-").replace("/", "-")
    out_dir = Path(args.output_dir) / f"{stamp}_{args.dataset}_{model_tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading %d %s items (seed=%d)...", args.n_queries, args.dataset, args.seed)
    questions = load_items(args.dataset, args.n_queries, args.seed)
    logger.info("Loaded %d items", len(questions))

    client = LLMClient(LLMConfig(provider="ollama", model=args.local_model))

    rows_path = out_dir / "rows.jsonl"
    n_written = 0
    with open(rows_path, "w") as f:
        for i, q in enumerate(questions):
            try:
                resp = client.chat_with_logprobs(
                    [ChatMessage(role="user", content=build_prompt(args.dataset, q))],
                    max_tokens=args.max_tokens, temperature=0.0,
                    top_logprobs=args.top_logprobs,
                )
            except Exception as e:  # noqa: BLE001 — one bad query shouldn't kill the run
                logger.warning("Query %d failed: %s", i, e)
                continue

            answer = (resp.content or "").strip()
            correct = label_correct(args.dataset, answer, q)
            # entropy needs the top-k distribution per position; store it only
            # when we actually collected k>1 (keeps files small otherwise).
            top_by_pos = resp.top_logprobs_by_position if args.top_logprobs > 1 else []
            f.write(json.dumps({
                "question_id": q["question_id"],
                "dataset": args.dataset,
                "local_model": args.local_model,
                "model_answer": answer,
                "correct": int(correct),
                "avg_logprob": resp.avg_logprob,
                "token_logprobs": resp.logprobs,
                "tokens": resp.tokens,
                "top_logprobs_by_position": top_by_pos,
                "n_tokens": len(resp.logprobs),
                "had_thinking": bool(getattr(resp, "had_thinking", False)),
            }) + "\n")
            n_written += 1
            if n_written % 25 == 0:
                f.flush()
                logger.info("Progress: %d/%d", n_written, len(questions))

    logger.info("Collected %d rows -> %s", n_written, rows_path)
    print(rows_path)  # so a caller can pipe straight into analyze
    return 0


# ── Phase 2: analyze ───────────────────────────────────────────────

# Function words + punctuation carry near-p=1 logprobs regardless of whether the
# final answer is correct, so averaging them dilutes the signal. `is_content`
# keeps only substantive tokens for the content-aware aggregations.
_STOPWORDS = frozenset("""
a an the this that these those and or but if then else so because as of to in on
at for with from by is are was were be been being am do does did have has had
i we you he she it they me my our your his her its their there here what which who
whom will would can could should may might must shall not no nor only just also
step let s t re ve ll m d = + - * / < > ( ) [ ] { } . , : ; ! ? # $ % ^ & | \ ~ `
""".split())


def _is_content(tok: str) -> bool:
    """A token counts as 'content' if, stripped and lowercased, it isn't a
    stopword/punctuation and contains at least one alphanumeric char."""
    t = tok.strip().lower()
    if not t or t in _STOPWORDS:
        return False
    return any(c.isalnum() for c in t)


def _entropy(top: dict) -> float:
    """Predictive entropy (nats) from a {token: logprob} top-k dict. Higher
    entropy = less confident, so we negate it later to keep 'higher=confident'."""
    if not top:
        return float("nan")
    lps = np.array(list(top.values()), dtype=np.float64)
    p = np.exp(lps)
    s = p.sum()
    if s <= 0:
        return float("nan")
    p = p / s
    return float(-(p * np.log(p + 1e-12)).sum())


def _pad_flat(vals: list[float], max_k: int) -> np.ndarray:
    """Running score at each K=1..max_k. Once tokens run out, hold the last
    value flat — models a short generation committing at its length."""
    out = np.empty(max_k, dtype=np.float64)
    if not vals:
        out[:] = np.nan
        return out
    arr = np.asarray(vals, dtype=np.float64)
    n = len(arr)
    for k in range(1, max_k + 1):
        out[k - 1] = arr[min(k, n) - 1]
    return out


def _running_scores(row: dict, max_k: int, bottom_k: int = 5) -> dict[str, np.ndarray]:
    """All running-aggregation curves for one row, each padded to max_k.

    Returns a dict keyed by aggregation name. Higher score always = more
    confident/correct.
      mean          — running mean of all token logprobs (the naive baseline)
      min           — running min (single least-confident token)
      content_mean  — running mean over CONTENT tokens only (the user's fix)
      bottom{k}_mean— mean of the k least-confident tokens so far (robust min)
      neg_entropy   — running mean of -entropy over content tokens (needs top-k)
    """
    lp = np.asarray(row.get("token_logprobs") or [], dtype=np.float64)
    toks = row.get("tokens") or []
    n = len(lp)
    curves: dict[str, list[float]] = {"mean": [], "min": [], "content_mean": [],
                                      f"bottom{bottom_k}_mean": [], "neg_entropy": []}
    if n == 0:
        return {k: _pad_flat([], max_k) for k in curves}

    tops = row.get("top_logprobs_by_position") or []
    have_tops = len(tops) == n
    # content mask (fall back to "all tokens are content" if token strings absent)
    if len(toks) == n:
        content = np.array([_is_content(t) for t in toks], dtype=bool)
    else:
        content = np.ones(n, dtype=bool)

    csum = np.cumsum(lp)
    running_min = np.inf
    c_sum = 0.0
    c_cnt = 0
    ent_sum = 0.0
    ent_cnt = 0
    for j in range(n):
        running_min = min(running_min, lp[j])
        curves["mean"].append(csum[j] / (j + 1))
        curves["min"].append(running_min)
        if content[j]:
            c_sum += lp[j]
            c_cnt += 1
            if have_tops:
                e = _entropy(tops[j])
                if not np.isnan(e):
                    ent_sum += e
                    ent_cnt += 1
        # content_mean: mean over content tokens seen so far; before any content
        # token appears, fall back to the plain running mean so K is defined.
        curves["content_mean"].append(c_sum / c_cnt if c_cnt else csum[j] / (j + 1))
        curves["neg_entropy"].append(-(ent_sum / ent_cnt) if ent_cnt else float("nan"))
        # bottom-k mean: mean of the k lowest logprobs among the first j+1
        kk = min(bottom_k, j + 1)
        curves[f"bottom{bottom_k}_mean"].append(float(np.sort(lp[: j + 1])[:kk].mean()))

    return {name: _pad_flat(vals, max_k) for name, vals in curves.items()}


def analyze(args: argparse.Namespace) -> int:
    rows_paths = [Path(p) for p in args.rows]
    rows: list[dict] = []
    for rp in rows_paths:
        with open(rp) as f:
            rows.extend(json.loads(line) for line in f if line.strip())
    # Only rows with at least one token and a usable label.
    rows = [r for r in rows if r.get("n_tokens", 0) > 0]
    if not rows:
        logger.error("No usable rows found.")
        return 2

    out_dir = rows_paths[0].parent
    labels = np.array([r["correct"] for r in rows], dtype=np.int32)
    n_pos, n_neg = int(labels.sum()), int((labels == 0).sum())
    n_tokens = np.array([r["n_tokens"] for r in rows], dtype=np.int32)

    # AUROC convention: higher score = more likely CORRECT. avg_logprob is
    # already monotonic in confidence (closer to 0 = more confident), so we use
    # it directly as the score.
    full_scores = np.array([r["avg_logprob"] for r in rows], dtype=np.float64)
    full_auroc, full_lo, full_hi = bootstrap_auroc_ci(full_scores, labels)

    max_k = int(args.max_k)
    # Stack every aggregation's running-score matrix: {method: (n_rows, max_k)}.
    per_row = [_running_scores(r, max_k) for r in rows]
    methods = list(per_row[0].keys())
    mats = {m: np.vstack([pr[m] for pr in per_row]) for m in methods}
    # Drop methods that are all-NaN (e.g. neg_entropy when top-k wasn't collected).
    methods = [m for m in methods if not np.isnan(mats[m]).all()]

    curve = []
    for k in range(1, max_k + 1):
        entry = {"k": k, "frac_live": float((n_tokens >= k).mean())}
        for m in methods:
            entry[f"auroc_{m}"] = float(auroc(mats[m][:, k - 1], labels))
        curve.append(entry)

    # First K within 0.05 of the full ceiling, per method.
    k_within = {}
    for m in methods:
        hit = [c["k"] for c in curve
               if not np.isnan(c[f"auroc_{m}"]) and c[f"auroc_{m}"] >= full_auroc - 0.05]
        k_within[m] = hit[0] if hit else None

    dataset_name = rows[0].get("dataset", "unknown")
    summary = {
        "dataset": dataset_name,
        "local_model": rows[0].get("local_model", "unknown"),
        "n": len(rows),
        "n_correct": n_pos,
        "n_incorrect": n_neg,
        "local_accuracy": float(labels.mean()),
        "n_tokens_median": float(np.median(n_tokens)),
        "n_tokens_mean": float(n_tokens.mean()),
        "n_tokens_p90": float(np.percentile(n_tokens, 90)),
        "full_avg_logprob_auroc": float(full_auroc),
        "full_avg_logprob_ci": [float(full_lo), float(full_hi)],
        "methods": methods,
        "k_within_0.05_of_full": k_within,
        "curve": curve,
    }
    (out_dir / "viability_summary.json").write_text(json.dumps(summary, indent=2))
    _plot(curve, methods, full_auroc, summary, out_dir / "viability_curve.png")

    # Console report — one column per aggregation method.
    show_k = [k for k in (1, 2, 3, 5, 8, 10, 15, 20, 30, 50, 100, 200) if k <= max_k]
    if max_k not in show_k:
        show_k.append(max_k)
    print(f"\n{'='*90}")
    print(" Paper A viability — streaming-logprob AUROC over token position (content-aware)")
    print(f"{'='*90}")
    print(f"  Model: {summary['local_model']}   Dataset: {dataset_name}")
    print(f"  N (usable): {summary['n']}  (correct={n_pos}, incorrect={n_neg})   "
          f"accuracy={summary['local_accuracy']:.3f}")
    print(f"  Answer length: median={summary['n_tokens_median']:.0f} "
          f"mean={summary['n_tokens_mean']:.1f} p90={summary['n_tokens_p90']:.0f} tokens")
    print(f"  Full avg_logprob AUROC (ceiling): {full_auroc:.3f} "
          f"[{full_lo:.3f}, {full_hi:.3f}]")
    header = f"  {'K':>4}  {'%live':>5}  " + "  ".join(f"{m[:13]:>13}" for m in methods)
    print("\n" + header)
    for c in curve:
        if c["k"] in show_k:
            cells = "  ".join(f"{c[f'auroc_{m}']:>13.3f}" for m in methods)
            print(f"  {c['k']:>4}  {c['frac_live']*100:>4.0f}%  {cells}")
    print("\n  First K within 0.05 of full ceiling, by method:")
    for m in methods:
        print(f"    {m:>16}: {k_within[m]}")
    # Verdict keys on the BEST early (K<=15) method vs ceiling.
    early = curve[:15]
    per_method_early = []
    for m in methods:
        vals = [c[f"auroc_{m}"] for c in early if not np.isnan(c[f"auroc_{m}"])]
        if vals:
            per_method_early.append(max(vals))
    best_early = max(per_method_early) if per_method_early else float("nan")
    if best_early >= full_auroc - 0.05:
        verdict = "VIABLE — some aggregation reaches the ceiling by K=15."
    elif best_early >= 0.65:
        verdict = f"PARTIAL — best early AUROC {best_early:.3f}; useful but below ceiling."
    else:
        verdict = f"NEGATIVE — best early AUROC {best_early:.3f}; no early signal, any aggregation."
    print(f"\n  Best early (K<=15) AUROC across methods: {best_early:.3f}")
    print(f"  Verdict: {verdict}")
    print(f"{'='*90}")
    print(f"Artifacts: {out_dir}")
    return 0


_METHOD_STYLE = {
    "mean": ("running-mean (all tokens)", "-", 2.0),
    "content_mean": ("running-mean (content tokens only)", "-", 2.0),
    "min": ("running-min", "--", 1.2),
    "bottom5_mean": ("bottom-5 mean", "--", 1.2),
    "neg_entropy": ("neg predictive entropy (content)", "-.", 1.5),
}


def _plot(curve, methods, full_auroc, summary, path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib unavailable — skipping plot")
        return
    ks = [c["k"] for c in curve]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for m in methods:
        label, ls, lw = _METHOD_STYLE.get(m, (m, "-", 1.2))
        ax.plot(ks, [c[f"auroc_{m}"] for c in curve], label=label, ls=ls, lw=lw)
    ax.axhline(full_auroc, color="k", ls=":", label=f"full avg_logprob ({full_auroc:.3f})")
    ax.axhline(0.5, color="gray", ls="-", lw=0.8, alpha=0.5, label="chance")
    ax.set_xlabel("token position K")
    ax.set_ylabel("AUROC vs local_correct")
    ax.set_title(f"Streaming-logprob viability — {summary['local_model']} "
                 f"({summary['dataset']}, n={summary['n']})")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    logger.info("Wrote %s", path)


# ── entry ──────────────────────────────────────────────────────────


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    pc = sub.add_parser("collect", help="generate answers + per-token logprobs")
    pc.add_argument("--dataset", choices=["triviaqa", "gsm8k"], default="triviaqa")
    pc.add_argument("--n-queries", type=int, default=200)
    pc.add_argument("--seed", type=int, default=42)
    pc.add_argument("--local-model", default="qwen2.5:7b")
    pc.add_argument("--max-tokens", type=int, default=50)
    pc.add_argument("--top-logprobs", type=int, default=1,
                    help="k for top_logprobs; >1 enables predictive-entropy aggregation")
    pc.add_argument("--output-dir", default="results/experiment/streaming_viability")
    pc.set_defaults(func=collect)

    pa = sub.add_parser("analyze", help="AUROC over token position")
    pa.add_argument("--rows", nargs="+", required=True, help="one or more rows.jsonl")
    pa.add_argument("--max-k", type=int, default=50)
    pa.set_defaults(func=analyze)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

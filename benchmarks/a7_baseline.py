"""A7 baseline — the NFR-2 ROI gate, run local-only ($0).

A7 asks whether the apprentice-agent bet is real:
  1. What is the cloud/local/memory *split* on real multi-step tasks?
  2. Is step difficulty *separable* — do tasks split into cheap (local-able)
     and hard (cloud) steps — or *entangled*, so per-step routing degenerates
     to whole-task routing and the win evaporates?

Cloud tool-calling isn't available here (the Bedrock backend raises on
``tools=``; no OpenAI key), so we can't put a dollar figure on cloud *share*.
But question 2 — the one that actually validates the routing bet — is
measurable local-only: run every task on Ollama through the tiered executor
and record each step's **intended tier**, derived from the local model's own
``avg_logprob`` against the real router thresholds:

    avg_logprob > high (-0.3)  → Tier 1 LOCAL   (confident, run local)
    avg_logprob < low  (-1.0)  → Tier 3 CLOUD   (would escalate)
    otherwise                  → Tier 2 VERIFY  (uncertain, self-consistency)

We then cross-tabulate intended tier against each task's a-priori difficulty
label. If easy tasks concentrate at Tier 1 and hard tasks shift toward
Tier 2/3, difficulty is separable and step-level routing pays off. A flat
distribution across labels means difficulty is entangled.

Memory-tier share is 0 by construction here: with no cloud escalation there is
nothing to learn, so the task-entry memory probe never hits. That's a property
of the local-only run, not a finding about the design.

Usage:
    python -m benchmarks.a7_baseline --pilot        # ~10 tasks
    python -m benchmarks.a7_baseline                # full 50
"""

from __future__ import annotations

import argparse
import logging
import os
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from autodidact.database import init_database
from autodidact.executor import Executor
from autodidact.llm_client import LLMClient, LLMConfig
from autodidact.routing.step_router import FixedThresholdRouter, _DEFAULT_THRESHOLD, Tier
from autodidact.tools import REGISTRY
from autodidact.trajectory_store import TrajectoryStore

from benchmarks.multistep_tasks import MultiStepTask, load_tasks, pilot_tasks, verifier_for

logger = logging.getLogger("a7")


@dataclass
class TaskOutcome:
    task_id: str
    difficulty: str
    steps_taken: int
    stop_reason: str
    completed: bool  # finished with a text answer (not budget-exhausted / empty)
    correct: Optional[bool]  # verifier result; None = no verifier (unchecked)
    # Intended tier counts for this task's steps.
    tier_counts: Counter = field(default_factory=Counter)
    # Max-intended tier over the task's steps (the tier the *task* would route to).
    peak_tier: str = "LOCAL"


def _intended_tier(avg_logprob: float | None) -> Tier:
    """The tier the router *would* pick from this step's confidence.

    Uses the real default thresholds so the local-only run reflects production
    routing intent. None (no logprob) maps to VERIFY, matching Threshold.tier_for.
    """
    return _DEFAULT_THRESHOLD.tier_for(avg_logprob)


def _run_one(task: MultiStepTask, local: LLMClient, conn) -> TaskOutcome:
    """Run a single task in an isolated sandbox dir and read back its trajectory."""
    store = TrajectoryStore(conn)
    # Local-only: no cloud. Uncertain/low-confidence steps degrade to local
    # execution, but we recover the *intended* tier from the recorded logprob.
    executor = Executor(
        local=local, cloud=None, tools=REGISTRY,
        router=FixedThresholdRouter(), store=store, max_iterations=12,
    )

    verify = verifier_for(task.task_id)
    correct: Optional[bool] = None
    with tempfile.TemporaryDirectory(prefix=f"a7_{task.task_id}_") as sandbox:
        for name, content in task.setup_files.items():
            (Path(sandbox) / name).write_text(content)
        prev = os.getcwd()
        os.chdir(sandbox)  # file tools confine to cwd
        try:
            result = executor.execute(task.prompt)
        finally:
            os.chdir(prev)
        # Verify against the actual sandbox state *before* the tempdir is torn down.
        if verify is not None:
            try:
                correct = bool(verify(Path(sandbox), result.answer or ""))
            except Exception as e:
                logger.warning("verifier %s raised: %s", task.task_id, e)
                correct = False

    # The executor started exactly one trajectory this call; recover its steps.
    # TrajectoryStore has no "latest" helper, so we read via the header table.
    steps = _latest_steps(conn)
    tier_counts: Counter = Counter()
    peak = Tier.LOCAL
    order = {Tier.LOCAL: 0, Tier.VERIFY: 1, Tier.CLOUD: 2}
    for s in steps:
        t = _intended_tier(s.avg_logprob)
        tier_counts[t.value] += 1
        if order[t] > order[peak]:
            peak = t

    completed = result.stop_reason == "done" and bool(result.answer)
    return TaskOutcome(
        task_id=task.task_id, difficulty=task.difficulty,
        steps_taken=result.steps_taken, stop_reason=result.stop_reason,
        completed=completed, correct=correct, tier_counts=tier_counts,
        peak_tier=peak.value,
    )


def _latest_steps(conn):
    """Read the most-recently-created trajectory's steps."""
    row = conn.execute(
        "SELECT id FROM execution_trajectories ORDER BY created_at DESC, rowid DESC LIMIT 1"
    ).fetchone()
    if row is None:
        return []
    return TrajectoryStore(conn).load_steps(row["id"])


def _report(outcomes: list[TaskOutcome]) -> str:
    lines: list[str] = []
    n = len(outcomes)
    total_steps = sum(o.steps_taken for o in outcomes)
    completed = sum(1 for o in outcomes if o.completed)

    # Aggregate intended-tier distribution.
    all_tiers: Counter = Counter()
    for o in outcomes:
        all_tiers.update(o.tier_counts)
    tier_total = sum(all_tiers.values()) or 1

    lines.append("# A7 Baseline (local-only) — NFR-2 ROI gate\n")
    lines.append(f"Tasks: {n}   Completed locally: {completed}/{n} "
                 f"({completed / n * 100:.0f}%)   Total steps: {total_steps}\n")
    lines.append("## Intended-tier distribution (from local avg_logprob vs thresholds)\n")
    lines.append("Cloud share here = the % of steps the router *would* escalate.\n")
    for tier in ("LOCAL", "VERIFY", "CLOUD"):
        c = all_tiers.get(tier, 0)
        lines.append(f"  {tier:6s}: {c:4d}  ({c / tier_total * 100:5.1f}%)")
    lines.append("  (memory: 0.0% by construction — no cloud escalation, nothing learned)\n")

    # The key A7 question: is difficulty separable? Cross-tab tier by difficulty.
    lines.append("## Separability: intended-tier mix by a-priori difficulty\n")
    by_diff: dict[str, Counter] = defaultdict(Counter)
    steps_by_diff: dict[str, int] = defaultdict(int)
    done_by_diff: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for o in outcomes:
        by_diff[o.difficulty].update(o.tier_counts)
        steps_by_diff[o.difficulty] += o.steps_taken
        done_by_diff[o.difficulty][0] += int(o.completed)
        done_by_diff[o.difficulty][1] += 1
    for diff in ("easy", "medium", "hard"):
        c = by_diff[diff]
        tot = sum(c.values()) or 1
        cloud_pct = c.get("CLOUD", 0) / tot * 100
        local_pct = c.get("LOCAL", 0) / tot * 100
        done, dtot = done_by_diff[diff]
        lines.append(
            f"  {diff:6s}: LOCAL {local_pct:5.1f}%  VERIFY {c.get('VERIFY', 0) / tot * 100:5.1f}%  "
            f"CLOUD {cloud_pct:5.1f}%   | completed {done}/{dtot}"
        )
    lines.append("")

    # The decisive table: is high local confidence actually *correct*? A high
    # LOCAL% is only good news if those local steps produce right answers.
    checked = [o for o in outcomes if o.correct is not None]
    lines.append("## Correctness (verified tasks only)")
    if not checked:
        lines.append("  (no verifiers ran)\n")
    else:
        n_correct = sum(1 for o in checked if o.correct)
        lines.append(f"  Verified: {len(checked)}/{n}   Correct: {n_correct}/{len(checked)} "
                     f"({n_correct / len(checked) * 100:.0f}%)\n")
        lines.append("  By a-priori difficulty:")
        for diff in ("easy", "medium", "hard"):
            grp = [o for o in checked if o.difficulty == diff]
            if grp:
                c = sum(1 for o in grp if o.correct)
                lines.append(f"    {diff:6s}: {c}/{len(grp)} correct")
        lines.append("")
        lines.append("  Correctness by task's peak intended tier:")
        for tier in ("LOCAL", "VERIFY", "CLOUD"):
            grp = [o for o in checked if o.peak_tier == tier]
            if grp:
                c = sum(1 for o in grp if o.correct)
                lines.append(f"    {tier:6s}: {c}/{len(grp)} correct")
        # The money line: confident-but-wrong tasks. These are the ones routing
        # SHOULD have escalated but (on confidence alone) wouldn't have.
        conf_wrong = [o for o in checked if o.peak_tier == "LOCAL" and not o.correct]
        lines.append("")
        lines.append(f"  Confident-but-WRONG (peak tier LOCAL, incorrect): "
                     f"{len(conf_wrong)}  {[o.task_id for o in conf_wrong]}")
        lines.append("  → these are the routing gap: high self-confidence, wrong answer,")
        lines.append("    no escalation triggered. If this set is large, logprob-only")
        lines.append("    routing is miscalibrated and GSA/verification earns its keep.\n")

    lines.append("## Read")
    lines.append("If LOCAL% falls and CLOUD% rises from easy→hard, difficulty is")
    lines.append("separable and per-step routing pays off. A flat mix means it's")
    lines.append("entangled. Separately, a large confident-but-wrong set means")
    lines.append("logprob alone is miscalibrated — the case for GSA + verification.\n")

    # Per-task detail.
    lines.append("## Per-task")
    for o in sorted(outcomes, key=lambda x: (x.difficulty, x.task_id)):
        tiers = " ".join(f"{k}:{v}" for k, v in sorted(o.tier_counts.items()))
        corr = "?" if o.correct is None else ("✓" if o.correct else "✗")
        lines.append(
            f"  {o.task_id} [{o.difficulty:6s}] {corr} steps={o.steps_taken:2d} "
            f"{'ok ' if o.completed else 'INC'} ({o.stop_reason})  {tiers}"
        )
    return "\n".join(lines)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="A7 baseline: multi-step local routing intent")
    p.add_argument("--pilot", action="store_true", help="Run ~10 stratified tasks instead of all 50")
    p.add_argument("--n-pilot", type=int, default=10)
    p.add_argument("--local-model", default="qwen2.5:7b")
    p.add_argument("--embedding-model", default="qllama/bge-large-en-v1.5")
    p.add_argument("--out", default="results/a7_baseline.md")
    args = p.parse_args()

    tasks = pilot_tasks(args.n_pilot) if args.pilot else load_tasks()
    logger.info("A7 %s run: %d tasks", "PILOT" if args.pilot else "FULL", len(tasks))

    local = LLMClient(LLMConfig(
        provider="ollama", model=args.local_model, embedding_model=args.embedding_model,
    ))
    conn = init_database(":memory:")

    outcomes: list[TaskOutcome] = []
    for i, task in enumerate(tasks, 1):
        logger.info("[%d/%d] %s (%s)", i, len(tasks), task.task_id, task.difficulty)
        try:
            outcomes.append(_run_one(task, local, conn))
        except Exception as e:
            logger.warning("task %s failed: %s", task.task_id, e)
            outcomes.append(TaskOutcome(
                task_id=task.task_id, difficulty=task.difficulty, steps_taken=0,
                stop_reason=f"error:{type(e).__name__}", completed=False,
            ))

    report = _report(outcomes)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report)
    print("\n" + report)
    logger.info("wrote %s", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

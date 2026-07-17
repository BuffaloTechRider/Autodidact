# Autodidact Experiment Guide

Everything you need to run, repeat, and vary the v0.1 ablation experiment. Written so future-you can come back in a month and know what to do without re-reading the whole spec.

## TL;DR — one-time setup, then three commands

```bash
# One-time setup (see "Prerequisites" below)
ollama pull qwen2.5:7b && ollama pull qllama/bge-large-en-v1.5
pip install -e '.[dev,bedrock]'

# Each experiment run:
./scripts/preflight.sh        # verify everything, get cost estimate (no spend)
./scripts/run_experiment.sh    # do the actual run
./scripts/monitor.sh           # optional: watch progress in another terminal
```

Read results at `results/experiment/MEMO.md` when the run completes.

## Prerequisites

1. **Ollama** running locally with the local model and embedding model pulled. Default: `qwen2.5:7b` + `qllama/bge-large-en-v1.5`.
2. **AWS credentials** (via `aws configure` or `AWS_PROFILE`) with Bedrock access in a region where the target Claude models are enabled.
3. **Python deps** installed: `pip install -e '.[dev,bedrock]'`. This installs numpy, scipy, sklearn, pydantic, requests, matplotlib, faiss, datasets, boto3.
4. **Disk**: ~5GB free (for Ollama models + HuggingFace MMLU-Pro download + SQLite DB).

Preflight script checks all four automatically:

```bash
./scripts/preflight.sh
```

If everything passes, it also prints the estimated cloud cost for the current configuration. Nothing is spent until you run `run_experiment.sh`.

## The three scripts

### `preflight.sh`

Read-only. Verifies Ollama reachable + models pulled, AWS identity valid, Bedrock models reachable, Python deps importable. Prints cost estimate. Safe to run anytime.

### `run_experiment.sh`

Runs the full pipeline end-to-end:

1. **Train RouteLLM baselines** (1-2h, ~$25 at premium). Also seeds the knowledge store with 100 MMLU-Pro queries.
2. **Run the ablation experiment** (3-4h, ~$27 at premium). For each of 1000 evaluation queries, computes all 6 confidence signals, scores both RouteLLM baselines, generates local + cloud answers, labels correctness.
3. **Analyze** (seconds, $0). Computes AUROC per signal and per 7 combos, paired bootstrap deltas, calibration plots, ROC overlay, and writes `MEMO.md`.

Flags:

- `--dry-run` — small-scale validation run (~$2, ~30 min): n_seed=20, n_eval=50, n_training=100. Uses run_id `dry-<timestamp>` to keep it isolated.
- `--skip-baselines` — skip step 1 if pickle files already exist.
- `--skip-main` — skip step 2 if experiment_results already populated. Use to only re-run analysis/memo with different formatting.

### `monitor.sh`

Live-refreshing view of in-flight experiment progress. Run in a second terminal:

```bash
./scripts/monitor.sh <run_id>    # or no arg to auto-detect latest
```

Shows row count, cost so far, local/cloud accuracy so far, GSA extraction-mode distribution (sanity check), and recent errors.

## Varying the experiment

All variation happens via environment variables to `run_experiment.sh`. No code changes required. The `RUN_ID` is what keeps different experiments separate in the DB.

### Vary the local model (size or family)

```bash
# Same-family size sweep
RUN_ID=qwen-3b  LOCAL_MODEL=qwen2.5:3b  ./scripts/run_experiment.sh
RUN_ID=qwen-7b  LOCAL_MODEL=qwen2.5:7b  ./scripts/run_experiment.sh
RUN_ID=qwen-14b LOCAL_MODEL=qwen2.5:14b ./scripts/run_experiment.sh

# Different families at similar size
RUN_ID=llama-8b LOCAL_MODEL=llama3.1:8b ./scripts/run_experiment.sh
RUN_ID=mistral-7b LOCAL_MODEL=mistral:7b ./scripts/run_experiment.sh
RUN_ID=phi-14b LOCAL_MODEL=phi3:14b ./scripts/run_experiment.sh
```

Before each, `ollama pull <model>`. The RouteLLM baselines depend on the local model (because `local_is_sufficient` is defined relative to it), so each new model triggers fresh baseline training. If you want to skip baseline training and reuse an existing one, you can't — but a comparison across local models with the SAME baseline is meaningless anyway, so this is correct behavior.

Each local-model run is fully isolated by `RUN_ID`. They coexist in the same DB. You can cross-compare memos later.

### Vary the cloud model

```bash
# Cheaper cloud (saves ~$15)
CLOUD_MODEL=anthropic.claude-3-haiku-20240307-v1:0  ./scripts/run_experiment.sh

# Mid-tier
CLOUD_MODEL=us.anthropic.claude-sonnet-4-20250514-v1:0  ./scripts/run_experiment.sh

# Premium (default)
CLOUD_MODEL=us.anthropic.claude-sonnet-4-5-20250929-v1:0  ./scripts/run_experiment.sh
```

Stronger cloud → better knowledge content → higher retrieval signal quality. Matters less than you'd think on MMLU-Pro (all Claude models get ~70-85% correct), but matters more on harder benchmarks.

### Vary the judge

```bash
# Cheap judge
JUDGE_MODEL=anthropic.claude-3-sonnet-20240229-v1:0 ./scripts/run_experiment.sh

# Premium judge (default)
JUDGE_MODEL=us.anthropic.claude-opus-4-5-20251101-v1:0 ./scripts/run_experiment.sh
```

Judge is only called when letter extraction fails (~10-30% of queries). Premium judge adds ~$10 at n=1000.

### Vary dataset size

```bash
# Smaller, cheaper, noisier
RUN_ID=quick N_EVAL=200 N_TRAINING=200 ./scripts/run_experiment.sh

# Default
N_EVAL=1000 N_TRAINING=1000 ./scripts/run_experiment.sh

# Bigger, more expensive, sharper statistical conclusions
RUN_ID=big N_EVAL=2000 N_TRAINING=2000 ./scripts/run_experiment.sh
```

Bootstrap CI widths scale roughly as 1/√n. Doubling n tightens CIs by ~30%.

### Vary the seed (reproducibility check)

```bash
RUN_ID=seed-1 EVAL_SEED=1 TRAIN_SEED=2 ./scripts/run_experiment.sh
RUN_ID=seed-2 EVAL_SEED=11 TRAIN_SEED=12 ./scripts/run_experiment.sh
```

Multiple seeds establish whether results are stable across which questions got sampled. Two matching seeds with different question selections producing similar AUROCs is evidence the signals are robust.

### Vary knowledge-store seed size (the "memory matters" curve)

```bash
for n in 50 100 250 500; do
  RUN_ID=seedsize-$n N_SEED=$n ./scripts/run_experiment.sh
done
```

Rebuilds the knowledge store fresh for each. Expected curve: knowledge_similarity AUROC grows with seed size up to a plateau. This is the direct evidence of the memory-growth story.

Note: baseline training ALSO reseeds with the same `n`, so `routellm_plus_ks` sees the same memory as our combos. Fair comparison.

### Dry-run before committing

```bash
./scripts/run_experiment.sh --dry-run
```

Uses n_seed=20, n_eval=50, n_training=100, ~$2, ~30 minutes. Validates that Ollama + Bedrock + the full pipeline all work end-to-end without burning a full budget. Recommended before any bigger run.

## Comparing runs

All experiment_results rows live in `autodidact_experiment.db`, keyed by `run_id`. To compare two runs:

```bash
# Summary of two runs side by side
python -c "
from benchmarks.ablation_analysis import compute_all, load_rows
from autodidact.database import init_database
conn = init_database('autodidact_experiment.db')
for rid in ['qwen-7b', 'llama-8b']:
    try:
        r = load_rows(conn, rid)
        result = compute_all(r, conn)
        hl = result['headline']
        print(f'{rid}: best={hl[\"best_combo\"]} auroc={hl[\"best_auroc\"]:.3f} vs_routellm_plus_ks={hl[\"gap_vs_routellm_plus_ks\"]:+.3f}')
    except Exception as e:
        print(f'{rid}: {e}')
"
```

Or re-run analysis on a specific run:

```bash
python -m benchmarks.ablation_analysis --run-id qwen-7b --output-dir results/qwen-7b
```

(use a per-run `--output-dir` if you want per-run plots and memos)

## Resuming a failed run

The experiment harness is resumable. If something dies mid-run (network blip, Ollama crash, you accidentally Ctrl+C), rerun with the same `RUN_ID`:

```bash
RUN_ID=v0.1-20260424 ./scripts/run_experiment.sh --skip-baselines
```

Queries that already have rows in `experiment_results` are skipped. Only the missing indices will be re-executed. No wasted cost.

Same for baseline training: the `routellm_training_rows` table caches labeled pairs per `training_seed`, so rerunning only fills in missing rows.

## Troubleshooting

### Ollama not reachable
```bash
ollama serve &       # start it
ollama list          # confirm models are pulled
```

### Bedrock model access denied
- Check the Bedrock console → Model access to ensure the model is enabled in your region.
- Newer models (Claude 3.5+, Claude 4+) require the `us.` inference-profile ID, not the plain model ID. E.g. `us.anthropic.claude-sonnet-4-5-20250929-v1:0`, not `anthropic.claude-sonnet-4-5-20250929-v1:0`.
- Run `aws bedrock list-inference-profiles --region us-west-2` to see available profiles.

### Bedrock rate-limit / throttle
Harness retries transient errors with exponential backoff. If sustained throttling, slow down: we don't have a rate-limit flag yet; if needed, edit `_with_retries` or add a `time.sleep()` per-call.

### Cost exceeded threshold
Preflight prints the estimate before spending. If the estimate is wrong (our per-query token assumption is conservative), bump `COST_THRESHOLD=200 ./scripts/run_experiment.sh`.

### Memo says "n/a" for most AUROCs
Usually means `load_rows` dropped most rows because they had NaN in some signal. Check:
```bash
sqlite3 autodidact_experiment.db "SELECT COUNT(*), COUNT(CASE WHEN error_info IS NULL THEN 1 END) FROM experiment_results WHERE run_id = '<your_run_id>'"
```
If error row count is high, inspect `error_info` per row.

### Local model never says YES or NO
Check GSA extraction mode:
```bash
sqlite3 autodidact_experiment.db "SELECT gsa_extraction_mode, COUNT(*) FROM experiment_results WHERE run_id = '<your_run_id>' GROUP BY gsa_extraction_mode"
```
If mostly `neutral`, the model isn't producing YES/NO at all (bad prompt following). Try a different local model, or adjust the prompt in `autodidact/signals/grounded_self_assessment.py`.

### Retrieval recall is ~0
At small seed sizes, most queries have no in-category match. Bump `N_SEED=500` or higher.

## Recommended follow-up experiments

See `.kiro/specs/autodidact-framework/tasks.md` section "v0.1.1+ follow-ups". Highlights:

- **Seed-size sweep** at 50/100/250/500/1000. Makes the memory-growth curve explicit.
- **Cross-model sweep** across 3B/7B/14B Qwen + Llama + Mistral. Shows whether signals generalize.
- **Cross-dataset replication** on TriviaQA or BBH. External validity for a paper claim.
- **Bigger n** (2000+) if v0.1 paired deltas are marginal.

## What to share

When an experiment finishes, the interesting artifacts live in `results/experiment/<run_id>/`:

- `MEMO.md` — the honest one-pager with all conclusions.
- `summary.json` — every AUROC number, CI, paired delta. Raw for reuse.
- `roc_overlay.png` — ROC curves for all 7 combos + 2 baselines on one axes.
- `calibration_best_*.png` — reliability diagram for the winning combo.
- `calibration_weak_*.png` — reliability diagrams for the two weakest signals (diagnostic).

Every run lives in its own subdirectory, so running new experiments never overwrites old memos. The `results/experiment/latest` symlink always points at the most recent analysis, so `cat results/experiment/latest/MEMO.md` does the right thing for quick lookups.

When you want to share or compare, point at the specific run_id directory rather than `latest` — that gives you a stable reference that won't move out from under you.

Commit the per-run directory to git alongside `LAB_NOTES.md` updates. That's your experiment record: numbers + code state + the story of what you learned.

## Versioning conventions

- **`run_id`** is the primary handle. Every row in `experiment_results` is keyed by it, every artifact directory is named by it, every memo cites it.
- **Default `run_id`** is `v0.1-<YYYYMMDD>` for the `run_experiment.sh` default, `dry-<YYYYMMDD>_<HHMM>` for `--dry-run`. Override with `RUN_ID=foo ./scripts/run_experiment.sh`.
- **Never reuse a run_id** across different configurations. Thanks to the UNIQUE constraint on `(run_id, query_index)`, resuming a run with the same id is safe (skips completed rows); starting a *different* experiment with the same id will mix data.
- **When you change a prompt, a model, a seed size** — anything that meaningfully changes what's being measured — pick a new run_id. Easier to compare later.
- **When you re-run analysis only** (no new experiment data), artifacts update in place within `<run_id>/`. The underlying DB rows are unchanged.

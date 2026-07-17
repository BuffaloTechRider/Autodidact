#!/usr/bin/env bash
# Full Autodidact v0.1 experiment run.
# 1. Train RouteLLM baselines (~1-2h, ~$25 at premium)
# 2. Run main ablation experiment (~3-4h, ~$27 at premium)
# 3. Analyze, produce summary.json, plots, and MEMO.md (seconds, $0)
#
# Usage:
#   ./scripts/run_experiment.sh
#   ./scripts/run_experiment.sh --dry-run       # small-scale validation run
#   ./scripts/run_experiment.sh --skip-baselines  # reuse pre-trained baselines
#   ./scripts/run_experiment.sh --skip-main       # only rerun analysis/memo
#
# Environment variables (defaults in scripts/EXPERIMENT_GUIDE.md):
#   RUN_ID           default: v0.1-<date>
#   LOCAL_MODEL      default: qwen2.5:7b
#   EMBEDDING_MODEL  default: qllama/bge-large-en-v1.5
#   CLOUD_MODEL      default: us.anthropic.claude-sonnet-4-5-20250929-v1:0
#   JUDGE_MODEL      default: us.anthropic.claude-opus-4-5-20251101-v1:0
#   BEDROCK_REGION   default: us-west-2
#   N_SEED           default: 100
#   N_EVAL           default: 1000
#   N_TRAINING       default: 1000
#   EVAL_SEED        default: 42
#   TRAIN_SEED       default: 43
#   DB_PATH          default: autodidact_experiment.db
#   OUTPUT_DIR       default: results/experiment
#   LOG_DIR          default: logs
#   COST_THRESHOLD   default: 100 (USD; aborts if estimate exceeds)
#
# What to check after:
#   cat results/experiment/MEMO.md
#   cat results/experiment/summary.json | python -m json.tool | less

set -euo pipefail

# ── Defaults ───────────────────────────────────────────────────────

# Record which env vars the user explicitly set so --dry-run doesn't clobber them.
USER_SET_N_SEED="${N_SEED+yes}"
USER_SET_N_EVAL="${N_EVAL+yes}"
USER_SET_N_TRAINING="${N_TRAINING+yes}"

RUN_ID="${RUN_ID:-v0.1-$(date +%Y%m%d)}"
LOCAL_MODEL="${LOCAL_MODEL:-qwen2.5:7b}"
EMBEDDING_MODEL="${EMBEDDING_MODEL:-qllama/bge-large-en-v1.5}"
CLOUD_MODEL="${CLOUD_MODEL:-us.anthropic.claude-sonnet-4-5-20250929-v1:0}"
JUDGE_MODEL="${JUDGE_MODEL:-us.anthropic.claude-opus-4-5-20251101-v1:0}"
BEDROCK_REGION="${BEDROCK_REGION:-us-west-2}"
N_SEED="${N_SEED:-1000}"
N_EVAL="${N_EVAL:-1000}"
N_TRAINING="${N_TRAINING:-1000}"
EVAL_SEED="${EVAL_SEED:-42}"
TRAIN_SEED="${TRAIN_SEED:-43}"
DB_PATH="${DB_PATH:-autodidact_experiment.db}"
OUTPUT_DIR="${OUTPUT_DIR:-results/experiment}"
LOG_DIR="${LOG_DIR:-logs}"
COST_THRESHOLD="${COST_THRESHOLD:-100}"

# ── Flags ──────────────────────────────────────────────────────────

DRY_RUN=0
SKIP_BASELINES=0
SKIP_MAIN=0
for arg in "$@"; do
  case "$arg" in
    --dry-run)
      DRY_RUN=1
      # --dry-run sets small defaults but RESPECTS env vars the user set explicitly.
      [ -z "$USER_SET_N_SEED" ]     && N_SEED=20
      [ -z "$USER_SET_N_EVAL" ]     && N_EVAL=50
      [ -z "$USER_SET_N_TRAINING" ] && N_TRAINING=100
      RUN_ID="dry-$(date +%Y%m%d_%H%M)"
      echo "DRY RUN: n_seed=$N_SEED, n_eval=$N_EVAL, n_training=$N_TRAINING"
      ;;
    --skip-baselines) SKIP_BASELINES=1 ;;
    --skip-main)      SKIP_MAIN=1 ;;
    -h|--help)
      sed -n '2,/^set -euo pipefail/p' "$0" | grep -E '^# ' | sed 's/^# //'
      exit 0
      ;;
    *) echo "Unknown flag: $arg"; exit 2 ;;
  esac
done

# ── Setup ──────────────────────────────────────────────────────────

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"
STAMP=$(date +%Y%m%d_%H%M%S)

echo "═══════════════════════════════════════════════════════════════"
echo "Autodidact v0.1 experiment run"
echo "═══════════════════════════════════════════════════════════════"
echo "  RUN_ID:          $RUN_ID"
echo "  LOCAL_MODEL:     $LOCAL_MODEL"
echo "  CLOUD_MODEL:     $CLOUD_MODEL"
echo "  JUDGE_MODEL:     $JUDGE_MODEL"
echo "  BEDROCK_REGION:  $BEDROCK_REGION"
echo "  N_SEED:          $N_SEED"
echo "  N_EVAL:          $N_EVAL"
echo "  N_TRAINING:      $N_TRAINING"
echo "  DB_PATH:         $DB_PATH"
echo "  OUTPUT_DIR:      $OUTPUT_DIR"
echo "  LOG_DIR:         $LOG_DIR"
echo "  skip_baselines:  $SKIP_BASELINES"
echo "  skip_main:       $SKIP_MAIN"
echo "  dry_run:         $DRY_RUN"
echo "═══════════════════════════════════════════════════════════════"
echo

# macOS: keep system awake during the run if caffeinate is present.
PREFIX=""
if command -v caffeinate >/dev/null 2>&1; then
  PREFIX="caffeinate -s"
  echo "Prefixing long-running commands with 'caffeinate -s' to prevent sleep."
fi

# ── Step 1: RouteLLM baselines ────────────────────────────────────

if [ "$SKIP_BASELINES" -eq 0 ]; then
  echo "── Step 1/3: Train RouteLLM baselines ──"
  BASE_LOG="$LOG_DIR/baseline_${STAMP}.log"
  echo "  Logging to: $BASE_LOG"
  set +e
  $PREFIX python -m benchmarks.routellm_baseline \
    --db-path "$DB_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --n-seed "$N_SEED" \
    --n-eval "$N_EVAL" \
    --n-training "$N_TRAINING" \
    --eval-seed "$EVAL_SEED" \
    --train-seed "$TRAIN_SEED" \
    --local-model "$LOCAL_MODEL" \
    --cloud-model "$CLOUD_MODEL" \
    --judge-model "$JUDGE_MODEL" \
    --embedding-model "$EMBEDDING_MODEL" \
    --bedrock-region "$BEDROCK_REGION" \
    2>&1 | tee "$BASE_LOG"
  STATUS=${PIPESTATUS[0]}
  set -e
  if [ "$STATUS" -ne 0 ]; then
    echo "✗ Baseline training failed (exit $STATUS). See $BASE_LOG."
    exit "$STATUS"
  fi
  if [ ! -f "$OUTPUT_DIR/routellm_no_memory.pkl" ] || [ ! -f "$OUTPUT_DIR/routellm_plus_ks.pkl" ]; then
    echo "✗ Baseline pickle files missing after training."
    exit 1
  fi
  echo "✓ Baselines trained."
  echo
else
  echo "── Step 1/3: SKIPPED (--skip-baselines) ──"
  if [ ! -f "$OUTPUT_DIR/routellm_no_memory.pkl" ] || [ ! -f "$OUTPUT_DIR/routellm_plus_ks.pkl" ]; then
    echo "✗ Cannot skip baselines — pickle files do not exist in $OUTPUT_DIR."
    echo "  Run once without --skip-baselines first."
    exit 1
  fi
  echo
fi

# ── Step 2: Main experiment ────────────────────────────────────────

if [ "$SKIP_MAIN" -eq 0 ]; then
  echo "── Step 2/3: Run ablation experiment ──"
  EXP_LOG="$LOG_DIR/experiment_${STAMP}.log"
  echo "  Logging to: $EXP_LOG"
  set +e
  $PREFIX python -m benchmarks.ablation_experiment \
    --run-id "$RUN_ID" \
    --db-path "$DB_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --n-seed "$N_SEED" \
    --n-eval "$N_EVAL" \
    --eval-seed "$EVAL_SEED" \
    --train-seed "$TRAIN_SEED" \
    --local-model "$LOCAL_MODEL" \
    --cloud-model "$CLOUD_MODEL" \
    --judge-model "$JUDGE_MODEL" \
    --embedding-model "$EMBEDDING_MODEL" \
    --bedrock-region "$BEDROCK_REGION" \
    --cost-threshold-usd "$COST_THRESHOLD" \
    --confirm-cost \
    2>&1 | tee "$EXP_LOG"
  STATUS=${PIPESTATUS[0]}
  set -e
  if [ "$STATUS" -ne 0 ]; then
    echo "✗ Main experiment failed (exit $STATUS). See $EXP_LOG."
    exit "$STATUS"
  fi
  echo "✓ Main experiment complete."
  echo
else
  echo "── Step 2/3: SKIPPED (--skip-main) ──"
  echo
fi

# ── Step 3: Analysis + memo ────────────────────────────────────────

echo "── Step 3/3: Analyze and generate memo ──"
ANA_LOG="$LOG_DIR/analysis_${STAMP}.log"
RUN_OUTPUT_DIR="$OUTPUT_DIR/$RUN_ID"
set +e
python -m benchmarks.ablation_analysis \
  --run-id "$RUN_ID" \
  --db-path "$DB_PATH" \
  --output-dir "$OUTPUT_DIR" \
  2>&1 | tee "$ANA_LOG"
STATUS=${PIPESTATUS[0]}
set -e
if [ "$STATUS" -ne 0 ]; then
  echo "✗ Analysis failed (exit $STATUS). See $ANA_LOG."
  exit "$STATUS"
fi

# Fill memo with full metadata. Write into the per-run subdirectory so we keep history.
python -m benchmarks.memo \
  --output-dir "$RUN_OUTPUT_DIR" \
  --local-model "$LOCAL_MODEL" \
  --cloud-model "$CLOUD_MODEL" \
  --embedding-model "$EMBEDDING_MODEL" \
  --eval-seed "$EVAL_SEED" \
  --train-seed "$TRAIN_SEED" \
  --n-training-rows "$N_TRAINING" \
  2>&1 | tee -a "$ANA_LOG"

echo
echo "═══════════════════════════════════════════════════════════════"
echo "✓ DONE."
echo "  Run ID:       $RUN_ID"
echo "  Artifacts:    $RUN_OUTPUT_DIR/"
echo "  Memo:         $RUN_OUTPUT_DIR/MEMO.md"
echo "  Summary:      $RUN_OUTPUT_DIR/summary.json"
echo "  Plots:        $RUN_OUTPUT_DIR/*.png"
echo "  Latest alias: $OUTPUT_DIR/latest -> $RUN_ID"
echo "═══════════════════════════════════════════════════════════════"
echo
echo "Read the memo:"
echo "  cat $OUTPUT_DIR/latest/MEMO.md"
echo "or directly:"
echo "  cat $RUN_OUTPUT_DIR/MEMO.md"

#!/usr/bin/env bash
# Preflight check before running an Autodidact experiment.
# Verifies Ollama, AWS, Python deps, and prints an estimated cost.
# Does NOT spend any cloud money.
#
# Usage:
#   ./scripts/preflight.sh
#
# Environment variables (all optional — see scripts/EXPERIMENT_GUIDE.md):
#   LOCAL_MODEL      default: qwen2.5:7b
#   EMBEDDING_MODEL  default: qllama/bge-large-en-v1.5
#   CLOUD_MODEL      default: us.anthropic.claude-sonnet-4-5-20250929-v1:0
#   JUDGE_MODEL      default: us.anthropic.claude-opus-4-5-20251101-v1:0
#   BEDROCK_REGION   default: us-west-2
#   N_SEED           default: 100
#   N_EVAL           default: 1000
#   N_TRAINING       default: 1000
#   OLLAMA_HOST      default: http://localhost:11434

set -uo pipefail

LOCAL_MODEL="${LOCAL_MODEL:-qwen2.5:7b}"
EMBEDDING_MODEL="${EMBEDDING_MODEL:-qllama/bge-large-en-v1.5}"
CLOUD_MODEL="${CLOUD_MODEL:-us.anthropic.claude-sonnet-4-5-20250929-v1:0}"
JUDGE_MODEL="${JUDGE_MODEL:-us.anthropic.claude-opus-4-5-20251101-v1:0}"
BEDROCK_REGION="${BEDROCK_REGION:-us-west-2}"
N_SEED="${N_SEED:-1000}"
N_EVAL="${N_EVAL:-1000}"
N_TRAINING="${N_TRAINING:-1000}"
OLLAMA_HOST="${OLLAMA_HOST:-http://localhost:11434}"

PASS=0
FAIL=0
WARN=0

ok()   { echo "  ✓ $1"; PASS=$((PASS+1)); }
bad()  { echo "  ✗ $1"; FAIL=$((FAIL+1)); }
warn() { echo "  ⚠ $1"; WARN=$((WARN+1)); }
hdr()  { echo; echo "── $1 ──"; }

hdr "Configuration"
cat <<EOF
  LOCAL_MODEL      = $LOCAL_MODEL
  EMBEDDING_MODEL  = $EMBEDDING_MODEL
  CLOUD_MODEL      = $CLOUD_MODEL
  JUDGE_MODEL      = $JUDGE_MODEL
  BEDROCK_REGION   = $BEDROCK_REGION
  N_SEED           = $N_SEED
  N_EVAL           = $N_EVAL
  N_TRAINING       = $N_TRAINING
  OLLAMA_HOST      = $OLLAMA_HOST
EOF

hdr "1. Ollama"
if ! command -v curl >/dev/null 2>&1; then
  bad "curl not installed; cannot check Ollama"
else
  if curl -s --max-time 5 "$OLLAMA_HOST/api/tags" >/dev/null 2>&1; then
    ok "Ollama reachable at $OLLAMA_HOST"
    tags_json=$(curl -s --max-time 5 "$OLLAMA_HOST/api/tags")
    # Match either an exact quoted tag (e.g. "qwen2.5:7b") OR a bare name followed
    # by a colon (e.g. "nomic-embed-text:latest" when the user didn't specify a tag).
    # The :latest tag is what Ollama assigns when you do `ollama pull <name>` without a tag.
    check_pulled() {
      local model="$1"
      # If user specified a tag (name:tag), match the exact quoted form.
      # Otherwise match name followed by ":" (any tag).
      if [[ "$model" == *:* ]]; then
        echo "$tags_json" | grep -q "\"${model}\""
      else
        echo "$tags_json" | grep -qE "\"${model}:[^\"]+\""
      fi
    }
    if check_pulled "$LOCAL_MODEL"; then
      ok "Local model pulled: $LOCAL_MODEL"
    else
      bad "Local model NOT pulled: $LOCAL_MODEL — run: ollama pull $LOCAL_MODEL"
    fi
    if check_pulled "$EMBEDDING_MODEL"; then
      ok "Embedding model pulled: $EMBEDDING_MODEL"
    else
      bad "Embedding model NOT pulled: $EMBEDDING_MODEL — run: ollama pull $EMBEDDING_MODEL"
    fi
  else
    bad "Ollama NOT reachable at $OLLAMA_HOST — start with: ollama serve"
  fi
fi

hdr "2. AWS credentials"
if ! command -v aws >/dev/null 2>&1; then
  bad "aws CLI not installed"
else
  if identity=$(aws sts get-caller-identity 2>&1); then
    who=$(echo "$identity" | python -c "import json,sys; d=json.load(sys.stdin); print(d.get('Arn','?'))" 2>/dev/null || echo "?")
    ok "AWS identity: $who"
  else
    bad "AWS credentials not configured — run: aws configure"
  fi
fi

hdr "3. Bedrock access"
if command -v aws >/dev/null 2>&1; then
  for model in "$CLOUD_MODEL" "$JUDGE_MODEL"; do
    result=$(aws bedrock-runtime converse \
      --region "$BEDROCK_REGION" \
      --model-id "$model" \
      --messages '[{"role":"user","content":[{"text":"ok"}]}]' \
      --inference-config maxTokens=4 \
      2>&1 | head -10)
    if echo "$result" | grep -q '"output"'; then
      ok "Bedrock model reachable: $model"
    elif echo "$result" | grep -q "AccessDeniedException"; then
      bad "Bedrock access DENIED for $model — enable in console"
    elif echo "$result" | grep -q "ValidationException"; then
      bad "Bedrock validation error for $model — check if model needs an inference-profile ID (us.* prefix)"
    elif echo "$result" | grep -q "ResourceNotFoundException"; then
      bad "Bedrock model not found: $model — check region $BEDROCK_REGION and model ID"
    else
      warn "Bedrock check inconclusive for $model; snippet: $(echo $result | head -c 120)"
    fi
  done
fi

hdr "4. Python deps"
python -c "
import sys
missing = []
for mod in ('numpy','scipy','sklearn','pydantic','requests','matplotlib','faiss','datasets','boto3'):
    try:
        __import__(mod)
    except Exception as e:
        missing.append((mod, str(e)))
if missing:
    for m, e in missing:
        print(f'  ✗ missing: {m} — {e}')
    sys.exit(1)
print('  ✓ all required deps importable')
" || FAIL=$((FAIL+1))

python -c "
try:
    import autodidact
    from autodidact.llm_client import LLMClient
    from autodidact.signals.grounded_self_assessment import GroundedSelfAssessment
    import benchmarks.ablation_experiment
    import benchmarks.ablation_analysis
    import benchmarks.memo
    import benchmarks.routellm_baseline
    print('  ✓ autodidact + benchmarks modules import')
except Exception as e:
    print(f'  ✗ import error: {e}')
    raise
" || FAIL=$((FAIL+1))

hdr "5. Cost estimate"
python -c "
from benchmarks.ablation_experiment import _estimate_cost, HarnessConfig
cfg = HarnessConfig(
    run_id='preflight', db_path=':memory:', output_dir='/tmp',
    n_seed=$N_SEED, n_eval=$N_EVAL, eval_seed=42, train_seed=43,
    local_model='$LOCAL_MODEL',
    cloud_model='$CLOUD_MODEL',
    judge_model='$JUDGE_MODEL',
    embedding_model='$EMBEDDING_MODEL',
    ollama_host=None, bedrock_region='$BEDROCK_REGION',
    confirm_cost=False, cost_threshold_usd=5.0,
)
c_main = _estimate_cost(cfg, n_eval=$N_EVAL, n_seed=$N_SEED)
c_baseline = _estimate_cost(cfg, n_eval=$N_TRAINING, n_seed=0)
print(f'  Baseline training: USD {c_baseline:.2f}')
print(f'  Main experiment:   USD {c_main:.2f}')
print(f'  GRAND TOTAL:       USD {c_baseline + c_main:.2f}')
print()
print('  (Estimate assumes ~1500 output tokens per cloud answer and ~30% judge-call rate.')
print('  Real cost for MMLU-Pro is typically 40-70% of this.)')
"

hdr "Summary"
echo "  passes:  $PASS"
echo "  warns:   $WARN"
echo "  fails:   $FAIL"
echo
if [ "$FAIL" -gt 0 ]; then
  echo "  ✗ Preflight FAILED. Fix the items above before running."
  exit 1
fi
echo "  ✓ Preflight PASSED. You can run:"
echo "      ./scripts/run_experiment.sh"

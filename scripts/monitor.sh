#!/usr/bin/env bash
# Live progress / cost / signal-health monitoring for an in-flight experiment.
# Run in a separate terminal while run_experiment.sh is executing.
#
# Usage:
#   ./scripts/monitor.sh [RUN_ID]
#
# If RUN_ID is not given, it shows whichever run_id has the most recent rows.

set -uo pipefail

DB_PATH="${DB_PATH:-autodidact_experiment.db}"

if [ ! -f "$DB_PATH" ]; then
  echo "DB not found: $DB_PATH (set DB_PATH=... if you used a different location)"
  exit 1
fi

RUN_ID="${1:-}"
if [ -z "$RUN_ID" ]; then
  RUN_ID=$(sqlite3 "$DB_PATH" "SELECT run_id FROM experiment_results ORDER BY created_at DESC LIMIT 1" 2>/dev/null || true)
  if [ -z "$RUN_ID" ]; then
    echo "No experiment_results rows yet. Pass a run_id, or wait for the harness to start writing."
    exit 1
  fi
  echo "Auto-detected RUN_ID=$RUN_ID (latest by created_at)"
fi

clear_and_print() {
  if command -v tput >/dev/null 2>&1; then tput cup 0 0; tput ed; else clear; fi
}

while true; do
  clear_and_print
  echo "═══════════════════════════════════════════════════════════════"
  echo "Autodidact experiment monitor — $(date +'%Y-%m-%d %H:%M:%S')"
  echo "  run_id: $RUN_ID"
  echo "  db:     $DB_PATH"
  echo "═══════════════════════════════════════════════════════════════"

  echo
  echo "── Progress ──"
  sqlite3 "$DB_PATH" <<SQL
.headers on
.mode column
.width 18 10 10 10
SELECT
  'eval_rows'             AS stage,
  COUNT(*)                AS rows,
  MIN(query_index)        AS min_idx,
  MAX(query_index)        AS max_idx
FROM experiment_results WHERE run_id = '$RUN_ID'
UNION ALL
SELECT
  'baseline_train_rows'   AS stage,
  COUNT(*)                AS rows,
  NULL                    AS min_idx,
  NULL                    AS max_idx
FROM routellm_training_rows
UNION ALL
SELECT
  'knowledge_entries'     AS stage,
  COUNT(*)                AS rows,
  NULL                    AS min_idx,
  NULL                    AS max_idx
FROM knowledge_entries WHERE valid_to IS NULL;
SQL

  echo
  echo "── Cost so far ──"
  sqlite3 "$DB_PATH" <<SQL
.headers on
.mode column
SELECT
  printf('%.4f', COALESCE(SUM(cost_cloud_answer_usd), 0))   AS cloud_usd,
  printf('%.4f', COALESCE(SUM(cost_judge_usd),  0))         AS judge_usd,
  printf('%.4f', COALESCE(SUM(cost_cloud_answer_usd + cost_judge_usd), 0)) AS total_usd
FROM experiment_results WHERE run_id = '$RUN_ID';
SQL

  echo
  echo "── Local accuracy so far (local_correct rate) ──"
  sqlite3 "$DB_PATH" <<SQL
.headers on
.mode column
SELECT
  COUNT(*)                                AS n_scored,
  ROUND(AVG(local_correct), 3)            AS local_acc,
  ROUND(AVG(cloud_correct), 3)            AS cloud_acc,
  ROUND(AVG(retrieval_recall_at_5), 3)    AS retrieval_recall_at_5,
  ROUND(AVG(judge_used), 3)               AS judge_rate
FROM experiment_results
WHERE run_id = '$RUN_ID' AND error_info IS NULL;
SQL

  echo
  echo "── GSA extraction-mode distribution (signal health) ──"
  sqlite3 "$DB_PATH" <<SQL
.headers on
.mode column
SELECT
  COALESCE(gsa_extraction_mode, 'NULL')   AS mode,
  COUNT(*)                                AS n,
  printf('%.1f%%', 100.0 * COUNT(*) / NULLIF((SELECT COUNT(*) FROM experiment_results WHERE run_id = '$RUN_ID'), 0)) AS pct
FROM experiment_results
WHERE run_id = '$RUN_ID'
GROUP BY gsa_extraction_mode
ORDER BY n DESC;
SQL

  echo
  echo "── Errors (last 5) ──"
  sqlite3 "$DB_PATH" <<SQL
.headers on
.mode column
.width 6 60
SELECT query_index, substr(error_info, 1, 60) AS error
FROM experiment_results
WHERE run_id = '$RUN_ID' AND error_info IS NOT NULL
ORDER BY query_index DESC LIMIT 5;
SQL

  echo
  echo "Press Ctrl+C to exit. Refreshing every 10s..."
  sleep 10
done

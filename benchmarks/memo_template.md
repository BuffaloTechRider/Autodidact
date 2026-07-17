# Autodidact v0.1 — Confidence Ablation Memo

**Run ID:** `{{run_id}}`
**Date:** {{date}}
**Local model:** {{local_model}}
**Cloud model:** {{cloud_model}} (answers + LLM-judge)
**Embedding model:** {{embedding_model}}
**Evaluation set:** {{n_total}} MMLU-Pro queries (stratified across categories, seed {{eval_seed}})
**Training set for RouteLLM baselines:** {{n_training_rows}} disjoint MMLU-Pro queries (seed {{train_seed}})

## Summary

{{summary_paragraph}}

## Per-Signal AUROC

How well each individual signal separates correct from incorrect local answers. AUROC of 0.5 is chance; 1.0 is perfect.

| Signal | AUROC | 95% CI | n |
|---|---|---|---|
{{per_signal_table}}

## Per-Combo AUROC

Fused-signal ablation. "Mean" is a simple arithmetic mean across signals; "Thompson" samples per-signal weights from their Beta(α, β) distributions each query.

| Combo | Mean AUROC | Mean CI | Thompson AUROC | Thompson CI | n |
|---|---|---|---|---|---|
{{per_combo_table}}

## Comparison to Prior Art

| Approach | AUROC | 95% CI |
|---|---|---|
| `routellm_no_memory` (memory-blind RouteLLM-style classifier) | {{routellm_no_memory_auroc}} | {{routellm_no_memory_ci}} |
| `routellm_plus_ks` (RouteLLM-style + knowledge similarity feature) | {{routellm_plus_ks_auroc}} | {{routellm_plus_ks_ci}} |
| **Our best combo (`{{best_combo}}`)** | **{{best_auroc}}** | {{best_ci}} |

**Headline number:** our best combo beats `routellm_plus_ks` by {{gap_vs_routellm_plus_ks}} AUROC.

{{prior_art_interpretation}}

## Calibration

![Best combo calibration]({{calibration_best_path}})

{{calibration_notes}}

## Retrieval Quality Stratification

Retrieval quality is a potential confound: if retrieval misses, knowledge-similarity and grounded-self-assessment both lose signal. The `retrieval_recall_at_5` proxy (does any top-5 retrieved entry match the query's MMLU-Pro category) lets us split the memo.

{{retrieval_section}}

## Recommendation

{{recommendation}}

### Reproducing this memo

```bash
python -m benchmarks.routellm_baseline --training-seed {{train_seed}}
python -m benchmarks.ablation_experiment --run-id {{run_id}} --local-model {{local_model}}
python -m benchmarks.ablation_analysis --run-id {{run_id}}
```

Artifacts: `results/experiment/summary.json`, `results/experiment/roc_overlay.png`, `results/experiment/calibration_*.png`, this memo.

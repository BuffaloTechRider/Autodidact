# A7 Baseline (local-only) — NFR-2 ROI gate

Tasks: 12   Completed locally: 12/12 (100%)   Total steps: 44

## Intended-tier distribution (from local avg_logprob vs thresholds)

Cloud share here = the % of steps the router *would* escalate.

  LOCAL :   28  ( 87.5%)
  VERIFY:    3  (  9.4%)
  CLOUD :    1  (  3.1%)
  (memory: 0.0% by construction — no cloud escalation, nothing learned)

## Separability: intended-tier mix by a-priori difficulty

  easy  : LOCAL 100.0%  VERIFY   0.0%  CLOUD   0.0%   | completed 4/4
  medium: LOCAL  78.6%  VERIFY  14.3%  CLOUD   7.1%   | completed 4/4
  hard  : LOCAL  90.9%  VERIFY   9.1%  CLOUD   0.0%   | completed 4/4

## Correctness (verified tasks only)
  Verified: 4/12   Correct: 4/4 (100%)

  By a-priori difficulty:
    easy  : 2/2 correct
    medium: 1/1 correct
    hard  : 1/1 correct

  Correctness by task's peak intended tier:
    LOCAL : 2/2 correct
    VERIFY: 1/1 correct
    CLOUD : 1/1 correct

  Confident-but-WRONG (peak tier LOCAL, incorrect): 0  []
  → these are the routing gap: high self-confidence, wrong answer,
    no escalation triggered. If this set is large, logprob-only
    routing is miscalibrated and GSA/verification earns its keep.

## Real cloud escalation (--cloud mode)
  Tasks that escalated: 1/12   Total escalations: 1   Cost: $0.0100
  Avg cost/task: $0.0008

## Read
If LOCAL% falls and CLOUD% rises from easy→hard, difficulty is
separable and per-step routing pays off. A flat mix means it's
entangled. Separately, a large confident-but-wrong set means
logprob alone is miscalibrated — the case for GSA + verification.

## Per-task
  e07 [easy  ] ✓ steps= 2 ok  (done)  LOCAL:1
  e08 [easy  ] ? steps= 3 ok  (done)  LOCAL:2
  e09 [easy  ] ✓ steps= 3 ok  (done)  LOCAL:2
  e14 [easy  ] ? steps= 3 ok  (done)  LOCAL:2
  h01 [hard  ] ? steps= 4 ok  (done)  LOCAL:3
  h08 [hard  ] ? steps= 3 ok  (done)  LOCAL:2
  h12 [hard  ] ✓ steps= 5 ok  (done)  LOCAL:3 VERIFY:1
  h13 [hard  ] ? steps= 3 ok  (done)  LOCAL:2
  m06 [medium] ? steps= 3 ok  (done)  LOCAL:1 VERIFY:1
  m08 [medium] ? steps= 4 ok  (done)  LOCAL:3
  m11 [medium] ✓ steps= 8 ok  (done)  CLOUD:1 LOCAL:5 VERIFY:1
  m16 [medium] ? steps= 3 ok  (done)  LOCAL:2
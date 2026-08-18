# A7 Baseline (local-only) — NFR-2 ROI gate

Tasks: 53   Completed locally: 52/53 (98%)   Total steps: 185

## Intended-tier distribution (from local avg_logprob vs thresholds)

Cloud share here = the % of steps the router *would* escalate.

  LOCAL :  118  ( 89.4%)
  VERIFY:   14  ( 10.6%)
  CLOUD :    0  (  0.0%)
  (memory: 0.0% by construction — no cloud escalation, nothing learned)

## Separability: intended-tier mix by a-priori difficulty

  easy  : LOCAL  96.2%  VERIFY   3.8%  CLOUD   0.0%   | completed 15/15
  medium: LOCAL  86.0%  VERIFY  14.0%  CLOUD   0.0%   | completed 18/18
  hard  : LOCAL  88.9%  VERIFY  11.1%  CLOUD   0.0%   | completed 19/20

## Correctness (verified tasks only)
  Verified: 20/53   Correct: 16/20 (80%)

  By a-priori difficulty:
    easy  : 6/6 correct
    medium: 4/5 correct
    hard  : 6/9 correct

  Correctness by task's peak intended tier:
    LOCAL : 12/14 correct
    VERIFY: 4/6 correct

  Confident-but-WRONG (peak tier LOCAL, incorrect): 2  ['m09', 'h14']
  → these are the routing gap: high self-confidence, wrong answer,
    no escalation triggered. If this set is large, logprob-only
    routing is miscalibrated and GSA/verification earns its keep.

## Read
If LOCAL% falls and CLOUD% rises from easy→hard, difficulty is
separable and per-step routing pays off. A flat mix means it's
entangled. Separately, a large confident-but-wrong set means
logprob alone is miscalibrated — the case for GSA + verification.

## Per-task
  e01 [easy  ] ✓ steps= 3 ok  (done)  LOCAL:2
  e02 [easy  ] ? steps= 3 ok  (done)  LOCAL:2
  e03 [easy  ] ✓ steps= 2 ok  (done)  LOCAL:1
  e04 [easy  ] ? steps= 2 ok  (done)  LOCAL:1
  e05 [easy  ] ? steps= 2 ok  (done)  LOCAL:1
  e06 [easy  ] ? steps= 5 ok  (done)  LOCAL:4
  e07 [easy  ] ✓ steps= 2 ok  (done)  LOCAL:1
  e08 [easy  ] ? steps= 3 ok  (done)  LOCAL:2
  e09 [easy  ] ✓ steps= 3 ok  (done)  LOCAL:2
  e10 [easy  ] ? steps= 2 ok  (done)  LOCAL:1
  e11 [easy  ] ✓ steps= 2 ok  (done)  LOCAL:1
  e12 [easy  ] ? steps= 2 ok  (done)  LOCAL:1
  e13 [easy  ] ✓ steps= 4 ok  (done)  LOCAL:3
  e14 [easy  ] ? steps= 3 ok  (done)  LOCAL:2
  e15 [easy  ] ? steps= 3 ok  (done)  LOCAL:1 VERIFY:1
  h01 [hard  ] ? steps= 4 ok  (done)  LOCAL:3
  h02 [hard  ] ? steps= 2 INC (done)  LOCAL:1
  h03 [hard  ] ? steps= 6 ok  (done)  LOCAL:3 VERIFY:2
  h04 [hard  ] ✗ steps= 4 ok  (done)  LOCAL:2 VERIFY:1
  h05 [hard  ] ? steps= 4 ok  (done)  LOCAL:3
  h06 [hard  ] ? steps= 7 ok  (done)  LOCAL:6
  h07 [hard  ] ✓ steps= 3 ok  (done)  LOCAL:2
  h08 [hard  ] ? steps= 3 ok  (done)  LOCAL:2
  h09 [hard  ] ? steps= 3 ok  (done)  LOCAL:2
  h10 [hard  ] ✓ steps= 3 ok  (done)  LOCAL:2
  h11 [hard  ] ? steps= 4 ok  (done)  LOCAL:3
  h12 [hard  ] ✓ steps= 5 ok  (done)  LOCAL:3 VERIFY:1
  h13 [hard  ] ? steps= 3 ok  (done)  LOCAL:2
  h14 [hard  ] ✗ steps= 5 ok  (done)  LOCAL:4
  h15 [hard  ] ? steps= 4 ok  (done)  LOCAL:3
  h16 [hard  ] ✓ steps= 6 ok  (done)  LOCAL:4 VERIFY:1
  h17 [hard  ] ? steps= 3 ok  (done)  LOCAL:2
  h18 [hard  ] ✗ steps= 8 ok  (done)  LOCAL:6 VERIFY:1
  h19 [hard  ] ✓ steps= 3 ok  (done)  LOCAL:2
  h20 [hard  ] ✓ steps= 3 ok  (done)  LOCAL:1 VERIFY:1
  m01 [medium] ? steps= 5 ok  (done)  LOCAL:3 VERIFY:1
  m02 [medium] ? steps= 2 ok  (done)  LOCAL:1
  m03 [medium] ✓ steps= 3 ok  (done)  LOCAL:2
  m04 [medium] ? steps= 5 ok  (done)  LOCAL:4
  m05 [medium] ? steps= 3 ok  (done)  LOCAL:2
  m06 [medium] ? steps= 3 ok  (done)  LOCAL:1 VERIFY:1
  m07 [medium] ? steps= 2 ok  (done)  LOCAL:1
  m08 [medium] ? steps= 4 ok  (done)  LOCAL:3
  m09 [medium] ✗ steps= 3 ok  (done)  LOCAL:2
  m10 [medium] ? steps= 5 ok  (done)  LOCAL:3 VERIFY:1
  m11 [medium] ✓ steps= 4 ok  (done)  LOCAL:3
  m12 [medium] ? steps= 3 ok  (done)  LOCAL:2
  m13 [medium] ✓ steps= 3 ok  (done)  LOCAL:1 VERIFY:1
  m14 [medium] ? steps= 3 ok  (done)  VERIFY:2
  m15 [medium] ? steps= 3 ok  (done)  LOCAL:2
  m16 [medium] ? steps= 3 ok  (done)  LOCAL:2
  m17 [medium] ? steps= 4 ok  (done)  LOCAL:3
  m18 [medium] ✓ steps= 3 ok  (done)  LOCAL:2
# A7 Baseline (cloud-enabled, hardened corpus) — NFR-2 ROI gate

> Clean run (fresh AWS token): 58/59 completed, no token failures, cost
> section populated. This supersedes the token-truncated earlier run.
>
> **Bottom line:** cloud share is tiny (**3/59 tasks, ~2% of steps, $0.02
> total**) because the local model is confident on almost everything — but
> that confidence is **wrong ~23% of the time** (24/31 verified correct;
> hard only 10/16), including **4 confident-but-wrong tasks** the router
> never escalated. So the *saving* is real (cheap) but *unguarded*: logprob
> alone doesn't catch the errors. This is the empirical case for GSA +
> verification over logprob-only routing, and a caution on per-step routing
> ROI (difficulty is entangled — tier mix is flat across easy/med/hard).

Tasks: 59   Completed locally: 58/59 (98%)   Total steps: 206

## Intended-tier distribution (from local avg_logprob vs thresholds)

Cloud share here = the % of steps the router *would* escalate.

  LOCAL :  133  ( 90.5%)
  VERIFY:   11  (  7.5%)
  CLOUD :    3  (  2.0%)
  (memory: 0.0% by construction — no cloud escalation, nothing learned)

## Separability: intended-tier mix by a-priori difficulty

  easy  : LOCAL  96.2%  VERIFY   3.8%  CLOUD   0.0%   | completed 15/15
  medium: LOCAL  86.7%  VERIFY   8.9%  CLOUD   4.4%   | completed 18/18
  hard  : LOCAL  90.8%  VERIFY   7.9%  CLOUD   1.3%   | completed 25/26

## Correctness (verified tasks only)
  Verified: 31/59   Correct: 24/31 (77%)

  By a-priori difficulty:
    easy  : 8/8 correct
    medium: 6/7 correct
    hard  : 10/16 correct

  Correctness by task's peak intended tier:
    LOCAL : 20/24 correct
    VERIFY: 3/6 correct
    CLOUD : 1/1 correct

  Confident-but-WRONG (peak tier LOCAL, incorrect): 4  ['m09', 'x04', 'x05', 'x06']
  → these are the routing gap: high self-confidence, wrong answer,
    no escalation triggered. If this set is large, logprob-only
    routing is miscalibrated and GSA/verification earns its keep.

## Real cloud escalation (--cloud mode)
  Tasks that escalated: 3/59   Total escalations: 3   Cost: $0.0205
  Avg cost/task: $0.0003

## Read
If LOCAL% falls and CLOUD% rises from easy→hard, difficulty is
separable and per-step routing pays off. A flat mix means it's
entangled. Separately, a large confident-but-wrong set means
logprob alone is miscalibrated — the case for GSA + verification.

## Per-task
  e01 [easy  ] ✓ steps= 3 ok  (done)  LOCAL:2
  e02 [easy  ] ? steps= 3 ok  (done)  LOCAL:2
  e03 [easy  ] ✓ steps= 2 ok  (done)  LOCAL:1
  e04 [easy  ] ✓ steps= 2 ok  (done)  LOCAL:1
  e05 [easy  ] ✓ steps= 2 ok  (done)  LOCAL:1
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
  h03 [hard  ] ? steps= 6 ok  (done)  LOCAL:4 VERIFY:1
  h04 [hard  ] ✗ steps= 4 ok  (done)  LOCAL:2 VERIFY:1
  h05 [hard  ] ✓ steps= 4 ok  (done)  LOCAL:3
  h06 [hard  ] ? steps= 7 ok  (done)  LOCAL:6
  h07 [hard  ] ✓ steps= 3 ok  (done)  LOCAL:2
  h08 [hard  ] ? steps= 3 ok  (done)  LOCAL:2
  h09 [hard  ] ? steps= 4 ok  (done)  LOCAL:3
  h10 [hard  ] ✓ steps= 4 ok  (done)  LOCAL:3
  h11 [hard  ] ? steps= 3 ok  (done)  LOCAL:2
  h12 [hard  ] ✗ steps= 4 ok  (done)  LOCAL:2 VERIFY:1
  h13 [hard  ] ? steps= 3 ok  (done)  LOCAL:2
  h14 [hard  ] ✓ steps= 7 ok  (done)  LOCAL:6
  h15 [hard  ] ? steps= 4 ok  (done)  LOCAL:3
  h16 [hard  ] ✓ steps= 5 ok  (done)  CLOUD:1 LOCAL:3
  h17 [hard  ] ? steps= 3 ok  (done)  LOCAL:2
  h18 [hard  ] ✓ steps= 3 ok  (done)  LOCAL:2
  h19 [hard  ] ✓ steps= 3 ok  (done)  LOCAL:2
  h20 [hard  ] ✓ steps= 3 ok  (done)  LOCAL:1 VERIFY:1
  x01 [hard  ] ✓ steps= 4 ok  (done)  LOCAL:3
  x02 [hard  ] ✗ steps= 3 ok  (done)  LOCAL:1 VERIFY:1
  x03 [hard  ] ✓ steps= 3 ok  (done)  LOCAL:1 VERIFY:1
  x04 [hard  ] ✗ steps= 4 ok  (done)  LOCAL:3
  x05 [hard  ] ✗ steps= 5 ok  (done)  LOCAL:4
  x06 [hard  ] ✗ steps= 4 ok  (done)  LOCAL:3
  m01 [medium] ? steps= 5 ok  (done)  LOCAL:3 VERIFY:1
  m02 [medium] ? steps= 2 ok  (done)  LOCAL:1
  m03 [medium] ✓ steps= 4 ok  (done)  LOCAL:3
  m04 [medium] ? steps= 4 ok  (done)  LOCAL:3
  m05 [medium] ✓ steps= 3 ok  (done)  LOCAL:2
  m06 [medium] ? steps= 3 ok  (done)  LOCAL:1 VERIFY:1
  m07 [medium] ? steps= 2 ok  (done)  LOCAL:1
  m08 [medium] ? steps= 5 ok  (done)  CLOUD:1 LOCAL:3
  m09 [medium] ✗ steps= 3 ok  (done)  LOCAL:2
  m10 [medium] ? steps= 5 ok  (done)  LOCAL:3 VERIFY:1
  m11 [medium] ✓ steps= 3 ok  (done)  LOCAL:2
  m12 [medium] ? steps= 5 ok  (done)  LOCAL:4
  m13 [medium] ✓ steps= 3 ok  (done)  LOCAL:1 VERIFY:1
  m14 [medium] ? steps= 2 ok  (done)  CLOUD:1
  m15 [medium] ✓ steps= 3 ok  (done)  LOCAL:2
  m16 [medium] ? steps= 4 ok  (done)  LOCAL:3
  m17 [medium] ? steps= 4 ok  (done)  LOCAL:3
  m18 [medium] ✓ steps= 3 ok  (done)  LOCAL:2
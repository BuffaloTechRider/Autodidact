# A7 Baseline (cloud-enabled, hardened corpus) — NFR-2 ROI gate

> **Caveat (read first):** the AWS/Midway session token expired ~13 min into
> this ~18-min run. `ExpiredTokenException` only fires on a Converse call, which
> only happens on **escalation** — so the 5 `error:ClientError` tasks
> (e15, m10, m14, h14, h18) are precisely the ones that *tried to escalate* and
> hit the dead token. Consequence: the **local-routing + correctness data (54
> valid tasks) is solid**, but the **cloud cost/share figure is unreliable** —
> escalations after token-death errored out instead of completing, so no cost
> section was emitted. Re-run with a fresh token to get the real cloud share.
>
> The signal that matters is intact and *strengthened* vs the first corpus:
> harder tasks + more verifiers → correctness fell to **72%** (hard: 8/14) and
> the confident-but-wrong set **tripled to 6** — direct evidence that
> logprob-only routing is miscalibrated (the case for GSA + verification).

Tasks: 59   Completed locally: 53/59 (90%)   Total steps: 188

## Intended-tier distribution (from local avg_logprob vs thresholds)

Cloud share here = the % of steps the router *would* escalate.

  LOCAL :  124  ( 92.5%)
  VERIFY:   10  (  7.5%)
  CLOUD :    0  (  0.0%)
  (memory: 0.0% by construction — no cloud escalation, nothing learned)

## Separability: intended-tier mix by a-priori difficulty

  easy  : LOCAL 100.0%  VERIFY   0.0%  CLOUD   0.0%   | completed 14/15
  medium: LOCAL  91.9%  VERIFY   8.1%  CLOUD   0.0%   | completed 16/18
  hard  : LOCAL  90.3%  VERIFY   9.7%  CLOUD   0.0%   | completed 23/26

## Correctness (verified tasks only)
  Verified: 29/59   Correct: 21/29 (72%)

  By a-priori difficulty:
    easy  : 8/8 correct
    medium: 5/7 correct
    hard  : 8/14 correct

  Correctness by task's peak intended tier:
    LOCAL : 17/23 correct
    VERIFY: 4/6 correct

  Confident-but-WRONG (peak tier LOCAL, incorrect): 6  ['m09', 'm11', 'x01', 'x04', 'x05', 'x06']
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
  e04 [easy  ] ✓ steps= 2 ok  (done)  LOCAL:1
  e05 [easy  ] ✓ steps= 3 ok  (done)  LOCAL:2
  e06 [easy  ] ? steps= 5 ok  (done)  LOCAL:4
  e07 [easy  ] ✓ steps= 2 ok  (done)  LOCAL:1
  e08 [easy  ] ? steps= 3 ok  (done)  LOCAL:2
  e09 [easy  ] ✓ steps= 3 ok  (done)  LOCAL:2
  e10 [easy  ] ? steps= 2 ok  (done)  LOCAL:1
  e11 [easy  ] ✓ steps= 2 ok  (done)  LOCAL:1
  e12 [easy  ] ? steps= 2 ok  (done)  LOCAL:1
  e13 [easy  ] ✓ steps= 4 ok  (done)  LOCAL:3
  e14 [easy  ] ? steps= 3 ok  (done)  LOCAL:2
  e15 [easy  ] ? steps= 0 INC (error:ClientError)  
  h01 [hard  ] ? steps= 4 ok  (done)  LOCAL:3
  h02 [hard  ] ? steps= 2 INC (done)  LOCAL:1
  h03 [hard  ] ? steps= 5 ok  (done)  LOCAL:3 VERIFY:1
  h04 [hard  ] ✗ steps= 4 ok  (done)  LOCAL:2 VERIFY:1
  h05 [hard  ] ✓ steps= 4 ok  (done)  LOCAL:3
  h06 [hard  ] ? steps= 7 ok  (done)  LOCAL:5 VERIFY:1
  h07 [hard  ] ✓ steps= 3 ok  (done)  LOCAL:2
  h08 [hard  ] ? steps= 3 ok  (done)  LOCAL:2
  h09 [hard  ] ? steps= 3 ok  (done)  LOCAL:2
  h10 [hard  ] ✓ steps= 3 ok  (done)  LOCAL:2
  h11 [hard  ] ? steps= 4 ok  (done)  LOCAL:3
  h12 [hard  ] ✓ steps= 5 ok  (done)  LOCAL:3 VERIFY:1
  h13 [hard  ] ? steps= 3 ok  (done)  LOCAL:2
  h14 [hard  ] ? steps= 0 INC (error:ClientError)  
  h15 [hard  ] ? steps= 4 ok  (done)  LOCAL:3
  h16 [hard  ] ✓ steps= 6 ok  (done)  LOCAL:5
  h17 [hard  ] ? steps= 3 ok  (done)  LOCAL:2
  h18 [hard  ] ? steps= 0 INC (error:ClientError)  
  h19 [hard  ] ✓ steps= 3 ok  (done)  LOCAL:2
  h20 [hard  ] ✓ steps= 4 ok  (done)  LOCAL:2 VERIFY:1
  x01 [hard  ] ✗ steps= 7 ok  (done)  LOCAL:6
  x02 [hard  ] ✗ steps= 3 ok  (done)  LOCAL:1 VERIFY:1
  x03 [hard  ] ✓ steps= 3 ok  (done)  LOCAL:1 VERIFY:1
  x04 [hard  ] ✗ steps= 4 ok  (done)  LOCAL:3
  x05 [hard  ] ✗ steps= 5 ok  (done)  LOCAL:4
  x06 [hard  ] ✗ steps= 4 ok  (done)  LOCAL:3
  m01 [medium] ? steps= 3 ok  (done)  LOCAL:1 VERIFY:1
  m02 [medium] ? steps= 2 ok  (done)  LOCAL:1
  m03 [medium] ✓ steps= 3 ok  (done)  LOCAL:2
  m04 [medium] ? steps= 4 ok  (done)  LOCAL:3
  m05 [medium] ✓ steps= 3 ok  (done)  LOCAL:2
  m06 [medium] ? steps= 4 ok  (done)  LOCAL:2 VERIFY:1
  m07 [medium] ? steps= 2 ok  (done)  LOCAL:1
  m08 [medium] ? steps= 4 ok  (done)  LOCAL:3
  m09 [medium] ✗ steps= 3 ok  (done)  LOCAL:2
  m10 [medium] ? steps= 0 INC (error:ClientError)  
  m11 [medium] ✗ steps= 4 ok  (done)  LOCAL:3
  m12 [medium] ? steps= 5 ok  (done)  LOCAL:4
  m13 [medium] ✓ steps= 3 ok  (done)  LOCAL:1 VERIFY:1
  m14 [medium] ? steps= 0 INC (error:ClientError)  
  m15 [medium] ✓ steps= 3 ok  (done)  LOCAL:2
  m16 [medium] ? steps= 3 ok  (done)  LOCAL:2
  m17 [medium] ? steps= 4 ok  (done)  LOCAL:3
  m18 [medium] ✓ steps= 3 ok  (done)  LOCAL:2
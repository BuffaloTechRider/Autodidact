# A7 + GSA pre-gate (v4 adversarial-trust) — verification-path validation

> **Verdict: the v4 GSA pre-gate does NOT earn its keep here.** Compared to the
> GSA-off cloud run (`a7_cloud_full.md`):
>
> | metric | GSA off | GSA on (v4) |
> |---|---|---|
> | escalated tasks | 3/59 | **28/59** |
> | escalations | 3 | **44** |
> | cost | $0.02 | **$0.25** (~12×) |
> | correctness | 24/31 (77%) | **21/30 (70%)** |
> | confident-but-wrong | 4 | **5** |
>
> The gate fires **far too often** (30% of steps → CLOUD) yet correctness went
> *down*, not up. Crucially, it did **not** rescue the target confident-but-wrong
> tasks: m09, x04, x05 stayed peak-LOCAL and wrong (the gate never escalated
> them); x06 errored on a transient network blip so is inconclusive. So the v4
> pre-gate escalates the *wrong* steps — it's trigger-happy on easy tasks (easy
> CLOUD share jumped 0% → 37%) while still missing the actual errors, because
> those errors are in the model's *reasoning on a step it's confident about*,
> which a "can you do the next action?" probe doesn't interrogate.
>
> **Implication:** a per-step pre-gate is the wrong shape for confident-but-wrong
> *final answers*. This is the empirical case for **post-hoc answer verification**
> (check the produced result, escalate on failure) over a pre-gate — see the
> A7 follow-up. Two runs also hit a transient EndpointConnectionError / expired
> token (x06), unrelated to the finding.

Tasks: 59   Completed locally: 55/59 (93%)   Total steps: 206

## Intended-tier distribution (from local avg_logprob vs thresholds)

Cloud share here = the % of steps the router *would* escalate.

  LOCAL :   98  ( 66.2%)
  VERIFY:    6  (  4.1%)
  CLOUD :   44  ( 29.7%)
  (memory: 0.0% by construction — no cloud escalation, nothing learned)

## Separability: intended-tier mix by a-priori difficulty

  easy  : LOCAL  63.0%  VERIFY   0.0%  CLOUD  37.0%   | completed 15/15
  medium: LOCAL  57.9%  VERIFY   2.6%  CLOUD  39.5%   | completed 16/18
  hard  : LOCAL  71.1%  VERIFY   6.0%  CLOUD  22.9%   | completed 24/26

## Correctness (verified tasks only)
  Verified: 30/59   Correct: 21/30 (70%)

  By a-priori difficulty:
    easy  : 8/8 correct
    medium: 4/7 correct
    hard  : 9/15 correct

  Correctness by task's peak intended tier:
    LOCAL : 13/18 correct
    VERIFY: 0/2 correct
    CLOUD : 8/10 correct

  Confident-but-WRONG (peak tier LOCAL, incorrect): 5  ['m09', 'm18', 'h14', 'x04', 'x05']
  → these are the routing gap: high self-confidence, wrong answer,
    no escalation triggered. If this set is large, logprob-only
    routing is miscalibrated and GSA/verification earns its keep.

## Real cloud escalation (--cloud mode)
  Tasks that escalated: 28/59   Total escalations: 44   Cost: $0.2533
  Avg cost/task: $0.0043

## Read
If LOCAL% falls and CLOUD% rises from easy→hard, difficulty is
separable and per-step routing pays off. A flat mix means it's
entangled. Separately, a large confident-but-wrong set means
logprob alone is miscalibrated — the case for GSA + verification.

## Per-task
  e01 [easy  ] ✓ steps= 3 ok  (done)  CLOUD:1 LOCAL:1
  e02 [easy  ] ? steps= 3 ok  (done)  LOCAL:2
  e03 [easy  ] ✓ steps= 2 ok  (done)  LOCAL:1
  e04 [easy  ] ✓ steps= 2 ok  (done)  CLOUD:1
  e05 [easy  ] ✓ steps= 3 ok  (done)  LOCAL:2
  e06 [easy  ] ? steps= 5 ok  (done)  CLOUD:4
  e07 [easy  ] ✓ steps= 2 ok  (done)  CLOUD:1
  e08 [easy  ] ? steps= 3 ok  (done)  LOCAL:2
  e09 [easy  ] ✓ steps= 3 ok  (done)  LOCAL:2
  e10 [easy  ] ? steps= 2 ok  (done)  CLOUD:1
  e11 [easy  ] ✓ steps= 2 ok  (done)  CLOUD:1
  e12 [easy  ] ? steps= 2 ok  (done)  LOCAL:1
  e13 [easy  ] ✓ steps= 4 ok  (done)  LOCAL:3
  e14 [easy  ] ? steps= 3 ok  (done)  LOCAL:2
  e15 [easy  ] ? steps= 3 ok  (done)  CLOUD:1 LOCAL:1
  h01 [hard  ] ? steps= 4 ok  (done)  LOCAL:3
  h02 [hard  ] ? steps= 2 INC (done)  LOCAL:1
  h03 [hard  ] ? steps= 5 ok  (done)  CLOUD:2 LOCAL:1 VERIFY:1
  h04 [hard  ] ✓ steps= 5 ok  (done)  CLOUD:1 LOCAL:2 VERIFY:1
  h05 [hard  ] ✓ steps= 4 ok  (done)  LOCAL:3
  h06 [hard  ] ? steps=10 ok  (done)  CLOUD:4 LOCAL:5
  h07 [hard  ] ✓ steps= 3 ok  (done)  LOCAL:2
  h08 [hard  ] ? steps= 4 ok  (done)  CLOUD:2 LOCAL:1
  h09 [hard  ] ? steps= 4 ok  (done)  CLOUD:1 LOCAL:2
  h10 [hard  ] ✓ steps= 3 ok  (done)  LOCAL:2
  h11 [hard  ] ? steps= 5 ok  (done)  LOCAL:4
  h12 [hard  ] ✓ steps= 5 ok  (done)  CLOUD:2 LOCAL:1 VERIFY:1
  h13 [hard  ] ? steps= 3 ok  (done)  LOCAL:2
  h14 [hard  ] ✗ steps= 5 ok  (done)  LOCAL:4
  h15 [hard  ] ? steps= 5 ok  (done)  CLOUD:3 LOCAL:1
  h16 [hard  ] ✓ steps= 5 ok  (done)  CLOUD:1 LOCAL:3
  h17 [hard  ] ? steps= 3 ok  (done)  LOCAL:2
  h18 [hard  ] ✗ steps= 6 ok  (done)  CLOUD:1 LOCAL:4
  h19 [hard  ] ✓ steps= 3 ok  (done)  LOCAL:2
  h20 [hard  ] ✓ steps= 5 ok  (done)  CLOUD:2 LOCAL:2
  x01 [hard  ] ✓ steps= 4 ok  (done)  LOCAL:3
  x02 [hard  ] ✗ steps= 3 ok  (done)  LOCAL:1 VERIFY:1
  x03 [hard  ] ✗ steps= 3 ok  (done)  LOCAL:1 VERIFY:1
  x04 [hard  ] ✗ steps= 4 ok  (done)  LOCAL:3
  x05 [hard  ] ✗ steps= 5 ok  (done)  LOCAL:4
  x06 [hard  ] ? steps= 0 INC (error:LLMClientError)  
  m01 [medium] ? steps= 4 ok  (done)  CLOUD:1 LOCAL:1 VERIFY:1
  m02 [medium] ? steps= 2 ok  (done)  CLOUD:1
  m03 [medium] ✓ steps= 3 ok  (done)  LOCAL:2
  m04 [medium] ? steps= 4 ok  (done)  CLOUD:1 LOCAL:2
  m05 [medium] ✓ steps= 3 ok  (done)  LOCAL:2
  m06 [medium] ? steps= 4 ok  (done)  CLOUD:3
  m07 [medium] ? steps= 3 ok  (done)  CLOUD:1 LOCAL:1
  m08 [medium] ? steps= 3 ok  (done)  LOCAL:2
  m09 [medium] ✗ steps= 3 ok  (done)  LOCAL:2
  m10 [medium] ? steps= 2 ok  (done)  CLOUD:1
  m11 [medium] ✓ steps= 3 ok  (done)  LOCAL:2
  m12 [medium] ? steps= 4 ok  (done)  CLOUD:3
  m13 [medium] ✗ steps= 2 INC (done)  CLOUD:1
  m14 [medium] ? steps= 2 ok  (done)  CLOUD:1
  m15 [medium] ✓ steps= 3 ok  (done)  LOCAL:2
  m16 [medium] ? steps= 4 ok  (done)  CLOUD:1 LOCAL:2
  m17 [medium] ? steps= 4 ok  (done)  CLOUD:1 LOCAL:2
  m18 [medium] ✗ steps= 3 INC (done)  LOCAL:2
# Problem Statement — Autodidact v2.0 (Apprentice Agent)

> **Lifecycle doc 1 of 4.** No solution language. If this doc names a module,
> API, or algorithm, that has leaked from design — move it to `03-design.md`.
> Next: [`01-requirements.md`](01-requirements.md).

## Who has the problem

Developers running agentic LLM workflows who pay per cloud token. Every task
step — trivial or hard — hits the expensive frontier model, even the ones a
cheap local model (or a past answer) could handle.

## When it bites

- A multi-step task fires N cloud calls; most steps are routine (file reads,
  boilerplate edits, lookups already answered once).
- Cost and latency scale with task length, not task difficulty.
- The system never gets cheaper: solving the same class of problem tomorrow
  costs exactly what it cost today. There is no compounding.

## What "solved" looks like

- Routine steps are handled locally or from memory; only genuinely hard steps
  escalate to the cloud.
- The share of steps needing the cloud **drops over time** as the agent learns
  from its own escalations.
- No accuracy regression: cheaper routing must not increase wrong answers.

## Explicitly out of scope

- Training or fine-tuning a custom model (routing is model-agnostic by design).
- Multi-user / hosted service concerns (auth, quotas, tenancy).
- Non-text modalities.

## Adversarial check (challenge the framing)

- **What am I assuming that might be wrong?** That a local 7B model is good enough
  for "routine" steps. If routine steps still fail locally often, step-level
  routing just adds latency (double-generation on Tier 2) without saving cloud
  calls. Mitigation is in the design: GSA pre-gate escalates immediately when the
  local model would hedge, so we don't pay double-generation on hopeless steps.
- **Also assuming** that step difficulty is separable — that a task cleanly splits
  into cheap and hard steps. If hardness is entangled (every step needs the whole
  task's context), per-step routing degenerates to whole-task routing and the win
  evaporates.
- **Cheapest way to invalidate before building:** instrument the existing v1.0
  agent to log the cloud/local/memory split on ~50 real multi-step tasks. If the
  cloud share is already low, the apprentice-agent ROI is small; if most steps are
  cloud, confirm how many are genuinely hard vs. just uncached. This is the NFR-2
  baseline measurement and should run before Phase B.
- **Who would say this isn't worth solving?** Someone who'd rather just use a
  bigger local model (no routing needed) or accept the cloud bill for simplicity.
  Our answer: the compounding-savings curve (NFR-2) is the differentiator; a static
  setup never gets cheaper.

## Source context

Derived from memory `project_overview.md` and `project_v2_design.md`.

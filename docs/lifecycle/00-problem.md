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

## Adversarial check (fill in — challenge the framing)

- What am I assuming that might be wrong? _(e.g. "local model is good enough for
  routine steps" — is it, on the target task distribution?)_
- Cheapest way to invalidate this before building? _(e.g. measure actual cloud /
  local / memory-hit split on 50 real tasks.)_
- Who would say this isn't worth solving, and why?

## Source context

Derived from memory `project_overview.md` and `project_v2_design.md`.

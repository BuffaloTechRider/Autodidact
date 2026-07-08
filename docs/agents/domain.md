# Domain Docs — Consumer Rules

This repo uses a **single-context** layout: one `CONTEXT.md` at the repo root and architectural decision records under `docs/adr/`.

## Layout

```
CONTEXT.md              # domain language, bounded contexts, key abstractions
docs/adr/
  001-initial-architecture.md
  002-routing-signal-choice.md
  ...
```

## Consumer rules for agent skills

### Reading CONTEXT.md

- Skills that need to understand the project's domain language (`improve-codebase-architecture`, `diagnose`, `tdd`, `grill-with-docs`) read `CONTEXT.md` before starting work.
- If `CONTEXT.md` doesn't exist yet, the skill should note this and offer to help create it — but never block on its absence.
- `CONTEXT.md` is the source of truth for terminology. If the code uses a term differently from `CONTEXT.md`, flag it as a potential inconsistency.

### Reading ADRs

- ADRs live in `docs/adr/` and follow the format: `NNN-short-title.md`.
- Skills read ADRs to understand past decisions before proposing changes that might contradict them.
- If a skill's recommendation conflicts with an existing ADR, it should cite the ADR and ask the user whether to supersede it.

### Writing

- Skills may propose updates to `CONTEXT.md` when they discover new domain terms or refine existing definitions.
- Skills may propose new ADRs when they make architectural decisions worth recording.
- All writes to domain docs require user confirmation — never auto-commit.

## Creating CONTEXT.md

If `CONTEXT.md` doesn't exist yet, a good starting template:

```markdown
# Autodidact — Domain Context

## What this project is

Autodidact is a self-learning AI framework that routes queries between cheap and expensive models, learning from every escalation.

## Key abstractions

- **Agent** — the central API; ties together routing, knowledge store, and LLM clients
- **Router** — decides local vs cloud based on confidence signals
- **Knowledge Store** — SQLite + FAISS store of past Q&A pairs from cloud escalations
- **Confidence Signal** — a scalar in [0,1] predicting whether the local model's answer is correct
- **logprob_uncertainty** — the primary signal; average per-token log-probability mapped through a sigmoid

## Bounded contexts

- `autodidact/` — core framework (routing, KB, signals, LLM client)
- `benchmarks/` — experiment harness and analysis (not shipped in the product)
- `results/` — experiment data and reports (not shipped)
```

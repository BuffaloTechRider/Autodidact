# Requirements — Autodidact v2.0

> **Lifecycle doc 2 of 4.** Every requirement has an ID and a **testable**
> acceptance criterion, and later maps to a test and a component in
> [`03-design.md`](03-design.md). Prev: [`00-problem.md`](00-problem.md).

## Functional (must)

| ID | Requirement | Acceptance criterion (testable) |
|----|-------------|--------------------------------|
| FR-1 | Route each step across tiers (memory / local-fast / local-verified / cloud) | Given a step with a known best tier, the router selects it; unit test per tier with fixed signals |
| FR-2 | Execute multi-step tasks via a ReAct loop with tool calls | A task requiring ≥2 tool calls completes end-to-end in an integration test |
| FR-3 | Learn a reusable skill from a cloud escalation | After a cloud-solved task, a skill exists and is retrievable by semantic match |
| FR-4 | Reuse a stored skill on a similar later task without escalating | Second run of a similar task resolves at memory/local tier — 0 cloud calls |
| FR-5 | Tools self-register and expose function-calling schemas | Registry lists all tools; schema validates against OpenAI function format |

## Non-functional (should)

| ID | Requirement | Acceptance criterion |
|----|-------------|---------------------|
| NFR-1 | No accuracy regression vs. v1 routing | Benchmark suite: answer-quality delta ≥ 0 within noise band |
| NFR-2 | Cloud-call share decreases with repetition | On a repeated task family, cloud share at run 10 < run 1 (measured) |
| NFR-3 | Routing adds no per-model training / no drift | Router works on a fresh model with zero pretraining; thresholds adapt online |
| NFR-4 | Surgical, reviewable increments | Each phase ships as an independent PR that builds + passes tests |

## Ranking

- **Must:** FR-1, FR-2, FR-5, NFR-1
- **Should:** FR-3, FR-4, NFR-2, NFR-3
- **Could:** _(defer — see `docs/FUTURE-LEARNINGS.md`)_

## Conflicts / ambiguities to resolve

- FR-4 (reuse without escalating) vs NFR-1 (no accuracy regression): a stale or
  wrong skill trades cost for correctness. **Open:** what's the skill-invalidation
  policy? _(decide in design)_
- "Similar task" (FR-3/FR-4) needs a concrete similarity threshold to be testable.

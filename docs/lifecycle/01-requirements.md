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

## Resolved decisions (were conflicts / ambiguities)

- **FR-4 vs NFR-1 — skill-invalidation policy (RESOLVED 2026-07, counter-based):**
  A skill becomes **trusted** (usable without cloud escalation) at
  `success_count >= 2` (matches config `skills.min_success_for_trust: 2`). On
  failure, `failure_count += 1`; the skill is **flagged** (auto-use disabled,
  re-validated via cloud on next use) when `failure_count >= 2` **OR**
  `failure_ratio > 0.5` over `>= 3` uses. Skills are **never deleted**, only
  demoted (Mem0/Hermes pattern). This satisfies NFR-1: a flagged skill routes to
  cloud, so reuse never regresses accuracy. Implemented via the
  `CorrectionInvalidation` stage. See `03-design.md` → skill-invalidation record.
- **FR-3/FR-4 — "similar task" threshold (RESOLVED 2026-07):** reuse the existing
  memory-tier bars from `agent.py`. Skill match `>= 0.80`
  (`MEMORY_DIRECT_THRESHOLD`) → load and follow the skill (FR-4 reuse path).
  `0.60–0.80` (`MEMORY_CONTEXT_THRESHOLD`) → load skill as reference/plan hint,
  route steps normally. `< 0.60` → new task, fresh learning (FR-3). Makes FR-3/FR-4
  testable: a query at cosine `>= 0.80` to a stored skill resolves at local/memory
  tier with 0 cloud calls.

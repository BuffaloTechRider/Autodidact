# Autodidact v2.0 — the apprentice agent (SSOT)

**Autodidact learns to do tasks once at cloud quality, then does them again for
free.** v1.0 was a self-learning Q&A router (local↔cloud by confidence). v2.0
turns that router into a task-executing agent: it runs multi-step tasks through a
ReAct loop, routes each *step* across tiers (memory / local-fast / local-verified /
cloud), and distills every cloud escalation into a reusable **skill**. Repeat a
task family and the cloud share falls toward zero.

**This directory is the single source of truth** for what v2.0 is and what's being
built. Chat is disposable; these files are the memory. For vision *beyond* the
current build, see [`../ROADMAP.md`](../ROADMAP.md). Superseded docs live in
[`../archive/`](../archive/README.md).

## Lifecycle docs (one per phase)

Each phase reviews and refines the previous doc before moving on:

```
00-problem → 01-requirements → 02-overview → 03-design → 04-tasks → implementation → tests
```

| Doc | Phase | Verify before moving on |
|-----|-------|-------------------------|
| [00-problem.md](00-problem.md) | Problem statement | Zero solution language; adversarial check filled in |
| [01-requirements.md](01-requirements.md) | Requirements | Every requirement has an ID + testable criterion |
| [02-overview.md](02-overview.md) | Approach | Chosen approach + rejected alternatives + why |
| [03-design.md](03-design.md) | Design & architecture | Every requirement ID traces to a component |
| [04-tasks.md](04-tasks.md) | Task list | Every task traces to an FR/NFR ID + a component |

**Design deep-dives** (referenced from `03-design.md`, not duplicated):
[03a-design-full.md](03a-design-full.md) (full v2 design),
[03b-rag-pipeline.md](03b-rag-pipeline.md) (retrieval),
[03c-hallucination.md](03c-hallucination.md) (grounding/hallucination).

## Status header

- **Phase A — Routing + Tools + Backend:** ✅ largely done (routing pipeline,
  GSA, Thompson posteriors, `tools` backend param, tool registry + terminal/file-ops
  recovered). A7 baseline measurement still open.
- **Phase B — Execution loop:** 🔧 executor + step router + trajectory
  persistence/compression + prompt caching landed; task detection/rendering + the
  ≥2-tool-call integration test remain.
- **Phase C — Skill learning:** ⬜ not started.
- **Phase D — Integration + polish:** ⬜ not started.

See [04-tasks.md](04-tasks.md) for the per-task breakdown and current state.

## Working agreement

- **Implementation:** small reviewable PRs; tests-first against the acceptance
  criteria (`"write tests for FR-3, then implement until they pass"`). Each phase
  ships as an independent PR that builds + passes tests (NFR-4).
- **Testing:** `/verify` to drive the feature end-to-end; `/code-review` to check
  repo standards + spec-match back to [01-requirements.md](01-requirements.md).
- `.kiro/specs/` is **non-canonical** (the Kiro IDE may read it); these docs are
  authoritative.

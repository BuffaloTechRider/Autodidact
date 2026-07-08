# Product Lifecycle Docs

Durable, written artifacts — one per phase. Each phase reviews and refines the
previous doc before moving on; chat is disposable, these files are the memory.

```
00-problem.md  → 01-requirements.md → 02-overview.md → 03-design.md → implementation → tests
```

| Doc | Phase | Verify before moving on |
|-----|-------|-------------------------|
| [00-problem.md](00-problem.md) | Problem statement | Zero solution language; adversarial check filled in |
| [01-requirements.md](01-requirements.md) | Requirements | Every requirement has an ID + testable criterion |
| [02-overview.md](02-overview.md) | Approach | Chosen approach + rejected alternatives + why |
| [03-design.md](03-design.md) | Design & architecture | Every requirement ID traces to a component |

**Implementation:** small reviewable PRs; tests-first against the acceptance
criteria (`"write tests for FR-3, then implement until they pass"`).
**Testing:** `/verify` to drive the feature end-to-end; `/code-review` to check
repo standards + spec-match back to `01-requirements.md`.

Existing deep-dive docs (`../DESIGN-V2.md`, `../RAG-PIPELINE.md`,
`../HALLUCINATION-PROBLEM.md`, `../FUTURE-LEARNINGS.md`) are referenced from these
rather than duplicated.

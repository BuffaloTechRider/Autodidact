# Design & Architecture — Autodidact v2.0

> **Lifecycle doc 4 of 4.** Interfaces, data flow, failure modes. Review the
> **interfaces** hardest — they're expensive to change. Every requirement ID
> from [`01-requirements.md`](01-requirements.md) must trace to a component below;
> anything unmapped is scope creep or a gap.
>
> This is the lifecycle **index**. The full, living design is
> [`03a-design-full.md`](03a-design-full.md); deep-dives in
> [`03b-rag-pipeline.md`](03b-rag-pipeline.md) and
> [`03c-hallucination.md`](03c-hallucination.md). Consider the `codebase-design`
> skill when shaping a new module's seams.

## Component map (requirement → component → status)

| Component | Path | Serves | Status |
|-----------|------|--------|--------|
| Routing pipeline | `autodidact/routing/stages.py` (Memory → GSA pre-gate → LocalGeneration → CloudEscalation → CorrectionInvalidation) | FR-1, NFR-1, NFR-3 | DONE (PR #64) |
| GSA signal | `autodidact/signals/grounded_self_assessment.py` | FR-1 | DONE |
| Thompson posteriors | `autodidact/database.py` | FR-1, NFR-3 | DONE |
| Tool registry | `autodidact/tools/registry.py` (self-register, function schemas, dispatch envelope) | FR-5 | ⚠️ **NOT IN TREE** — source `.py` absent from disk and git history (stale bytecode only in `tools/__pycache__/`); restore or re-implement before relying on FR-5 |
| Terminal / file ops tools | `autodidact/tools/terminal.py`, `file_ops.py` | FR-2, FR-5 | ⚠️ **NOT IN TREE** — same as above (bytecode for `terminal`, `file_ops`, `fuzzy_match` exists, no source) |
| Backend `tools` param | `autodidact/llm/backend.py`, `ollama.py`, `openai.py` | FR-2, FR-5 | DONE (commit `dfaaa54`) |
| ReAct executor | `autodidact/executor.py` _(not built)_ | FR-2 | **TODO — Phase B** |
| Planner (task → steps) | `autodidact/planner.py` _(not built)_ | FR-2 | **TODO — Phase B/D** |
| Skill store | `autodidact/skills/store.py` _(not built)_ | FR-3, FR-4, NFR-2 | **TODO — Phase C** |
| Skill reviewer | `autodidact/skills/reviewer.py` _(not built)_ | FR-3 | **TODO — Phase C** |

## Data flow (target)

```
task ──▶ planner ──▶ [step] ──▶ executor (ReAct)
                                   │
                                   ▼
                          routing/stages.py
              Memory ─▶ GSA pre-gate ─▶ Local ─▶ Cloud
                                   │
                    cloud escalation ──▶ skills/reviewer ──▶ skills/store
                                                                  │
                                          next similar step ◀─ retrieval (memory tier)
```

## Interfaces to nail down before coding Phase B (fill in)

- `executor.step(state) -> StepResult` — what's in `state` / `StepResult`?
- Where does `routing/stages.py` splice into each ReAct turn? (memory note:
  splice the routing hook at each turn, reuse Hermes loop skeleton.)
- `skills/store.py`: `find(query) -> Skill?`, `record(outcome)`, `invalidate(...)`
  — resolves the FR-4 vs NFR-1 conflict from requirements (skill-invalidation policy).

## Component decision records

### Component: skills/store (Phase C)
**Interface:** `find(query) -> Skill?`, `record(outcome)`, `invalidate(id)`
**Options considered:**
- SQLite + structured rows — semantic-queryable, versionable, dedupable · schema migration cost
- Flat Markdown files (Hermes-style) — dead simple, git-diffable · no semantic query, manual dedup
**Chosen:** SQLite + structured skills, compact index + on-demand view.
**Why:** FR-4 requires retrieval by semantic match; Markdown-only can't satisfy it.
Borrows Hermes' compact-index idea without its Markdown-only storage.
**Match thresholds (RESOLVED 2026-07):** `find(query)` uses the `agent.py`
memory-tier bars — `>= 0.80` follow skill (FR-4 reuse), `0.60–0.80` load as
reference hint only, `< 0.60` treat as new task (FR-3). Reuses
`MEMORY_DIRECT_THRESHOLD` / `MEMORY_CONTEXT_THRESHOLD`.
**Serves:** FR-3, FR-4, NFR-2

### Component: skill-invalidation policy (resolves FR-4 vs NFR-1)
**The conflict:** reuse a stored skill (FR-4, cheaper) vs. no accuracy regression
(NFR-1) — a stale skill trades cost for correctness.
**Options considered:**
- Invalidate on first failure — simple, safe · one flaky run nukes a good skill
- Success/failure counter + threshold — resilient to noise · needs tuning, slower to retire bad skills
**Chosen (RESOLVED 2026-07):** counter-based, via the `CorrectionInvalidation` stage.
- **Trust** (auto-use without cloud) at `success_count >= 2` — reuses config
  `skills.min_success_for_trust: 2`.
- **Flag** (disable auto-use, re-validate via cloud on next use) when
  `failure_count >= 2` **OR** `failure_ratio > 0.5` over `>= 3` uses.
- **Never delete** — only demote (Mem0/Hermes pattern). A flagged skill routes to
  cloud, so NFR-1 holds.
**Why:** matches the existing routing stage and the memory-tier vocabulary; the
double-failure / ratio guard stops one flaky run from retiring a good skill while
retiring a genuinely broken one within a few uses. Thresholds may be re-tuned in
Phase C against real reuse data, but Phase C is **no longer blocked** on this.
**Serves:** FR-4, NFR-1

### Component: executor loop (Phase B)
**Interface:** `executor.step(state) -> StepResult` — `state`/`StepResult` shape still open.
**Options considered:**
- Reuse Hermes loop skeleton + trajectory compression, splice `routing/stages.py` per turn — proven shape, less to invent · must adapt Hermes' state model to ours
- Bespoke loop — fits our routing exactly · reinvents trajectory handling
**Chosen:** Hermes skeleton + our routing hook at each turn.
**Why:** memory `hermes_reference_baseline.md` — borrow proven patterns or beat
them with justification; the loop is not where our novelty lives.
**Serves:** FR-2

## Failure modes to design for

- Stale/wrong skill reused → need invalidation on failure (CorrectionInvalidation).
- Tool call errors → dispatch envelope `{ok, result|error}` already standardizes this.
- Local model hallucinates confidently → GSA + self-consistency gate (built).

## Build plan

Phase A — **PARTIAL**: routing pipeline (`routing/stages.py`) and backend `tools`
param (commit `dfaaa54`) are DONE, but the tool registry and terminal/file-ops
tools are **not in the tree** (see status table). Phase B (executor + planner)
depends on FR-5 tools; restore or re-implement `autodidact/tools/` before starting
it. Then Phase C (skills), Phase D (CLI `do`/`skills`, end-to-end demo). Ship each
as an independent PR (NFR-4).

# Tasks — Autodidact v2.0 (Apprentice Agent)

> **Lifecycle doc 5 of 5 — the single task list.** Every task traces to a
> requirement ID in [`01-requirements.md`](01-requirements.md) and a component in
> [`03-design.md`](03-design.md). This replaces the old `.kiro/specs/*/tasks.md`
> files (archived/non-canonical) and the `.scratch/` issue convention for v2.0 work.
> Prev: [`03-design.md`](03-design.md).
>
> **Status legend:** ✅ done · 🔧 in progress · ⬜ todo · ⚠️ regressed / needs recovery

---

## Phase A — Routing + Tools + Backend (foundation)

| # | Task | Serves | Status | Notes |
|---|------|--------|--------|-------|
| A1 | Routing pipeline (Memory → GSA pre-gate → LocalGeneration → CloudEscalation → CorrectionInvalidation) | FR-1, NFR-1, NFR-3 | ✅ | `autodidact/routing/stages.py` |
| A2 | GSA signal | FR-1 | ✅ | `autodidact/signals/grounded_self_assessment.py` |
| A3 | Per-category Thompson posteriors | FR-1, NFR-3 | ✅ | `autodidact/database.py` |
| A4 | Backend `tools` parameter (Ollama + OpenAI) | FR-2, FR-5 | ✅ | commit `dfaaa54`; `autodidact/llm/{backend,ollama,openai}.py` |
| A5 | Tool registry (self-register, function schemas, dispatch envelope) | FR-5 | ✅ | Recovered. `autodidact/tools/registry.py`; tests in `tests/test_tools.py`. |
| A6 | Terminal + file-ops tools | FR-2, FR-5 | ✅ | `autodidact/tools/{terminal,file_ops,fuzzy_match}.py`; tests in `tests/test_tools.py`, `tests/test_fuzzy_match.py`. |
| A7 | NFR-2 baseline: log cloud/local/memory split on ~50 real multi-step tasks | NFR-2 | ⬜ | The adversarial-check measurement from `00-problem.md`. Do before Phase B to confirm ROI. |

## Phase B — Execution Loop

| # | Task | Serves | Status | Notes |
|---|------|--------|--------|-------|
| B1 | Executor interface (`ExecutionResult` shape + `execute()`) | FR-2 | ✅ | `autodidact/executor.py`; tests in `tests/test_executor.py`. `ExecutionResult` mirrors the DESIGN-V2 contract. |
| B2 | `executor.py` — ReAct loop with step-level routing | FR-2 | ✅ | Tiered loop reuses `routing/step_router.py` (`StepRouter`) per turn; commits `5c01b71`, `8d2adde`. |
| B3 | `planner.py` — task → steps decomposition | FR-2 | ⬜ | **Descoped.** `run(plan=True)` does this as one cloud pre-pass injected as context (no subsystem). A standalone `planner.py` is only needed if skill-guided planning (D2) wants structured steps. Self-decompose (`plan=False`) is the default, matching Hermes/Claude Code. |
| B4 | Execution trace recording | FR-2, FR-3 | 🔧 | Landed as `trajectory_store.py` (`StepRecord`/`TrajectoryStore`, resume) + `trajectory_compress.py`, **not** the `execution_traces` table named here. Reconcile schema vs. skill reviewer needs (Phase C). |
| B5 | Executor as single front door in `agent.py` | FR-2 | ✅ | **Revised from "task-vs-Q&A detection" — no classifier.** `Agent.run(task, plan=)` drives the executor; Q&A finishes turn 1 with no tool call, so it subsumes questions without a heuristic (matches Hermes/Claude Code). Task-level learning-from-escalation wired post-hoc (reuses `_learn`). `query()`/`chat()` kept intact per NFR-1 (staged; shim conversion is a fast-follow). Tests: `TestRunFrontDoor` in `tests/test_agent.py`. **Two known gaps, deliberately deferred:** (a) step router is `FixedThresholdRouter` (non-learning) — Thompson per-step posteriors are a follow-up; (b) `run()` only escalates low-confidence *tool calls*, not low-confidence *text* answers, so pure-Q&A confidence escalation still belongs to `query()`. |
| B6 | Execution-mode rendering in `thought_renderer.py` ([STEP]/[SKILL]/[ESCALATING]) | FR-2 | 🔧 | Minimal progress printing wired in the `do` CLI command (`tool_result`/`cloud_call`/`memory_hit`). Full `[STEP]/[SKILL]/[ESCALATING]` renderer in `thought_renderer.py` still todo. |

## Phase C — Skill Learning

| # | Task | Serves | Status | Notes |
|---|------|--------|--------|-------|
| C1 | `skills/store.py` — CRUD + semantic search + lifecycle counters | FR-3, FR-4, NFR-2 | ⬜ | Match thresholds resolved: ≥0.80 follow / 0.60–0.80 hint / <0.60 new (see `03-design.md`). |
| C2 | `skills/loader.py` — compact index in prompt, on-demand `skill_view` | FR-3, FR-4 | ⬜ | |
| C3 | Skill-invalidation via `CorrectionInvalidation` (trust≥2 / flag≥2 fails or >50% ratio / never delete) | FR-4, NFR-1 | ⬜ | Policy resolved 2026-07 — no longer a blocker. |
| C4 | Extend `learning_extractor.py` — extract procedures/tool patterns from cloud tool calls | FR-3 | ⬜ | Fact extraction already exists. |
| C5 | `skills/reviewer.py` — post-execution review (patch > create) | FR-3 | ⬜ | Synchronous in v2.0 (simplicity). |
| C6 | Skills + execution_traces schema migrations | FR-3 | ⬜ | `database.py`. |

## Phase D — Integration + Polish

| # | Task | Serves | Status | Notes |
|---|------|--------|--------|-------|
| D1 | CLI: `autodidact do`, `autodidact skills {list,view,search}` | FR-2, FR-3 | 🔧 | `autodidact do "<task>" [--plan]` landed (wired to `Agent.run`). `skills` sub-commands await Phase C. |
| D2 | Skill-guided execution: planner structures steps from loaded skill | FR-2, FR-4 | ⬜ | |
| D3 | End-to-end demo: "learn once, execute free forever" (day1 escalate → day2 skill → day3 $0) | FR-3, FR-4, NFR-2 | ⬜ | Success criterion from `03a-design-full.md`. |
| D4 | Cost dashboard for execution mode | NFR-2 | ⬜ | |

---

## Acceptance criteria (from `01-requirements.md`)

- **FR-1:** router selects the correct tier for a step with fixed signals (unit test per tier). — A1 done.
- **FR-2:** a task with ≥2 tool calls completes end-to-end (integration test). — B2/B5 done; `test_multi_step_task_dispatches_tools` (real write_file→read_file dispatch through `run()`).
- **FR-3:** after a cloud-solved task, a skill exists and is retrievable by semantic match. — C1/C4.
- **FR-4:** second run of a similar task (cosine ≥0.80) resolves at memory/local tier, **0 cloud calls**. — C1/C3/D2.
- **FR-5:** registry lists all tools; schemas validate against OpenAI function format. — A5 (regressed).
- **NFR-1:** answer-quality delta ≥ 0 vs v1 within noise band. — regression suite each phase.
- **NFR-2:** cloud share at run 10 < run 1 on a repeated task family. — A7 baseline, then D3/D4.
- **NFR-3:** router works on a fresh model, thresholds adapt online, no per-model training. — A1/A3 done.
- **NFR-4:** each phase ships as an independent PR that builds + passes tests.

## Immediate next actions

1. **Run A7** (cloud/local/memory split baseline) to confirm the apprentice-agent ROI — the last open Phase A item and the adversarial-check measurement from `00-problem.md`.
2. **B5 follow-ups:** (a) convert `query()`/`chat()` to shims over `run()` now that the front door is proven; (b) a learning `StepRouter` (Thompson per-step posteriors) to replace `FixedThresholdRouter`; (c) escalate low-confidence *text* answers in the loop so `run()` fully subsumes `query()`'s Q&A routing.
3. **B6:** full execution-mode renderer (`[STEP]/[SKILL]/[ESCALATING]`) in `thought_renderer.py` (minimal progress printing already in `do`).
4. **B4 reconciliation:** decide whether `trajectory_store` satisfies the execution-trace requirement or a distinct `execution_traces` table is still needed for the Phase C skill reviewer.

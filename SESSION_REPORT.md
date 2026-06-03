# Autodidact — Session Report

Running log of what was done, decisions made, and implementation progress.

---

## Session: 2026-05-03

### Standing Rules (from user)
1. **Test-driven-development:** Write tests for a function/flow before implementing it.
2. **Report file:** Keep this file updated with what was done and how.

### Task: Council Debate on Two Articles

**What:** Read and debated two arxiv articles in a council-of-5 format, analyzing applicability to Autodidact.

**Articles:**
1. **Memoria** (2512.12686v1) — Modular agentic memory framework with session summarization + weighted knowledge graph
2. **"Agentic Memory is a Memo, Not True Memory"** (2604.27707v1) — Position paper arguing all current agentic memory is lookup, not learning. Proves a Generalization Gap theorem.

**How:** Convened 5 council members (Architect, Pragmatist, Researcher, Product Thinker, Skeptic) to debate 5 topics across both articles.

**Key Findings:**
- Autodidact v1.0 IS a "memo system" by Xu et al.'s taxonomy — this is fine for v1.0's scope
- Memoria's KG triplet approach is interesting for v1.1 skill extraction but adds LLM cost
- The consolidation pipeline (episodic → parametric via LoRA) is already on our Phase 4 roadmap; Xu et al. provides theoretical justification
- Memory poisoning is a real concern for v2.0 (multi-agent Hive), not v1.0 (single-user local)
- One v1.0 implementation detail: store correction history rather than overwriting when user corrects an answer

**Output:** `COUNCIL_ARTICLES_DEBATE.md` — full deliberation with votes and consensus.

**Impact on v1.0:** No scope changes. One minor implementation note for user corrections (R5.5).

---

### Status Check: What's Been Built

**92 tests passing.** All core code is implemented.

#### Completed (Tasks 1-4):

| Task | What | Status |
|------|------|--------|
| 1. Agent Class (R1) | `Agent.__init__`, `query()`, `from_config()`, `savings()`, `correct()` | ✅ Done |
| 1.3 Two-stage routing (R4) | Stage 1: KB search (sim>0.85 → memory, 0.60-0.85 → context). Stage 2: logprob → escalate if below threshold | ✅ Done |
| 1.4 Learning from escalation (R5) | Store Q&A + embeddings on cloud escalation, dedup at sim>0.95 | ✅ Done |
| 2. Visible Learning UX (R2) | ThoughtRenderer with [THINKING]/[MEMORY]/[LOCAL]/[CLOUD]/[LEARNED] tags, cost display, session summary | ✅ Done |
| 3. CLI (R3) | `init`, `chat`, `query`, `savings`, `memory stats`, `memory search` via typer | ✅ Done |
| 4. Cost Tracking (R6) | Per-query logging to query_log table, savings calculator | ✅ Done |

#### Remaining (Tasks 5-6):

| Task | What | Status |
|------|------|--------|
| 5.1 pyproject.toml | Entry point, deps, optional extras | ❌ Not done |
| 5.2 README rewrite | Learning story, 3-command quickstart, GIF | ❌ Not done |
| 5.3 PyPI setup | Package publishing | ❌ Not done |
| 6.1 Demo script | Pre-scripted interaction for recording | ❌ Not done |
| 6.2 Blog post | Launch blog post | ❌ Not done |
| 6.3 Launch | GitHub + Reddit + HN | ❌ Not done |

#### Existing Infrastructure (from v0.1 experiments):

| Module | What | Tests |
|--------|------|-------|
| `llm_client.py` | Ollama + OpenAI-compat + Bedrock, retry logic, throttle handling | 3 tests |
| `knowledge_store.py` | SQLite + FAISS, Ebbinghaus decay, STM/LTM tiers, scoped search, mixed-dim detection | 11 tests |
| `confidence_evaluator.py` | 5-signal Thompson Sampling fusion, logprob_uncertainty, energy scorer | 5 tests |
| `database.py` | SQLite WAL, 8 tables, schema migrations | (tested via integration) |
| `types.py` | Pydantic models: KnowledgeEntry, RoutingDecision, SignalScores, AgentMetrics, Config | (tested via usage) |
| `signals/grounded_self_assessment.py` | GSA v3 retrieval-conditional, 3-tier extraction fallback | 7 tests |
| `agent.py` | Core routing engine, learning, corrections, savings | 6 tests |
| `cli.py` | All 6 commands | 7 tests |
| `thought_renderer.py` | Terminal formatting for visible learning | 5 tests |
| `cost_tracking` | Query log persistence, cumulative savings | 2 tests |

**Total: 92 tests, all passing.**

### Demo Prototype Analysis (2026-05-03)

Reviewed `demo-prototype/` (lean TS demo for Vietnam AI Stars pitch) and `prototype/` (full TS vision).

**Reusable for v1.0:**
1. **LearningExtractor** — extracts structured knowledge from cloud responses via LLM (demo stores structured entries, Python stores raw Q&A). Simple: one local LLM call, extract JSON, fallback to raw.
2. **Progress callbacks** — `onProgress` pattern for real-time UI updates during query processing. Cleaner than post-hoc ThoughtRenderer.

**Reference for later (not v1.0):**
- SkillStore + SkillEvolver + SkillFormat → v1.1
- SelfVerification (periodic re-testing of stored knowledge) → v1.1
- ContextBuilder (layered L0-L3 prompts with token budgets) → v1.1
- ToolRegistry (learned API/tool registry with verification + decay) → v2.0
- UserProfile (preferences, vocabulary, conventions) → v2.0

#### What's Next

Remaining work is packaging and launch (Tasks 5-6). The core product code is complete. User requested TDD approach going forward.

---

### Implementation: LearningExtractor + Progress Callbacks (2026-05-03)

**Approach:** TDD — wrote 17 failing tests first, then implemented to green.

#### 1. LearningExtractor (`autodidact/learning_extractor.py`)

**What:** Extracts structured knowledge entries from cloud responses via a local LLM call. Ported from `demo-prototype/src/learning-extractor.ts`.

**How it works:**
- On cloud escalation, sends the Q&A to the local LLM with an extraction prompt
- LLM returns JSON with `knowledge` (facts) and `skills` (procedures)
- 3-tier JSON parsing: direct parse → strip markdown code block → regex extract `{...}`
- Fallback: if extraction fails (bad JSON, LLM exception, empty result), stores raw answer as single entry with lower confidence (0.7 vs 0.9)
- Content truncated to 500 chars to prevent KB bloat

**Integration into Agent._learn():**
- `_learn()` now returns `(learned: bool, count: int)` instead of just `bool`
- Uses LearningExtractor to get structured entries, then stores each one with embeddings
- First entry gets `verbatim_response` (full cloud answer), subsequent entries don't (saves space)
- Deduplication still works: checks for sim > 0.95 before inserting

**Tests (9):** `tests/test_learning_extractor.py`
- Successful extraction: valid JSON → multiple knowledge entries, skills with steps
- JSON resilience: markdown code blocks, JSON embedded in text
- Fallback: invalid JSON, LLM exception, empty knowledge array
- Content limits: truncation at 500 chars
- ExtractionResult structure

#### 2. Progress Callbacks (`on_progress` in Agent.query())

**What:** Real-time event emission during query processing. Ported from `demo-prototype/src/agent.ts` `ProgressCallback` pattern.

**How it works:**
- `Agent.query()` now accepts `on_progress: Callable[[dict], None] = None`
- Events are plain dicts with a `"type"` key (no new classes — matches existing style)
- Event types emitted:
  - `thinking` — memory search done, includes `memory_hits` count and `best_similarity`
  - `memory_hit` — answering from memory, includes `similarity`, `memory_source`, `age_days`
  - `local_done` — local model answered, includes `confidence`
  - `cloud_call` — escalating to cloud, includes `model`
  - `cloud_done` — cloud response received, includes `cost`, `model`, `latency_ms`
  - `learning` — knowledge stored, includes `knowledge_count`
- Backward compatible: `on_progress=None` (default) means no callbacks, existing code unchanged

**Tests (8):** `tests/test_progress_callbacks.py`
- Signature: accepts kwarg, works without it
- Local route: thinking → local_done with confidence
- Cloud route: thinking → cloud_call → cloud_done (with cost) → learning (with count)
- Memory route: thinking → memory_hit

**Test results:** 110 passed, 0 failed (92 existing + 17 new + 1 pre-existing pass).

---

### Full Vision Spec Created (2026-05-03)

**What:** Created `.kiro/specs/autodidact-full/` — the master spec covering the entire roadmap from v1.0 through Phase 4.

**Why:** The `autodidact-full` folder existed but was empty. This spec ensures nothing gets lost across sessions. Each phase has its own detailed spec when it's time to build; this document is the index and the contract.

**Contents:**
- `requirements.md` — Full requirements for all phases:
  - Phase 1 v1.0 (current, shipping) — references existing product-v1 spec
  - Phase 1 v1.1 (planned) — 9 requirement groups: tiered memory, contradiction detection, skill extraction, GSA pre-filter, self-verification, context builder, integrations, multi-turn memory, document ingestion
  - Phase 2 (Hive) — agent registry, skill transfer, knowledge tokens, reputation, security
  - Phase 3 (Hierarchical) — tiered agents, N-tier routing
  - Phase 4 (Self-Improvement) — consolidation pipeline, safety, online learning, harness evolution, CGT benchmark
  - Cross-cutting: security per phase, honest terminology, article learnings, prototype reference table
- `tasks.md` — Master task list with 27 top-level tasks across all phases, linking to detailed specs

**Sources synthesized:** VISION.md, ROADMAP.md, CONTEXT.md, COUNCIL_ARTICLES_DEBATE.md, COMPASS.md, prototype/ components, demo-prototype/ components

---

### Council Debate: MemPalace Framework (2026-05-03)

**What:** Council of 5 analyzed [MemPalace](https://github.com/mempalace/mempalace) — an open-source AI memory system with 96.6% R@5 on LongMemEval, structured palace hierarchy, temporal knowledge graph, 29 MCP tools, and agent diaries.

**Output:** `COUNCIL_MEMPALACE_DEBATE.md`

**Key findings:**

1. **Architecturally aligned** — MemPalace's wings/rooms/halls map directly to our domain/topic/category. Their temporal validity (valid_from/valid_to) matches ours. We built the same abstractions independently.

2. **Different products** — MemPalace is passive memory (store and retrieve conversations). Autodidact is active learning (route, escalate, learn, improve). They don't have confidence routing, cloud escalation, learning from escalations, or cost tracking.

3. **Temporal KG is a clean reference** — Their entities + triples tables with temporal validity is the right design for our v1.1 triple extraction (extends LearningExtractor).

4. **MCP-first distribution** — 29 MCP tools as one integration surface vs our planned LangChain + proxy + plugin. Worth studying for v1.1.

5. **Benchmarking discipline** — Committed per-question results, reproducible commands, honest about limitations. Model for our v1.0 launch.

**Actionable items:**

| Item | Timeframe |
|------|-----------|
| Wire up scoped search (domain/topic filtering) in agent query flow | v1.1 |
| Use MemPalace's KG schema as reference for triple extraction | v1.1 |
| Agent diary for self-verification logging and session summaries | v1.1 |
| MCP server (5-8 tools) as primary integration surface | v1.1 |
| Add retrieval precision measurement to quality benchmark | v1.0 |
| Cross-wing tunnels concept for Hive agent discovery | v2.0 |

---

### Requirements Update: Zero-Friction Setup + Cloud-to-Cloud Routing (2026-05-03)

**What changed in v1.0 requirements:**

1. **R3 (CLI):** `autodidact init` now references R8 zero-friction wizard instead of "simple config, no auto-install"

2. **R7 (Multi-Provider):** Added AC5 — cloud-to-cloud routing mode. Cheap cloud model fills the "local" slot, expensive cloud fills the "cloud" slot. No Ollama required. Logprob_uncertainty on cheap model's output decides escalation.

3. **R8 (Config) → renamed to "Zero-Friction Setup":** Complete rewrite:
   - Three setup modes: local+cloud (default), cloud+cloud, local-only
   - Ollama auto-detection: `which ollama`, offer install command per platform
   - Model auto-detection: `ollama list`, auto-pull if missing
   - Cloud provider presets: OpenAI, OpenRouter, DeepSeek, Bedrock with pre-filled URLs and model lists
   - Cloud-to-cloud setup: pick cheap + expensive providers/models

4. **Deferred list:** Removed "Cloud-to-cloud routing" from v1.1 deferred list (now v1.0)

5. **Success criteria:** Updated "Setup to first query in under 3 minutes" to "regardless of starting point (no Ollama, no models, no config)" + added "Cloud-to-cloud mode works for users without GPU"

**Files updated:**
- `.kiro/specs/autodidact-product-v1/requirements.md` — R3, R7, R8, deferred list, success criteria
- `.kiro/specs/autodidact-product-v1/tasks.md` — Task 5.2 added (setup wizard)
- `.kiro/specs/autodidact-full/requirements.md` — v1.0 ships list, v1.1 integrations list
- `.kiro/specs/autodidact-full/tasks.md` — Task 5 description updated

**Implementation status:** Tests written (`tests/test_setup_wizard.py`, 14 tests), setup wizard module created (`autodidact/setup_wizard.py`, 14/14 passing). CLI `init` command not yet rewritten to use the wizard. Agent not yet wired for cloud-to-cloud config loading.

---

### Requirements Update: Document Ingestion / Cold Start Fix (2026-05-03)

**Problem:** Empty-brain cold start. Users get no value from the memory system until they've asked enough questions to build up the KB.

**Council debate:** Why build basic RAG when mature frameworks exist? Council concluded: we don't need a framework. We need ~100 lines of glue code. Our existing infrastructure (LLMClient.embed, FAISS, SQLite) handles embedding + storage + retrieval. We just need file walking + text chunking.

**Decision:** Lean document ingestion for v1.0. Code intelligence via GitNexus MCP in v1.1.

**GitNexus analysis:** [GitNexus](https://github.com/abhigyanpatwari/GitNexus) builds code knowledge graphs with AST parsing (14 languages), dependency resolution, call chain tracing, clustering, and 16 MCP tools. Too much scope for v1.0 (would delay launch by months). For v1.1, `autodidact learn --code <repo>` should delegate to GitNexus MCP rather than rebuilding AST parsing. Key insight: precompute structure at index time, don't explore at query time.

**What changed:**
- **R3 (CLI):** Added AC7 — `autodidact learn <path>`
- **R9 (new):** Document Ingestion requirement with 11 acceptance criteria:
  - Walk directory, chunk text files (~500 tokens, 50 token overlap)
  - Embed via configured model, store in `document_chunks` table (separate from `knowledge_entries` per AD-002)
  - Retrieve at query time alongside agent memory, different prompt framing
  - Deduplication on re-ingestion, progress display, stats command
  - Supported: .md, .txt, .py, .ts, .js, .json, .yaml, .csv, .html, .rst, + more
  - Optional PDF via `[pdf]` extra
- **Deferred list:** Replaced `autodidact learn <file>` with `AST-aware code ingestion via GitNexus MCP`
- **v1.0 tasks:** Added Task 5.3 (document ingestion)
- **Full vision spec:** Moved document ingestion to v1.0, updated v1.1 R1.1.9 to "Code-Aware Document Ingestion"

**Files updated:**
- `.kiro/specs/autodidact-product-v1/requirements.md` — R3, R9 (new), deferred list
- `.kiro/specs/autodidact-product-v1/tasks.md` — Task 5.3 added
- `.kiro/specs/autodidact-full/requirements.md` — v1.0 ships list, v1.1 R1.1.9
- `.kiro/specs/autodidact-full/tasks.md` — Task 14 updated

---

## Catch-up: v1.0 launch → v1.0.7 (2026-05-09 → 2026-06-01)

This section reconstructs the gap between the 2026-05-03 entries above and the
current repo state. Source: git history (tags v1.0.0–v1.0.7) and per-commit
diffs. The detailed user-facing record lives in `CHANGELOG.md` — this is the
working narrative.

### Shipped: v1.0.0 (2026-05-09) — first public release

Tagged and published to PyPI. Packaging (Task 5), README, LICENSE, CHANGELOG,
and the OIDC trusted-publishing release workflow all landed. The launch
included the setup wizard, cloud-to-cloud routing, document ingestion, visible
learning UX, learning extractor, knowledge store, GSA v3, and the full CLI.

### Shipped: v1.0.1 → v1.0.7 (2026-05-13 → 2026-05-22)

Seven point releases in nine days. Highlights:

| Version | Theme |
|---------|-------|
| 1.0.1 | Streaming chat, `/learn` slash command, live Bedrock + OpenRouter discovery, Ollama auto-install, GSA `think=false`, retry-policy split, logprob dropped from chat path |
| 1.0.2 | Hybrid BM25+vector retrieval (RRF), background document synthesis, local+local mode, memory hits generate full answers |
| 1.0.3 | Custom Server wizard mode (llama.cpp/LM Studio/vLLM), CI test workflow, hardened Ollama install |
| 1.0.4 | Memory-recall answers stream like every other route |
| 1.0.5 | More refusal-detector markers, per-session savings summary, wizard default → Google |
| 1.0.6 | Tagging-only bump |
| 1.0.7 | PDF/docx made optional (lazy import), `openai` promoted to core dep |

Test suite grew from 110 → **542 passing** (+5 skipped). New test files since
the launch include: chat backends, chat slash commands (cloud/gsa/learn),
chunk sizing for BGE, document store, GSA routing gate / error scoping /
no-think, hardware detect, init sanity checks, Ollama install flow, OpenRouter
discovery, progress events, retry policy, routing pipeline, streaming agent,
and wizard list pickers.

### Refactors (PRs #68–70, late May)

- `setup_wizard.py` split from a monolith into a per-concern `setup_wizard/`
  package.
- Interactive prompts moved out of `cli.py` into the wizard package
  (`flow.py`, `ollama.py`).

### Session: 2026-06-03 — release hygiene

**Task:** Get the test suite green and fix release-process debt found on resume.

**Findings:** `test_pypi_release.py` had 1 failure + 5 errors:
1. `CHANGELOG.md` stopped at 1.0.1 — versions 1.0.2 → 1.0.7 shipped with no
   changelog entries, and the release test enforces that the current version
   (1.0.7) appears.
2. The `build_artifacts` fixture only caught `FileNotFoundError`, but
   `python -m build` with the module absent exits 1 → `CalledProcessError` →
   hard `pytest.fail` instead of skipping.

**Fixes:**
- Backfilled `CHANGELOG.md` entries for 1.0.2–1.0.7, reconstructed from git
  diffs, plus per-version `compare/` link references.
- `test_pypi_release.py`: added an upfront `importlib.util.find_spec("build")`
  check so the build-artifact tests skip cleanly when `build` isn't installed
  (CI, which has `build`, still runs the real assertions).

**Result:** 542 passed, 5 skipped, 0 failed.

**Note:** changelog dates are git tag dates; the `(#NN)` PR refs on the older
1.0.2/1.0.3 entries are inferred from commit context and not cross-checked
against GitHub.

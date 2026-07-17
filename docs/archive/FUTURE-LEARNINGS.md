# Future Learnings — Ideas for v2.1+

**Source:** Deep analysis of ATLAS, MemPalace, and DeepSeek TUI (May 2026).
These are validated patterns we deliberately defer from v2.0 to keep scope tight. Each entry includes the source project, what it does, why it matters, and how it maps to Autodidact.

---

## From ATLAS (Self-Hosted Coding Agent)

**Repository:** https://github.com/itigges22/ATLAS
**Core thesis:** Intelligence lives in the scaffolding, not the weights.

### 1. Energy-Based Confidence from Self-Embeddings

**What it is:** A small MLP (2M params) trained on the model's own 4096-dim embeddings to predict code correctness without executing it. Contrastive loss: correct code → low energy, incorrect → high energy. Val AUC 0.9467.

**Why it matters:** A learned confidence signal that complements our logprob_uncertainty. Logprobs measure token-level certainty; energy scores measure output-level quality. Together they'd be a stronger routing signal than either alone.

**How to adopt:** After v2.0 accumulates execution traces with pass/fail labels, train a small MLP on the embedding of the agent's response to predict success. Use as a secondary signal alongside logprob_uncertainty. Requires: embedding extraction from responses, labeled outcomes, periodic retraining.

**Estimated effort:** 2-3 days once we have 200+ labeled traces.

### 2. Thompson Sampling with Cost-Weighted Efficiency

**What it is:** 4 compute routes from cheapest (cost=1) to expensive (cost=1500). Beta posteriors learn which route works per difficulty level. Efficiency = p_success / cost. Bayesian bandit allocates compute optimally.

**Why it matters:** Our current threshold is static (0.7). ATLAS' approach learns the optimal threshold per task difficulty over time. Easy tasks get fast-tracked; hard tasks get full compute.

**How to adopt:** Bin tasks by estimated difficulty (heuristic or cheap classifier). Maintain Beta(α,β) per {difficulty_bin, compute_route} pair. Update on success/failure. Route selection maximizes expected_success / cost. We already have Thompson Sampling infrastructure in `confidence_evaluator.py`.

**Estimated effort:** 1 week. Mostly wiring existing infrastructure to execution traces.

### 3. Per-Task Tier Classification

**What it is:** Not every task needs the full execution pipeline. ATLAS classifies files into T1 (simple, direct write) and T2 (complex, full multi-phase pipeline) based on structural indicators.

**Why it matters:** "Fix a typo in README" shouldn't go through 20-iteration execution with skill search. A fast path for trivial tasks saves latency and compute.

**How to adopt:** Before entering the execution loop, classify task complexity:
- T0 (Q&A): No tools needed. Existing v1.0 pipeline.
- T1 (simple action): 1-2 tool calls expected. Skip skill search, don't store as skill.
- T2 (complex task): Full execution loop with skill search and learning.

Heuristic signals: query length, action verb count, number of entities mentioned, presence of multi-step language ("first... then...").

**Estimated effort:** 2-3 days.

### 4. Constraint-Driven Candidate Diversity (PlanSearch)

**What it is:** Instead of temperature sampling for diverse candidates, extract N distinct constraint sets from the problem. Each constraint eliminates ~70% of solution space. Generate one candidate per constraint set → structurally different approaches.

**Why it matters:** When a task fails, retrying with temperature diversity often produces the same mistake. Constraint-driven retry produces genuinely different approaches.

**How to adopt:** On execution failure, instead of simple retry:
1. Analyze why it failed (extract constraints from the error)
2. Generate alternative approaches that satisfy the new constraints
3. Try the most different approach next

**Estimated effort:** 1 week. Requires prompt engineering + failure analysis.

### 5. Budget Forcing with Wait Injection

**What it is:** Control thinking token allocation per step. When the model terminates reasoning too early, inject "Wait, let me reconsider." to force deeper thinking.

**Why it matters:** For hard steps, more thinking tokens = better tool call quality. For easy steps, less thinking = faster execution.

**How to adopt:** Map step confidence to thinking budget: high confidence → minimal thinking, low confidence → extended thinking (via system prompt instructions or temperature/sampling adjustments).

**Estimated effort:** 1 day. Prompt-level change.

### 6. Pattern Cache with BM25 Retrieval

**What it is:** Redis-backed cache of solved patterns indexed by BM25. Different Ebbinghaus half-lives per pattern type (API=7d, architectural=30d, idioms=90d).

**Why it matters:** We already have Ebbinghaus decay in knowledge_store.py. ATLAS validates that type-specific half-lives work better than uniform decay.

**How to adopt:** Add a `decay_class` field to knowledge entries. Map: tool_pattern=7d, procedure=30d, domain_fact=90d. Adjust stability multiplier per class.

**Estimated effort:** Half day. Schema change + minor logic adjustment.

---

## From MemPalace (Local-First AI Memory)

**Repository:** https://github.com/mempalace/mempalace
**Core thesis:** Never summarize. Raw text with good embeddings beats extraction.

### 7. Hybrid BM25 + Vector Search

**What it is:** Final score = `0.6 * vector_similarity + 0.4 * normalized_bm25_score`. Pure vector search misses keyword matches; pure BM25 misses semantic matches. The hybrid outperforms both.

**Why it matters:** Our FAISS-only search misses cases where the query uses the exact same keywords as a stored entry but the embedding doesn't quite match (happens with technical terms, API names, etc.).

**How to adopt:**
1. Add SQLite FTS5 index on `knowledge_entries.content` and `knowledge_entries.question`
2. On search: run FAISS + FTS5 in parallel
3. Re-rank combined candidates: `0.6 * cosine_sim + 0.4 * norm_bm25`
4. Return top-k from re-ranked list

**Estimated effort:** 2-3 days. FTS5 is built into SQLite.

### 8. Two-Tier Index (Closet/Drawer Pattern)

**What it is:** Compact pointer documents (closets: `topic|entities|→drawer_ids`) enable fast scanning. Full content (drawers) only retrieved when a closet matches.

**Why it matters:** As our knowledge store grows past 5,000 entries, searching all embeddings for every query gets expensive. A compact index layer pre-filters candidates.

**How to adopt:** Create a `knowledge_index` table with: topic keywords, entity names, and pointers to knowledge_entry IDs. On search, first query the index (cheap text match), then FAISS only within the candidate set.

**Estimated effort:** 3-4 days. New table + index builder + filtered FAISS search.

### 9. Temporal Knowledge Graph with Triples

**What it is:** SQLite table of `(subject, predicate, object, valid_from, valid_to)` triples. Point-in-time queries: "What was true about X on date Y?" Invalidation sets `valid_to` without deleting.

**Why it matters:** For skills and tool knowledge: "API endpoint X was at URL A until March, now it's at URL B." Temporal validity prevents serving stale information without losing history.

**How to adopt:** We already have `valid_from`/`valid_to` on knowledge_entries. Extend to a separate `knowledge_triples` table for structured relationships:
- `("deploy_script", "uses", "docker_registry_v2", valid_from="2026-05-01")`
- `("staging_url", "is", "staging.example.com", valid_from="2026-01-01")`
- Skills can reference triples for dynamic facts (e.g., "get current staging URL from KG")

**Estimated effort:** 1 week. New table, query API, integration with skill execution.

### 10. 4-Layer Context Stack

**What it is:**
- L0: Identity (~100 tokens) — always loaded
- L1: Essential context (~500-800 tokens) — auto-generated summary, always loaded
- L2: On-demand (~200-500 each) — loaded when topic/skill matches
- L3: Deep search (unlimited) — full semantic retrieval

Wake-up cost: ~600-900 tokens. 95% of context stays free.

**Why it matters:** As skills accumulate, the system prompt bloats. Layered loading keeps it lean while still having access to deep knowledge.

**How to adopt:** Map to our system:
- L0: Agent identity + routing instructions (always)
- L1: Top-5 most-used skills summary + user profile (always)
- L2: Skill index (always, but compact — name + description only)
- L3: Full skill bodies + deep KB search (on-demand via skill_view)

This is partially implemented in v2.0's skill loader (compact index + on-demand). Formalize the remaining layers.

**Estimated effort:** 2-3 days.

### 11. Write-Ahead Log for Mutations

**What it is:** Every knowledge write is logged to a WAL file before execution. Full audit trail, rollback capability.

**Why it matters:** If a background reviewer writes a bad skill or corrupts a knowledge entry, we can trace and undo it.

**How to adopt:** Create `~/.autodidact/wal.jsonl`. Before any INSERT/UPDATE/DELETE on knowledge_entries or skills tables, append a record: `{timestamp, operation, table, id, old_value, new_value}`. Periodic compaction removes entries older than 7 days.

**Estimated effort:** 1 day.

### 12. Frozen Memory Snapshot at Session Start

**What it is:** Memory is loaded once into the system prompt at session start. Never changes mid-session. Writes go to disk immediately but only take effect next session. Preserves prompt prefix cache stability.

**Why it matters:** If a provider supports prefix caching (OpenAI, DeepSeek, Anthropic), a stable system prompt means the prefix is cached after the first call. Mutating the system prompt mid-conversation invalidates the cache and increases cost.

**How to adopt:** At session start, snapshot the skill index and essential context. Lock it for the session. New skills created during execution are usable in the NEXT session (or explicitly reloaded if the user asks).

**Estimated effort:** 1 day. Mostly a discipline change in how we build system prompts.

---

## From DeepSeek TUI (Rust Coding Agent)

**Repository:** https://github.com/Hmbown/DeepSeek-TUI
**Core thesis:** Claude Code-class experience optimized for cheap/fast models.

### 13. Per-Turn Model Routing via Pre-Screening

**What it is:** Before each turn, a cheap Flash-class call evaluates the message and picks: which model + which thinking level (off/high/max). Routing decision per-turn, not per-session.

**Why it matters:** Different turns in the same execution have different difficulty. "List files in the directory" is trivial; "refactor this authentication module" is hard. Spending the same compute on both wastes money.

**How to adopt:** Before each executor iteration:
1. Quick classification call (local model, 20 tokens max): "Is this step trivial/medium/hard?"
2. Trivial → local model, no extended thinking
3. Medium → local model, standard thinking
4. Hard → escalate to cloud immediately (skip the failed local attempt)

This avoids wasting a full local generation + logprob check on steps we can predict will fail.

**Estimated effort:** 2-3 days. Adds a cheap pre-screening call.

### 14. LSP Diagnostics Injection

**What it is:** After every file edit, LSP servers (pyright, rust-analyzer, etc.) provide diagnostics. These are injected as system messages before the next API call.

**Why it matters:** The agent shouldn't have to run the code to discover syntax errors or type mismatches. Instant feedback from the language server catches problems before they compound.

**How to adopt:**
1. Detect project language from file extensions
2. If LSP is available (pyright for Python, tsc for TypeScript), run it after file edits
3. Inject diagnostics into the next executor iteration as a tool result: `{"diagnostics": [{"file": "x.py", "line": 42, "message": "..."}]}`

**Estimated effort:** 3-5 days. Needs LSP client integration (or just run CLI linters: `pyright --outputjson`, `tsc --noEmit`).

### 15. Side-Git Workspace Snapshots

**What it is:** A separate `.git` directory that never touches the user's repo takes per-turn snapshots. Enables undo without affecting user's git history.

**Why it matters:** When the agent makes a bad edit during execution, the user needs to undo it cleanly. If the agent is using the user's git repo, it creates noise in their history.

**How to adopt:** Before execution starts:
1. Create a snapshot branch: `git stash create` or copy working tree state
2. After each file write, snapshot the current state
3. On failure/rollback: restore from snapshot
4. On success: clean up snapshots

Simpler alternative: just use `git stash` before execution and `git stash pop` on failure.

**Estimated effort:** 2 days.

### 16. RLM (Recursive Language Model) Pattern

**What it is:** A Python REPL with `llm_query()` helpers. When the agent needs to process something too large for context, it writes a script to chunk and process it.

**Why it matters:** Some tasks involve large files that don't fit in context (50K-line logs, large codebases). Instead of failing or truncating, the agent programs its own context management.

**How to adopt:** Add a `code_execute` tool that runs Python in a sandbox with access to `query_local(prompt)` and `query_cloud(prompt)` helper functions. The agent can write: "Read the 10K-line file in chunks, extract errors from each chunk, then summarize."

**Estimated effort:** 1 week. Needs a sandboxed Python executor with LLM access.

### 17. Capacity Controller with Risk Bands

**What it is:** Tracks context usage and applies escalating guardrails:
- Green: normal execution
- Yellow: shorter responses, prioritize finishing
- Orange: force compaction (summarize history)
- Red: split into sub-task, hand off to fresh context

**Why it matters:** Without this, long executions silently hit context limits and produce garbage. With it, the system gracefully degrades.

**How to adopt:** Track token count across iterations. At 50% capacity: warn. At 70%: summarize early iterations into a compact "what happened so far" block. At 85%: force task split or completion.

**Estimated effort:** 2-3 days.

### 18. Durable Task System with Verification Gates

**What it is:** Background tasks survive restarts. Each task has structured state, verification criteria, and progress tracking.

**Why it matters:** For long-running tasks (CI pipelines, deployment monitoring), the agent needs persistence across sessions.

**How to adopt:** `execution_traces` table already stores completed tasks. Extend with:
- `status`: pending/running/paused/complete/failed
- `checkpoint`: JSON blob of current state (which step, accumulated results)
- `resume()` method that continues from last checkpoint

**Estimated effort:** 3-5 days.

### 19. Steer Mechanism (User Correction Mid-Stream)

**What it is:** Users can inject follow-up messages while the model is mid-execution. The injected message appears in the next API call.

**Why it matters:** If the user sees the agent going in the wrong direction during a 10-step execution, they should be able to say "no, use the staging database not prod" without waiting for completion.

**How to adopt:** In the execution loop, check for user input between iterations. If present, inject it as a user message in the conversation before the next LLM call. Requires async input handling in the CLI.

**Estimated effort:** 2-3 days.

---

## From Karpathy's LLM Wiki (2026)

**Source:** https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f
**Core thesis:** LLMs should incrementally build and maintain persistent knowledge artifacts, not re-derive from raw chunks on every query. The wiki is a "persistent, compounding artifact."

### 20. Knowledge Lint Operation

**What it is:** A periodic health-check pass over the knowledge store and skill store. The LLM scans for contradictions between entries, stale facts superseded by newer sources, orphan entries that never get retrieved, skills that conflict with each other, and gaps where important concepts are mentioned but not stored.

**Why it matters:** As the knowledge store and skill store grow (especially with document synthesis adding entries at ingest time), inconsistencies will accumulate. Two documents may state contradictory facts. A skill may reference an API endpoint that changed. Without active maintenance, the knowledge base rots — the agent gives stale or conflicting answers.

**How to adopt:**
1. Add a `lint` CLI command: `autodidact lint`
2. Implementation: load all knowledge entries + skills, group by topic/embedding cluster
3. For each cluster, prompt the local model: "Do any of these entries contradict each other? Are any clearly outdated?"
4. Output: a report of issues found, with suggested actions (merge, archive, flag for user review)
5. Optionally auto-fix low-risk issues (e.g., archive entries with 0 retrieval hits in 30 days)

**Relationship to other patterns:** This subsumes the Hermes "Curator" pattern (periodic skill consolidation/archival) and adds contradiction detection. Karpathy validates independently.

**Estimated effort:** 3-4 days. Mostly prompt engineering + a scan loop.

### 21. Cross-Referencing Between Knowledge Entries

**What it is:** Knowledge entries and skills can reference each other, forming a lightweight knowledge graph. Skills link to prerequisite skills ("deploy_to_staging" requires "build_docker_image"). Knowledge entries link to related entries ("PTO policy" links to "HR contact info"). The agent can follow links to gather richer context.

**Why it matters:** Currently entries are isolated atoms — the agent finds them individually by embedding similarity. But knowledge is relational: understanding "how to deploy" requires understanding "how to build" first. Cross-references let the agent pull in prerequisite context automatically when loading a skill, and discover related information it wouldn't have found by embedding search alone.

**How to adopt:**
1. Add a `related_ids` field to knowledge entries and a `prerequisite_skills` field to skills
2. During the skill reviewer's post-execution pass, detect when one skill's steps depend on another skill's output
3. During document synthesis, detect when extracted facts reference concepts already in the store — link them
4. At retrieval time: when a knowledge entry or skill is loaded, optionally follow 1 hop of links to gather related context
5. The lint operation (item #20) can detect missing links: "these two entries are about the same topic but don't reference each other"

**Relationship to other patterns:** This is a simplified version of MemPalace's temporal knowledge graph (#9 in this doc), without the full `(subject, predicate, object, valid_from, valid_to)` triple structure. Start with simple bidirectional links; evolve to triples if the use case demands it.

**Estimated effort:** 3-5 days. Schema change + link creation in reviewer + optional link-following at retrieval.

---

## Priority Ranking for v2.1

Based on impact/effort ratio:

| Priority | Feature | Source | Effort | Impact |
|----------|---------|--------|--------|--------|
| 1 | Knowledge lint operation | Karpathy/Hermes | 3-4d | High — prevents knowledge rot |
| 2 | Hybrid BM25+vector search | MemPalace | 2-3d | High — improves all retrieval |
| 3 | Cross-referencing between entries | Karpathy/MemPalace | 3-5d | High — relational context |
| 4 | Per-task tier classification | ATLAS | 2-3d | High — avoids over-engineering simple tasks |
| 5 | Frozen memory snapshot | MemPalace | 1d | Medium — prefix cache savings |
| 6 | Write-ahead log | MemPalace | 1d | Medium — safety + auditability |
| 7 | Capacity controller | DeepSeek | 2-3d | High — prevents context overflow |
| 8 | Cost-weighted Thompson Sampling | ATLAS | 1w | High — learns optimal routing |
| 9 | LSP diagnostics injection | DeepSeek | 3-5d | Medium — faster error detection |
| 10 | 4-layer context stack | MemPalace | 2-3d | Medium — prompt efficiency |
| 11 | Steer mechanism | DeepSeek | 2-3d | Medium — better UX |
| 12 | Type-specific decay classes | ATLAS | 0.5d | Low — fine-tuning existing feature |
| 13 | Per-turn pre-screening | DeepSeek | 2-3d | Medium — reduce wasted local attempts |
| 14 | Side-git snapshots | DeepSeek | 2d | Medium — safer file edits |
| 15 | Energy-based confidence | ATLAS | 3d + data | Medium — needs labeled data first |
| 16 | Two-tier index | MemPalace | 3-4d | Medium — matters at 5K+ entries |
| 17 | RLM recursive processing | DeepSeek | 1w | Low — niche use case |
| 18 | Temporal knowledge graph | MemPalace | 1w | Low — subsumed by #3 initially |
| 19 | PlanSearch diversity | ATLAS | 1w | Low — matters for repeated failures |
| 20 | Durable tasks | DeepSeek | 3-5d | Low — matters for long-running ops |
| 21 | Budget forcing | ATLAS | 1d | Low — model-specific optimization |

---

## Cross-Cutting Patterns (Validated by Multiple Projects)

These patterns appear independently in 2+ of the analyzed projects, suggesting they're fundamental rather than project-specific:

1. **Ebbinghaus decay for knowledge** — ATLAS (pattern cache), Autodidact v1 (knowledge store)
2. **Compact index + on-demand loading** — Hermes (skills), MemPalace (closets), DeepSeek (skills), Karpathy (index.md)
3. **Active maintenance/lint** — Hermes (Curator), Karpathy (lint operation), MemPalace (WAL + consistency checks)
4. **Background self-improvement** — Hermes (review fork), ATLAS (online retraining)
5. **Hybrid search (vector + keyword)** — MemPalace (BM25+vector), ATLAS (BM25 patterns)
6. **Frozen context at session start** — Hermes (memory snapshot), DeepSeek (system prompt), Karpathy (frozen wiki at session start)
7. **Thompson/Bayesian routing** — ATLAS (Beta posteriors), Autodidact v1 (Thompson fusion)
8. **Provenance tracking** — Hermes (skill source), MemPalace (WAL), ATLAS (replay buffer)
9. **Tool result type coercion** — Hermes (fix LLM type errors), universal good practice
10. **Synthesize-once, retrieve-many** — Karpathy (wiki pages compiled from raw sources), Autodidact v2.0 (document synthesis on ingest)

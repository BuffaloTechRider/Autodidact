# Autodidact v2.0 — Full Design (deep-dive appendix)

**The Apprentice Agent: An AI That Learns to Do, Not Just Answer**

> ℹ️ **Design appendix `03a` of the lifecycle SSOT.** This is the full v2.0
> (apprentice-agent) design narrative. The traceable index —
> requirement→component→path→**status** — is [`03-design.md`](03-design.md); trust
> it over any path named here. Some module paths below are aspirational and differ
> from what was actually built:
> - Routing lives in `autodidact/routing/stages.py`, **not** `router.py`.
> - `knowledge_store.py` is still in use; `knowledge_memory.py` was not created.
> - `executor.py`, `planner.py`, and `skills/` are **not yet built** (Phases B/C).
> - The tool registry and `terminal.py`/`file_ops.py` are **not in the tree**
>   (only stale bytecode remains); the backend `tools` param IS done (`dfaaa54`).
>
> **Scope:** "v2.0" in this repo means the **apprentice agent** (this doc). The
> Hive network, tiered hierarchy, and LoRA consolidation are later phases, tracked
> as vision-only in [`../ROADMAP.md`](../ROADMAP.md) — not part of this SSOT.

---

## Context

### What v1.0 Is (Implemented)

Autodidact v1.0 is a Q&A routing agent. It answers questions by routing between cheap (local) and expensive (cloud) models based on logprob confidence, learns from cloud escalations by storing Q&A pairs, and gets cheaper over time as its knowledge store grows.

**v1.0 components (4,400 LOC):**

| Module | What it does |
|--------|-------------|
| `agent.py` | Query routing: memory check → local generation → cloud escalation → learn |
| `llm_client.py` | Ollama + OpenAI-compatible + Bedrock backends with retry |
| `knowledge_store.py` | SQLite + FAISS, Ebbinghaus decay, STM/LTM tiers |
| `confidence_evaluator.py` | Thompson Sampling fusion of 5 signals (logprob dominant) |
| `learning_extractor.py` | Extracts structured facts from cloud responses |
| `document_store.py` | Document ingestion for cold start (chunk + embed + store) |
| `signals/grounded_self_assessment.py` | Pre-generation Y/N confidence probe |
| `thought_renderer.py` | Visible learning UX with [THINKING]/[MEMORY]/[LOCAL]/[CLOUD] tags |
| `setup_wizard.py` | Ollama detection, model pulling, cloud provider presets |
| `cli.py` | `init`, `chat`, `query`, `savings`, `memory` commands |
| `types.py` | Pydantic domain models |
| `database.py` | SQLite schema with migrations |

**v1.0 can:**
- Answer questions from memory (semantic retrieval)
- Route by confidence (logprob_uncertainty, AUROC 0.65-0.83)
- Learn facts from cloud escalations
- Track cost savings
- Ingest documents for cold start

**v1.0 cannot:**
- Execute tasks (tool calling, shell commands, file operations)
- Learn procedures (multi-step skills)
- Discover and learn tool interfaces
- Self-improve its skills after execution
- Act as an agent (plan, execute, verify)

### What v2.0 Is

v2.0 transforms Autodidact from a Q&A router into a **learning agent** — an apprentice that starts knowing nothing, acquires skills from cloud escalation, and gets increasingly autonomous over time.

The core differentiator: **confidence-based routing at the step level during task execution.** The agent tries each step locally, escalates individual steps it's uncertain about, and stores the cloud's approach as a reusable skill for next time.

### Inspiration Sources

| Source | What we take | What we leave |
|--------|-------------|---------------|
| **Hermes Agent** | Self-registering tool registry, background skill review, curator pattern, skill lifecycle tracking, frozen memory snapshot, compact index + on-demand loading | 12K-LOC monolith, Markdown-only skills without execution guarantees, tiny bounded memory (3.5KB), no routing |
| **ATLAS** | Thompson Sampling with cost weighting, per-task tier classification, Ebbinghaus-decay pattern cache, energy-based scoring | Single-model-only, coding-specific pipeline, Docker microservice complexity |
| **MemPalace** | Temporal knowledge graph, hybrid BM25+vector search, 4-layer context stack, write-ahead log, closet/drawer index pattern | ChromaDB fragility, fixed chunk size, regex-only entity extraction |
| **DeepSeek TUI** | Per-turn model routing via pre-screening, LSP diagnostics injection, durable task system, capacity controller | 190K LOC Rust, DeepSeek-specific features, minimal memory |

---

## Architecture

### Layer Diagram

```
┌──────────────────────────────────────────────────────────────────────┐
│                         User Request                                  │
│  "Deploy to staging" / "What's our PTO policy?" / "Fix the CI"       │
└────────────────────────────────┬─────────────────────────────────────┘
                                 ▼
┌──────────────────────────────────────────────────────────────────────┐
│                   PARALLEL RETRIEVAL (all tiers at once)              │
│                                                                      │
│  embed query ONCE → search simultaneously:                           │
│    Tier 1+2: Knowledge pages (FAISS summaries + FTS5 content)        │
│    Tier 3:   Raw doc chunks  (FAISS chunks + FTS5 BM25+RRF)          │
│                                                                      │
│  Merge → assemble context (pages first, chunks fill budget)          │
│  Output: context + knowledge_similarity score                        │
└────────────────────────────────┬─────────────────────────────────────┘
                                 ▼
┌──────────────────────────────────────────────────────────────────────┐
│                        AGENT DISPATCHER                               │
│  Classify: is this a question (Q&A) or a task (needs execution)?     │
│  Q&A → Q&A pipeline                                                  │
│  Task → execution loop                                               │
└──────────┬─────────────────────────────────────────────┬─────────────┘
           │                                             │
           ▼                                             ▼
┌────────────────────────┐            ┌─────────────────────────────────┐
│    Q&A PIPELINE (v2)   │            │        EXECUTION LOOP (v2)      │
│                        │            │                                 │
│  1. Direct memory hit? │            │  1. Search skill pages          │
│     → return ($0.00)   │            │  2. Plan steps (from skill      │
│  2. GSA pre-screen     │            │     or LLM-generated)           │
│     (uses context)     │            │  3. For each step:              │
│  3. Generate locally   │            │     a. Try locally (tiered)     │
│  4. Tiered routing     │            │     b. If uncertain → verify    │
│     (logprob + verify) │            │     c. If fails → escalate      │
│  5. Cloud if uncertain │            │     d. Execute tool call         │
│  6. Learn → integrate  │            │     e. Verify result             │
│     into topic page    │            │  4. After completion → learn     │
│  7. Update Thompson    │            │     → integrate into pages       │
│                        │            │                                 │
└────────────────────────┘            └──────────────────┬──────────────┘
                                                         │
                                                         ▼
┌──────────────────────────────────────────────────────────────────────┐
│                     BACKGROUND REVIEW                                 │
│  Post-execution analysis (async, doesn't block user):                │
│  - New procedure discovered? → Create/update skill page              │
│  - Cloud showed a new technique? → Integrate into knowledge page     │
│  - Skill failed at a step? → Patch the skill page                   │
│  - Facts learned along the way? → Integrate into topic pages         │
└──────────────────────────────────────────────────────────────────────┘
```

### Knowledge Storage: Topic Pages + Chunk-Based Search

Knowledge is organized into **topic pages** (compiled markdown in SQLite), not flat entries. Each page is a self-contained document on one topic, continuously updated as the agent learns. Search operates on **page chunks** (512-token segments), not summaries.

**Key principle:** LLM is used for WRITING (compiling knowledge). LLM is NEVER used for INDEXING (chunking + embedding is deterministic). This keeps search reliable, fast, and free of hallucination.

```sql
-- The compiled knowledge pages (source of truth)
CREATE TABLE knowledge_pages (
    id            TEXT PRIMARY KEY,
    topic         TEXT NOT NULL UNIQUE,     -- "deployment", "python-async", "auth-api"
    category      TEXT NOT NULL,            -- concept/procedure/entity/domain
    status        TEXT NOT NULL DEFAULT 'active',  -- active/core/archived
    content       TEXT NOT NULL,            -- full compiled markdown
    sources       TEXT NOT NULL DEFAULT '[]',
    entry_count   INTEGER NOT NULL DEFAULT 0,
    access_count  INTEGER NOT NULL DEFAULT 0,
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL,
    accessed_at   TEXT NOT NULL,
    stability     REAL NOT NULL DEFAULT 1.0  -- Ebbinghaus: grows with access
);

-- Page chunks for vector search (re-generated on every page update)
CREATE TABLE knowledge_page_chunks (
    id            TEXT PRIMARY KEY,
    page_id       TEXT NOT NULL REFERENCES knowledge_pages(id) ON DELETE CASCADE,
    chunk_index   INTEGER NOT NULL,
    content       TEXT NOT NULL,            -- 512-token section-aware segment
    embedding     BLOB NOT NULL,           -- USearch/FAISS vector
    UNIQUE(page_id, chunk_index)
);

-- FTS5 on full page content for keyword search
CREATE VIRTUAL TABLE knowledge_pages_fts USING fts5(
    topic, content,
    content='knowledge_pages',
    content_rowid='rowid',
    tokenize='porter unicode61'
);
```

**Why chunk-based search, not summaries:**
- Summaries truncate/lose information → missed search hits
- Summaries require an LLM call on every page update → slow, fragile, can hallucinate
- Chunks cover EVERY word of the page → no information loss
- If one chunk mentions "rollback", a query about rollback finds it with high similarity
- No extra LLM dependency for indexing

**Dual chunking strategy:**

| Layer | Strategy | Why |
|-------|----------|-----|
| Tier 1 (knowledge pages) | **Section-aware:** split on `##` headings, each section = one chunk if ≤512 tokens. Large sections split on paragraphs. Heading prefix retained on every chunk for self-containedness. No overlap. | Pages are structured markdown WE compile — we control the format |
| Tier 3 (raw documents) | **Markdown-aware overlap:** split on headings first, then paragraphs, ~50 token overlap. Code blocks kept atomic. | User's files have unpredictable structure — overlap protects against bad boundaries |

The LLM compilation prompt enforces structured headings: "Use ## section headings. Each section covers ONE subtopic. Keep sections under 400 tokens." This makes pages naturally chunk-friendly.

**Write path (when knowledge arrives):**
```
New fact: "Deploy requires VPN connection"
  → Search existing page chunks → find "deployment" page
  → LLM integrates fact into page content (the compile step)
  → Re-chunk page into 512-token sections (deterministic, no LLM)
  → Re-embed each chunk (deterministic, no LLM)
  → FTS5 auto-updates via trigger
```

**Read path (when user asks a question):**
```
Query: "how do I rollback a deployment?"
  → Embed query (one call)
  → IN PARALLEL:
      Vector: search page_chunks → find relevant chunks → map to pages
      BM25: FTS5 MATCH on pages_fts → find matching pages
  → RRF fusion at PAGE level (dedupe chunks from same page)
  → Load top-K pages' full content → inject as context
  → knowledge_similarity = best chunk score (feeds routing)
```

**Vector index: USearch (replaces FAISS)**
- HNSW algorithm (approximate nearest neighbor, O(log n) search)
- <1MB install, zero native deps (vs FAISS ~10MB + BLAS requirement)
- 10x faster than FAISS at scale (>5K vectors)
- Simpler API: `index.add(id, vec)`, `index.search(vec, k)`
- Builds cleanly on all platforms (ARM Mac, Linux, Windows)

**Why pages, not flat entries:** A flat bag of 142 facts about "deployment" gives the LLM 3 random fragments. A compiled page gives the FULL picture — steps, URLs, requirements, all in one read. The LLM sees complete, organized knowledge instead of scattered atoms.

**Priority decay:** `retention = exp(-hours_since_access / stability)` where `stability = base × (1 + ln(1 + access_count))`. Frequently-used pages stay active. Unused pages fade to archive. Memory stays lean.

### Three-Tier Retrieval (parallel, not cascade)

| Tier | Storage | What's indexed | Search method |
|------|---------|----------------|---------------|
| Tier 1 (active pages) | `knowledge_pages` + `knowledge_page_chunks` | USearch: chunk embeddings. FTS5: full page content | Hybrid RRF at page level |
| Tier 2 (archived) | Archive JSONL + chunk embeddings kept in USearch | Same as Tier 1 | Same search, restore on hit |
| Tier 3 (raw docs) | `document_chunks` + `document_chunks_fts` | USearch: chunk embeddings. FTS5: chunk content | Hybrid BM25+vector RRF |

All tiers searched in ONE pass. No cascade latency. ~100ms total retrieval.

### Optional: Reranking Stage (v2.0)

After RRF produces the top-20 candidates, an optional reranker precisely scores each (query, chunk) pair to pick the best 3 for context injection. OFF in v1.5 (RRF is good enough), ON in v2.0 when routing accuracy becomes critical for tool execution.

```
RRF top-20 candidates → Reranker scores each → Top-3 by reranker score → context
```

**Reranker options (in order of preference for our stack):**

| Option | Latency | Deps | Notes |
|--------|---------|------|-------|
| Ollama-served reranker (Qwen3-Reranker 0.6B) | ~200-500ms | Ollama (already running) | Best fit — no new deps, good quality |
| Cross-encoder via sentence-transformers | ~50-200ms | sentence-transformers (~500MB) | Faster but large dependency |
| LLM-as-judge ("rate relevance 1-5") | ~2-4s | Local LLM (already have) | Too slow for every query |

**When reranking matters:** The difference between "top-3 by RRF" and "top-3 by reranker" is small for simple factual queries but significant for nuanced queries where multiple pages are partially relevant. For tool execution (v2.0), picking the wrong skill page means executing the wrong steps — reranking pays for itself there.

**Implementation:** A `rerank(query, candidates, limit)` method on KnowledgeMemory. When a reranker model is configured (via config.yaml), it runs. When not configured, it's a no-op pass-through (RRF results returned directly).

### Module Map

```
autodidact/
├── agent.py                  # EXTEND: add task detection, dispatch to executor
├── executor.py               # NEW: ReAct loop with step-level routing
├── planner.py                # NEW: task → steps decomposition using skills
├── tools/
│   ├── __init__.py
│   ├── registry.py           # NEW: self-registering tool collection
│   ├── terminal.py           # NEW: shell command execution
│   ├── file_ops.py           # NEW: read/write/search/patch files
│   ├── web.py                # NEW: web search + fetch (optional)
│   └── memory_tool.py        # NEW: expose memory ops as agent-callable tool
├── skills/
│   ├── __init__.py
│   ├── store.py              # NEW: skill CRUD + semantic retrieval
│   ├── loader.py             # NEW: compact index for prompt, on-demand loading
│   └── reviewer.py           # NEW: background self-improvement after execution
├── knowledge_memory.py       # NEW: page-based knowledge store (replaces knowledge_store.py)
│                             #   search (chunk vectors + FTS5 + RRF), integrate, decay, archive
├── knowledge_store.py        # DEPRECATE: replaced by knowledge_memory.py (kept for migration)
├── document_store.py         # EXTEND: add synthesis pass after chunking (ingest → internalize)
├── learning_extractor.py     # EXTEND: extract skills + tool patterns + document synthesis
├── router.py                 # NEW: tiered routing with Thompson Sampling per category
├── llm_client.py             # EXTEND: add tools parameter to chat calls
├── thought_renderer.py       # EXTEND: add [STEP], [SKILL], [ESCALATING] tags
├── setup_wizard.py           # EXISTS: unchanged
├── cli.py                    # EXTEND: add `do`, `skills` commands
├── types.py                  # EXTEND: add Skill, ToolCall, Step types
└── database.py               # EXTEND: add skills table
```

---

## Core Components

### 1. Tool Registry (`tools/registry.py`)

Adapted from Hermes' self-registering pattern. Each tool file registers itself at import time.

**Design decisions:**
- Tools are Python functions with a typed schema (OpenAI function-calling format)
- Auto-discovered: any `tools/*.py` file with a `register()` call is picked up
- Dispatch returns JSON strings (consistent with OpenAI tool result format)
- Type coercion on arguments (fix LLM type errors silently)
- Toolsets for grouping (enable/disable sets of tools per context)

```python
@dataclass
class ToolEntry:
    name: str
    description: str
    schema: dict          # OpenAI function-calling format
    handler: Callable[[dict], str]  # args → JSON result string
    toolset: str = "default"

class ToolRegistry:
    _tools: dict[str, ToolEntry] = {}

    @classmethod
    def register(cls, name: str, *, description: str, schema: dict,
                 handler: Callable, toolset: str = "default") -> None: ...

    def get_schemas(self, toolsets: list[str] | None = None) -> list[dict]: ...

    def dispatch(self, name: str, arguments: dict) -> str: ...
```

**Initial tools (v2.0):**

| Tool | Toolset | Purpose |
|------|---------|---------|
| `terminal` | terminal | Execute shell commands |
| `read_file` | file | Read file contents |
| `write_file` | file | Create/overwrite files |
| `edit_file` | file | Patch existing files (find/replace) |
| `search_files` | file | Regex grep across files |
| `list_directory` | file | List directory contents |
| `skill_view` | skills | Load a skill's full content |
| `memory_search` | memory | Search the knowledge store |

**Not in v2.0:** web search, browser, delegate/subagent, code execution sandbox. These are v2.1+ additions.

### 2. Tiered Routing Engine (`router.py`)

The routing system that decides where to handle each query or execution step. Replaces the v1.0 "logprob-only" approach with a multi-signal, tiered system that works with any local model and improves over time without retraining.

**Why v1.0's logprob routing fails:**
- **Hallucination with high confidence:** Made-up facts are fluent and predictable. High logprob ≠ correct. The model doesn't *know* it's wrong.
- **Refusal with high confidence:** "I don't have access to real-time information" is a highly rehearsed RLHF phrase. Token certainty is near-perfect.
- **Root cause:** Logprob measures token predictability, not answer correctness.

**Design principles:**
- No per-model pre-training required — works from day 1 with any local model
- No MLP that needs constant retraining as memory grows
- Uses signals that are model-agnostic and either stable or improve with memory growth
- Adapts per-user per-model via lightweight Thompson Sampling (6 Beta distributions, not a neural network)

#### Routing Tiers

```
Tier 0: Memory hit           — free, instant, no generation
Tier 1: Local, single-pass   — 1 generation, no verification (fast path)
Tier 2: Local, verified      — 2 generations + knowledge check (still free $)
Tier 3: Cloud escalation     — expensive, authoritative
```

#### Routing Flow

```
┌─────────────────────────────────────────────────────────────┐
│ 1. CATEGORY CLASSIFICATION (keyword heuristic, ~0 cost)     │
│    → factual / realtime / reasoning / creative / code / kb  │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. MEMORY CHECK                                             │
│    hit > 0.85 → Tier 0 (return from memory)                 │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. GSA PRE-SCREEN (1 token, knowledge-conditional)          │
│    NO / score < 0.3 → Tier 3 (escalate immediately)         │
│    YES / score > 0.3 → continue to local generation         │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. LOCAL GENERATION (with logprobs)                         │
│    Get category-specific threshold from Thompson posterior  │
│                                                             │
│    logprob > high_threshold → Tier 1 (trust, return)        │
│    logprob < low_threshold  → Tier 3 (escalate)             │
│    middle zone              → Tier 2 (verify)               │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ 5. VERIFICATION (Tier 2 only)                               │
│    a) Self-consistency: re-generate at temp=0.3, compare    │
│    b) Knowledge-verify: check response vs memory store      │
│    Both pass → return local answer                          │
│    Either fails → Tier 3 (escalate to cloud)                │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ 6. OUTCOME RECORDING (on every escalation)                  │
│    Compare local answer vs cloud answer → agree/disagree    │
│    Update Thompson posterior for this category              │
│    Optionally update energy_scorer if enough data           │
└─────────────────────────────────────────────────────────────┘
```

#### Signals Used (no training required)

| Signal | Catches | Cost | Drifts with memory? |
|--------|---------|------|---------------------|
| GSA pre-screen | Refusals, known-unknowns | 1 token | No |
| Logprob | General uncertainty | Free (already computed) | No |
| Self-consistency | Hallucinations, random errors | 2x generation | No |
| Knowledge verification | Factual hallucinations | 1 retrieval | Gets **better** |
| Category × Thompson | Per-type thresholds | ~0 (cached posteriors) | Adapts automatically |

#### Per-Category Thompson Sampling

Instead of a single global threshold (0.7), maintain a Beta(α, β) posterior per query category. Each category learns its own threshold from outcomes.

```python
categories = {
    "factual":   Beta(α=1, β=1),  # starts uniform, learns quickly
    "realtime":  Beta(α=1, β=1),
    "reasoning": Beta(α=1, β=1),
    "creative":  Beta(α=1, β=1),
    "code":      Beta(α=1, β=1),
    "personal":  Beta(α=1, β=1),  # KB-backed queries
}

# At query time:
category = classify(query)
p_local_success = categories[category].mean()  # α / (α + β)
# Adaptive threshold: high success rate → lower bar for local
high_threshold = 0.9 - (0.3 * p_local_success)  # ranges 0.6-0.9
low_threshold = 0.5 - (0.2 * p_local_success)   # ranges 0.3-0.5

# After escalation, compare local vs cloud:
if local_was_correct:
    categories[category].α += 1
else:
    categories[category].β += 1
```

**Why this works:**
- A category where local succeeds often (factual, creative) → threshold drops → fewer escalations
- A category where local fails often (realtime, hard reasoning) → threshold stays high → more escalations
- Learns per-model: a 70B model will earn lower thresholds faster than a 7B
- No retraining, no MLP, no drift — just 6 lightweight counters updated on each escalation
- Thompson exploration built in: occasionally tries the "wrong" tier to discover if performance has changed

#### GSA Prompt Design (v4 — knowledge-conditional with calibrated scale)

The v1.0 GSA prompt ("Are you confident you can answer?") is too vague. Models say YES for things they'll hallucinate about (RLHF eagerness) and NO for things they can answer (RLHF hedging).

**v4 prompt (production):**

```
Given this question: "{query}"
{optional: "Here is what you know about this topic:\n{top 2-3 memory hits}"}

Based on your training knowledge{optional: " and the information above"},
can you give a specific, factual answer (not a hedge or disclaimer)?

Respond with exactly one token: YES or NO.
```

Key design choices:
- "specific, factual answer (not a hedge or disclaimer)" — directly targets the refusal failure mode
- Knowledge-conditional: when memory has strong hits, show them. When not, bare prompt (indistinguishable from "retrieval never happened" — same v3 principle from EXP-005)
- Extract from logprob distribution over YES/NO, not just the generated token

**v4-scale prompt (research/experimentation):**

```
Given this question: "{query}"

How likely is it that you can give a correct, specific answer?
1 = Would definitely need to guess or refuse
2 = Probably can't answer reliably
3 = Uncertain — might be right, might be wrong
4 = Fairly confident I know the answer
5 = I definitely know this

Respond with exactly one digit.
```

Score: use logprob distribution across 1-5 (the spread carries more information than the chosen token). Flat distribution = genuine uncertainty. Spike on 5 = confident.

#### Self-Consistency Verification

Only triggered for queries in the "uncertain zone" (Tier 2) — approximately 30-40% of queries.

```python
def verify_self_consistency(local_client, messages, first_response):
    """Re-generate at temperature=0.3 and compare key claims."""
    second_response = local_client.chat(messages, temperature=0.3)

    # Compare: do both responses agree on factual claims?
    # Simple: token overlap on named entities, numbers, key nouns
    # Better (v2.1): LLM-as-judge "do these agree on the facts?"
    claims_a = extract_claims(first_response)
    claims_b = extract_claims(second_response)
    return claims_agree(claims_a, claims_b)
```

#### Knowledge Verification

After generation, check if the response contradicts anything in the knowledge store:

```python
def verify_against_knowledge(response, memory, embed_client):
    """Check response doesn't contradict stored knowledge."""
    # Embed key claims from the response
    claims = extract_claims(response)
    for claim in claims:
        claim_emb = embed_client.embed(claim)
        hits = memory.search(claim_emb, limit=3, min_similarity=0.70)
        for hit in hits:
            if contradicts(claim, hit.entry.content):
                return False  # contradiction found → escalate
    return True  # no contradiction → safe
```

This signal gets STRONGER as memory grows — more knowledge = more facts to verify against. The opposite of drift.

---

### 3. Executor (`executor.py`)

The ReAct execution loop with step-level tiered routing. This is the central new component.

**Design decisions:**
- Iterative: LLM generates one action at a time, sees the result, decides next action
- Tiered confidence: each LLM turn routes through the same tiered system as Q&A (GSA → logprob → verify)
- Skill-aware: if a matching skill exists, its steps guide the execution plan
- Bounded: max iterations (default 20), max escalations per task (default 5)
- Observable: emits progress events for the thought renderer

```python
@dataclass
class ExecutionResult:
    answer: str
    steps_taken: int
    escalations: int
    tools_used: list[str]
    cost_usd: float
    skill_used: str | None       # which skill guided the execution
    skill_created: str | None    # new skill created from this execution

class Executor:
    def __init__(self, agent: Agent, tools: ToolRegistry, router: Router): ...

    def execute(self, task: str, *, context: str = None,
                on_progress: ProgressCallback = None) -> ExecutionResult:
        """
        The loop:
        1. Build messages with system prompt + tool schemas + history
        2. Generate with local model (with logprobs)
        3. Route via tiered system:
           - High confidence → execute locally
           - Uncertain zone → verify (self-consistency on tool call)
           - Low confidence → escalate to cloud
        4. If response contains tool_calls → execute them, append results
        5. If response is text (no tool call) → task complete
        6. After completion → trigger skill learning
        7. Record outcome → update Thompson posteriors
        """
```

**Step-level tiered routing — how it works:**

```python
# Each executor iteration:
tool_call = local_model.generate_with_tools(messages)
logprob = avg_logprob(tool_call_tokens)
threshold = router.get_threshold(category="code")  # from Thompson posterior

if logprob > threshold.high:
    # Tier 1: trust it
    execute(tool_call)
elif logprob < threshold.low:
    # Tier 3: escalate immediately
    tool_call = cloud_model.generate_with_tools(messages)
    execute(tool_call)
else:
    # Tier 2: uncertain — verify via self-consistency on the tool call
    tool_call_2 = local_model.generate_with_tools(messages, temperature=0.3)
    if tool_call.name == tool_call_2.name and tool_call.args == tool_call_2.args:
        execute(tool_call)  # consistent → probably correct
    else:
        tool_call = cloud_model.generate_with_tools(messages)  # disagreement → escalate
        execute(tool_call)
```

What this means in practice:
- Local model is confident about `terminal("pytest")` → Tier 1, executes locally, free
- Local model is uncertain about `terminal("kubectl set image ...")` → Tier 2, re-generates, disagrees with self → Tier 3, escalates to cloud
- Local model gives same tool call both times for `edit_file(...)` → Tier 2 passes, executes locally

**Why step-level routing matters:**
- A task might have 5 steps. 4 are easy, 1 is hard.
- Without step-level routing: entire task goes to cloud ($0.05)
- With step-level routing: 4 steps local + 1 escalated ($0.01)
- After learning: all 5 steps local ($0.00)

### 3. Skill System (`skills/`)

Skills are stored procedures with structured steps, semantic retrieval, and quality tracking. This is our key improvement over Hermes' Markdown-only approach.

**Design decisions:**
- Skills are stored in SQLite (not files) — queryable, versionable, embeddable
- Each skill has an embedding of its trigger/description for semantic retrieval
- Steps can optionally include tool name + argument templates (structured, not just prose)
- Success/failure tracking per skill gates automatic use (unvalidated skills get flagged)
- Skills are loaded on-demand: only the compact index is in the system prompt

#### Skill Schema

```python
class SkillStep(BaseModel):
    order: int
    description: str
    tool: str | None = None           # which tool this step uses
    args_template: dict | None = None # argument pattern with {placeholders}
    verification: str | None = None   # how to verify this step succeeded

class Skill(BaseModel):
    id: str
    name: str                         # short identifier
    description: str                  # one-line (for the index in prompt)
    trigger: str                      # when should this skill be loaded
    steps: list[SkillStep]
    tools_required: list[str]
    source: str                       # "cloud_escalation" | "user" | "reviewer"
    success_count: int = 0
    failure_count: int = 0
    last_used: str | None = None
    created_at: str
    updated_at: str
    version: int = 1
    embedding: list[float] | None = None
```

#### Skill Store (`skills/store.py`)

```python
class SkillStore:
    def insert(self, skill: Skill) -> Skill: ...
    def search(self, query_embedding: np.ndarray, limit: int = 5) -> list[ScoredSkill]: ...
    def get(self, skill_id: str) -> Skill | None: ...
    def update(self, skill_id: str, **changes) -> None: ...
    def record_outcome(self, skill_id: str, success: bool) -> None: ...
    def get_index(self) -> list[SkillIndexEntry]: ...  # compact: name + description only
```

#### Skill Loader (`skills/loader.py`)

Builds the compact skill index injected into the system prompt:

```
## Available Skills
When a task matches one of these skills, load it with skill_view(name) before proceeding.

- deploy_to_staging: Deploy the app to staging via Docker + kubectl
- run_tests: Run pytest with coverage reporting
- create_pr: Create a GitHub pull request from the current branch
```

The full skill body (steps, tool patterns, verification) is only loaded when the agent calls `skill_view(name)`. This keeps the system prompt lean (Hermes pattern).

#### Skill Lifecycle

```
Creation                    Usage                     Maintenance
─────────                   ─────                     ───────────
Cloud escalation            Agent finds matching      Reviewer patches
  → extractor finds         skill via semantic        after execution
    procedure               search → loads it →       deviates from
  → stored as skill         executes steps            stored steps
    (source: "cloud")       → tracks success/fail
                                                      Curator archives
User teaches                                          after 30 days
  → stored as skill                                   unused (v2.1)
    (source: "user")
```

### 4. Background Skill Reviewer (`skills/reviewer.py`)

After task execution completes, a lightweight review pass examines the execution trace and decides what to persist. Inspired by Hermes' background review fork, but triggered by *execution outcomes* rather than conversation turns.

**Design decisions:**
- Runs synchronously after task completion (not in a background thread for v2.0 — simplicity first)
- Examines: which steps escalated, which tools were called, what failed
- Follows strict preference order (from Hermes): patch existing skill > create new
- Uses the local model for review (cheap, fast) — falls back to storing raw trace if extraction fails

```python
class SkillReviewer:
    def review(self, trace: ExecutionTrace) -> ReviewResult:
        """
        Signals that trigger skill creation/update:
        1. Cloud escalation happened during execution
           → Extract the procedure cloud demonstrated
        2. Execution deviated from loaded skill
           → Patch the skill's failing steps
        3. New tool combination discovered
           → Create a new skill
        4. Existing skill succeeded
           → Increment success_count (no LLM call needed)
        5. Existing skill failed
           → Increment failure_count, flag for revision
        """
```

**What gets stored:**

| Signal | Action |
|--------|--------|
| Cloud showed a new tool call pattern | Create skill with the tool sequence |
| Cloud corrected a step the local model got wrong | Patch that step in the existing skill |
| Entire execution succeeded with no escalation | `skill.success_count += 1` |
| Execution failed despite having a skill | `skill.failure_count += 1`; if ratio > 0.5, flag |
| No matching skill existed and task succeeded | Create new skill from the execution trace |

### 5. Extended LLM Client (`llm_client.py` changes)

The LLM client needs to support function calling for the executor.

**Changes needed:**
- Add `tools` parameter to `chat()` and `chat_with_logprobs()`
- Parse tool_calls from response (OpenAI format: `response.choices[0].message.tool_calls`)
- Return tool calls in a structured format alongside content
- Ollama: use the native function calling support (Ollama 0.5+)
- OpenAI-compatible: standard function calling

```python
@dataclass
class ToolCallResult:
    id: str
    name: str
    arguments: dict

class ChatResponseWithTools(ChatResponse):
    tool_calls: list[ToolCallResult] = []
```

### 6. Extended Learning Extractor

Currently extracts facts only. v2.0 extends it to also extract procedures and tool patterns from cloud responses.

**Changes:**
- When cloud response contains tool calls → extract as a skill
- When cloud response describes a multi-step procedure → extract as a skill
- Existing fact extraction continues unchanged

### 7. Document Synthesis on Ingest (`document_store.py` + `learning_extractor.py`)

**Problem:** `autodidact learn` currently chunks documents and embeds them for raw retrieval — pure RAG. The agent re-derives knowledge from raw chunks on every query. Nothing is internalized.

**Insight (Karpathy's "LLM Wiki" pattern):** Humans don't re-read source pages every time they're asked a question. They internalize: build mental models, summaries, key facts. The agent should do the same — "compile" documents into knowledge at ingest time, not re-derive at query time.

**Design:** During `autodidact learn`, after chunking for raw retrieval (unchanged), also run a **synthesis pass** that extracts structured knowledge and stores it in the agent's knowledge store.

```
Current (v1.0):
  document → chunk → embed → store chunks → retrieve raw chunks at query time

New (v2.0):
  document → chunk → embed → store chunks (backup for detail retrieval)
         ↘ synthesize → extract facts/concepts → store in knowledge store
```

**How it works:**

1. After chunking, group chunks into sections (~2000 tokens each, respecting natural boundaries)
2. For each section, call `LearningExtractor.extract_from_document(section_text, source_file)` — a new method that prompts for key facts, concepts, and relationships (not Q&A format)
3. Store extracted entries in the knowledge store with `source: "document_ingest"` and a reference back to the source file
4. At query time: the existing memory-first path catches these synthesized entries before falling back to raw chunk retrieval

**Why both layers survive:**
- Synthesized knowledge handles "what is X?" and "how does Y relate to Z?" — common questions answered from internalized understanding
- Raw chunks handle "what exactly does page 47 say about..." — specific detail that synthesis may have dropped
- The knowledge store is the fast path; raw retrieval is the fallback for precision

**Extraction prompt (sketch):**

```
Given this section of a document, extract the key facts, concepts, and
relationships. For each, provide:
- A concise statement of the fact/concept
- A natural question someone might ask that this answers
- Key terms for retrieval

Do NOT extract trivial or obvious information. Focus on what would be
valuable to recall without re-reading the source.
```

**Implementation:** ~80 lines added to `document_store.py` (synthesis pass after chunking) + a new `extract_from_document()` method on `LearningExtractor` (~40 lines, reuses existing extraction infra with a different prompt).

**Scaling concern:** A 50-page PDF might produce 25 sections × 1 LLM call each = 25 calls to the local model during ingest. This is acceptable — ingest is a one-time operation, not query-time latency. For very large documents, batch sections or cap at N extractions.

### 8. CLI Extensions

```
autodidact do "task description"     # Execute a task (new)
autodidact skills list               # List learned skills
autodidact skills view <name>        # Show skill details
autodidact skills search "query"     # Semantic skill search
```

The `do` command enters the execution loop. The `chat` command stays as Q&A (v1 behavior) but can dispatch to the executor if the user's message looks like a task.

### 9. Extended Thought Renderer

New tags for execution mode:

```
[SKILL] Loading procedure: deploy_to_staging v2
[STEP 1/4] Run test suite
  → terminal: pytest --tb=short
  → confidence: 0.91 [LOCAL]
[STEP 2/4] Build Docker image
  → terminal: docker build -t myapp:latest .
  → confidence: 0.88 [LOCAL]
[STEP 3/4] Push to registry
  → confidence: 0.48 [ESCALATING to cloud]
  → cloud: docker push registry.example.com/myapp:latest
  → [LEARNED] Updated skill step 3 with registry URL pattern
[STEP 4/4] Deploy to k8s
  → terminal: kubectl set image deployment/myapp ...
  → confidence: 0.85 [LOCAL]

Done. 3/4 steps local, 1 escalated. Cost: $0.004.
```

---

## Data Flow: A Complete Execution

**Scenario:** User asks "Run the tests and fix any failures"

```
1. DISPATCH
   Agent classifies: this is a task (mentions "run" + "fix" → action verbs)

2. SKILL SEARCH
   Embed "run tests and fix failures"
   Search skill store → finds "run_pytest" (0.82 similarity)
   Load skill: steps = [run pytest, parse failures, fix each failure, re-run]

3. EXECUTION LOOP
   Iteration 1:
     Local model: tool_call(terminal, {command: "pytest --tb=short"})
     Logprob: 0.94, category "code" threshold: 0.78
     → Tier 1 (above high_threshold) → execute locally
     Result: "2 failures: test_auth.py::test_login, test_db.py::test_migrate"

   Iteration 2:
     Local model: tool_call(read_file, {path: "test_auth.py"})
     Logprob: 0.96 → Tier 1 → execute locally
     Result: <file contents>

   Iteration 3:
     Local model: tool_call(edit_file, {path: "src/auth.py", ...})
     Logprob: 0.62 → uncertain zone → Tier 2: VERIFY
     Re-generate: tool_call(edit_file, {path: "src/auth.py", different args...})
     Tool calls DISAGREE → Tier 3: ESCALATE
     Cloud model: tool_call(edit_file, {path: "src/auth.py", old: "...", new: "..."})
     Execute cloud's edit.
     → Thompson update: code category β += 1 (local was wrong)

   Iteration 4:
     Local model: tool_call(terminal, {command: "pytest test_auth.py --tb=short"})
     Logprob: 0.93 → Tier 1 → execute locally
     Result: "1 passed"

   ... (continue for second failure) ...

   Iteration N:
     Local model: text response "All tests pass. Fixed auth token validation
                  and database migration ordering."
     → Task complete.

4. BACKGROUND REVIEW
   Reviewer examines trace:
   - Step 3 escalated → cloud showed a specific edit pattern for auth
   - Extract: "When fixing auth token validation, check the decode() call
     matches the encode() algorithm"
   - Update skill "run_pytest" step 3 with this pattern? No — this is a
     new fact about auth, not a general test-fixing procedure.
   - Store as knowledge entry instead.

5. RESULT
   ExecutionResult(
     answer="All tests pass. Fixed 2 failures.",
     steps_taken=8,
     escalations=1,
     cost_usd=0.004,
     skill_used="run_pytest",
   )
```

---

## Database Schema Additions

```sql
-- Skills table
CREATE TABLE IF NOT EXISTS skills (
    id              TEXT PRIMARY KEY,
    name            TEXT NOT NULL UNIQUE,
    description     TEXT NOT NULL,
    trigger_text    TEXT NOT NULL,
    steps           TEXT NOT NULL DEFAULT '[]',    -- JSON array of SkillStep
    tools_required  TEXT NOT NULL DEFAULT '[]',    -- JSON array of tool names
    source          TEXT NOT NULL CHECK(source IN ('cloud_escalation','user','reviewer')),
    success_count   INTEGER NOT NULL DEFAULT 0,
    failure_count   INTEGER NOT NULL DEFAULT 0,
    last_used       TEXT,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL,
    version         INTEGER NOT NULL DEFAULT 1,
    embedding       BLOB
);

CREATE INDEX IF NOT EXISTS idx_skills_name ON skills(name);
CREATE INDEX IF NOT EXISTS idx_skills_source ON skills(source);
CREATE INDEX IF NOT EXISTS idx_skills_last_used ON skills(last_used);

-- Execution trace log (for reviewer and debugging)
CREATE TABLE IF NOT EXISTS execution_traces (
    id              TEXT PRIMARY KEY,
    task_text       TEXT NOT NULL,
    skill_used      TEXT,                          -- skill ID if one was loaded
    steps           TEXT NOT NULL DEFAULT '[]',    -- JSON: [{tool, args, result, escalated, confidence, tier}]
    outcome         TEXT NOT NULL CHECK(outcome IN ('success','failure','timeout')),
    total_cost      REAL NOT NULL DEFAULT 0.0,
    escalation_count INTEGER NOT NULL DEFAULT 0,
    created_at      TEXT NOT NULL
);

-- Per-category Thompson Sampling posteriors for tiered routing.
-- Replaces the per-signal Thompson params from v1.0 experiments.
-- Each row tracks how well LOCAL performs for a given query category.
CREATE TABLE IF NOT EXISTS routing_posteriors (
    category        TEXT PRIMARY KEY,
    alpha           REAL NOT NULL DEFAULT 1.0,     -- successes (local was correct)
    beta_param      REAL NOT NULL DEFAULT 1.0,     -- failures (local was wrong)
    total_queries   INTEGER NOT NULL DEFAULT 0,
    last_updated    TEXT NOT NULL
);

-- Seed initial categories with uniform priors
INSERT OR IGNORE INTO routing_posteriors (category, alpha, beta_param, total_queries, last_updated)
VALUES
    ('factual', 1.0, 1.0, 0, ''),
    ('realtime', 1.0, 1.0, 0, ''),
    ('reasoning', 1.0, 1.0, 0, ''),
    ('creative', 1.0, 1.0, 0, ''),
    ('code', 1.0, 1.0, 0, ''),
    ('personal', 1.0, 1.0, 0, '');
```

---

## Configuration

v2.0 config extends v1.0 YAML:

```yaml
# ~/.autodidact/config.yaml
local:
  model: qwen2.5:7b
  embedding_model: qllama/bge-large-en-v1.5
cloud:
  provider: openai
  model: gpt-4o
routing:
  # Tiered routing replaces the single confidence_threshold from v1.0.
  # These are starting defaults — Thompson Sampling adapts them per category.
  tier1_threshold: 0.80       # above this → trust local (fast path)
  tier3_threshold: 0.45       # below this → escalate immediately
  # Middle zone (0.45-0.80) → Tier 2 verification (self-consistency + KB check)
  gsa_escalate_threshold: 0.3 # GSA p_yes below this → skip local entirely
  verify_temperature: 0.3     # temperature for self-consistency re-generation
memory:
  path: ~/.autodidact/memory.db
# v2.0 additions:
execution:
  max_iterations: 20
  max_escalations_per_task: 5
  toolsets: [terminal, file, skills, memory]
skills:
  auto_create: true           # create skills from successful executions
  min_success_for_trust: 2    # use skill without flagging after N successes
  review_after_execution: true
```

---

## Success Criteria

1. **The apprentice demo works:** "Deploy to staging" fails on day 1 (escalates to cloud), succeeds on day 2 (from learned skill), costs $0.00 on day 3+.
2. **Step-level routing is measurable:** After 10 task executions, show that only uncertain steps escalate (not entire tasks).
3. **Skills accumulate:** After 20 task executions across various domains, the skill store has 5-10 validated skills with success_count > 2.
4. **Cost curve decreases:** Plot cost per task over time. It should trend downward as skills are acquired.
5. **v1.0 Q&A still works:** The existing `autodidact chat` / `autodidact query` commands work exactly as before. Task mode is additive.
6. **Document synthesis works:** After `autodidact learn` on a 10-page document, the agent can answer questions from internalized knowledge (memory hit) without falling back to raw chunk retrieval for common questions about the document's content.

---

## Implementation Plan

### Phase A: Tool Foundation + Document Synthesis + Routing Engine (Week 1)

- `tools/registry.py` — registry pattern with auto-discovery
- `tools/terminal.py` — shell execution with timeout and output truncation
- `tools/file_ops.py` — read, write, edit, search, list
- Extend `llm_client.py` — add `tools` parameter to chat methods
- `router.py` — tiered routing engine:
  - Category classification (keyword heuristic)
  - GSA v4 prompt (knowledge-conditional, "specific factual answer" framing)
  - Per-category Thompson Sampling posteriors (Beta distributions)
  - Self-consistency verification
  - Knowledge verification (response vs memory store)
  - Wire into existing `agent.py` Q&A path (replace inline `_compute_confidence`)
- Document synthesis on ingest: `LearningExtractor.extract_from_document()` + synthesis pass in `document_store.py`
- Tests for tool registry, individual tools, routing engine, and document synthesis

### Phase B: Execution Loop (Week 2)

- `executor.py` — ReAct loop with tool dispatch
- Step-level tiered routing (reuses `router.py` — same GSA/logprob/verify/Thompson system)
- Execution trace recording
- Extend `agent.py` — task detection and dispatch to executor
- Extend `thought_renderer.py` — execution-mode rendering
- Tests for executor

### Phase C: Skill Learning (Week 3)

- `skills/store.py` — CRUD + semantic search + lifecycle tracking
- `skills/loader.py` — compact index in prompt, on-demand loading via skill_view tool
- Extend `learning_extractor.py` — extract procedures from cloud tool calls
- `skills/reviewer.py` — post-execution review
- Database schema additions (skills table, execution_traces table)
- Tests for skill store and reviewer

### Phase D: Integration + Polish (Week 4)

- CLI: `autodidact do`, `autodidact skills` commands
- Skill-guided execution: planner uses loaded skills to structure step sequence
- End-to-end test: demonstrate the "learn once, execute free forever" loop
- Cost dashboard updates for execution mode
- Documentation

---

## Patterns Adopted from Open Source Projects

Specific implementations studied and adapted (all MIT/Apache licensed):

### From qmd (24K stars, MIT)
- **Weighted RRF fusion:** `weight / (k + rank + 1)` with k=60. Original query gets 2x weight, expansions 1x. Top-rank bonus: +0.05 for rank 1, +0.02 for rank 2-3.
- **BM25 normalization:** `|score| / (1 + |score|)` maps FTS5 negative rank to [0,1).
- **Break-point scored chunking:** heading=100, code block=80, hr=60, paragraph=20, list=5, newline=1. Squared-distance decay within search window.
- **Strong signal detection:** If top BM25 > 0.85 AND gap to #2 > 0.15 → answer is obvious, skip expensive vector search.
- **Position-aware score blending (v2.0):** Top-3 results get 75% RRF protection from reranker; lower ranks get 40%. Prevents reranker from demoting obviously-relevant top hits.
- **Reranker caching:** Cache reranker scores by chunk text (not file path). Identical chunks scored once.
- **Query expansion (v2.0):** Grammar-constrained generation of lex/vec/hyde variants.

### From Hermes Agent (148K stars, MIT)
- **FTS5 schema pattern:** External-content FTS5 with sync triggers on INSERT/UPDATE/DELETE. Trust scoring per fact (retrieval_count, helpful_count).
- **Memory context fencing:** `<memory-context>[System note: recalled knowledge, NOT new user input.]...content...</memory-context>` — separates memory from user input in prompts.
- **Tool registry:** Module-level `registry.register()` calls, AST-scanned auto-discovery, TTL-cached availability checks (~30s), generation counter for cache invalidation.
- **Curator lifecycle:** Inactivity-triggered (7 days default). Stale after 30 days unused, archived after 90. Never deletes — archives to recoverable location. Merges narrow skills into umbrella skills.
- **Skill format:** YAML frontmatter (name, description, version, platforms, tags) + structured sections (When to Use, Prerequisites, Procedure, Verification, Pitfalls).

### From Mem0 (25K stars, Apache 2.0)
- **MD5 hash deduplication:** `hashlib.md5(text.encode()).hexdigest()` check before insert. O(1) exact-duplicate prevention. Maintains seen_hashes within batch too.
- **Extraction principles:** Contextually rich (not atomic fragments). Preserve specific details (proper nouns, quantities, commands). Temporal grounding (relative → absolute dates). Link related memories via IDs.
- **History/audit table:** old_content, new_content, event (ADD/UPDATE/DELETE), timestamp. Full change log for every knowledge mutation.
- **Entity-boosted search (v2.0):** Extract named entities from query, search entity store, boost memories linked to matching entities.
- **Sigmoid BM25 normalization:** Query-length-adaptive midpoint and steepness for sigmoid normalization of BM25 scores.

### From ATLAS (2K stars, AGPL — patterns only, no code copied)
- **Per-task tier classification:** T1 (simple, direct) vs T2 (complex, multi-pass). Don't over-engineer trivial queries.
- **Thompson Sampling with cost weighting:** Beta posteriors per {category × tier}. Efficiency = p_success / cost.

---

## What's NOT in v2.0

Deferred to v2.1+:
- Knowledge lint operation — periodic health check: find contradictions between entries, stale facts superseded by newer sources, orphan knowledge with no retrieval hits, skills that conflict with each other (Karpathy "LLM Wiki" pattern, independently validated by Hermes Curator)
- Cross-referencing between knowledge entries — skills link to prerequisite skills, knowledge entries link to related entries, forming a lightweight knowledge graph (Karpathy wiki cross-references + MemPalace temporal KG, simplified)
- Web search / browser tools
- Sub-agent / delegation (Hermes/DeepSeek pattern)
- Curator (periodic skill consolidation/archival) — Hermes pattern, subsumed by lint operation above
- Background async review (v2.0 reviews synchronously for simplicity)
- MCP server mode
- OpenAI-compatible proxy (`autodidact serve`)
- Hybrid BM25+vector search (MemPalace pattern)
- Temporal knowledge graph (MemPalace pattern)
- Energy-based confidence from embeddings (ATLAS pattern)
- Thompson Sampling with cost weighting on routes (ATLAS pattern)
- Per-turn model routing via pre-screening call (DeepSeek pattern)
- LSP diagnostics injection (DeepSeek pattern)
- Workspace snapshots (DeepSeek pattern)
- RLM recursive processing (DeepSeek pattern)

---

## Open Questions

1. **Task detection heuristic:** How does the agent distinguish "what is X?" (Q&A) from "do X" (task)? Options: keyword heuristics, LLM classification call, user-explicit (`/do` prefix).

2. ~~**Tool call confidence signal:** Is avg_logprob over the tool_call tokens a valid confidence signal?~~ **RESOLVED:** Logprob alone is insufficient. v2.0 uses tiered routing: logprob determines which tier (fast-path / verify / escalate), self-consistency + knowledge verification catch hallucinations, GSA pre-screen catches refusals. See "Tiered Routing Engine" section.

3. **Skill granularity:** When should the reviewer create a skill vs store a fact? Rule of thumb: if it took 3+ tool calls, it's a skill. If it's a single fact, it's knowledge.

4. **Safety:** The terminal tool can execute arbitrary commands. v2.0 ships with an approval gate (user confirms before execution). YOLO mode is opt-in. Future: sandboxing.

5. **Context window management:** Long executions (20 iterations) may exceed context. Options: trim early iterations, summarize completed steps, or split into sub-tasks.

6. **Self-consistency cost:** Tier 2 verification doubles local generation time for uncertain queries (~30-40% of traffic). Acceptable for Q&A; potentially expensive in the executor where each iteration might be 3-5 seconds. Mitigation: for tool calls, compare only the tool name + argument structure (much cheaper than full-text comparison).

7. **Knowledge verification for new domains:** When the agent has no memory yet (cold start), knowledge verification can't catch anything. The system relies on GSA + logprob + self-consistency only. Acceptable — as memory grows, knowledge verification activates automatically.

8. **Category classification accuracy:** The keyword heuristic for query classification is crude. Misclassification means the wrong Thompson threshold is applied. Mitigation: start all categories at Beta(1,1) (uniform) so misclassification has minimal early impact; categories self-correct as outcomes accumulate.

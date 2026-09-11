# Autodidact — Interview Deep-Dive & Talking Points

> Personal prep doc for a system deep-dive interview (AI/Agent product engineering).
> Grounded in the actual codebase (~11.6K LOC, 639 tests). Not a spec — a study
> sheet organized around *how I'd explain each thing out loud* and *the trade-offs
> I actually made*.

---

## 0. The 60-second pitch (memorize this)

> Autodidact is a **local-first AI agent that gets cheaper the more you use it.**
> Most agent frameworks send every step to a frontier model, so cost scales with
> *task length*, not *task difficulty*, and the system never improves — solving the
> same problem tomorrow costs exactly what it cost today.
>
> Autodidact routes each step across three tiers — **memory (free) → local model
> (free) → cloud (expensive)** — and it only escalates the steps it's genuinely
> uncertain about. Critically, **every cloud escalation is distilled into permanent
> local knowledge**, so the *share* of steps needing the cloud drops over time. It's
> an apprentice: asks a lot on day one, handles most things itself by week two.
>
> The hard technical problem is the routing decision — *knowing what you don't know*
> — which I solve with model-agnostic confidence signals and an online bandit,
> deliberately avoiding any per-model training.

**Headline numbers (from the dev benchmark run):** ~67% of repetitive
codebase/doc queries intercepted locally; ~70% cost saved over 30 dev queries.
*(Be honest that these are dev-run figures, not a controlled paper benchmark.)*

---

## 1. The problem framing (why this exists)

The framing is the strongest part of the story — lead with it.

- **Who hurts:** developers running agentic LLM workflows paying per cloud token.
- **When it bites:** a multi-step task fires N cloud calls; most steps are routine
  (file reads, boilerplate, lookups already answered once). **Cost and latency scale
  with task length, not difficulty.**
- **The missing property: compounding.** A static setup never gets cheaper. Autodidact's
  differentiator is the *downward cost curve* — the fraction of cloud calls falls as
  memory grows.
- **What "solved" looks like:** routine steps handled locally/from memory; cloud share
  drops over time; **no accuracy regression** (cheaper routing must not raise error rate).

**Adversarial check I put in my own design doc** (shows maturity — bring this up):
- *"What if a local 7B isn't good enough for 'routine' steps?"* Then step-routing just
  adds latency (double-generation) without saving calls. **Mitigation:** the GSA pre-gate
  escalates *immediately* when the local model would hedge, so we don't pay for
  double-generation on hopeless steps.
- *"What if step difficulty isn't separable?"* If every step needs whole-task context,
  per-step routing degenerates to whole-task routing and the win evaporates. This is a
  real risk I can't fully rule out — I'd instrument the cloud/local/memory split on real
  tasks before over-investing.

---

## 2. Architecture at a glance (the whiteboard sketch)

```
User request
   │
   ▼
┌───────────────────────────────────────────────┐
│  RETRIEVAL  (embed once, search all tiers)      │
│  Tier 1/2: knowledge (FAISS vectors + FTS5)     │
│  Tier 3:   raw doc chunks (FAISS + BM25, RRF)   │
│  → context + knowledge_similarity score          │
└───────────────────────────────────────────────┘
   │
   ▼
┌───────────────────────────────────────────────┐
│  DISPATCH: question (Q&A) or task (execution)?  │
└───────────────┬───────────────────┬────────────┘
        Q&A     │                   │   Task
                ▼                   ▼
   ┌────────────────────┐   ┌─────────────────────────┐
   │ Q&A pipeline        │   │ ReAct executor           │
   │  memory hit?         │   │  plan steps (skill/LLM)  │
   │  GSA pre-screen      │   │  per step: tiered route  │
   │  local gen + logprob │   │  tool call → verify      │
   │  verify → cloud?     │   │  bounded iters/escals    │
   │  learn → integrate   │   │  learn → integrate       │
   └────────────────────┘   └─────────────────────────┘
                │                   │
                ▼                   ▼
      ┌───────────────────────────────────┐
      │ BACKGROUND REVIEW (async)          │
      │ new skill? patch skill? new facts? │
      └───────────────────────────────────┘
```

**Module map (what actually exists in `autodidact/`):**

| Module | Role |
|---|---|
| `agent.py` | Orchestrator: retrieval → dispatch → route → learn |
| `routing/step_router.py`, `routing/stages.py` | Tiered routing engine + pipeline stages |
| `signals/grounded_self_assessment.py` | GSA pre-screen (YES/NO logprob probe) |
| `executor.py` | ReAct loop with step-level routing (v2) |
| `tools/registry.py`, `file_ops.py`, `terminal.py` | Self-registering tool registry |
| `knowledge_store.py` | SQLite + FAISS + FTS5, Ebbinghaus decay, STM/LTM |
| `document_store.py` | Ingestion (chunk + embed) + hybrid search |
| `learning_extractor.py` | Turns cloud responses into structured facts |
| `llm/` (backends) | Ollama / OpenAI-compatible / Bedrock adapters |
| `config.py`, `database.py`, `types.py` | Typed config, schema+migrations, Pydantic models |

**Design stance to state clearly:** *the LLM WRITES knowledge but NEVER INDEXES it.*
Chunking + embedding is deterministic. This keeps retrieval reliable, fast, and free of
hallucination — the model can't corrupt the index.

---

## 2.5 Systems & infrastructure (map to the four eng pillars)

> This section exists to answer the *systems* interviewer directly. Each claim points at
> real code — don't overclaim beyond it. The honest framing: this is a **single-process,
> embedded runtime**, not a distributed service — so I speak to the *primitives*
> (state machines, durable state, event pipelines, resilience, telemetry) that would
> port directly to a distributed agentic runtime, and I'm explicit about where the
> single-node boundary is.

### 2.5.1 Distributed-systems primitives: state machine + durable state + resume

The v2 executor is a **crash-safe, resumable state machine**, which is the piece most
relevant to an agentic-runtime role.

- **Explicit state machine.** A task run (`execution_trajectories`) has a status enum
  enforced *at the schema level*: `CHECK(status IN ('running','done','budget_exhausted','failed'))`.
  Status *drives resume eligibility* — only `running` trajectories are resumable.
- **Write-ahead durability.** `PRAGMA journal_mode = WAL` (`database.py:7`). Every ReAct
  iteration is checkpointed as it happens: `TrajectoryStore.record_step()` **commits
  immediately** (`trajectory_store.py:86`) so a task killed mid-run (crash, SIGKILL, budget
  exhaustion) resumes from its **last committed step** instead of re-paying for the whole run.
- **Event-sourced replay.** State isn't stored as a blob — it's an **append-only log of
  steps** (`trajectory_steps`, one row per iteration, `UNIQUE(trajectory_id, step_index)`).
  `replay_messages()` deterministically **rebuilds the LLM message list by replaying committed
  steps in order**. This is event sourcing: the step log is the source of truth, the message
  list is a projection. Same pattern powers both resume *and* offline learning (query completed
  trajectories for skill extraction + threshold tuning).
- **The one-liner:** *"Each ReAct step is a committed event; resume is a deterministic replay
  of the event log — so an interrupted agent never pays twice for work it already did."*
- **Honest boundary:** consensus/multi-node is **out of scope** — this is one process with one
  SQLite writer. But the *durable-state + replay + status-machine* design is exactly what you'd
  lift into a distributed executor (swap SQLite for a durable log/queue, the state machine and
  replay logic are unchanged). Say that explicitly rather than pretending it's distributed.

### 2.5.2 Search & retrieval indexing (the deep bucket — full detail in §4)

Everything a search interviewer wants, and it's all real:
- **Inverted index:** SQLite **FTS5** external-content table (`document_chunks_fts`) with
  `porter unicode61` tokenization, kept in sync by INSERT/DELETE/UPDATE **triggers** — a real
  on-disk inverted index searched in C, not a Python scan.
- **Vector embeddings:** 1024-dim bge-large, stored as BLOBs *in the same row as the text*
  (no desync), searched via **FAISS `IndexFlatIP`** (exact cosine) held resident in RAM as a
  **rebuildable cache** over SQLite-as-truth.
- **Chunking strategy (tuned, not arbitrary) — content-type-aware** (full evolution story in
  §4.8): **AST-aware for code** (tree-sitter, whole functions), **heading-section-aware for
  markdown/prose** with **atomic tables** (`chunk_markdown`), **boundary-aware sliding window**
  for everything else. Unifying rule: every split piece **carries its structural header**
  (function signature / heading trail / table column header). ~384-token target, ~50-token
  overlap. The 384 is *empirical* — was 500, lowered after live testing showed 500 crowded
  bge-large's 512-token window on token-dense Python (`document_store.py:63`). Hard cap 480
  tokens enforced with the **real BGE tokenizer** (chars-per-token heuristic as offline fallback).
- **Hybrid indexing:** dense (FAISS) + sparse (BM25/FTS5) fused with **Reciprocal Rank Fusion
  (k=60)** for ordering, cosine for thresholding. FAISS/USearch have no built-in hybrid — the
  fusion layer is hand-built (`search_hybrid`).

### 2.5.3 Backend/serving: resilience, pooling, sub-second tool APIs

- **Provider-agnostic gateway.** One `ChatBackend` interface, three adapters (Ollama /
  OpenAI-compatible / Bedrock). Routing/executor code asks for `chat_with_logprobs` and gets a
  uniform result regardless of provider.
- **Exponential backoff tuned from a real incident.** `_with_retries` with backoff
  `(1,2,4,8,16)s` (`llm/backend.py:88`). The comment records *why* it's that long: EXP-003
  observed **69 consecutive Bedrock throttle failures across ~12s** — the short `(0.5,1,2)`
  schedule couldn't clear the burst. **Retry only on allow-listed transient exceptions; HTTP
  4xx propagates immediately** (don't retry a bad request). This is the "rate-limit / 429
  handling" story, grounded in a real throttle incident.
- **Timeouts everywhere.** Per-provider read/connect timeouts (default 300s, sized for cold-start
  model loads), with actionable error messages ("model may need a longer timeout for cold starts").
- **Sub-second tool execution API with a uniform envelope.** `ToolRegistry.dispatch()` always
  returns a JSON envelope: `{"ok": true, "result": ...}` or `{"ok": false, "error": "..."}`
  (`tools/registry.py:131`). Handler exceptions are **caught and serialized as errors** — a
  tool crash never crashes the loop, it becomes an observation the agent reasons about.
  Argument **type coercion** silently fixes the LLM passing `"5"` where an int is expected.
- **Connection/worker pooling.** Ingestion uses a `ThreadPoolExecutor` (configurable
  `ingest.workers`, default 4); **all DB writes funnel to one thread** because SQLite is
  single-writer — the pool parallelizes IO-bound embed calls, not the writer. That's the
  correct pooling model for a single-writer store, and I can explain why a naive multi-writer
  pool would corrupt/lock the DB.
- **Honest boundary:** "high QPS gateway / rate limiting" — this is a *client* of LLM APIs, not
  a multi-tenant gateway serving high QPS. What I have is the *client-side* resilience
  (backoff, timeouts, circuit-breaker-shaped retry, graceful degradation). I'd say that plainly
  and then describe how I'd build the server side if asked.

### 2.5.4 ML infra / MLOps: eval pipelines, deterministic replay, telemetry

- **Evaluation pipeline (this is genuinely strong — see the research doc).** A full
  `benchmarks/` suite: per-signal AUROC with **1000-sample bootstrap CIs**, calibration
  diagrams, ablations, a RouteLLM-style trained baseline on a **disjoint** train split, and a
  memo generator that fills a template with the numbers. This is a real offline eval harness,
  not ad-hoc scripts.
- **Deterministic replay / reproducibility.** Seeded, resumable KB seeding (`benchmarks/seeding.py`:
  *"given the same seed, the same queries"*); experiments keyed by `run_id`; the trajectory
  replay above is itself deterministic. Reproducibility was designed in, not bolted on.
- **Telemetry / cost accounting.** Per-step cost is recorded on the trajectory
  (`cost_usd`, `escalations`, `tools_used`, `avg_logprob` per step in `trajectory_steps`).
  `Agent.savings()` computes cumulative **actual cost vs. estimated all-cloud cost** — the
  product's core metric (the savings curve) is a first-class telemetry output, not an
  afterthought. Cost rates are per-provider $/M-tokens.
- **Model serving:** local models served via **Ollama** (the setup wizard detects it, pulls
  models, starts the daemon); cloud via provider SDKs. I don't run a custom serving engine — I
  integrate existing ones — and I'd say so.

**How to use §2.5 in the room:** lead with 2.5.1 (state machine + durable replay) for a
systems/distributed interviewer — it's the most transferable to an agentic *runtime*. Pivot to
2.5.2 for search-heavy roles, 2.5.4 for MLOps-heavy roles. Always name the single-node boundary
before they find it.

---

## 3. Routing & confidence — *"knowing what you don't know"*

This is the intellectual core. If they push anywhere, push here.

### 3.1 Why the naive approach (logprob-only) fails

v1.0 routed on token logprob alone. It breaks in two directions, and I can name both:
- **Confident hallucination:** made-up facts are fluent and *predictable* → high logprob,
  wrong answer. The model doesn't know it's wrong.
- **Confident refusal:** "I don't have access to real-time info" is a rehearsed RLHF
  phrase → near-perfect token certainty, but it's a non-answer.
- **Root cause (the one-liner):** *logprob measures token predictability, not answer
  correctness.*

### 3.2 The multi-signal solution

Combine signals that are each model-agnostic and either **stable** or **improve** as
memory grows — explicitly *no per-model training*:

| Signal | Catches | Cost | Drifts? |
|---|---|---|---|
| **GSA pre-screen** | refusals, known-unknowns | 1 token | No |
| **Logprob** | general uncertainty | free (already computed) | No |
| **Self-consistency** | hallucinations, random errors | 2× generation | No |
| **Knowledge verification** | factual hallucination | 1 retrieval | **Gets better** |
| **Category × Thompson** | per-type thresholds | ~0 (cached) | Adapts |

### 3.3 GSA (Grounded Self-Assessment) — the pre-gate

- A **1-token** probe *before* generating: "can you give a specific, factual answer
  (not a hedge)? Respond YES or NO."
- **Score from the logprob distribution over YES/NO**, not the emitted token:
  `p_yes = exp(YES) / (exp(YES) + exp(NO))`. The distribution carries more information
  than the hard label. (Bedrock fallback: no logprobs → hard YES/NO = 1.0/0.0.)
- **Knowledge-conditional:** when memory has strong hits (score above a floor), inject
  them into the prompt; when it doesn't, use the *bare* prompt. Key subtlety I learned
  experimentally (lab notes P9/P10): injecting *weak* retrieval hits **degrades** the
  signal — priming the model with "here's what you know" when the hits are junk pushes
  it toward false YES. So it's all-or-nothing on retrieval conditioning.
- **Prompt wording matters:** "not a hedge or disclaimer" directly targets the refusal
  failure mode. This wording went through several versions (v4 is production).

### 3.4 Per-category Thompson Sampling (the online adaptation)

- Instead of one global 0.7 threshold, keep a **Beta(α, β) per query category**
  (factual, realtime, reasoning, creative, code, personal).
- Threshold derived from the posterior mean: category where local usually succeeds →
  threshold drops → fewer escalations; category where local usually fails → threshold
  stays high → more escalations.
- Update is trivial: after an escalation, compare local vs cloud → `α += 1` if local was
  right, else `β += 1`. **Six counters, not a neural net.**
- **Why a bandit and not an MLP:** no retraining, no drift, works on a fresh model from
  day 1, and Thompson exploration is built in (occasionally tries the "wrong" tier to
  detect if performance changed). This ties to my planned **paper on non-stationary
  bandit routing** — the reward distribution shifts as memory grows, which vanilla
  Thompson doesn't model.

### 3.5 The tier ladder (say this crisply)

```
Tier 0: memory hit          — free, instant, no generation   (similarity > ~0.85)
Tier 1: local single-pass   — 1 generation, trust it          (logprob > high thr)
Tier 2: local verified      — 2 gens + knowledge check, free  (middle zone, ~30-40%)
Tier 3: cloud escalation    — expensive, authoritative        (logprob < low thr, or GSA=NO)
```

The two verification signals in Tier 2:
- **Self-consistency:** regenerate at temp 0.3, compare claims. Disagreement → escalate.
- **Knowledge verification:** check the answer doesn't contradict the store. *This signal
  strengthens as memory grows — the opposite of drift.*

---

## 4. Memory & retrieval — the RAG substrate

This is where we spent the most engineering effort this cycle, so I can go deep.

### 4.1 Storage model (know the schema cold)

Everything is in **one SQLite file** — no server, no Docker (contrast with pgvector,
which would mean shipping users a Postgres daemon; that breaks the local-first premise).

- **`knowledge_entries`** — learned facts. Columns include `content`, `question`,
  `embedding` (BLOB), `confidence`, `tier` (STM/LTM), `usage_count`, `valid_from`/
  `valid_to` (temporal), `domain`/`topic`/`category` (scope).
- **`document_chunks`** — ingested doc/code chunks. `content` (text) and `embedding`
  (BLOB) live **in the same row** — text and vector can't desync, no join, no ID mapping.
- **`document_chunks_fts`** — FTS5 virtual table (external-content), kept in sync by
  triggers. This is a *real on-disk inverted index* that SQLite understands and searches
  in C. Contrast: the vector BLOB is opaque bytes to SQLite — all vector math happens in
  Python, so the vector index must live *outside* SQLite (in FAISS, in RAM).

### 4.2 The three search paths and hybrid fusion

- **Vector search:** cosine over embeddings.
- **BM25 (`search_bm25`):** `WHERE document_chunks_fts MATCH ?` — SQLite's FTS5 engine
  computes ranking, returns ranked rows.
- **Hybrid (`search_hybrid`):** run both, fuse with **Reciprocal Rank Fusion (k=60)** for
  *ordering*, but return the **cosine score** for *thresholding* (so downstream 0.75/0.30
  cutoffs stay meaningful). BM25-only hits get a small penalty since they lack a cosine
  score. **FAISS/USearch have no built-in hybrid — you assemble dense + sparse + fusion
  yourself, which is exactly what this does.**

### 4.3 FAISS: exact vs approximate (a decision I can defend)

- Uses **`IndexFlatIP`** on L2-normalized vectors (inner product = cosine) — **exact**,
  not HNSW/IVF.
- **Why exact:** `IndexFlatIP` is still O(n·d), but it moves the dot products from a
  Python loop into vectorized C/BLAS and keeps vectors resident in RAM (vs re-reading and
  re-deserializing every BLOB per query). That's a 10–100× wall-clock win with **zero
  change in results.**
- **Why not HNSW/IVF (yet):** they're *approximate* (recall < 100%) and only pay off at
  ~millions of vectors — a single big codebase is tens of thousands of chunks. And
  crucially, **approximate recall is wrong for dedup**: dedup asks "is there a
  near-identical entry?" — a recall miss inserts a redundant copy. Exact never misses.
  HNSW/IVF is a deliberate *future scale decision*, not a default.

### 4.4 FAISS as a rebuildable cache (the key architectural insight)

- **SQLite is the source of truth; FAISS is a disposable in-RAM accelerator.** FAISS never
  reads SQLite; it holds its own copy of the vectors. On restart, the index is gone and
  gets rebuilt from SQLite. No `write_index` file to keep in sync — sidesteps the
  "two files drift apart" bug class.
- **IDs:** `IndexFlatIP` doesn't store your IDs — it returns positional indices. I keep a
  parallel `_faiss_ids` list mapping position → UUID, then fetch the full row from SQLite.
  (Could use `IndexIDMap2` for int64 IDs, but our IDs are UUIDs, so the side-list is
  simpler.)
- **Memory:** n × dim × 4 bytes. At 1024-dim: 10k chunks ≈ 41MB, 100k ≈ 410MB, 1M ≈ 4.1GB.
  Fine until the millions.

### 4.5 The incremental-index problem (best "issue faced" story in this area)

**Original pattern:** any mutation set `_faiss_dirty = True`; next search does a full O(n)
rebuild. I claimed this was fine for "write-rarely-read-often."

**The bug in my own reasoning (caught during design review):** the knowledge store is
*not* write-rarely — it **learns on every cloud escalation**. The real cycle is
`query → search → escalate → learn (invalidate + insert) → next query → search → REBUILD`.
So it pays a full O(n) rebuild on essentially *every* post-escalation query, and the cost
grows as the KB accumulates. My framing was exactly backwards.

**The fix (partially landed):** make `insert()` do an **incremental `_faiss_add`** — append
the single normalized vector to the live index, O(dim), no rebuild.

**The subtler problem that surfaced — dedup completeness:** if `invalidate()` just leaves
dead vectors in the index (relying on a `valid_to IS NULL` post-filter to drop them from
results), then under churn the index fills with near-identical dead copies. A dedup search
with `k = 3` can get *3 dead copies* in its top-k, post-filter drops all 3, returns empty →
concludes "no duplicate exists" → inserts a redundant copy. **The post-filter guarantees
correctness (never return a stale entry) but NOT completeness (never miss a live one)** —
and dedup needs completeness.

**The design I converged on (tombstone + periodic compaction):**
- `insert` → incremental add, O(dim).
- `invalidate` → O(1) tombstone (keep `valid_to`), don't rebuild.
- `search` → over-fetch `k = limit / (1 - dead_fraction)` (bounded to ~2×limit) + an
  adaptive re-query fallback if too few live hits survive.
- **compaction** → rebuild when dead fraction > ~30%, which *bounds* the over-fetch.
- These two levers are a matched pair: the compaction threshold caps the dead fraction,
  which caps how far you must over-fetch.

**Why not USearch (which has cheap graph deletes)?** USearch's HNSW gives ~O(log n)
removal — tempting for a churn-heavy store. But (a) it's *approximate*, which as noted is
wrong for dedup recall; (b) the O(n) compaction on a few-thousand-vector store is
sub-millisecond today, so the delete-cost problem is a scale problem I don't have yet;
(c) it's a new dependency and would diverge from the FAISS path already proven in-repo.
Migrating both stores to USearch is a deliberate future task, not a now-decision.

### 4.6 Ebbinghaus decay & tiers (the "human memory" flourish)

- `retention = exp(-hours_since_access / stability)`, where
  `stability = base × (1 + ln(1 + access_count))`. Frequently-used knowledge stays;
  unused knowledge fades. Memory stays lean without manual eviction.
- **STM → LTM promotion** after N accesses; decay cycle expires stale LTM entries. This is
  also the natural place to run index **compaction** — a full pass already happens there.

### 4.7 Known gaps I'd name proactively

- **No metadata filtering on document search** — can't scope to `src/auth/` or "Python
  files only." Every query hits the whole corpus. The knowledge store *does* have scoped
  search, but it implements scope via *filtered brute-force* (can't combine the FAISS
  index with a `WHERE` predicate). Fixing "filtered ANN" properly is the state-of-the-art
  frontier (Qdrant's filterable HNSW, or `sqlite-vec` to get ANN + SQL filter in one
  engine). Roadmap, not shipped.
- **AST chunking covers Python/JS/TS only.** Go/Rust/Java/C/etc. fall back to the
  boundary-aware window (correct, just not function-aware). The mechanism generalizes — it's
  registering more tree-sitter grammars, not new logic. Highest-value next chunking step.
- **Structured data (JSON/YAML/CSV) has no structure-aware chunker** — sliced as text, so a
  chunk can split an object/record. A `chunk_json`/`chunk_csv` (split on top-level keys / row
  groups) would be the analogue of what `chunk_markdown` did for prose.
- **Table *embedding* still weak** (splitting is solved, §4.9): an intact numeric table embeds
  poorly. Higher-ceiling fixes (LLM-summary indexing, text-to-SQL) conflict with the
  deterministic-index rule — a deliberate not-yet, see §4.9.

### 4.8 Parsing & chunking evolution (a clean "it broke, I fixed it" arc)

One of the strongest concrete engineering stories — a naive v1 with two real failure modes,
rebuilt into content-type-aware pipelines. Tell it as *symptom → root cause → fix*.

**Where it started (v1, naive):**
- **Parsing:** essentially plain-text only. `read_text()` for text; PDF/DOCX unsupported or a
  naive extractor that flattened layout into mojibake (tables, reading order lost).
- **Chunking:** fixed-size sliding window with overlap, applied blindly to everything.

**Failure mode 1 — chunks missing info.** A blind window cuts mid-function or mid-sentence.
The embedding of a *fragment* retrieves poorly, and the model gets a chunk that references a
variable/step defined in the chunk *before* it → **incomplete answers.** Root cause: the
splitter respected byte offsets, not *meaning*.

**Failure mode 2 — binary docs unusable / crashes.** PDFs flattened to garbage; a missing
parser lib crashed ingestion instead of skipping the file.

**Failure mode 3 (found in the wild) — oversized chunks silently fail to embed.** Live
`autodidact learn .` on the repo emitted `Ollama HTTP 500: input length exceeds context
length` on specific chunks of `agent.py` — the 4-chars/token heuristic under-counts on dense
code, so "500-token" chunks were really ~521 BGE tokens, over the 512 model cap. The info in
those chunks just *vanished* from the index. (Regression-tested now in `test_chunk_token_aware.py`.)

**The fix — parsing: a degrade-gracefully chain** (`_read_text_from_file`):
1. **docling first** for PDF/DOCX → converts to **Markdown preserving layout + tables** (the
   mojibake fix). Optional dep, lazily loaded/cached.
2. **pymupdf / python-docx fallback** when docling absent or conversion fails.
3. **Skip-with-actionable-`ImportError`** ("`pip install pymupdf`") if neither — never crash.
   Text coverage widened from `.md`-ish to ~30 extensions (code, config, data, markup).
- **Principle:** *soft dependencies, best-parser-if-present, skip loud if none.*

**The fix — chunking: three content-type paths** (chosen in `_prepare_file`):
1. **Code → `chunk_code_ast` (tree-sitter).** Chunks on **semantic boundaries**: each top-level
   def/class is its own chunk (a function is *never* split across chunks). Small nodes (imports)
   grouped; big classes split into methods with the **class signature prepended as a header** so
   a method chunk keeps its context.
2. **Markdown/prose → `chunk_markdown` (added this session — §note below).** Splits on `#`
   headings; a section that fits stays **intact** (kills failure-mode-1 for docs); an oversized
   section is split and every piece after the first **re-states its heading trail** (`# Guide` →
   `## Deploy`) so a mid-section chunk still says what it's about.
3. **Everything else → `chunk_text` (boundary-aware window).** Still overlap-based, but
   `_find_split_point` breaks on the best boundary in priority order — **paragraph → line →
   sentence → word** — never mid-word.

**Failure mode 4 — tables shredded mid-row.** A Markdown table (incl. those docling extracts
from PDF/DOCX) is a 2D structure; a 1D window cut severs rows from their column header, leaving
fragments like `| 4.2 | 17 |` with no idea what the columns mean → useless for retrieval *and*
for the model. **Fix — table-aware chunking in `chunk_markdown`:** within an oversized section,
`_segment_section_blocks` separates prose runs from table blocks; a table that fits stays
**atomic** (never split), and a table larger than the target is **row-split with the column
header + delimiter row repeated on every piece** (`_chunk_markdown_table`) — the exact same
"prepend the structural header" trick used for headings and function signatures. Verified
end-to-end: a table inside a large section survives as one chunk carrying its heading trail.

**The safety net all paths share — hard token cap.** `_enforce_cap` uses the **real BGE
tokenizer** to recursively split anything over 480 tokens (correct by construction), with a
stricter 2.5-chars/token fallback when the tokenizer can't load offline. This is what closes
failure-mode-3 permanently.

**The unifying principle across every path — "carry your structural header":**

| Content | Primary unit | Header prepended to split pieces | Overlap? |
|---|---|---|---|
| Code (AST) | function / class | class/def **signature** | No (header instead) |
| Markdown prose | `#` section | **heading trail** (`# A` / `## B`) | Only inside an oversized section |
| Markdown table | the whole table | **column header + delimiter row** | No |
| Generic text | window | — (arbitrary boundary) | Yes, ~50 tok |

*Overlap is the fallback for arbitrary cuts; semantic units don't overlap, they carry a header.*

> **Honesty note for the interview:** the heading-section-aware prose chunker (`chunk_markdown`)
> and the table-aware handling are the **most recent additions** (with tests), made to close the
> gaps discussed here — before that, prose used the boundary-aware window. If asked "is that
> shipped?", the honest answer is "the AST path and boundary window shipped earlier; the
> heading-section + atomic-table paths are recent, tested, not yet battle-tested in production."

**One-liner:** *"We moved from splitting on byte offsets to splitting on meaning — functions
for code, sections and tables for docs — and made every chunk self-contained by prepending its
structural header, because a chunk that's lost its context retrieves like noise."*

**Deliberate non-goal:** no semantic/embedding-based chunking (embed sentences + cluster). The
chunking stays *deterministic* — no LLM in the index path — which is the same "LLM writes but
never indexes" principle. A legitimate future step to name, not a gap I'm hiding.

### 4.9 Table & figure semantics — the SOTA and where I stop (a strong "I know the frontier" answer)

Great question to be ready for, because it separates "I use RAG libraries" from "I understand
the failure modes." Tables/figures are **2D/spatial** structures forced through a **1D/linear**
chunk-and-embed pipeline. Two distinct failures: **splitting** (row severed from header — fixed
in §4.8) and **embedding** (even an intact table of numbers embeds weakly — a query like "Q3
revenue?" won't match a grid of digits).

**What I built (surgical, fits our constraints):** atomic-table chunking + row-split-with-
repeated-header. Solves *splitting*. Deterministic, zero new deps.

**The SOTA for the harder *embedding/representation* half** (know these as classes, not tool
trivia):
- **Summary/description indexing (multi-vector retriever)** — LLM writes a NL summary of the
  table ("quarterly revenue by region, 2023-25"); **embed the summary**, **return the full
  table**. Retrieval becomes prose-to-prose (strong signal). *Tension: uses an LLM at index
  time — violates our "LLM writes but never indexes" rule.* That's the trade-off I'd flag before
  adopting it.
- **Structured extraction → text-to-SQL / table-QA** — push tables into SQL/DataFrame, answer
  numeric questions by querying, not embedding. Best for precise aggregates ("sum of Q3"), which
  vectors are bad at.
- **Multimodal / vision embeddings (ColPali / ColQwen)** — embed the *rendered page image*
  directly, skip text extraction entirely. The 2024-25 frontier for visually-rich/scanned docs;
  heavier infra, less mature for precise numeric QA.
- **Layout-aware parsing (already have via docling)** — the foundation; it preserves tables as
  Markdown so they *survive* extraction. Our gap was purely downstream (now closed for splitting).

**How I'd frame the stopping point:** *"Atomic-table chunking is the surgical fix that respects
our deterministic-indexing constraint. LLM-summary indexing and text-to-SQL have a higher
ceiling but require relaxing 'no LLM in the index path' — a deliberate architectural decision.
ColPali-style multimodal retrieval is where the field is heading for figure-heavy docs, but it's
a bigger infra bet, so it's a roadmap item, not a now-decision."* Same judgment shape as the
FAISS→USearch and SQLite→sqlite-vec calls: know the SOTA, pick the constraint-respecting fix,
name what adopting the frontier would cost.

---

## 5. Agent execution — the ReAct loop

### 5.0 Control flow: dispatch, planning, and "who decides" (read this before the diagram)

**Correct the architecture diagram out loud.** The layer diagram shows a `DISPATCH: question
or task?` box and the design doc lists a `planner.py`. **Neither is shipped as drawn.** Be
precise about what actually exists — it's a stronger answer than the diagram.

- **No standalone dispatch classifier.** `agent.query()` is the Q&A pipeline; the `Executor`
  exists and is fully tested but **isn't wired into `agent.query()` or a CLI command** yet.
  Today the "dispatch" is *which CLI command the user runs* (`query`/`chat` = Q&A). The
  automatic question-vs-task router is the **integration seam I haven't closed.**
- **How dispatch is *meant* to work (and how I'd build it): fuse it into tool-calling.** Don't
  build a brittle upfront "Q&A vs task" classifier that can be confidently wrong before
  generating (the §3.1 failure mode again). Instead expose tool schemas and let the model's
  turn-1 output *be* the decision: **emits text → it was a question (done); emits a tool call →
  it's a task step (enter the loop).** The executor already does exactly this per turn
  (`executor.py:262`). So dispatch and routing collapse into one local generation that produces
  both the act-or-answer decision *and* the `avg_logprob` routing signal.
- **No planner. It's pure ReAct, not Plan-and-Execute.** `planner.py` does **not exist**. The
  model is handed `[system, user=task]` and **never asked to produce a step list.** Decomposition
  is *implicit and emergent* — the model decides **one action at a time**, sees the tool result,
  decides the next. The "plan" is just the sequence of tool calls that happened.
- **Who plans / who routes:** an **outer orchestrator loop** (`while steps < max_iterations`)
  drives it; **each turn is routed independently** (GSA gate → local/cloud → tier). So no single
  model plans upfront — whichever tier handles a given turn decides only *that* turn's action.
  Local might do steps 1-3, cloud handles a hard step 4, local resumes step 5.
- **GSA veto escalates a *step*, not the *task*.** If the GSA gate vetoes (`p_yes < threshold`),
  that step goes straight to cloud and local generation is skipped (`executor.py:244`) — but the
  cloud result returns *into the same loop*, and the next turn re-runs the gate. Cloud never
  "takes over" the task.

**ReAct (shipped) vs Plan-and-Execute (designed, not built) — the trade-off to articulate:**

| | ReAct (shipped) | Plan-first (`planner.py`, not built) |
|---|---|---|
| Decomposition | Implicit, one action at a time | Explicit step list upfront |
| Adapts to results | Yes — each step sees prior output | Plan can go stale as reality diverges |
| Cost | Cheaper on short tasks | Extra planning call(s) upfront |
| Long-task coherence | Can wander / re-try the same thing (bounded by `max_iterations`) | Global structure holds it together |
| Routing fit | Natural — route each step's confidence as proposed | Would route the plan, then each step |

*Framing:* "I shipped ReAct deliberately — simpler, adapts to tool results, fits step-level
routing. The cost is it can wander on long tasks (hence the hard bounds + trajectory
compression), and with no global view it can re-escalate the same stuck sub-problem. A planner
or reflection step is the fix when tasks get long enough that coherence beats per-step
adaptivity — roadmap, not a now-decision."

### 5.1 The executor (`executor.py`)

- **Iterative ReAct:** the LLM emits one action at a time, sees the tool result, decides
  the next action.
- **Step-level routing:** *each* LLM turn goes through the same tier ladder as Q&A
  (GSA → logprob → verify → cloud). This is the v2 differentiator — routing at the *step*
  granularity, not the whole task.
- **Bounded:** `max_iterations = 20`, `max_escalations = 5`. When the escalation budget is
  spent, a Tier-3 decision *falls back to running the local proposal* rather than blocking
  — graceful degradation, not failure.
- **Tier 0 shortcut:** a fresh, high-similarity whole-task memory hit can short-circuit
  the entire loop.

### 5.2 Tool registry (`tools/registry.py`)

- **Self-registering** (borrowed from Hermes): each `tools/*.py` registers at import.
- Tools are Python functions with an **OpenAI function-calling schema**; dispatch returns
  **JSON strings** (consistent result format).
- **Type coercion on arguments** — silently fix the LLM's type errors (it'll pass `"5"`
  where you want `5`).
- Implemented tools: `file_ops` (read/write/search/patch), `terminal` (shell),
  `fuzzy_match`.

### 5.3 Skill learning & background review (the "gets smarter" loop)

- After a task, a **background (async) review** runs so it doesn't block the user:
  new procedure discovered → create/patch a skill page; cloud showed a new technique →
  integrate into a knowledge page; facts learned along the way → integrate into topic
  pages.
- **Design principle:** skills are structured (SQLite + compact index + on-demand load),
  *not* Markdown-only like Hermes — because Markdown-only skills have no execution
  guarantees.

### 5.4 What I deliberately left out of v2 (scope discipline)

Async/concurrent *tool* execution (executor is single-threaded), a full planner module,
multi-user concerns. Naming these shows I scope rather than gold-plate.

### 5.5 Adding a reflection step — effort estimate (a good "how would you extend it" answer)

**Why:** pure ReAct has no global view, so it can re-escalate the same stuck sub-problem or loop
until `max_iterations`. A reflection step — periodically asking "am I making progress / am I
stuck / should I change approach?" — is the standard fix (Reflexion-style).

**Where it slots in:** the loop is already a clean state machine (`executor.py:227`) with all the
state reflection needs *already in hand* — `messages` (full history), `steps`, `escalations`,
`tools_used`, and the per-step `tool_result` envelopes (`ok`/`error`). Reflection is a new
branch near the top of the loop body, gated by a cadence (e.g. every N steps, or after a tool
error/repeat).

**Effort: roughly a day for a solid v1, half a day for a crude one.** Small, because the
scaffolding exists:

| Piece | Effort | Notes |
|---|---|---|
| Reflection prompt + call | S | One extra LLM turn: "given the goal + last K steps, are we progressing? next move or give up?" Reuse the existing `chat_with_logprobs`. |
| Trigger logic | S | Every N steps, or on repeated/errored tool calls. Repeat-detection helper (`_same_call`) already exists. |
| Feed result back | S | Append reflection as a system/assistant note to `messages`; the loop already carries history. |
| New `stop_reason` | XS | Add `"gave_up"` alongside `done`/`budget_exhausted`; `_finish` already takes a reason. |
| Persist it | S | `trajectory_steps` schema would want a `kind` (action vs reflection) column — a migration. Or log reflections as a step with a null `tool_name`. |
| Routing the reflection call | M | Which tier runs reflection? Cheapest-correct is *always local* (it's meta-reasoning, not the task) — but that's a design call worth stating. |
| Tests | M | Loop-detection scenario, give-up scenario; mirror `test_executor.py` fakes. |

**The real cost isn't code — it's the design questions** (this is the interview-worthy part):
1. **Cadence vs. cost.** Every-step reflection ~doubles LLM calls. Every-N-steps or
   trigger-on-stall (repeat/error) is the tradeoff — I'd trigger on *evidence of trouble*
   (repeated `_same_call`, consecutive `ok:false`), not a fixed clock.
2. **Which tier reflects?** Reflection routed like any step could escalate to cloud, but that's
   paying frontier price for meta-reasoning. Pin it local unless it, too, hedges.
3. **Reflection ≠ planning.** This adds *self-correction within* ReAct, not upfront
   decomposition. Cheaper than a planner and complementary — reflection catches "I'm stuck,"
   a planner prevents "no global structure." I'd add reflection first (higher ROI, less code).

**Framing:** "A day, because the executor is already a checkpointed state machine holding the
history and outcomes reflection needs — the work is a triggered extra turn plus a `stop_reason`
and a schema column. The hard part is the *policy* — when to reflect and at which tier — not the
plumbing."

---

## 6. Trade-offs & issues — the money section

Interviewers probe here hardest. Each of these is a real decision with a real "why."

| Decision | Chose | Rejected | Why |
|---|---|---|---|
| **Confidence** | Model-agnostic signals + Thompson | Logprob-only; energy MLP | Logprob misroutes on confident halluc/refusal; MLP needs per-model retraining + drifts as memory grows (violates the no-training NFR) |
| **Vector index** | FAISS `IndexFlatIP` (exact) | HNSW/IVF; USearch | Exact needed for dedup recall; approximate only pays at millions of vectors; USearch = new dep + divergence |
| **Vector DB** | SQLite BLOB + in-RAM FAISS | pgvector; Pinecone/Qdrant | Local-first/offline/private premise; a server DB breaks "it's just a file" |
| **Index freshness** | Incremental add + tombstone + compaction | Full rebuild per mutation | KB writes on *every* escalation → per-query O(n) rebuild was the hidden cost |
| **Knowledge writing** | LLM writes, deterministic code indexes | LLM-generated summaries as index | Summaries lose info + need an LLM call per update + can hallucinate the index |
| **Chunking** | Content-type-aware (AST / heading-section / boundary window) | Fixed-size blind window | Blind cuts mid-function/mid-sentence → fragment chunks retrieve poorly → incomplete answers |
| **Parsing** | docling → pymupdf/docx → skip-loud chain | Single hard-dep parser | Best layout-preserving parser if present; never crash on a missing lib or one bad file |
| **Hybrid search** | BM25 (FTS5) + vector + RRF | vector-only | Keyword recall for exact terms (identifiers, error codes) that embeddings miss |
| **Ingestion** | File-level thread pool (default 4) | Serial; process pool; native batch | Embeds are IO-bound → threads overlap them; DB writes stay on one thread (SQLite single-writer). Native batch is moot for local Ollama (one prompt at a time) |
| **Config** | Typed Pydantic + strict YAML validation | dict-typed config | Fail at load with a readable error, not deep in a run |

**Two "I was wrong" stories (these land well — show them I self-correct):**
1. **The FAISS rebuild framing** (§4.5): I called the KB "write-rarely" in my own doc; it's
   the opposite. Caught it in review, which changed the whole index design from
   rebuild-on-mutation to incremental-add + compaction.
2. **Weak-retrieval GSA priming** (§3.3): I assumed injecting retrieved hits into the GSA
   prompt always helps. Experiments (P9/P10) showed *weak* hits degrade the signal, so I
   made retrieval conditioning all-or-nothing above a score floor.

**Concurrency correctness detail worth mentioning:** the knowledge store shares one SQLite
connection between the main thread and a background `_learn` thread. CPython can segfault
if one sqlite3 connection is used from two threads at once, so every store method is guarded
by a re-entrant lock (`@_synchronized` on an `RLock`) — reentrant so a public method can
call another without deadlocking.

---

## 7. Results & honesty about them

- **~67%** of repetitive codebase/doc queries intercepted by local memory/RAG.
- **~70%** cost saved over 30 standard dev queries.
- **639 tests**, ~11.6K LOC.
- **Be upfront:** these are dev-run figures, not a controlled academic benchmark. The
  scientifically honest version is the planned papers — a proper AUROC/calibration study of
  the routing signals, and the non-stationary bandit analysis. GSA logprob routing showed
  AUROC ~0.65–0.83 depending on category, which is *useful but not a solved problem* — the
  multi-signal fusion is what makes it production-viable.

---

## 8. Likely interviewer questions + crisp answers

**Q: How do you know when to trust the local model?**
A: I don't rely on one signal. GSA pre-gate (1-token YES/NO from the logprob distribution)
catches refusals and known-unknowns before generating; logprob catches general uncertainty;
self-consistency (regenerate, compare claims) catches hallucination; knowledge-verification
catches factual contradictions. A per-category Thompson bandit sets the thresholds and adapts
online. No single signal is trusted alone because each has a known blind spot.

**Q: Why not just fine-tune a router / use a classifier?**
A: It would need per-model retraining and would drift as memory grows — every new model or
knowledge update invalidates it. My constraint was "works on a fresh model from day 1, no
training." A 6-parameter bandit gives online adaptation with zero training and built-in
exploration.

**Q: Why SQLite + FAISS instead of a real vector DB?**
A: Local-first is the product premise — offline, private, zero infra. A server DB (pgvector,
Pinecone) means shipping users a daemon, which contradicts "it's just a file." SQLite gives
transactional text+vector storage and FTS5 for BM25 for free; FAISS is an in-RAM accelerator
rebuilt from SQLite, so there's no persistence-sync bug. At millions of vectors I'd revisit
(sqlite-vec or LanceDB — still embedded).

**Q: What breaks at scale?**
A: Three things, in order: (1) the exact FAISS scan gets slow past ~1M vectors → switch to
HNSW/IVF; (2) the whole index must fit in RAM → quantization or memory-mapped USearch;
(3) no metadata filtering on doc search → need filtered-ANN (Qdrant-style or sqlite-vec). None
bite at a single-codebase scale, which is why I didn't build them speculatively.

**Q: What was the hardest bug / most surprising thing?**
A: The dedup-completeness failure (§4.5). The `valid_to` post-filter felt obviously correct —
it *is* correct for "don't return stale entries" — but it silently broke *completeness* under
churn, because dead vectors crowd out live ones in the top-k. It taught me to separate
correctness from completeness explicitly when reasoning about filtered search.

**Q: How does it actually get cheaper over time?**
A: Every cloud escalation is distilled by the learning extractor into a structured fact (or a
skill), written into the knowledge store, and indexed deterministically. The next similar
query hits Tier 0 (memory) or lets the local model answer with that context, so it never pays
the cloud twice for the same class of problem. The Thompson thresholds also relax for
categories where local keeps succeeding, so fewer things escalate at all.

**Q: If you rebuilt it, what would you change?**
A: (1) Start with the incremental-add + tombstone index instead of rebuild-on-mutation — I
took the "simple" path and it was wrong for the write pattern. (2) Design document metadata
filtering in from the start. (3) Instrument the cloud/local/memory split earlier — my whole
ROI thesis rests on step difficulty being separable, and I'd want that measured before
building the step-router, not after.

---

## 9. One-liners to have ready

- *"Cost scales with task length, not difficulty — Autodidact fixes that by only escalating
  the hard steps and remembering the answers."*
- *"The LLM writes knowledge but never indexes it — indexing is deterministic so it can't
  hallucinate the search layer."*
- *"Logprob measures token predictability, not correctness."*
- *"SQLite is the source of truth; FAISS is a rebuildable cache — no sync bug possible."*
- *"The post-filter guarantees correctness, not completeness — and dedup needs completeness."*
- *"Six Beta counters, not a neural net."*

---

## 10. Real measured results (cite these, they're from actual runs)

From `results/summary.json` and the benchmark suite — use these instead of vaguer README numbers when pressed:

- **Learning curve** (`learning_curve.json`, 200 queries): **final local-resolution rate 0.765**, only **47 total escalations** across 200 queries. The curve *starts at 0% local* (empty brain, everything escalates) and climbs to ~77% — that's the compounding thesis, measured. Knowledge count grows 1:1 with escalations (47), i.e. every escalation is retained.
- **Thompson vs baselines** (`thompson_calibration.json`, 500 queries): **Thompson 0.746 accuracy vs fixed-threshold 0.75 vs random 0.504.** Honest read: Thompson **matches** a well-tuned fixed threshold while **adapting online without knowing the right threshold in advance** — the fixed baseline had oracle tuning; Thompson earned it. Random (0.504) shows routing matters at all.
- **Ebbinghaus decay:** final precision **1.0 with decay vs 1.0 without** on this set — i.e. decay costs no accuracy while keeping memory lean. (Be honest: on this benchmark it didn't *help* precision; its value is memory hygiene, not accuracy.)

**How to frame the Thompson "tie":** *"The point wasn't to beat a hand-tuned threshold — it was to reach it without hand-tuning, online, per-category, on a fresh model. A fixed 0.75 only works because someone found 0.75; Thompson finds it per category and re-finds it when the model or memory changes."*

---

## 11. Deeper dives to have loaded (for the 60-min back-half)

The interviewer will pick 2-3 and go deep. Have these ready cold.

### 11.1 The learning-extractor path (how a cloud answer becomes memory)

Cloud escalation → `learning_extractor` pulls **structured facts** (not the raw blob) from the response → each fact gets `question`, `content`, `domain/topic/category`, an embedding → dedup check against existing memory → insert into STM. Next similar query retrieves it. **Why structured, not raw:** a raw transcript retrieves poorly (mixed topics, conversational filler dilute the embedding); atomized facts each embed cleanly and retrieve precisely. Trade-off: extraction is itself an LLM call, so it's part of escalation cost — amortized because it's paid once and reused forever.

### 11.2 Answer-embedding column (a forward-compat design bet)

`knowledge_entries` stores both `embedding` (question-side) and `answer_embedding`. v1 only searches question-side; the answer embedding is stored-but-unused. **Why store it now:** re-embedding the whole KB later (to experiment with answer-side or hybrid retrieval) costs a full re-seed; storing it at write time is nearly free. This is a deliberate "cheap now vs expensive later" bet — the kind of forward-compat call worth defending.

### 11.3 Mixed-embedding-dimension guard (a real failure mode I designed around)

If someone swaps the embedding model (e.g. nomic 768-dim → bge 1024-dim) without re-seeding, the store would contain mixed-dim vectors — FAISS can't build an index over them and cosine between different-dim vectors is meaningless. Instead of a cryptic numpy broadcast error deep in a search, I **detect it on insert and on index build** and raise a `MixedEmbeddingDimensionError` with recovery instructions. Lesson: *fail loud and early at the boundary, with an actionable message.*

### 11.4 Per-consumer similarity thresholds (one knob, many appetites)

`search(min_similarity=...)` is a per-call override, not a global config mutation. Different consumers want different floors: GSA wants strong-hits-or-nothing (0.70+), answer-injection wants medium (0.60), and "knowledge similarity as a routing feature" wants raw top-k (0.0). **Why per-call:** mutating a shared config for each consumer is a race condition waiting to happen (the background learn thread shares the store). Passing the floor per call keeps it stateless and thread-safe.

### 11.5 Backend abstraction (Ollama / OpenAI-compatible / Bedrock)

One `ChatBackend` interface, three adapters. The routing/executor code is provider-agnostic — it asks for `chat_with_logprobs` and gets a uniform result. **The sharp edge:** not all providers return logprobs (Bedrock doesn't expose them the same way), so GSA has a **hard-label fallback** (raw YES/NO → 1.0/0.0) when the softmax path is unavailable. Design principle: degrade the *quality* of a signal gracefully rather than making the signal a hard dependency on a provider feature.

### 11.6 Ingestion concurrency (the change I shipped this cycle)

Ingesting a big codebase was fully serial — one embed call per chunk in a Python loop. I added a **file-level `ThreadPoolExecutor`**: read+chunk+embed run on worker threads, but **all SQLite writes stay on the calling thread** (SQLite is single-writer; `pool.map` preserves order so `chunk_index` and dedup stay deterministic). Default 4 workers, configurable via `ingest.workers` with a `--workers` CLI override. **The honest caveat I'd volunteer:** against a *local* Ollama on one GPU the win is modest (the GPU serializes the actual embedding compute; threads only overlap HTTP/parse overhead) — the big win would be against a cloud embedding API. I built it anyway because it's correct and cheap, and it future-proofs the cloud-embedding path.

---

## 12. Where the whole thesis could be wrong (steelman the critique)

Have an answer for the person who thinks the project is misguided:

- **"Just use a bigger local model — no routing needed."** Valid if a 14B/32B runs on the user's hardware and hits the quality bar. Routing wins when the local model is *cheap but imperfect* and the cloud is *expensive but authoritative* — the middle regime. If hardware gets cheap enough, the routing ROI shrinks. My bet is that the frontier keeps moving, so there's always a cheap/expensive gap to arbitrage.
- **"Step difficulty isn't separable."** The core risk from §1. If every step needs whole-task context, per-step routing collapses to whole-task routing. I mitigate with the GSA pre-gate but can't fully eliminate it — I'd measure the cloud/local/memory split on real tasks before over-investing.
- **"Learned knowledge goes stale / gets poisoned."** A wrong cloud answer becomes permanent wrong memory. Ebbinghaus decay helps (unused facts fade) but doesn't catch confident-but-wrong facts that keep getting used. This is exactly why **Paper B includes a poisoning-robustness experiment** — I know it's a hole.
- **"Retrieval is the bottleneck, not routing."** If retrieval misses, both knowledge-similarity and GSA lose signal (they're retrieval-conditional). The ablation stratifies by `retrieval_recall_at_5` to separate "routing is bad" from "retrieval is bad." Honest: retrieval quality is a real confound in my numbers.

---

## 13. 60-minute time budget (how to pace it)

| Time | Segment | What to cover |
|---|---|---|
| 0-5 | Pitch + problem | §0, §1 — get the compounding thesis across fast |
| 5-12 | Architecture | §2 whiteboard, dispatch, the three tiers |
| 12-22 | **Systems & infra** | §2.5 — state machine + durable replay (lead here for a systems interviewer), hybrid index, resilience, telemetry |
| 22-35 | Routing deep-dive | §3 — GSA, multi-signal, Thompson; the intellectual core |
| 35-48 | Memory/retrieval OR execution | §4 or §5 depending on their interest; §11 dives on demand |
| 48-56 | Trade-offs + results | §6 decision table, §7/§10 honest numbers, the two "I was wrong" stories |
| 56-60 | "What would you change / what's next" | §12 steelman, §13 wrap, roadmap |

**Tailor the front-load to the role:** for a distributed-systems/agentic-runtime interviewer,
spend the 12-22 block on §2.5.1 (state machine, WAL, event-sourced replay, resume) — it's the
most transferable. For search-infra, expand §2.5.2 + §4. For MLOps, expand §2.5.4 + the research
doc's eval methodology.

**If they only remember three things, make them:** (1) cost scales with difficulty not length + compounding; (2) knowing-what-you-don't-know via model-agnostic multi-signal routing, no training; (3) the dedup correctness-vs-completeness insight (proves you reason rigorously about edge cases).

**For a systems interviewer specifically**, the three become: (1) crash-safe resumable state machine — every ReAct step is a committed event, resume is deterministic replay of the log; (2) SQLite-as-truth + FAISS-as-rebuildable-cache (no sync-bug class) with hybrid FTS5+vector+RRF retrieval; (3) client-side resilience tuned from a real incident (69-throttle Bedrock burst → the backoff schedule), plus per-step cost telemetry driving the savings metric.
```

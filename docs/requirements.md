# Autodidact — Full Vision Requirements

**Purpose:** This is the master spec. It captures the entire roadmap from v1.0 through Phase 4 so nothing gets lost across sessions. Each phase has its own detailed spec when it's time to build; this document is the index and the contract for what each phase delivers.

**Last updated:** 2026-05-03

**Source documents:** VISION.md, ROADMAP.md, CONTEXT.md, COUNCIL_ARTICLES_DEBATE.md, COUNCIL_MEMPALACE_DEBATE.md, paper/COMPASS.md

---

## The Vision

Autodidact works like a new employee. On day one, it asks a lot of questions. By week two, it handles most tasks independently. By month three, it's the expert. Every interaction makes it smarter. Every escalation makes it more capable. It never forgets what it learned.

---

## Phase 1: The Self-Learning Agent

### v1.0 — Core Product (current, shipping)

**Spec:** `.kiro/specs/autodidact-product-v1/`
**Status:** Tasks 1-4 complete, Tasks 5-6 (packaging/launch) remaining

**What ships:**
- Python SDK: `from autodidact import Agent`
- CLI: `autodidact init`, `autodidact chat`, `autodidact query`, `autodidact savings`, `autodidact memory`
- Zero-friction setup wizard: auto-detect/install Ollama, auto-pull models, cloud provider presets
- Three setup modes: local+cloud (default), cloud+cloud (no Ollama needed), local-only
- Two-stage confidence routing: memory check (pre-gen) → logprob_uncertainty (post-gen)
- Cloud-to-cloud routing: cheap cloud ↔ expensive cloud, no local model required
- Document ingestion: `autodidact learn <path>` — chunk text files, embed, store for RAG context (cold start fix)
- Knowledge store: SQLite + FAISS, learns from every cloud escalation
- LearningExtractor: structured knowledge extraction from cloud responses via local LLM
- Progress callbacks: real-time `on_progress` events during query processing
- Visible learning UX: [THINKING], [MEMORY], [LOCAL], [CLOUD], [LEARNED] tags
- Cost tracking: per-query and cumulative savings
- Multi-provider: Ollama (local), OpenAI-compatible + Bedrock (cloud), OpenRouter, DeepSeek
- Config: YAML file, env var overrides, cloud provider presets

**What's validated (v0.1 experiments, $123, 11 experiments):**
- logprob_uncertainty AUROC 0.65-0.83 across 3 models × 2 datasets
- Zero-shot signals beat supervised baselines on cross-dataset transfer
- Knowledge store with 1000 entries → 89% in-category retrieval recall
- Answer quality +12pp when retrieval context available

**Acceptance criteria for v1.0 launch:**
- "Magic moment" (answer from memory) within first 10 interactions
- Setup to first query in under 3 minutes (Ollama pre-installed)
- After 100 domain-specific interactions, 40%+ queries answered from memory or locally
- 92+ tests passing (currently 110)

### v1.1 — Memory Intelligence (planned, months 2-3)

**What ships:**

#### R1.1.1: Tiered Memory with Consolidation
```
HOT:  Working Memory    → conversation history (in-session, RAM)
WARM: Short-Term Memory → recent escalation answers (SQLite+FAISS)
COOL: Long-Term Memory  → proven knowledge, promoted after repeated use
COLD: Archive           → rarely-used, dumped to JSONL, removed from FAISS, restorable
```
- Memory lifecycle: new → STM → (used N times) → LTM → (unused T days) → Archive → (accessed) → STM
- `autodidact maintain` consolidation job: deduplicate, compress clusters, promote/demote, validate stale, report
- Provenance tracking on consolidation operations (which entries merged/compressed) — per council debate on Xu et al.

#### R1.1.2: Contradiction Detection
- When cloud gives a different answer to a question already in memory, flag it
- Optionally ask user which is correct
- Store correction history (old answer + new answer + timestamp), never overwrite — per council debate on memory poisoning

#### R1.1.3: Skill Extraction
- After cloud escalation, extract PROCEDURES (not just facts) from the response
- Store as reusable skills: name, description, ordered steps
- Skills have a `validated` flag — set True after successful use
- Consider structured triplet extraction (Memoria-style) piggybacking on existing cloud calls — no extra LLM cost

#### R1.1.4: GSA v3 Pre-Generation Filter
- Use Grounded Self-Assessment as a fast pre-generation confidence check
- Skip full local generation when GSA is very confident or very uncertain
- Reduces latency on easy queries (no need for full generation + logprob check)

#### R1.1.5: Self-Verification System
- Periodic re-testing of stored knowledge against local model
- Self-test questions generated at extraction time (or default: "Is the following true? {fact}")
- LLM-as-judge for contradiction detection
- Flag stale entries, trigger re-verification with cloud
- Time-based and query-count-based triggers
- Reference: `prototype/src/components/self-verification.ts`

#### R1.1.6: Context Builder (Layered Prompts)
- L0: Always loaded — identity preamble
- L1: Always loaded — critical facts + user profile summary
- L2: Loaded when query matches known domain/topic
- L3: Loaded when confidence is low — full knowledge + skills dump
- Token budgets per layer to prevent prompt bloat
- Reference: `prototype/src/components/context-builder.ts`

#### R1.1.7: Integrations
- LangChain / LlamaIndex drop-in: `AutodidactLLM` wrapper
- OpenAI-compatible proxy mode: `autodidact serve` (works with Cursor, Aider, any tool)
- MCP server: 5-8 tools (query, search_memory, savings, memory_stats, learn_document) as primary integration surface — per MemPalace council debate. One integration surface vs many separate plugins.
- External retriever hook: `agent = Agent(..., retriever=my_func)`

#### R1.1.8: Multi-Turn Conversation Memory
- Conversation sessions persisted across restarts
- Session summarization for long conversations
- Session-scoped memory (what was discussed in this session)
- Agent diary: persistent stream of observations, findings, patterns — per MemPalace council debate. Used for self-verification logging and session summaries.

#### R1.1.9: Code-Aware Document Ingestion
- `autodidact learn --code <repo>` — AST-aware code ingestion via GitNexus MCP
- Function-level chunking instead of text-level chunking
- Dependency graph, call chains, import resolution
- Advanced document parsing: tables, images from PDFs (via Unstructured.io or LlamaIndex)
- Incremental re-indexing: detect changed files, re-chunk only those
- Web page ingestion: URL fetching + HTML parsing
- Reference: [GitNexus](https://github.com/abhigyanpatwari/GitNexus) — code knowledge graph with 14-language AST parsing, MCP server, and agent skills. Consider MCP integration rather than rebuilding.

#### R1.1.10: Scoped Search
- Wire up domain/topic/category filtering in agent query flow — per MemPalace council debate
- Our KnowledgeStore already has `KnowledgeScope(domain, topic, category)` and scoped search infrastructure
- Currently unused in Agent._check_memory() — flat search across everything
- As KB grows, scoped search improves both speed and relevance

---

## Phase 2: The Hive — Collective Intelligence Network (months 6-18)

**Status:** Planned (after v1.0 ships and has users)

Agents teaching agents. A decentralized marketplace where agents discover experts, acquire skills, and trade knowledge.

### R2.1: Agent Registry
- Discovery service for finding expert agents by skill/domain
- Agents register their capabilities (domains, skills, knowledge areas)
- Search by skill, domain, or query similarity

### R2.2: Skill Transfer Protocol
- DISCOVER → NEGOTIATE → TRANSFER → VALIDATE pipeline
- Transfer methods:
  - Prompt chain export/import (lightweight, works immediately)
  - Knowledge graph subset sync (structured knowledge transfer)
  - LoRA adapter download (deep skill transfer, requires GPU)
  - Live tutoring sessions (streaming Q&A between agents)

### R2.3: Knowledge Tokens
- Credit system for knowledge exchange
- Agents earn tokens by teaching, spend tokens by learning
- Prevents free-riding in the network

### R2.4: Reputation System
- Agents rated by teaching success rate
- Knowledge from low-reputation agents gets quarantined (not blindly trusted)
- Reputation-weighted trust for knowledge acceptance — per council debate on memory poisoning

### R2.5: Security — Memory Poisoning Defense
- Provenance tracking on all transferred knowledge (which agent, when, what context)
- Validation pipeline for incoming knowledge (test against local model before accepting)
- Quarantine for unverified knowledge from new/low-reputation agents
- Audit trail for knowledge lineage
- Reference: Xu et al. "Agentic Memory is a Memo" — persistent memory poisoning analysis

---

## Phase 3: The Organization — Hierarchical Agent Network (months 18-36)

**Status:** Vision

Multi-tier agent hierarchy that mirrors how human organizations work.

### R3.1: Tiered Agent Hierarchy
| Human Role | Agent Tier | Model Size | Scope |
|---|---|---|---|
| Intern | Specialist Agent | 3B | Narrow, routine queries |
| Junior Engineer | Domain Agent | 7B | Team-level knowledge |
| Senior Engineer | Senior Agent | 70B | Org-level knowledge |
| Principal Engineer | Principal Agent | Frontier | World knowledge, novel problems |

### R3.2: N-Tier Routing
- Same confidence routing from Phase 1, generalized to N tiers
- Each tier escalates UP when uncertain
- Knowledge flows DOWN after escalation (senior teaches junior)
- Cost optimization: most queries handled by cheapest tier

### R3.3: Organizational Properties
- Fault tolerance: if a tier is unavailable, queries route to next tier
- Organic growth: new specialist agents added without restructuring
- Knowledge distribution: specialized knowledge lives where it's needed

---

## Phase 4: Continuous Self-Improvement (months 24+, research)

**Status:** Research exploration

Moving beyond memory-based learning (episodic lookup) to actual model improvement (parametric learning). This is where Autodidact stops being a "memo system" (per Xu et al.) and becomes a true learning system.

### R4.1: Consolidation Pipeline (Episodic → Parametric)
- Periodic LoRA fine-tuning on accumulated knowledge store
- The knowledge store becomes training data for the local model
- Three design principles from Xu et al. "Agentic Memory is a Memo":
  1. Treat agentic memory as episodic lookup (don't expect it to generalize) ✅ already true
  2. Build a consolidation pathway from episodic store to parametric memory
  3. Make consolidation safe (provenance, versioned checkpoints, regression guards)

### R4.2: Consolidation Safety
- **Provenance tracking:** which knowledge entries contributed to which LoRA
- **Versioned checkpoints:** rollback if fine-tuning degrades performance
- **Regression guards:** test suite that catches capability loss after fine-tuning
- **Validation set:** held-out queries to measure before/after accuracy

### R4.3: Online Continual Learning
- Real-time learning from every interaction (not just periodic batch)
- Catastrophic forgetting mitigation (EWC, replay buffers, progressive expansion)
- Reference: Xu et al. Appendix D — comparison of continual learning methods

### R4.4: Harness Evolution (AHE-inspired)
- Tools, middleware, and memory evolve automatically
- Agent discovers and learns new tools from cloud escalations
- Tool registry with verification and decay
- Reference: `prototype/src/components/tool-registry.ts`, `prototype/src/components/skill-evolver.ts`

### R4.5: Compositional Generalization Benchmark
- Measure compositional generalization over time (CGT), not just recall accuracy
- Does the fine-tuned model handle novel concept combinations better than retrieval-only?
- Per Xu et al. Theorem 1: retrieval needs Ω(k²) examples for compositional tasks; parametric learning needs O(d/δ)
- This is the benchmark that proves Phase 4 actually works

---

## Cross-Cutting Concerns

### Security
| Phase | Threat | Mitigation |
|---|---|---|
| v1.0 | User correction injection | Store correction history, don't overwrite |
| v1.1 | Stale knowledge serving wrong answers | Self-verification system, staleness indicators |
| v2.0 | Memory poisoning via agent-to-agent transfer | Provenance tracking, reputation-weighted trust, quarantine |
| Phase 4 | Poisoned training data for LoRA | Provenance tracking, regression guards, validation sets |

### Honest Terminology
- v1.0 "learning" = storing Q&A pairs from cloud escalations (episodic memory)
- v1.1 "learning" = structured extraction + skill storage + self-verification
- Phase 4 "learning" = actual model weight changes (parametric memory)
- CONTEXT.md clarifies this internally. Marketing uses "learning" for all phases. UX makes the distinction visible via [MEMORY] vs [LOCAL] tags.

### What the Articles Taught Us (Council Debate, 2026-05-03)
- **Memoria (2512.12686):** Validates decay-weighted retrieval. KG triplets interesting for v1.1 skill extraction but add LLM cost. No v1.0 changes.
- **Xu et al. "Agentic Memory is a Memo" (2604.27707):** Autodidact v1.0 IS a memo system by their taxonomy. Acceptable for v1.0. The Generalization Gap theorem (Ω(k²) for retrieval vs O(d/δ) for parametric) provides theoretical justification for Phase 4. Consolidation safety principles (provenance, versioning, regression guards) added to Phase 4 requirements.

### What MemPalace Taught Us (Council Debate, 2026-05-03)
- **Architecturally aligned:** Their wings/rooms/halls = our domain/topic/category. Their temporal validity = our valid_from/valid_to. Independent convergence.
- **Temporal KG:** Their entities + triples tables with temporal validity is the reference design for v1.1 triple extraction (R1.1.3).
- **MCP-first distribution:** 29 MCP tools as one integration surface. Added to v1.1 R1.1.7.
- **Agent diaries:** Persistent working notes for self-verification logging and session summaries. Added to v1.1 R1.1.8.
- **Scoped search:** Their wing/room filtering maps to our domain/topic scoping. Infrastructure exists, needs wiring. Added as v1.1 R1.1.10.
- **Benchmarking discipline:** Committed per-question results, reproducible commands. Model for v1.0 launch.
- **Retrieval precision benchmark:** Need to measure "when agent says [MEMORY], is the answer correct?" Added to v1.0 quality benchmark plan.
- Reference: `COUNCIL_MEMPALACE_DEBATE.md`

### What GitNexus Taught Us (Analysis, 2026-05-03)
- **Code knowledge graph:** AST parsing → dependency graph → call chains → clustering → execution flows. 14 languages, Tree-sitter, MCP server with 16 tools.
- **Not for v1.0:** Building our own AST parser is months of work. GitNexus already exists.
- **v1.1 integration path:** `autodidact learn --code <repo>` delegates to GitNexus MCP rather than rebuilding. Added to v1.1 R1.1.9.
- **Key insight:** "Traditional Graph RAG gives LLM raw edges and hopes it explores enough. GitNexus precomputes structure at index time." This is the right approach for code intelligence — precompute, don't explore at query time.
- Reference: [GitNexus](https://github.com/abhigyanpatwari/GitNexus)

---

## Existing Specs Index

| Spec | Scope | Status |
|---|---|---|
| `.kiro/specs/autodidact-framework/` | v0.1 experiment (confidence ablation) | ✅ Complete |
| `.kiro/specs/autodidact-demo-prototype/` | Vietnam AI Stars demo | ✅ Complete |
| `.kiro/specs/autodidact-product-v1/` | v1.0 product (CLI agent) | 🔧 In progress (tasks 5-6 remaining) |
| `.kiro/specs/autodidact-full/` | Full vision (this document) | 📋 Living document |

---

## Prototype Components Reference

The TypeScript prototypes (`demo-prototype/`, `prototype/`) contain reference implementations for future phases:

| Component | Location | Target Phase |
|---|---|---|
| LearningExtractor | `demo-prototype/src/learning-extractor.ts` | ✅ Ported to Python (v1.0) |
| Progress callbacks | `demo-prototype/src/agent.ts` | ✅ Ported to Python (v1.0) |
| SkillStore | `prototype/src/components/skill-store.ts` | v1.1 |
| SkillEvolver | `prototype/src/components/skill-evolver.ts` | v1.1 / Phase 4 |
| SkillFormat (MD export/import) | `prototype/src/components/skill-format.ts` | v1.1 |
| SelfVerification | `prototype/src/components/self-verification.ts` | v1.1 |
| ContextBuilder (L0-L3) | `prototype/src/components/context-builder.ts` | v1.1 |
| ToolRegistry | `prototype/src/components/tool-registry.ts` | Phase 4 |
| UserProfile | `prototype/src/components/user-profile.ts` | v2.0 |
| MetricsTracker | `prototype/src/components/metrics-tracker.ts` | v1.1 |

---

## Timeline

| Phase | Timeline | Milestone |
|---|---|---|
| v1.0 | May 2026 (weeks 1-4) | Open source launch, first users |
| v1.1 | June-July 2026 (months 2-3) | Memory intelligence, skill extraction, integrations |
| v2.0 | Nov 2026-May 2027 (months 6-18) | Hive network beta, skill marketplace |
| v3.0 | May 2027-May 2028 (months 18-36) | Enterprise hierarchical deployment |
| Phase 4 | 2028+ (months 24+) | Research: LoRA consolidation, online learning |

## How Phases Build on Each Other

```
Phase 1 v1.0: Single agent learns from cloud escalations (episodic memory)
         ↓ (foundation: routing + KB + confidence signals)
Phase 1 v1.1: Tiered memory, skill extraction, self-verification
         ↓ (richer memory: skills + validated knowledge)
Phase 2: Agents learn from each other via skill transfer
         ↓ (network effect: knowledge marketplace)
Phase 3: Agents organized in cost-optimized hierarchies
         ↓ (enterprise scale: N-tier routing)
Phase 4: Agents improve their own weights from accumulated knowledge
         (research frontier: episodic → parametric consolidation)
```

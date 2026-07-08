# Autodidact Roadmap

## Vision

Autodidact is a self-learning AI agent that makes any AI system cheaper and smarter over time. Route easy queries to a cheap model, escalate hard ones to an expensive model, and learn from every escalation so the system handles more locally as it grows.

**Updated 2026-04-30** after v0.1 experiments validated the core mechanism across 3 model families and 2 datasets. See `VISION.md` for the full narrative and `results/experiment/v0.1_report.md` for the evidence.

---

## Phase 1: The Self-Learning Agent (Current — shipping v1.0)

**Status:** Core mechanism validated. Building the product.

The foundation — a framework for building AI systems with intelligent routing and learning from escalations.

### What's Validated (v0.1 experiments, $123, 11 experiments)
- **logprob_uncertainty** is the dominant routing signal: AUROC 0.65-0.83 across 3 models × 2 datasets, zero training cost
- Zero-shot signals beat supervised baselines on cross-dataset transfer (RouteLLM collapses on new datasets; logprob doesn't)
- Retrieval-conditional self-assessment (GSA v3) improves calibration when KB is dense enough
- Knowledge store with 1000 entries gives 89% in-category retrieval recall
- Answer quality improves +12pp when retrieval context is available

### What's Shipping (v1.0)
- Python SDK: `from autodidact import Agent`
- CLI: `autodidact init`, `autodidact chat`, `autodidact savings`
- Zero-friction setup wizard (auto-detect/install Ollama, pull models, configure cloud)
- logprob-based confidence routing (local ↔ cloud)
- Knowledge store that learns from every cloud escalation
- Cost dashboard showing real savings
- LangChain / LlamaIndex drop-in integration
- OpenAI-compatible proxy mode (works with Cursor, Aider, any tool)
- Cloud-to-cloud routing mode (no local model required)

### What We Learned Doesn't Work (Yet)
- Naive multi-signal fusion (mean of 6 signals) is worse than the best single signal
- Knowledge similarity as a routing signal is structurally inverted on MCQ benchmarks
- Thompson Sampling fusion without a feedback loop collapses to naive mean
- Energy scorer needs online learning to activate (deferred to v1.1)

### v1.1 Planned: Memory Intelligence
The "human memory" system — tiered memory with consolidation:
- **Working Memory** → conversation history (in-session)
- **Short-Term Memory** → recent escalation answers (SQLite+FAISS, decays via Ebbinghaus)
- **Long-Term Memory** → proven knowledge, promoted after repeated use
- **Cold Archive** → rarely-used knowledge archived to disk files, removed from FAISS to save RAM, restorable on access
- **Consolidation job** (`autodidact maintain`) — deduplicate, compress related entries, promote/demote tiers, validate stale entries
- **Contradiction detection** — flag when cloud gives a different answer than what's in memory
- **Skill extraction** — extract reusable procedures from cloud escalations, not just answers
- **External retriever hook** — plug in your own RAG pipeline alongside the agent's internal memory

### Infrastructure (built during v0.1)
- `autodidact/llm_client.py` — Ollama + Bedrock + OpenAI-compatible with retry logic
- `autodidact/knowledge_store.py` — SQLite + FAISS with Ebbinghaus decay, mixed-dim detection
- `autodidact/confidence_evaluator.py` — logprob_uncertainty + Thompson Sampling
- `autodidact/signals/grounded_self_assessment.py` — GSA v3 retrieval-conditional
- 55 tests passing

---

## Phase 2: The Hive — Collective Intelligence Network

**Status:** Planned (after v1.0 ships and has users)

Agents teaching agents. A decentralized marketplace where agents discover experts, acquire skills, and trade knowledge.

### Core Concepts
- **Agent Registry** — discovery service for finding expert agents by skill/domain
- **Skill Transfer Protocol** — DISCOVER → NEGOTIATE → TRANSFER → VALIDATE
- **Knowledge Tokens** — credit system for knowledge exchange
- **Course Builder** — agents package expertise into transferable skill packages
- **Reputation System** — agents rated by teaching success rate

### Skill Transfer Methods
- Prompt chain export/import (lightweight, works today)
- Knowledge graph subset sync (structured knowledge transfer)
- LoRA adapter download (deep skill transfer, requires GPU)
- Live tutoring sessions (streaming Q&A between agents)

---

## Phase 3: The Organization — Hierarchical Agent Network

**Status:** Vision

Multi-tier agent hierarchy that mirrors how human organizations work.

| Human Organization | Agent Organization |
|---|---|
| Intern | Specialist Agent (3B model, narrow scope, cheap) |
| Junior Engineer | Domain Agent (7B model, team-level knowledge) |
| Senior Engineer | Senior Agent (70B model, org-level knowledge) |
| Principal Engineer | Principal Agent (frontier model, world knowledge) |

### How It Works
- Each tier runs on hardware appropriate to its scope
- Agents escalate UP the hierarchy when uncertain (same confidence routing as Phase 1)
- Knowledge flows DOWN after escalation (senior teaches junior)
- Lower tiers handle high-volume, routine queries cheaply
- Upper tiers handle complex, novel queries with more compute

### Key Properties
- **Cost optimization** — most queries handled by cheap lower-tier agents
- **Knowledge distribution** — specialized knowledge lives where it's needed
- **Fault tolerance** — if a tier is unavailable, queries route to the next tier
- **Organic growth** — new specialist agents can be added without restructuring

---

## Phase 4: Continuous Self-Improvement (Research)

**Status:** Research exploration

Moving beyond memory-based learning to actual model improvement.

### Learning Levels
1. **Knowledge retrieval** (Phase 1) — "I remember the answer"
2. **Skill/procedure replay** (Phase 1) — "I remember how to do this"
3. **Harness evolution** (Phase 4) — tools, middleware, and memory evolve automatically (inspired by AHE — Agentic Harness Engineering)
4. **Periodic LoRA fine-tuning** (Phase 4) — "I've internalized this domain"
5. **Online continual learning** (Phase 4) — "I learn in real-time from every interaction"

### Research Questions
- Can we fine-tune a local model on its own accumulated knowledge store?
- How do we prevent catastrophic forgetting during incremental fine-tuning?
- Can the Hive network distribute fine-tuning across agents?
- What's the optimal balance between memory-based and weight-based learning?
- Can harness evolution (AHE-style) be applied online, not just on benchmarks?

---

## Timeline

| Phase | Timeline | Milestone |
|---|---|---|
| Phase 1 v1.0 | Weeks 1-4 (May 2026) | Open source launch, first users |
| Phase 1 v1.1 | Months 2-3 | Adaptive routing, multi-turn memory |
| Phase 2 | Months 6-18 | Hive network beta, skill marketplace |
| Phase 3 | Months 18-36 | Enterprise hierarchical deployment |
| Phase 4 | Months 24+ | Research collaboration, LoRA integration |

## How Phases Build on Each Other

```
Phase 1: Single agent learns from cloud escalations
         ↓ (foundation: routing + KB + confidence signals)
Phase 2: Agents learn from each other via skill transfer
         ↓ (network effect: knowledge marketplace)
Phase 3: Agents organized in cost-optimized hierarchies
         ↓ (enterprise scale: N-tier routing)
Phase 4: Agents improve their own weights from accumulated knowledge
         (research frontier: online learning)
```

Each phase uses the infrastructure from the previous phase. Phase 1's routing generalizes to N-tier in Phase 3. Phase 1's KB becomes the training data for Phase 4's fine-tuning. Phase 2's skill transfer uses Phase 1's knowledge store format.

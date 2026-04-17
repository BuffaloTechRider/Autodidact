# Autodidact Roadmap

## Vision

Autodidact is a self-learning, local-first AI agent framework that mimics how humans learn and work. The long-term vision spans three phases: individual learning, collective intelligence, and organizational hierarchy.

---

## Phase 1: The Self-Learning Agent (Current)

**Status:** In development

The foundation — a single agent that learns from every interaction.

### Core
- Multi-signal confidence evaluation (5 signals + Thompson Sampling)
- LOCAL/CLOUD routing with learning from every escalation
- Hierarchical knowledge store (domain/topic/category) with temporal validity
- STM/LTM memory tiers with Ebbinghaus decay
- Procedural skill store with versioning and self-improvement
- Autonomous tool discovery from cloud escalations
- Multi-turn conversation with per-turn routing
- User/team profile personalization

### Infrastructure
- Python core with ChromaDB/FAISS hybrid search
- TypeScript SDK for developer ecosystem
- Benchmark suite (LongMemEval, learning curve, Thompson calibration)
- AWS Bedrock + OpenAI-compatible cloud routing

### Novel Contributions
1. Thompson Sampling for agent routing (no published precedent)
2. Ebbinghaus-inspired STM→LTM memory consolidation
3. Self-improving procedural memory with closed-loop optimization
4. Learning-from-escalation loop (escalate once, learn forever)
5. Energy scorer that bootstraps from the agent's own learning history

---

## Phase 2: The Hive — Collective Intelligence Network

**Status:** Planned

Agents teaching agents. A decentralized marketplace where agents discover experts, acquire skills, and trade knowledge.

### Core Concepts
- **Agent Registry** — discovery service for finding expert agents by skill/domain
- **Skill Transfer Protocol** — DISCOVER → NEGOTIATE → TRANSFER → VALIDATE
- **Knowledge Tokens** — credit system for knowledge exchange (teacher earns tokens when student passes validation)
- **Course Builder** — agents package their expertise into transferable skill packages
- **Reputation System** — agents rated by teaching success rate

### Skill Transfer Methods
- Prompt chain export/import (lightweight, works today)
- Knowledge graph subset sync (structured knowledge transfer)
- LoRA adapter download (deep skill transfer, requires GPU)
- Live tutoring sessions (streaming Q&A between agents)

### Token Economics
- Teacher earns tokens when student passes validation test
- Tokens redeemable for compute credits or API access
- Reputation score based on teaching success rate
- No blockchain — simple centralized ledger with audit log

### Architecture
- HTTP/WebSocket REST API for agent-to-agent communication
- SQLite + signed receipts for token ledger
- HuggingFace Hub or local filesystem for LoRA storage
- Compatible with agentskills.io open standard

---

## Phase 3: The Organization — Hierarchical Agent Network

**Status:** Vision

Multi-tier agent hierarchy that mirrors how human organizations work.

### The Analogy
```
Human Organization          Agent Organization
─────────────────          ──────────────────
Intern                  →  Specialist Agent (3B model, narrow scope, cheap)
Junior Engineer         →  Domain Agent (7B model, team-level knowledge)
Senior Engineer         →  Senior Agent (70B model, org-level knowledge)
Principal Engineer      →  Principal Agent (frontier model, world knowledge)
```

### How It Works
- Each tier runs on hardware appropriate to its scope
- Agents escalate UP the hierarchy when uncertain (same confidence routing as Phase 1)
- Knowledge flows DOWN after escalation (senior teaches junior)
- Each agent has its own knowledge store scoped to its domain
- Lower tiers handle high-volume, routine queries cheaply
- Upper tiers handle complex, novel queries with more compute

### Key Properties
- **Cost optimization** — most queries handled by cheap lower-tier agents
- **Knowledge distribution** — specialized knowledge lives where it's needed
- **Fault tolerance** — if a tier is unavailable, queries route to the next tier
- **Organic growth** — new specialist agents can be added without restructuring
- **Mentorship** — senior agents actively improve junior agents' knowledge

### Enterprise Use Cases
- **Engineering org**: Intern agents per microservice, senior agent for architecture decisions
- **Legal firm**: Specialist agents per practice area, senior agent for cross-practice questions
- **Customer support**: Tier-1 agents for common questions, escalation to human-assisted agents for complex cases

### Technical Foundation
- Phase 1's LOCAL/CLOUD routing generalizes to N-tier routing
- Phase 2's Hive protocol enables agent-to-agent communication
- The confidence evaluator already supports extensible route interfaces
- Knowledge store scoping (domain/topic) maps to organizational boundaries

---

## Phase 4: Continuous Self-Improvement (Research)

**Status:** Research exploration

Moving beyond memory-based learning to actual model improvement.

### Learning Levels
1. **Knowledge retrieval** (Phase 1) — "I remember the answer"
2. **Skill/procedure replay** (Phase 1) — "I remember how to do this"
3. **Periodic LoRA fine-tuning** (Phase 4) — "I've internalized this domain"
4. **Online continual learning** (Phase 4) — "I learn in real-time from every interaction"

### Research Questions
- Can we fine-tune a local model on its own accumulated knowledge store?
- How do we prevent catastrophic forgetting during incremental fine-tuning?
- Can the Hive network distribute fine-tuning across agents?
- What's the optimal balance between memory-based and weight-based learning?

---

## Timeline (Estimated)

| Phase | Timeline | Milestone |
|-------|----------|-----------|
| Phase 1 | Months 1-6 | Open source launch, first enterprise design partners |
| Phase 2 | Months 6-18 | Hive network beta, skill marketplace |
| Phase 3 | Months 18-36 | Enterprise hierarchical deployment |
| Phase 4 | Months 24+ | Research collaboration, LoRA integration |

---

## How Phases Build on Each Other

```
Phase 1: Single agent learns from cloud
         ↓ (foundation)
Phase 2: Agents learn from each other
         ↓ (network effect)
Phase 3: Agents organized in hierarchies
         ↓ (enterprise scale)
Phase 4: Agents improve their own weights
         (research frontier)
```

Each phase uses the infrastructure from the previous phase. Phase 2's agent-to-agent communication uses Phase 1's learning loop. Phase 3's hierarchy uses Phase 2's Hive protocol. Phase 4's fine-tuning uses Phase 1's knowledge store as training data.

# Autodidact — Vision

**An AI that learns like a human.**

On day one, it asks a lot of questions. By week two, it handles most tasks independently. By month three, it's the expert. Every interaction makes it smarter. Every escalation makes it more capable. It never forgets what it learned.

*Updated 2026-04-30, grounded in v0.1 experiment results and council review.*

---

## The Human Analogy

When you encounter a question or task, you go through this sequence:

1. **Do I know the answer?** → Check your memory
2. **Do I know HOW to find it?** → Recall a procedure
3. **Am I confident I can do it?** → Self-assess
4. **If yes** → Do it yourself (free, fast)
5. **If no** → Ask someone smarter (costs time, but you get the right answer)
6. **Remember what you learned** → Store the knowledge
7. **Remember HOW you did it** → Store the procedure
8. **Next time, start from step 1** → You're smarter now

A new employee does this every day. The more tasks they do, the more knowledgeable they become, and the fewer questions they need to ask. Eventually they become the expert others ask.

**Autodidact makes AI work the same way.**

## The Problem

Every AI system today is either:
- **Smart but expensive** — cloud LLMs (Claude, GPT-4o) answer everything well but cost $0.01-0.05 per query. At scale, that's thousands per month.
- **Cheap but limited** — local LLMs (Qwen, Llama, Mistral) are free but fail on hard queries.
- **Static** — neither gets better over time. The 1000th query costs the same as the 1st.

No system learns from its own experience. No system routes intelligently between cheap and expensive. No system gets cheaper as it gets smarter.

## The Solution

Autodidact is an AI agent that:

1. **Thinks first.** Checks its memory for similar past questions. If it's seen this before, it answers from memory — instantly, for free.

2. **Tries locally.** If no memory match, generates an answer with the local model and checks its own confidence. If confident, returns the answer.

3. **Asks when uncertain.** If not confident, escalates to a cloud model. Gets the right answer.

4. **Learns from every escalation.** Stores the question + answer. Next time a similar question comes in, it answers from memory.

5. **Gets smarter over time.** The more it's used, the more it knows, the fewer escalations it needs, the cheaper it gets.

## What We've Validated (v0.1 Experiments)

These aren't aspirational claims. They're measured results from 11 experiments across 3 model families and 2 datasets.

### Finding 1: Zero-shot confidence signals work

`logprob_uncertainty` — the average per-token log-probability of the model's own answer — predicts correctness at AUROC 0.65-0.83 across all models and datasets tested. It requires zero training data, zero per-model tuning, and zero setup cost.

| Dataset | qwen2.5:7b | llama3.1:8b | mistral:7b |
|---|---|---|---|
| MMLU-Pro (MCQ) | 0.714 | 0.650 | 0.678 |
| TriviaQA (open-ended) | 0.828 | 0.800 | 0.717 |

### Finding 2: Supervised routing doesn't generalize across datasets

RouteLLM-style supervised classifiers achieve 0.64-0.68 AUROC on the dataset they were trained on, but collapse to chance (0.51-0.56) on a different dataset. Our zero-shot signals work on both datasets at zero cost.

### Finding 3: The system gets cheaper over time

With a 1000-entry knowledge base, 89% of queries have a semantically relevant retrieved entry. Answer quality improves by +12 percentage points when retrieval context is available (p=0.012, n=100).

### Finding 4: Signal quality correlates with model calibration

Models with calibration-aware RLHF (Qwen) produce stronger logprob signals. Models without (Llama) benefit more from explicit self-assessment prompting. The framework adapts to both.

## Architecture

```
User Query
    ↓
┌──────────────────────────────────────────────────────┐
│                   Autodidact Agent                    │
│                                                      │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────┐  │
│  │  Document     │  │ Agent Memory │  │Conversation│  │
│  │  Store (RAG)  │  │  (learned    │  │  History   │  │
│  │  user's docs  │  │   from cloud)│  │  (session) │  │
│  └──────┬───────┘  └──────┬───────┘  └─────┬─────┘  │
│         │                 │                 │        │
│         └────────┬────────┴────────┬────────┘        │
│                  ▼                 ▼                  │
│         ┌──────────────────────────────┐             │
│         │      Combined Context        │             │
│         │  docs + memory + history     │             │
│         └──────────────┬───────────────┘             │
│                        ▼                             │
│         ┌──────────────────────────────┐             │
│         │    Confidence Evaluator      │             │
│         │  Stage 1: memory check       │             │
│         │  Stage 2: logprob routing    │             │
│         └──────┬──────────────┬────────┘             │
│           confident          uncertain               │
│                │                  │                   │
│         ┌──────▼──────┐   ┌──────▼──────┐           │
│         │ Local Model │   │ Cloud Model │           │
│         │  (Ollama)   │   │ (OpenAI/    │           │
│         │   FREE      │   │  Bedrock)   │           │
│         └─────────────┘   └──────┬──────┘           │
│                                  │                   │
│                           ┌──────▼──────┐           │
│                           │   LEARN     │           │
│                           │ Store Q&A   │           │
│                           │ in memory   │           │
│                           └─────────────┘           │
└──────────────────────────────────────────────────────┘
    ↓
Answer + [LOCAL] / [CLOUD] / [MEMORY] + Cost
```

### Agent Memory Tiers (v1.0 foundation, v1.1 full system)

```
HOT:  Working Memory    → conversation history (in-session, RAM)
WARM: Short-Term Memory → recent escalation answers (SQLite+FAISS)
COOL: Long-Term Memory  → proven knowledge, promoted after repeated use
COLD: Archive           → rarely-used, archived to disk, restorable

New entry → STM → (used N times) → LTM → (unused T days) → Archive
                                                              ↓
                                          (accessed again) → STM
```

v1.0 ships STM+LTM with Ebbinghaus decay. v1.1 adds cold archive, consolidation job, and contradiction detection.

## Product Principles

**1. Zero-friction setup.** `pip install autodidact && autodidact init` handles everything — detects/installs Ollama, pulls a model, configures cloud API keys. No docs required.

**2. Works with existing systems.** Autodidact is a drop-in replacement for your LLM client. If you use LangChain, swap `ChatOpenAI` for `AutodidactLLM`. If you use raw API calls, point them at Autodidact's proxy. Your RAG pipeline, your frontend, your prompts — all unchanged.

**3. Measurable value from day one.** Every query shows whether it was routed locally (free) or to cloud (paid). `autodidact savings` shows cumulative cost savings. The value is verifiable, not aspirational.

**4. Gets smarter, not just cheaper.** The knowledge store grows from every cloud escalation. The routing threshold adapts. The system doesn't just cache — it learns semantic patterns that transfer to similar-but-not-identical queries.

**5. Domain-agnostic.** Customer support, internal Q&A, coding, legal, medical triage — any domain where queries repeat and knowledge accumulates. No domain-specific code in the framework.

## Roadmap

### Phase 1: The Self-Learning Agent (v1.0 — shipping)

The foundation. A single agent that routes intelligently and learns from escalations.

**What ships:**
- Python SDK (`from autodidact import Agent`)
- CLI (`autodidact chat`, `autodidact init`, `autodidact savings`)
- Zero-friction setup wizard (auto-detect/install Ollama, pull models, configure cloud)
- logprob-based confidence routing (validated across 3 models, 2 datasets)
- Knowledge store with semantic retrieval (SQLite + FAISS)
- Retrieval-conditional self-assessment (GSA v3)
- Cost dashboard (per-session and cumulative savings)
- LangChain/LlamaIndex integration (`AutodidactLLM` drop-in)
- OpenAI-compatible proxy mode (works with any tool)
- Cloud-to-cloud routing mode (no local model required)

**What we validated:** logprob_uncertainty AUROC 0.65-0.83, retrieval improves answer quality by +12pp, zero-shot beats supervised routing on cross-dataset transfer.

### Phase 2: The Hive — Agents Teaching Agents (v2.0 — planned)

Agents that share knowledge with each other. A company's support agent teaches the sales agent about product features. A legal agent teaches the HR agent about compliance.

**Core concepts:**
- Agent Registry — discover expert agents by skill/domain
- Skill Transfer Protocol — DISCOVER → NEGOTIATE → TRANSFER → VALIDATE
- Knowledge Tokens — credit system for knowledge exchange
- Reputation System — agents rated by teaching success rate

### Phase 3: The Organization — Hierarchical Agent Network (v3.0 — vision)

Multi-tier agent hierarchy that mirrors how human organizations work.

- Specialist agents (3B models, narrow scope, cheap) handle routine queries
- Domain agents (7B models, team-level knowledge) handle department-specific queries
- Senior agents (70B models, org-level knowledge) handle cross-cutting queries
- Principal agents (frontier models) handle novel, complex queries

Each tier escalates UP when uncertain. Knowledge flows DOWN after escalation. The confidence evaluator from Phase 1 generalizes to N-tier routing.

### Phase 4: Continuous Self-Improvement (v4.0 — research)

Moving beyond memory-based learning to actual model improvement.

- Periodic LoRA fine-tuning on accumulated knowledge
- Online continual learning from every interaction
- Harness evolution (inspired by AHE — tools, middleware, and memory evolve automatically)

## What Changed From the Original Roadmap

The original roadmap (pre-v0.1 experiments) assumed:
- Multi-signal Thompson Sampling fusion would be the core mechanism
- 6 signals would be better than 1
- Knowledge similarity would be a strong signal
- The product would be a coding agent

**What the experiments taught us:**
- One signal (logprob_uncertainty) beats all fusions. Simplicity wins.
- Knowledge similarity is structurally inverted on MCQ benchmarks. Needs different retrieval design.
- Naive fusion hurts when signals have heterogeneous quality. Adaptive weighting (v2.0) may help.
- The product is domain-agnostic, not coding-specific. The routing + learning pattern applies everywhere.
- Zero-shot signals beat supervised baselines on cross-dataset transfer. This is the key differentiator.

**What stayed the same:**
- The escalation-based learning loop (escalate once, learn forever)
- The hierarchical knowledge store with temporal validity
- The multi-phase roadmap (single agent → hive → organization)
- The local-first philosophy (your data stays on your machine)

## The Honest Pitch

> Autodidact is an AI agent that learns like a new employee. It thinks first, asks when uncertain, remembers what it learned, and gets more independent over time. Every escalation makes it smarter. Every interaction makes it cheaper.

> Start with any local model + any cloud model. On day one, most queries go to cloud. By week two, most go local. By month three, it's the expert. You can see it learning — every response shows the agent's thought process, what it remembered, what it escalated, and what it just learned.

> Under the hood: intelligent routing via logprob confidence (validated across 3 model families and 2 datasets), a growing knowledge store that learns from every cloud escalation, and a visible thought process that makes the learning tangible.

> This isn't a framework you import. It's an agent you talk to. And it gets smarter every day.

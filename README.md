# Autodidact

**The AI agent that teaches itself.**

Autodidact is a self-learning, local-first AI agent framework. It escalates once, learns forever. Every cloud API call becomes a permanent learning — knowledge, skills, and tools extracted and stored locally. Over time, the agent resolves more queries without touching the cloud.

Day 1: 100% cloud. Month 3: 80% local. The cost goes down. The performance goes up. Your data stays private.

```
Query → Confidence Evaluation → LOCAL (from memory) or CLOUD (learn + store)
                                         ↓
                              Next time → LOCAL (no cloud needed)
```

## Why Autodidact?

| Problem | How Autodidact Solves It |
|---------|------------------------|
| Cloud AI is expensive | Local resolution rate increases over time — cost drops |
| AI forgets between sessions | Persistent hierarchical memory with temporal validity |
| AI doesn't improve | Self-improving skills, Thompson Sampling routing, Ebbinghaus memory |
| Privacy concerns | Local-first — cloud is opt-in, one-time per learning |
| One-size-fits-all | Learns YOUR domain, YOUR tools, YOUR preferences |

## How It Works

**5-signal confidence evaluation** decides whether to answer locally or escalate:

1. **Knowledge Similarity** — Do I have relevant knowledge? (cosine similarity with 0.75 threshold)
2. **Logprob Uncertainty** — How certain is the local model? (token probabilities)
3. **Self-Consistency** — Do multiple attempts agree? (2-sample agreement)
4. **Query Classification** — Is this factual, reasoning, or real-time? (route accordingly)
5. **Energy Scoring** — Have I succeeded on similar queries before? (learned from history)

Signals are fused via **Thompson Sampling** — a Bayesian bandit that learns which signals to trust from outcomes. No manual tuning needed.

## Novel Contributions

These are publishable, benchmarkable innovations — not just engineering:

1. **Thompson Sampling for agent routing** — First Bayesian bandit approach to local/cloud routing decisions
2. **Ebbinghaus-inspired memory consolidation** — STM→LTM promotion with spaced repetition decay
3. **Self-improving procedural memory** — Skills evolve based on success/failure patterns
4. **Learning-from-escalation loop** — Every cloud call is a training signal
5. **Energy scorer bootstrapping** — Agent learns its own competence boundary from history

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    User Query                            │
└──────────────────────┬──────────────────────────────────┘
                       ▼
         ┌──────────────────────────┐
         │   Confidence Evaluator   │
         │   (5 signals + Thompson) │
         └─────────┬────────────────┘
                   │
        ┌──────────┴──────────┐
        │                     │
   score ≥ 0.7          score < 0.7
        │                     │
        ▼                     ▼
  ┌───────────┐      ┌───────────────┐
  │   LOCAL   │      │     CLOUD     │
  │  Answer   │      │   Escalate    │
  │  from     │      │   + Learn     │
  │  memory   │      │   + Store     │
  └───────────┘      └───────┬───────┘
                              │
                    ┌─────────┴─────────┐
                    │  Learning         │
                    │  Extractor        │
                    │  → Knowledge      │
                    │  → Skills         │
                    │  → Tools          │
                    └───────────────────┘
```

## Quick Start

```bash
# Install
pip install autodidact  # Python core
# or
npm install autodidact  # TypeScript SDK

# Initialize
autodidact init

# Query (uses Ollama by default)
autodidact query "What is Thompson Sampling?"
# → CLOUD: learned! (first time)

autodidact query "Explain Thompson Sampling"
# → LOCAL: answered from memory (no cloud needed)
```

## Key Features

- **Multi-signal confidence routing** with Thompson Sampling fusion
- **Hierarchical knowledge store** (domain/topic/category) with scoped search
- **Temporal validity** — knowledge has time windows, not binary stale flags
- **STM/LTM memory tiers** with Ebbinghaus forgetting curve
- **Autonomous tool discovery** — learns APIs from cloud responses
- **Multi-turn conversations** with per-turn routing
- **Self-verification** — periodically validates stored knowledge
- **Skill self-improvement** — procedures evolve based on outcomes
- **User/team profiles** — personalizes across sessions
- **Model-agnostic** — works with any model (3B to frontier)
- **Cost tracking** — see exactly how much you're saving

## Roadmap

| Phase | What | Status |
|-------|------|--------|
| Phase 1 | Self-learning agent framework | In development |
| Phase 2 | Hive — agents teaching agents | Planned |
| Phase 3 | Hierarchical agent organization | Vision |
| Phase 4 | Continuous self-improvement (LoRA) | Research |

See [ROADMAP.md](ROADMAP.md) for the full vision.

## Tech Stack

- **Python core** — ChromaDB/FAISS, BM25+semantic hybrid search, benchmarks
- **TypeScript SDK** — Developer-facing API, npm ecosystem
- **SQLite** — All state in one portable file (WAL mode)
- **Model-agnostic** — Ollama, vLLM, OpenAI, Anthropic, AWS Bedrock

## Benchmarks

Coming soon. We benchmark:
- Retrieval quality (LongMemEval)
- Learning curve (local resolution rate over N queries)
- Thompson Sampling calibration accuracy
- Ebbinghaus decay effectiveness

## Contributing

We're looking for contributors and co-founders. See [CONTRIBUTING.md](CONTRIBUTING.md).

**Areas where we need help:**
- Python core implementation (ChromaDB, FAISS, hybrid search)
- Benchmark suite development
- Energy scorer research (embedding-based confidence)
- Multi-turn conversation optimization
- Documentation and examples

## License

MIT

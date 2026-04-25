# Autodidact

**A self-learning, local-first AI agent that gets cheaper and more private over time.**

Autodidact runs a small local model (via Ollama, vLLM, or similar) for most queries, and escalates to a cloud model (Bedrock, OpenAI, Anthropic) only when it's uncertain. Every cloud escalation becomes a permanent learning, stored locally. Over time, the local model handles more queries on its own.

```
Query → Confidence Evaluation → LOCAL (answer from memory) or CLOUD (escalate + learn)
                                         ↓
                              Next similar query → LOCAL (no cloud needed)
```

## Why

| Problem                    | How Autodidact helps                                                         |
| -------------------------- | ---------------------------------------------------------------------------- |
| Cloud AI is expensive      | Local resolution rate grows over time, cloud cost drops                       |
| AI forgets between sessions | Persistent hierarchical memory with temporal validity windows                |
| One-size-fits-all agents   | Learns from your actual queries and tools, not a generic corpus              |
| Privacy                    | Local-first — the cloud sees a query only when the local model can't handle it |

## How it works

A **5-signal confidence evaluator** decides whether the local model can answer, or whether to escalate:

1. **Knowledge similarity** — is there relevant knowledge already stored? (cosine similarity, 0.75 floor to prevent wrong-knowledge injection)
2. **Logprob uncertainty** — how confident was the local model in the tokens it produced?
3. **Self-consistency** — do two independent local attempts agree on the key facts?
4. **Query classification** — is this factual, reasoning, creative, or real-time?
5. **Energy scorer** — has the local model succeeded on similar queries in the past? (logistic regression on query embeddings, trained on the agent's own history, activates after 50 labeled examples)

Signals are fused via **Thompson Sampling** — each signal has a Beta(α, β) distribution that updates from outcomes, so the router learns which signals to trust without manual tuning.

## Quick Start

> v0.1 is Python-only. TypeScript SDK is post-launch.

```bash
# Install from source (pip package not yet published)
git clone https://github.com/BuffaloTechRider/EvoAgent
cd EvoAgent
pip install -e '.[dev,bedrock]'

# Make sure Ollama is running locally with a chat model and embedder
ollama pull qwen2.5:7b
ollama pull qllama/bge-large-en-v1.5

# Run the demo (uses Ollama as local, Bedrock as cloud if AWS creds present)
autodidact demo
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for full setup details once the launch tasks are complete.

## What's in v0.1

- Multi-signal confidence routing with Thompson Sampling fusion
- Hierarchical knowledge store (domain/topic/category) with scoped search and temporal validity
- STM → LTM promotion with Ebbinghaus-inspired decay (`R(t) = e^(-t/S)`)
- FAISS-backed vector retrieval (not brute-force cosine)
- Knowledge extraction from cloud escalations (verbatim + structured)
- Cost-tracked cloud escalation via AWS Bedrock or OpenAI-compatible APIs
- Quality-aware benchmark on MMLU-Pro measuring local resolution rate **and** answer accuracy over time
- SQLite-backed persistent state (WAL mode, one portable file)
- Model-agnostic — tested with local models from 3B to 70B

## What's NOT in v0.1 (deferred)

Being honest about scope:

- No multi-turn conversation sessions or conversation branching
- No skill extraction or skill store (only knowledge extraction)
- No autonomous tool discovery or tool registry
- No self-verification system for stored knowledge
- No user profile / multi-user support
- No multi-provider cloud router (one provider at a time)
- No TypeScript SDK, MCP server, or hybrid BM25+semantic search

All of these are tracked in [ROADMAP.md](ROADMAP.md) and are good starter issues for contributors post-launch.

## On novelty

An earlier version of this README claimed the Thompson Sampling router and the multi-signal confidence evaluator were novel contributions. After a proper literature review in late 2025, that framing was wrong. Bayesian bandit routing for LLMs, multi-signal hallucination detection, and Ebbinghaus-inspired memory have all been published recently. See the [Prior Art section of the design doc](.kiro/specs/autodidact-framework/design.md#prior-art) (or `design.md` once the spec is public) for citations.

What Autodidact is, honestly: a **well-engineered open-source closed loop** — confidence gate, knowledge retrieval, cloud escalation, learning extraction, quality-aware evaluation — assembled from techniques that each work on their own, now measured end-to-end with answer accuracy preserved. The value is the engineering and the measurement, not the individual algorithms.

## Benchmark

Launch target: MMLU-Pro, 500 stratified queries, Ollama + Bedrock. The benchmark reports four numbers over windowed queries:

- Local resolution rate (how often we answered without cloud)
- Local accuracy (correctness on LOCAL decisions)
- Cloud accuracy (correctness on CLOUD decisions)
- Overall accuracy

Compared against two baselines: `cloud_only` (upper bound) and `local_only` (no learning loop). Results will land in `results/quality_benchmark/` before launch.

## Roadmap

| Phase   | What                                         | Status          |
| ------- | -------------------------------------------- | --------------- |
| Phase 1 | Self-learning agent, Python core, v0.1 launch | In development |
| Phase 2 | Multi-turn, skill/tool extraction, TS SDK    | Planned         |
| Phase 3 | Hive — agents teaching agents                 | Vision          |
| Phase 4 | Hierarchical agent networks                   | Vision          |

See [ROADMAP.md](ROADMAP.md) for details.

## Tech Stack

- **Python 3.10+**
- **SQLite** (WAL mode) — all state in one portable file
- **FAISS** — vector retrieval
- **Pydantic v2** — config and schema validation
- **Ollama / OpenAI-compatible / AWS Bedrock** — LLM backends

## Contributing

We're looking for contributors. See [CONTRIBUTING.md](CONTRIBUTING.md).

Good first issues after launch:

- Multi-turn conversation session manager
- Skill extraction from cloud responses
- Autonomous tool discovery and registry
- Self-verification system
- Additional benchmarks (TriviaQA, LongMemEval)
- TypeScript SDK

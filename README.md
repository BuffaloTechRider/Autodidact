# Autodidact

**A self-evolving AI agent that learns like a new employee.**

Autodidact is an AI agent with a local brain that works like a human, or a new employee. When asked a question or given a task, it thinks first and evaluates whether it can handle the task or answer the question by itself (local brain). If yes, it executes. If not, it escalates — by searching Google or asking a more powerful cloud model — just like how humans work. After the escalation, it learns the new knowledge, skills, or tool usages so next time it won't have to ask similar questions again.

On day one, it asks a lot of questions. By week two, it handles most tasks independently. By month three, it's the expert. Every cloud escalation becomes permanent local knowledge. Every interaction makes it smarter. It never forgets what it learned.

```
Query → Think  (check memory)
      → Try    (local model answers if confident)
      → Ask    (escalate to cloud when uncertain)
      → Learn  (store the answer for next time)
      ──────────────────────────────────────────
      Next similar query → Answer from memory, $0.00
```

## Three-command quickstart

```bash
pip install autodidact             # or: pip install "autodidact[openai,bedrock,pdf]"
autodidact init                    # zero-friction setup: auto-detects Ollama, pulls models, configures cloud
autodidact chat                    # start talking to the agent
```

That's it. `autodidact init` walks you through three setup modes:

1. **Local + Cloud** (default) — Ollama local model + cloud API for escalation. Best cost savings.
2. **Cloud + Cloud** — cheap cloud model (gpt-4o-mini, DeepSeek, etc.) + expensive cloud model. No GPU or Ollama required.
3. **Local only** — Ollama only. Free. No escalation learning.

If Ollama isn't installed, the wizard shows the install command for your platform. If your model isn't pulled, it pulls it automatically. You're in chat within three minutes of `pip install`, regardless of starting point.

## How it works — the human analogy

When you encounter a question, you go through this sequence:

1. **Do I know the answer?** → Check your memory
2. **Am I confident I can answer it?** → Self-assess
3. **If yes** → Answer (free, fast)
4. **If no** → Ask someone smarter (costs time, but you get the right answer)
5. **Remember what you learned** → Store it
6. **Next time, start from step 1** → You're smarter now

A new employee does this every day. The more tasks they do, the more knowledgeable they become, the fewer questions they ask. Eventually, they're the person others come to.

**Autodidact makes AI work the same way.**

### The visible thought process

Every response shows the agent's reasoning, in real time:

```
You> What's our PTO policy?
[THINKING] Checking memory... no relevant entries yet (0 hits)
[CLOUD] Escalated — gpt-4o-mini took 1.2s
[LEARNED] ✅ Stored: "Company PTO is 20 days per year, accrued monthly."
  💰 $0.003 | Confidence: 0.34 → escalated | ✅ Learned

You> How much vacation do I get?
[THINKING] Checking memory... found 1 similar entry (similarity: 0.91)
[MEMORY] Company PTO is 20 days per year, accrued monthly.
  ↳ Recalled from: "What's our PTO policy?" (learned 0 days ago)
  💰 $0.00 | Confidence: 0.91 | Route: memory
```

That's the "magic moment" — the user watching the agent answer from learned knowledge, for free, because it remembered a question it was asked moments ago.

## Solving the cold start

A brand-new agent has an empty brain. `autodidact learn` seeds it with existing knowledge:

```bash
autodidact learn ~/docs/policies/     # ingest a folder of docs
autodidact learn ./README.md          # ingest a single file
autodidact learn --stats              # show what's been ingested
```

Supports `.md`, `.txt`, `.py`, `.ts`, `.yaml`, `.json`, `.csv`, `.html`, and 15+ other text formats. PDFs via `pip install "autodidact[pdf]"`. Chunks are stored separately from learned Q&A (one is reference material, the other is experience), but both get retrieved and injected into the prompt at query time.

## What's in v1.0

- **Zero-friction setup wizard.** Auto-detects Ollama, pulls models, presets for OpenAI / OpenRouter / DeepSeek / Bedrock.
- **Three setup modes.** Local+Cloud (default), Cloud+Cloud (no GPU), Local-only (free).
- **Confidence-based routing.** `logprob_uncertainty` decides when to escalate (validated AUROC 0.65-0.83 across 3 model families × 2 datasets).
- **Learning from escalations.** Structured knowledge extraction from cloud responses, deduplication on insert, staleness-aware re-verification.
- **Cold-start fix.** `autodidact learn <path>` ingests docs so the agent has knowledge from day one.
- **Visible learning UX.** `[THINKING]`, `[MEMORY]`, `[LOCAL]`, `[CLOUD]`, `[LEARNED]` tags show what the agent is doing and why.
- **Cost tracking.** `autodidact savings` reports cumulative cost avoided vs an all-cloud baseline.
- **Local-first.** All state in one portable SQLite file (`~/.autodidact/memory.db`). Works offline after setup.
- **Multi-provider.** Ollama local. OpenAI-compatible cloud (OpenRouter, DeepSeek, Together, Anthropic proxies). AWS Bedrock via optional `[bedrock]` extra.

## Commands

```
autodidact init             Zero-friction setup wizard
autodidact chat             Interactive chat with visible thought process
autodidact query "q"        Single-query mode
autodidact learn <path>     Ingest documents (cold-start fix)
autodidact savings          Cumulative cost savings
autodidact memory stats     Knowledge store size + breakdown
autodidact memory search    Search what the agent has learned
```

## What's NOT in v1.0

- No multi-turn conversation sessions across restarts (in-session history only)
- No skill extraction or skill store (only fact extraction)
- No autonomous tool discovery
- No self-verification cycle (stale entries are flagged but not proactively re-checked)
- No MCP server (coming in v1.1)
- No TypeScript SDK

All of these are in the roadmap. See [ROADMAP.md](ROADMAP.md) and `.kiro/specs/autodidact-full/requirements.md` for what ships when.

## What we **have** verified empirically:

- `logprob_uncertainty` is the dominant routing signal (AUROC 0.65-0.83 across 3 model families × 2 datasets).
- Zero-shot inference-time signals match supervised routing baselines (RouteLLM) at zero per-model training cost.
- Naive multi-signal fusion hurts — the best single signal beats the mean of all 6 signals.
- Signal quality correlates with RLHF calibration training across model families (Qwen > Llama).

Full write-up: [`paper/blog-post.md`](paper/blog-post.md). Research findings have their own home at [zero-shot-llm-confidence](https://github.com/paulnnguyen/zero-shot-llm-confidence).

## Roadmap

| Phase   | What                                             | Status           |
|---------|--------------------------------------------------|------------------|
| v1.0    | Zero-friction self-learning agent                | **Shipping now** |
| v1.1    | Skill extraction, self-verification, MCP server  | Planned          |
| v2.0    | Hive — agents teaching each other                | Planned          |
| v3.0    | Hierarchical agent networks                      | Vision           |
| Phase 4 | LoRA consolidation (episodic → parametric)       | Research         |

See [ROADMAP.md](ROADMAP.md) for the timeline.

## Tech stack

- **Python 3.10+**
- **SQLite** (WAL mode) — all state in one portable file
- **FAISS** — vector retrieval
- **Pydantic v2** — validation
- **Typer + Rich** — CLI
- **Ollama / OpenAI-compatible / AWS Bedrock** — LLM backends

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

Good first issues after launch:

- Multi-turn session persistence across restarts
- Skill extraction (extract procedures from cloud, not just facts)
- MCP server for Claude Desktop / Cursor / Gemini CLI
- Self-verification cycle
- Additional benchmarks (TriviaQA, LongMemEval)
- TypeScript SDK

## License

MIT — see [LICENSE](LICENSE).

---

Built by [BuffaloTechRider](https://github.com/BuffaloTechRider). Repository: [BuffaloTechRider/Autodidact](https://github.com/BuffaloTechRider/Autodidact).

# Autodidact

**A self-evolving AI agent that learns like a new employee.**

Autodidact is an AI agent with a local brain that works like a human, or a new employee. When asked a question or given a task, it thinks first and evaluates whether it can handle the task or answer the question by itself (local brain). If yes, it executes. If not, it escalates - by searching Google or asking a more powerful cloud model - just like how humans work. After the escalation, it learns the new knowledge, skills, or tool usages so next time it won't have to ask similar questions again.

On day one, it asks a lot of questions. By week two, it handles most tasks independently. By month three, it's the expert. Every cloud escalation becomes permanent local knowledge. Every interaction makes it smarter. It never forgets what it learned.

```
Query → Think  (check memory)
      → Try    (local model answers if confident)
      → Ask    (escalate to cloud when uncertain)
      → Learn  (store the answer for next time)
      ──────────────────────────────────────────
      Next similar query → Answer from memory, $0.00
```

## Four-command quickstart

```bash
pip install autodidact             # or: pip install "autodidact[openai,bedrock,pdf]"
autodidact init                    # zero-friction setup: auto-detects Ollama, pulls models, configures cloud
autodidact learn <path>            # A brand-new agent has an empty brain. `autodidact learn` seeds it with existing knowledge. <Path> points to the folder having documents or code you want the agent to learn.
autodidact chat                    # start talking to the agent
```

> **Windows note:** If `autodidact` isn't found after install, use `python -m autodidact` instead (e.g. `python -m autodidact init`). This happens when Python's `Scripts/` folder isn't in your PATH.

That's it. `autodidact init` walks you through five setup modes:

1. **Local + Cloud** (default) — Ollama local model + cloud API for escalation. Best cost savings.
2. **Cloud + Cloud** — cheap cloud model + expensive cloud model. No GPU or Ollama required.
3. **Local + Local** — small Ollama + big Ollama. Fully offline, still learns from escalations. Free.
4. **Custom local** — any OpenAI-compatible server (llama.cpp, LM Studio, vLLM, LocalAI) + optional cloud.
5. **Local only** — single Ollama model. Free. No escalation learning.

If Ollama isn't installed, the wizard offers to install it (with retry on failure). If your model isn't pulled, it pulls it automatically. If Ollama isn't running, it starts the daemon for you. On corporate networks where Ollama can't be installed, mode 2 or 4 work without it.

## How it works - the human analogy

When you encounter a question, you go through this sequence:

1. **Do I know the answer?** → Check your memory
2. **Am I confident I can answer it?** → Self-assess
3. **If yes** → Answer (free, fast)
4. **If no** → Ask someone smarter (costs time and often money too, but you get the right answer)
5. **Remember what you learned** → Store it
6. **Next time, start from step 1** → You're smarter now

A new employee does this every day. The more tasks they do, the more knowledgeable they become, the fewer questions they ask. Eventually, they're the person others come to.

**Autodidact makes AI work the same way.**

### Solving the cold start

A brand-new agent has an empty brain. `autodidact learn` seeds it with existing knowledge:

```bash
autodidact learn ~/docs/policies/     # ingest a folder of docs
autodidact learn ./README.md          # ingest a single file
autodidact learn --stats              # show what's been ingested
```

Supports `.md`, `.txt`, `.py`, `.ts`, `.js`, `.yaml`, `.json`, `.csv`, `.html`, and 15+ other text formats. Code files are split on function/class boundaries via tree-sitter (`pip install "autodidact[code]"`). PDFs via `pip install "autodidact[pdf]"`. Chunks are stored separately from learned Q&A (one is reference material, the other is experience), but both get retrieved and injected into the prompt at query time.

### See it learn

Feed it your docs:

```bash
$ autodidact learn ./engineering-docs/
[1] deployment-guide.md → 8 chunks
[2] architecture.md     → 15 chunks
─── Ingestion Complete ───
Files: 2 · Chunks: 23 · Synthesizing knowledge in background...
```

Ask something the docs alone can't fully answer:


```bash
you> How do I fix "connection refused" on staging?

[CLOUD] Three common causes, ranked by frequency:
  1. VPN dropped after sleep — `vpn connect staging`
  2. Service crashed         — `kubectl get pods -n staging`
  3. Stale DNS post-deploy   — `sudo dscacheutil -flushcache`

↳ Source: deployment-guide.md
💰 $0.012 | Route: cloud | ✅ Learned
```

The docs had the deployment steps, but the local model does not have the *troubleshooting wisdom* or not confident enough in in reasoning or judgemnt. Cloud provided it. The agent learned it.

Next time:

```bash
you> Staging is down again, connection errors

[LOCAL] This is almost always the VPN (it drops after sleep). Quick fix:
  1. `vpn connect staging`
  2. Still failing? `kubectl get pods -n staging` — service may have crashed
  3. After a deploy, flush DNS: `sudo dscacheutil -flushcache`
  ↳ Context: memory (2 facts)
  💰 $0.00 | Route: local
```

Same knowledge. Zero cost. The answer is *better* than raw docs because it leads with the most likely cause (learned from the cloud's reasoning, not just document text).

**That's the loop.** Every escalation makes the agent smarter. Every smart answer saves money. Over time, cloud calls approach zero.


## What's in v1.0.x

- **Zero-friction setup wizard.** Auto-detects Ollama, pulls models, starts daemon, retries on failure. Installs via Homebrew (macOS) or official installer. Presets for 11 cloud providers including Google AI Studio (free tier, no credit card).
- **Five setup modes.** Local+Cloud, Cloud+Cloud, Local+Local, Custom server, Local-only. Works everywhere — GPU, no GPU, corporate network, offline.
- **AST-aware code chunking.** `autodidact learn` uses tree-sitter to split code on function/class boundaries (Python, JS, TS). Each chunk is a complete semantic unit with its class header preserved. Non-code files use overlap-based text splitting.
- **Hybrid retrieval.** BM25 keyword search (FTS5) + vector similarity, merged via Reciprocal Rank Fusion. RRF orders results; cosine similarity scores them — so downstream thresholds remain meaningful.
- **Document synthesis.** `autodidact learn` doesn't just index — it extracts key facts into memory (background, non-blocking). The agent answers from internalized knowledge, not raw chunks.
- **Confidence-based routing.** GSA pre-screen + logprob uncertainty + refusal detection. Escalates when uncertain, stays local when confident. Non-answer detection prevents learning from "I don't know" cloud responses.
- **Learning from escalations.** Structured knowledge extraction from cloud responses (background, non-blocking). Deduplication on insert. Memory recall at 0.80+ similarity serves learned answers directly.
- **Visible learning UX.** `[THINKING]`, `[MEMORY]`, `[LOCAL]`, `[CLOUD]`, `[LEARNED]` tags show what the agent is doing and why.
- **Cost tracking.** `autodidact savings` reports cumulative cost avoided vs an all-cloud baseline.
- **Local-first.** All state in one portable SQLite file (`~/.autodidact/memory.db`). Works offline after setup.
- **Multi-provider.** Ollama, any OpenAI-compatible server (llama.cpp, LM Studio, vLLM), AWS Bedrock, Google AI Studio, OpenRouter, and 8 more. 11 cloud provider presets.

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

## What's NOT in v1.0.x (coming in v1.5 and v2.0)

- No conversational query rewriting (v1.5 — rewrite follow-up queries into self-contained searches using conversation history)
- No markdown-aware chunking (v1.5 — respect tables, code fences, headings as atomic units; tree-sitter markdown grammar)
- No parent-child retrieval (v1.5 — index small chunks, return parent section on hit)
- No contextual chunking (v1.5 — prepend LLM-generated context to chunks before embedding, bridges NL↔code gap)
- No topic-based knowledge pages (v1.5 — knowledge compiled into pages, not flat facts)
- No OpenAI-compatible proxy mode (v1.5 — `autodidact serve`)
- No agentic retrieval (v2.0 — model reads files on demand via tools, no static chunking needed)
- No tool execution (v2.0 — terminal, file ops, ReAct loop)
- No skill learning from tasks (v2.0 — learns procedures, not just facts)
- No reranking (v2.0 — cross-encoder on retrieval candidates)
- No MCP server (v2.0)

All of these are designed and planned.

## What we **have** verified empirically:

- `logprob_uncertainty` is the dominant routing signal (AUROC 0.65-0.83 across 3 model families × 2 datasets).
- Zero-shot inference-time signals match supervised routing baselines (RouteLLM) at zero per-model training cost.
- Naive multi-signal fusion hurts - the best single signal beats the mean of all 6 signals.
- Signal quality correlates with RLHF calibration training across model families (Qwen > Llama).

Full write-up: [`paper`](https://arxiv.org/html/2605.02241v1). Research findings have their own home at [zero-shot-llm-confidence](https://github.com/BuffaloTechRider/zero-shot-llm-confidence-estimation).

## Roadmap

| Version | What | Status |
|---------|------|--------|
| v1.0.4  | AST-aware chunking, Google AI Studio provider, memory transfer, non-answer filtering | **Current** |
| v1.5    | Query rewriting, markdown-aware chunking, parent-child retrieval, contextual chunking, topic pages, `autodidact serve` proxy | Planned |
| v2.0    | Agentic retrieval (readFile), tool execution, skill learning, tiered routing, reranking, MCP server | Designed |
| v3.0    | Agent network — agents teaching each other | Planned |

## Tech stack

- **Python 3.10+**
- **SQLite** (WAL mode) - all state in one portable file
- **FAISS** - vector retrieval
- **tree-sitter** - AST-aware code chunking (optional, for `.py`, `.js`, `.ts`)
- **Pydantic v2** - validation
- **Typer + Rich** - CLI
- **Ollama / OpenAI-compatible / AWS Bedrock / Google AI Studio** - LLM backends

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

Good first issues:

- `autodidact serve` — OpenAI-compatible proxy (drop-in for Cursor, Aider, any tool)
- MCP server for Claude Desktop / Cursor / Gemini CLI
- PDF document ingestion (`unstructured` parser)
- Topic-based knowledge pages (v1.5 core feature)
- Skill extraction from cloud responses (procedures, not just facts)
- `autodidact status` dashboard (learning curve + cost savings visualization)

## License

MIT - see [LICENSE](LICENSE).

---

Built by [BuffaloTechRider](https://github.com/BuffaloTechRider). Repository: [BuffaloTechRider/Autodidact](https://github.com/BuffaloTechRider/Autodidact).

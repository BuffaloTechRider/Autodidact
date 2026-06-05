# Changelog

All notable changes to Autodidact will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.0.7] — 2026-05-22

### Changed

- **PDF and Word support are now optional.** `pymupdf` and `python-docx`
  moved out of the core dependency list; `DocumentStore` imports them lazily
  and raises an actionable `ImportError` ("Install with pip install pymupdf")
  only when a `.pdf` or `.docx` is actually ingested. Keeps the base install
  lean for users who only ingest text files. (#52)
- **`openai` promoted to a core dependency.** It was an optional `[openai]`
  extra, but the OpenAI-compatible client path (OpenRouter, DeepSeek,
  Together, and cloud-to-cloud mode) is common enough that requiring an extra
  was a footgun. The `[openai]` extra is removed.

## [1.0.6] — 2026-05-18

Release-tagging bump. No functional changes over 1.0.5.

## [1.0.5] — 2026-05-18

### Changed

- **Refusal detector recognizes more non-answer phrasings** — added markers
  for "I cannot/can't tell you", "I cannot/can't answer", "I don't have access
  to", and context-grounded refusals ("the provided context does not contain",
  "not available in the provided context"). Catches more cases where the local
  model declines, so they escalate instead of being stored as answers.
- **Cloud provider default switched to Google** in the setup wizard's provider
  picker.

### Fixed

- **Chat session summary now reports the current session, not all-time DB
  stats.** On exit, `chat` prefers per-session stats and recomputes savings
  from this session's cloud cost, so the closing summary reflects what just
  happened rather than the cumulative database total.

## [1.0.4] — 2026-05-18

### Changed

- **Memory-recall answers now stream.** When a query is answered from memory,
  the local model generates the answer via the streaming path (`_call_local`)
  like every other route, instead of a blocking `chat()` call — tokens render
  as they arrive.

## [1.0.3] — 2026-05-17

### Added

- **Custom Server wizard mode** — `autodidact init` can point the local slot at
  any OpenAI-compatible server (llama.cpp, LM Studio, vLLM) instead of Ollama. (#28)
- **CI test workflow** for PR status checks.

### Changed

- **Ollama install flow hardened** — retries the installer, offers a manual
  fallback with a wait loop, and auto-starts the daemon. (#28)
- **Cost calculation updated.**

### Fixed

- **botocore test skips when the module isn't installed** instead of failing.

## [1.0.2] — 2026-05-15

### Added

- **Hybrid retrieval (BM25 + vector)** — document chunks get an FTS5 virtual
  table with sync triggers; `search_hybrid()` merges keyword (BM25) and vector
  results via Reciprocal Rank Fusion (k=60). The agent uses hybrid search for
  document context.
- **Document synthesis on ingest** — a non-blocking background thread extracts
  key facts from document sections after chunking, batch-inserting them
  (10 facts/commit) to minimize write-lock contention. `KnowledgeStore` gains
  `insert_batch()`.
- **Local + Local setup mode** — small Ollama model + big Ollama model, fully
  offline and still learning. Fourth option in the setup wizard.

### Changed

- **Memory hits generate a full answer** via the local model using the stored
  facts as context, rather than returning raw stored text.
- **Cloud-escalation learning runs in the background** (non-blocking), and the
  SQLite connection allows cross-thread access (`check_same_thread=False`) to
  support the background threads.

### Fixed

- **Chunk cap enforced with the BGE tokenizer** during ingestion, so chunks
  can't overflow the embedding model's context.

### Housekeeping

- Moved `confidence_evaluator.py` to `benchmarks/` — it's unused in production
  (the validated single-signal `logprob_uncertainty` path replaced the
  multi-signal fusion).

## [1.0.1] — 2026-05-13

Polish, performance, and correctness on top of 1.0.0. No breaking changes.

### Added

- **Streaming chat** for both local and cloud paths — tokens render as they
  arrive instead of waiting for the full response. Spinner disappears once
  the first token lands. Anthropic extended-thinking content streams under
  a `[THINKING]` header. (#18)
- **`/learn <path>` chat slash command** — ingest documents from inside
  a chat session without leaving the REPL. Bypasses LLM routing and calls
  `DocumentStore.ingest()` directly. Supports `/learn .`, `/learn ~/x`, and
  paths with whitespace. (#22)
- **Live Bedrock model discovery** — `autodidact init` now queries
  `list_foundation_models` + `list_inference_profiles` against the user's
  region and presents only models they can actually invoke. Replaces the
  static preset, which shipped invented IDs and IDs unavailable in some
  regions. (#24)
- **Live OpenRouter catalog browse** — picker gets a "Browse all OpenRouter
  models" entry that hits `/v1/models`, sorts cheapest first, and shows
  per-1M-token pricing alongside each ID. 364+ models available without
  leaving the wizard. (#24)
- **Wizard installs Ollama and starts the daemon when missing** —
  `autodidact init` offers to run the official curl installer on macOS
  and Linux, and to start the daemon if it is down. macOS Gatekeeper and
  Login Items hint included on permission errors. (#16)
- **Smaller default local models** — recommendations shifted down one tier
  per hardware bucket. 32GB Apple Silicon now defaults to `qwen3:8b` instead
  of `qwen3:14b`; the minimal tier stays at `qwen3:0.6b`. (#21)

### Changed

- **GSA forces `think=false`** to keep the YES/NO probe calibrated; reasoning
  tokens were dragging probabilities toward the middle. The chat path keeps
  thinking enabled by default. (#17)
- **Default LLM client timeout** raised from 60s to 300s — large local
  models can take longer than 60s on first-token cold-start. (#17)
- **Retry policy split** by exception class — `ConnectionError` and
  `ConnectTimeout` retry with exponential backoff; `ReadTimeout` fails
  fast. Hung requests no longer multiply latency. (#17)
- **Logprob requests dropped from the chat path** — adds ~150ms per Ollama
  call and the post-local logprob gate is no longer used for routing. GSA
  pre-gate plus the refusal detector cover the same need without the cost.
  (#20)
- **Bedrock model preset is now empty** — discovery is the source of truth.
  Falls back to free-form input on any discovery failure. (#24)

### Fixed

- **GSA error message no longer hides Bedrock errors.** A wide `try/except`
  in `Agent.query()` was wrapping both the GSA probe and the cloud
  escalation. A `ValidationException` from Bedrock surfaced as
  "GSA probe failed, skipping gate: ..." while local silently produced an
  answer. The `try` is now scoped to the probe only; cloud errors
  propagate. (#24)
- **Logprob check no longer triggers false escalations on thinking
  responses.** Thinking-token logprobs were averaged in with content,
  dragging the local confidence below threshold. Routing now skips the
  logprob signal when the response includes thinking tokens. (#19)
- **Document chunks no longer overflow BGE-large's 512-token context.**
  Default chunk size lowered to 384 tokens with a hard cap at 480. Live
  `/learn .` no longer fails with "input length exceeds context length"
  on code files. (#23)
- **Ollama model verification rewritten to use `/api/show`.** Catches the
  cloud-only manifest case — some Ollama tags pull a tiny manifest that
  points at remote inference; the old `ollama list` parse couldn't tell
  these apart from real local models. (#16)
- **Wizard tests can no longer install Ollama on the host.** A conftest
  fixture now blocks real installer invocation in tests, regression-guarding
  an earlier accident. (#16)
- **Embedding model namespace fix** — pulled the right tag for
  `qllama/bge-large-en-v1.5`. (#7)


## [1.0.0] — 2026-05-09

The first shippable release. A self-evolving AI agent with a local brain,
confidence-based routing, and learning from cloud escalations.

### Added

- **Agent core** — `Agent.query()` with three-stage routing (memory → local → cloud),
  `Agent.correct()` for user corrections, `Agent.savings()` for cost tracking.
- **Zero-friction setup wizard** (`autodidact init`) — auto-detects Ollama,
  auto-pulls missing models, cloud provider presets for OpenAI, OpenRouter,
  DeepSeek, and Bedrock. Three setup modes: local+cloud, cloud+cloud, local-only.
- **Cloud-to-cloud routing** — the "local" slot can hold a cheap cloud model
  (gpt-4o-mini, DeepSeek, etc.) that escalates to an expensive cloud model.
  No Ollama or GPU required.
- **Document ingestion** (`autodidact learn <path>`) — cold-start fix. Chunks
  and embeds text files (.md, .txt, .py, .ts, .json, .yaml, + 15 others),
  respects `.gitignore`, deduplicates on re-ingestion. PDF support via the
  optional `[pdf]` extra.
- **Visible learning UX** — `[THINKING]`, `[MEMORY]`, `[LOCAL]`, `[CLOUD]`,
  `[LEARNED]` tags rendered with rich; real-time `on_progress` callback
  hook for custom UIs.
- **Learning extractor** — structured knowledge extraction from cloud responses
  via the local LLM, with a raw-answer fallback when JSON parsing fails.
- **Knowledge store** — SQLite + FAISS, STM/LTM tiers, Ebbinghaus decay,
  deduplication, scoped search by domain/topic/category.
- **Confidence evaluator** — 5-signal Thompson Sampling fusion. `logprob_uncertainty`
  is the dominant signal (validated AUROC 0.65-0.83 across 3 model families × 2
  datasets).
- **Grounded self-assessment (GSA v3)** — retrieval-conditional pre-response
  confidence signal with 3-tier extraction fallback.
- **CLI** — `init`, `chat`, `query`, `learn`, `savings`, `memory stats`,
  `memory search`.
- **Multi-provider LLM client** — Ollama (local), OpenAI-compatible
  (OpenRouter, DeepSeek, Together, etc.), AWS Bedrock via optional
  `[bedrock]` extra. Retry logic with exponential backoff, throttle handling.

### Fixed

- Stale memory entries now fall through to local generation instead of
  escalating straight to cloud. Escalation only happens when local confidence
  is also low — matching the original routing intent (escalate when uncertain,
  not when memory is merely old).
- Packaging bug: `dependencies = [...]` was mis-nested under `[project.urls]`,
  causing `pip install autodidact` to install with zero runtime dependencies.
  Moved inside `[project]` and added a regression guard test.

### Notes

- Research findings (zero-shot confidence signals vs supervised routing) have
  their own home at [zero-shot-llm-confidence](https://github.com/paulnnguyen/zero-shot-llm-confidence).
- Algorithms are not individually novel; the contribution is the well-engineered
  closed loop and the end-to-end measurement with answer accuracy preserved.

[Unreleased]: https://github.com/BuffaloTechRider/Autodidact/compare/v1.0.7...HEAD
[1.0.7]: https://github.com/BuffaloTechRider/Autodidact/compare/v1.0.6...v1.0.7
[1.0.6]: https://github.com/BuffaloTechRider/Autodidact/compare/v1.0.5...v1.0.6
[1.0.5]: https://github.com/BuffaloTechRider/Autodidact/compare/v1.0.4...v1.0.5
[1.0.4]: https://github.com/BuffaloTechRider/Autodidact/compare/v1.0.3...v1.0.4
[1.0.3]: https://github.com/BuffaloTechRider/Autodidact/compare/v1.0.2...v1.0.3
[1.0.2]: https://github.com/BuffaloTechRider/Autodidact/compare/v1.0.1...v1.0.2
[1.0.1]: https://github.com/BuffaloTechRider/Autodidact/compare/v1.0.0...v1.0.1
[1.0.0]: https://github.com/BuffaloTechRider/Autodidact/releases/tag/v1.0.0

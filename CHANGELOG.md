# Changelog

All notable changes to Autodidact will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/BuffaloTechRider/Autodidact/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/BuffaloTechRider/Autodidact/releases/tag/v1.0.0

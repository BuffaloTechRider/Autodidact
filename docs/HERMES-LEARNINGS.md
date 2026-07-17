# Hermes Agent — Learnings for Autodidact v2

**Source:** Structured analysis of the [Hermes agent](https://github.com/nousresearch/hermes-agent) (MIT, Nous Research), June 2026. Three parallel analyses (tools, skills, agent loop) compared Hermes' implementation against Autodidact's v2 design, then verified the cited files against the clone.

**Guiding principle:** For components that are *not* Autodidact's moat — tool calling, skills, the ReAct loop — don't reinvent the wheel; adopt battle-tested patterns. But Hermes is a large multi-user monolith (~2,700 Python files; individual files up to 700 KB — `cli.py` is 722 KB, `run_agent.py` 254 KB, `hermes_state.py` 245 KB). So we **adopt patterns and design decisions, never vendor code**. Copying from those files would import exactly the complexity `CLAUDE.md` forbids.

**Autodidact's moat stays ours:** confidence-based local/cloud routing (GSA → logprob → self-consistency → Thompson-sampled escalation) and learning-from-escalation have no analog in Hermes — it has no confidence routing at all. Everything below is scaffolding *around* that moat.

---

## 1. Tools

**How Hermes does it:** Self-registering modules (each `tools/*.py` calls `registry.register()` at import). `ToolEntry` holds an OpenAI function-calling schema + handler + toolset + optional `check_fn` availability probe. Dispatch returns JSON strings. A generation counter invalidates cached schemas; an RLock guards mutation; TTL-cached availability probes suppress transient failures. AST-based auto-discovery finds 80+ tools across built-ins + plugins.

### ADOPTED (PR #73)

| Pattern | Source | Why |
|---------|--------|-----|
| **Fuzzy find-and-replace for `edit_file`** | `tools/fuzzy_match.py` (865 lines, 9 strategies) | LLM-generated `old` drifts on whitespace/indentation; exact match fails, and cloud-escalation retries drift the same way → the learning loop breaks. We ported the 2 highest-value strategies (`exact → line-trimmed`) + indent re-anchoring (`_reindent_replacement`); dropped the long tail (unicode NFC, escape-drift, block-anchor, context-aware). |
| **Path confinement** | `tools/path_security.py` (`validate_within_dir`) | The file tools could read `/etc/passwd` via `../` or a symlink. All five now resolve and reject paths escaping the working directory. |

### SKIP (monolith-scale for a ~5 K LOC focused agent)

| Pattern | Why skip |
|---------|----------|
| AST-based auto-discovery | Explicit imports in `tools/__init__.py` scale fine to ~20 tools. |
| TTL-cached `check_fn` availability probes + transient-failure grace | Our tools are local; no Docker/Modal/network deps to flake. Revisit if a tool probes external services. |
| Generation counter + LRU-memoized schema assembly | The executor fetches schemas ~once per task, not per gateway turn. Add the counter only if we add MCP/plugin tools. |
| RLock / thread-safe snapshots | Executor is single-threaded. Add if a background skill reviewer runs concurrently. |
| Dynamic per-tool schema overrides | No tool has runtime-dependent schema fields yet. |
| Plugin override policy, toolset aliases, legacy-name maps | No plugin ecosystem, no v1 toolset schema to deprecate. |
| Write-approval gates | Skill writes are governed by `learning_extractor`; per-write approval is overkill. |

### DEFERRED

- **Config-backed output limits** — the hardcoded 16 KB terminal cap is adequate; config plumbing for one constant would be speculative. Revisit if power users need to tune it.

**Verdict on our registry:** sufficient for the ReAct executor. Fuzzy matching was the one genuinely blocking gap (now closed); everything else is insurance for scale we don't have.

---

## 2. Skills

**How Hermes does it:** Markdown files (`skills/<category>/<name>/SKILL.md`) with YAML frontmatter (`name`, `description`, `version`, `platforms`, `tags`, `related_skills`). A **compact index** (names + one-line descriptions, ~500–1000 tokens) goes in the system prompt; full bodies load on demand via a `skill_view(name, file_path?)` tool. Two-layer cache (in-process LRU + disk snapshot keyed on mtime). Support dirs (`references/`, `templates/`) are progressive-disclosure attachments, not standalone skills. A separate Curator job archives skills unused for 30/90 days based on a `skill_usage.json` sidecar.

### ADOPT (Phase C — when we build `skills/`)

| Pattern | Why |
|---------|-----|
| **Compact index in prompt + on-demand `skill_view`** | Validates our existing plan. Keep names+descriptions in the prompt; load full body only when the model requests it. Scales to many skills without bloating the prompt. |
| **Platform-aware gating** | Add `platforms: list[str]`; filter at index-build time. |
| **Tool-availability gating** | Add `conditions: {required_tools?, required_toolsets?}`; only surface skills whose tools are enabled. |
| **Progressive-disclosure attachments** | Model auxiliary files (references/templates) keyed by `(skill_id, path)`, loaded on request — not as standalone skills. |
| **On-disk index snapshot** | Cache the built index; rebuild only when a skill changes. (Cheap with our SQLite backing.) |
| **`related_skills` links** | Lightweight discoverability field; optional. |

### SKIP — keep our structured design instead

Hermes' skills are **Markdown-only prose** — read-only advice with no execution guarantees, no argument templates, no success/failure tracking, no semantic (embedding) retrieval. Our design deliberately diverges to **SQLite-stored, embedded, structured skills** (steps with `tool` + `args_template`, `success_count`/`failure_count`, `source` provenance).

**This divergence is correct and central to the moat**, not NIH:
- Markdown skills **cannot be auto-patched**; structured steps can — and the skill reviewer patching skills from cloud escalations *is* the learning mechanism.
- Keyword-only matching can't find "run the suite" → a `pytest` skill; embeddings can.
- Prose steps can't be re-executed without LLM re-planning; `args_template` + context can.

Also skip for v2.0: external skill directories (multi-profile sharing), Curator as a separate daemon (run review synchronously post-execution instead).

---

## 3. Agent Loop (Executor)

**How Hermes does it:** One unified `while` loop (`agent/conversation_loop.py`, ~5,000 lines). Per iteration: check interrupt → build messages → API call with inline retry/fallback-provider loop → normalize response across transports → parse & validate tool calls (invalid JSON → retry the API call, don't execute) → append assistant message → execute tools (mutating the messages list in place) → append `role="tool"` results → compress trajectory if over token budget → loop. Bounded by `max_iterations` + an iteration budget with a one-call grace. Trajectory compression (`trajectory_compressor.py`, 68 KB) protects head + tail and LLM-summarizes the middle.

### ADOPT (Phase B — when we build `executor.py`)

| Pattern | Why | Effort |
|---------|-----|--------|
| **Unified `while` loop** (retry/fallback inline, no state machine) | Simplest structure that handles the ReAct cycle. | Low |
| **Tool-call parse + validate, retry on invalid JSON** | Invalid/truncated tool args → re-ask the model, don't execute. | Low |
| **Append tool results in place** (`role="tool"` messages) | Standard, minimal. | Low |
| **Iteration bounds + escalation budget** | Prevents infinite loops; caps cost. | Low |
| **Trajectory compression** (protect head+tail, summarize middle) | DESIGN-V2 flags context-window growth as an open risk; Ollama's 4–8 K window overflows on multi-step tasks. Adopt the *strategy*, not their 1.5 K-LOC service — target ~150 LOC with a local-model summarizer. | Medium |
| **Simplified error classification** (retry 5xx/timeout, fallback on 429, fail fast on 401/403) | Resilience without provider-specific machinery. | Low |
| **Trajectory persistence + resume** (per-iteration checkpoint to SQLite; resume interrupted task) | Not just crash-insurance: for a *learning* agent the trajectory (messages + tool results + per-step routing decisions + escalations) is a first-class learning artifact. A task that already spent cloud-escalation dollars must not be thrown away on interrupt. **Match** Hermes' resume; **beat** it by making trajectories queryable/replayable and feeding them to skill extraction + threshold tuning. Adopt the capability at our scale (~SQLite table), not their 245 KB `hermes_state.py`. | Medium |
| **Cache-aware message construction + explicit cloud cache_control** | Byte-stable system prefix (system prompt + tool schemas); ephemeral content (memory/context injections) goes into user messages, never the system prompt. The executor re-sends the prefix ~20×/task, so this (a) warms Ollama's local KV/context cache on the hot path and (b) lets the escalation path pass provider cache params (Anthropic `cache_control`) where cloud cost concentrates. Cheap to adopt now, expensive to retrofit once ephemeral content has leaked into the system prompt. | Low–Medium |

### SKIP (v2.0)

Provider transport abstraction (we already have 3 backends), async/concurrent tool execution, streaming display + TTS + stale-stream health checks, thinking-block edge-case handling, Nous-specific rate-limit guard.

> **Revised 2026-07-08:** session persistence/resume and prompt caching were moved SKIP → ADOPT. Rationale: the "match or beat Hermes" bar (CLAUDE.md §5) applies — both intersect the moat (persistence feeds learning; caching hits the cloud-escalation cost we exist to reduce), and caching discipline is cheap now / costly to retrofit. We adopt the *capabilities* at Autodidact scale, not Hermes' heavy implementations.

### Integration — splicing the moat into a Hermes-style loop

The loop *structure* is Hermes'; the **routing decision at each turn is ours** (`autodidact/routing/stages.py`). Per iteration, after the local model proposes a tool call:

```
logprob   = response.avg_logprob
category  = classify(tool_call)                 # "code" / "terminal" / ...
threshold = router.get_threshold(category)      # Thompson posterior

if   logprob > threshold.high:  execute locally                 # Tier 1
elif logprob < threshold.low:   escalate to cloud, execute      # Tier 3
else:                                                            # Tier 2
    regen = local_model(messages, temperature=0.3)
    if regen.tool_call == tool_call:  execute locally           # self-consistent
    else:                             escalate to cloud, execute # disagreement
```

This is the one place a "commodity" component carries the moat: adopt Hermes' loop skeleton, but every turn routes through our tiered confidence system.

---

## Summary

| Area | Adopt from Hermes | Keep ours / skip |
|------|-------------------|------------------|
| **Tools** | Fuzzy edit (done), path confinement (done) | Registry is otherwise sufficient; skip availability probes, plugin policy, AST discovery |
| **Skills** | Compact index + on-demand load, platform/tool gating | Keep SQLite + structured + embedded + success-tracked; skip Markdown-only, Curator daemon |
| **Loop** | Loop skeleton, tool parse/validate, compression, budgets, trajectory persistence+resume, cache-aware construction | Splice in our tiered routing; skip transports, streaming, async tool exec |

**Bottom line:** Autodidact and Hermes are complementary, not competing. Hermes solves multi-user/plugin/platform scale; Autodidact solves confidence routing + learning, which Hermes never attempted. We take Hermes' scaffolding patterns and keep our moat.

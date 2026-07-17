# Autodidact — Roadmap (vision, not spec)

> **This file is vision-only.** It lists what's shipped and what's planned *beyond*
> the current build so nothing is lost — but it is **not** a spec and carries no
> testable requirements. The single source of truth for what's being built now (the
> **v2.0 apprentice agent**) is [`lifecycle/`](lifecycle/README.md). When a roadmap
> item becomes the active build, it graduates into the lifecycle docs and is removed
> from here.
>
> Detailed historical specs for these future items live in
> [`archive/`](archive/README.md) (`requirements-fullvision.md`, `FUTURE-LEARNINGS.md`).

---

## Shipped

- **v1.0 — Self-learning Q&A agent.** logprob-confidence routing (local↔cloud),
  knowledge store (SQLite + FAISS) that learns from cloud escalations, document
  ingestion for cold start, cost dashboard, zero-friction setup, visible-learning
  UX ([MEMORY]/[LOCAL]/[CLOUD]). Validated: logprob AUROC 0.65–0.83 across 3 models
  × 2 datasets; retrieval +12pp answer quality; zero-shot beats supervised routing
  on cross-dataset transfer.

## Building now → see the SSOT

- **v2.0 — The Apprentice Agent.** Step-level tiered routing inside a ReAct
  execution loop, learning reusable skills from cloud escalations. **Full spec and
  tasks:** [`lifecycle/`](lifecycle/README.md). Do not duplicate v2.0 detail here.

---

## Next: v1.1 — Memory Intelligence (planned, not spec'd)

Richer memory, orthogonal to the apprentice-agent execution work. Candidate items
(from `archive/requirements-fullvision.md` R1.1.x):

- **Tiered memory + consolidation** — HOT/WARM/COOL/COLD (working → STM → LTM →
  archive), `autodidact maintain` job (dedupe, compress, promote/demote, validate
  stale), provenance on consolidation ops.
- **Contradiction detection** — flag when cloud disagrees with stored memory; store
  correction history, never overwrite.
- **Self-verification** — periodic re-testing of stored knowledge, staleness flags,
  LLM-as-judge contradiction checks.
- **Document synthesis on ingest** — compile documents into knowledge at ingest
  time, not re-derive at query time (design in `lifecycle/03b-rag-pipeline.md` §Will).
- **Code-aware ingestion** — delegate to GitNexus MCP for AST/function-level
  chunking (R1.1.9). Integrate, don't vendor.
- **Scoped search** — wire existing `KnowledgeScope(domain, topic, category)` into
  the query path (R1.1.10).
- **Integrations** — LangChain/LlamaIndex `AutodidactLLM` drop-in, OpenAI-compatible
  proxy (`autodidact serve`), MCP server, external retriever hook.

## Later: Phase 2 — The Hive (agents teaching agents)

A decentralized network where agents discover experts and trade knowledge. Core
concepts: Agent Registry, Skill Transfer Protocol (DISCOVER → NEGOTIATE → TRANSFER
→ VALIDATE), Knowledge Tokens, Reputation System, memory-poisoning defense
(provenance, quarantine, reputation-weighted trust).

> **Naming note:** older docs called the Hive "v2.0." In this repo **v2.0 = the
> apprentice agent**; the Hive is a later phase.

## Later: Phase 3 — The Organization (hierarchical agent network)

Multi-tier hierarchy mirroring a human org: Specialist (3B) → Domain (7B) → Senior
(70B) → Principal (frontier). Escalate UP when uncertain, knowledge flows DOWN.
The Phase-1 confidence router generalizes to N-tier routing.

## Vision: Phase 4 — Continuous Self-Improvement (research)

Beyond memory-based (episodic) learning to parametric learning: periodic LoRA
fine-tuning on the accumulated knowledge store, online continual learning,
consolidation safety (provenance, versioned checkpoints, regression guards). This
is where Autodidact stops being a "memo system" and starts changing weights.

---

## How the phases build

```
v1.0  single agent learns from cloud escalations (episodic memory)   ✅ shipped
  ↓   foundation: routing + KB + confidence signals
v2.0  apprentice agent: step-level routing + skill learning          🔧 building (lifecycle/)
  ↓   richer execution: skills + traces
v1.1  memory intelligence: tiered memory, synthesis, integrations    planned
  ↓
Phase 2  agents teach agents via skill transfer (Hive)               later
  ↓
Phase 3  agents organized in cost-optimized hierarchies              later
  ↓
Phase 4  agents improve their own weights (episodic → parametric)    research
```

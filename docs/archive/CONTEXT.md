# Autodidact — Domain Context

> ⚠️ **v1.0-era document — partially superseded by v2.0.** This file describes the
> shipped v1.0 domain model. The v2.0 apprentice-agent design (`docs/DESIGN-V2.md`
> + `docs/lifecycle/`) supersedes several decisions here, specifically:
> - **Routing (AD-001, AD-003):** v1.0 used logprob-only, post-generation, single
>   threshold 0.7. v2.0 replaced this with a tiered pipeline (GSA pre-gate +
>   logprob + self-consistency + knowledge-verify, per-category Thompson
>   thresholds), implemented in `autodidact/routing/stages.py`. See
>   `lifecycle/02-overview.md`.
> - **Vector index:** "FAISS" throughout this doc is superseded by **USearch** in
>   v2.0 (`DESIGN-V2.md` → "Vector index: USearch (replaces FAISS)").
> - **Confidence Threshold:** the single 0.7 threshold below is superseded by the
>   4-tier thresholds in `DESIGN-V2.md`.
>
> Terminology and bounded-context sections below remain accurate. Treat the
> architectural decisions as v1.0 history unless echoed in the lifecycle docs.

## What this project is

Autodidact is a self-learning AI agent that routes queries between a cheap local model and an expensive cloud model, learning from every cloud escalation to handle more queries locally over time.

## Key Terms

### Agent
The central runtime. Accepts a user query, decides how to answer it (from memory, locally, or via cloud), and learns from the outcome. Not a chatbot framework — a single intelligent process that manages routing and memory.

### Routing
The decision of whether to answer a query locally (free, fast) or escalate to cloud (paid, smarter). Based on confidence signals computed from the local model's own output. NOT load balancing — it's a quality-based decision, not a capacity-based one.

### Escalation
When the agent decides the local model can't handle a query and forwards it to the cloud model. Every escalation is a learning opportunity — the cloud's answer gets stored in memory.

### Learning (v1.0: memory storage only)
In v1.0, "learning" means storing the cloud model's answer in the knowledge store so similar future queries can be answered from memory. This is semantic caching with retrieval, not model fine-tuning or skill extraction.

**Important:** the README and marketing say "learns from every interaction." In v1.0, this specifically means "stores Q&A pairs from cloud escalations." It does NOT mean the model's weights change, the agent discovers new tools, or procedures are extracted. Those are v1.1+ features.

### Confidence
A scalar in [0, 1] predicting whether the local model's answer is correct. Computed from `logprob_uncertainty` — the average per-token log-probability of the generated answer, mapped through a sigmoid. Higher = more likely correct.

NOT the same as the model saying "I'm confident." It's a statistical property of the token distribution, not a self-report.

### Memory (Agent Memory / Knowledge Store)
The agent's internal store of past Q&A pairs learned from cloud escalations. Stored in SQLite with FAISS vector index for semantic search. Separate from any external document store the user might have.

**Not the same as:** conversation history (which is per-session and in-memory), or a document store / RAG pipeline (which contains the user's source documents).

### Document Store
The user's source documents (PDFs, wiki pages, markdown files). Ingested via `autodidact learn`. Stored in the same SQLite+FAISS infrastructure as agent memory but tagged differently. Used for RAG-style context injection.

**In v1.0:** the document store and agent memory share the same physical database but are logically separate (different `source` tags). In v1.1, they may be separated into distinct stores if the use cases diverge.

### Embedding Model
A separate model (default `qllama/bge-large-en-v1.5`, ~358MB, 1024-dim) used to vectorize queries, knowledge entries, and document chunks for similarity search. Distinct from the chat model — see AD-004 for why we don't reuse the chat model for this.

### Confidence Threshold
The routing decision boundary. Queries with confidence >= threshold are answered locally. Below threshold → escalate to cloud. Default 0.7 based on v0.1 experiments. Configurable per deployment.

### Magic Moment
The product term for when the agent answers a query from memory for the first time. The user asked something, the agent escalated to cloud, learned the answer, and now answers a similar question instantly and for free. This is the moment the user understands the value proposition.

## Bounded Contexts

- `autodidact/` — core agent code (routing, memory, signals, LLM client)
- `benchmarks/` — v0.1 experiment harness and analysis (not shipped in the product)
- `results/` — experiment data and reports (not shipped)
- `paper/` — academic paper drafts (not shipped)
- `.kiro/specs/` — spec files for Kiro IDE (not shipped)

## Architectural Decisions

### AD-001: logprob_uncertainty as the primary routing signal
**Decision:** use average per-token log-probability as the sole routing signal in v1.0.
**Context:** v0.1 experiments tested 6 signals across 3 models and 2 datasets. logprob_uncertainty was the best single signal on every model and dataset (AUROC 0.65-0.83). Multi-signal fusion was worse than logprob alone.
**Alternatives considered:** Thompson Sampling fusion of 6 signals (worse), GSA-only (weaker), RouteLLM-style supervised classifier (doesn't transfer across datasets).

### AD-002: Two separate retrieval stores (document store + agent memory)
**Decision:** keep user documents and agent-learned Q&A in logically separate stores.
**Context:** they serve different purposes. Documents answer "what do the source materials say?" Agent memory answers "have I been asked this before?" Mixing them would confuse retrieval — a document chunk about PTO policy is different from a past Q&A about PTO policy.
**Trade-off:** more complex than a single store, but cleaner semantics and easier to reason about.

### AD-003: Post-generation routing (not pre-generation)
**Decision:** generate the local answer first, then check confidence, then decide to escalate.
**Context:** logprob_uncertainty requires a full generation to compute. Pre-generation signals (GSA) are weaker (AUROC 0.56-0.64 vs 0.65-0.83). The latency cost of generating-then-discarding on escalated queries is accepted as a trade-off for better routing accuracy.
**Mitigation:** Stage 1 memory check (pre-generation) catches queries that can be answered from memory without any generation. Only queries that miss memory AND fail local confidence hit the double-generation cost.

### AD-004: Dedicated embedding model, separate from the chat model
**Decision:** ship a separate, embedding-specific model (`qllama/bge-large-en-v1.5`) for memory and document retrieval, in addition to the chat model. Do not reuse the chat model's hidden states as embeddings.
**Context:** chat models are decoder-only and trained for next-token prediction; embedding models are bidirectional / encoder-style and trained with contrastive objectives that explicitly pull semantically similar texts together. A chat model's mean-pooled hidden state is technically a vector, but it has no contrastive signal in its training and produces poor retrieval quality.
**Measured:** Task 13's retrieval upgrade (`benchmarks/validate_retrieval_upgrade.py` against P15) showed switching from `nomic-embed-text` (a weaker embedding model) to `bge-large-en-v1.5` raised retrieval recall@5 from ~17% to ≥40% on the same KB and queries. Memory routing depends on retrieval quality, so this isn't a marginal optimization — it's the difference between memory hits actually firing and silently missing.
**Cost:** an extra ~358MB pull alongside the chat model. Negligible vs. the chat model's 5+ GB.
**Alternatives considered:** (1) reuse chat model via mean-pooled hidden states — rejected, no contrastive training; (2) `qwen3-embedding:8b` — a dedicated embedding model in the qwen3 family, aesthetically nicer but ~5GB and not yet benchmarked in our setup; flagged as a v1.1 experiment.
**Trade-off accepted:** two-model setup over one-model simplicity, because measured retrieval quality is decisive for the routing story.

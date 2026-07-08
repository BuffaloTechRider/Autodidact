# RAG Pipeline — Decisions and Direction

**Status:** Decision record. Living document.
**Last updated:** 2026-05-13
**Triggered by:** Live `autodidact learn .` failures on dense code (#PR-after-1.0.1).
**Anchors:** `requirements.md` R1.1.9, `DESIGN-V2.md` §7, `ROADMAP.md` v1.1.

---

## Why this document exists

The first ingestion bug (BGE-large rejecting 521-token chunks) was tactical. The next instinct was strategic — "should we adopt a RAG framework?" This doc fixes the answer in writing so the next time the question comes up, we don't re-litigate.

## Today's pipeline (v1.0)

`autodidact/document_store.py` does five things in ~510 lines:

1. **Walk** — file-system traversal with extension allowlist + .gitignore filter.
2. **Chunk** — character-based splitter with paragraph/line/sentence boundary preference. Hard cap enforced via real BGE token count (post 1.0.2).
3. **Embed** — Ollama `/api/embeddings` against `qllama/bge-large-en-v1.5`.
4. **Write** — SQLite for chunks, FAISS for the index. Single backend. Single dim (1024).
5. **Retrieve** — flat ANN search, top-k by cosine. No reranking, no metadata filtering, no hybrid lexical search.

This is *raw RAG*. The agent fetches chunks at query time and the LLM re-derives knowledge from them on every call.

## What we will and will not do

### Will

- **Document synthesis on ingest** (DESIGN-V2 §7, v1.1).
  After chunking, run a synthesis pass that calls the local LLM to extract structured facts/concepts/relationships from each section, and store the result in the existing `KnowledgeStore` (the same store `LearningExtractor` populates from cloud escalations). The agent's existing memory-first path catches these synthesized entries before falling back to raw chunk retrieval.
  - Raw chunks remain as the precision fallback ("what exactly does page 47 say...").
  - Synthesis is local-LLM cost — acceptable at ingest time, not at query time.
  - This is the v2.0-shaped move that makes the system *less* dependent on chunk-quality and retrieval-recall.

- **Code-aware ingestion** (R1.1.9, v1.1).
  Delegate to GitNexus MCP for AST parsing, function-level chunking, dependency graphs, and call chains. Don't rebuild a 14-language tree-sitter pipeline.
  - GitNexus is a separate MCP server. We integrate, we do not vendor.

- **Advanced document parsing** for non-code (R1.1.9, v1.1).
  Use `unstructured` or `LlamaIndex.readers` for PDFs with tables/images and HTML pages. Drop them in behind the existing `chunk_text` interface as alternative parsers, selected by extension.
  - Parser swap is a small surface; it does not infect retrieval, scoring, or storage.

- **Incremental re-indexing** (R1.1.9, v1.1).
  Track per-file content hash and re-chunk only changed files. Today every `autodidact learn .` re-walks the whole tree and dedupes by content; this is correct but wasteful at scale.

- **Scoped search** (R1.1.10, v1.1).
  Wire the existing `KnowledgeScope(domain, topic, category)` into `Agent._check_memory()`. Infrastructure is already there, just unused. Helps both speed and relevance as the KB grows.

- **Hybrid BM25 + vector retrieval** (DESIGN-V2 §16, v2.0).
  When semantic similarity hits a wall on rare technical terms (model names, error codes, identifiers), pair the FAISS recall with a BM25 pass and merge by RRF. Optional, only if benchmarks show it's needed.

### Will not

- **Adopt LangChain or LlamaIndex as a routing/agent layer.** They impose their own abstractions over confidence routing, memory tiers, and the [MEMORY]/[LOCAL]/[CLOUD] visible-learning UX. Our differentiator is the routing + learning loop; we don't outsource it.

- **Adopt Chonkie / semantic-text-splitter as a chunking framework.** The current splitter is ~80 lines. The bug we just hit was about token *counting*, not chunking *strategy*. A framework brings new versions to track without reducing our code.

- **Replace SQLite + FAISS with ChromaDB / Qdrant / Weaviate.** Our store is single-process, single-user, single-tenant by design. The MemPalace council debate explicitly flagged ChromaDB fragility. FAISS scales to millions of chunks on a laptop.

- **Add dense reranking (cross-encoder)** before the synthesis layer ships. Reranking helps recall@k; it does not fix retrieval correctness when chunks are wrong. If the synthesis layer turns "raw chunks" into the precision-fallback path it should be, top-k recall matters less.

- **Add embedding-model abstraction.** v1.0 is BGE-large via Ollama. v1.1 may add OpenAI text-embedding-3-* for cloud-cloud users. Until that's a real demand, one model, one dim, one backend.

## Boundaries we hold

These are non-negotiable for v1.x.

| Property | Why |
|---|---|
| Local-first (Ollama embeddings work offline) | The product's promise. Cloud RAG defeats the differentiator. |
| Single-process, single-DB | Ops simplicity. No vector DB to run. |
| Chunks → KnowledgeStore is the same store as cloud escalations | One memory surface, not two retrieval systems. |
| Memory-first routing (no chunk retrieval before checking the KB) | DESIGN-V2 §7 architecture. Chunks are fallback, not primary. |

## Decision log

| Date | Decision | Reason |
|---|---|---|
| 2026-05-13 | Use BGE tokenizer for cap enforcement, not char heuristic | Live failure on agent.py: 521-token chunks despite 1500-char cap |
| 2026-05-13 | Add `tokenizers>=0.20` as a hard dep | ~5MB, Rust-backed, already battle-tested upstream |
| 2026-05-13 | Document synthesis (DESIGN-V2 §7) confirmed as the v1.1 RAG direction, not framework adoption | We're a routing+learning system that happens to use RAG, not a RAG product |
| 2026-05-13 | GitNexus MCP confirmed as the v1.1 code-ingest path, not in-tree AST parsing | Months of work avoided; integration via MCP, not vendoring |

## What "good RAG" looks like, by phase

**v1.0 (today, post-1.0.2):**
- Chunks fit the embedding model's context window. Always. (✓ now)
- Top-k retrieval surfaces relevant chunks for the cloud judge.

**v1.1 (months 2–3):**
- Memory-first answers most queries from the synthesized KB without raw chunk retrieval.
- Code ingest via GitNexus produces function-level chunks with dependency context.
- Incremental re-index: changing one file re-chunks one file.

**v2.0+:**
- Hybrid BM25 + vector retrieval where it earns its keep.
- Cross-encoder reranking only if measured benefit > latency cost.
- External retriever hook (`Agent(..., retriever=my_func)`) for users who already have a pipeline.

## What this doc is not

It is not a ban on changing direction. It is a checkpoint of why we're *here* so future debates start from "what's changed since 2026-05-13?" rather than zero.

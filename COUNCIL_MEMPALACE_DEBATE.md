# Council Deliberation: MemPalace Framework Analysis

**Date:** 2026-05-03
**Context:** Autodidact v1.0 — analyzing MemPalace to identify learnings for our agent

## What MemPalace Is

[MemPalace](https://github.com/mempalace/mempalace) is an open-source AI memory system. Key properties:

- **Verbatim storage** — stores original text, never summarizes or paraphrases
- **Structured index** — Wings (people/projects) → Rooms (topics) → Halls (categories: facts, events, discoveries, preferences, advice) → Drawers (original text chunks)
- **Pluggable retrieval** — ChromaDB default, interface-based backend swap
- **Temporal knowledge graph** — entity-relationship triples with validity windows (valid_from/valid_to), SQLite-backed
- **Cross-wing navigation** — "tunnels" connect the same topic across different projects/people
- **Agent diaries** — each specialist agent gets its own wing and diary stream
- **29 MCP tools** — full read/write/navigate/graph/diary interface
- **Local-first** — nothing leaves your machine unless you opt in
- **96.6% R@5 raw on LongMemEval** — no LLM required for core retrieval; 98.4% with hybrid heuristics

## Council Members

- **C1 (The Architect):** System design, how ideas map to Autodidact
- **C2 (The Pragmatist):** What's shippable, what's speculative
- **C3 (The Researcher):** Evaluates claims and evidence
- **C4 (The Product Thinker):** User experience and value proposition
- **C5 (The Skeptic):** Challenges both MemPalace and any proposed changes

---

## DEBATE 1: Verbatim Storage vs Our Approach

**C1:** MemPalace's core philosophy is "store verbatim, never summarize." Our LearningExtractor does the opposite — we extract structured facts from cloud responses. These are fundamentally different design choices.

**C3:** MemPalace's approach is optimized for *recall* — "what did we discuss about X?" Our approach is optimized for *reuse* — "answer this new question using what you learned." Verbatim storage is better for conversation history. Structured extraction is better for knowledge that needs to compose with new queries.

**C5:** But MemPalace's 96.6% R@5 on LongMemEval is impressive. Our knowledge store hasn't been benchmarked on anything comparable. Are we sure extraction is better than verbatim for our use case?

**C4:** Different products, different needs. MemPalace is a *memory system* — it remembers conversations. Autodidact is a *learning agent* — it learns from escalations to answer future questions. A user asking "what did we discuss about auth?" wants verbatim recall. A user asking "what's our PTO policy?" wants a direct answer, not a transcript of the conversation where they first asked.

**C1:** That said, we already store `verbatim_response` on the first extracted entry. We have both: structured facts for reuse AND the original cloud response for reference. MemPalace's approach validates that keeping the original is important.

**CONSENSUS:** Our extraction + verbatim storage approach is correct for our use case. MemPalace validates that keeping the original text matters (we already do this). No change needed.

---

## DEBATE 2: The Palace Structure (Wings/Rooms/Halls)

**C1:** MemPalace's hierarchical structure is interesting:
- **Wings** = projects/people → maps to our `domain` field
- **Rooms** = topics → maps to our `topic` field
- **Halls** = categories (facts, events, discoveries, preferences, advice) → maps to our `KnowledgeCategory` enum

We already have this structure! Our `KnowledgeEntry` has `domain`, `topic`, and `category` fields. And our `KnowledgeCategory` enum has the exact same values: FACTS, EVENTS, DISCOVERIES, PREFERENCES, ADVICE.

**C3:** The difference is that MemPalace *uses* this structure for scoped search. Their `mempalace_search` accepts `wing` and `room` filters. Our `KnowledgeStore.search()` has a `scope` parameter with `KnowledgeScope(domain, topic, category)` — but I'm not sure we actually use it in the agent's query flow.

**C5:** Let me check... In `Agent._check_memory()`, we call `self.memory.search(q_emb, limit=5, min_similarity=0.0)` with no scope. We're doing flat search across everything. The scoping infrastructure exists but isn't wired up.

**C4:** For v1.0 with a small knowledge store, flat search is fine. But as the KB grows (v1.1+), scoped search becomes important. If a user is asking about "auth migration" and we have 10,000 entries across 50 domains, scoping to the relevant domain first would improve both speed and relevance.

**C2:** This is a v1.1 concern. The infrastructure is already built. We just need to wire it up when the KB is large enough to benefit. No v1.0 change.

**CONSENSUS:** Our domain/topic/category structure already mirrors MemPalace's wings/rooms/halls. The scoping infrastructure exists but isn't used in the agent flow yet. Wire it up in v1.1 when KB size warrants it. No v1.0 change.

---

## DEBATE 3: Temporal Knowledge Graph

**C1:** MemPalace has a temporal entity-relationship graph: subject → predicate → object with `valid_from`/`valid_to`. This is exactly what the Memoria paper proposed (council debate #2) and what we discussed for v1.1 skill extraction.

Their implementation is clean: SQLite with two tables (`entities` and `triples`), temporal validity windows, invalidation support, timeline queries. The API is simple: `add_triple`, `query_entity`, `invalidate`, `timeline`.

**C3:** This is more structured than our knowledge store. We store flat Q&A pairs. They store relationships. The question is: does the structure help retrieval?

**C5:** For their use case (conversation memory), yes — "who works on what" is naturally a graph query. For our use case (answering questions from learned knowledge), it's less clear. Our queries are "what's the answer to X?" not "what's the relationship between A and B?"

**C4:** But there's a hybrid opportunity. When we extract knowledge from cloud responses, some facts ARE relationships: "Python was created by Guido van Rossum in 1991" → (Python, created_by, Guido van Rossum, valid_from=1991). Storing these as triples alongside our Q&A pairs would enable richer queries.

**C1:** We already have `valid_from`/`valid_to` on `KnowledgeEntry`. The temporal validity infrastructure exists. What we don't have is the triple structure. Adding it would be a v1.1 feature — extend the LearningExtractor to also extract entity-relationship triples, store them in a separate table, and use them for context enrichment.

**C2:** This is interesting but not v1.0. The LearningExtractor already extracts structured facts. Adding triple extraction is a natural extension for v1.1. Reference MemPalace's schema as a design template.

**CONSENSUS:** MemPalace's temporal KG validates the direction we discussed in the Memoria council debate. Our `valid_from`/`valid_to` infrastructure already exists. Triple extraction is a v1.1 feature — use MemPalace's schema (entities + triples tables) as a reference design. No v1.0 change.

---

## DEBATE 4: Cross-Wing Navigation (Tunnels)

**C1:** MemPalace's "tunnels" are explicit cross-domain connections. When the same topic appears in different wings (projects), a tunnel links them. This enables graph traversal: "start at auth-migration in project A, find related content in project B."

**C3:** This is relevant for our v2.0 Hive (agents teaching agents). When Agent A knows about "auth migration" and Agent B asks about it, the tunnel concept maps to skill transfer discovery.

**C5:** For a single-agent system (v1.0-v1.1), tunnels are overkill. We have one knowledge store, not multiple wings. Cross-domain connections happen naturally through embedding similarity.

**C4:** Agreed. But the concept is worth noting for v2.0. When agents have separate knowledge stores, explicit connections between them (tunnels) would help with knowledge discovery.

**CONSENSUS:** Tunnels are a v2.0 concept for the Hive. Not relevant for v1.0-v1.1. Note in the full vision spec as a reference for agent-to-agent knowledge discovery.

---

## DEBATE 5: Agent Diaries

**C1:** Each MemPalace agent gets its own diary — a persistent stream of observations, findings, decisions, and patterns. This is separate from the knowledge store. It's the agent's *working notes*, not its *learned knowledge*.

**C4:** This is interesting for Autodidact. Our agent currently has no persistent self-reflection. It learns facts from escalations, but it doesn't record observations like "I've been getting a lot of questions about auth lately" or "my confidence on Python questions has been improving."

**C3:** The diary concept maps to what the prototype's `MetricsTracker` does — recording query patterns, success rates, and trends. But MemPalace's diary is more qualitative: it's the agent writing notes to its future self.

**C5:** This is a nice-to-have, not a must-have. The agent's query_log already captures quantitative patterns. A qualitative diary adds complexity without clear user value in v1.0.

**C2:** Agreed. But for v1.1's self-verification system, the diary concept is useful. The verification cycle could write diary entries: "Verified 20 entries, 18 passed, 2 flagged stale." This gives the agent a persistent record of its own maintenance activities.

**C1:** One concrete idea: after each session, the agent could write a brief diary entry summarizing what it learned. "Session 2026-05-03: 12 queries, learned 3 new facts about Python deployment, confidence improving on DevOps topics." This would feed into the v1.1 context builder (L1 layer: user profile summary).

**CONSENSUS:** Agent diaries are a v1.1 feature, not v1.0. Useful for self-verification logging and session summaries. Note as a reference for v1.1 context builder and self-verification.

---

## DEBATE 6: MCP Server (29 Tools)

**C1:** MemPalace exposes 29 MCP tools covering reads, writes, navigation, knowledge graph, and agent diaries. This makes it usable from any MCP-compatible tool (Claude Code, Cursor, Gemini CLI, etc.).

**C4:** This is a distribution strategy, not a feature. MemPalace is a *memory layer* that other tools consume. Autodidact is an *agent* that uses memory internally. Different product shapes.

**C3:** But the MCP angle is relevant for our v1.1 integrations. Instead of building a LangChain wrapper AND an OpenAI proxy AND a Cursor plugin, we could expose Autodidact's capabilities as MCP tools. One integration surface, many consumers.

**C5:** MCP is already in our v1.1 roadmap (deferred from v1.0). MemPalace's tool design is a good reference for what to expose. But we shouldn't copy their 29-tool surface — our agent is simpler. Maybe 5-8 tools: query, search_memory, savings, memory_stats, learn_document.

**C2:** Agreed. MCP is v1.1. Use MemPalace's tool design as a reference for the interface, but scope to our actual capabilities.

**CONSENSUS:** MCP server is v1.1. MemPalace's 29-tool design is a reference, but we'd expose 5-8 tools matching our agent's capabilities. No v1.0 change.

---

## DEBATE 7: Deduplication and Mining

**C1:** MemPalace has `mempalace_check_duplicate` (similarity threshold 0.85-0.87) and deterministic drawer IDs for exact-match dedup. We have deduplication at sim > 0.95 in `Agent._learn()`. Their threshold is lower.

**C3:** 0.85 is aggressive — it would merge entries that are related but not identical. Our 0.95 is conservative — only near-exact duplicates get merged. For a learning agent, conservative is better. You don't want to merge "What's Python's GIL?" with "What's Python's GC?" just because they're 0.87 similar.

**C5:** Agreed. Our 0.95 threshold is correct for our use case. MemPalace's 0.85 makes sense for conversation chunks where slight rewordings are common.

**CONSENSUS:** Our deduplication threshold (0.95) is correct. No change.

---

## DEBATE 8: Benchmarking

**C3:** MemPalace benchmarks on LongMemEval (96.6% R@5), LoCoMo (88.9% R@10), ConvoMem (92.9%), and MemBench (80.3%). They're transparent about methodology and commit per-question results.

**C5:** We have no comparable benchmark for our knowledge store. Our v0.1 experiments measured *routing signal quality* (AUROC), not *retrieval quality* (R@5). We measured retrieval recall at 89% in-category, but that's a different metric on a different task.

**C4:** For v1.0 launch, we should have a retrieval benchmark. Not necessarily LongMemEval (that's conversation recall, not Q&A retrieval), but something that measures "after N escalations, what percentage of similar future queries get answered from memory?"

**C1:** This is already in our launch plan as the "quality benchmark" — MMLU-Pro, 500 queries, measuring local resolution rate and accuracy over time. But we should also measure retrieval precision: when the agent says [MEMORY], is the retrieved answer actually correct?

**CONSENSUS:** We need a retrieval quality benchmark for v1.0 launch. MemPalace's transparency about benchmarking methodology is a good model. Add retrieval precision measurement to the quality benchmark plan.

---

## FINAL CONSENSUS

| Topic | Decision | Timeframe |
|---|---|---|
| Verbatim vs extraction | Our extraction + verbatim approach is correct. No change. | — |
| Palace structure (wings/rooms/halls) | Already mirrors our domain/topic/category. Wire up scoped search in v1.1. | v1.1 |
| Temporal knowledge graph | Use MemPalace's schema as reference for v1.1 triple extraction. | v1.1 |
| Cross-wing tunnels | v2.0 concept for Hive agent-to-agent discovery. | v2.0 |
| Agent diaries | v1.1 for self-verification logging and session summaries. | v1.1 |
| MCP server | v1.1, 5-8 tools. Use MemPalace's design as reference. | v1.1 |
| Deduplication threshold | Our 0.95 is correct. No change. | — |
| Benchmarking | Add retrieval precision to v1.0 quality benchmark. | v1.0 |

### Key Takeaways

1. **We're architecturally aligned.** Our domain/topic/category/valid_from/valid_to structure already mirrors MemPalace's wings/rooms/halls/temporal validity. We built the same abstractions independently.

2. **MemPalace is a memory system; Autodidact is a learning agent.** They store and retrieve. We store, retrieve, AND route + learn + improve. Different products with overlapping infrastructure.

3. **Their temporal KG is a clean reference design** for our v1.1 triple extraction. Two tables (entities + triples), temporal validity, invalidation — simple and effective.

4. **Their MCP-first distribution** is worth studying for v1.1. One integration surface (MCP) vs many (LangChain + proxy + plugin) is simpler to maintain.

5. **Their benchmarking discipline** (committed per-question results, reproducible commands, honest about limitations) is a model for our v1.0 launch.

6. **What they DON'T have that we do:** confidence routing, cloud escalation, learning from escalations, cost tracking, visible thought process. MemPalace is passive memory; Autodidact is active learning.

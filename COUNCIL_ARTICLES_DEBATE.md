# Council Deliberation: Two Articles on Agentic Memory

**Date:** 2026-05-03
**Context:** Autodidact v1.0 — self-learning agent with confidence routing and knowledge store

## The Articles

1. **Memoria** (arXiv 2512.12686v1) — "Memoria: A Scalable Agentic Memory Framework for Personalized Conversational AI"
   - Sarin, Singh, Sarmah, Mehta
   - A modular memory framework: dynamic session summarization + weighted knowledge graph for user modeling
   - Exponential decay weighting, SQL + vector DB, incremental KG triplet extraction

2. **"Contextual Agentic Memory is a Memo, Not True Memory"** (arXiv 2604.27707v1) — Xu, Dai, Zhang (CUHK/Zhejiang)
   - A position paper arguing ALL current agentic memory (vector stores, RAG, scratchpads) is **lookup, not memory**
   - Proves a Generalization Gap: retrieval needs Ω(k²) examples for compositional tasks; parametric learning needs O(d/δ)
   - Identifies the "Frozen Novice Problem": agents never `.train()`, only `.predict(C)`
   - Calls for a consolidation pipeline: episodic store → parametric weight updates (LoRA, MEMIT, etc.)

## Council Members

- **C1 (The Architect):** Focuses on system design and how ideas map to Autodidact's architecture.
- **C2 (The Pragmatist):** Cares about what's shippable in v1.0 and what's speculative.
- **C3 (The Researcher):** Evaluates the theoretical claims and experimental evidence.
- **C4 (The Product Thinker):** Asks "does this change the user experience or value proposition?"
- **C5 (The Skeptic):** Challenges both articles and any proposed changes.

---

## DEBATE 1: Is Autodidact's Knowledge Store "Just a Memo"?

**C3:** The Xu et al. paper makes a strong theoretical argument. Autodidact's knowledge store IS exemplar-based lookup. We store Q&A pairs and retrieve by similarity. By their taxonomy, we have Episodic memory (external store) but no Experiential memory (weight changes). We are, by their definition, a "well-organized novice."

**C5:** Hold on. Their Theorem 1 (Generalization Gap) applies to *compositional* tasks — combining k base concepts in novel ways. Autodidact v1.0 targets *repetitive* domains: customer support, internal Q&A, knowledge work. These are high-overlap, low-composition domains. The gap theorem's Ω(k²) requirement matters less when most queries are near-duplicates of past queries.

**C1:** That's a fair point, but it doesn't invalidate the critique. Even in repetitive domains, there's a long tail. The 89% retrieval hit rate we measured means 11% of queries have no relevant memory. For those, the local model is still frozen at its pretrained capability. Over time, as the easy queries get cached, the *remaining* queries are increasingly the compositional/novel ones — exactly where the Generalization Gap bites.

**C4:** From a product perspective, the "Frozen Novice" label is uncomfortable but accurate for v1.0. Our agent gets better at *recall* (answering from memory) but not at *reasoning* (handling novel combinations). The user might notice: "It remembers what I asked before, but it doesn't seem *smarter* on new things."

**C2:** For v1.0, this is fine. We're shipping a product, not solving AGI. The knowledge store delivers measurable value (cost savings, faster responses). The Frozen Novice problem is a v2.0+ concern.

**CONSENSUS:** Autodidact v1.0 IS a "memo system" by Xu et al.'s taxonomy. This is acceptable for v1.0's scope (repetitive domain queries). But the critique correctly identifies the ceiling: as easy queries get cached, the remaining hard queries expose the frozen model's limitations. This should inform the roadmap.

---

## DEBATE 2: What Can We Learn from Memoria's Architecture?

**C1:** Memoria's two-component design (session summarization + weighted KG) maps interestingly to Autodidact:
- Their **session summarization** ≈ our conversation history (v1.0 has this in-session)
- Their **weighted KG** ≈ our knowledge store, but structured as triplets rather than Q&A pairs
- Their **exponential decay** ≈ our Ebbinghaus decay (already built)

The key difference: Memoria extracts *structured triplets* (subject, predicate, object) from conversations. We store *verbatim Q&A pairs*. Triplets are more composable — you can answer "What does Paul prefer for deployment?" by combining (Paul, prefers, Docker) + (Paul, uses, AWS) even if that exact question was never asked.

**C3:** Their experimental results are modest: 87.1% vs 85.7% (full context) on single-session, 80.8% vs 78.2% on knowledge-update. The real win is latency: 38.7% reduction by using ~400 tokens instead of 115K. But Autodidact already gets this benefit from retrieval — we inject relevant context, not the full history.

**C5:** The KG triplet extraction is expensive. They use GPT-4.1-mini for every message to extract triplets. For Autodidact, where the whole point is cost reduction, adding an LLM call per message to build a KG defeats the purpose.

**C4:** I see one useful idea: **structured extraction from cloud escalations**. When we escalate to cloud and get an answer, instead of storing just the Q&A pair, we could also extract key facts as structured entries. This is already in the v1.1 roadmap as "skill extraction." Memoria's triplet approach is one way to do it.

**C2:** For v1.0, no changes. The Q&A store works. For v1.1, consider structured extraction as part of the "skill extraction" feature — but use the cloud model that already answered (no extra LLM call) to also return structured facts.

**CONSENSUS:** Memoria validates our decay-weighted retrieval approach but doesn't offer anything we need for v1.0. The structured triplet idea is worth revisiting in v1.1's skill extraction feature, but only if it can piggyback on existing cloud calls (no extra LLM cost).

---

## DEBATE 3: The Consolidation Pipeline — Should It Be on the Roadmap?

**C3:** Xu et al.'s strongest recommendation is: build a consolidation channel from episodic store to parametric memory. Concretely: periodically fine-tune the local model (LoRA) on accumulated knowledge. This is already in Autodidact's Phase 4 roadmap ("Periodic LoRA fine-tuning on accumulated knowledge"). The paper provides theoretical justification for why this matters.

**C1:** The consolidation pipeline they describe has three principles:
1. Treat agentic memory as episodic lookup (not expected to generalize) ✅ — we already do this
2. Build a consolidation pathway (episodic → parametric) — this is Phase 4
3. Make consolidation safe (provenance, versioned checkpoints, regression guards) — not on our roadmap yet

Principle 3 is the interesting one. If we ever do LoRA fine-tuning on accumulated knowledge, we need:
- Provenance tracking: which knowledge entries contributed to which LoRA
- Versioned checkpoints: rollback if fine-tuning degrades performance
- Regression guards: test suite that catches capability loss

**C5:** This is Phase 4 material. We're debating v1.0. Let's not roadmap-creep.

**C2:** Agreed. But I'd add one sentence to the Phase 4 section of ROADMAP.md citing this paper's consolidation safety principles. It's a good framework for when we get there.

**C4:** The product implication is interesting though. Xu et al. say agents should *tell the user* they're doing lookup, not learning. Our visible learning UX (R2) already does this — we show `[MEMORY]` vs `[LOCAL]` vs `[CLOUD]`. But we call it "learning" in the marketing. Xu et al. would say: storing a Q&A pair is not learning, it's note-taking. Learning is when the model's weights change.

**C1:** That's a fair semantic point, but "learning" is the right product term. Users understand "the agent learned this" to mean "it will remember this." We don't need to explain the episodic/parametric distinction to users. The CONTEXT.md already clarifies internally: "In v1.0, 'learning' means storing Q&A pairs from cloud escalations."

**CONSENSUS:** The consolidation pipeline (episodic → parametric via LoRA) is correctly placed in Phase 4. Xu et al.'s safety principles (provenance, versioning, regression guards) should be noted in the roadmap. No changes to v1.0.

---

## DEBATE 4: The Security Argument — Persistent Memory Poisoning

**C3:** Xu et al. make a compelling security argument: persistent memory converts transient prompt injection into permanent compromise. If a malicious query poisons the knowledge store, every future retrieval from that entry is compromised. They cite MINJA (98.2% injection success persisting across sessions) and PoisonedRAG (5 adversarial texts → 90% attack success).

**C1:** This is directly relevant to Autodidact. Our knowledge store persists cloud answers permanently. If a user asks a question that triggers a poisoned response from the cloud model, that poisoned answer lives in our KB forever and gets served to similar future queries.

**C5:** The attack surface is narrower than they claim for our case. Autodidact stores *cloud model answers*, not arbitrary user content. The attacker would need to either: (a) compromise the cloud model's response, or (b) craft a query that causes the cloud to return a poisoned answer that then poisons future retrievals. Both are harder than injecting into a user-writable memory store.

**C4:** But user correction (`/correct`) IS a user-writable path into the KB. If we implement that in v1.0, we need to think about what happens when a user (or an automated system using the agent) writes malicious corrections.

**C2:** For v1.0, the mitigation is simple: the knowledge store is local, single-user, and the user controls what goes in. Multi-user scenarios (v2.0 Hive) are where poisoning becomes a real threat. We should note this in the v2.0 security considerations.

**C1:** One concrete v1.0 action: when implementing user correction (`/correct`, `/wrong`), we should store the *original* answer alongside the correction, not overwrite it. This gives us an audit trail and the ability to detect if corrections are being used to inject bad data.

**CONSENSUS:** Memory poisoning is a real concern but primarily for multi-user scenarios (v2.0+). For v1.0: (1) store correction history rather than overwriting, (2) note the security consideration in the Hive roadmap section.

---

## DEBATE 5: Applicability to Autodidact — What Changes?

**C1:** Let me summarize what's actionable across all timeframes:

### v1.0 (now — no changes to scope)
- **No architectural changes.** The Q&A knowledge store is the right v1.0 approach.
- **One implementation detail:** when implementing user corrections (R5.5), store correction history (old answer + new answer + timestamp) rather than overwriting. This is a one-field addition to the schema.

### v1.1 (Memory Intelligence — minor roadmap refinement)
- **Structured extraction:** When extracting knowledge from cloud escalations, consider extracting structured facts (triplets or key-value pairs) alongside the verbatim Q&A. This makes the knowledge store more composable for novel queries. Aligns with the existing "skill extraction" roadmap item.
- **Consolidation job safety:** The `autodidact maintain` consolidation job should track provenance (which entries were merged/compressed) for auditability.

### v2.0 (Hive — security consideration)
- **Memory poisoning defense:** When agents share knowledge (Skill Transfer Protocol), implement provenance tracking and validation. An agent should not blindly trust knowledge from another agent.
- **Reputation-weighted trust:** The existing Reputation System concept should factor into knowledge acceptance — knowledge from low-reputation agents gets quarantined.

### Phase 4 (Continuous Self-Improvement — theoretical grounding)
- **Xu et al. provides the theoretical justification** for why Phase 4 matters. The Generalization Gap theorem proves that retrieval-only systems have a provable ceiling on compositional tasks. LoRA fine-tuning on accumulated knowledge is the path to breaking through that ceiling.
- **Consolidation safety principles:** provenance tracking, versioned checkpoints, regression guards. These should be first-class requirements when Phase 4 is specced.
- **Benchmark design:** When evaluating Phase 4, measure *compositional generalization over time* (CGT), not just recall accuracy. Does the fine-tuned model handle novel concept combinations better than the retrieval-only version?

**C4:** The most important takeaway for the product: Autodidact v1.0 is honestly positioned. We call it "learning" but CONTEXT.md clarifies it's memory storage. The visible UX (`[MEMORY]` tag) makes it clear when the agent is recalling vs reasoning. The Phase 4 roadmap already targets the right next step (LoRA). These papers validate our trajectory without requiring changes to v1.0.

**C2:** Agreed. Ship v1.0 as designed. The papers confirm we're building the right foundation — the episodic store that Phase 4's consolidation pipeline will consume.

**C5:** One caution: don't let these papers inflate the roadmap. Phase 4 is "research exploration" for a reason. LoRA fine-tuning on accumulated knowledge is an unsolved problem (catastrophic forgetting, data quality, evaluation). The papers identify the gap but don't solve it.

**C3:** Fair. But Xu et al.'s framing is useful for the pitch: "v1.0 gives you a smart filing system. Future versions will give you an agent that actually internalizes what it's learned." That's a compelling upgrade story.

---

## FINAL CONSENSUS

| Topic | Decision | Timeframe |
|---|---|---|
| Is Autodidact a "memo system"? | Yes, by Xu et al.'s taxonomy. Acceptable for v1.0. | Acknowledged |
| Memoria's KG triplets | Not needed for v1.0. Consider for v1.1 skill extraction. | v1.1 |
| Consolidation pipeline | Already on roadmap (Phase 4). Add safety principles. | Phase 4 |
| Memory poisoning | Store correction history in v1.0. Full defense in v2.0. | v1.0 (minor), v2.0 |
| Structured extraction | Piggyback on cloud calls in v1.1. No extra LLM cost. | v1.1 |
| Compositional generalization benchmark | Add to Phase 4 evaluation plan. | Phase 4 |
| Marketing language ("learning") | Keep. CONTEXT.md clarifies internally. UX makes it visible. | No change |

**Bottom line:** These papers validate Autodidact's trajectory. v1.0 is correctly scoped as an episodic memory system. The path from v1.0 (memo) → Phase 4 (true learning via consolidation) is exactly what Xu et al. prescribe. No changes to v1.0 scope. Minor roadmap annotations for v1.1+.

# v1.0.4 Session: Issues, Debugging, and Design Decisions

This document captures the issues encountered during the v1.0.4 development session, how each was diagnosed, the options considered, and why we chose the approach we did.

---

## 1. Hybrid Search Scoring Bug (RRF scores vs cosine thresholds)

### Symptom
GSA always vetoed local answers (`gsa=0.00 | gsa_vetoed`) even after ingesting relevant documents. The doc bypass check and document context injection never triggered.

### Debugging
- Traced the GSA bypass logic: `has_doc_context = bool(doc_hits and doc_hits[0].score >= 0.75)`
- `search_hybrid()` returned RRF scores (Reciprocal Rank Fusion)
- Calculated max possible RRF score: `2/60 = 0.033` — can never reach 0.75
- Same issue with context injection threshold (0.30) — also unreachable

### Root Cause
`search_hybrid()` returned raw RRF scores (0-0.033 range) but all downstream code compared against cosine similarity thresholds (0-1 range). Two different scales, one comparison.

### Options Considered
1. **Normalize RRF to [0,1]** — divide by max theoretical score. Problem: normalized RRF compresses into a narrow band (~0.87-1.0), making thresholds meaningless.
2. **Lower thresholds to RRF scale** — change 0.75 → 0.030, 0.30 → 0.020. Unintuitive.
3. **Use cosine as the score, RRF only for ordering** — RRF determines result order (chunks found by both methods rank higher), but the returned score is the vector cosine similarity.

### Decision: Option 3
RRF is a rank fusion technique — it only knows ordinal position, not absolute relevance. Cosine similarity is the only signal with meaningful absolute semantics (0.9 = "very similar", 0.3 = "barely related"). Using cosine as the score preserves the semantic meaning of all thresholds while still benefiting from RRF's multi-signal ordering.

---

## 2. Document Context Injection — Chunk Relevance Filter

### Symptom
After fixing hybrid search, needed to decide what threshold filters out irrelevant chunks.

### Options
- **0.30** (original) — too low, lets in noise
- **0.50** — filters to genuinely relevant content
- **0.75** — too strict for code retrieval

### Decision: 0.50
Below 0.50 cosine with bge-large, chunks are "loosely related" at best. For code/technical content, 0.50 is the floor where a chunk is likely to actually help the model answer.

---

## 3. GSA Doc Bypass Threshold

### Symptom
Abstract queries like "how is routing implemented" scored 0.68-0.74 against code chunks. The 0.75 threshold meant the bypass never triggered for these queries.

### Debugging
Ran actual cosine similarity measurements against the document store for test queries. Found that `bge-large-en-v1.5` on abstract queries against code tops out around 0.74.

### Options
- **0.65** — covers all test queries
- **0.70** — covers "routing" (0.745) but not "knowledge map" (0.686)
- **Keep 0.75** — almost never triggers

### Decision: Initially 0.70, later raised to 0.75
Started at 0.70 but after testing showed that broad questions (scoring 0.65-0.68) actually produce better answers from cloud, we raised back to 0.75. The doc bypass should only trigger for queries with very strong doc matches. Broad architectural questions are better served by the cloud → learn → memory path.

---

## 4. AST-Aware Code Chunking (tree-sitter)

### Symptom
Local model answers were shallow — it saw 500-char code fragments that cut mid-function. The cloud model also claimed "I can't see the full implementation" because only fragments were injected as context.

### Debugging
Research showed frontier tools (Claude Code, Cursor, Aider) don't use fixed-size chunking for code. State of the art is either agentic retrieval (model reads files on demand) or AST-boundary chunking.

### Options
1. **Adjacent chunk expansion** — fetch ±1 neighbors, stitch together. Simple but adds overlap duplication and latency.
2. **Larger chunks** — 1024 tokens instead of 512. Worse retrieval precision.
3. **AST-aware splitting (tree-sitter)** — split on function/class boundaries. Each chunk is a complete semantic unit.
4. **Full agentic retrieval** — model reads files via tools. Best results but requires v2 executor.

### Decision: Option 3 (tree-sitter AST chunking)
- Each chunk is a complete function/method with its class header prepended
- No overlap needed (boundaries are semantic, not arbitrary)
- Retrieval precision stays high (small, focused chunks embed well)
- Each chunk carries its enclosing context ("class Agent:\n    def query(...)")
- Falls back to text chunking for non-code files
- Added as optional dependency (`pip install autodidact[code]`)

We initially also implemented adjacent chunk expansion (option 1) but removed it after realizing AST chunks are already complete semantic units — expansion just added unrelated adjacent methods for +280ms latency.

---

## 5. Chunk Size Exceeding Embedding Model Limit

### Symptom
After implementing AST chunking, Ollama returned 500 errors: "input length exceeds context length." Some chunks were 600-700 tokens, exceeding BGE-large's 512-token context.

### Root Cause
Token estimation used 4 chars/token (prose heuristic) for code that's actually ~2.5-3 chars/token. A 1800-char chunk "looks" like 450 tokens but is actually 650 real tokens.

### Fix
Three-layer defense:
1. Use `_FALLBACK_CHARS_PER_TOKEN = 2.5` for code (conservative estimate)
2. `_subsplit_with_header` reserves room for the prepended class/method signature
3. Final `_enforce_cap` pass uses the **real BGE tokenizer** — any chunk still over gets bisected

Result: max chunk size = 480 tokens (verified across all source files).

---

## 6. Refusal Detector False Positive

### Symptom
Local model correctly answered "how confidence evaluation works" with a detailed explanation. Mid-response it explained the refusal detection feature: "The system flags explicit **I don't know** responses." The refusal detector matched this as a real refusal and escalated to cloud.

### Root Cause
Substring-based refusal detection scanned the entire response. The phrase "I don't know" appeared inside the model's explanation of the feature, not as a self-report of inability.

### Options
1. **Check first 200 chars only** — real refusals happen upfront; explanations come later
2. **Require sentence-start position** — more precise but complex regex
3. **Exclude matches inside quotes/backticks** — handles quoting but fragile

### Decision: Option 1 (first 200 chars)
Simple, effective, no false negatives for real refusals. If the model wrote 200+ chars of valid content before mentioning "I don't know," it's clearly explaining, not refusing.

---

## 7. System Prompt Causing Hallucinated Disclaimers

### Symptom
Local model added fake disclaimers: "unimplemented features (e.g., full Ollama integration) are not assumed" and "Scope: Focuses on documented components." These features ARE implemented — the model was editorializing.

### Root Cause
System prompt said: "If the context doesn't contain enough information to fully answer, say so rather than making things up." The model over-interpreted this as "always add a limitations section."

### Decision
Replaced with trust-based framing:
- "The user will lose trust if you fabricate facts or code — DO NOT make up facts, code, or features that don't exist."
- "Do NOT editorialize about what you can or cannot see in the context."

This blocks fabrication without encouraging disclaimers. Real limitations from the code (e.g., "Bedrock doesn't support embeddings") can still be stated because they're facts.

---

## 8. Non-Answer Learning Pollution

### Symptom
Cloud responded "I don't have reliable information about Claude 4 Opus, check these links." This non-answer was stored as learned knowledge. Future similar queries would recall "I don't know" from memory.

### Decision
Added `_cloud_response_is_non_answer()` check before learning. Scans first 300 chars for markers like "I don't have reliable," "rather than risk giving you inaccurate," etc. Non-answers are not stored. The `[LEARNED]` indicator only shows when actual knowledge was extracted.

---

## 9. Memory Transfer Threshold (MEMORY_DIRECT_THRESHOLD)

### Symptom
After Q1 went to cloud and learned, Q2 (rephrased same topic, 0.789 similarity) didn't serve from memory — it re-escalated to cloud. Three similar questions each cost money independently.

### Debugging
Measured cross-query similarity: Q1↔Q2 = 0.789, Q1↔Q3 = 0.754, Q2↔Q3 = 0.801. All above 0.60 (context injection) but below 0.85 (direct memory serving).

### Options
- **Lower to 0.78** — catches Q2 from Q1's learning
- **Lower to 0.75** — catches all cross-matches, small risk of serving loosely related content
- **Keep 0.85, fix GSA instead**
- **Keep 0.85, accept cloud for first-time broad queries**

### Decision: 0.80
Tested local answer quality when serving Q2/Q3 from Q1's memory: good enough (accurate, structured, covers correct components) though less comprehensive than cloud. At 0.80 cosine with bge-large, queries are clearly about the same topic. Risk of false matches is low.

The rationale: first broad query → cloud (builds trust with comprehensive answer). Rephrased follow-ups → memory (saves money, quality is acceptable since user already got the comprehensive version).

---

## 10. Savings Calculation Underestimating

### Symptom
Session with 1 cloud ($0.035) and 2 local/memory ($0.00) showed "Saved: 15%" instead of the expected ~67%.

### Root Cause
The savings estimate used `$0.003` as the hypothetical cloud cost for local/memory queries. But actual cloud cost was $0.035 — 10x higher. The estimate was nonsensical.

### Options
- **Fixed minimum** ($0.003) — what we had, clearly wrong
- **Average cloud cost** — better but diluted by cheap queries
- **Max cloud cost** — assumes local/memory queries would have been similarly expensive to the most expensive cloud call in the session

### Decision: Max cloud cost
Local/memory queries that avoided cloud are typically similar in complexity to the cloud queries that taught the system. Using max is the most honest estimate of "what was avoided." Result: savings now shows 67% for a 2/3 local session.

---

## 11. Bedrock Temperature Deprecation

### Symptom
`ValidationException: temperature is deprecated for this model` when using Claude Opus 4 on Bedrock.

### Options
1. **Catch error and retry without temperature** — adds a failed round-trip on every call
2. **Don't send temperature to Bedrock** — simple, zero latency cost
3. **Model-specific allowlist** — complex to maintain

### Decision: Option 2
Removed `temperature` from both Bedrock paths entirely. Also removed `temperature=0.0` from cloud calls in the agent. Reasoning: temperature=0.0 is just "be deterministic" — these models default to low randomness anyway. No point sending a parameter that some models reject. Local model still uses temperature=0.0 (needed for calibrated logprob confidence).

---

## 12. Ollama Version Mismatch After Install

### Symptom
Homebrew installed Ollama 0.24.0, but the model registry required a newer version. Error: "pull model manifest: 412 — requires a newer version."

### Root Cause
Two issues: (1) Homebrew formula lags behind Ollama releases. (2) An old Ollama daemon (0.4.2 from a previous install) was still running, ignoring the new binary.

### Decision
Three-step recovery in `_pull_and_verify`:
1. Restart daemon (`pkill -9` all ollama processes, then start fresh) — handles stale daemon
2. If still 412, fall back to official curl installer (gets absolute latest) — handles Homebrew lag
3. If still fails, show clear "update Ollama" message with download link

Also: install via Homebrew first on macOS (works on corporate VPNs), fall back to curl installer.

---

## 13. Memory Path Not Streaming

### Symptom
When answering from memory, the spinner showed "Recalling from memory..." for 5-10 seconds, then the entire response appeared at once. No token-by-token streaming.

### Root Cause
Memory path used `self._local_client.chat()` (blocking, non-streaming) while the local path used `self._call_local()` (streaming with token callbacks).

### Decision
Changed memory path to use `self._call_local(messages, _emit)` — same streaming approach as local generation. User now sees tokens appear as they're generated regardless of routing decision.

---

## 14. Google AI Studio Provider

### Symptom
Users with Claude Pro / ChatGPT Plus subscriptions can't use Autodidact — subscriptions don't include API access.

### Decision
Added Google AI Studio as a provider:
- Free tier (500 req/day, no credit card)
- Uses OpenAI-compatible endpoint (`generativelanguage.googleapis.com/v1beta/openai/`)
- Mapped through existing OpenAI provider code path (just different base_url)
- Wizard shows helpful note: "ChatGPT/Claude subscriptions do NOT include API access"
- Links to get free/cheap API keys from Google, OpenAI, Anthropic

Also mapped all OpenAI-compatible providers (google, openrouter, deepseek, groq, etc.) through a single `_OPENAI_COMPAT_PROVIDERS` set in `_parse_model_string`.

---

## Summary of Key Design Principles Applied

1. **Prefer model-agnostic solutions** — threshold tuning that works for one model breaks on another. We chose structural fixes (skip GSA when context is available, check first 200 chars) over calibration fixes.

2. **Quality over cost savings** — when in doubt, let the cloud answer. First impressions matter more than saving $0.03.

3. **Defensive fallbacks** — three-layer chunk size enforcement, three-step Ollama version recovery, streaming fallback for non-Ollama providers.

4. **Don't learn garbage** — non-answer filtering prevents memory pollution from "I don't know" cloud responses.

5. **Show progress** — every waiting period gets a spinner with context ("Checking memory," "Confirming with local brain," "Updating Ollama").

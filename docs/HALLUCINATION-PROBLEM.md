# Silent Hallucination: Problem Statement and Options

**Status:** open question, not yet implemented
**Owner:** paulnn
**Last updated:** 2026-05-11

## The failure mode

During local testing:

```
you> tell me more about openclaw
[LOCAL] It seems there might be a typo in your query. Are you referring 
to OpenCL (Open Computing Language)?
Confidence: 0.90 | Route: local

you> no it is not correct, I mean the new agent framework openclaw, not opencl
[LOCAL] I understand now. You are referring to OpenCLAW, which is a newer 
agent-based modeling framework. Here's some information about it:
- Modular Architecture: Allows users to define and simulate various 
  types of agents...
- Scalability: Can handle large-scale simulations efficiently...
Confidence: 0.91 | Route: local
```

OpenCLAW does not exist. The model fabricated a plausible framework description.

Routing:
- Turn 1: caught by the refusal detector (PR #9) — "are you referring to" triggers.
- Turn 2: **not caught by anything.** The answer is confident, fluent, and committed. Shipped to the user at high confidence.

## Why each current signal fails

**Logprobs (our only routing signal today).** Token-level logprobs measure how likely each next token is given the previous ones. "Modular architecture" is a likely continuation of "key features:" regardless of whether OpenCLAW exists. avg_logprob stays high throughout fabrication.

**Refusal detector (PR #9).** Triggers on voluntary surrender markers ("I don't know", "did you mean"). A fabricated answer is the opposite — it's enthusiastic agreement.

**Sycophancy pressure.** Instruction-tuned models are trained to be helpful. When the user asserts "I mean the new agent framework openclaw", the model treats that as ground truth and generates what a real framework description would look like. The user's pushback is read as permission to hallucinate more confidently.

**No retrieval grounding.** Memory is empty for "openclaw" (because it isn't real). No documents either. Model generates purely from parametric knowledge — except it has no parametric knowledge, so it fills the void with genre templates.

## Why this is hard in general

This is a known open problem in LLM research. The core mismatch:
- Our routing operates at the token level.
- Hallucination lives at the claim level — a stream of fluent tokens that collectively describe something nonexistent.

No single mechanical signal available at inference time cleanly separates "confident true answer" from "confident fabricated answer."

## Options, scored honestly

| Option | What it does | Catches the openclaw case? | Cost | Shippable in v1.x? |
|---|---|---|---|---|
| **A. Correction detection** — detect user pushback ("no that's wrong", "actually not X") in chat REPL, treat as implicit `correct()` | Forces cloud on follow-up after user disagrees | No for turn 1, yes for turn 2+ | Tiny | v1.0.1 |
| **B. Ungrounded warning** — when a query has no memory / doc hits AND contains capitalized proper nouns not in a small known set, prefix: "I have no learned knowledge about this, guessing:" | Surfaces uncertainty for niche terms | Partial (warns, doesn't prevent) | Small | v1.0.1 |
| **C. Wire up GSA signal** — `autodidact/signals/grounded_self_assessment.py` is half-built; use it as second gate after logprob confidence | Catches hallucinations where the model's self-grade drops | Maybe (GSA has its own FP rate, needs tuning) | +1 LLM call per query | v1.1 |
| **D. Consensus routing** — if local confidence is high BUT the query contains novel proper nouns not in memory/docs, silently cross-check with cloud | Catches many confidently-wrong niche answers | Yes (for detectable class) | $ on every rare-term query | v1.1 possible, needs cost controls |
| **E. Real RAG with web tool** — give model a search tool; ground answers in citations | Fixes the root cause | Yes | Big feature; changes architecture | v2.0 |
| **F. Manual override** — `!cloud` prefix or `/cloud` slash command in the chat REPL to force escalation | Gives user agency when they know local will hallucinate | No (but user can fix) | Trivial | v1.0.1 |
| **G. Document the limitation** — honest README section, expose `escalated_on_refusal` and a new `hallucination_risk` flag for unbacked proper nouns | Sets user expectations; pairs with F | No | Trivial | v1.0.1 |

## Recommendation

**v1.0.1: A + F + G.** A is targeted and cheap — the moment the user says "no, that's wrong", we already have `Agent.correct()`; we just need to wire it into the chat REPL. F gives manual control. G sets expectations.

**v1.1: C.** GSA is already in the repo. Wiring + tuning is a real task but not a big one.

**v2.0: E.** Web tool is the right long-term answer. RAG with citations is the only approach that doesn't try to out-guess the fundamental limits of logprob signals.

## Rejected non-starters

- **Temperature = 0 as a "fix".** Already at 0. Doesn't help — deterministic fabrication is still fabrication.
- **"Just lower the threshold to 0.95".** Would cause massive false escalation on genuine answers (the 90+% confident ones that are correct). Makes cost worse, doesn't catch the problem answer (0.91 in the openclaw case).
- **"Add more examples to the system prompt".** Brittle, bloats context, model still hallucinates.

## Open questions

1. Should `hallucination_risk` be a binary flag or a float? (Binary is easier to explain to users; float is more honest.)
2. Where does the "known proper nouns" set for option B come from? (Memory? A prelude of capitalized tokens from the local model's training set?)
3. Is GSA measuring the right thing for this case? It grades answer-vs-question relevance, not answer-vs-reality. Worth auditing before committing to option C.
4. Can we use the embedding distance of the answer to the query (or to the documents) as a cheap "relevance score"? Might catch answers that wander too far from any grounded context.

## Related

- PR #9: refusal detector — catches the adjacent class of confident-hedge failures.
- `autodidact/signals/grounded_self_assessment.py` — existing but unwired.
- `Agent.correct()` — existing API for user-signaled corrections; not yet integrated into chat REPL.

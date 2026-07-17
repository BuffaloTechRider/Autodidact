# Overview / Approach — Autodidact v2.0

> **Lifecycle doc 3 of 4.** Chosen approach + rejected alternatives + **why**.
> The "why" is what saves you in 3 months. Draft candidates in Plan Mode.
> Prev: [`01-requirements.md`](01-requirements.md) · Next: [`03-design.md`](03-design.md).

## Chosen approach

Step-level **tiered routing** inside a ReAct execution loop, with a
**skill-learning** feedback path that turns cloud escalations into reusable
stored procedures. Routing signals are model-agnostic (GSA pre-screen,
self-consistency, knowledge verification) with per-category thresholds adapted
online via Thompson Sampling.

Full design: [`03a-design-full.md`](03a-design-full.md).

## Candidate approaches considered

| Approach | Pros | Cons | Verdict |
|----------|------|------|---------|
| Logprob-only confidence (v1) | Simple, already built | Models hallucinate AND refuse at high confidence → misroutes | **Rejected** — insufficient signal |
| Energy-based MLP (ATLAS-style) | Strong published results | Needs per-model training; drifts as memory grows | **Rejected** — retraining + drift violate NFR-3 |
| Model-agnostic signals + adaptive thresholds | No training, no drift, works on fresh models | More moving parts (GSA, self-consistency) | **Chosen** |

## Key decisions (with rationale)

- **Routing (May 2026):** logprob alone fails; use GSA + self-consistency +
  knowledge verification + Thompson-sampled per-category thresholds. No retraining.
- **Retrieval (May 2026):** chunk pages (512-token, section-aware) + USearch over
  chunk vectors + FTS5, RRF at page level. LLM writes knowledge, never indexes it
  (chunk+embed is deterministic). USearch over FAISS (smaller, no native deps).
- **Skills:** SQLite + structured skills with a compact index + on-demand view
  (not Markdown-only), borrowing selectively from Hermes — see `docs/HERMES-LEARNINGS.md`.

## Related prior art

- Hermes agent — loop skeleton, trajectory compression, skill index patterns.
  Adopt/skip decisions captured per-area; see memory `hermes_reference_baseline.md`.

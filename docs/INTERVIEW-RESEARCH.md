# Autodidact — Research Interview Deep-Dive (AI/ML Research Role)

> Prep doc for a 60-min research interview. Emphasis: the *research questions*, the
> *methodology*, the *metrics*, the *ablations*, and — importantly — the *negative
> finding I killed*. Grounded in the real experiment suite (`benchmarks/`, `results/`).
> A research interviewer cares less about the product and more about: did you ask a
> sharp question, did you measure it honestly, and did you know when to stop.
>
> Companion artifact: **`paper/upcoming_papers_plan.md`** (the full Paper A negative +
> Paper B flagship plan). This doc is the interview-facing summary; that's the record.

---

## 0. The research thesis in 60 seconds

> The core research question is **zero-shot confidence estimation for small LLMs**:
> given a cheap local model, can we know — *without any per-model training* — when to
> trust its answer vs. escalate to an expensive model? And a second, more novel one:
> when the local model's *capability grows over time* (because the system accumulates
> memory), the routing problem becomes **non-stationary** — the reward distribution of
> the "answer locally" arm shifts as memory fills. Standard routing theory assumes a
> fixed arm. I formalize this as a **non-stationary multi-armed bandit** and adapt the
> thresholds online.
>
> The empirical backbone is an **ablation over model-agnostic confidence signals** with
> AUROC + bootstrap CIs + calibration diagrams, benchmarked against a RouteLLM-style
> trained baseline. Along the way I ran a viability gate on a promising idea —
> streaming-logprob early-exit — and **killed it with a robust negative finding**, which
> I think is one of the more valuable outcomes.

---

## 1. Research questions (what I'm actually asking)

1. **RQ1 (signal quality):** How well does each *model-agnostic, training-free* signal
   separate correct from incorrect local answers? (AUROC vs `local_correct`.)
2. **RQ2 (fusion):** Does fusing signals beat the best single signal, and does an online
   Thompson fusion beat naive mean fusion — *without* training data?
3. **RQ3 (prior art):** Does training-free fusion approach or beat a RouteLLM-style
   *trained* router, especially one given a knowledge-similarity feature?
4. **RQ4 (non-stationarity):** As memory grows, how do (a) local capability, (b) per-signal
   AUROC, and (c) the optimal threshold evolve? Does a non-stationary Thompson variant beat
   vanilla Thompson?
5. **RQ5 (early exit — killed):** Can early-token logprobs predict final correctness, so we
   route *before* finishing local generation? **Answer: no, for small LLMs.**

---

## 2. Experimental setup (be precise — researchers probe methodology)

- **Local models:** `qwen2.5:7b`, `qwen3:8b` (Ollama). **Cloud:** frontier model, used both
  as the authoritative answerer *and* the LLM-judge for labeling. **Embeddings:**
  `qllama/bge-large-en-v1.5` (1024-dim, local).
- **Datasets:**
  - **MMLU-Pro** (stratified across categories, fixed seed) — the main ablation set.
    Multiple-choice → clean `local_correct` labels via answer matching.
  - **TriviaQA / GSM8K** — used specifically for the streaming study, because MMLU-Pro
    answers are a *single letter* (no token-position curve to measure). Free-form answers
    give a real running-logprob signal **and** free exact-match labels (no judge, $0).
- **Labeling:** MCQ by exact answer match; free-form by exact-match against answer aliases;
  otherwise LLM-judge. I ran a **`label_noise_audit`** because judge/label noise is an
  obvious confound — you can't report AUROC to 3 decimals on noisy labels and pretend the
  labels are clean.
- **Train/eval disjointness:** RouteLLM baselines train on a **disjoint** MMLU-Pro split
  (separate seed) from the eval set — otherwise the "trained baseline" is cheating.
- **Uncertainty:** every AUROC reported with a **1000-sample bootstrap 95% CI**. Point
  estimates without CIs on a few-hundred-query eval set are not defensible.

**The signals evaluated** (`SIGNAL_COLUMNS` in `ablation_analysis.py`):
`knowledge_similarity`, `query_classification`, `energy_scorer`,
`grounded_self_assessment`, `logprob_uncertainty`, `self_consistency`.

---

## 3. Metrics & why each one (know the "why", not just the name)

- **AUROC (vs `local_correct`):** the primary metric — threshold-independent separation of
  correct/incorrect. 0.5 = chance, 1.0 = perfect. *Why AUROC and not accuracy:* routing
  quality is about *ranking* which queries to escalate, and the operating threshold is
  itself a tunable — AUROC evaluates the ranker independent of where you set the cut.
- **Signed AUROC** (in the GSA prompt study): keep the sign so an *inverted* signal (AUROC <
  0.5) is visible rather than hidden. This mattered — see §5.
- **Calibration / reliability diagrams:** AUROC says "can it rank"; calibration says "do the
  probabilities mean what they say." A signal can rank well but be miscalibrated (all scores
  squished into 0.4-0.6), which breaks a fixed-threshold policy. I plot reliability for the
  best combo and the two worst signals.
- **Bootstrap CIs:** to state whether a combo *actually* beats another or is within noise.
- **Retrieval-quality stratification:** split results by `retrieval_recall_at_5` — because
  `knowledge_similarity` and `grounded_self_assessment` are **retrieval-conditional**; if
  retrieval misses, they lose signal, and I need to separate "the signal is bad" from
  "retrieval was bad on this slice."

---

## 4. Key results (from `results/`, real numbers)

### 4.1 Learning curve (the compounding claim, measured)
`learning_curve.json`, 200 queries: **final local-resolution rate 0.765**, **47 escalations
total**, knowledge count grows 1:1 with escalations. Curve starts at **0%** (empty memory,
everything escalates) and climbs to ~77%. This is the *capability-growth curve* that RQ4 is
built on — the local arm demonstrably improves as memory fills.

### 4.2 Thompson vs baselines
`thompson_calibration.json`, 500 queries: **Thompson 0.746 vs fixed-threshold 0.75 vs random
0.504.**
- **Interpretation (state this carefully):** Thompson **matches an oracle-tuned fixed
  threshold** while learning it *online, per-category, without prior knowledge of the right
  value*. The fixed baseline is an *upper bound with hindsight*; Thompson reaching it is the
  result, not a loss. Random at 0.504 confirms routing is non-trivial on this set.
- **The research honesty move:** I don't claim Thompson *beats* fixed here — on a
  *stationary* benchmark it can't, because a well-tuned constant is optimal when nothing
  moves. **The win is in the non-stationary regime (RQ4/Paper B)**, which this stationary
  benchmark doesn't exercise. Saying this unprompted signals I understand my own result.

### 4.3 Ebbinghaus decay
Final precision **1.0 with decay = 1.0 without** on this set. Honest read: decay is a
**memory-hygiene** mechanism (bounds store growth, fades unused facts), not an accuracy
lever — and the experiment confirms it costs no accuracy. I'd *not* oversell it as an
accuracy improvement.

### 4.4 Ablation (the RQ1-RQ3 core)
Per-signal AUROC with bootstrap CIs, per-combo mean-fusion vs Thompson-fusion, and the gap
vs `routellm_no_memory` and `routellm_plus_ks`. Headline framing: **training-free fusion is
competitive with a trained RouteLLM-style router that's even been handed the
knowledge-similarity feature** — which is the whole point (no training, no drift, works on a
fresh model).

---

## 5. The GSA prompt study (a clean example of experimental hygiene)

Great story for a research interviewer — it shows I debug *signals*, not just code.

- **Symptom (LAB_NOTES P7):** the initial GSA prompt made `qwen2.5:7b` answer "NO" to ~99% of
  queries *even ones it could answer correctly* → **AUROC 0.183** — highly informative but
  **inverted** (the model's self-report was anti-correlated with correctness because the
  prompt triggered RLHF hedging).
- **The wrong fix I rejected:** just flip the sign post-hoc (AUROC 0.183 → 0.817). That's
  overfitting to an artifact — the inversion isn't stable across models/prompts.
- **The right fix:** *ask a different question.* Ran a prompt study over multiple variants on
  an MMLU-Pro subset, scored by **positive** AUROC, and chose the wording that made the
  signal informative *the right way up.* Winner emphasizes "specific, factual answer (not a
  hedge or disclaimer)" — directly targeting the refusal failure mode.
- **Lesson:** signed AUROC exposed an inverted signal that unsigned AUROC would have hidden;
  and the fix was upstream (prompt design) not downstream (sign flip).

Also from the ablation lab notes (P9/P10): **injecting *weak* retrieval hits into the GSA
prompt degrades the signal** — priming the model with "here's what you know" when the hits
are junk pushes it toward false YES. So retrieval-conditioning is **all-or-nothing above a
score floor**, not "always inject context." A subtle, measured interaction effect.

---

## 6. The negative finding — Streaming-Logprob Early-Exit (Paper A, killed)

**This is the most research-credible thing I did. Lead with it if they value rigor.**

- **Hypothesis:** if early-token logprobs predict final correctness, we could route *during*
  local generation (early exit) and save compute.
- **Viability gate design:** for each token position K, compute running-mean (and running-min)
  logprob over the first K tokens, AUROC vs `local_correct`, and compare to the full-answer
  `avg_logprob` AUROC (**the ceiling**). Pre-registered decision rule:
  - within 0.05 of ceiling by token 10-15 → viable;
  - needs 50+ tokens → modest, reconsider;
  - never stabilizes → negative, move on.
- **Result:** early-token confidence **does not** predict final correctness for small LLMs.
  Best early-K (K≤15) AUROC ~0.72 on qwen3 vs a ceiling of 0.886 — never within 0.05 before
  generation ends. **The logprob signal is a property of the *whole* generation, not a
  prefix.**
- **The steelman I tested (this is the important part):** a collaborator (Paul) flagged
  **mean-dilution** — maybe naive mean over all tokens buries the signal. So I added
  per-token logprob capture (`ChatResponseWithLogprobs.tokens`) and tested **content-token-only
  mean, bottom-k mean, and predictive entropy.** Content-aware aggregation *does* beat naive
  mean early by ~0.04-0.05 (dilution is real!) — **but still doesn't reach the ceiling before
  generation ends.** Ruling out the obvious fix makes the negative *stronger*, not weaker.
- **Salvage value:** full-generation `avg_logprob` AUROC on free-form (0.84-0.89) **beats**
  MMLU-Pro MCQ (~0.71). Interpretation: single-letter MCQ answers give the logprob almost
  nothing to work with; free-form generation carries real signal. This *reinforces* the main
  design decision — **route AFTER full local generation, not before.**
- **Outcome:** publishable as a short/workshop negative result; and I **pivoted to Paper B as
  the flagship** (which never depended on A). Knowing when to kill a direction is the skill.

---

## 7. Paper B — Non-Stationary Bandit Routing (the flagship contribution)

- **Gap in prior art:** routing/cascade literature (RouteLLM, FrugalGPT, cascades) assumes a
  **fixed** local model. Autodidact's local arm *grows* — memory accumulation changes the
  probability it answers correctly. So the bandit is **non-stationary**: the reward
  distribution of the "local" arm drifts upward (and per-category at different rates).
- **Novelty:** formalize growing-memory routing as a **non-stationary MAB**, derive adaptive
  per-category thresholds, and propose **NS-Thompson Sampling** (e.g. discounted/sliding-window
  posteriors) that tracks the drift instead of averaging over stale history.
- **Five planned experiments:**
  1. **Capability-growth curve** — local accuracy vs memory size (the §4.1 curve, formalized).
  2. **Per-signal AUROC evolution** — does `knowledge_similarity` AUROC *rise* with memory (it
     should — more to verify against) while `logprob` stays flat?
  3. **Threshold adaptation** — does the learned per-category threshold *move* over time?
  4. **NS-TS vs vanilla TS** — the headline: does modeling non-stationarity help vs. a
     stationary bandit that eventually washes out old data too slowly / fast?
  5. **Poisoning robustness** — a wrong cloud answer becomes permanent memory; how badly does
     that corrupt routing, and does decay/verification mitigate it?
- **Target:** EMNLP 2026 / NeurIPS 2027. ~2-3 months, ~$250-400 cloud budget.
- **Why it's genuinely novel:** the non-stationarity here is *endogenous* — the agent's own
  learning causes the drift — which is different from typical non-stationary bandits where the
  environment changes exogenously. That framing is the contribution.

---

## 8. Threats to validity (name them before they do)

- **Label noise:** judge-based labels are imperfect; I ran `label_noise_audit` and report AUROC
  with CIs so a 0.74 vs 0.75 gap isn't over-interpreted.
- **Retrieval confound:** retrieval-conditional signals (`knowledge_similarity`, GSA) inflate/
  deflate with retrieval quality; hence the `recall_at_5` stratification.
- **Small eval sets:** hundreds of queries, not thousands — bootstrap CIs are wide; I state
  effects as "within noise" when they are.
- **Single-domain skew:** MMLU-Pro is academic MCQ; free-form generalization is tested only on
  TriviaQA/GSM8K. Real dev-workflow queries (the product's actual use) are under-sampled in the
  formal benchmarks — a gap I'd close before strong external claims.
- **Model coverage:** two small Qwen models. Cross-model transfer (`routellm_cross_model_transfer`,
  `cross_model_analysis`) partially addresses "does this generalize across local models," but
  two families isn't a strong claim.
- **Stationary benchmark for a non-stationary method:** the Thompson result (§4.2) is on a
  stationary set, which structurally *cannot* show Thompson's advantage — the whole point of
  Paper B is that the interesting regime isn't in the current numbers yet.

---

## 9. Likely research-interviewer questions + answers

**Q: Why AUROC over accuracy or F1?**
A: Routing is a ranking problem — *which* queries to escalate — and the operating threshold is
itself a free parameter (and one I adapt online). AUROC evaluates the ranker independent of the
cut point; accuracy conflates ranker quality with threshold choice.

**Q: Your Thompson result just ties the fixed baseline. So why bother?**
A: On a stationary benchmark a well-tuned constant is optimal — Thompson *can't* beat it, it can
only reach it, which it does *without* being handed the right threshold, per-category, online.
The value shows up under non-stationarity (Paper B): when the local arm's success rate drifts as
memory grows, the "right" fixed threshold changes and a constant goes stale. That regime isn't in
this benchmark, which is exactly why Paper B builds a non-stationary one.

**Q: How do you know your labels are trustworthy?**
A: I don't fully — which is why there's a label-noise audit and every number has a bootstrap CI.
I avoid claiming sub-CI differences. For free-form I prefer exact-match against aliases (no judge)
precisely to remove judge noise where possible.

**Q: The streaming idea failed. Isn't that wasted work?**
A: The opposite — it's a clean, pre-registered negative with the obvious rescue (content-aware
aggregation) tested and ruled out, which strengthens it. And it produced a positive corollary:
full-generation logprob carries far more signal on free-form than MCQ, which validates
"route after full local generation." A negative that sharpens the main design is a good outcome.

**Q: What's actually novel vs. RouteLLM / FrugalGPT / cascades?**
A: Two things. (1) Training-free, model-agnostic multi-signal fusion that's competitive with a
*trained* router given the knowledge feature — no per-model training, no drift. (2) The
non-stationary framing: prior work assumes a fixed local model; here the local arm's competence
is *endogenously* growing because the system learns from its own escalations. Formalizing and
handling that drift is the contribution.

**Q: What would break your conclusions?**
A: If the capability-growth curve is an artifact of query repetition rather than genuine
generalization (memorizing exact queries vs learning transferable knowledge) — I'd test with
held-out *paraphrased* queries. And if per-signal AUROC *doesn't* evolve with memory, the
non-stationary premise weakens; experiment 2 in Paper B is designed to falsify exactly that.

**Q: How would you scale this study to be publication-strong?**
A: More models (3+ families, 2+ sizes each), larger eval (thousands of queries), a
domain-realistic query set beyond academic MCQ, paraphrase-held-out capability curves, and the
full NS-TS vs TS comparison with regret curves — not just final accuracy.

---

## 10. One-liners (research flavor)

- *"Logprob measures token predictability, not correctness — that's why single-signal routing fails."*
- *"The routing problem is non-stationary because the agent's own learning grows the local arm — prior work assumes a fixed model."*
- *"Signed AUROC caught an inverted signal that unsigned would have hidden — the fix was the prompt, not a sign flip."*
- *"Ruling out the obvious rescue makes a negative result stronger, not weaker."*
- *"Thompson doesn't beat a hand-tuned threshold on a stationary set — it reaches it without the hand-tuning, which is the point."*
- *"Full-generation logprob beats early-token logprob AND beats MCQ logprob — route after generating, on free-form."*

---

## Appendix: experiment → file map (so I can point to real code)

| Experiment | Script | Output |
|---|---|---|
| Per-signal + fusion ablation, calibration, RouteLLM gap | `benchmarks/ablation_analysis.py` | `results/summary.json`, memo |
| GSA prompt study (signed AUROC) | `benchmarks/gsa_prompt_study.py` | per-variant table |
| GSA retrieval-conditioning rerun | `benchmarks/gsa_retrieval_rerun.py` | — |
| Learning / capability-growth curve | `benchmarks/learning_curve.py`, `learning_curve_v2.py` | `results/learning_curve.json/.png` |
| Thompson vs fixed vs random | `benchmarks/thompson_calibration.py` | `results/thompson_calibration.json/.png` |
| Ebbinghaus decay | `benchmarks/ebbinghaus_decay.py` | `results/ebbinghaus_decay.json/.png` |
| Streaming early-exit viability (Paper A, negative) | `benchmarks/streaming_viability.py`, `streaming_logprob_collect.py` | curve + summary |
| RouteLLM baselines + transfer | `benchmarks/routellm_baseline.py`, `routellm_cross_model_transfer.py` | — |
| Threshold sweep | `benchmarks/threshold_sweep_study.py` | — |
| Label-noise audit | `benchmarks/label_noise_audit.py` | — |
| Cross-model analysis | `benchmarks/cross_model_analysis.py` | — |

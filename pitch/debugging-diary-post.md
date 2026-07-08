# Five things I got wrong building a confidence evaluator for local LLMs

*An in-progress lab-notebook post from the Autodidact project.*

I've been building [Autodidact](https://github.com/BuffaloTechRider/EvoAgent), a local-first AI agent framework. The central piece is a **confidence evaluator** — something that decides whether a small local model (Qwen 2.5 7B, Llama 3.1 8B, Mistral 7B) can answer a question, or whether to escalate to a cloud model (Claude 4.5, GPT, etc.).

If it works, you get cheap local inference most of the time and cloud only when needed. If it doesn't, you either hallucinate (local answers wrong questions) or escalate everything (pay cloud prices for no benefit).

I spent the last two weeks building this, measuring it on [MMLU-Pro](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro), and debugging surprises. This post is the findings I wish I'd known before starting. All numbers here are from real runs against Ollama + AWS Bedrock; code and raw data are in the repo.

None of what follows is final. The v0.1 main experiment is still running at the time of writing; the blog post where I claim "we've measured X and Y on 3 models with 95% CIs" will come in a few weeks. This post is the journey.

## The setup in one paragraph

Query comes in → compute six "confidence signals" (logprob uncertainty, self-consistency across two samples, knowledge-similarity against a growing vector store, a lightweight query classifier, a learned "energy" score on query embeddings, and a grounded self-assessment where the model is asked YES/NO whether it can answer). Fuse them via Thompson Sampling. If the fused score is above a threshold, answer local; otherwise escalate to cloud and store the answer for next time. The idea isn't mine — versions of each signal are in [Kadavath et al.](https://arxiv.org/abs/2207.05221), [Wang et al. self-consistency](https://arxiv.org/abs/2203.11171), and [RouteLLM](https://arxiv.org/abs/2406.18665). What I'm doing is composing them and measuring what actually works.

Here's what I got wrong.

## 1. Ollama silently changed its logprob API

The first week, my logprob-based confidence signal had AUROC 0.500 with a zero-width confidence interval across 50 queries. Every single query produced the same value. Dead signal.

Turns out: my code passed `options.include_logprobs = true` inside the Ollama request body (the old way). Ollama 0.12.11+ moved logprob support to **top-level** `logprobs: true` and `top_logprobs: N` fields. Older versions silently ignore unknown parameters. I wasn't getting an error; I was just getting every response with no logprobs, hitting my graceful fallback for every query.

The fix was five lines. The damage was spending a day debugging "is the signal broken or just weak?"

**Lesson:** Probe every critical LLM API field end-to-end against your specific backend version BEFORE running a multi-hour experiment. My preflight script now fires a single "YES or NO" probe and asserts `top_logprobs_by_position[0]` has at least two distinct tokens. 30 seconds saved 4 hours.

If you're using Ollama for anything that needs logprobs (uncertainty estimation, hallucination detection, confidence routing), check your version: `curl -s http://localhost:11434/api/version` and ensure it's ≥ 0.12.11.

## 2. "Grounding" self-assessment on retrieval actively hurt it

I had a signal called **Grounded Self-Assessment** (GSA): before answering, show the model what its knowledge store retrieved for this query, then ask "do you have enough information to answer correctly?" The intuition: the model will say YES when retrieved context is good, NO when it's missing.

Sounds right. Also wrong.

I ran an ablation with four prompt variants on qwen2.5:7b, with vs. without retrieved hits injected. The "with hits" condition had AUROC **0.538**. The "without hits" condition had AUROC **0.620**. Injecting retrieved context **lowered** the signal by 0.08 AUROC.

Why? I looked at the 5 queries where retrieval returned hits above the 0.75 similarity threshold. On two queries where the model would have answered CORRECTLY without retrieval, injecting retrieved content caused the model to say "NO, I don't have enough information." The retrieval was tangentially relevant — it didn't contain the specific answer — and the model interpreted its presence as a cue to look for the answer in the retrieval, didn't find it, and said NO.

The model had the answer in its weights. The retrieval made it second-guess itself.

**Lesson:** Retrieval-augmented confidence isn't free. "Does this retrieved content help?" is different from "Do I know the answer?" If your prompt emphasizes the former, you're measuring retrieval quality, not model knowledge. For my use case, dropping retrieval injection from the confidence prompt (keeping it only in the main-answer prompt, separately) was the right call.

This has implications beyond my project. Many RAG systems inject retrieved context into "should I answer" gates. That might be hurting them.

## 3. The winning prompt varies by model family

After fixing GSA, I ran the prompt ablation on three models: qwen2.5:7b, llama3.1:8b, mistral:7b-instruct. Four prompt variants each, 100 queries each, no retrieval injection.

| Model | "current" | "direct" | "confidence" | "prediction" |
|---|---|---|---|---|
| qwen2.5:7b | 0.470 | 0.591 | **0.636** | 0.525 |
| llama3.1:8b | **0.545** | 0.535 | 0.519 | 0.515 |
| mistral:7b-instruct | 0.562 | **0.747** | 0.684 | 0.657 |

Bolded is the winner per row. Three different winning prompts across three models. And Llama's "winner" at 0.545 is barely above chance; all four variants fall inside the bootstrap CI of each other. Llama has essentially **no self-assessment signal at all** at this model size with any of these prompts.

Mistral is the most interesting. AUROC 0.747 with the "direct" prompt — the highest single-signal AUROC I've measured. Same prompt gives 0.591 on Qwen and 0.535 on Llama. The model matters, and it matters a lot.

**Lesson:** Don't ship "here's the prompt that works for self-assessment." Ship "here's a short probe you run on install to pick the prompt." My v0.1 now does a ~10-minute mini-study on first boot to pick the local model's best prompt. It's the honest thing to do.

This is also consistent with the [Kadavath et al.](https://arxiv.org/abs/2207.05221) observation that calibrated self-assessment depends heavily on RLHF program. Qwen's post-training includes self-critique data. Mistral's does something else. Llama 3.1 seems to not calibrate its confidence at all — it's been trained for helpfulness and harmlessness, not for honest uncertainty.

If you care about LLM calibration, **pick your base model for calibration, not just capability.** Mistral-7B-Instruct punches above its weight here.

## 4. Retrieval quality is my actual bottleneck, not the LLM

I ran an answer-quality study: 60 queries × 3 models × 2 conditions (with retrieval injected into the answer prompt vs without). The delta was:

| Model | Delta (with − without) | p-value |
|---|---|---|
| qwen2.5:7b | +0.017 | 1.000 |
| llama3.1:8b | +0.033 | 0.625 |
| mistral:7b-instruct | -0.017 | 1.000 |

All statistically a wash at n=60. My first interpretation: "retrieval injection doesn't help answer quality."

That's what the data says. But look at `retrieval_recall_at_5`: **10 out of 60 queries actually had any retrieval hit** above the 0.75 similarity threshold. Across ALL three models. Same number because retrieval depends on the embedding model, not the LLM.

I'm measuring the retrieval-helps-answer effect on n=10 samples. That's not underpowered; that's uninterpretable.

The knowledge store has 500 entries. MMLU-Pro has ~14 subject categories. Most eval queries have zero category-matched entries in the store. The embedder is nomic-embed-text. Cross-encoder reranking: none. Question-to-answer rewriting (HyDE): none. It's an entry-level RAG.

**Lesson:** Before concluding "retrieval doesn't help," check whether retrieval is even happening. When it's not, you're measuring the noise floor of your LLM's hallucination rate.

The next step in the project is upgrading the retrieval pipeline (bge-large-en-v1.5 embedder + bge-reranker-base cross-encoder) and re-measuring. I have three specific predictions I'll report back on: retrieval_recall_at_5 should rise from 17% to ≥40%, knowledge-similarity AUROC should rise from 0.484 to ≥0.60, and the retrieval-injection answer-quality delta should finally swing positive.

If those predictions all hold, the confidence evaluator story gets much stronger. If they don't, I need to rethink.

## 5. Multi-signal ensembles are not automatically better

I put a lot of design effort into Thompson Sampling fusion — the Bayesian bandit that combines all six confidence signals into one routing decision. Surely the ensemble beats any single signal.

Not so much. On qwen2.5:7b:
- Single signal (logprob_uncertainty): AUROC 0.642
- Six-signal mean ensemble: AUROC 0.615
- Six-signal Thompson fusion: AUROC 0.564

The ensemble is WORSE than the best single signal, because half of my signals are near-chance (query classification keyword-based, weakly correlated with correctness) and they dilute the good signals.

This shouldn't have surprised me. It's a classic "averaging a good signal with noise gives you a worse signal" trap. Thompson Sampling eventually learns to downweight the bad signals, but the bootstrap period is long, and on a 1000-query benchmark the bandit hasn't converged yet.

**Lesson:** Simple baselines matter. Before adding ensemble fusion, measure whether one signal alone already wins. If it does, the ensemble's job is to NOT make things worse — a much weaker claim than "the ensemble is better."

The v0.1 main experiment will retest this on three models with proper statistical power. I'm not confident Thompson fusion beats single-signal, and the paper I want to write has to contend with that.

## What I still don't know

Honest list:

1. **Does Mistral's 0.747 AUROC replicate at n=1000?** My measurement is at n=100. Bootstrap CI is ±0.09. The real value could be anywhere from 0.65 to 0.84.
2. **Does logprob_uncertainty generalize cross-model?** Measured on Qwen only so far. Theory says yes (calibration theory is model-agnostic); data hasn't confirmed.
3. **Does retrieval quality actually matter for the confidence story, or is the embedder we use a red herring?** The answer is the v0.1 main experiment's gate.
4. **Does a growing knowledge base change signal quality over time?** This is my actual research question, deferred to v0.2. The idea: as the local model's effective memory grows via cloud escalation, the signals should become more discriminating. If they don't, the whole "local agent that gets smarter over time" story is weaker than I'd like.
5. **Is Thompson Sampling the right fusion strategy for signals that might be correlated?** Classical TS assumes independent arms. Our signals are almost certainly correlated (good self-assessment and good logprobs probably co-occur). The math gets messier here than the paper implies.

## What's next

Three-week plan: finish the retrieval upgrade, re-measure the three predictions above, then run the main experiment across all 3 models at n=1000 with RouteLLM-style baselines. That produces the data for a technical report on GitHub. If it works, v0.2 runs the same experiment with growing memory (the thing I actually want to study) and probably becomes a workshop or short paper submission.

The code is in [github.com/BuffaloTechRider/EvoAgent](https://github.com/BuffaloTechRider/EvoAgent). The debugging notes are in `results/experiment/LAB_NOTES.md` in the repo — more honest than this post, less polished.

If you've built something adjacent (LLM routing, confidence estimation, RAG for local models), I'd genuinely like to hear where your findings agree or disagree with mine. Email [your email] or DM on [your handle]. The next post will land with real cross-model numbers when the main experiment finishes.

---

*Autodidact is solo-maintained and under active construction. Nothing here is advice. It's a lab notebook made public so people can push back on it.*

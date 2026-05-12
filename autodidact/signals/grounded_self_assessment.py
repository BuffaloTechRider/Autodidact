"""Self-Assessment signal.

A pre-response confidence signal. The local model is asked a single Y/N
question about whether it can answer the user's query. The YES-token
probability is the signal value. Cheap — one generation of a single token.

## Versions

### v3 (current default) — retrieval-conditional prompting

Prompt version: ``gsa-v3-retrieval-conditional``.

When a caller passes ``retrieved_hits`` AND at least one hit's score clears the
``min_similarity`` threshold (default 0.70), the prompt includes the strong
hits as recalled memory. Otherwise the prompt is IDENTICAL to the bare v2
variant — no mention of retrieval, no "no relevant knowledge retrieved" text,
nothing the model can distinguish from a query that just never had retrieval.
This design choice is critical: EXP-005 and P9/P10 both showed that telling
the model "no knowledge was retrieved" primes it toward NO.

EXP-005 (n=931, qwen2.5:7b) validated this design:
  - ``gsa_v3_070`` AUROC = 0.599 (strong hits shown on 31% of queries)
  - ``gsa`` v2 baseline AUROC = 0.562 (always bare)
  - ``gsa_v3_060`` AUROC = 0.511 (strong hits shown on 83% — marginal hits hurt)

### v2 — bare prompt always (legacy)

Prompt version: ``gsa-v2-confidence``.

Pre-dates the retrieval-quality upgrade (Task 13). Reachable via the
constructor flag ``use_v2_legacy=True`` for replicating pre-v3 runs.

Rationale for v2: an earlier v1 design injected top-k hits ALWAYS (with a
"(no relevant knowledge retrieved)" placeholder when empty) and that
degraded signal quality — see P9/P10 in LAB_NOTES. v2 stripped retrieval
entirely; v3 puts it back but ONLY when strong AND ALWAYS with indistinguishable
fallback.

## Signal extraction

Three-tier fallback (shared by v2 and v3):
1. **Logprob softmax (preferred).** If the backend returns top-logprobs,
   compute p_yes = exp(YES) / (exp(YES) + exp(NO)) from the first generated
   position.
2. **Text hard label (Bedrock fallback).** If logprobs are unavailable but
   the raw token says YES or NO, return 1.0 or 0.0.
3. **Neutral 0.5.** Both failed; carries no information for this query.

Requirement 2 from the v0.1 spec is satisfied here.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional

from autodidact.llm_client import ChatMessage, LLMClient
from autodidact.types import KnowledgeEntry

logger = logging.getLogger(__name__)

# Default similarity threshold for showing retrieved hits in the v3 prompt.
# Below this score, the prompt falls back to the bare v2 form. EXP-005 found
# 0.70 beats 0.60 on qwen2.5:7b + bge-large + MMLU-Pro.
DEFAULT_MIN_SIMILARITY_V3 = 0.70

# Bumped when the default prompt behavior changes. Old runs keep their label
# so we can replicate or compare across versions.
PROMPT_VERSION = "gsa-v3-retrieval-conditional"
PROMPT_VERSION_V2 = "gsa-v2-confidence"
PROMPT_VERSION_V4 = "gsa-v4-adversarial-trust"

# The bare prompt. Used both by v2 (always) and by v3 when no hits clear the
# threshold. CRITICAL: the bare prompt is identical in both cases so the
# model cannot distinguish "retrieval was run and returned nothing strong"
# from "retrieval was never attempted."
BARE_PROMPT_TEMPLATE = (
    "Question from the user:\n"
    "{query}\n\n"
    "Do you have specific, reliable knowledge to answer this question? "
    "Answering NO when uncertain is the right choice. "
    "The user would rather get a cloud-backed answer than a confidently-wrong local one. "
    "Respond with exactly one token: YES or NO."
)

# The with-retrieval prompt. Used by v3 when ≥1 hit clears min_similarity.
WITH_RETRIEVAL_PROMPT_TEMPLATE = (
    "Question from the user:\n"
    "{query}\n\n"
    "Here is what you recall from your knowledge base:\n"
    "{hits_block}\n\n"
    "Do you have specific, reliable knowledge to answer this question? "
    "Answering NO when uncertain is the right choice. "
    "The user would rather get a cloud-backed answer than a confidently-wrong local one. "
    "Respond with exactly one token: YES or NO."
)

# ── v4: adversarial trust framing (opt-in) ────────────────────────
# Uses explicit cost-of-wrong-answer language to counteract sycophancy bias
# toward YES. Kept opt-in until experimentally validated against v3 on the
# same benchmark (EXP-005 equivalent). Default remains v3.
BARE_PROMPT_TEMPLATE_V4 = (
    "Question from the user:\n"
    "{query}\n\n"
    "If you say YES and your answer turns out to be wrong, the user loses "
    "trust. Say NO unless you are confident.\n\n"
    "Respond with exactly one token: YES or NO."
)

WITH_RETRIEVAL_PROMPT_TEMPLATE_V4 = (
    "Question from the user:\n"
    "{query}\n\n"
    "Here is what you recall from your knowledge base:\n"
    "{hits_block}\n\n"
    "If you say YES and your answer turns out to be wrong, the user loses "
    "trust. Say NO unless you are confident.\n\n"
    "Respond with exactly one token: YES or NO."
)

# Back-compat alias: earlier code imported `PROMPT_TEMPLATE` as the bare form.
PROMPT_TEMPLATE = BARE_PROMPT_TEMPLATE

# Accept a wide-ish set of tokens because tokenizers vary.
_YES_TOKENS = {"YES", "Yes", "yes", "Y", "y", " YES", " Yes", " yes", " Y", " y"}
_NO_TOKENS = {"NO", "No", "no", "N", "n", " NO", " No", " no", " N", " n"}


@dataclass
class SelfAssessmentResult:
    """Return type carrying both the scalar signal and debug info.

    Internal-only; see the Pydantic-vs-dataclass convention in autodidact/types.py.
    """

    p_yes: float
    yes_logprob: Optional[float]
    no_logprob: Optional[float]
    extraction_mode: str  # 'logprob_softmax' | 'text_hard' | 'neutral'
    recognized: bool       # True when extraction_mode != 'neutral'
    raw_response: str
    # v3 bookkeeping: whether the prompt included retrieval and how many hits it used.
    # v2 always reports (False, 0).
    had_retrieval: bool = False
    n_hits_used: int = 0


# Backwards-compat alias. Existing code that imports GroundedSelfAssessmentResult
# keeps working.
GroundedSelfAssessmentResult = SelfAssessmentResult


class SelfAssessment:
    """Prompt the local model with a single Y/N self-confidence probe.

    By default runs the v3 retrieval-conditional prompt. Two knobs control
    which prompt template is used:

    - ``prompt_version="v4"`` switches to the adversarial-trust prompt, which
      uses cost-of-wrong-answer framing to counteract sycophancy-driven YES bias.
      Opt-in until experimentally validated.
    - ``use_v2_legacy=True`` reproduces v2 runs (bare prompt always, retrieved
      hits silently ignored). Equivalent to ``prompt_version="v2"``.

    Parameters
    ----------
    llm_client : LLMClient
        Must be configured against the LOCAL model. Logprob access is strongly
        preferred; if unavailable, the signal gracefully degrades through the
        three-tier fallback described in the module docstring.
    min_similarity : float
        Similarity threshold above which a retrieved hit is considered "strong
        enough" to include in the prompt. Defaults to 0.70 per EXP-005.
    use_v2_legacy : bool
        If True, always use the bare v2 prompt. Mutually exclusive with
        ``prompt_version``. Kept for backwards compatibility.
    prompt_version : str | None
        One of {"v2", "v3", "v4"} or None. None defaults to "v3".
    """

    def __init__(
        self,
        llm_client: LLMClient,
        min_similarity: float = DEFAULT_MIN_SIMILARITY_V3,
        use_v2_legacy: bool = False,
        prompt_version: Optional[str] = None,
    ) -> None:
        if prompt_version is not None and use_v2_legacy:
            raise ValueError(
                "Pass either prompt_version or use_v2_legacy, not both."
            )
        if prompt_version is not None and prompt_version not in ("v2", "v3", "v4"):
            raise ValueError(
                f"prompt_version must be one of 'v2', 'v3', 'v4'; got {prompt_version!r}"
            )
        self.llm_client = llm_client
        self.min_similarity = float(min_similarity)
        if use_v2_legacy:
            self._version = "v2"
        elif prompt_version is not None:
            self._version = prompt_version
        else:
            self._version = "v3"
        # Preserve legacy attribute so existing callers reading it still work.
        self.use_v2_legacy = self._version == "v2"

    @property
    def prompt_version(self) -> str:
        """Return the prompt version string persisted to experiment rows."""
        if self._version == "v2":
            return PROMPT_VERSION_V2
        if self._version == "v4":
            return PROMPT_VERSION_V4
        return PROMPT_VERSION  # v3 default

    def compute(
        self,
        query: str,
        retrieved_hits: Optional[list] = None,
    ) -> SelfAssessmentResult:
        """Return the YES probability for the self-confidence probe.

        ``retrieved_hits`` accepts either a list of ``KnowledgeEntry`` (v2
        call sites) or a list of ``ScoredKnowledgeEntry`` (v3 call sites
        using retrieval-conditional prompting). When v2-legacy mode is on,
        the argument is ignored entirely.
        """
        prompt, had_retrieval, n_hits_used = self._build_prompt(query, retrieved_hits)
        messages = [ChatMessage(role="user", content=prompt)]
        response = self.llm_client.chat_with_logprobs(
            messages,
            max_tokens=1,
            temperature=0.0,
            top_logprobs=5,
        )
        raw = response.content

        # Tier 1 — logprob softmax
        if response.top_logprobs_by_position:
            first_pos = response.top_logprobs_by_position[0]
            yes_lp = self._first_matching_logprob(first_pos, _YES_TOKENS)
            no_lp = self._first_matching_logprob(first_pos, _NO_TOKENS)
            if yes_lp is not None or no_lp is not None:
                missing_lp = -20.0
                yes_eff = yes_lp if yes_lp is not None else missing_lp
                no_eff = no_lp if no_lp is not None else missing_lp
                max_lp = max(yes_eff, no_eff)
                yes_exp = math.exp(yes_eff - max_lp)
                no_exp = math.exp(no_eff - max_lp)
                p_yes = yes_exp / (yes_exp + no_exp)
                p_yes = max(0.0, min(1.0, p_yes))
                return SelfAssessmentResult(
                    p_yes=p_yes,
                    yes_logprob=yes_lp,
                    no_logprob=no_lp,
                    extraction_mode="logprob_softmax",
                    recognized=True,
                    raw_response=raw,
                    had_retrieval=had_retrieval,
                    n_hits_used=n_hits_used,
                )
            logger.debug(
                "SelfAssessment: top-logprobs available but no YES/NO token; falling through to text (%r)",
                raw[:40],
            )

        # Tier 2 — text hard label
        text_label = self._parse_yes_no_from_text(raw)
        if text_label is not None:
            return SelfAssessmentResult(
                p_yes=1.0 if text_label else 0.0,
                yes_logprob=None,
                no_logprob=None,
                extraction_mode="text_hard",
                recognized=True,
                raw_response=raw,
                had_retrieval=had_retrieval,
                n_hits_used=n_hits_used,
            )

        # Tier 3 — neutral fallback
        logger.warning(
            "SelfAssessment: could not extract YES/NO from response=%r; returning neutral 0.5",
            raw[:40],
        )
        return SelfAssessmentResult(
            p_yes=0.5,
            yes_logprob=None,
            no_logprob=None,
            extraction_mode="neutral",
            recognized=False,
            raw_response=raw,
            had_retrieval=had_retrieval,
            n_hits_used=n_hits_used,
        )

    # ── Internal helpers ─────────────────────────────────────────

    def _build_prompt(
        self, query: str, retrieved_hits: Optional[list]
    ) -> tuple[str, bool, int]:
        """Choose the prompt template and render it.

        Returns (prompt_text, had_retrieval, n_hits_used). ``had_retrieval``
        is True when the with-retrieval template was used; False when the
        bare fallback was used. This lets callers audit how often retrieval
        was actually shown at runtime.
        """
        if self._version == "v2":
            if retrieved_hits:
                logger.debug(
                    "SelfAssessment (v2): ignoring %d retrieved hits",
                    len(retrieved_hits),
                )
            return (BARE_PROMPT_TEMPLATE.format(query=query.strip()), False, 0)

        # v3 and v4 are both retrieval-conditional. Pick templates per version.
        if self._version == "v4":
            bare_tpl = BARE_PROMPT_TEMPLATE_V4
            with_retr_tpl = WITH_RETRIEVAL_PROMPT_TEMPLATE_V4
        else:
            bare_tpl = BARE_PROMPT_TEMPLATE
            with_retr_tpl = WITH_RETRIEVAL_PROMPT_TEMPLATE

        strong_hits = self._filter_strong_hits(retrieved_hits)
        if strong_hits:
            hits_block = _render_hits_block(strong_hits)
            prompt = with_retr_tpl.format(query=query.strip(), hits_block=hits_block)
            return (prompt, True, len(strong_hits))

        # No strong hits — bare prompt, indistinguishable from "never-searched".
        return (bare_tpl.format(query=query.strip()), False, 0)

    def _filter_strong_hits(self, retrieved_hits: Optional[list]) -> list:
        """Return only hits whose score is above min_similarity.

        Accepts either ``ScoredKnowledgeEntry`` (has ``.score`` and ``.entry``)
        or bare ``KnowledgeEntry`` (no score — in that case we can't filter,
        so we treat them all as "no score known" and fall back to the bare
        prompt to be safe).
        """
        if not retrieved_hits:
            return []
        strong: list = []
        for h in retrieved_hits:
            score = getattr(h, "score", None)
            if score is None:
                # v2-style caller passed a KnowledgeEntry list. No score info
                # means we can't enforce threshold — treat as "don't show" so
                # the bare prompt is used and the model isn't primed either way.
                continue
            if score >= self.min_similarity:
                strong.append(h)
        return strong

    @staticmethod
    def _parse_yes_no_from_text(text: str) -> Optional[bool]:
        """Return True for YES, False for NO, None if unparseable."""
        if not text:
            return None
        stripped = text.strip().strip(".,;:!?)('\"`").upper()
        if not stripped:
            return None
        first_word = stripped.split()[0] if stripped.split() else stripped
        if first_word in {"YES", "Y"}:
            return True
        if first_word in {"NO", "N"}:
            return False
        if stripped.startswith("YES") or stripped.startswith("Y "):
            return True
        if stripped.startswith("NO") or stripped.startswith("N "):
            return False
        return None

    @staticmethod
    def _first_matching_logprob(
        logprobs_at_pos: dict[str, float], candidates: set[str]
    ) -> Optional[float]:
        """Return the logprob of the first candidate token present, or None."""
        for token, lp in logprobs_at_pos.items():
            if token in candidates:
                return float(lp)
        for token, lp in logprobs_at_pos.items():
            stripped = token.strip().upper()
            if stripped in {"YES", "Y"} and candidates is _YES_TOKENS:
                return float(lp)
            if stripped in {"NO", "N"} and candidates is _NO_TOKENS:
                return float(lp)
        return None


def _render_hits_block(hits: list) -> str:
    """Format a list of ScoredKnowledgeEntry into a numbered block for the prompt."""
    lines: list[str] = []
    for i, h in enumerate(hits, start=1):
        entry = getattr(h, "entry", h)  # accept ScoredKnowledgeEntry or KnowledgeEntry
        content = (getattr(entry, "content", "") or "")[:400].strip()
        q = getattr(entry, "question", None)
        if q:
            lines.append(f"{i}. (memory of: {q.strip()[:120]})\n   {content}")
        else:
            lines.append(f"{i}. {content}")
    return "\n".join(lines)


# Backwards-compat alias for callers that still import the v1 class name.
GroundedSelfAssessment = SelfAssessment

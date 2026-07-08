"""Correctness labeling for MMLU-Pro responses.

Two strategies, applied in order:
1. Regex letter extraction from the response. Covers most well-formed responses.
2. LLM-as-judge fallback via Bedrock (or any LLMClient) for ambiguous free-form responses.

The label function returns (correct: bool, used_judge: bool) so the harness can
track how often the expensive judge path was needed.

Requirement 5 (R5) from the v0.1 spec is satisfied here.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from autodidact.llm_client import ChatMessage, LLMClient

logger = logging.getLogger(__name__)


# Regex patterns tried in order. First match wins.
# MMLU-Pro has up to 10 options (A-J), hence the character class.
_LETTER_PATTERNS: list[re.Pattern] = [
    # "The answer is (X)" / "Answer: X" / "answer is X"
    re.compile(r"(?i)answer\s*(?:is|:)\s*\(?\s*([A-J])\b"),
    # "(X) is correct" / "X is correct"
    re.compile(r"(?i)\(?\s*([A-J])\s*\)?\s+is\s+(?:the\s+)?correct"),
    # Standalone letter at end of the response, possibly with trailing punctuation
    re.compile(r"(?im)^\s*\(?([A-J])\)?\s*[\.\)\]]?\s*$"),
    # Any final "... X." or "... X)" on the last line
    re.compile(r"(?m)\b([A-J])\b\s*[\.\)\]]\s*$"),
    # Very loose: first standalone A-J token wrapped in punctuation
    re.compile(r"(?:^|\s)\(([A-J])\)(?:\s|$|[\.,:;])"),
]


def extract_letter(response_text: str) -> Optional[str]:
    """Extract the answer letter from a model response.

    Returns the uppercase letter (A-J) on success, None otherwise.
    """
    if not response_text:
        return None
    text = response_text.strip()
    # Early fast path: single-letter response.
    if len(text) == 1 and text.upper() in "ABCDEFGHIJ":
        return text.upper()
    for pattern in _LETTER_PATTERNS:
        m = pattern.search(text)
        if m:
            return m.group(1).upper()
    return None


def label_by_letter_match(response_text: str, ground_truth_letter: str) -> Optional[int]:
    """Score via letter extraction. Returns 1/0, or None if extraction failed."""
    extracted = extract_letter(response_text)
    if extracted is None:
        return None
    return 1 if extracted == ground_truth_letter.upper() else 0


_JUDGE_PROMPT = """You are a strict grader. Decide whether the MODEL_ANSWER correctly answers the QUESTION.

The correct answer is letter {letter}, which corresponds to: "{ground_truth_answer}"

QUESTION:
{question}

MODEL_ANSWER:
{model_answer}

Respond with exactly one token: YES or NO."""


def label_by_judge(
    judge_client: LLMClient,
    question: str,
    model_answer: str,
    ground_truth_letter: str,
    ground_truth_answer: str,
) -> int:
    """Score via LLM-as-judge. Returns 1 (correct) or 0 (incorrect)."""
    prompt = _JUDGE_PROMPT.format(
        letter=ground_truth_letter.upper(),
        ground_truth_answer=ground_truth_answer,
        question=question.strip(),
        model_answer=(model_answer or "").strip(),
    )
    resp = judge_client.chat(
        [ChatMessage(role="user", content=prompt)],
        max_tokens=4,
        temperature=0.0,
    )
    text = (resp.content or "").strip().upper()
    if text.startswith("YES"):
        return 1
    if text.startswith("NO"):
        return 0
    logger.warning(
        "Judge returned ambiguous response %r for question %r; defaulting to 0",
        text[:40], question[:80],
    )
    return 0


def label_answer(
    response_text: str,
    ground_truth_letter: str,
    ground_truth_answer: str,
    question: str,
    judge_client: Optional[LLMClient] = None,
) -> tuple[int, bool]:
    """Label one response. Returns (correct 0/1, used_judge bool).

    Tries letter extraction first; falls back to LLM-as-judge if extraction fails
    and a judge client is provided. If letter extraction fails and no judge is
    available, returns (0, False) and logs a warning.
    """
    by_letter = label_by_letter_match(response_text, ground_truth_letter)
    if by_letter is not None:
        return by_letter, False
    if judge_client is None:
        logger.warning(
            "Letter extraction failed and no judge configured; scoring as incorrect. "
            "Response excerpt: %r",
            (response_text or "")[:80],
        )
        return 0, False
    score = label_by_judge(
        judge_client, question, response_text, ground_truth_letter, ground_truth_answer
    )
    return score, True

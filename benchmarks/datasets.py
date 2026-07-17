"""MMLU-Pro dataset loader for the v0.1 ablation experiment.

Two subsets used:
- Experiment_Corpus (seed 42, 500 queries) — first 100 for knowledge seeding, remaining 400 for evaluation
- Training_Corpus (seed 43, 1000 queries) — for the RouteLLM baseline classifiers

The two subsets MUST be disjoint by query_id. The `load_two_disjoint_subsets`
helper enforces that with an assertion.
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class MMLUProQuery:
    """One MMLU-Pro question with everything we need downstream."""

    query_id: str
    query_text: str
    ground_truth_letter: str
    ground_truth_answer: str
    category: str
    options: list[str]

    def formatted_prompt(self) -> str:
        """Canonical prompt format for local and cloud models to answer."""
        options_block = "\n".join(f"{chr(65 + i)}. {opt}" for i, opt in enumerate(self.options))
        return f"""Answer the following multiple-choice question. Respond with the letter of the correct option.

Question: {self.query_text}

Options:
{options_block}

Answer:"""


def load_mmlu_pro_subset(
    n: int,
    seed: int,
    split: str = "test",
) -> list[MMLUProQuery]:
    """Load n MMLU-Pro queries stratified across categories with a fixed seed.

    Requires `datasets` package (HuggingFace). The dataset is cached locally
    by the `datasets` library after first download.
    """
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "The `datasets` package is required for MMLU-Pro loading. "
            "Install with `pip install datasets`."
        ) from e

    raw = load_dataset("TIGER-Lab/MMLU-Pro", split=split)

    # Group by category for stratified sampling.
    by_category: dict[str, list[dict]] = defaultdict(list)
    for item in raw:
        by_category[item["category"]].append(item)

    categories = sorted(by_category.keys())
    if not categories:
        raise RuntimeError(f"MMLU-Pro split '{split}' returned no categories")

    # Stratified round-robin sampling: in each category shuffle deterministically,
    # then round-robin pick until we have n.
    rng = random.Random(seed)
    shuffled: dict[str, list[dict]] = {}
    for cat in categories:
        items = list(by_category[cat])
        rng.shuffle(items)
        shuffled[cat] = items

    picked: list[dict] = []
    i = 0
    while len(picked) < n:
        progress = False
        for cat in categories:
            if i < len(shuffled[cat]):
                picked.append(shuffled[cat][i])
                progress = True
                if len(picked) >= n:
                    break
        if not progress:
            break
        i += 1

    if len(picked) < n:
        raise RuntimeError(
            f"Requested {n} queries but MMLU-Pro split '{split}' only has {len(picked)}"
        )

    # Convert to MMLUProQuery objects.
    out: list[MMLUProQuery] = []
    for item in picked:
        options = list(item.get("options") or [])
        answer_index = int(item.get("answer_index", 0) or 0)
        if not options or answer_index >= len(options):
            continue
        ground_truth_letter = chr(65 + answer_index)
        ground_truth_answer = str(options[answer_index])
        q_id = _derive_query_id(item)
        out.append(
            MMLUProQuery(
                query_id=q_id,
                query_text=str(item["question"]),
                ground_truth_letter=ground_truth_letter,
                ground_truth_answer=ground_truth_answer,
                category=str(item["category"]),
                options=[str(o) for o in options],
            )
        )

    return out[:n]


def load_two_disjoint_subsets(
    n_eval: int = 500,
    n_train: int = 1000,
    eval_seed: int = 42,
    train_seed: int = 43,
) -> tuple[list[MMLUProQuery], list[MMLUProQuery]]:
    """Load evaluation and training corpora and assert they share no query IDs."""
    eval_set = load_mmlu_pro_subset(n_eval, eval_seed)
    train_set = load_mmlu_pro_subset(n_train, train_seed)

    eval_ids = {q.query_id for q in eval_set}
    train_ids = {q.query_id for q in train_set}
    overlap = eval_ids & train_ids
    if overlap:
        # Deterministically drop the overlap from the training set so the
        # two subsets are disjoint by query_id. Then re-fill from the tail
        # of the training shuffle with queries that aren't in eval.
        train_set = [q for q in train_set if q.query_id not in eval_ids]
        # If we lost some, pull more from the dataset with the same seed+offset.
        # The simplest fix: reload more and filter.
        if len(train_set) < n_train:
            extra = load_mmlu_pro_subset(n_train + len(overlap) * 2, train_seed)
            extra = [q for q in extra if q.query_id not in eval_ids]
            # Dedupe by query_id while preserving order.
            seen: set[str] = set()
            dedup: list[MMLUProQuery] = []
            for q in extra:
                if q.query_id in seen:
                    continue
                seen.add(q.query_id)
                dedup.append(q)
            train_set = dedup[:n_train]

    # Final assertion — the tests in task 4.2 rely on this contract.
    eval_ids = {q.query_id for q in eval_set}
    train_ids = {q.query_id for q in train_set}
    assert eval_ids.isdisjoint(train_ids), (
        "Evaluation and training corpora must not share query IDs after dedup"
    )
    return eval_set, train_set


def _derive_query_id(item: dict) -> str:
    """Derive a stable query_id. MMLU-Pro ships its own id-like field in some releases.

    Fall back to hashing the question text if no id is present.
    """
    for key in ("id", "question_id", "qid"):
        if key in item and item[key] is not None:
            return f"mmlupro:{item[key]}"
    # Fall back: hash the question text. This is stable across runs but
    # loses information if two different releases modify text slightly.
    import hashlib
    h = hashlib.sha1(str(item.get("question", "")).encode("utf-8")).hexdigest()[:16]
    return f"mmlupro:sha1-{h}"

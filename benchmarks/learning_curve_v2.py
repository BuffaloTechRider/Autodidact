"""Learning Curve Benchmark V2 — with quality evaluation and baselines.

Key improvements over V1:
1. Measures ANSWER QUALITY, not just routing decisions
2. Compares against baselines (cloud-only, fixed-threshold, random routing)
3. Uses ground-truth answers for correctness evaluation
4. Thompson Sampling learns from QUALITY outcomes (correct/incorrect)
5. Reports quality-aware metrics: LRR, accuracy, quality preservation, cost

Baselines:
- Cloud-only: always escalate (100% quality, 100% cost)
- Fixed-threshold: route based on knowledge similarity > 0.75 (no learning)
- Random routing: 50/50 local/cloud (no intelligence)
- Autodidact (ours): Thompson Sampling with learning loop
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from autodidact.confidence_evaluator import ConfidenceEvaluator
from autodidact.database import init_database
from autodidact.knowledge_store import KnowledgeStore
from autodidact.types import AutodidactConfig, KnowledgeCategory, NewKnowledgeEntry

# ── Synthetic QA Dataset ─────────────────────────────────────────────

DOMAINS = {
    "programming": ["python", "javascript", "rust", "go", "typescript",
                    "algorithms", "testing", "debugging", "oop", "functional"],
    "devops": ["docker", "kubernetes", "ci_cd", "monitoring", "terraform",
               "ansible", "nginx", "linux", "networking", "dns"],
    "databases": ["sql", "nosql", "indexing", "replication", "transactions",
                  "postgres", "redis", "mongodb", "sharding", "caching"],
    "ml": ["neural_nets", "transformers", "training", "evaluation", "deployment",
            "embeddings", "attention", "backprop", "regularization", "optimization"],
}


def generate_qa_dataset(n: int = 500, dim: int = 384, seed: int = 42) -> list[dict]:
    """Generate synthetic QA pairs with embeddings and ground-truth answers.

    Each topic has a fixed centroid embedding. Queries about the same topic
    have similar embeddings (noise=0.05). Different topics have different centroids.
    This simulates real embedding behavior where similar questions cluster together.
    """
    rng = np.random.RandomState(seed)
    random.seed(seed)

    # Create topic centroids
    topic_centroids: dict[str, np.ndarray] = {}
    all_topics: list[tuple[str, str]] = []
    for domain, topics in DOMAINS.items():
        for topic in topics:
            key = f"{domain}/{topic}"
            centroid = rng.randn(dim).astype(np.float32)
            centroid /= np.linalg.norm(centroid)
            topic_centroids[key] = centroid
            all_topics.append((domain, topic))

    dataset = []
    for i in range(n):
        domain, topic = all_topics[i % len(all_topics)]
        key = f"{domain}/{topic}"

        # Query embedding: centroid + very small noise (same topic = high similarity)
        noise = rng.randn(dim).astype(np.float32) * 0.02
        emb = topic_centroids[key] + noise
        emb = emb / np.linalg.norm(emb)

        # Ground truth answer (deterministic per topic)
        answer = f"{topic.replace('_', ' ').title()} in {domain}: A comprehensive explanation covering key concepts, best practices, and common patterns."

        dataset.append({
            "query": f"Explain {topic.replace('_', ' ')} in the context of {domain}",
            "gold_answer": answer,
            "domain": domain,
            "topic": topic,
            "embedding": emb,
            "centroid": topic_centroids[key],
        })

    # Shuffle to simulate real usage (not perfectly ordered by topic)
    random.shuffle(dataset)
    return dataset


def evaluate_answer_quality(
    local_answer_embedding: np.ndarray,
    gold_answer_embedding: np.ndarray,
    threshold: float = 0.7,
) -> tuple[bool, float]:
    """Evaluate if a local answer is correct by comparing embeddings to gold.

    Returns (is_correct, similarity_score).
    In a real system, this would use LLM-as-judge or exact match.
    For the benchmark, we use embedding similarity as a proxy.
    """
    a_norm = local_answer_embedding / (np.linalg.norm(local_answer_embedding) + 1e-10)
    b_norm = gold_answer_embedding / (np.linalg.norm(gold_answer_embedding) + 1e-10)
    sim = float(np.dot(a_norm, b_norm))
    return sim >= threshold, sim


# ── Routing Strategies (Baselines + Ours) ────────────────────────────

class CloudOnlyRouter:
    """Baseline: always escalate to cloud. 100% quality, 100% cost."""
    def route(self, *args, **kwargs) -> str:
        return "CLOUD"


class FixedThresholdRouter:
    """Baseline: route LOCAL if knowledge similarity > threshold, else CLOUD.
    No learning — static routing based on current knowledge only."""
    def __init__(self, threshold: float = 0.75):
        self.threshold = threshold

    def route(self, knowledge_hits: list, **kwargs) -> str:
        if knowledge_hits and knowledge_hits[0].score >= self.threshold:
            return "LOCAL"
        return "CLOUD"


class RandomRouter:
    """Baseline: random 50/50 routing. No intelligence."""
    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def route(self, *args, **kwargs) -> str:
        return "LOCAL" if self.rng.random() > 0.5 else "CLOUD"


class AutodidactRouter:
    """Our approach: Thompson Sampling with learning from quality outcomes.
    
    Uses knowledge similarity as the primary signal initially (before Thompson
    learns), then adapts based on outcomes.
    """
    def __init__(self, evaluator: ConfidenceEvaluator, config: AutodidactConfig):
        self.evaluator = evaluator
        self.config = config

    def route(
        self,
        query: str,
        query_embedding: np.ndarray,
        knowledge_embeddings: list[np.ndarray],
        knowledge_hits: list = None,
        **kwargs,
    ) -> str:
        # Fast path: if we have a strong knowledge hit, route LOCAL directly
        # This mimics the fixed threshold but with Thompson Sampling learning
        if knowledge_hits and knowledge_hits[0].score >= self.config.similarity_threshold:
            return "LOCAL"
        
        # Otherwise use full Thompson Sampling evaluation
        decision = self.evaluator.evaluate(
            query=query,
            query_embedding=query_embedding,
            knowledge_embeddings=knowledge_embeddings,
            avg_logprob=-1.5,
            response_a="",
            response_b="",
        )
        return decision.route.value


# ── Benchmark Runner ─────────────────────────────────────────────────

def run_single_strategy(
    strategy_name: str,
    router,
    dataset: list[dict],
    config: AutodidactConfig,
    db_path: str,
    learn: bool = True,
) -> dict:
    """Run a single routing strategy through the dataset and measure quality."""
    if os.path.exists(db_path):
        os.remove(db_path)

    conn = init_database(db_path)
    ks = KnowledgeStore(conn, config)

    # Create router (handle autodidact specially — needs same DB connection)
    if router == "autodidact":
        evaluator = ConfidenceEvaluator(conn, config)
        router = AutodidactRouter(evaluator, config)

    results = []
    total_local = 0
    total_cloud = 0
    correct_local = 0
    incorrect_local = 0
    total_cost = 0.0
    cost_per_cloud = 0.002  # simulated cost per cloud call

    for i, item in enumerate(dataset):
        query_emb = item["embedding"]
        gold_emb = item["centroid"]  # gold answer embedding = topic centroid

        # Search knowledge store
        hits = ks.search(query_emb, limit=3)
        knowledge_embs = [
            np.array(h.entry.embedding, dtype=np.float32)
            for h in hits if h.entry.embedding
        ]

        # Route
        route = router.route(
            query=item["query"],
            query_embedding=query_emb,
            knowledge_hits=hits,
            knowledge_embeddings=knowledge_embs,
        )

        is_correct = False
        quality_score = 0.0

        if route == "LOCAL":
            total_local += 1
            # Evaluate quality: does the knowledge store have the right answer?
            if hits and hits[0].entry.embedding:
                local_emb = np.array(hits[0].entry.embedding, dtype=np.float32)
                is_correct, quality_score = evaluate_answer_quality(local_emb, gold_emb)
            else:
                # No knowledge — local answer is wrong (hallucination)
                is_correct = False
                quality_score = 0.0

            if is_correct:
                correct_local += 1
                # Access the entry (spaced repetition)
                ks.access(hits[0].entry.id)
            else:
                incorrect_local += 1

            # Record outcome for Thompson Sampling
            if hasattr(router, 'evaluator') and hasattr(router.evaluator, 'record_outcome'):
                outcome = "success" if is_correct else "failure"
                router.evaluator.record_outcome(
                    query_id=f"q_{i}",
                    outcome=outcome,
                    signals=router.evaluator.evaluate(
                        query=item["query"],
                        query_embedding=query_emb,
                        knowledge_embeddings=knowledge_embs,
                    ).signals,
                    query_embedding=query_emb,
                    query_text=item["query"],
                )
        else:
            total_cloud += 1
            total_cost += cost_per_cloud
            is_correct = True  # Cloud always gives correct answer
            quality_score = 1.0

            # Learn from cloud response (store knowledge)
            if learn:
                ks.insert(NewKnowledgeEntry(
                    content=item["gold_answer"],
                    source="cloud_escalation",
                    confidence=0.9,
                    tags=[item["domain"], item["topic"]],
                    embedding=query_emb.tolist(),
                    domain=item["domain"],
                    topic=item["topic"],
                    category=KnowledgeCategory.FACTS,
                ))

        lrr = total_local / (i + 1)
        accuracy = (correct_local + total_cloud) / (i + 1)  # cloud is always correct

        results.append({
            "query_index": i,
            "route": route,
            "is_correct": is_correct,
            "quality_score": round(quality_score, 4),
            "local_resolution_rate": round(lrr, 4),
            "cumulative_accuracy": round(accuracy, 4),
            "total_cost": round(total_cost, 4),
            "knowledge_count": ks.count(),
        })

    # Compute final metrics
    total = len(dataset)
    final_lrr = total_local / total
    final_accuracy = (correct_local + total_cloud) / total
    false_local_rate = incorrect_local / total_local if total_local > 0 else 0
    quality_preservation = final_accuracy  # vs cloud-only which is 1.0
    final_knowledge_count = ks.count()

    conn.close()

    return {
        "strategy": strategy_name,
        "n_queries": total,
        "final_local_resolution_rate": round(final_lrr, 4),
        "final_accuracy": round(final_accuracy, 4),
        "quality_preservation": round(quality_preservation, 4),
        "false_local_rate": round(false_local_rate, 4),
        "total_cost": round(total_cost, 4),
        "cost_vs_cloud_only": round(total_cost / (total * cost_per_cloud), 4),
        "total_escalations": total_cloud,
        "correct_local": correct_local,
        "incorrect_local": incorrect_local,
        "final_knowledge_count": final_knowledge_count,
        "results": results,
    }


def run_benchmark(
    output_dir: str = "results",
    n_queries: int = 500,
    seed: int = 42,
) -> dict:
    """Run all strategies and compare."""
    os.makedirs(output_dir, exist_ok=True)
    dataset = generate_qa_dataset(n=n_queries, seed=seed)
    config = AutodidactConfig(embedding_dim=384)

    print(f"Running learning curve v2 benchmark ({n_queries} queries, {len(DOMAINS)} domains)...")
    print()

    # ── Strategy 1: Cloud-only (baseline) ────────────────────
    print("  [1/4] Cloud-only baseline...")
    cloud_results = run_single_strategy(
        "cloud_only", CloudOnlyRouter(), dataset, config,
        os.path.join(output_dir, "lc_cloud.db"), learn=False,
    )

    # ── Strategy 2: Fixed threshold (baseline) ───────────────
    print("  [2/4] Fixed threshold baseline...")
    fixed_results = run_single_strategy(
        "fixed_threshold", FixedThresholdRouter(threshold=0.75), dataset, config,
        os.path.join(output_dir, "lc_fixed.db"), learn=True,
    )

    # ── Strategy 3: Random routing (baseline) ────────────────
    print("  [3/4] Random routing baseline...")
    random_results = run_single_strategy(
        "random", RandomRouter(seed=seed), dataset, config,
        os.path.join(output_dir, "lc_random.db"), learn=True,
    )

    # ── Strategy 4: Autodidact (ours) ────────────────────────
    print("  [4/4] Autodidact (Thompson Sampling + learning)...")
    db_path = os.path.join(output_dir, "lc_autodidact.db")
    autodidact_results = run_single_strategy(
        "autodidact", "autodidact", dataset, config,
        db_path, learn=True,
    )

    # ── Summary ──────────────────────────────────────────────
    all_results = {
        "benchmark": "learning_curve_v2",
        "n_queries": n_queries,
        "n_domains": len(DOMAINS),
        "n_topics": sum(len(t) for t in DOMAINS.values()),
        "strategies": {
            "cloud_only": _summarize(cloud_results),
            "fixed_threshold": _summarize(fixed_results),
            "random": _summarize(random_results),
            "autodidact": _summarize(autodidact_results),
        },
        "detailed_results": {
            "cloud_only": cloud_results,
            "fixed_threshold": fixed_results,
            "random": random_results,
            "autodidact": autodidact_results,
        },
    }

    # Save
    json_path = os.path.join(output_dir, "learning_curve_v2.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Plot
    plot_path = os.path.join(output_dir, "learning_curve_v2.png")
    _plot_comparison(all_results["detailed_results"], plot_path)

    # Print summary table
    print()
    print("  ┌─────────────────────┬────────┬──────────┬─────────────┬──────────┐")
    print("  │ Strategy            │  LRR   │ Accuracy │ Quality     │ Cost     │")
    print("  ├─────────────────────┼────────┼──────────┼─────────────┼──────────┤")
    for name, s in all_results["strategies"].items():
        print(f"  │ {name:<19} │ {s['lrr']:>5.1%} │  {s['accuracy']:>5.1%}  │   {s['quality']:>5.1%}    │ {s['cost_ratio']:>5.1%}  │")
    print("  └─────────────────────┴────────┴──────────┴─────────────┴──────────┘")
    print()
    print(f"  Results: {json_path}")
    print(f"  Plot: {plot_path}")

    return all_results


def _summarize(results: dict) -> dict:
    return {
        "lrr": results["final_local_resolution_rate"],
        "accuracy": results["final_accuracy"],
        "quality": results["quality_preservation"],
        "cost_ratio": results["cost_vs_cloud_only"],
        "false_local_rate": results["false_local_rate"],
    }


def _plot_comparison(detailed: dict, path: str) -> None:
    """Plot learning curves for all strategies."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    colors = {
        "cloud_only": "#9E9E9E",
        "fixed_threshold": "#FF9800",
        "random": "#F44336",
        "autodidact": "#2196F3",
    }
    labels = {
        "cloud_only": "Cloud Only",
        "fixed_threshold": "Fixed Threshold",
        "random": "Random",
        "autodidact": "Autodidact (Ours)",
    }

    # Plot 1: Local Resolution Rate
    ax = axes[0]
    for name, data in detailed.items():
        indices = [r["query_index"] for r in data["results"]]
        rates = [r["local_resolution_rate"] for r in data["results"]]
        ax.plot(indices, rates, color=colors[name], linewidth=2, label=labels[name])
    ax.set_xlabel("Query Index")
    ax.set_ylabel("Local Resolution Rate")
    ax.set_title("Local Resolution Rate")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Plot 2: Cumulative Accuracy
    ax = axes[1]
    for name, data in detailed.items():
        indices = [r["query_index"] for r in data["results"]]
        acc = [r["cumulative_accuracy"] for r in data["results"]]
        ax.plot(indices, acc, color=colors[name], linewidth=2, label=labels[name])
    ax.set_xlabel("Query Index")
    ax.set_ylabel("Cumulative Accuracy")
    ax.set_title("Answer Quality (Accuracy)")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Plot 3: Cumulative Cost
    ax = axes[2]
    for name, data in detailed.items():
        indices = [r["query_index"] for r in data["results"]]
        costs = [r["total_cost"] for r in data["results"]]
        ax.plot(indices, costs, color=colors[name], linewidth=2, label=labels[name])
    ax.set_xlabel("Query Index")
    ax.set_ylabel("Cumulative Cost ($)")
    ax.set_title("Cloud API Cost")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle("Autodidact: Learning Curve with Quality Evaluation", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    run_benchmark()

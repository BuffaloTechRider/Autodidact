"""Learning Curve Benchmark — the most compelling visual for the paper.

Simulates an agent starting from empty knowledge, processing queries sequentially,
learning from cloud escalations, and tracking local resolution rate over time.
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

# ── Synthetic dataset ────────────────────────────────────────────────

DOMAINS = {
    "programming": {
        "topics": ["python", "javascript", "rust", "algorithms", "testing"],
        "category": KnowledgeCategory.FACTS,
    },
    "devops": {
        "topics": ["docker", "kubernetes", "ci_cd", "monitoring", "terraform"],
        "category": KnowledgeCategory.FACTS,
    },
    "databases": {
        "topics": ["sql", "nosql", "indexing", "replication", "transactions"],
        "category": KnowledgeCategory.FACTS,
    },
    "ml": {
        "topics": ["neural_nets", "transformers", "training", "evaluation", "deployment"],
        "category": KnowledgeCategory.DISCOVERIES,
    },
    "security": {
        "topics": ["auth", "encryption", "vulnerabilities", "compliance", "networking"],
        "category": KnowledgeCategory.ADVICE,
    },
}


def generate_synthetic_dataset(n: int = 200, dim: int = 384, seed: int = 42) -> list[dict]:
    """Generate synthetic Q&A pairs with embeddings across 5 domains."""
    rng = np.random.RandomState(seed)
    random.seed(seed)
    dataset = []

    # Create topic centroids for embedding clustering (tighter clusters)
    domain_names = list(DOMAINS.keys())
    topic_centroids: dict[str, np.ndarray] = {}
    for d in domain_names:
        for t in DOMAINS[d]["topics"]:
            key = f"{d}/{t}"
            c = rng.randn(dim).astype(np.float32)
            c /= np.linalg.norm(c)
            topic_centroids[key] = c

    for i in range(n):
        domain = domain_names[i % len(domain_names)]
        info = DOMAINS[domain]
        topic = info["topics"][i % len(info["topics"])]
        key = f"{domain}/{topic}"

        # Embedding near topic centroid with small noise (so repeats match)
        noise = rng.randn(dim).astype(np.float32) * 0.03
        emb = topic_centroids[key] + noise
        emb = emb / np.linalg.norm(emb)

        dataset.append({
            "query": f"How does {topic} work in {domain}?",
            "answer": f"{topic} in {domain}: This is the ground truth explanation for query {i}.",
            "domain": domain,
            "topic": topic,
            "category": info["category"],
            "embedding": emb,
        })

    return dataset


def run_learning_curve(
    output_dir: str = "results",
    n_queries: int = 200,
    seed: int = 42,
) -> dict:
    """Run the learning curve benchmark.

    Returns JSON-serializable results with per-query metrics.
    """
    os.makedirs(output_dir, exist_ok=True)
    db_path = os.path.join(output_dir, "learning_curve.db")

    # Clean slate
    if os.path.exists(db_path):
        os.remove(db_path)

    config = AutodidactConfig(db_path=db_path, embedding_dim=384)
    conn = init_database(db_path)
    ks = KnowledgeStore(conn, config)
    ce = ConfidenceEvaluator(conn, config)

    dataset = generate_synthetic_dataset(n=n_queries, dim=config.embedding_dim, seed=seed)

    results = []
    total_escalations = 0
    local_resolutions = 0

    for i, item in enumerate(dataset):
        query_emb = item["embedding"]

        # Search knowledge store
        hits = ks.search(query_emb, limit=3)
        knowledge_embs = [np.array(h.entry.embedding, dtype=np.float32) for h in hits if h.entry.embedding]

        # Evaluate confidence
        decision = ce.evaluate(
            query=item["query"],
            query_embedding=query_emb,
            knowledge_embeddings=knowledge_embs,
            avg_logprob=-1.5,
            response_a=item["answer"][:50],
            response_b=item["answer"][:50],
        )

        # Determine if local can resolve (has a match above threshold)
        local_can_resolve = len(hits) > 0 and hits[0].score >= config.similarity_threshold

        if local_can_resolve:
            local_resolutions += 1
            # Access the matched entry (spaced repetition)
            ks.access(hits[0].entry.id)
        else:
            total_escalations += 1
            # Learn from "cloud" response
            ks.insert(NewKnowledgeEntry(
                content=item["answer"],
                source="cloud_escalation",
                confidence=0.8,
                tags=[item["domain"], item["topic"]],
                embedding=query_emb.tolist(),
                domain=item["domain"],
                topic=item["topic"],
                category=item["category"],
            ))

        rate = local_resolutions / (i + 1)
        results.append({
            "query_index": i,
            "local_resolution_rate": round(rate, 4),
            "total_escalations": total_escalations,
            "knowledge_count": ks.count(),
            "route": "LOCAL" if local_can_resolve else "CLOUD",
        })

    conn.close()

    output = {
        "benchmark": "learning_curve",
        "n_queries": n_queries,
        "final_local_rate": results[-1]["local_resolution_rate"],
        "total_escalations": total_escalations,
        "final_knowledge_count": results[-1]["knowledge_count"],
        "results": results,
    }

    # Save JSON
    json_path = os.path.join(output_dir, "learning_curve.json")
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)

    # Generate plot
    plot_path = os.path.join(output_dir, "learning_curve.png")
    _plot_learning_curve(results, plot_path)

    print(f"Learning curve: {n_queries} queries, final local rate={output['final_local_rate']:.2%}")
    print(f"  Results: {json_path}")
    print(f"  Plot: {plot_path}")

    return output


def _plot_learning_curve(results: list[dict], path: str) -> None:
    """Generate matplotlib plot of the learning curve."""
    indices = [r["query_index"] for r in results]
    rates = [r["local_resolution_rate"] for r in results]
    counts = [r["knowledge_count"] for r in results]

    fig, ax1 = plt.subplots(figsize=(10, 6))

    color1 = "#2196F3"
    ax1.set_xlabel("Query Index", fontsize=12)
    ax1.set_ylabel("Local Resolution Rate", color=color1, fontsize=12)
    ax1.plot(indices, rates, color=color1, linewidth=2, label="Local Resolution Rate")
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.set_ylim(0, 1)

    ax2 = ax1.twinx()
    color2 = "#FF9800"
    ax2.set_ylabel("Knowledge Count", color=color2, fontsize=12)
    ax2.plot(indices, counts, color=color2, linewidth=1.5, linestyle="--", label="Knowledge Count")
    ax2.tick_params(axis="y", labelcolor=color2)

    fig.suptitle("Autodidact Learning Curve", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    run_learning_curve()

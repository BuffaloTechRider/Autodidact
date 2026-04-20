"""Ebbinghaus Decay Benchmark — Memory Consolidation.

Measures retrieval quality with and without Ebbinghaus decay by inserting
entries over simulated time, accessing some frequently, and comparing
precision of retrieval with stale entries pruned vs retained.
"""

from __future__ import annotations

import json
import os
import random
from datetime import datetime, timedelta, timezone

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from autodidact.database import init_database
from autodidact.knowledge_store import KnowledgeStore
from autodidact.types import AutodidactConfig, KnowledgeCategory, NewKnowledgeEntry


def _make_entry(
    content: str,
    domain: str,
    topic: str,
    embedding: np.ndarray,
) -> NewKnowledgeEntry:
    return NewKnowledgeEntry(
        content=content,
        source="cloud_escalation",
        confidence=0.8,
        tags=[domain, topic],
        embedding=embedding.tolist(),
        domain=domain,
        topic=topic,
        category=KnowledgeCategory.FACTS,
    )


def run_ebbinghaus_decay(
    output_dir: str = "results",
    n_entries: int = 500,
    seed: int = 42,
) -> dict:
    """Run the Ebbinghaus decay benchmark.

    Compares retrieval precision with decay (stale pruned) vs without (all retained).
    """
    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.RandomState(seed)
    random.seed(seed)
    dim = 384

    # Create two stores: one with decay, one without
    db_decay_path = os.path.join(output_dir, "ebbinghaus_decay.db")
    db_nodecay_path = os.path.join(output_dir, "ebbinghaus_nodecay.db")
    for p in [db_decay_path, db_nodecay_path]:
        if os.path.exists(p):
            os.remove(p)

    config_decay = AutodidactConfig(
        db_path=db_decay_path, decay_threshold=0.1, base_stability=1.0
    )
    config_nodecay = AutodidactConfig(
        db_path=db_nodecay_path, decay_threshold=0.0, base_stability=1.0
    )

    conn_decay = init_database(db_decay_path)
    conn_nodecay = init_database(db_nodecay_path)
    ks_decay = KnowledgeStore(conn_decay, config_decay)
    ks_nodecay = KnowledgeStore(conn_nodecay, config_nodecay)

    # Generate entries with domain clustering
    domains = ["programming", "devops", "databases", "ml", "security"]
    centroids = {d: rng.randn(dim).astype(np.float32) for d in domains}
    for c in centroids.values():
        c /= np.linalg.norm(c)

    # Track which entries are "relevant" (frequently accessed = still useful)
    entry_ids_decay: list[str] = []
    entry_ids_nodecay: list[str] = []
    frequently_accessed: set[int] = set()

    # Insert entries over simulated time — start from now
    base_time = datetime.now(timezone.utc)

    for i in range(n_entries):
        domain = domains[i % len(domains)]
        noise = rng.randn(dim).astype(np.float32) * 0.03
        emb = centroids[domain] + noise
        emb = emb / np.linalg.norm(emb)

        entry_data = _make_entry(
            content=f"Knowledge entry {i} about {domain}",
            domain=domain,
            topic=f"topic_{i % 10}",
            embedding=emb,
        )

        e1 = ks_decay.insert(entry_data)
        e2 = ks_nodecay.insert(entry_data)
        entry_ids_decay.append(e1.id)
        entry_ids_nodecay.append(e2.id)

        # 30% of entries get frequent access (these are "relevant")
        if rng.random() < 0.3:
            frequently_accessed.add(i)

    # Simulate access patterns: frequently accessed entries get 5-10 accesses
    for idx in frequently_accessed:
        n_accesses = rng.randint(5, 11)
        for _ in range(n_accesses):
            ks_decay.access(entry_ids_decay[idx])
            ks_nodecay.access(entry_ids_nodecay[idx])

    # Promote frequently accessed to LTM
    for idx in frequently_accessed:
        ks_decay.promote_to_ltm(entry_ids_decay[idx])
        ks_nodecay.promote_to_ltm(entry_ids_nodecay[idx])

    # Run decay cycles at intervals and measure precision
    results_decay: list[dict] = []
    results_nodecay: list[dict] = []

    n_cycles = 20
    hours_per_cycle = 2.0

    for cycle in range(n_cycles):
        current_time = base_time + timedelta(hours=hours_per_cycle * (cycle + 1))

        # Run decay on the decay store
        decay_result = ks_decay.run_decay_cycle(current_time)

        # Measure retrieval precision for both stores
        # Query with random domain centroid
        test_domain = random.choice(domains)
        query_emb = centroids[test_domain] + rng.randn(dim).astype(np.float32) * 0.02
        query_emb = query_emb / np.linalg.norm(query_emb)

        hits_decay = ks_decay.search(query_emb, limit=10)
        hits_nodecay = ks_nodecay.search(query_emb, limit=10)

        # Precision: fraction of returned entries that are from the correct domain
        def precision(hits, domain):
            if not hits:
                return 0.0
            correct = sum(1 for h in hits if h.entry.domain == domain)
            return correct / len(hits)

        p_decay = precision(hits_decay, test_domain)
        p_nodecay = precision(hits_nodecay, test_domain)

        elapsed_hours = hours_per_cycle * (cycle + 1)
        results_decay.append({
            "cycle": cycle,
            "elapsed_hours": elapsed_hours,
            "precision": round(p_decay, 4),
            "valid_entries": ks_decay.count(),
            "expired": decay_result["expired"],
            "promoted": decay_result["promoted"],
        })
        results_nodecay.append({
            "cycle": cycle,
            "elapsed_hours": elapsed_hours,
            "precision": round(p_nodecay, 4),
            "valid_entries": ks_nodecay.count(),
            "expired": 0,
            "promoted": 0,
        })

    conn_decay.close()
    conn_nodecay.close()

    output = {
        "benchmark": "ebbinghaus_decay",
        "n_entries": n_entries,
        "n_cycles": n_cycles,
        "frequently_accessed_count": len(frequently_accessed),
        "results_with_decay": results_decay,
        "results_without_decay": results_nodecay,
    }

    json_path = os.path.join(output_dir, "ebbinghaus_decay.json")
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)

    plot_path = os.path.join(output_dir, "ebbinghaus_decay.png")
    _plot_decay(results_decay, results_nodecay, plot_path)

    print(f"Ebbinghaus decay: {n_entries} entries, {n_cycles} decay cycles")
    print(f"  With decay — final precision: {results_decay[-1]['precision']:.2%}, entries: {results_decay[-1]['valid_entries']}")
    print(f"  Without decay — final precision: {results_nodecay[-1]['precision']:.2%}, entries: {results_nodecay[-1]['valid_entries']}")
    print(f"  Results: {json_path}")
    print(f"  Plot: {plot_path}")

    return output


def _plot_decay(decay: list[dict], nodecay: list[dict], path: str) -> None:
    """Plot precision over time for both conditions."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    hours_d = [r["elapsed_hours"] for r in decay]
    hours_n = [r["elapsed_hours"] for r in nodecay]

    # Precision
    ax1.plot(hours_d, [r["precision"] for r in decay],
             color="#4CAF50", linewidth=2, label="With Decay", marker="o", markersize=4)
    ax1.plot(hours_n, [r["precision"] for r in nodecay],
             color="#F44336", linewidth=2, label="Without Decay", marker="s", markersize=4)
    ax1.set_xlabel("Elapsed Hours", fontsize=12)
    ax1.set_ylabel("Retrieval Precision", fontsize=12)
    ax1.set_ylim(0, 1.05)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_title("Retrieval Precision Over Time")

    # Valid entries
    ax2.plot(hours_d, [r["valid_entries"] for r in decay],
             color="#4CAF50", linewidth=2, label="With Decay", marker="o", markersize=4)
    ax2.plot(hours_n, [r["valid_entries"] for r in nodecay],
             color="#F44336", linewidth=2, label="Without Decay", marker="s", markersize=4)
    ax2.set_xlabel("Elapsed Hours", fontsize=12)
    ax2.set_ylabel("Valid Entries", fontsize=12)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_title("Knowledge Store Size Over Time")

    fig.suptitle("Ebbinghaus Decay: Memory Consolidation", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    run_ebbinghaus_decay()

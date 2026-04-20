"""Tests for Knowledge Store."""

import math
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from autodidact.database import init_database
from autodidact.knowledge_store import KnowledgeStore
from autodidact.types import (
    AutodidactConfig,
    KnowledgeCategory,
    KnowledgeScope,
    MemoryTier,
    NewKnowledgeEntry,
)


@pytest.fixture
def setup():
    """Create in-memory database and knowledge store."""
    config = AutodidactConfig(db_path=":memory:", embedding_dim=32)
    conn = init_database(":memory:")
    ks = KnowledgeStore(conn, config)
    return ks, conn, config


def _random_embedding(dim: int = 32, seed: int = 0) -> list[float]:
    rng = np.random.RandomState(seed)
    emb = rng.randn(dim).astype(np.float32)
    emb /= np.linalg.norm(emb)
    return emb.tolist()


class TestInsert:
    """Test that insert creates STM entries."""

    def test_insert_creates_stm_entry(self, setup):
        ks, conn, config = setup
        entry = ks.insert(NewKnowledgeEntry(
            content="Python is a programming language",
            embedding=_random_embedding(),
            domain="programming",
            topic="python",
        ))
        assert entry.tier == MemoryTier.STM
        assert entry.usage_count == 0
        assert entry.valid_to is None

    def test_insert_sets_valid_from(self, setup):
        ks, conn, config = setup
        entry = ks.insert(NewKnowledgeEntry(
            content="Test content",
            embedding=_random_embedding(),
        ))
        assert entry.valid_from is not None
        assert entry.valid_to is None

    def test_insert_preserves_domain_topic(self, setup):
        ks, conn, config = setup
        entry = ks.insert(NewKnowledgeEntry(
            content="Docker containers",
            embedding=_random_embedding(),
            domain="devops",
            topic="docker",
            category=KnowledgeCategory.FACTS,
        ))
        assert entry.domain == "devops"
        assert entry.topic == "docker"
        assert entry.category == KnowledgeCategory.FACTS


class TestSearch:
    """Test search with minimum threshold."""

    def test_search_returns_above_threshold(self, setup):
        ks, conn, config = setup
        emb = _random_embedding(seed=1)
        ks.insert(NewKnowledgeEntry(
            content="Python basics",
            embedding=emb,
            domain="programming",
        ))
        query = np.array(emb, dtype=np.float32)
        hits = ks.search(query, limit=5)
        assert len(hits) == 1
        assert hits[0].score >= config.similarity_threshold

    def test_search_filters_below_threshold(self, setup):
        ks, conn, config = setup
        emb1 = np.array([1.0, 0.0, 0.0] + [0.0] * 29, dtype=np.float32)
        emb1 /= np.linalg.norm(emb1)
        ks.insert(NewKnowledgeEntry(
            content="Entry A",
            embedding=emb1.tolist(),
        ))
        # Query with orthogonal vector — should get no results
        query = np.array([0.0, 1.0, 0.0] + [0.0] * 29, dtype=np.float32)
        query /= np.linalg.norm(query)
        hits = ks.search(query, limit=5)
        assert len(hits) == 0

    def test_search_excludes_invalidated(self, setup):
        ks, conn, config = setup
        emb = _random_embedding(seed=2)
        entry = ks.insert(NewKnowledgeEntry(
            content="Outdated info",
            embedding=emb,
        ))
        ks.invalidate(entry.id)
        query = np.array(emb, dtype=np.float32)
        hits = ks.search(query, limit=5)
        assert len(hits) == 0


class TestEbbinghausDecay:
    """Test Ebbinghaus decay formula."""

    def test_retention_formula(self):
        # R(t) = e^(-t/S), S = base_stability × (1 + ln(1 + access_count))
        # With access_count=0, S=1*(1+ln(1))=1, R(1)=e^(-1)≈0.368
        r = KnowledgeStore.ebbinghaus_retention(1.0, 0, 1.0)
        assert abs(r - math.exp(-1)) < 1e-6

    def test_more_accesses_slower_decay(self):
        r_low = KnowledgeStore.ebbinghaus_retention(5.0, 1, 1.0)
        r_high = KnowledgeStore.ebbinghaus_retention(5.0, 10, 1.0)
        assert r_high > r_low, "More accesses should mean slower decay"

    def test_zero_time_full_retention(self):
        r = KnowledgeStore.ebbinghaus_retention(0.0, 5, 1.0)
        assert abs(r - 1.0) < 1e-6

    def test_decay_cycle_expires_stale_ltm(self, setup):
        ks, conn, config = setup
        emb = _random_embedding(seed=3)
        entry = ks.insert(NewKnowledgeEntry(
            content="Will decay",
            embedding=emb,
        ))
        # Promote to LTM
        ks.promote_to_ltm(entry.id)
        # Run decay far in the future — should expire
        future = datetime.now(timezone.utc) + timedelta(hours=100)
        result = ks.run_decay_cycle(future)
        assert result["expired"] >= 1

        # Verify entry is invalidated
        refreshed = ks.get(entry.id)
        assert refreshed is not None
        assert refreshed.valid_to is not None


class TestSTMPromotion:
    """Test STM → LTM promotion."""

    def test_promotion_on_access(self, setup):
        ks, conn, config = setup
        emb = _random_embedding(seed=4)
        entry = ks.insert(NewKnowledgeEntry(
            content="Frequently accessed",
            embedding=emb,
        ))
        assert entry.tier == MemoryTier.STM

        # Access enough times to trigger promotion
        for _ in range(config.stm_promotion_accesses):
            ks.access(entry.id)

        # Run decay cycle to trigger promotion check
        ks.run_decay_cycle()
        refreshed = ks.get(entry.id)
        assert refreshed is not None
        assert refreshed.tier == MemoryTier.LTM

    def test_manual_promotion(self, setup):
        ks, conn, config = setup
        entry = ks.insert(NewKnowledgeEntry(
            content="Manual promote",
            embedding=_random_embedding(seed=5),
        ))
        ks.promote_to_ltm(entry.id)
        refreshed = ks.get(entry.id)
        assert refreshed is not None
        assert refreshed.tier == MemoryTier.LTM
        assert refreshed.promoted_at is not None


class TestScopedSearch:
    """Test scoped search by domain/topic/category."""

    def test_scoped_by_domain(self, setup):
        ks, conn, config = setup
        emb = _random_embedding(seed=10)
        ks.insert(NewKnowledgeEntry(
            content="Python stuff",
            embedding=emb,
            domain="programming",
            topic="python",
        ))
        ks.insert(NewKnowledgeEntry(
            content="Docker stuff",
            embedding=emb,  # same embedding for test
            domain="devops",
            topic="docker",
        ))

        query = np.array(emb, dtype=np.float32)
        scope = KnowledgeScope(domain="programming")
        hits = ks.search(query, limit=10, scope=scope)
        for h in hits:
            assert h.entry.domain == "programming"

    def test_scoped_by_topic(self, setup):
        ks, conn, config = setup
        emb = _random_embedding(seed=11)
        ks.insert(NewKnowledgeEntry(
            content="Python basics",
            embedding=emb,
            domain="programming",
            topic="python",
        ))
        ks.insert(NewKnowledgeEntry(
            content="Rust basics",
            embedding=emb,
            domain="programming",
            topic="rust",
        ))

        query = np.array(emb, dtype=np.float32)
        scope = KnowledgeScope(domain="programming", topic="python")
        hits = ks.search(query, limit=10, scope=scope)
        for h in hits:
            assert h.entry.topic == "python"

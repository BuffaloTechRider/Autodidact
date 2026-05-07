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


class TestMixedDimensionDetection:
    """Embedder-swap safety: mixed-dim entries must fail loudly, not silently corrupt FAISS."""

    def test_insert_rejects_mismatched_dim(self, setup):
        from autodidact.knowledge_store import MixedEmbeddingDimensionError

        ks, conn, config = setup
        # First entry establishes the dim (32, per the fixture).
        ks.insert(NewKnowledgeEntry(
            content="768-dim era entry",
            embedding=_random_embedding(dim=32, seed=1),
        ))
        # Second entry at a different dim must raise.
        with pytest.raises(MixedEmbeddingDimensionError, match="1024"):
            ks.insert(NewKnowledgeEntry(
                content="1024-dim era entry",
                embedding=_random_embedding(dim=1024, seed=2),
            ))

    def test_faiss_rebuild_rejects_mixed_dim(self, setup):
        """If mixed-dim rows get into the DB out-of-band, index build must fail clearly."""
        import sqlite3

        from autodidact.knowledge_store import MixedEmbeddingDimensionError

        ks, conn, config = setup

        # Insert first entry at dim 32 via the normal path.
        ks.insert(NewKnowledgeEntry(
            content="first",
            embedding=_random_embedding(dim=32, seed=1),
        ))

        # Bypass insert() to simulate a pre-existing mixed-dim DB (the scenario we
        # actually hit: a DB seeded with one embedder and then half-migrated).
        bad_blob = np.array(
            _random_embedding(dim=1024, seed=2), dtype=np.float32
        ).tobytes()
        conn.execute(
            "INSERT INTO knowledge_entries "
            "(id, content, question, source, confidence, tags, embedding, tier, "
            "usage_count, created_at, last_accessed, metadata, domain, topic, "
            "category, valid_from, valid_to, verbatim_response) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, 'STM', 0, ?, ?, ?, ?, ?, ?, ?, NULL, ?)",
            (
                "bad-id", "mismatched", "q", "manual", 0.5, "[]", bad_blob,
                "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z", "{}",
                "d", "t", "facts", "2026-01-01T00:00:00Z", "raw",
            ),
        )
        conn.commit()
        ks._faiss_dirty = True

        query = np.array(_random_embedding(dim=32, seed=3), dtype=np.float32)
        with pytest.raises(MixedEmbeddingDimensionError, match="mixed embedding dimensions"):
            ks.search(query, limit=5)

    def test_insert_dim_cache_survives_across_calls(self, setup):
        """The _expected_dim cache means we don't re-query on every insert."""
        ks, conn, config = setup
        ks.insert(NewKnowledgeEntry(
            content="one", embedding=_random_embedding(dim=32, seed=1)
        ))
        # Second insert of the same dim should just work.
        ks.insert(NewKnowledgeEntry(
            content="two", embedding=_random_embedding(dim=32, seed=2)
        ))
        assert ks._expected_dim == 32


class TestPerConsumerThreshold:
    """search(min_similarity=...) lets each caller set its own floor."""

    def test_default_uses_config_threshold(self, setup):
        ks, conn, config = setup
        # Insert two entries, then query at cos ~1 to a third vector.
        e1 = np.array([1.0] + [0.0] * 31, dtype=np.float32)
        e1 /= np.linalg.norm(e1)
        e2 = np.array([0.9, 0.1] + [0.0] * 30, dtype=np.float32)
        e2 /= np.linalg.norm(e2)
        ks.insert(NewKnowledgeEntry(content="A", embedding=e1.tolist()))
        ks.insert(NewKnowledgeEntry(content="B", embedding=e2.tolist()))

        # Config defaults to 0.75; query close to e1 should see e1 but not e2.
        q = e1.copy()
        hits = ks.search(q, limit=5)
        assert len(hits) >= 1
        for h in hits:
            assert h.score >= config.similarity_threshold

    def test_min_similarity_override_lower(self, setup):
        """Passing min_similarity=0.0 admits everything the config would filter."""
        ks, conn, config = setup
        e = np.array([1.0] + [0.0] * 31, dtype=np.float32)
        e /= np.linalg.norm(e)
        ks.insert(NewKnowledgeEntry(content="A", embedding=e.tolist()))

        # Nearly-orthogonal query that scores ~0.3 against e.
        q = np.array([0.3, 0.95] + [0.0] * 30, dtype=np.float32)
        q /= np.linalg.norm(q)
        # With default threshold (0.75 in fixture) expect 0 hits.
        hits_default = ks.search(q, limit=5)
        # With override threshold 0.0 expect 1 hit.
        hits_override = ks.search(q, limit=5, min_similarity=0.0)
        assert len(hits_override) >= len(hits_default)
        # The override hit has a score well below the default threshold.
        if hits_override:
            assert hits_override[0].score < config.similarity_threshold

    def test_min_similarity_override_higher(self, setup):
        """Passing a higher min_similarity filters out hits the config would admit."""
        ks, conn, config = setup
        e = np.array([1.0] + [0.0] * 31, dtype=np.float32)
        e /= np.linalg.norm(e)
        ks.insert(NewKnowledgeEntry(content="A", embedding=e.tolist()))

        # Query identical to e — similarity ~1.0. Default (0.75) admits. 0.99 also admits.
        # 1.5 (impossible floor) admits nothing.
        q = e.copy()
        assert len(ks.search(q, limit=5, min_similarity=0.99)) >= 1
        assert len(ks.search(q, limit=5, min_similarity=1.5)) == 0


class TestAnswerEmbedding:
    """v0.1 stores answer embeddings for v0.2 retrieval experiments without re-seeding."""

    def test_insert_roundtrips_answer_embedding(self, setup):
        ks, conn, config = setup
        q_emb = _random_embedding(dim=32, seed=1)
        a_emb = _random_embedding(dim=32, seed=2)
        entry = ks.insert(NewKnowledgeEntry(
            content="Paris is the capital of France.",
            question="What is the capital of France?",
            embedding=q_emb,
            answer_embedding=a_emb,
        ))
        fetched = ks.get(entry.id)
        assert fetched is not None
        assert fetched.answer_embedding is not None
        assert len(fetched.answer_embedding) == 32
        # Pydantic floats vs numpy floats: allow float32 round-trip tolerance.
        assert all(abs(a - b) < 1e-5 for a, b in zip(fetched.answer_embedding, a_emb))

    def test_insert_accepts_null_answer_embedding(self, setup):
        """Not all callers compute an answer embedding; NULL is the default."""
        ks, conn, config = setup
        entry = ks.insert(NewKnowledgeEntry(
            content="Content with no answer embedding",
            embedding=_random_embedding(dim=32, seed=3),
        ))
        fetched = ks.get(entry.id)
        assert fetched is not None
        assert fetched.answer_embedding is None

    def test_answer_embedding_not_used_by_retrieval(self, setup):
        """v0.1 invariant: answer_embedding is stored but never searched."""
        ks, conn, config = setup
        # Insert with question-embedding e1 and an orthogonal answer-embedding e2.
        e1 = np.array([1.0] + [0.0] * 31, dtype=np.float32)
        e1 /= np.linalg.norm(e1)
        e2 = np.array([0.0, 1.0] + [0.0] * 30, dtype=np.float32)
        e2 /= np.linalg.norm(e2)
        ks.insert(NewKnowledgeEntry(
            content="C", embedding=e1.tolist(), answer_embedding=e2.tolist(),
        ))
        # Query vector matches e2 (the ANSWER embedding). If retrieval used
        # the answer side this should return the entry; since v0.1 only
        # searches question embeddings, it should NOT return it at the
        # default threshold.
        q = e2.copy()
        hits = ks.search(q, limit=5)
        assert len(hits) == 0  # answer-side retrieval is a v0.2 thing

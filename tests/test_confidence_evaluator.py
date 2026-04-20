"""Tests for Thompson Sampling confidence evaluator."""

import numpy as np
import pytest

from autodidact.confidence_evaluator import ConfidenceEvaluator
from autodidact.database import init_database
from autodidact.types import AutodidactConfig, SignalScores


@pytest.fixture
def setup():
    """Create in-memory database and evaluator."""
    config = AutodidactConfig(db_path=":memory:", embedding_dim=32)
    conn = init_database(":memory:")
    ce = ConfidenceEvaluator(conn, config)
    return ce, conn, config


class TestBetaSampling:
    """Test that Beta sampling produces valid values."""

    def test_sampled_theta_in_unit_interval(self, setup):
        ce, conn, config = setup
        rng = np.random.RandomState(42)
        emb = rng.randn(32).astype(np.float32)

        # Run evaluate many times — all fused scores should be in [0, 1]
        for _ in range(50):
            decision = ce.evaluate(
                query="test query",
                query_embedding=emb,
                knowledge_embeddings=[],
                avg_logprob=-1.5,
                response_a="hello world",
                response_b="hello world",
            )
            assert 0.0 <= decision.fused_score <= 1.0

    def test_fusion_weights_are_positive(self, setup):
        ce, conn, config = setup
        emb = np.random.randn(32).astype(np.float32)
        decision = ce.evaluate(
            query="test",
            query_embedding=emb,
            knowledge_embeddings=[],
        )
        for name, weight in decision.fusion_weights.items():
            assert weight > 0.0, f"Weight for {name} should be positive"


class TestFusion:
    """Test that fusion is a valid weighted average."""

    def test_fused_score_is_weighted_average(self, setup):
        ce, conn, config = setup
        emb = np.random.randn(32).astype(np.float32)
        decision = ce.evaluate(
            query="what is python",
            query_embedding=emb,
            knowledge_embeddings=[],
            avg_logprob=-1.0,
            response_a="Python is a language",
            response_b="Python is a language",
        )
        # Manually verify: fused = Σ(θᵢ × sᵢ) / Σ(θᵢ)
        signals = decision.signals.model_dump()
        weighted_sum = 0.0
        weight_total = 0.0
        for name, theta in decision.fusion_weights.items():
            val = signals[name]
            if val is not None:
                weighted_sum += theta * val
                weight_total += theta
        expected = weighted_sum / weight_total if weight_total > 0 else 0.0
        assert abs(decision.fused_score - expected) < 1e-6


class TestOutcomeUpdates:
    """Test that outcome recording updates α/β correctly."""

    def test_success_increments_alpha(self, setup):
        ce, conn, config = setup
        before = ce.get_signal_weights()
        signals = SignalScores(
            knowledge_similarity=0.8,
            logprob_uncertainty=0.6,
            self_consistency=0.7,
            query_classification=0.5,
            energy_scorer=None,
        )
        ce.record_outcome("q1", "success", signals)
        after = ce.get_signal_weights()

        for name in ["knowledge_similarity", "logprob_uncertainty",
                      "self_consistency", "query_classification"]:
            assert after[name]["alpha"] == before[name]["alpha"] + 1
            assert after[name]["beta_param"] == before[name]["beta_param"]

    def test_failure_increments_beta(self, setup):
        ce, conn, config = setup
        before = ce.get_signal_weights()
        signals = SignalScores(
            knowledge_similarity=0.8,
            logprob_uncertainty=0.6,
            self_consistency=0.7,
            query_classification=0.5,
            energy_scorer=None,
        )
        ce.record_outcome("q1", "failure", signals)
        after = ce.get_signal_weights()

        for name in ["knowledge_similarity", "logprob_uncertainty",
                      "self_consistency", "query_classification"]:
            assert after[name]["alpha"] == before[name]["alpha"]
            assert after[name]["beta_param"] == before[name]["beta_param"] + 1

    def test_energy_scorer_not_updated_when_none(self, setup):
        ce, conn, config = setup
        before = ce.get_signal_weights()
        signals = SignalScores(
            knowledge_similarity=0.8,
            logprob_uncertainty=0.6,
            self_consistency=0.7,
            query_classification=0.5,
            energy_scorer=None,
        )
        ce.record_outcome("q1", "success", signals)
        after = ce.get_signal_weights()
        assert after["energy_scorer"]["alpha"] == before["energy_scorer"]["alpha"]


class TestKnowledgeSimilarityThreshold:
    """Test that knowledge_similarity returns 0 below threshold."""

    def test_below_threshold_returns_zero(self, setup):
        ce, conn, config = setup
        query = np.array([1.0, 0.0, 0.0] + [0.0] * 29, dtype=np.float32)
        # Orthogonal embedding — similarity near 0
        kb = np.array([0.0, 1.0, 0.0] + [0.0] * 29, dtype=np.float32)
        score = ce.compute_knowledge_similarity(query, [kb])
        assert score == 0.0

    def test_above_threshold_returns_similarity(self, setup):
        ce, conn, config = setup
        query = np.ones(32, dtype=np.float32)
        query /= np.linalg.norm(query)
        # Nearly identical embedding
        kb = query + np.random.randn(32).astype(np.float32) * 0.01
        kb /= np.linalg.norm(kb)
        score = ce.compute_knowledge_similarity(query, [kb])
        assert score > 0.75

    def test_empty_knowledge_returns_zero(self, setup):
        ce, conn, config = setup
        query = np.random.randn(32).astype(np.float32)
        score = ce.compute_knowledge_similarity(query, [])
        assert score == 0.0


class TestEnergyScorer:
    """Test energy scorer activation at 50 examples."""

    def test_disabled_below_threshold(self, setup):
        ce, conn, config = setup
        emb = np.random.randn(32).astype(np.float32)
        result = ce.compute_energy_score(emb)
        assert result is None

    def test_activates_at_threshold(self, setup):
        ce, conn, config = setup
        rng = np.random.RandomState(42)

        # Add 50 examples (the minimum)
        signals = SignalScores(
            knowledge_similarity=0.8,
            logprob_uncertainty=0.6,
            self_consistency=0.7,
            query_classification=0.5,
            energy_scorer=None,
        )
        for i in range(50):
            emb = rng.randn(32).astype(np.float32)
            outcome = "success" if i % 2 == 0 else "failure"
            ce.record_outcome(
                f"q{i}", outcome, signals,
                query_embedding=emb,
                query_text=f"query {i}",
            )

        # Force retrain
        ce._train_energy_model()

        # Now energy scorer should be active
        test_emb = rng.randn(32).astype(np.float32)
        result = ce.compute_energy_score(test_emb)
        assert result is not None
        assert 0.0 <= result <= 1.0

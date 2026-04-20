"""Thompson Sampling confidence evaluator — the core novel contribution.

Fuses five independent signals via Bayesian bandit to produce routing decisions.
Each signal maintains Beta(α, β) parameters that self-calibrate from outcomes.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Optional

import numpy as np
from scipy.stats import beta as beta_dist
from sklearn.linear_model import LogisticRegression

from autodidact.types import (
    AutodidactConfig,
    RouteName,
    RoutingDecision,
    SignalScores,
)

SIGNAL_NAMES = [
    "knowledge_similarity",
    "logprob_uncertainty",
    "self_consistency",
    "query_classification",
    "energy_scorer",
]


class ConfidenceEvaluator:
    """Multi-signal confidence evaluator with Thompson Sampling fusion.

    Computes five signals, samples θ ~ Beta(α, β) per signal, and produces
    a weighted-average fused score for routing decisions.
    """

    def __init__(self, conn: sqlite3.Connection, config: AutodidactConfig) -> None:
        self.conn = conn
        self.config = config
        self._ensure_thompson_params()
        self._energy_model: Optional[LogisticRegression] = None
        self._energy_example_count = self._get_energy_example_count()
        self._load_energy_model()

    # ── Signal computation ───────────────────────────────────────────

    def compute_knowledge_similarity(
        self, query_embedding: np.ndarray, knowledge_embeddings: list[np.ndarray]
    ) -> float:
        """Max cosine similarity against knowledge store. Returns 0 if below threshold."""
        if not knowledge_embeddings:
            return 0.0
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-10)
        max_sim = 0.0
        for emb in knowledge_embeddings:
            emb_norm = emb / (np.linalg.norm(emb) + 1e-10)
            sim = float(np.dot(query_norm, emb_norm))
            max_sim = max(max_sim, sim)
        if max_sim < self.config.similarity_threshold:
            return 0.0
        return max_sim

    def compute_logprob_uncertainty(
        self, avg_logprob: float, scale: float = 2.0, shift: float = 3.0
    ) -> float:
        """Map average log-probability to confidence via sigmoid."""
        x = avg_logprob * scale + shift
        return float(1.0 / (1.0 + np.exp(-x)))

    def compute_self_consistency(self, response_a: str, response_b: str) -> float:
        """Measure factual agreement between two responses.

        Simple token-overlap heuristic suitable for benchmarks.
        Production would use LLM-as-judge.
        """
        tokens_a = set(response_a.lower().split())
        tokens_b = set(response_b.lower().split())
        if not tokens_a or not tokens_b:
            return 0.0
        intersection = tokens_a & tokens_b
        union = tokens_a | tokens_b
        return len(intersection) / len(union)

    def compute_query_classification(self, query: str) -> float:
        """Classify query type and return confidence bias.

        Factual → 0.7 (bias toward local), reasoning → 0.5,
        real-time → 0.2 (bias toward cloud), creative → 0.4.
        """
        query_lower = query.lower()
        realtime_keywords = [
            "current", "latest", "today", "now", "weather",
            "price", "stock", "news", "live",
        ]
        factual_keywords = [
            "what is", "define", "explain", "how does", "describe",
            "meaning of", "difference between",
        ]
        creative_keywords = [
            "write", "create", "generate", "compose", "design",
            "imagine", "story", "poem",
        ]

        if any(kw in query_lower for kw in realtime_keywords):
            return 0.2
        if any(kw in query_lower for kw in factual_keywords):
            return 0.7
        if any(kw in query_lower for kw in creative_keywords):
            return 0.4
        return 0.5

    def compute_energy_score(self, query_embedding: np.ndarray) -> Optional[float]:
        """Logistic regression on query embeddings trained on pass/fail history.

        Returns None if fewer than energy_scorer_min_examples exist.
        """
        if self._energy_model is None:
            return None
        prob = self._energy_model.predict_proba(query_embedding.reshape(1, -1))[0]
        # Index 1 = probability of 'pass'
        pass_idx = list(self._energy_model.classes_).index("pass")
        return float(prob[pass_idx])

    # ── Thompson Sampling fusion ─────────────────────────────────────

    def evaluate(
        self,
        query: str,
        query_embedding: np.ndarray,
        knowledge_embeddings: list[np.ndarray],
        avg_logprob: float = -1.5,
        response_a: str = "",
        response_b: str = "",
    ) -> RoutingDecision:
        """Compute all signals, sample θ ~ Beta(α,β), fuse, and route."""
        signals = SignalScores(
            knowledge_similarity=self.compute_knowledge_similarity(
                query_embedding, knowledge_embeddings
            ),
            logprob_uncertainty=self.compute_logprob_uncertainty(avg_logprob),
            self_consistency=self.compute_self_consistency(response_a, response_b),
            query_classification=self.compute_query_classification(query),
            energy_scorer=self.compute_energy_score(query_embedding),
        )

        # Sample θ from Beta(α, β) for each active signal
        params = self._get_thompson_params()
        fusion_weights: dict[str, float] = {}
        weighted_sum = 0.0
        weight_total = 0.0

        signal_dict = signals.model_dump()
        for name in SIGNAL_NAMES:
            value = signal_dict[name]
            if value is None:
                continue
            alpha = params[name]["alpha"]
            beta_p = params[name]["beta_param"]
            theta = float(beta_dist.rvs(alpha, beta_p))
            fusion_weights[name] = theta
            weighted_sum += theta * value
            weight_total += theta

        fused_score = weighted_sum / weight_total if weight_total > 0 else 0.0
        route = RouteName.LOCAL if fused_score >= self.config.confidence_threshold else RouteName.CLOUD
        query_id = str(uuid.uuid4())

        return RoutingDecision(
            route=route,
            signals=signals,
            fusion_weights=fusion_weights,
            fused_score=fused_score,
            query_id=query_id,
        )

    # ── Outcome recording ────────────────────────────────────────────

    def record_outcome(
        self,
        query_id: str,
        outcome: str,
        signals: SignalScores,
        query_embedding: Optional[np.ndarray] = None,
        query_text: str = "",
    ) -> None:
        """Update Thompson Sampling α/β based on success/failure."""
        now = datetime.now(timezone.utc).isoformat()
        signal_dict = signals.model_dump()

        for name in SIGNAL_NAMES:
            if signal_dict[name] is None:
                continue
            if outcome == "success":
                self.conn.execute(
                    "UPDATE thompson_params SET alpha = alpha + 1, updated_at = ? WHERE signal_name = ?",
                    (now, name),
                )
            else:
                self.conn.execute(
                    "UPDATE thompson_params SET beta_param = beta_param + 1, updated_at = ? WHERE signal_name = ?",
                    (now, name),
                )
        self.conn.commit()

        # Record energy scorer example
        if query_embedding is not None and query_text:
            self._add_energy_example(query_text, query_embedding, outcome)

    def get_signal_weights(self) -> dict[str, dict[str, float]]:
        """Return current Beta(α, β) parameters for all signals."""
        return self._get_thompson_params()

    # ── Energy scorer management ─────────────────────────────────────

    def _add_energy_example(
        self, query_text: str, query_embedding: np.ndarray, outcome: str
    ) -> None:
        """Add a labeled example and retrain if needed."""
        now = datetime.now(timezone.utc).isoformat()
        es_outcome = "pass" if outcome == "success" else "fail"
        self.conn.execute(
            "INSERT INTO energy_scorer_examples (id, query_text, query_embedding, outcome, created_at) VALUES (?, ?, ?, ?, ?)",
            (str(uuid.uuid4()), query_text, query_embedding.tobytes(), es_outcome, now),
        )
        self.conn.commit()
        self._energy_example_count += 1

        if (
            self._energy_example_count >= self.config.energy_scorer_min_examples
            and self._energy_example_count % self.config.energy_scorer_retrain_interval == 0
        ):
            self._train_energy_model()

    def _train_energy_model(self) -> None:
        """Train logistic regression on accumulated pass/fail embeddings."""
        rows = self.conn.execute(
            "SELECT query_embedding, outcome FROM energy_scorer_examples"
        ).fetchall()
        if len(rows) < self.config.energy_scorer_min_examples:
            return

        X = np.array([np.frombuffer(r["query_embedding"], dtype=np.float32) for r in rows])
        y = [r["outcome"] for r in rows]

        # Need both classes
        unique_classes = set(y)
        if len(unique_classes) < 2:
            return

        model = LogisticRegression(max_iter=500, random_state=42)
        model.fit(X, y)
        self._energy_model = model

        # Persist model
        now = datetime.now(timezone.utc).isoformat()
        weights_blob = model.coef_.astype(np.float32).tobytes()
        bias = float(model.intercept_[0])
        self.conn.execute("DELETE FROM energy_scorer_model")
        self.conn.execute(
            "INSERT INTO energy_scorer_model (id, weights, bias, example_count, trained_at) VALUES (1, ?, ?, ?, ?)",
            (weights_blob, bias, len(rows), now),
        )
        self.conn.commit()

    def _load_energy_model(self) -> None:
        """Load persisted energy scorer model if available."""
        row = self.conn.execute("SELECT * FROM energy_scorer_model WHERE id = 1").fetchone()
        if row is None:
            return
        if row["example_count"] < self.config.energy_scorer_min_examples:
            return

        # Reload from stored examples to get a proper sklearn model
        rows = self.conn.execute(
            "SELECT query_embedding, outcome FROM energy_scorer_examples"
        ).fetchall()
        if len(rows) < self.config.energy_scorer_min_examples:
            return

        X = np.array([np.frombuffer(r["query_embedding"], dtype=np.float32) for r in rows])
        y = [r["outcome"] for r in rows]
        if len(set(y)) < 2:
            return

        model = LogisticRegression(max_iter=500, random_state=42)
        model.fit(X, y)
        self._energy_model = model

    def _get_energy_example_count(self) -> int:
        row = self.conn.execute("SELECT COUNT(*) as cnt FROM energy_scorer_examples").fetchone()
        return row["cnt"] if row else 0

    # ── Thompson params persistence ──────────────────────────────────

    def _ensure_thompson_params(self) -> None:
        """Initialize Beta(1,1) for each signal if not present."""
        now = datetime.now(timezone.utc).isoformat()
        for name in SIGNAL_NAMES:
            self.conn.execute(
                "INSERT OR IGNORE INTO thompson_params (signal_name, alpha, beta_param, updated_at) VALUES (?, 1.0, 1.0, ?)",
                (name, now),
            )
        self.conn.commit()

    def _get_thompson_params(self) -> dict[str, dict[str, float]]:
        rows = self.conn.execute("SELECT signal_name, alpha, beta_param FROM thompson_params").fetchall()
        return {r["signal_name"]: {"alpha": r["alpha"], "beta_param": r["beta_param"]} for r in rows}

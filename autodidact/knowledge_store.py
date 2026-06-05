"""Hierarchical Knowledge Store with Ebbinghaus Decay.

SQLite-backed storage with domain/topic/category hierarchy,
cosine similarity search, STM/LTM tiers, and spaced-repetition decay.
"""

from __future__ import annotations

import functools
import json
import math
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import faiss

from autodidact.types import (
    AutodidactConfig,
    KnowledgeCategory,
    KnowledgeEntry,
    KnowledgeScope,
    MemoryTier,
    NewKnowledgeEntry,
)


class ScoredKnowledgeEntry:
    """A knowledge entry with its similarity score."""

    def __init__(self, entry: KnowledgeEntry, score: float) -> None:
        self.entry = entry
        self.score = score


def _synchronized(method):
    """Serialize a method on ``self.lock``.

    The sqlite connection and FAISS index are shared with the background
    ``_learn`` thread spawned on cloud escalation. CPython segfaults when one
    sqlite3 connection is used from two threads at once, so every method that
    touches the connection or the index must hold the lock.
    """

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        with self.lock:
            return method(self, *args, **kwargs)

    return wrapper


class MixedEmbeddingDimensionError(ValueError):
    """Raised when the knowledge store contains embeddings of mixed dimensions.

    This typically means the embedding model was changed (e.g. nomic-embed-text at
    768 dim swapped for bge-large-en-v1.5 at 1024 dim) without re-seeding the KB.
    FAISS cannot build an index over mixed-dim vectors, and cosine similarity
    between different-dim vectors is meaningless anyway.

    Recovery: drop the knowledge_entries table and re-seed.

        sqlite3 autodidact_experiment.db "DELETE FROM knowledge_entries"
        python -m benchmarks.seeding --n 500 --seed 42 \\
            --embedding-model qllama/bge-large-en-v1.5
    """


class KnowledgeStore:
    """Hierarchical knowledge store with Ebbinghaus decay and cosine similarity search."""

    def __init__(self, conn: sqlite3.Connection, config: AutodidactConfig) -> None:
        self.conn = conn
        self.config = config
        # Reentrant so a @_synchronized public method can call another one
        # (e.g. _faiss_search → get) without deadlocking. Shared with the Agent,
        # which writes query_log on the same connection from the main thread
        # while the background _learn thread writes here.
        self.lock = threading.RLock()
        self._faiss_index: Optional[faiss.IndexFlatIP] = None
        self._faiss_ids: list[str] = []
        self._faiss_dirty = True
        # Cached dim of first valid entry; None means "not yet checked / store empty".
        # Reset on _faiss_dirty to catch any out-of-band mutations.
        self._expected_dim: Optional[int] = None

    @_synchronized
    def insert(self, entry: NewKnowledgeEntry) -> KnowledgeEntry:
        """Insert a new knowledge entry into STM."""
        now = datetime.now(timezone.utc).isoformat()
        entry_id = str(uuid.uuid4())

        embedding_blob = None
        if entry.embedding is not None:
            new_dim = len(entry.embedding)
            existing_dim = self._get_existing_embedding_dim()
            if existing_dim is not None and existing_dim != new_dim:
                raise MixedEmbeddingDimensionError(
                    f"Cannot insert embedding of dim {new_dim}: knowledge store "
                    f"already contains {existing_dim}-dim embeddings. This usually "
                    f"means the embedding model was changed without re-seeding. "
                    f"Drop the knowledge_entries table and re-run seeding. "
                    f"See autodidact.knowledge_store.MixedEmbeddingDimensionError "
                    f"for recovery instructions."
                )
            embedding_blob = np.array(entry.embedding, dtype=np.float32).tobytes()

        # Answer-side embedding is analysis-only in v0.1 (not wired into retrieval).
        # We validate dim against the same expected_dim so mixed-embedder bugs are
        # caught on this column too.
        answer_embedding_blob = None
        if entry.answer_embedding is not None:
            ans_dim = len(entry.answer_embedding)
            existing_dim = self._get_existing_embedding_dim()
            if existing_dim is not None and existing_dim != ans_dim:
                raise MixedEmbeddingDimensionError(
                    f"Cannot insert answer_embedding of dim {ans_dim}: knowledge store "
                    f"already contains {existing_dim}-dim embeddings."
                )
            answer_embedding_blob = np.array(entry.answer_embedding, dtype=np.float32).tobytes()

        self.conn.execute(
            """INSERT INTO knowledge_entries
            (id, content, question, source, confidence, tags, embedding, answer_embedding,
             tier, usage_count,
             created_at, last_accessed, metadata, domain, topic, category,
             valid_from, valid_to, verbatim_response)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'STM', 0, ?, ?, ?, ?, ?, ?, ?, NULL, ?)""",
            (
                entry_id,
                entry.content,
                entry.question,
                entry.source,
                entry.confidence,
                json.dumps(entry.tags),
                embedding_blob,
                answer_embedding_blob,
                now,
                now,
                json.dumps(entry.metadata),
                entry.domain,
                entry.topic,
                entry.category.value,
                now,
                entry.verbatim_response,
            ),
        )
        self.conn.commit()
        self._faiss_dirty = True

        return KnowledgeEntry(
            id=entry_id,
            content=entry.content,
            question=entry.question,
            source=entry.source,
            confidence=entry.confidence,
            tags=entry.tags,
            embedding=entry.embedding,
            answer_embedding=entry.answer_embedding,
            tier=MemoryTier.STM,
            usage_count=0,
            created_at=now,
            last_accessed=now,
            domain=entry.domain,
            topic=entry.topic,
            category=entry.category,
            valid_from=now,
            verbatim_response=entry.verbatim_response,
        )

    @_synchronized
    def insert_batch(self, entries: list[NewKnowledgeEntry]) -> int:
        """Insert multiple entries in a single transaction. Returns count inserted."""
        if not entries:
            return 0
        # Validate dim on first entry with an embedding.
        for entry in entries:
            if entry.embedding is not None:
                new_dim = len(entry.embedding)
                existing_dim = self._get_existing_embedding_dim()
                if existing_dim is not None and existing_dim != new_dim:
                    raise MixedEmbeddingDimensionError(
                        f"Cannot batch-insert embeddings of dim {new_dim}: "
                        f"store contains {existing_dim}-dim embeddings."
                    )
                break
        now = datetime.now(timezone.utc).isoformat()
        count = 0
        for entry in entries:
            entry_id = str(uuid.uuid4())
            embedding_blob = None
            if entry.embedding is not None:
                embedding_blob = np.array(entry.embedding, dtype=np.float32).tobytes()
            answer_embedding_blob = None
            if entry.answer_embedding is not None:
                answer_embedding_blob = np.array(entry.answer_embedding, dtype=np.float32).tobytes()
            self.conn.execute(
                """INSERT INTO knowledge_entries
                (id, content, question, source, confidence, tags, embedding, answer_embedding,
                 tier, usage_count,
                 created_at, last_accessed, metadata, domain, topic, category,
                 valid_from, valid_to, verbatim_response)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'STM', 0, ?, ?, ?, ?, ?, ?, ?, NULL, ?)""",
                (
                    entry_id, entry.content, entry.question, entry.source,
                    entry.confidence, json.dumps(entry.tags), embedding_blob,
                    answer_embedding_blob, now, now, json.dumps(entry.metadata),
                    entry.domain, entry.topic, entry.category.value, now,
                    entry.verbatim_response,
                ),
            )
            count += 1
        self.conn.commit()
        self._faiss_dirty = True
        return count

    @_synchronized
    def search(
        self,
        query_embedding: np.ndarray,
        limit: int = 5,
        scope: Optional[KnowledgeScope] = None,
        as_of: Optional[str] = None,
        min_similarity: Optional[float] = None,
    ) -> list[ScoredKnowledgeEntry]:
        """Search knowledge entries by cosine similarity with minimum threshold.

        Uses FAISS for fast approximate nearest neighbor search when no scope
        filters are applied. Falls back to filtered brute-force when scoped.

        Parameters
        ----------
        min_similarity
            Per-call override for the similarity floor. If None, uses
            ``config.similarity_threshold``. Different consumers have
            different information appetites (see EXP-002 and LAB_NOTES P19):
            GSA wants strong hits or none (0.70+), answer-injection wants
            medium (0.60), knowledge_similarity-as-feature wants raw top-k
            (0.0). Pass a per-call value rather than mutating config.
        """
        threshold = (
            min_similarity
            if min_similarity is not None
            else self.config.similarity_threshold
        )
        # If scoped or historical, use filtered search (can't use FAISS index directly)
        if scope or as_of:
            return self._filtered_search(query_embedding, limit, scope, as_of, threshold)

        # Use FAISS for unscoped search on current entries
        return self._faiss_search(query_embedding, limit, threshold)

    def _faiss_search(
        self, query_embedding: np.ndarray, limit: int, threshold: float
    ) -> list[ScoredKnowledgeEntry]:
        """Fast search using FAISS inner-product index."""
        self._rebuild_faiss_if_dirty()

        if self._faiss_index is None or self._faiss_index.ntotal == 0:
            return []

        # Normalize query for cosine similarity via inner product
        query_norm = query_embedding.astype(np.float32).reshape(1, -1)
        faiss.normalize_L2(query_norm)

        # Search more than limit to account for threshold filtering
        k = min(limit * 3, self._faiss_index.ntotal)
        scores, indices = self._faiss_index.search(query_norm, k)

        results: list[ScoredKnowledgeEntry] = []
        for i in range(k):
            idx = int(indices[0][i])
            sim = float(scores[0][i])
            if idx < 0 or sim < threshold:
                continue
            entry_id = self._faiss_ids[idx]
            entry = self.get(entry_id)
            if entry and entry.valid_to is None:
                results.append(ScoredKnowledgeEntry(entry=entry, score=sim))
            if len(results) >= limit:
                break

        return results

    def _rebuild_faiss_if_dirty(self) -> None:
        """Rebuild the FAISS index from current valid entries."""
        if not self._faiss_dirty:
            return

        rows = self.conn.execute(
            "SELECT id, embedding FROM knowledge_entries WHERE embedding IS NOT NULL AND valid_to IS NULL"
        ).fetchall()

        if not rows:
            self._faiss_index = None
            self._faiss_ids = []
            self._faiss_dirty = False
            return

        # Determine embedding dimension from first entry and validate all others match.
        # A mixed-dim store typically indicates an embedder swap without re-seeding;
        # numpy would raise a cryptic broadcast error below — raise a clearer one now.
        first_emb = np.frombuffer(rows[0]["embedding"], dtype=np.float32)
        dim = len(first_emb)
        self._expected_dim = dim

        # Build normalized embedding matrix
        embeddings = np.zeros((len(rows), dim), dtype=np.float32)
        ids: list[str] = []
        for i, row in enumerate(rows):
            emb = np.frombuffer(row["embedding"], dtype=np.float32)
            if len(emb) != dim:
                raise MixedEmbeddingDimensionError(
                    f"Knowledge store contains mixed embedding dimensions: "
                    f"first entry has dim {dim}, entry at index {i} has dim {len(emb)}. "
                    f"This usually means the embedding model was changed without "
                    f"re-seeding. Drop the knowledge_entries table and re-run seeding. "
                    f"See autodidact.knowledge_store.MixedEmbeddingDimensionError "
                    f"for recovery instructions."
                )
            embeddings[i] = emb
            ids.append(row["id"])

        faiss.normalize_L2(embeddings)

        # Inner product on normalized vectors = cosine similarity
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)

        self._faiss_index = index
        self._faiss_ids = ids
        self._faiss_dirty = False

    def _filtered_search(
        self,
        query_embedding: np.ndarray,
        limit: int,
        scope: Optional[KnowledgeScope] = None,
        as_of: Optional[str] = None,
        threshold: Optional[float] = None,
    ) -> list[ScoredKnowledgeEntry]:
        """Filtered brute-force search for scoped/historical queries."""
        if threshold is None:
            threshold = self.config.similarity_threshold
        # Build scope filter
        conditions = ["valid_to IS NULL"]
        params: list = []

        if as_of:
            conditions = [
                "valid_from <= ?",
                "(valid_to IS NULL OR valid_to > ?)",
            ]
            params = [as_of, as_of]

        if scope:
            if scope.domain:
                conditions.append("domain = ?")
                params.append(scope.domain)
            if scope.topic:
                conditions.append("topic = ?")
                params.append(scope.topic)
            if scope.category:
                conditions.append("category = ?")
                params.append(scope.category.value)

        where_clause = " AND ".join(conditions)
        rows = self.conn.execute(
            f"SELECT * FROM knowledge_entries WHERE embedding IS NOT NULL AND {where_clause}",
            params,
        ).fetchall()

        if not rows:
            return []

        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-10)
        scored: list[ScoredKnowledgeEntry] = []

        for row in rows:
            emb = np.frombuffer(row["embedding"], dtype=np.float32)
            emb_norm = emb / (np.linalg.norm(emb) + 1e-10)
            sim = float(np.dot(query_norm, emb_norm))

            if sim < threshold:
                continue

            entry = self._row_to_entry(row)
            scored.append(ScoredKnowledgeEntry(entry=entry, score=sim))

        scored.sort(key=lambda x: x.score, reverse=True)
        return scored[:limit]

    @_synchronized
    def get(self, entry_id: str) -> Optional[KnowledgeEntry]:
        """Get a knowledge entry by ID."""
        row = self.conn.execute(
            "SELECT * FROM knowledge_entries WHERE id = ?", (entry_id,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_entry(row)

    @_synchronized
    def access(self, entry_id: str) -> None:
        """Record an access — updates usage_count and last_accessed."""
        now = datetime.now(timezone.utc).isoformat()
        self.conn.execute(
            "UPDATE knowledge_entries SET usage_count = usage_count + 1, last_accessed = ? WHERE id = ?",
            (now, entry_id),
        )
        self.conn.commit()

    @_synchronized
    def promote_to_ltm(self, entry_id: str) -> None:
        """Promote an STM entry to LTM."""
        now = datetime.now(timezone.utc).isoformat()
        self.conn.execute(
            "UPDATE knowledge_entries SET tier = 'LTM', promoted_at = ? WHERE id = ? AND tier = 'STM'",
            (now, entry_id),
        )
        self.conn.commit()

    @_synchronized
    def invalidate(self, entry_id: str) -> None:
        """Soft-invalidate by setting valid_to."""
        now = datetime.now(timezone.utc).isoformat()
        self.conn.execute(
            "UPDATE knowledge_entries SET valid_to = ? WHERE id = ?",
            (now, entry_id),
        )
        self.conn.commit()
        self._faiss_dirty = True

    @_synchronized
    def run_decay_cycle(self, current_time: Optional[datetime] = None) -> dict[str, int]:
        """Apply Ebbinghaus decay. Returns counts of expired and promoted entries.

        R(t) = e^(-t/S) where S = base_stability × (1 + ln(1 + access_count))
        """
        if current_time is None:
            current_time = datetime.now(timezone.utc)

        now_iso = current_time.isoformat()
        expired = 0
        promoted = 0

        # Check STM entries for promotion
        stm_rows = self.conn.execute(
            "SELECT id, usage_count FROM knowledge_entries WHERE tier = 'STM' AND valid_to IS NULL"
        ).fetchall()
        for row in stm_rows:
            if row["usage_count"] >= self.config.stm_promotion_accesses:
                self.conn.execute(
                    "UPDATE knowledge_entries SET tier = 'LTM', promoted_at = ? WHERE id = ?",
                    (now_iso, row["id"]),
                )
                promoted += 1

        # Apply Ebbinghaus decay to LTM entries
        ltm_rows = self.conn.execute(
            "SELECT id, last_accessed, usage_count FROM knowledge_entries WHERE tier = 'LTM' AND valid_to IS NULL"
        ).fetchall()
        for row in ltm_rows:
            last_accessed = datetime.fromisoformat(row["last_accessed"])
            if last_accessed.tzinfo is None:
                last_accessed = last_accessed.replace(tzinfo=timezone.utc)
            elapsed_hours = (current_time - last_accessed).total_seconds() / 3600.0
            stability = self.config.base_stability * (
                1 + math.log(1 + row["usage_count"])
            )
            retention = math.exp(-elapsed_hours / stability) if stability > 0 else 0.0

            if retention < self.config.decay_threshold:
                self.conn.execute(
                    "UPDATE knowledge_entries SET valid_to = ? WHERE id = ?",
                    (now_iso, row["id"]),
                )
                expired += 1

        self.conn.commit()
        return {"expired": expired, "promoted": promoted}

    @staticmethod
    def ebbinghaus_retention(
        elapsed_hours: float, access_count: int, base_stability: float = 1.0
    ) -> float:
        """Compute Ebbinghaus retention: R(t) = e^(-t/S).

        S = base_stability × (1 + ln(1 + access_count))
        """
        stability = base_stability * (1 + math.log(1 + access_count))
        if stability <= 0:
            return 0.0
        return math.exp(-elapsed_hours / stability)

    @_synchronized
    def get_stats(self) -> dict:
        """Return knowledge store statistics."""
        total = self.conn.execute("SELECT COUNT(*) as cnt FROM knowledge_entries WHERE valid_to IS NULL").fetchone()["cnt"]
        stm = self.conn.execute("SELECT COUNT(*) as cnt FROM knowledge_entries WHERE tier = 'STM' AND valid_to IS NULL").fetchone()["cnt"]
        ltm = self.conn.execute("SELECT COUNT(*) as cnt FROM knowledge_entries WHERE tier = 'LTM' AND valid_to IS NULL").fetchone()["cnt"]
        return {"total": total, "stm": stm, "ltm": ltm}

    def _get_existing_embedding_dim(self) -> Optional[int]:
        """Return the embedding dim of the first valid entry with a non-null embedding.

        Used by insert() to detect embedder swaps without re-seeding. Cached on
        first call; invalidated whenever the FAISS index is marked dirty (which
        _rebuild_faiss_if_dirty also repopulates as a side effect).
        """
        if self._expected_dim is not None:
            return self._expected_dim

        row = self.conn.execute(
            "SELECT embedding FROM knowledge_entries "
            "WHERE embedding IS NOT NULL AND valid_to IS NULL "
            "LIMIT 1"
        ).fetchone()
        if row is None:
            return None

        dim = len(np.frombuffer(row["embedding"], dtype=np.float32))
        self._expected_dim = dim
        return dim

    @_synchronized
    def list_domains(self) -> list[str]:
        rows = self.conn.execute("SELECT DISTINCT domain FROM knowledge_entries WHERE valid_to IS NULL").fetchall()
        return [r["domain"] for r in rows]

    @_synchronized
    def list_topics(self, domain: str) -> list[str]:
        rows = self.conn.execute(
            "SELECT DISTINCT topic FROM knowledge_entries WHERE domain = ? AND valid_to IS NULL",
            (domain,),
        ).fetchall()
        return [r["topic"] for r in rows]

    @_synchronized
    def get_all_embeddings(self) -> list[np.ndarray]:
        """Return all valid entry embeddings (for confidence evaluator)."""
        rows = self.conn.execute(
            "SELECT embedding FROM knowledge_entries WHERE embedding IS NOT NULL AND valid_to IS NULL"
        ).fetchall()
        return [np.frombuffer(r["embedding"], dtype=np.float32) for r in rows]

    @_synchronized
    def count(self) -> int:
        row = self.conn.execute("SELECT COUNT(*) as cnt FROM knowledge_entries WHERE valid_to IS NULL").fetchone()
        return row["cnt"]

    # ── Internal ─────────────────────────────────────────────────────

    def _row_to_entry(self, row: sqlite3.Row) -> KnowledgeEntry:
        embedding = None
        if row["embedding"]:
            embedding = np.frombuffer(row["embedding"], dtype=np.float32).tolist()
        # answer_embedding was added in a later schema migration; older rows
        # may lack the column entirely.
        answer_embedding = None
        try:
            if row["answer_embedding"]:
                answer_embedding = np.frombuffer(row["answer_embedding"], dtype=np.float32).tolist()
        except (IndexError, KeyError):
            answer_embedding = None
        # Defensive: older rows may lack the 'question' column after migration
        question = None
        try:
            question = row["question"]
        except (IndexError, KeyError):
            question = None
        return KnowledgeEntry(
            id=row["id"],
            content=row["content"],
            question=question,
            source=row["source"],
            confidence=row["confidence"],
            tags=json.loads(row["tags"]),
            embedding=embedding,
            answer_embedding=answer_embedding,
            tier=MemoryTier(row["tier"]),
            usage_count=row["usage_count"],
            created_at=row["created_at"],
            last_accessed=row["last_accessed"],
            promoted_at=row["promoted_at"],
            metadata=json.loads(row["metadata"]),
            domain=row["domain"],
            topic=row["topic"],
            category=KnowledgeCategory(row["category"]),
            valid_from=row["valid_from"],
            valid_to=row["valid_to"],
            verbatim_response=row["verbatim_response"],
        )

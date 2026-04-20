"""Hierarchical Knowledge Store with Ebbinghaus Decay.

SQLite-backed storage with domain/topic/category hierarchy,
cosine similarity search, STM/LTM tiers, and spaced-repetition decay.
"""

from __future__ import annotations

import json
import math
import sqlite3
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


class KnowledgeStore:
    """Hierarchical knowledge store with Ebbinghaus decay and cosine similarity search."""

    def __init__(self, conn: sqlite3.Connection, config: AutodidactConfig) -> None:
        self.conn = conn
        self.config = config
        self._faiss_index: Optional[faiss.IndexFlatIP] = None
        self._faiss_ids: list[str] = []
        self._faiss_dirty = True

    def insert(self, entry: NewKnowledgeEntry) -> KnowledgeEntry:
        """Insert a new knowledge entry into STM."""
        now = datetime.now(timezone.utc).isoformat()
        entry_id = str(uuid.uuid4())

        embedding_blob = None
        if entry.embedding is not None:
            embedding_blob = np.array(entry.embedding, dtype=np.float32).tobytes()

        self.conn.execute(
            """INSERT INTO knowledge_entries
            (id, content, source, confidence, tags, embedding, tier, usage_count,
             created_at, last_accessed, metadata, domain, topic, category,
             valid_from, valid_to, verbatim_response)
            VALUES (?, ?, ?, ?, ?, ?, 'STM', 0, ?, ?, ?, ?, ?, ?, ?, NULL, ?)""",
            (
                entry_id,
                entry.content,
                entry.source,
                entry.confidence,
                json.dumps(entry.tags),
                embedding_blob,
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
            source=entry.source,
            confidence=entry.confidence,
            tags=entry.tags,
            embedding=entry.embedding,
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

    def search(
        self,
        query_embedding: np.ndarray,
        limit: int = 5,
        scope: Optional[KnowledgeScope] = None,
        as_of: Optional[str] = None,
    ) -> list[ScoredKnowledgeEntry]:
        """Search knowledge entries by cosine similarity with minimum threshold.

        Uses FAISS for fast approximate nearest neighbor search when no scope
        filters are applied. Falls back to filtered brute-force when scoped.
        """
        # If scoped or historical, use filtered search (can't use FAISS index directly)
        if scope or as_of:
            return self._filtered_search(query_embedding, limit, scope, as_of)

        # Use FAISS for unscoped search on current entries
        return self._faiss_search(query_embedding, limit)

    def _faiss_search(
        self, query_embedding: np.ndarray, limit: int
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
            if idx < 0 or sim < self.config.similarity_threshold:
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

        # Determine embedding dimension from first entry
        first_emb = np.frombuffer(rows[0]["embedding"], dtype=np.float32)
        dim = len(first_emb)

        # Build normalized embedding matrix
        embeddings = np.zeros((len(rows), dim), dtype=np.float32)
        ids: list[str] = []
        for i, row in enumerate(rows):
            emb = np.frombuffer(row["embedding"], dtype=np.float32)
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
    ) -> list[ScoredKnowledgeEntry]:
        """Filtered brute-force search for scoped/historical queries."""
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

            if sim < self.config.similarity_threshold:
                continue

            entry = self._row_to_entry(row)
            scored.append(ScoredKnowledgeEntry(entry=entry, score=sim))

        scored.sort(key=lambda x: x.score, reverse=True)
        return scored[:limit]

    def get(self, entry_id: str) -> Optional[KnowledgeEntry]:
        """Get a knowledge entry by ID."""
        row = self.conn.execute(
            "SELECT * FROM knowledge_entries WHERE id = ?", (entry_id,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_entry(row)

    def access(self, entry_id: str) -> None:
        """Record an access — updates usage_count and last_accessed."""
        now = datetime.now(timezone.utc).isoformat()
        self.conn.execute(
            "UPDATE knowledge_entries SET usage_count = usage_count + 1, last_accessed = ? WHERE id = ?",
            (now, entry_id),
        )
        self.conn.commit()

    def promote_to_ltm(self, entry_id: str) -> None:
        """Promote an STM entry to LTM."""
        now = datetime.now(timezone.utc).isoformat()
        self.conn.execute(
            "UPDATE knowledge_entries SET tier = 'LTM', promoted_at = ? WHERE id = ? AND tier = 'STM'",
            (now, entry_id),
        )
        self.conn.commit()

    def invalidate(self, entry_id: str) -> None:
        """Soft-invalidate by setting valid_to."""
        now = datetime.now(timezone.utc).isoformat()
        self.conn.execute(
            "UPDATE knowledge_entries SET valid_to = ? WHERE id = ?",
            (now, entry_id),
        )
        self.conn.commit()
        self._faiss_dirty = True

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

    def get_stats(self) -> dict:
        """Return knowledge store statistics."""
        total = self.conn.execute("SELECT COUNT(*) as cnt FROM knowledge_entries WHERE valid_to IS NULL").fetchone()["cnt"]
        stm = self.conn.execute("SELECT COUNT(*) as cnt FROM knowledge_entries WHERE tier = 'STM' AND valid_to IS NULL").fetchone()["cnt"]
        ltm = self.conn.execute("SELECT COUNT(*) as cnt FROM knowledge_entries WHERE tier = 'LTM' AND valid_to IS NULL").fetchone()["cnt"]
        return {"total": total, "stm": stm, "ltm": ltm}

    def list_domains(self) -> list[str]:
        rows = self.conn.execute("SELECT DISTINCT domain FROM knowledge_entries WHERE valid_to IS NULL").fetchall()
        return [r["domain"] for r in rows]

    def list_topics(self, domain: str) -> list[str]:
        rows = self.conn.execute(
            "SELECT DISTINCT topic FROM knowledge_entries WHERE domain = ? AND valid_to IS NULL",
            (domain,),
        ).fetchall()
        return [r["topic"] for r in rows]

    def get_all_embeddings(self) -> list[np.ndarray]:
        """Return all valid entry embeddings (for confidence evaluator)."""
        rows = self.conn.execute(
            "SELECT embedding FROM knowledge_entries WHERE embedding IS NOT NULL AND valid_to IS NULL"
        ).fetchall()
        return [np.frombuffer(r["embedding"], dtype=np.float32) for r in rows]

    def count(self) -> int:
        row = self.conn.execute("SELECT COUNT(*) as cnt FROM knowledge_entries WHERE valid_to IS NULL").fetchone()
        return row["cnt"]

    # ── Internal ─────────────────────────────────────────────────────

    def _row_to_entry(self, row: sqlite3.Row) -> KnowledgeEntry:
        embedding = None
        if row["embedding"]:
            embedding = np.frombuffer(row["embedding"], dtype=np.float32).tolist()
        return KnowledgeEntry(
            id=row["id"],
            content=row["content"],
            source=row["source"],
            confidence=row["confidence"],
            tags=json.loads(row["tags"]),
            embedding=embedding,
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

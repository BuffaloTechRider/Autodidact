"""Autodidact Agent - the self-learning AI that routes and remembers.

The Agent is the central API. It accepts a user query, decides how to answer
it (from memory, locally, or via cloud), and learns from every cloud escalation.

Usage:
    from autodidact import Agent

    agent = Agent(local_model="ollama/qwen2.5:7b", cloud_model="openai/gpt-4o")
    response = agent.query("What is the capital of France?")
    print(response.answer)       # "Paris"
    print(response.routed_to)    # "local"
    print(response.confidence)   # 0.92
    print(response.cost_usd)     # 0.0

See CONTEXT.md for precise definitions of routing, escalation, learning, and memory.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from autodidact.database import init_database
from autodidact.document_store import DocumentStore, ScoredChunk
from autodidact.knowledge_store import KnowledgeStore, ScoredKnowledgeEntry
from autodidact.learning_extractor import ExtractionResult, LearningExtractor
from autodidact.llm_client import ChatMessage, ChatResponseWithLogprobs, LLMClient, LLMConfig
from autodidact.signals.grounded_self_assessment import SelfAssessment
from autodidact.types import AutodidactConfig, NewKnowledgeEntry

# Type alias for progress callbacks.
ProgressCallback = Optional[Callable[[dict], None]]

logger = logging.getLogger(__name__)

# ── Cost rates (USD per million tokens) for savings estimation ─────
# Subset of the rates from benchmarks/ablation_experiment.py.
# Users can override via config; these are sensible defaults.
_DEFAULT_COST_RATES = {
    "gpt-4o": {"input": 2.5, "output": 10.0},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
    "claude-sonnet-4-5": {"input": 3.0, "output": 15.0},
    "claude-haiku": {"input": 0.25, "output": 1.25},
}

# ── Similarity thresholds for memory retrieval tiers ───────────────
MEMORY_DIRECT_THRESHOLD = 0.85   # return stored answer directly
MEMORY_CONTEXT_THRESHOLD = 0.60  # inject as reference context
MEMORY_STALENESS_DAYS = 7        # re-verify entries older than this


# ── Response model ─────────────────────────────────────────────────

@dataclass
class QueryResponse:
    """Result of an agent.query() call."""

    answer: str
    routed_to: str  # "local", "cloud", or "memory"
    confidence: float  # 0.0-1.0; for memory answers, similarity score
    cost_usd: float
    learned: bool  # True if a new KB entry was stored
    latency_ms: int
    memory_source: Optional[str] = None  # the past question it recalled, if any
    memory_age_days: Optional[float] = None  # how old the memory entry is
    stale: bool = False  # True if memory answer is older than staleness threshold
    escalated_on_refusal: bool = False  # True if local refused and we forced cloud
    escalated_on_gsa: bool = False  # True if GSA pre-gate vetoed local
    gsa_p_yes: Optional[float] = None  # p_yes from the pre-local self-assessment probe


@dataclass
class SavingsReport:
    """Cumulative cost savings statistics."""

    total_queries: int = 0
    local_queries: int = 0
    cloud_queries: int = 0
    memory_queries: int = 0
    total_cost_usd: float = 0.0
    estimated_all_cloud_cost_usd: float = 0.0
    saved_usd: float = 0.0
    saved_pct: float = 0.0
    facts_learned: int = 0


# ── Agent ──────────────────────────────────────────────────────────

class Agent:
    """The self-learning AI agent.

    Accepts queries, routes between local and cloud models based on confidence,
    and learns from every cloud escalation by storing Q&A pairs in a knowledge
    store for future retrieval.
    """

    def __init__(
        self,
        local_model: Optional[str] = None,
        cloud_model: Optional[str] = None,
        *,
        cloud_provider: str = "openai",
        cloud_base_url: Optional[str] = None,
        cloud_api_key_env: Optional[str] = None,
        cloud_region: str = "us-west-2",
        cloud_bedrock: Optional[dict] = None,
        local_base_url: Optional[str] = None,
        local_api_key_env: Optional[str] = None,
        local_region: str = "us-west-2",
        local_bedrock: Optional[dict] = None,
        embedding_model: Optional[str] = None,
        db_path: str = "~/.autodidact/memory.db",
        confidence_threshold: float = 0.7,
        staleness_days: float = MEMORY_STALENESS_DAYS,
        gsa_enabled: bool = True,
        gsa_threshold: float = 0.5,
    ) -> None:
        self.confidence_threshold = confidence_threshold
        self.staleness_days = staleness_days
        self.gsa_enabled = gsa_enabled
        self.gsa_threshold = gsa_threshold

        # Expand ~ in db_path and ensure parent dir exists.
        self._db_path = str(Path(db_path).expanduser())
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)

        # Initialize database and knowledge store.
        self._conn = init_database(self._db_path)
        self._config = AutodidactConfig(db_path=self._db_path)
        self.memory = KnowledgeStore(self._conn, self._config)

        # "Local" model client — normally Ollama, but can be any provider for
        # cloud-to-cloud mode (cheap cloud model in the local slot, expensive
        # in the cloud slot).
        self._local_client: Optional[LLMClient] = None
        self._local_model_name = local_model
        if local_model:
            # Parse "ollama/qwen2.5:7b" → provider="ollama", model="qwen2.5:7b".
            # "openai/gpt-4o-mini" → provider="openai", model="gpt-4o-mini".
            provider, model = _parse_model_string(local_model, default_provider="ollama")
            emb_model = embedding_model or "qllama/bge-large-en-v1.5"
            # Strip ONLY known provider prefixes (ollama/, openai/, bedrock/) —
            # not arbitrary namespaces. Ollama models commonly live under
            # third-party namespaces like 'qllama/bge-large-en-v1.5' or
            # 'hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF' where the slash is
            # part of the model's identity. Stripping those breaks embedding
            # lookups (Ollama 404s on the bare name).
            if "/" in emb_model:
                first_segment, rest = emb_model.split("/", 1)
                if first_segment.lower() in ("ollama", "openai", "bedrock"):
                    emb_model = rest
            local_config_kwargs: dict = {
                "provider": provider,
                "model": model,
                "embedding_model": emb_model,
            }
            # Wire provider-specific settings when the "local" slot isn't Ollama.
            if provider == "openai":
                local_config_kwargs["base_url"] = local_base_url or "https://api.openai.com/v1"
                local_config_kwargs["api_key_env"] = local_api_key_env or "OPENAI_API_KEY"
            elif provider == "bedrock":
                local_config_kwargs["region"] = local_region
                if local_bedrock:
                    _apply_bedrock_auth(local_config_kwargs, local_bedrock)
            self._local_client = LLMClient(LLMConfig(**local_config_kwargs))

        # Cloud model client.
        self._cloud_client: Optional[LLMClient] = None
        self._cloud_model_name = cloud_model
        if cloud_model:
            provider, model = _parse_model_string(cloud_model, default_provider=cloud_provider)
            config_kwargs: dict = {"provider": provider, "model": model}
            if provider == "openai":
                config_kwargs["base_url"] = cloud_base_url or "https://api.openai.com/v1"
                config_kwargs["api_key_env"] = cloud_api_key_env or "OPENAI_API_KEY"
            elif provider == "bedrock":
                config_kwargs["region"] = cloud_region
                if cloud_bedrock:
                    _apply_bedrock_auth(config_kwargs, cloud_bedrock)
            self._cloud_client = LLMClient(LLMConfig(**config_kwargs))

        # Embedding client — reuse local client if available, else cloud.
        self._embed_client = self._local_client or self._cloud_client

        # Session stats.
        self._session_stats = SavingsReport()

        # Conversation history (in-session only).
        self._history: list[dict] = []  # [{"role": "user"/"assistant", "content": "..."}]

        # Document store for ingested source materials (R9). Separate from
        # agent memory per AD-002. None unless attached via attach_document_store()
        # or set by the caller directly. Retrieved alongside memory at query
        # time with different prompt framing.
        self.documents: Optional[DocumentStore] = None

        # GSA (grounded self-assessment) probe — runs before local generation
        # when enabled. Built lazily on first use so tests can patch it.
        self._gsa: Optional[SelfAssessment] = None

    # ── Public API ────────────────────────────────────────────────

    def attach_document_store(self, store: DocumentStore) -> None:
        """Wire an existing DocumentStore into this agent.

        Document chunks will be retrieved alongside agent memory at query
        time and injected into the prompt with distinct framing ('from your
        documents' vs 'from past interactions').
        """
        self.documents = store

    def query(
        self,
        question: str,
        context: Optional[str] = None,
        *,
        on_progress: ProgressCallback = None,
    ) -> QueryResponse:
        """Ask the agent a question. It thinks, routes, and learns.

        Parameters
        ----------
        question
            The user's question.
        context
            Optional external context (e.g., from a RAG pipeline). Injected
            into the prompt alongside any memory context the agent retrieves.
        on_progress
            Optional callback for real-time UI updates. Called with a dict
            containing at minimum a "type" key. Event types:
            - thinking: memory search started, includes memory_hits count
            - memory_hit: answering from memory (high similarity)
            - local_done: local model answered, includes confidence
            - cloud_call: escalating to cloud, includes model name
            - cloud_done: cloud response received, includes cost and model
            - learning: knowledge stored, includes knowledge_count
        """
        started = time.perf_counter()

        def _emit(event: dict) -> None:
            if on_progress is not None:
                on_progress(event)

        # ── Stage 1: Check memory ────────────────────────────────
        memory_hits = self._check_memory(question)
        best_hit = memory_hits[0] if memory_hits else None

        _emit({
            "type": "thinking",
            "memory_hits": len(memory_hits),
            "best_similarity": best_hit.score if best_hit else 0.0,
        })

        if best_hit and best_hit.score >= MEMORY_DIRECT_THRESHOLD:
            # Direct memory answer — no generation needed.
            entry = best_hit.entry
            self.memory.access(entry.id)
            age_days = self._entry_age_days(entry)
            is_stale = age_days > self.staleness_days

            if is_stale:
                # Stale memory: fall through to Stage 2 (local generation).
                # Many facts are stable for months or years; escalating every
                # stale hit wastes cloud dollars on queries local can handle.
                # Only escalate if local is ALSO uncertain — matching the
                # original routing intent: escalate when uncertain, not when
                # memory is merely old.
                logger.info(
                    "Memory hit is stale (%.1f days old); falling through to local",
                    age_days,
                )
            else:
                _emit({
                    "type": "memory_hit",
                    "similarity": best_hit.score,
                    "memory_source": entry.question,
                    "age_days": age_days,
                })

                latency = _elapsed_ms(started)
                self._record_query("memory", 0.0, best_hit.score, latency, question=question)
                self._append_history(question, entry.content)
                return QueryResponse(
                    answer=entry.content,
                    routed_to="memory",
                    confidence=best_hit.score,
                    cost_usd=0.0,
                    learned=False,
                    latency_ms=latency,
                    memory_source=entry.question,
                    memory_age_days=age_days,
                    stale=False,
                )

        # ── Stage 1.5: GSA pre-gate ──────────────────────────────
        # Ask the local model itself: "can you answer this?" before spending
        # time generating a full response. If it self-reports NO with high
        # probability, skip local and escalate directly. This catches the
        # class of failures where the model would fabricate a plausible-
        # sounding answer (high logprobs) to a question it has no real
        # knowledge of — logprob-based confidence can't see the hallucination
        # but a separate Y/N self-probe often can.
        gsa_p_yes: Optional[float] = None
        # Tolerate agents built via Agent.__new__ in older tests that don't
        # set the GSA attrs. getattr defaults match __init__ defaults.
        gsa_enabled = getattr(self, "gsa_enabled", True)
        gsa_threshold = getattr(self, "gsa_threshold", 0.5)
        if (
            gsa_enabled
            and self._local_client is not None
            and self._cloud_client is not None
        ):
            try:
                if getattr(self, "_gsa", None) is None:
                    self._gsa = SelfAssessment(self._local_client)
                gsa_result = self._gsa.compute(question, retrieved_hits=memory_hits)
                gsa_p_yes = gsa_result.p_yes
                if gsa_p_yes < gsa_threshold:
                    # Model self-reports it can't answer — skip local entirely.
                    resp = self._escalate_to_cloud(
                        question, context, memory_hits, started, _emit,
                    )
                    resp.escalated_on_gsa = True
                    resp.gsa_p_yes = gsa_p_yes
                    return resp
            except Exception as e:
                # GSA is a best-effort signal. A probe failure must not block
                # the actual query — fall through to the normal path.
                logger.warning("GSA probe failed, skipping gate: %s", e)

        # ── Stage 2: Generate locally ────────────────────────────
        if self._local_client is None:
            # No local model — go straight to cloud.
            if self._cloud_client is None:
                return QueryResponse(
                    answer="No model configured. Run `autodidact init` to set up.",
                    routed_to="local", confidence=0.0, cost_usd=0.0,
                    learned=False, latency_ms=_elapsed_ms(started),
                )
            return self._escalate_to_cloud(question, context, memory_hits, started, _emit)

        messages = self._build_messages(question, context, memory_hits)
        local_resp = self._call_local(messages, _emit)
        confidence = self._compute_confidence(local_resp)

        # Refusal override: the local model can hedge or ask for clarification
        # with highly confident tokens, which fools logprob-based routing.
        # When the content looks like a voluntary surrender, escalate regardless
        # of confidence — a hedge answered by cloud is strictly better than a
        # confident hedge returned to the user.
        refused = _looks_like_refusal(local_resp.content)

        if confidence >= self.confidence_threshold and not refused:
            # Local is confident AND didn't refuse — return its answer.
            _emit({"type": "local_done", "confidence": confidence})
            latency = _elapsed_ms(started)
            cost = 0.0
            self._record_query("local", cost, confidence, latency, question=question)
            self._append_history(question, local_resp.content)
            return QueryResponse(
                answer=local_resp.content,
                routed_to="local",
                confidence=confidence,
                cost_usd=cost,
                learned=False,
                latency_ms=latency,
                gsa_p_yes=gsa_p_yes,
            )

        # ── Stage 3: Escalate to cloud ───────────────────────────
        if self._cloud_client is None:
            # No cloud model — return local answer anyway with low confidence.
            _emit({"type": "local_done", "confidence": confidence})
            latency = _elapsed_ms(started)
            self._record_query("local", 0.0, confidence, latency, question=question)
            self._append_history(question, local_resp.content)
            return QueryResponse(
                answer=local_resp.content,
                routed_to="local",
                confidence=confidence,
                cost_usd=0.0,
                learned=False,
                latency_ms=latency,
                gsa_p_yes=gsa_p_yes,
            )

        resp = self._escalate_to_cloud(
            question, context, memory_hits, started, _emit,
            escalated_on_refusal=refused,
        )
        resp.gsa_p_yes = gsa_p_yes
        return resp

    def correct(
        self,
        question: str,
        *,
        on_progress: ProgressCallback = None,
    ) -> QueryResponse:
        """User says the last answer was wrong. Re-escalate to cloud and learn.

        Invalidates any matching memory entry and forces a fresh cloud answer.
        Streams cloud tokens through ``on_progress`` (same contract as
        ``query``) so the chat REPL can show output live.
        """
        started = time.perf_counter()

        def _emit(event: dict) -> None:
            if on_progress is not None:
                on_progress(event)

        # Invalidate the closest memory entry for this question.
        if self._embed_client:
            q_emb = self._embed_client.embed(question)
            hits = self.memory.search(q_emb, limit=1, min_similarity=0.80)
            for h in hits:
                self.memory.invalidate(h.entry.id)
                logger.info("Invalidated memory entry %s for correction", h.entry.id)

        if self._cloud_client is None:
            return QueryResponse(
                answer="No cloud model configured — cannot re-verify.",
                routed_to="local", confidence=0.0, cost_usd=0.0,
                learned=False, latency_ms=_elapsed_ms(started),
            )

        return self._escalate_to_cloud(question, None, [], started, _emit)

    def savings(self) -> SavingsReport:
        """Return cumulative cost savings across all sessions (R6 AC2).

        Reads from the query_log table for totals that survive across sessions,
        and counts facts_learned from knowledge_entries.
        """
        row = self._conn.execute(
            "SELECT "
            "  COUNT(*) AS total, "
            "  SUM(CASE WHEN routing_decision = 'local' THEN 1 ELSE 0 END) AS local_n, "
            "  SUM(CASE WHEN routing_decision = 'cloud' THEN 1 ELSE 0 END) AS cloud_n, "
            "  SUM(CASE WHEN routing_decision = 'memory' THEN 1 ELSE 0 END) AS memory_n, "
            "  COALESCE(SUM(cost), 0.0) AS total_cost "
            "FROM query_log"
        ).fetchone()

        total = row["total"]
        total_cost = row["total_cost"]
        # Estimate: every query would cost at least $0.003 if sent to cloud.
        all_cloud_est = total * 0.003
        # For cloud queries, use actual cost if it's higher than the minimum.
        cloud_actual_row = self._conn.execute(
            "SELECT COALESCE(SUM(CASE WHEN cost > 0.003 THEN cost ELSE 0.003 END), 0.0) AS est "
            "FROM query_log"
        ).fetchone()
        all_cloud_est = cloud_actual_row["est"] if total > 0 else 0.0

        saved = all_cloud_est - total_cost
        saved_pct = (saved / all_cloud_est * 100) if all_cloud_est > 0 else 0.0

        # Count facts learned from knowledge store.
        facts_row = self._conn.execute(
            "SELECT COUNT(*) AS n FROM knowledge_entries WHERE source = 'cloud_escalation'"
        ).fetchone()
        facts_learned = facts_row["n"]

        return SavingsReport(
            total_queries=total,
            local_queries=row["local_n"],
            cloud_queries=row["cloud_n"],
            memory_queries=row["memory_n"],
            total_cost_usd=total_cost,
            estimated_all_cloud_cost_usd=all_cloud_est,
            saved_usd=saved,
            saved_pct=saved_pct,
            facts_learned=facts_learned,
        )

    # ── Internal ──────────────────────────────────────────────────

    def _call_local(
        self,
        messages: list[ChatMessage],
        emit: Callable[[dict], None],
    ) -> ChatResponseWithLogprobs:
        """Call the local model. Streams when on Ollama; falls back otherwise.

        For Ollama we use ``chat_stream_ollama`` and forward each chunk through
        the agent's progress callback as ``token`` events tagged with
        ``source='local'`` and phase=='content' or 'thinking'. The CLI
        renderer can then show tokens as they arrive, masking generation
        latency.

        For non-Ollama providers (OpenAI-compatible, Bedrock) we fall back to
        the non-streaming ``chat_with_logprobs`` path. Streaming for those
        providers is a future enhancement on the local-model side.

        Tolerates clients that don't expose ``config.provider`` (notably mock
        clients in tests) — falls back to the non-streaming path in that case.
        """
        assert self._local_client is not None
        config = getattr(self._local_client, "config", None)
        provider = getattr(config, "provider", None) if config is not None else None

        if provider == "ollama":
            def _on_chunk(chunk: dict) -> None:
                emit({"type": "token", "source": "local", **chunk})

            return self._local_client.chat_stream_ollama(
                messages,
                on_token=_on_chunk,
                max_tokens=1024,
                temperature=0.0,
                top_logprobs=1,
            )

        # Non-Ollama or test mock: no streaming, use the normal path.
        return self._local_client.chat_with_logprobs(
            messages, max_tokens=1024, temperature=0.0, top_logprobs=1,
        )

    def _call_cloud(
        self,
        messages: list[ChatMessage],
        emit: Callable[[dict], None],
    ) -> ChatResponse:
        """Call the cloud model with streaming when supported.

        Forwards each chunk through ``emit`` as a ``token`` event tagged with
        ``source='cloud'``. Falls back to non-streaming for clients without a
        recognised provider (notably MagicMock fixtures in tests that don't
        configure ``config.provider``).
        """
        assert self._cloud_client is not None
        config = getattr(self._cloud_client, "config", None)
        provider = getattr(config, "provider", None) if config is not None else None

        if provider in ("ollama", "openai", "bedrock"):
            def _on_chunk(chunk: dict) -> None:
                emit({"type": "token", "source": "cloud", **chunk})
            return self._cloud_client.chat_stream(
                messages,
                on_token=_on_chunk,
                max_tokens=1024,
                temperature=0.0,
            )

        # Test fallback or unknown provider: no streaming.
        return self._cloud_client.chat(messages, max_tokens=1024, temperature=0.0)

    def _check_memory(self, question: str) -> list[ScoredKnowledgeEntry]:
        """Search the knowledge store for similar past Q&A."""
        if self._embed_client is None:
            return []
        try:
            q_emb = self._embed_client.embed(question)
            return self.memory.search(q_emb, limit=5, min_similarity=0.0)
        except Exception as e:
            logger.warning("Memory search failed: %s", e)
            return []

    def _escalate_to_cloud(
        self,
        question: str,
        context: Optional[str],
        memory_hits: list[ScoredKnowledgeEntry],
        started: float,
        emit: Callable[[dict], None] = lambda e: None,
        *,
        escalated_on_refusal: bool = False,
    ) -> QueryResponse:
        """Send to cloud, learn from the answer."""
        assert self._cloud_client is not None

        emit({"type": "cloud_call", "model": self._cloud_model_name or "unknown"})

        messages = self._build_messages(question, context, memory_hits)
        cloud_resp = self._call_cloud(messages, emit)
        cost = self._estimate_cost(cloud_resp.input_tokens, cloud_resp.output_tokens)

        emit({
            "type": "cloud_done",
            "model": self._cloud_model_name or "unknown",
            "cost": cost,
            "latency_ms": cloud_resp.latency_ms,
        })

        # Learn from escalation.
        learned, knowledge_count = self._learn(question, cloud_resp.content)

        if learned:
            emit({"type": "learning", "knowledge_count": knowledge_count})

        latency = _elapsed_ms(started)
        self._record_query("cloud", cost, 0.0, latency, learned=learned, question=question)
        self._append_history(question, cloud_resp.content)
        return QueryResponse(
            answer=cloud_resp.content,
            routed_to="cloud",
            confidence=0.0,  # we escalated because confidence was low
            cost_usd=cost,
            learned=learned,
            latency_ms=latency,
            escalated_on_refusal=escalated_on_refusal,
        )

    def _learn(self, question: str, answer: str) -> tuple[bool, int]:
        """Store knowledge from a cloud escalation. Returns (learned, count).

        Uses the LearningExtractor to extract structured knowledge entries
        from the cloud response. Falls back to storing the raw Q&A pair.
        """
        if self._embed_client is None:
            return False, 0
        try:
            # Extract structured knowledge via local LLM (if available).
            extractor_client = self._local_client or self._cloud_client
            if extractor_client:
                extractor = LearningExtractor(extractor_client)
                extraction = extractor.extract(question, answer)
            else:
                # No LLM for extraction — raw fallback.
                extraction = ExtractionResult(
                    knowledge=[NewKnowledgeEntry(
                        content=answer[:500],
                        source="cloud_escalation",
                        confidence=0.9,
                        domain="general",
                        topic="learned",
                        metadata={"extracted_from": question[:200]},
                    )],
                    skills=[],
                )

            # Deduplication: check if a very similar question already exists.
            q_emb = self._embed_client.embed(question)
            existing = self.memory.search(q_emb, limit=1, min_similarity=0.95)
            if existing:
                old = existing[0].entry
                self.memory.invalidate(old.id)
                logger.debug("Deduplicated: replacing entry %s with updated answer", old.id)

            # Store each extracted knowledge entry.
            stored_count = 0
            for entry in extraction.knowledge:
                try:
                    content_emb = self._embed_client.embed(entry.content)
                    # Use the question embedding for the question field,
                    # and the content embedding for the answer embedding.
                    entry.question = question
                    entry.embedding = q_emb.tolist()
                    entry.answer_embedding = content_emb.tolist()
                    entry.verbatim_response = answer if stored_count == 0 else None
                    self.memory.insert(entry)
                    stored_count += 1
                except Exception as e:
                    logger.warning("Failed to store extracted entry: %s", e)

            if stored_count > 0:
                self._session_stats.facts_learned += stored_count
                return True, stored_count

            return False, 0
        except Exception as e:
            logger.warning("Failed to learn from escalation: %s", e)
            return False, 0

    def _build_messages(
        self,
        question: str,
        context: Optional[str],
        memory_hits: list[ScoredKnowledgeEntry],
    ) -> list[ChatMessage]:
        """Build the prompt with all available context."""
        parts: list[str] = []

        # System message.
        parts.append("You are a helpful assistant. Answer the user's question accurately and concisely.")

        # Memory context (from agent's learned knowledge).
        memory_context = self._format_memory_context(memory_hits)
        if memory_context:
            parts.append(f"\n{memory_context}")

        # Document context (from user's ingested source materials — R9 AC8).
        # Framed distinctly from memory so the model knows "source material"
        # vs "something you answered before".
        document_context = self._format_document_context(question)
        if document_context:
            parts.append(f"\n{document_context}")

        # External context (from user's RAG pipeline or caller-supplied).
        if context:
            parts.append(f"\nRelevant context:\n{context}")

        system = "\n".join(parts)
        messages = [ChatMessage(role="system", content=system)]

        # Conversation history.
        for turn in self._history[-10:]:  # last 10 turns
            messages.append(ChatMessage(role=turn["role"], content=turn["content"]))

        # Current question.
        messages.append(ChatMessage(role="user", content=question))
        return messages

    def _format_memory_context(self, hits: list[ScoredKnowledgeEntry]) -> str:
        """Format memory hits into context for the prompt."""
        relevant = [h for h in hits if h.score >= MEMORY_CONTEXT_THRESHOLD]
        if not relevant:
            return ""
        lines = ["Here is what you recall from past interactions:"]
        for i, h in enumerate(relevant[:3], 1):
            q = h.entry.question or "unknown question"
            a = (h.entry.content or "")[:500]
            lines.append(f"{i}. (Previously asked: {q.strip()[:120]})\n   {a.strip()}")
        return "\n".join(lines)

    def _format_document_context(self, question: str) -> str:
        """Retrieve and format document chunks relevant to the question (R9 AC8).

        Returns empty string if no document store is attached, the store is
        empty, or no chunk is above the relevance threshold.
        """
        store = getattr(self, "documents", None)
        if store is None:
            return ""
        try:
            hits = store.search(question, limit=3)
        except Exception as e:
            logger.warning("Document retrieval failed: %s", e)
            return ""
        # Threshold is intentionally lower than memory's MEMORY_CONTEXT_THRESHOLD —
        # docs are reference material, we inject them more liberally.
        relevant = [h for h in hits if h.score >= 0.30]
        if not relevant:
            return ""
        lines = ["Here is relevant information from your documents:"]
        for i, h in enumerate(relevant, 1):
            content = (h.content or "")[:500].strip()
            source = Path(h.source_file).name if h.source_file else "document"
            lines.append(f"{i}. (from {source})\n   {content}")
        return "\n".join(lines)

    def _compute_confidence(self, resp: ChatResponseWithLogprobs) -> float:
        """Compute logprob_uncertainty from a local model response."""
        avg_lp = resp.avg_logprob
        if avg_lp is None:
            return 0.5  # neutral if no logprobs available
        # Sigmoid mapping: same as confidence_evaluator.compute_logprob_uncertainty
        x = avg_lp * 2.0 + 3.0
        return float(1.0 / (1.0 + math.exp(-x)))

    def _estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Rough cost estimate for a cloud call."""
        model = self._cloud_model_name or ""
        # Try to match against known rates.
        for key, rates in _DEFAULT_COST_RATES.items():
            if key in model.lower():
                return (input_tokens * rates["input"] + output_tokens * rates["output"]) / 1_000_000
        # Default: assume $3/M input, $15/M output (Sonnet-class).
        return (input_tokens * 3.0 + output_tokens * 15.0) / 1_000_000

    def _entry_age_days(self, entry) -> float:
        """How many days old is this knowledge entry?"""
        try:
            created = datetime.fromisoformat(entry.created_at)
            if created.tzinfo is None:
                created = created.replace(tzinfo=timezone.utc)
            now = datetime.now(timezone.utc)
            return (now - created).total_seconds() / 86400.0
        except Exception:
            return 0.0

    def _record_query(
        self, routed_to: str, cost: float, confidence: float, latency_ms: int,
        learned: bool = False, question: str = "",
    ) -> None:
        """Update session stats and persist to query_log table."""
        import uuid

        s = self._session_stats
        s.total_queries += 1
        s.total_cost_usd += cost
        # Estimate what this query would have cost if sent to cloud.
        s.estimated_all_cloud_cost_usd += max(cost, 0.003)  # minimum $0.003 per query
        if routed_to == "local":
            s.local_queries += 1
        elif routed_to == "cloud":
            s.cloud_queries += 1
        elif routed_to == "memory":
            s.memory_queries += 1

        # Persist to query_log (R6 AC1, AC4).
        try:
            now = datetime.now(timezone.utc).isoformat()
            self._conn.execute(
                "INSERT INTO query_log "
                "(id, query_text, routing_decision, signals, fusion_weights, "
                "fused_score, cost, latency_ms, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    str(uuid.uuid4()),
                    question,
                    routed_to,
                    "{}",   # signals — not used in product v1
                    "{}",   # fusion_weights — not used in product v1
                    confidence,
                    cost,
                    latency_ms,
                    now,
                ),
            )
            self._conn.commit()
        except Exception as e:
            logger.warning("Failed to persist query log: %s", e)

    def _append_history(self, question: str, answer: str) -> None:
        """Add a turn to conversation history."""
        self._history.append({"role": "user", "content": question})
        self._history.append({"role": "assistant", "content": answer})


# ── Helpers ────────────────────────────────────────────────────────

def _parse_model_string(model_str: str, default_provider: str = "ollama") -> tuple[str, str]:
    """Parse 'provider/model' into (provider, model). If no slash, use default_provider."""
    if "/" in model_str:
        parts = model_str.split("/", 1)
        provider = parts[0].lower()
        model = parts[1]
        # Map common provider names.
        if provider in ("ollama", "openai", "bedrock"):
            return provider, model
        # Treat unknown prefixes as part of the model name (e.g., "qllama/bge-large").
        return default_provider, model_str
    return default_provider, model_str


def _apply_bedrock_auth(config_kwargs: dict, bedrock_cfg: dict) -> None:
    """Translate a Bedrock config dict from YAML into LLMConfig kwargs.

    Input shape (as written by the setup wizard):
        {"auth_mode": "iam_user",
         "access_key_id": "...", "secret_access_key": "...",
         "session_token": "...",              # optional
         "region": "us-west-2"}
        {"auth_mode": "api_key", "api_key": "bedrock-...", "region": "..."}
        {"auth_mode": "default", "region": "..."}

    Output: kwargs that go straight into LLMConfig(...).
    """
    auth_mode = bedrock_cfg.get("auth_mode", "default")
    config_kwargs["bedrock_auth_mode"] = auth_mode
    if "region" in bedrock_cfg and bedrock_cfg["region"]:
        config_kwargs["region"] = bedrock_cfg["region"]
    if auth_mode == "iam_user":
        config_kwargs["bedrock_access_key_id"] = bedrock_cfg.get("access_key_id")
        config_kwargs["bedrock_secret_access_key"] = bedrock_cfg.get("secret_access_key")
        if bedrock_cfg.get("session_token"):
            config_kwargs["bedrock_session_token"] = bedrock_cfg["session_token"]
    elif auth_mode == "api_key":
        config_kwargs["bedrock_api_key"] = bedrock_cfg.get("api_key")


def _elapsed_ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)


# ── Refusal detector ───────────────────────────────────────────────
#
# Local models emit hedges and clarifying questions with very confident tokens,
# which tricks the logprob-based confidence score. This detector catches those
# voluntary-surrender responses so the router can override and escalate.
#
# Principle: only flag phrases that *explicitly* signal the model believes it
# can't answer. A factual statement that happens to mention "I don't know"
# in a quote is rare enough we can live with a small false-positive rate.

_REFUSAL_MARKERS = (
    # No real-time / live data
    "i don't have real-time",
    "i do not have real-time",
    "i don't have access to real-time",
    "i don't have current",
    "i can't access",
    "i cannot access",
    "i can't browse",
    "i cannot browse",
    "i'm unable to",
    "i am unable to",
    "i don't have the ability",
    "i do not have the ability",
    # Training cutoff hedges
    "as of my last update",
    "as of my knowledge cutoff",
    "my knowledge is limited to",
    "my training data",
    # Explicit "I don't know"
    "i don't know",
    "i do not know",
    "i'm not sure what",
    "i am not sure what",
    # Clarification requests (model is punting the question back)
    "did you mean",
    "are you referring to",
    "could you clarify",
    "can you clarify",
    "please clarify",
    "there might be a typo",
)


def _looks_like_refusal(text: str) -> bool:
    """Return True if the text reads like a voluntary surrender from the model.

    Catches hedges ('I don't have real-time data'), clarification requests
    ('Did you mean X?'), and explicit I-don't-knows. Case-insensitive,
    substring-based — deliberately simple; false positives are cheap (one
    extra cloud call) but false negatives ship bad answers to users.
    """
    if not text:
        return False
    lowered = text.lower()
    return any(marker in lowered for marker in _REFUSAL_MARKERS)

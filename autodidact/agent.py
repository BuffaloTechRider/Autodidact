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
MEMORY_STALENESS_DAYS = 30        # re-verify entries older than this


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
    context_sources: list[str] = field(default_factory=list)  # what context was used: "memory", "docs:file.md"
    memory_source: Optional[str] = None  # the past question it recalled, if any
    memory_age_days: Optional[float] = None  # how old the memory entry is
    memory_similarity: Optional[float] = None  # best memory hit score
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
            entry = best_hit.entry
            self.memory.access(entry.id)
            age_days = self._entry_age_days(entry)
            is_stale = age_days > self.staleness_days

            if is_stale:
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

                if self._local_client is not None:
                    # Generate a full answer using memory as context.
                    messages, ctx_sources = self._build_messages(question, context, memory_hits)
                    local_resp = self._local_client.chat(
                        messages, max_tokens=4096, temperature=0.0,
                    )
                    answer = local_resp.content

                    latency = _elapsed_ms(started)
                    self._record_query("memory", 0.0, best_hit.score, latency, question=question)
                    self._append_history(question, answer)
                    return QueryResponse(
                        answer=answer,
                        routed_to="memory",
                        confidence=best_hit.score,
                        cost_usd=0.0,
                        learned=False,
                        latency_ms=latency,
                        context_sources=ctx_sources,
                        memory_source=entry.question,
                        memory_age_days=age_days,
                        stale=False,
                    )

        # ── Stage 1.5: GSA pre-gate ──────────────────────────────
        # Ask the local model "can you answer this?" before full generation.
        # GSA sees both memory hits AND document hits so it knows what
        # context the local model will have available.
        gsa_p_yes: Optional[float] = None
        gsa_enabled = getattr(self, "gsa_enabled", True)
        gsa_threshold = getattr(self, "gsa_threshold", 0.5)
        best_similarity = best_hit.score if best_hit else 0.0
        skip_gsa = best_similarity >= MEMORY_DIRECT_THRESHOLD
        if (
            gsa_enabled
            and not skip_gsa
            and self._local_client is not None
            and self._cloud_client is not None
        ):
            # Check if documents have relevant context — if so, GSA should
            # know about it (prevents false escalation on doc-answerable queries).
            has_doc_context = False
            store = getattr(self, "documents", None)
            if store is not None:
                try:
                    doc_hits = store.search_hybrid(question, limit=2)
                    has_doc_context = bool(doc_hits and doc_hits[0].score >= 0.75)
                except Exception:
                    pass

            # Skip GSA if strong document context exists — the local model
            # will have the information it needs to answer.
            if has_doc_context:
                skip_gsa = True

            if not skip_gsa:
                try:
                    if getattr(self, "_gsa", None) is None:
                        self._gsa = SelfAssessment(self._local_client)
                    gsa_result = self._gsa.compute(question, retrieved_hits=memory_hits)
                    gsa_p_yes = gsa_result.p_yes
                except Exception as e:
                    logger.warning("GSA probe failed, skipping gate: %s", e)
                    gsa_p_yes = None

            if gsa_p_yes is not None and gsa_p_yes < gsa_threshold:
                # Model self-reports it can't answer — skip local entirely.
                # Note: errors from _escalate_to_cloud propagate (no try/except
                # wrapping it) so the user sees real problems like "Bedrock
                # rejected request: ValidationException".
                resp = self._escalate_to_cloud(
                    question, context, memory_hits, started, _emit,
                )
                resp.escalated_on_gsa = True
                resp.gsa_p_yes = gsa_p_yes
                return resp

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

        messages, ctx_sources = self._build_messages(question, context, memory_hits)
        local_resp = self._call_local(messages, _emit)

        refused = _looks_like_refusal(local_resp.content)

        if not refused:
            confidence = 1.0
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
                context_sources=ctx_sources,
                memory_similarity=best_hit.score if best_hit else None,
                gsa_p_yes=gsa_p_yes,
            )

        # ── Stage 3: Escalate to cloud ───────────────────────────
        if self._cloud_client is None:
            # No cloud model — return local answer anyway with neutral confidence.
            _emit({"type": "local_done", "confidence": 0.5})
            latency = _elapsed_ms(started)
            self._record_query("local", 0.0, 0.5, latency, question=question)
            self._append_history(question, local_resp.content)
            return QueryResponse(
                answer=local_resp.content,
                routed_to="local",
                confidence=0.5,
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
    ) -> ChatResponse:
        """Call the local model. Streams when on Ollama; falls back otherwise.

        Notably DOES NOT request logprobs. A benchmark (May 2026) showed
        Ollama adds ~150ms per call when ``logprobs=True``, even at
        ``top_logprobs=1``. Since the agent's post-local routing no longer
        consults logprobs (GSA + refusal detector handle that role), paying
        the overhead is wasted.

        For Ollama we use ``chat_stream_ollama_no_logprobs`` and forward
        each chunk through the agent's progress callback as ``token``
        events tagged with ``source='local'`` and ``phase='content'`` or
        ``'thinking'``.

        For non-Ollama providers and test mocks (no recognised
        ``config.provider``), we fall back to plain ``chat()`` —
        also no logprobs.
        """
        assert self._local_client is not None
        config = getattr(self._local_client, "config", None)
        provider = getattr(config, "provider", None) if config is not None else None

        if provider == "ollama":
            def _on_chunk(chunk: dict) -> None:
                emit({"type": "token", "source": "local", **chunk})

            return self._local_client.chat_stream_ollama_no_logprobs(
                messages,
                on_token=_on_chunk,
                max_tokens=4096,
                temperature=0.0,
            )

        # Non-Ollama or test mock: no streaming, no logprobs.
        return self._local_client.chat(
            messages, max_tokens=4096, temperature=0.0,
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
                max_tokens=4096,
                temperature=0.0,
            )

        # Test fallback or unknown provider: no streaming.
        return self._cloud_client.chat(messages, max_tokens=4096, temperature=0.0)

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

        messages, _ = self._build_messages(question, context, memory_hits)
        cloud_resp = self._call_cloud(messages, emit)
        cost = self._estimate_cost(cloud_resp.input_tokens, cloud_resp.output_tokens)

        emit({
            "type": "cloud_done",
            "model": self._cloud_model_name or "unknown",
            "cost": cost,
            "latency_ms": cloud_resp.latency_ms,
        })

        # Learn from escalation in background — don't block the user.
        import threading
        t = threading.Thread(
            target=self._learn,
            args=(question, cloud_resp.content),
            daemon=True,
        )
        t.start()
        self._last_learn_thread = t

        latency = _elapsed_ms(started)
        self._record_query("cloud", cost, 0.0, latency, learned=True, question=question)
        self._append_history(question, cloud_resp.content)
        return QueryResponse(
            answer=cloud_resp.content,
            routed_to="cloud",
            confidence=0.0,
            cost_usd=cost,
            learned=True,
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
    ) -> tuple[list[ChatMessage], list[str]]:
        """Build the prompt with all available context.

        Returns (messages, context_sources) where context_sources lists
        what was injected: "memory:N facts", "docs:filename.md", etc.
        """
        parts: list[str] = []
        sources: list[str] = []

        # System message.
        parts.append(
            "You are a helpful assistant. Answer the user's question accurately and concisely.\n"
            "IMPORTANT: Only state what you know from the context provided below or your training. "
            "If you reference code, quote it exactly as shown — never fabricate or paraphrase code snippets. "
            "If the context doesn't contain enough information to fully answer, say so rather than making things up."
        )

        # Memory context (from agent's learned knowledge).
        memory_context = self._format_memory_context(memory_hits)
        if memory_context:
            parts.append(f"\n{memory_context}")
            relevant = [h for h in memory_hits if h.score >= MEMORY_CONTEXT_THRESHOLD]
            sources.append(f"memory ({len(relevant[:3])} facts)")

        # Document context (from user's ingested source materials — R9 AC8).
        doc_sources = self._format_document_context_with_sources(question)
        if doc_sources:
            doc_context, doc_files = doc_sources
            parts.append(f"\n{doc_context}")
            for f in doc_files:
                sources.append(f"docs:{f}")

        # External context (from user's RAG pipeline or caller-supplied).
        if context:
            parts.append(f"\nRelevant context:\n{context}")
            sources.append("external")

        system = "\n".join(parts)
        messages = [ChatMessage(role="system", content=system)]

        # Conversation history.
        for turn in self._history[-10:]:  # last 10 turns
            messages.append(ChatMessage(role=turn["role"], content=turn["content"]))

        # Current question.
        messages.append(ChatMessage(role="user", content=question))
        return messages, sources

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

    def _format_document_context_with_sources(self, question: str) -> Optional[tuple[str, list[str]]]:
        """Retrieve and format document chunks relevant to the question.

        Returns (context_string, [filenames]) or None if nothing found.
        """
        store = getattr(self, "documents", None)
        if store is None:
            return None
        try:
            hits = store.search_hybrid(question, limit=3)
        except Exception as e:
            logger.warning("Document retrieval failed: %s", e)
            return None
        relevant = [h for h in hits if h.score >= 0.30]
        if not relevant:
            return None
        lines = ["Here is relevant information from your documents:"]
        files: list[str] = []
        for i, h in enumerate(relevant, 1):
            content = (h.content or "")[:500].strip()
            source = Path(h.source_file).name if h.source_file else "document"
            lines.append(f"{i}. (from {source})\n   {content}")
            if source not in files:
                files.append(source)
        return "\n".join(lines), files

    def _compute_confidence(self, resp: ChatResponseWithLogprobs) -> float:
        """Compute logprob_uncertainty from a local model response.

        On thinking-model responses (qwen3 with reasoning, DeepSeek-R1, etc.)
        the avg_logprob is over BOTH thinking and content tokens. Thinking
        tokens are inherently noisy ("explore option A... or maybe B...")
        which drags the average below threshold even on correct answers.

        For thinking responses we return max-confidence (1.0) so the
        post-local gate doesn't penalize them. The refusal detector and the
        pre-local GSA gate still fire — those are the right tools for
        thinking models.
        """
        if resp.had_thinking:
            return 1.0

        avg_lp = resp.avg_logprob
        if avg_lp is None:
            return 0.5  # neutral if no logprobs available
        # Sigmoid mapping: x = avg_logprob * scale + shift
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

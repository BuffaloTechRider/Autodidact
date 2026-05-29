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
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional, TYPE_CHECKING

import numpy as np

from autodidact.database import init_database
from autodidact.document_store import DocumentStore, ScoredChunk
from autodidact.knowledge_store import KnowledgeStore, ScoredKnowledgeEntry
from autodidact.learning_extractor import ExtractionResult, LearningExtractor
from autodidact.llm_client import ChatMessage, ChatResponseWithLogprobs, LLMClient, LLMConfig
from autodidact.signals.grounded_self_assessment import SelfAssessment
from autodidact.types import AutodidactConfig, NewKnowledgeEntry
from autodidact.routing import RoutingState, run_pipeline
from autodidact.routing.stages import (
    CloudEscalationDeps,
    CloudEscalationStage,
    CorrectionInvalidationDeps,
    CorrectionInvalidationStage,
    GsaPreGateDeps,
    GsaPreGateStage,
    LocalGenerationDeps,
    LocalGenerationStage,
    MemoryStage,
    MemoryStageDeps,
)

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
MEMORY_DIRECT_THRESHOLD = 0.80   # return stored answer directly
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
        gsa_threshold: float = 0.55,
    ) -> None:
        """Construct an Agent from kwargs (legacy / programmatic surface).

        For YAML-based construction prefer ``AgentConfig.from_yaml(path).build_agent()``,
        which validates the config strictly. This kwarg form is the
        programmatic shim — it builds the same internal state but doesn't
        enforce strict validation, since trusted callers (tests, libraries)
        own their own correctness.

        Empty kwargs (no models) ⇒ a no-op Agent (memory + DB only). Useful
        for tests that wire models in afterwards via private attributes.
        """
        # Always init the shared state first so a no-op Agent is still
        # usable for tests that bypass model config.
        self._init_state(
            confidence_threshold=confidence_threshold,
            staleness_days=staleness_days,
            gsa_enabled=gsa_enabled,
            gsa_threshold=gsa_threshold,
            db_path=db_path,
        )

        # Build the local LLMClient, if any.
        if local_model:
            provider, model = _parse_model_string(local_model, default_provider="ollama")
            self._local_client = LLMClient(_kwargs_to_llmconfig(
                provider=provider,
                model=model,
                embedding_model=embedding_model,
                base_url=local_base_url,
                api_key_env=local_api_key_env,
                region=local_region,
                bedrock=local_bedrock,
                is_local_slot=True,
            ))
            self._local_model_name = local_model

        # Build the cloud LLMClient, if any.
        if cloud_model:
            provider, model = _parse_model_string(cloud_model, default_provider=cloud_provider)
            self._cloud_client = LLMClient(_kwargs_to_llmconfig(
                provider=provider,
                model=model,
                embedding_model=None,  # cloud slot doesn't own embeddings
                base_url=cloud_base_url,
                api_key_env=cloud_api_key_env,
                region=cloud_region,
                bedrock=cloud_bedrock,
                is_local_slot=False,
            ))
            self._cloud_model_name = cloud_model

        self._embed_client = self._local_client or self._cloud_client
        self._build_default_stages()

    @classmethod
    def _from_config(cls, cfg: "AgentConfig") -> "Agent":
        """Construct an Agent from a strict AgentConfig.

        Used by ``AgentConfig.build_agent()``. The kwarg ``__init__`` above
        is the loose programmatic shim. Both end up with the same internal
        state via ``_init_state`` and shared LLMConfig translation helpers.
        """
        from autodidact.config import apply_bedrock_auth_to_llm_kwargs

        agent = cls.__new__(cls)
        agent._init_state(
            confidence_threshold=cfg.routing.confidence_threshold,
            staleness_days=cfg.routing.staleness_days,
            gsa_enabled=cfg.gsa.enabled,
            gsa_threshold=cfg.gsa.threshold,
            db_path=cfg.memory.path,
        )

        # Local slot.
        local = cfg.local
        provider = local.provider or "ollama"
        local_kwargs: dict = {
            "provider": "openai" if provider in _OPENAI_COMPAT_PROVIDERS else provider,
            "model": local.model,
            "embedding_model": _normalize_embedding_model(local.embedding_model),
        }
        if provider in _OPENAI_COMPAT_PROVIDERS:
            local_kwargs["base_url"] = local.base_url or "https://api.openai.com/v1"
            local_kwargs["api_key_env"] = local.api_key_env or "OPENAI_API_KEY"
            if local.api_key and local_kwargs["api_key_env"]:
                os.environ.setdefault(local_kwargs["api_key_env"], local.api_key)
        elif provider == "bedrock" and local.bedrock is not None:
            apply_bedrock_auth_to_llm_kwargs(local_kwargs, local.bedrock)
        agent._local_client = LLMClient(LLMConfig(**local_kwargs))
        agent._local_model_name = (
            f"{provider}/{local.model}" if provider != "ollama" else local.model
        )

        # Cloud slot.
        if cfg.cloud is not None:
            cloud = cfg.cloud
            cloud_kwargs: dict = {
                "provider": "openai" if cloud.provider in _OPENAI_COMPAT_PROVIDERS else cloud.provider,
                "model": cloud.model,
            }
            if cloud.provider in _OPENAI_COMPAT_PROVIDERS:
                cloud_kwargs["base_url"] = cloud.base_url or "https://api.openai.com/v1"
                cloud_kwargs["api_key_env"] = cloud.api_key_env or "OPENAI_API_KEY"
                if cloud.api_key and cloud_kwargs["api_key_env"]:
                    os.environ.setdefault(cloud_kwargs["api_key_env"], cloud.api_key)
            elif cloud.provider == "bedrock" and cloud.bedrock is not None:
                apply_bedrock_auth_to_llm_kwargs(cloud_kwargs, cloud.bedrock)
            agent._cloud_client = LLMClient(LLMConfig(**cloud_kwargs))
            agent._cloud_model_name = f"{cloud.provider}/{cloud.model}"

        agent._embed_client = agent._local_client or agent._cloud_client
        agent._build_default_stages()
        return agent

    def _init_state(
        self,
        *,
        confidence_threshold: float,
        staleness_days: float,
        gsa_enabled: bool,
        gsa_threshold: float,
        db_path: str,
    ) -> None:
        """Common state initialization shared by __init__ and _from_config."""
        self.confidence_threshold = confidence_threshold
        self.staleness_days = staleness_days
        self.gsa_enabled = gsa_enabled
        self.gsa_threshold = gsa_threshold

        self._db_path = str(Path(db_path).expanduser())
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = init_database(self._db_path)
        self._config = AutodidactConfig(db_path=self._db_path)
        self.memory = KnowledgeStore(self._conn, self._config)

        self._local_client: Optional[LLMClient] = None
        self._cloud_client: Optional[LLMClient] = None
        self._embed_client: Optional[LLMClient] = None
        self._local_model_name: Optional[str] = None
        self._cloud_model_name: Optional[str] = None
        self._session_stats = SavingsReport()
        self._history: list[dict] = []
        self.documents: Optional[DocumentStore] = None
        self._gsa: Optional[SelfAssessment] = None
        self._query_stages: Optional[list] = None
        self._correct_stages: Optional[list] = None

    # ── Public API ────────────────────────────────────────────────

    def attach_document_store(self, store: DocumentStore) -> None:
        """Wire an existing DocumentStore into this agent.

        Document chunks will be retrieved alongside agent memory at query
        time and injected into the prompt with distinct framing ('from your
        documents' vs 'from past interactions').
        """
        self.documents = store

    # ── Routing pipeline construction ─────────────────────────────

    def _build_default_stages(self) -> None:
        """Wire the default query and correct pipelines.

        Called from __init__. Tests that bypass __init__ can either
        - call this method themselves, or
        - set _query_stages / _correct_stages directly with stubs.
        """
        memory_deps = MemoryStageDeps(
            check_memory_fn=self._check_memory,
            knowledge_store_access=self.memory.access,
            entry_age_fn=self._entry_age_days,
            staleness_days=self.staleness_days,
            has_local_client=self._local_client is not None,
            build_messages_fn=self._build_messages,
            call_local_fn=self._call_local,
            record_query_fn=self._record_query,
            append_history_fn=self._append_history,
        )

        gsa_deps = GsaPreGateDeps(
            gsa_enabled=getattr(self, "gsa_enabled", True),
            gsa_threshold=getattr(self, "gsa_threshold", 0.55),
            has_local_client=self._local_client is not None,
            has_cloud_client=self._cloud_client is not None,
            document_search_fn=self._gsa_doc_search,
            build_gsa_fn=self._build_gsa_probe,
            escalate_fn=self._escalate_for_pipeline,
        )

        local_deps = LocalGenerationDeps(
            has_local_client=self._local_client is not None,
            has_cloud_client=self._cloud_client is not None,
            build_messages_fn=self._build_messages,
            call_local_fn=self._call_local,
            refusal_detector=_looks_like_refusal,
            record_query_fn=self._record_query,
            append_history_fn=self._append_history,
        )

        cloud_deps = CloudEscalationDeps(
            has_cloud_client=self._cloud_client is not None,
            escalate_fn=self._escalate_for_pipeline,
            record_query_fn=self._record_query,
            append_history_fn=self._append_history,
        )

        correction_deps = CorrectionInvalidationDeps(
            has_embed_client=self._embed_client is not None,
            embed_fn=(self._embed_client.embed if self._embed_client else (lambda t: None)),
            memory_search_fn=self.memory.search,
            memory_invalidate_fn=self.memory.invalidate,
        )

        self._query_stages = [
            MemoryStage(memory_deps),
            GsaPreGateStage(gsa_deps),
            LocalGenerationStage(local_deps),
            CloudEscalationStage(cloud_deps, force=False),
        ]
        self._correct_stages = [
            CorrectionInvalidationStage(correction_deps),
            CloudEscalationStage(cloud_deps, force=True),
        ]

    def _gsa_doc_search(self, question: str, q_emb):
        """Adapter for GsaPreGateStage's document_search_fn.

        Returns [] when no document store is attached so the stage's
        threshold check (>= 0.75) is a no-op.
        """
        store = getattr(self, "documents", None)
        if store is None:
            return []
        return store.search(question, limit=1, query_embedding=q_emb)

    def _build_gsa_probe(self):
        """Adapter for GsaPreGateStage's build_gsa_fn — lazy, patchable.

        Tests preset agent._gsa (e.g. with MagicMock) to control the probe.
        Production constructs SelfAssessment(self._local_client) on first use.
        """
        if getattr(self, "_gsa", None) is None:
            self._gsa = SelfAssessment(self._local_client)
        return self._gsa

    def _escalate_for_pipeline(self, state: RoutingState):
        """Adapter for stages' escalate_fn. Forwards to _escalate_to_cloud
        with the right escalated_on_refusal flag from RoutingState."""
        return self._escalate_to_cloud(
            state.question, state.context, state.memory_hits,
            state.started, state.emit,
            escalated_on_refusal=state.refused,
        )

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
            - gsa_check: GSA pre-gate probe is running
            - local_done: local model answered, includes confidence
            - cloud_call: escalating to cloud, includes model name
            - cloud_done: cloud response received, includes cost and model
            - token: streaming token; includes phase, source, text
        """
        started = time.perf_counter()

        def _emit(event: dict) -> None:
            if on_progress is not None:
                on_progress(event)

        if getattr(self, "_query_stages", None) is None:
            self._build_default_stages()

        state = RoutingState(
            question=question,
            context=context,
            started=started,
            emit=_emit,
        )
        return run_pipeline(self._query_stages, state)

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

        if getattr(self, "_correct_stages", None) is None:
            self._build_default_stages()

        state = RoutingState(
            question=question,
            context=None,
            started=started,
            emit=_emit,
        )
        return run_pipeline(self._correct_stages, state)

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
        # Estimate what all queries would have cost if sent to cloud.
        # Use the max actual cloud cost as the per-query estimate for local/memory queries.
        max_cloud_row = self._conn.execute(
            "SELECT COALESCE(MAX(cost), 0.015) AS max_cost FROM query_log WHERE cost > 0"
        ).fetchone()
        max_cloud_cost = max_cloud_row["max_cost"] if max_cloud_row["max_cost"] else 0.015
        cloud_actual_row = self._conn.execute(
            "SELECT COALESCE(SUM(CASE WHEN cost > 0 THEN cost ELSE ? END), 0.0) AS est "
            "FROM query_log",
            (max_cloud_cost,),
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
            )

        # Test fallback or unknown provider: no streaming.
        return self._cloud_client.chat(messages, max_tokens=4096)

    def _check_memory(self, question: str) -> tuple[list[ScoredKnowledgeEntry], Optional[np.ndarray]]:
        """Search the knowledge store for similar past Q&A.

        Returns (hits, query_embedding) so callers can reuse the embedding.
        """
        if self._embed_client is None:
            return [], None
        try:
            q_emb = self._embed_client.embed(question)
            hits = self.memory.search(q_emb, limit=5, min_similarity=0.0)
            return hits, q_emb
        except Exception as e:
            logger.warning("Memory search failed: %s", e)
            return [], None

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
        # Skip learning if cloud gave a non-answer.
        will_learn = not _cloud_response_is_non_answer(cloud_resp.content)
        if will_learn:
            import threading
            t = threading.Thread(
                target=self._learn,
                args=(question, cloud_resp.content),
                daemon=True,
            )
            t.start()
            self._last_learn_thread = t

        latency = _elapsed_ms(started)
        self._record_query("cloud", cost, 0.0, latency, learned=will_learn, question=question)
        self._append_history(question, cloud_resp.content)
        return QueryResponse(
            answer=cloud_resp.content,
            routed_to="cloud",
            confidence=0.0,
            cost_usd=cost,
            learned=will_learn,
            latency_ms=latency,
            escalated_on_refusal=escalated_on_refusal,
        )

    def _learn(self, question: str, answer: str) -> tuple[bool, int]:
        """Store knowledge from a cloud escalation. Returns (learned, count).

        Uses the LearningExtractor to extract structured knowledge entries
        from the cloud response. Falls back to storing the raw Q&A pair.
        Skips learning if the cloud response is a non-answer (refusal/hedging).
        """
        if self._embed_client is None:
            return False, 0
        if _cloud_response_is_non_answer(answer):
            logger.info("Cloud response is a non-answer; skipping learning.")
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
            "Use the context below and your training to answer confidently. "
            "The user will lose trust if you fabricate facts or code — DO NOT make up "
            "facts, code, or features that don't exist. Quote code exactly as shown. "
            "Do NOT editorialize about what you can or cannot see in the context."
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
            hits = store.search_hybrid(question, limit=5)
        except Exception as e:
            logger.warning("Document retrieval failed: %s", e)
            return None
        relevant = [h for h in hits if h.score >= 0.50]
        if not relevant:
            return None
        lines = ["Here is relevant information from your documents:"]
        files: list[str] = []
        for i, h in enumerate(relevant, 1):
            content = (h.content or "")[:1500].strip()
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
        # Use the max cloud cost seen so far (local/memory queries are typically
        # similar complexity to the cloud calls that taught the system).
        if cost > 0:
            s.estimated_all_cloud_cost_usd += cost
            s._max_cloud_cost = max(getattr(s, "_max_cloud_cost", 0.0), cost)
        else:
            cloud_est = getattr(s, "_max_cloud_cost", 0.0) or 0.015
            s.estimated_all_cloud_cost_usd += cloud_est
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

_OPENAI_COMPAT_PROVIDERS = frozenset({
    "openai", "google", "openrouter", "deepseek", "mistral",
    "groq", "together", "fireworks", "xai",
})


def _normalize_embedding_model(name: Optional[str]) -> Optional[str]:
    """Strip a known provider prefix from an embedding model name.

    Ollama models commonly live under third-party namespaces like
    ``qllama/bge-large-en-v1.5`` or ``hf.co/bartowski/...`` where the
    slash is part of the model's identity. Strip ONLY known provider
    prefixes (``ollama/``, ``openai/``, ``bedrock/``), not arbitrary
    namespaces.

    Returns the default ``qllama/bge-large-en-v1.5`` when name is None.
    """
    name = name or "qllama/bge-large-en-v1.5"
    if "/" in name:
        first, rest = name.split("/", 1)
        if first.lower() in ("ollama", "openai", "bedrock"):
            return rest
    return name


def _kwargs_to_llmconfig(
    *,
    provider: str,
    model: str,
    embedding_model: Optional[str],
    base_url: Optional[str],
    api_key_env: Optional[str],
    region: str,
    bedrock: Optional[dict],
    is_local_slot: bool,
) -> LLMConfig:
    """Translate kwarg-style construction into an LLMConfig.

    Used by ``Agent.__init__`` (the loose programmatic shim). The strict
    YAML path uses ``Agent._from_config`` directly with an AgentConfig,
    which performs equivalent translation but with typed inputs.
    """
    kwargs: dict = {
        "provider": provider,
        "model": model,
    }
    if is_local_slot:
        kwargs["embedding_model"] = _normalize_embedding_model(embedding_model)
    if provider == "openai":
        kwargs["base_url"] = base_url or "https://api.openai.com/v1"
        kwargs["api_key_env"] = api_key_env or "OPENAI_API_KEY"
    elif provider == "bedrock":
        kwargs["region"] = region
        if bedrock:
            _apply_bedrock_auth(kwargs, bedrock)
    return LLMConfig(**kwargs)


def _parse_model_string(model_str: str, default_provider: str = "ollama") -> tuple[str, str]:
    """Parse 'provider/model' into (provider, model). If no slash, use default_provider."""
    if "/" in model_str:
        parts = model_str.split("/", 1)
        provider = parts[0].lower()
        model = parts[1]
        if provider == "ollama":
            return "ollama", model
        if provider == "bedrock":
            return "bedrock", model
        if provider in _OPENAI_COMPAT_PROVIDERS:
            return "openai", model
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
    ('Did you mean X?'), and explicit I-don't-knows. Only checks the first
    200 characters — real refusals happen upfront; mentions deeper in the
    response are the model explaining/quoting, not refusing.
    """
    if not text:
        return False
    lowered = text[:200].lower()
    return any(marker in lowered for marker in _REFUSAL_MARKERS)


_NON_ANSWER_MARKERS = (
    "i don't have reliable",
    "i don't have specific",
    "i don't have detailed",
    "i don't have information",
    "i do not have reliable",
    "i do not have specific",
    "i cannot provide accurate",
    "i can't provide accurate",
    "i cannot tell you",
    "i can't tell you",
    "i cannot answer",
    "i can't answer",
    "i don't have access to",
    "i do not have access to",
    "i'd recommend checking",
    "i would recommend checking",
    "rather than risk giving you inaccurate",
    "i don't have up-to-date",
    "i do not have up-to-date",
    "my training data doesn't include",
    "beyond my knowledge cutoff",
    "i'm not able to confirm",
    "the provided context does not contain",
    "not available in the provided context",
    "no information about this in the context",
)


def _cloud_response_is_non_answer(text: str) -> bool:
    """Return True if the cloud response is essentially 'I don't know, check elsewhere.'

    These responses should NOT be stored as learned knowledge — they would
    pollute memory with non-answers that get recalled on future similar queries.
    """
    if not text:
        return True
    lowered = text[:300].lower()
    return any(marker in lowered for marker in _NON_ANSWER_MARKERS)

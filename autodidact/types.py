"""Core data types for Autodidact framework."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class KnowledgeCategory(str, Enum):
    FACTS = "facts"
    EVENTS = "events"
    DISCOVERIES = "discoveries"
    PREFERENCES = "preferences"
    ADVICE = "advice"


class MemoryTier(str, Enum):
    STM = "STM"
    LTM = "LTM"


class RouteName(str, Enum):
    LOCAL = "LOCAL"
    CLOUD = "CLOUD"


class SignalScores(BaseModel):
    knowledge_similarity: float = Field(ge=0, le=1)
    logprob_uncertainty: float = Field(ge=0, le=1)
    self_consistency: float = Field(ge=0, le=1)
    query_classification: float = Field(ge=0, le=1)
    energy_scorer: Optional[float] = Field(default=None, ge=0, le=1)


class RoutingDecision(BaseModel):
    route: RouteName
    signals: SignalScores
    fusion_weights: dict[str, float]
    fused_score: float
    query_id: str


class KnowledgeScope(BaseModel):
    domain: Optional[str] = None
    topic: Optional[str] = None
    category: Optional[KnowledgeCategory] = None


class KnowledgeEntry(BaseModel):
    id: str
    content: str
    source: str
    confidence: float = 0.5
    tags: list[str] = Field(default_factory=list)
    embedding: Optional[list[float]] = None
    tier: MemoryTier = MemoryTier.STM
    usage_count: int = 0
    created_at: str
    last_accessed: str
    promoted_at: Optional[str] = None
    metadata: dict = Field(default_factory=dict)
    domain: str = "general"
    topic: str = "uncategorized"
    category: KnowledgeCategory = KnowledgeCategory.FACTS
    valid_from: str
    valid_to: Optional[str] = None
    verbatim_response: Optional[str] = None


class NewKnowledgeEntry(BaseModel):
    content: str
    source: str = "manual"
    confidence: float = 0.5
    tags: list[str] = Field(default_factory=list)
    embedding: Optional[list[float]] = None
    domain: str = "general"
    topic: str = "uncategorized"
    category: KnowledgeCategory = KnowledgeCategory.FACTS
    metadata: dict = Field(default_factory=dict)
    verbatim_response: Optional[str] = None


class SkillEntry(BaseModel):
    id: str
    name: str
    description: str
    steps: list[dict] = Field(default_factory=list)
    tool_references: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    embedding: Optional[list[float]] = None
    version: int = 1
    success_count: int = 0
    failure_count: int = 0
    created_at: str
    updated_at: str


class AgentMetrics(BaseModel):
    total_queries: int = 0
    local_resolutions: int = 0
    cloud_escalations: int = 0
    local_resolution_rate: float = 0.0
    knowledge_count: int = 0
    total_cost: float = 0.0


class AutodidactConfig(BaseModel):
    db_path: str = "autodidact.db"
    confidence_threshold: float = 0.7
    similarity_threshold: float = 0.75
    stm_promotion_accesses: int = 3
    decay_threshold: float = 0.1
    base_stability: float = 1.0
    energy_scorer_min_examples: int = 50
    energy_scorer_retrain_interval: int = 50
    embedding_dim: int = 384

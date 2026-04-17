// ── Agent ────────────────────────────────────────────────────
export { Agent } from './components/agent.js';
export type { AgentComponents } from './components/agent.js';

// ── Components ──────────────────────────────────────────────
export { KnowledgeStore } from './components/knowledge-store.js';
export { SkillStore } from './components/skill-store.js';
export { ConfidenceEvaluator } from './components/confidence-evaluator.js';
export { CloudRouter } from './components/cloud-router.js';
export { LearningExtractor } from './components/learning-extractor.js';
export { SelfVerificationSystem } from './components/self-verification.js';
export { SkillEvolver } from './components/skill-evolver.js';
export { UserProfile } from './components/user-profile.js';
export { MetricsTracker } from './components/metrics-tracker.js';
export { LLMClient } from './components/llm-client.js';
export { ContextBuilder } from './components/context-builder.js';
export { ToolRegistry } from './components/tool-registry.js';
export { BedrockRouter } from './components/bedrock-router.js';
export type { BedrockProviderConfig } from './components/bedrock-router.js';

// ── Types and Interfaces ────────────────────────────────────
export type {
    ChatMessage,
    ChatOptions,
    ChatResponse,
    EvaluationContext,
    RoutingResult,
    SignalScores,
    SignalWeights,
    QueryOutcome,
    KnowledgeCategory,
    KnowledgeScope,
    TemporalQuery,
    ContextLayerLevel,
    ContextLayerConfig,
    ContextResult,
    KnowledgeEntry,
    NewKnowledgeEntry,
    SkillStep,
    SkillEntry,
    NewSkillEntry,
    SkillExecutionResult,
    CloudProvider,
    CloudResponse,
    EscalationRecord,
    ExtractionResult,
    SelfTestQuestion,
    VerificationResult,
    AgentResponse,
    AgentMetrics,
    AutodidactConfig,
    AutodidactError,
    SkillReviewResult,
    UserProfileData,
    ProfileObservation,
    KnowledgeStoreStats,
    SkillStoreStats,
    ExpireResult,
    ILLMClient,
    IConfidenceEvaluator,
    IKnowledgeStore,
    IContextBuilder,
    ISkillStore,
    ICloudRouter,
    ILearningExtractor,
    ISelfVerificationSystem,
    IAgent,
    ISkillEvolver,
    IUserProfile,
    IMetricsTracker,
    ScoredKnowledgeEntry,
    IToolRegistry,
    ToolDefinition,
    ToolConfig,
    NewToolDefinition,
    ToolExecutionResult,
    ToolRegistryStats,
} from './types.js';

// ── Config ──────────────────────────────────────────────────
export { resolveConfig, validateConfig, DEFAULT_CONFIG } from './config.js';

// ── Skill Format ────────────────────────────────────────────
export { exportSkill, importSkill, exportAll } from './components/skill-format.js';

// ── Database ────────────────────────────────────────────────
export { initDatabase } from './database.js';

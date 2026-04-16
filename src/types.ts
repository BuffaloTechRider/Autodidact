// ── Chat Types ──────────────────────────────────────────────
export interface ChatMessage {
    role: 'system' | 'user' | 'assistant';
    content: string;
}

export interface ChatOptions {
    temperature?: number;
    maxTokens?: number;
    model?: string;
}

export interface ChatResponse {
    content: string;
    usage: { promptTokens: number; completionTokens: number };
    model: string;
}

// ── Confidence Evaluator Types ──────────────────────────────
export interface EvaluationContext {
    knowledgeHits: KnowledgeEntry[];
    skillHits: SkillEntry[];
}

export interface RoutingResult {
    decision: 'LOCAL' | 'HEDGE' | 'ESCALATE';
    signals: SignalScores;
    fusedScore: number;
    queryId: string;
}

export interface SignalScores {
    knowledgeSimilarity: number;
    skillCoverage: number;
    queryComplexity: number;
    selfAssessment: number;
}

export interface SignalWeights {
    knowledgeSimilarity: { alpha: number; beta: number };
    skillCoverage: { alpha: number; beta: number };
    queryComplexity: { alpha: number; beta: number };
    selfAssessment: { alpha: number; beta: number };
}

export type QueryOutcome = 'success' | 'failure';

// ── Knowledge Store Types ───────────────────────────────────
export interface KnowledgeEntry {
    id: string;
    content: string;
    source: 'cloud_escalation' | 'manual' | 'self_verification';
    confidence: number;
    tags: string[];
    embedding: number[] | null;
    tier: 'STM' | 'LTM';
    usageCount: number;
    createdAt: string;
    lastAccessed: string;
    promotedAt: string | null;
    isStale: boolean;
    selfTestQuestions: string[];
    metadata: Record<string, unknown>;
}

export interface NewKnowledgeEntry {
    content: string;
    source: 'cloud_escalation' | 'manual' | 'self_verification';
    confidence?: number;
    tags?: string[];
    embedding?: number[];
    selfTestQuestions?: string[];
    metadata?: Record<string, unknown>;
}

// ── Skill Store Types ───────────────────────────────────────
export interface SkillStep {
    order: number;
    description: string;
    input: string;
    output: string;
    toolName?: string;
}

export interface SkillEntry {
    id: string;
    name: string;
    description: string;
    steps: SkillStep[];
    tags: string[];
    embedding: number[] | null;
    version: number;
    parentId: string | null;
    successCount: number;
    failureCount: number;
    totalLatencyMs: number;
    invocationCount: number;
    createdAt: string;
    updatedAt: string;
    metadata: Record<string, unknown>;
}

export interface NewSkillEntry {
    name: string;
    description: string;
    steps: SkillStep[];
    tags?: string[];
    embedding?: number[];
    metadata?: Record<string, unknown>;
}

export interface SkillExecutionResult {
    success: boolean;
    latencyMs: number;
}

// ── Cloud Router Types ──────────────────────────────────────
export interface CloudProvider {
    name: string;
    baseUrl: string;
    apiKey: string;
    model: string;
    costPer1kTokens: number;
    timeoutMs: number;
    priority: number;
}

export interface CloudResponse {
    content: string;
    provider: string;
    model: string;
    cost: number;
    latencyMs: number;
}

export interface EscalationRecord {
    id: string;
    queryId: string;
    provider: string;
    model: string;
    cost: number;
    latencyMs: number;
    success: boolean;
    error: string | null;
    createdAt: string;
}

// ── Learning Extractor Types ────────────────────────────────
export interface ExtractionResult {
    knowledge: NewKnowledgeEntry[];
    skills: NewSkillEntry[];
    selfTestQuestions: SelfTestQuestion[];
}

export interface SelfTestQuestion {
    knowledgeId: string;
    question: string;
}

// ── Self-Verification Types ─────────────────────────────────
export interface VerificationResult {
    tested: number;
    passed: number;
    failed: number;
    staleEntries: string[];
}

// ── Agent Types ─────────────────────────────────────────────
export interface AgentResponse {
    content: string;
    routing: RoutingResult;
    cost: number;
    latencyMs: number;
    sourcesUsed: string[];
}

export interface AgentMetrics {
    localResolutionRate: number;
    knowledgeGrowthRate: number;
    cumulativeCostAvoided: number;
    selfTestPassRate: number;
    confidenceCalibration: number;
    totalQueries: number;
    totalEscalations: number;
    totalKnowledgeEntries: number;
    totalSkillEntries: number;
}

// ── Configuration ───────────────────────────────────────────
export interface AutodidactConfig {
    localLLM: { baseUrl: string; apiKey?: string; model: string; timeoutMs?: number };

    knowledgeStore: {
        stmTtlMs: number;
        promotionWindowMs: number;
        ltmBaseStabilityHours: number;
        decayThreshold: number;
        maxEntries?: number;
    };

    confidenceEvaluator: {
        localThreshold: number;
        hedgeThreshold: number;
        initialAlpha: number;
        initialBeta: number;
    };

    cloudRouter: {
        providers: CloudProvider[];
        maxRetries?: number;
    };

    selfVerification: {
        enabled: boolean;
        intervalMs: number;
        batchSize: number;
        queryCountThreshold: number;
    };

    skillEvolver: {
        enabled: boolean;
        reviewThreshold: number;
        minSuccessRate: number;
    };

    userProfile: {
        enabled: boolean;
        defaultProfile: string;
        autoExtract: boolean;
    };

    database: {
        path: string;
    };
}

// ── Error Type ──────────────────────────────────────────────
export interface AutodidactError {
    code: string;
    message: string;
    component: string;
    details?: unknown;
    timestamp: string;
}

// ── Skill Evolver Types ─────────────────────────────────────
export interface SkillReviewResult {
    skillId: string;
    skillName: string;
    previousVersion: number;
    action: 'kept' | 'evolved' | 'failed';
    newVersion?: number;
    reason: string;
}

// ── User Profile Types ──────────────────────────────────────
export interface UserProfileData {
    name: string;
    preferences: Record<string, string>;
    vocabulary: string[];
    conventions: string[];
    interactionCount: number;
    createdAt: string;
    updatedAt: string;
}

export interface ProfileObservation {
    type: 'preference' | 'vocabulary' | 'convention';
    key: string;
    value: string;
}

// ── Store Stats Types ───────────────────────────────────────
export interface KnowledgeStoreStats {
    total: number;
    stm: number;
    ltm: number;
    stale: number;
}

export interface SkillStoreStats {
    total: number;
    averageSuccessRate: number;
    versionCounts: Record<string, number>;
}

export interface ExpireResult {
    expired: number;
    promoted: number;
}

// ── Component Interfaces ────────────────────────────────────
export interface ILLMClient {
    chat(messages: ChatMessage[], options?: ChatOptions): Promise<ChatResponse>;
    embed(text: string): Promise<number[]>;
}

export interface IConfidenceEvaluator {
    evaluate(query: string, context: EvaluationContext): Promise<RoutingResult>;
    recordOutcome(queryId: string, outcome: QueryOutcome): void;
    getSignalWeights(): SignalWeights;
}

export interface IKnowledgeStore {
    insert(entry: NewKnowledgeEntry): KnowledgeEntry;
    search(query: string, embedding: number[], limit?: number): KnowledgeEntry[];
    get(id: string): KnowledgeEntry | null;
    access(id: string): void;
    promoteToLTM(id: string): void;
    expire(id: string): void;
    runDecayCycle(): ExpireResult;
    getStats(): KnowledgeStoreStats;
}

export interface ISkillStore {
    insert(entry: NewSkillEntry): SkillEntry;
    search(query: string, embedding: number[], limit?: number): SkillEntry[];
    get(id: string): SkillEntry | null;
    updateMetrics(id: string, result: SkillExecutionResult): void;
    getVersion(id: string, version: number): SkillEntry | null;
    getStats(): SkillStoreStats;
}

export interface ICloudRouter {
    escalate(query: string, context?: string): Promise<CloudResponse>;
    getProviders(): CloudProvider[];
    getEscalationLog(): EscalationRecord[];
}

export interface ILearningExtractor {
    extract(query: string, response: string): Promise<ExtractionResult>;
}

export interface ISelfVerificationSystem {
    runVerificationCycle(): Promise<VerificationResult>;
    getPassRate(): number;
}

export interface IAgent {
    query(text: string): Promise<AgentResponse>;
    getMetrics(): AgentMetrics;
    getConfig(): AutodidactConfig;
}

export interface ISkillEvolver {
    reviewSkill(skillId: string): Promise<SkillReviewResult>;
    checkAndEvolve(): Promise<SkillReviewResult[]>;
}

export interface IUserProfile {
    get(profileName: string): UserProfileData | null;
    update(profileName: string, observations: ProfileObservation[]): void;
    getContext(profileName: string): string;
    list(): string[];
    reset(profileName: string): void;
}

export interface IMetricsTracker {
    recordQuery(entry: {
        id: string;
        queryText: string;
        routingDecision: string;
        signals: SignalScores;
        fusedScore: number;
        cost: number;
        latencyMs: number;
    }): void;
    recordOutcome(queryId: string, outcome: QueryOutcome, responseText?: string): void;
    getMetrics(): AgentMetrics;
}

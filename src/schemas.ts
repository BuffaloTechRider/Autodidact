import { z } from 'zod';

// ── Skill Step Schema ───────────────────────────────────────
export const SkillStepSchema = z.object({
    order: z.number().int(),
    description: z.string(),
    input: z.string(),
    output: z.string(),
    toolName: z.string().optional(),
});

// ── Signal Scores Schema ────────────────────────────────────
export const SignalScoresSchema = z.object({
    knowledgeSimilarity: z.number().min(0).max(1),
    skillCoverage: z.number().min(0).max(1),
    queryComplexity: z.number().min(0).max(1),
    selfAssessment: z.number().min(0).max(1),
});

// ── Routing Result Schema ───────────────────────────────────
export const RoutingResultSchema = z.object({
    decision: z.enum(['LOCAL', 'HEDGE', 'ESCALATE']),
    signals: SignalScoresSchema,
    fusedScore: z.number().min(0).max(1),
    queryId: z.string(),
});

// ── Chat Response Schema ────────────────────────────────────
export const ChatResponseSchema = z.object({
    content: z.string(),
    usage: z.object({
        promptTokens: z.number().int().min(0),
        completionTokens: z.number().int().min(0),
    }),
    model: z.string(),
});

// ── Cloud Provider Schema ───────────────────────────────────
export const CloudProviderSchema = z.object({
    name: z.string().min(1),
    baseUrl: z.string().url(),
    apiKey: z.string(),
    model: z.string().min(1),
    costPer1kTokens: z.number().min(0),
    timeoutMs: z.number().int().positive(),
    priority: z.number().int().min(0),
});

// ── Knowledge Category Schema ────────────────────────────────
export const KnowledgeCategorySchema = z.enum(['facts', 'events', 'discoveries', 'preferences', 'advice']);

// ── Knowledge Scope Schema ──────────────────────────────────
export const KnowledgeScopeSchema = z.object({
    domain: z.string().optional(),
    topic: z.string().optional(),
    category: KnowledgeCategorySchema.optional(),
});

// ── Temporal Query Schema ───────────────────────────────────
export const TemporalQuerySchema = z.object({
    asOf: z.string().optional(),
});

// ── New Knowledge Entry Schema ──────────────────────────────
export const NewKnowledgeEntrySchema = z.object({
    content: z.string().min(1),
    source: z.enum(['cloud_escalation', 'manual', 'self_verification']),
    confidence: z.number().min(0).max(1).optional(),
    tags: z.array(z.string()).optional(),
    embedding: z.array(z.number()).optional(),
    selfTestQuestions: z.array(z.string()).optional(),
    metadata: z.record(z.unknown()).optional(),
    domain: z.string().optional(),
    topic: z.string().optional(),
    category: KnowledgeCategorySchema.optional(),
});

// ── New Skill Entry Schema ──────────────────────────────────
export const NewSkillEntrySchema = z.object({
    name: z.string().min(1),
    description: z.string().min(1),
    steps: z.array(SkillStepSchema).min(1),
    tags: z.array(z.string()).optional(),
    embedding: z.array(z.number()).optional(),
    metadata: z.record(z.unknown()).optional(),
});

// ── Tool Config Schema ───────────────────────────────────────
export const ToolConfigSchema = z.object({
    url: z.string().optional(),
    method: z.string().optional(),
    headers: z.record(z.string()).optional(),
    authType: z.enum(['none', 'api_key', 'bearer', 'basic']).optional(),
    authKey: z.string().optional(),
    code: z.string().optional(),
    command: z.string().optional(),
    timeout: z.number().positive().optional(),
});

// ── New Tool Definition Schema ──────────────────────────────
export const NewToolDefinitionSchema = z.object({
    name: z.string().min(1),
    description: z.string().min(1),
    type: z.enum(['http', 'code', 'shell']),
    config: ToolConfigSchema,
    source: z.enum(['built_in', 'user_registered', 'learned']).optional(),
    learnedFromEscalation: z.string().optional(),
});

// ── Extraction Result Schema ────────────────────────────────
export const ExtractionResultSchema = z.object({
    knowledge: z.array(NewKnowledgeEntrySchema),
    skills: z.array(NewSkillEntrySchema),
    selfTestQuestions: z.array(
        z.object({
            knowledgeId: z.string(),
            question: z.string(),
        })
    ),
    tools: z.array(NewToolDefinitionSchema),
});

// ── Autodidact Config Schema ────────────────────────────────
export const AutodidactConfigSchema = z.object({
    localLLM: z.object({
        baseUrl: z.string().url(),
        apiKey: z.string().optional(),
        model: z.string().min(1),
        timeoutMs: z.number().int().positive().optional(),
    }),

    knowledgeStore: z
        .object({
            stmTtlMs: z.number().int().positive().default(3_600_000),
            promotionWindowMs: z.number().int().positive().default(3_600_000),
            ltmBaseStabilityHours: z.number().positive().default(168),
            decayThreshold: z.number().min(0).max(1).default(0.1),
            maxEntries: z.number().int().positive().optional(),
        })
        .default({}),

    confidenceEvaluator: z
        .object({
            localThreshold: z.number().min(0).max(1).default(0.7),
            hedgeThreshold: z.number().min(0).max(1).default(0.4),
            initialAlpha: z.number().positive().default(1.0),
            initialBeta: z.number().positive().default(1.0),
        })
        .default({}),

    cloudRouter: z.object({
        providers: z.array(CloudProviderSchema),
        maxRetries: z.number().int().positive().optional(),
    }),

    selfVerification: z
        .object({
            enabled: z.boolean().default(true),
            intervalMs: z.number().int().positive().default(86_400_000),
            batchSize: z.number().int().positive().default(20),
            queryCountThreshold: z.number().int().positive().default(50),
        })
        .default({}),

    skillEvolver: z
        .object({
            enabled: z.boolean().default(true),
            reviewThreshold: z.number().int().positive().default(10),
            minSuccessRate: z.number().min(0).max(1).default(0.6),
        })
        .default({}),

    userProfile: z
        .object({
            enabled: z.boolean().default(true),
            defaultProfile: z.string().default('default'),
            autoExtract: z.boolean().default(true),
        })
        .default({}),

    database: z
        .object({
            path: z.string().default('./autodidact.db'),
        })
        .default({}),

    contextLayers: z
        .object({
            l0TokenBudget: z.number().int().positive().default(50),
            l1TokenBudget: z.number().int().positive().default(120),
            l2TokenBudget: z.number().int().positive().default(500),
            l3TokenBudget: z.number().int().positive().default(1000),
            l3Threshold: z.number().min(0).max(1).default(0.5),
        })
        .default({}),

    toolRegistry: z
        .object({
            enabled: z.boolean().default(true),
            autoVerify: z.boolean().default(true),
            decayThreshold: z.number().min(0).max(1).default(0.1),
        })
        .default({}),
});

// ── Validation Helpers ──────────────────────────────────────
export function validateConfig(data: unknown) {
    return AutodidactConfigSchema.parse(data);
}

export function validateChatResponse(data: unknown) {
    return ChatResponseSchema.parse(data);
}

export function validateExtractionResult(data: unknown) {
    return ExtractionResultSchema.parse(data);
}

export function validateNewKnowledgeEntry(data: unknown) {
    return NewKnowledgeEntrySchema.parse(data);
}

export function validateNewSkillEntry(data: unknown) {
    return NewSkillEntrySchema.parse(data);
}

export function validateCloudProvider(data: unknown) {
    return CloudProviderSchema.parse(data);
}

export function safeValidateConfig(data: unknown) {
    return AutodidactConfigSchema.safeParse(data);
}

export function safeValidateChatResponse(data: unknown) {
    return ChatResponseSchema.safeParse(data);
}

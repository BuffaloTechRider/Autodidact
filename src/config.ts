import { AutodidactConfigSchema } from './schemas.js';
import type { AutodidactConfig } from './types.js';

/**
 * Default configuration with sensible defaults from the design document.
 */
export const DEFAULT_CONFIG: AutodidactConfig = {
    localLLM: {
        baseUrl: 'http://localhost:11434/v1',
        model: 'llama3',
        timeoutMs: 30_000,
    },

    knowledgeStore: {
        stmTtlMs: 3_600_000,           // 1 hour
        promotionWindowMs: 3_600_000,   // 1 hour
        ltmBaseStabilityHours: 168,     // 7 days
        decayThreshold: 0.1,
    },

    confidenceEvaluator: {
        localThreshold: 0.7,
        hedgeThreshold: 0.4,
        initialAlpha: 1.0,
        initialBeta: 1.0,
    },

    cloudRouter: {
        providers: [],
    },

    selfVerification: {
        enabled: true,
        intervalMs: 86_400_000,         // 24 hours
        batchSize: 20,
        queryCountThreshold: 50,
    },

    skillEvolver: {
        enabled: true,
        reviewThreshold: 10,
        minSuccessRate: 0.6,
    },

    userProfile: {
        enabled: true,
        defaultProfile: 'default',
        autoExtract: true,
    },

    database: {
        path: './autodidact.db',
    },
};

/**
 * Deep-merge a partial user config with the defaults.
 */
export function resolveConfig(partial?: Partial<AutodidactConfig>): AutodidactConfig {
    if (!partial) return { ...DEFAULT_CONFIG };

    return {
        localLLM: {
            ...DEFAULT_CONFIG.localLLM,
            ...partial.localLLM,
        },
        knowledgeStore: {
            ...DEFAULT_CONFIG.knowledgeStore,
            ...partial.knowledgeStore,
        },
        confidenceEvaluator: {
            ...DEFAULT_CONFIG.confidenceEvaluator,
            ...partial.confidenceEvaluator,
        },
        cloudRouter: {
            ...DEFAULT_CONFIG.cloudRouter,
            ...partial.cloudRouter,
        },
        selfVerification: {
            ...DEFAULT_CONFIG.selfVerification,
            ...partial.selfVerification,
        },
        skillEvolver: {
            ...DEFAULT_CONFIG.skillEvolver,
            ...partial.skillEvolver,
        },
        userProfile: {
            ...DEFAULT_CONFIG.userProfile,
            ...partial.userProfile,
        },
        database: {
            ...DEFAULT_CONFIG.database,
            ...partial.database,
        },
    };
}

/**
 * Validate config data with Zod. Throws descriptive errors on invalid fields.
 */
export function validateConfig(data: unknown): AutodidactConfig {
    return AutodidactConfigSchema.parse(data) as AutodidactConfig;
}

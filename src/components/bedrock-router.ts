import {
    BedrockRuntimeClient,
    ConverseCommand,
} from '@aws-sdk/client-bedrock-runtime';
import type {
    AutodidactError,
    CloudProvider,
    CloudResponse,
    EscalationRecord,
    ICloudRouter,
} from '../types.js';
import { generateId } from '../utils/id.js';
import type { Logger } from '../utils/logger.js';
import { defaultLogger } from '../utils/logger.js';
import { nowISO } from '../utils/time.js';
import type Database from 'better-sqlite3';

export interface BedrockProviderConfig {
    region: string;
    modelId: string;
    costPer1kInputTokens: number;
    costPer1kOutputTokens: number;
    maxTokens?: number;
    temperature?: number;
}

/**
 * CloudRouter implementation that uses AWS Bedrock Converse API.
 * Uses IAM credentials from the environment (AWS_PROFILE, AWS_ACCESS_KEY_ID, etc.)
 */
export class BedrockRouter implements ICloudRouter {
    private readonly db: Database.Database;
    private readonly providers: BedrockProviderConfig[];
    private readonly logger: Logger;

    constructor(
        db: Database.Database,
        providers: BedrockProviderConfig[],
        logger: Logger = defaultLogger,
    ) {
        this.db = db;
        this.providers = providers;
        this.logger = logger;
    }

    async escalate(query: string, context?: string): Promise<CloudResponse> {
        const queryId = generateId();
        const errors: { provider: string; error: string }[] = [];

        for (const provider of this.providers) {
            const startTime = Date.now();
            try {
                const client = new BedrockRuntimeClient({ region: provider.region });

                const userContent = context
                    ? `Context:\n${context}\n\nQuestion: ${query}`
                    : query;

                const command = new ConverseCommand({
                    modelId: provider.modelId,
                    messages: [{
                        role: 'user',
                        content: [{ text: userContent }],
                    }],
                    inferenceConfig: {
                        maxTokens: provider.maxTokens ?? 2048,
                        temperature: provider.temperature ?? 0.7,
                    },
                });

                const response = await client.send(command);
                const latencyMs = Date.now() - startTime;

                const outputText = response.output?.message?.content?.[0]?.text ?? '';
                const inputTokens = response.usage?.inputTokens ?? 0;
                const outputTokens = response.usage?.outputTokens ?? 0;

                const cost =
                    (inputTokens / 1000) * provider.costPer1kInputTokens +
                    (outputTokens / 1000) * provider.costPer1kOutputTokens;

                this.recordEscalation(queryId, provider.modelId, cost, latencyMs, true, null);

                this.logger.info('BedrockRouter.escalate: success', {
                    model: provider.modelId,
                    latencyMs,
                    cost,
                    inputTokens,
                    outputTokens,
                });

                return {
                    content: outputText,
                    provider: 'bedrock',
                    model: provider.modelId,
                    cost,
                    latencyMs,
                };
            } catch (err) {
                const latencyMs = Date.now() - startTime;
                const errorMsg = err instanceof Error ? err.message : String(err);
                errors.push({ provider: provider.modelId, error: errorMsg });
                this.recordEscalation(queryId, provider.modelId, 0, latencyMs, false, errorMsg);

                this.logger.warn('BedrockRouter.escalate: provider failed', {
                    model: provider.modelId,
                    error: errorMsg,
                });
            }
        }

        const error: AutodidactError = {
            code: 'CLOUD_ALL_FAILED',
            message: `All ${this.providers.length} Bedrock providers failed`,
            component: 'BedrockRouter',
            details: { errors, queryId },
            timestamp: nowISO(),
        };
        throw error;
    }

    getProviders(): CloudProvider[] {
        return this.providers.map(p => ({
            name: `bedrock:${p.modelId}`,
            baseUrl: `https://bedrock-runtime.${p.region}.amazonaws.com`,
            apiKey: 'iam',
            model: p.modelId,
            costPer1kTokens: p.costPer1kInputTokens,
            timeoutMs: 120_000,
            priority: 1,
        }));
    }

    getEscalationLog(): EscalationRecord[] {
        const rows = this.db
            .prepare(`SELECT * FROM escalation_log ORDER BY created_at DESC`)
            .all() as any[];
        return rows.map(row => ({
            id: row.id,
            queryId: row.query_id,
            provider: row.provider,
            model: row.model,
            cost: row.cost,
            latencyMs: row.latency_ms,
            success: row.success === 1,
            error: row.error,
            createdAt: row.created_at,
        }));
    }

    private recordEscalation(
        queryId: string, model: string, cost: number,
        latencyMs: number, success: boolean, error: string | null,
    ): void {
        this.db.prepare(
            `INSERT INTO escalation_log (id, query_id, provider, model, cost, latency_ms, success, error, created_at)
             VALUES (?, ?, 'bedrock', ?, ?, ?, ?, ?, ?)`,
        ).run(generateId(), queryId, model, cost, latencyMs, success ? 1 : 0, error, nowISO());
    }
}

import type Database from 'better-sqlite3';
import type {
    AutodidactError,
    CloudProvider,
    CloudResponse,
    EscalationRecord,
    ICloudRouter,
} from '../types.js';
import { LLMClient } from './llm-client.js';
import { generateId } from '../utils/id.js';
import type { Logger } from '../utils/logger.js';
import { defaultLogger } from '../utils/logger.js';
import { nowISO } from '../utils/time.js';

export class CloudRouter implements ICloudRouter {
    private readonly db: Database.Database;
    private readonly providers: CloudProvider[];
    private readonly logger: Logger;

    constructor(
        db: Database.Database,
        providers: CloudProvider[],
        logger: Logger = defaultLogger,
    ) {
        this.db = db;
        // Sort by priority first, then by cost (cheapest first)
        this.providers = [...providers].sort((a, b) => {
            if (a.priority !== b.priority) return a.priority - b.priority;
            return a.costPer1kTokens - b.costPer1kTokens;
        });
        this.logger = logger;
    }

    async escalate(query: string, context?: string): Promise<CloudResponse> {
        const queryId = generateId();
        const errors: { provider: string; error: string }[] = [];

        for (const provider of this.providers) {
            const startTime = Date.now();
            try {
                const client = new LLMClient(
                    {
                        baseUrl: provider.baseUrl,
                        apiKey: provider.apiKey,
                        model: provider.model,
                        timeoutMs: provider.timeoutMs,
                    },
                    this.logger,
                );

                const messages = [
                    ...(context
                        ? [{ role: 'system' as const, content: context }]
                        : []),
                    { role: 'user' as const, content: query },
                ];

                const response = await client.chat(messages);
                const latencyMs = Date.now() - startTime;
                const totalTokens =
                    response.usage.promptTokens +
                    response.usage.completionTokens;
                const cost = (totalTokens / 1000) * provider.costPer1kTokens;

                // Record success in escalation_log
                this.recordEscalation(
                    queryId,
                    provider,
                    cost,
                    latencyMs,
                    true,
                    null,
                );

                this.logger.info('CloudRouter.escalate: success', {
                    provider: provider.name,
                    latencyMs,
                    cost,
                });

                return {
                    content: response.content,
                    provider: provider.name,
                    model: provider.model,
                    cost,
                    latencyMs,
                };
            } catch (err) {
                const latencyMs = Date.now() - startTime;
                const errorMsg =
                    err instanceof Error
                        ? err.message
                        : isAutodidactError(err)
                            ? err.message
                            : String(err);

                errors.push({ provider: provider.name, error: errorMsg });

                // Record failure in escalation_log
                this.recordEscalation(
                    queryId,
                    provider,
                    0,
                    latencyMs,
                    false,
                    errorMsg,
                );

                this.logger.warn('CloudRouter.escalate: provider failed', {
                    provider: provider.name,
                    error: errorMsg,
                });
            }
        }

        // All providers failed
        const error: AutodidactError = {
            code: 'CLOUD_ALL_FAILED',
            message: `All ${this.providers.length} cloud providers failed`,
            component: 'CloudRouter',
            details: { errors, queryId },
            timestamp: nowISO(),
        };

        this.logger.error('CloudRouter.escalate: all providers failed', {
            errors,
        });

        throw error;
    }

    getProviders(): CloudProvider[] {
        return [...this.providers];
    }

    getEscalationLog(): EscalationRecord[] {
        const rows = this.db
            .prepare(
                `SELECT * FROM escalation_log ORDER BY created_at DESC`,
            )
            .all() as RawEscalationRow[];

        return rows.map((row) => ({
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

    // ── Private helpers ─────────────────────────────────────

    private recordEscalation(
        queryId: string,
        provider: CloudProvider,
        cost: number,
        latencyMs: number,
        success: boolean,
        error: string | null,
    ): void {
        const id = generateId();
        const now = nowISO();

        this.db
            .prepare(
                `INSERT INTO escalation_log
                (id, query_id, provider, model, cost, latency_ms, success, error, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)`,
            )
            .run(
                id,
                queryId,
                provider.name,
                provider.model,
                cost,
                latencyMs,
                success ? 1 : 0,
                error,
                now,
            );
    }
}

function isAutodidactError(err: unknown): err is AutodidactError {
    return (
        typeof err === 'object' &&
        err !== null &&
        'code' in err &&
        'component' in err &&
        'timestamp' in err
    );
}

/** Raw SQLite row shape for escalation_log */
interface RawEscalationRow {
    id: string;
    query_id: string;
    provider: string;
    model: string;
    cost: number;
    latency_ms: number;
    success: number;
    error: string | null;
    created_at: string;
}

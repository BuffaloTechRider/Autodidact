import type Database from 'better-sqlite3';
import type {
    IToolRegistry,
    NewToolDefinition,
    ToolConfig,
    ToolDefinition,
    ToolExecutionResult,
    ToolRegistryStats,
} from '../types.js';
import { generateId } from '../utils/id.js';
import type { Logger } from '../utils/logger.js';
import { defaultLogger } from '../utils/logger.js';
import { nowISO } from '../utils/time.js';

export interface ToolRegistryConfig {
    enabled: boolean;
    autoVerify: boolean;
    decayThreshold: number;
}

const DEFAULT_CONFIG: ToolRegistryConfig = {
    enabled: true,
    autoVerify: true,
    decayThreshold: 0.1,
};

interface RawToolRow {
    id: string;
    name: string;
    description: string;
    type: string;
    config: string;
    source: string;
    status: string;
    confidence: number;
    usage_count: number;
    success_count: number;
    failure_count: number;
    created_at: string;
    updated_at: string;
    learned_from_escalation: string | null;
}

export class ToolRegistry implements IToolRegistry {
    private readonly db: Database.Database;
    private readonly config: ToolRegistryConfig;
    private readonly logger: Logger;

    constructor(
        db: Database.Database,
        config?: Partial<ToolRegistryConfig>,
        logger: Logger = defaultLogger,
    ) {
        this.db = db;
        this.config = { ...DEFAULT_CONFIG, ...config };
        this.logger = logger;
    }

    register(tool: NewToolDefinition): ToolDefinition {
        const id = generateId();
        const now = nowISO();
        const source = tool.source ?? 'user_registered';
        const status = source === 'built_in' ? 'verified' : 'unverified';
        const confidence = source === 'built_in' ? 1.0 : 0.5;

        this.db
            .prepare(
                `INSERT INTO tool_registry
                (id, name, description, type, config, source, status, confidence,
                 usage_count, success_count, failure_count, created_at, updated_at,
                 learned_from_escalation)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, 0, 0, ?, ?, ?)`,
            )
            .run(
                id,
                tool.name,
                tool.description,
                tool.type,
                JSON.stringify(tool.config),
                source,
                status,
                confidence,
                now,
                now,
                tool.learnedFromEscalation ?? null,
            );

        this.logger.debug('ToolRegistry.register: created tool', { id, name: tool.name, source });
        return this.getById(id)!;
    }

    get(name: string): ToolDefinition | null {
        const row = this.db
            .prepare(`SELECT * FROM tool_registry WHERE name = ?`)
            .get(name) as RawToolRow | undefined;
        return row ? this.rowToTool(row) : null;
    }

    getById(id: string): ToolDefinition | null {
        const row = this.db
            .prepare(`SELECT * FROM tool_registry WHERE id = ?`)
            .get(id) as RawToolRow | undefined;
        return row ? this.rowToTool(row) : null;
    }

    async execute(name: string, params: Record<string, string>): Promise<ToolExecutionResult> {
        const tool = this.get(name);
        if (!tool) {
            return { success: false, output: '', error: 'Tool not found', latencyMs: 0 };
        }
        if (tool.status === 'failed' || tool.status === 'dormant') {
            return { success: false, output: '', error: `Tool is ${tool.status}`, latencyMs: 0 };
        }

        if (tool.type === 'http') {
            let url = tool.config.url ?? '';
            for (const [key, value] of Object.entries(params)) {
                url = url.replace(`{{${key}}}`, encodeURIComponent(value));
            }

            const start = Date.now();
            try {
                const controller = new AbortController();
                const timeout = tool.config.timeout ?? 30_000;
                const timer = setTimeout(() => controller.abort(), timeout);

                const response = await fetch(url, {
                    method: tool.config.method ?? 'GET',
                    headers: tool.config.headers,
                    signal: controller.signal,
                });
                clearTimeout(timer);

                const text = await response.text();
                const latencyMs = Date.now() - start;
                const success = response.ok;
                this.recordUsage(tool.id, success, latencyMs);
                return { success, output: text, latencyMs };
            } catch (err) {
                const latencyMs = Date.now() - start;
                this.recordUsage(tool.id, false, latencyMs);
                return {
                    success: false,
                    output: '',
                    error: err instanceof Error ? err.message : String(err),
                    latencyMs,
                };
            }
        }

        return {
            success: false,
            output: '',
            error: `Tool type '${tool.type}' execution not yet supported (sandbox required)`,
            latencyMs: 0,
        };
    }

    async verify(name: string): Promise<boolean> {
        const tool = this.get(name);
        if (!tool) return false;

        if (tool.type !== 'http') {
            this.logger.warn('ToolRegistry.verify: only http tools can be auto-verified', { name });
            return false;
        }

        const result = await this.execute(name, {});
        const now = nowISO();

        if (result.success) {
            this.db
                .prepare(`UPDATE tool_registry SET status = 'verified', confidence = MIN(confidence + 0.2, 1.0), updated_at = ? WHERE name = ?`)
                .run(now, name);
            this.logger.info('ToolRegistry.verify: tool verified', { name });
            return true;
        } else {
            this.db
                .prepare(`UPDATE tool_registry SET status = 'failed', updated_at = ? WHERE name = ?`)
                .run(now, name);
            this.logger.warn('ToolRegistry.verify: tool verification failed', { name, error: result.error });
            return false;
        }
    }

    list(filter?: { source?: string; status?: string }): ToolDefinition[] {
        let sql = `SELECT * FROM tool_registry WHERE 1=1`;
        const params: string[] = [];

        if (filter?.source) {
            sql += ` AND source = ?`;
            params.push(filter.source);
        }
        if (filter?.status) {
            sql += ` AND status = ?`;
            params.push(filter.status);
        }

        sql += ` ORDER BY name`;
        const rows = this.db.prepare(sql).all(...params) as RawToolRow[];
        return rows.map((r) => this.rowToTool(r));
    }

    getStats(): ToolRegistryStats {
        const count = (sql: string): number =>
            (this.db.prepare(sql).get() as { cnt: number }).cnt;

        return {
            total: count(`SELECT COUNT(*) as cnt FROM tool_registry`),
            verified: count(`SELECT COUNT(*) as cnt FROM tool_registry WHERE status = 'verified'`),
            failed: count(`SELECT COUNT(*) as cnt FROM tool_registry WHERE status = 'failed'`),
            dormant: count(`SELECT COUNT(*) as cnt FROM tool_registry WHERE status = 'dormant'`),
            builtIn: count(`SELECT COUNT(*) as cnt FROM tool_registry WHERE source = 'built_in'`),
            learned: count(`SELECT COUNT(*) as cnt FROM tool_registry WHERE source = 'learned'`),
            userRegistered: count(`SELECT COUNT(*) as cnt FROM tool_registry WHERE source = 'user_registered'`),
        };
    }

    applyDecay(decayRate: number): void {
        const now = nowISO();
        // Reduce confidence of all non-built_in tools that haven't been used recently
        this.db
            .prepare(
                `UPDATE tool_registry
                 SET confidence = MAX(confidence - ?, 0.0), updated_at = ?
                 WHERE source != 'built_in' AND status != 'failed'`,
            )
            .run(decayRate, now);

        // Mark tools below threshold as dormant
        this.db
            .prepare(
                `UPDATE tool_registry
                 SET status = 'dormant', updated_at = ?
                 WHERE confidence < ? AND status NOT IN ('failed', 'dormant') AND source != 'built_in'`,
            )
            .run(now, this.config.decayThreshold);

        this.logger.debug('ToolRegistry.applyDecay: decay applied', { decayRate });
    }

    // ── Private helpers ─────────────────────────────────────

    private recordUsage(id: string, success: boolean, _latencyMs: number): void {
        const now = nowISO();
        if (success) {
            this.db
                .prepare(
                    `UPDATE tool_registry
                     SET usage_count = usage_count + 1, success_count = success_count + 1,
                         confidence = MIN(confidence + 0.05, 1.0), updated_at = ?
                     WHERE id = ?`,
                )
                .run(now, id);
        } else {
            this.db
                .prepare(
                    `UPDATE tool_registry
                     SET usage_count = usage_count + 1, failure_count = failure_count + 1,
                         confidence = MAX(confidence - 0.1, 0.0), updated_at = ?
                     WHERE id = ?`,
                )
                .run(now, id);
        }
    }

    private rowToTool(row: RawToolRow): ToolDefinition {
        return {
            id: row.id,
            name: row.name,
            description: row.description,
            type: row.type as ToolDefinition['type'],
            config: JSON.parse(row.config) as ToolConfig,
            source: row.source as ToolDefinition['source'],
            status: row.status as ToolDefinition['status'],
            confidence: row.confidence,
            usageCount: row.usage_count,
            successCount: row.success_count,
            failureCount: row.failure_count,
            createdAt: row.created_at,
            updatedAt: row.updated_at,
            learnedFromEscalation: row.learned_from_escalation ?? undefined,
        };
    }
}

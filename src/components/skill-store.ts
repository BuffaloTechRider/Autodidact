import type Database from 'better-sqlite3';
import type {
    ISkillStore,
    NewSkillEntry,
    SkillEntry,
    SkillExecutionResult,
    SkillStoreStats,
} from '../types.js';
import { cosineSimilarity } from '../utils/cosine-similarity.js';
import {
    deserializeEmbedding,
    serializeEmbedding,
} from '../utils/embeddings.js';
import { generateId } from '../utils/id.js';
import type { Logger } from '../utils/logger.js';
import { defaultLogger } from '../utils/logger.js';
import { nowISO } from '../utils/time.js';

export class SkillStore implements ISkillStore {
    private readonly db: Database.Database;
    private readonly logger: Logger;

    constructor(db: Database.Database, logger: Logger = defaultLogger) {
        this.db = db;
        this.logger = logger;
    }

    insert(entry: NewSkillEntry): SkillEntry {
        const id = generateId();
        const now = nowISO();
        const embeddingBlob = entry.embedding
            ? serializeEmbedding(entry.embedding)
            : null;

        this.db
            .prepare(
                `INSERT INTO skill_entries
                (id, name, description, steps, tags, embedding, version,
                 parent_id, success_count, failure_count, total_latency_ms,
                 invocation_count, created_at, updated_at, metadata)
                VALUES (?, ?, ?, ?, ?, ?, 1, NULL, 0, 0, 0, 0, ?, ?, ?)`,
            )
            .run(
                id,
                entry.name,
                entry.description,
                JSON.stringify(entry.steps),
                JSON.stringify(entry.tags ?? []),
                embeddingBlob,
                now,
                now,
                JSON.stringify(entry.metadata ?? {}),
            );

        this.logger.debug('SkillStore.insert: created skill', { id, name: entry.name });

        return this.get(id)!;
    }

    search(
        _query: string,
        embedding: number[],
        limit: number = 10,
    ): SkillEntry[] {
        const rows = this.db
            .prepare(`SELECT * FROM skill_entries`)
            .all() as RawSkillRow[];

        const scored: { entry: SkillEntry; score: number }[] = [];

        for (const row of rows) {
            const entry = this.rowToEntry(row);
            if (!entry.embedding || entry.embedding.length === 0) {
                continue;
            }
            const score = cosineSimilarity(embedding, entry.embedding);
            scored.push({ entry, score });
        }

        scored.sort((a, b) => b.score - a.score);
        return scored.slice(0, limit).map((s) => s.entry);
    }

    get(id: string): SkillEntry | null {
        const row = this.db
            .prepare(`SELECT * FROM skill_entries WHERE id = ?`)
            .get(id) as RawSkillRow | undefined;

        if (!row) {
            return null;
        }
        return this.rowToEntry(row);
    }

    getVersion(id: string, version: number): SkillEntry | null {
        // The interface parameter is named 'id' but per the task spec,
        // this method looks up by name + version number.
        const row = this.db
            .prepare(`SELECT * FROM skill_entries WHERE name = ? AND version = ?`)
            .get(id, version) as RawSkillRow | undefined;

        if (!row) {
            return null;
        }
        return this.rowToEntry(row);
    }

    updateMetrics(id: string, result: SkillExecutionResult): void {
        const entry = this.get(id);
        if (!entry) {
            this.logger.warn('SkillStore.updateMetrics: skill not found', { id });
            return;
        }

        const now = nowISO();
        if (result.success) {
            this.db
                .prepare(
                    `UPDATE skill_entries
                     SET success_count = success_count + 1,
                         total_latency_ms = total_latency_ms + ?,
                         invocation_count = invocation_count + 1,
                         updated_at = ?
                     WHERE id = ?`,
                )
                .run(result.latencyMs, now, id);
        } else {
            this.db
                .prepare(
                    `UPDATE skill_entries
                     SET failure_count = failure_count + 1,
                         total_latency_ms = total_latency_ms + ?,
                         invocation_count = invocation_count + 1,
                         updated_at = ?
                     WHERE id = ?`,
                )
                .run(result.latencyMs, now, id);
        }

        this.logger.debug('SkillStore.updateMetrics: updated', {
            id,
            success: result.success,
        });
    }

    getStats(): SkillStoreStats {
        const total = (
            this.db
                .prepare(`SELECT COUNT(*) as cnt FROM skill_entries`)
                .get() as { cnt: number }
        ).cnt;

        const rows = this.db
            .prepare(
                `SELECT name, version, success_count, failure_count, invocation_count
                 FROM skill_entries`,
            )
            .all() as {
                name: string;
                version: number;
                success_count: number;
                failure_count: number;
                invocation_count: number;
            }[];

        let totalSuccessRate = 0;
        let countWithInvocations = 0;
        const versionCounts: Record<string, number> = {};

        for (const row of rows) {
            if (row.invocation_count > 0) {
                totalSuccessRate += row.success_count / row.invocation_count;
                countWithInvocations++;
            }
            const key = row.name;
            versionCounts[key] = (versionCounts[key] ?? 0) + 1;
        }

        const averageSuccessRate =
            countWithInvocations > 0 ? totalSuccessRate / countWithInvocations : 0;

        return { total, averageSuccessRate, versionCounts };
    }

    // ── Private helpers ─────────────────────────────────────

    private rowToEntry(row: RawSkillRow): SkillEntry {
        return {
            id: row.id,
            name: row.name,
            description: row.description,
            steps: JSON.parse(row.steps),
            tags: JSON.parse(row.tags) as string[],
            embedding: row.embedding
                ? deserializeEmbedding(row.embedding as Buffer)
                : null,
            version: row.version,
            parentId: row.parent_id,
            successCount: row.success_count,
            failureCount: row.failure_count,
            totalLatencyMs: row.total_latency_ms,
            invocationCount: row.invocation_count,
            createdAt: row.created_at,
            updatedAt: row.updated_at,
            metadata: JSON.parse(row.metadata) as Record<string, unknown>,
        };
    }
}

/** Raw SQLite row shape for skill_entries */
interface RawSkillRow {
    id: string;
    name: string;
    description: string;
    steps: string;
    tags: string;
    embedding: Buffer | null;
    version: number;
    parent_id: string | null;
    success_count: number;
    failure_count: number;
    total_latency_ms: number;
    invocation_count: number;
    created_at: string;
    updated_at: string;
    metadata: string;
}

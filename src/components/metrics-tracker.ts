import type Database from 'better-sqlite3';
import type {
    AgentMetrics,
    IMetricsTracker,
    QueryOutcome,
    SignalScores,
} from '../types.js';
import { generateId } from '../utils/id.js';
import type { Logger } from '../utils/logger.js';
import { defaultLogger } from '../utils/logger.js';
import { nowISO } from '../utils/time.js';

export class MetricsTracker implements IMetricsTracker {
    private readonly db: Database.Database;
    private readonly logger: Logger;

    constructor(db: Database.Database, logger: Logger = defaultLogger) {
        this.db = db;
        this.logger = logger;
    }

    recordQuery(entry: {
        id: string;
        queryText: string;
        routingDecision: string;
        signals: SignalScores;
        fusedScore: number;
        cost: number;
        latencyMs: number;
    }): void {
        const now = nowISO();

        this.db
            .prepare(
                `INSERT INTO query_log
                (id, query_text, routing_decision, signals, fused_score,
                 outcome, response_text, cost, latency_ms, created_at)
                VALUES (?, ?, ?, ?, ?, NULL, NULL, ?, ?, ?)`,
            )
            .run(
                entry.id,
                entry.queryText,
                entry.routingDecision,
                JSON.stringify(entry.signals),
                entry.fusedScore,
                entry.cost,
                entry.latencyMs,
                now,
            );

        this.logger.debug('MetricsTracker.recordQuery', { id: entry.id });
    }

    recordOutcome(queryId: string, outcome: QueryOutcome, responseText?: string): void {
        this.db
            .prepare(
                `UPDATE query_log
                 SET outcome = ?, response_text = ?
                 WHERE id = ?`,
            )
            .run(outcome, responseText ?? null, queryId);

        this.logger.debug('MetricsTracker.recordOutcome', { queryId, outcome });
    }

    getMetrics(): AgentMetrics {
        // Total queries
        const totalQueries = (
            this.db
                .prepare(`SELECT COUNT(*) as cnt FROM query_log`)
                .get() as { cnt: number }
        ).cnt;

        // Total escalations
        const totalEscalations = (
            this.db
                .prepare(
                    `SELECT COUNT(*) as cnt FROM query_log WHERE routing_decision = 'ESCALATE'`,
                )
                .get() as { cnt: number }
        ).cnt;

        // Local resolution rate
        const localResolutionRate =
            totalQueries > 0
                ? (totalQueries - totalEscalations) / totalQueries
                : 0;

        // Total knowledge entries
        const totalKnowledgeEntries = (
            this.db
                .prepare(`SELECT COUNT(*) as cnt FROM knowledge_entries`)
                .get() as { cnt: number }
        ).cnt;

        // Total skill entries
        const totalSkillEntries = (
            this.db
                .prepare(`SELECT COUNT(*) as cnt FROM skill_entries`)
                .get() as { cnt: number }
        ).cnt;

        // Knowledge growth rate (entries per hour)
        const knowledgeGrowthRate = this.computeKnowledgeGrowthRate();

        // Cumulative cost avoided
        const cumulativeCostAvoided = this.computeCostAvoided();

        // Self-test pass rate from verification_log
        const selfTestPassRate = this.computeSelfTestPassRate();

        // Confidence calibration
        const confidenceCalibration = this.computeConfidenceCalibration();

        return {
            localResolutionRate,
            knowledgeGrowthRate,
            cumulativeCostAvoided,
            selfTestPassRate,
            confidenceCalibration,
            totalQueries,
            totalEscalations,
            totalKnowledgeEntries,
            totalSkillEntries,
        };
    }

    // ── Private helpers ─────────────────────────────────────

    private computeKnowledgeGrowthRate(): number {
        // Compute entries added per hour based on first and last entry timestamps
        const range = this.db
            .prepare(
                `SELECT MIN(created_at) as first_at, MAX(created_at) as last_at, COUNT(*) as cnt
                 FROM knowledge_entries`,
            )
            .get() as { first_at: string | null; last_at: string | null; cnt: number };

        if (!range.first_at || !range.last_at || range.cnt <= 1) {
            return range.cnt;
        }

        const firstMs = new Date(range.first_at).getTime();
        const lastMs = new Date(range.last_at).getTime();
        const hoursElapsed = (lastMs - firstMs) / (1000 * 60 * 60);

        if (hoursElapsed <= 0) {
            return range.cnt;
        }

        return range.cnt / hoursElapsed;
    }

    private computeCostAvoided(): number {
        // Estimate cost avoided: for each LOCAL/HEDGE query, estimate what it would
        // have cost if escalated, using the average escalation cost
        const avgCost = this.db
            .prepare(
                `SELECT AVG(cost) as avg_cost FROM query_log
                 WHERE routing_decision = 'ESCALATE' AND cost > 0`,
            )
            .get() as { avg_cost: number | null };

        if (!avgCost.avg_cost) {
            return 0;
        }

        const localCount = (
            this.db
                .prepare(
                    `SELECT COUNT(*) as cnt FROM query_log
                     WHERE routing_decision IN ('LOCAL', 'HEDGE')`,
                )
                .get() as { cnt: number }
        ).cnt;

        return localCount * avgCost.avg_cost;
    }

    private computeSelfTestPassRate(): number {
        const row = this.db
            .prepare(
                `SELECT COUNT(*) as total,
                        SUM(CASE WHEN passed = 1 THEN 1 ELSE 0 END) as passed_count
                 FROM verification_log`,
            )
            .get() as { total: number; passed_count: number } | undefined;

        if (!row || row.total === 0) {
            return 0;
        }
        return row.passed_count / row.total;
    }

    private computeConfidenceCalibration(): number {
        // Percentage of routing decisions that led to successful outcomes
        const row = this.db
            .prepare(
                `SELECT COUNT(*) as total,
                        SUM(CASE WHEN outcome = 'success' THEN 1 ELSE 0 END) as success_count
                 FROM query_log
                 WHERE outcome IS NOT NULL`,
            )
            .get() as { total: number; success_count: number } | undefined;

        if (!row || row.total === 0) {
            return 0;
        }
        return row.success_count / row.total;
    }
}

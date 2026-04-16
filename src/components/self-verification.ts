import type Database from 'better-sqlite3';
import type {
    IKnowledgeStore,
    ILLMClient,
    ISelfVerificationSystem,
    KnowledgeEntry,
    VerificationResult,
} from '../types.js';
import { generateId } from '../utils/id.js';
import type { Logger } from '../utils/logger.js';
import { defaultLogger } from '../utils/logger.js';
import { nowISO } from '../utils/time.js';

export interface SelfVerificationConfig {
    enabled: boolean;
    intervalMs: number;
    batchSize: number;
    queryCountThreshold: number;
}

const DEFAULT_CONFIG: SelfVerificationConfig = {
    enabled: true,
    intervalMs: 86_400_000, // 24 hours
    batchSize: 20,
    queryCountThreshold: 50,
};

export class SelfVerificationSystem implements ISelfVerificationSystem {
    private readonly db: Database.Database;
    private readonly llmClient: ILLMClient;
    private readonly knowledgeStore: IKnowledgeStore;
    private readonly config: SelfVerificationConfig;
    private readonly logger: Logger;
    private queriesSinceLastVerification: number = 0;
    private lastVerificationTime: number = Date.now();

    constructor(
        llmClient: ILLMClient,
        knowledgeStore: IKnowledgeStore,
        db: Database.Database,
        config?: Partial<SelfVerificationConfig>,
        logger: Logger = defaultLogger,
    ) {
        this.llmClient = llmClient;
        this.knowledgeStore = knowledgeStore;
        this.db = db;
        this.config = { ...DEFAULT_CONFIG, ...config };
        this.logger = logger;
    }

    async runVerificationCycle(): Promise<VerificationResult> {
        const result: VerificationResult = {
            tested: 0,
            passed: 0,
            failed: 0,
            staleEntries: [],
        };

        if (!this.config.enabled) {
            return result;
        }

        // Select batch of LTM entries, oldest-verified-first
        const entries = this.selectBatch();
        if (entries.length === 0) {
            this.logger.debug('SelfVerificationSystem: no LTM entries to verify');
            return result;
        }

        for (const entry of entries) {
            result.tested++;

            try {
                const question = this.getOrGenerateQuestion(entry);
                const modelAnswer = await this.askLocalModel(question);
                const passed = await this.judgeContradiction(entry.content, modelAnswer);

                this.recordVerification(entry.id, question, modelAnswer, passed);

                if (passed) {
                    result.passed++;
                } else {
                    result.failed++;
                    result.staleEntries.push(entry.id);
                    this.flagAsStale(entry.id);
                }
            } catch (err) {
                this.logger.error(
                    'SelfVerificationSystem: verification failed for entry',
                    { entryId: entry.id, error: err instanceof Error ? err.message : String(err) },
                );
                result.failed++;
            }
        }

        // Reset counters
        this.queriesSinceLastVerification = 0;
        this.lastVerificationTime = Date.now();

        this.logger.info('SelfVerificationSystem: cycle complete', result);
        return result;
    }

    getPassRate(): number {
        const row = this.db
            .prepare(
                `SELECT
                    COUNT(*) as total,
                    SUM(CASE WHEN passed = 1 THEN 1 ELSE 0 END) as passed_count
                 FROM verification_log`,
            )
            .get() as { total: number; passed_count: number } | undefined;

        if (!row || row.total === 0) {
            return 0;
        }
        return row.passed_count / row.total;
    }

    shouldRunCycle(queryCount?: number): boolean {
        if (!this.config.enabled) return false;

        if (queryCount !== undefined) {
            this.queriesSinceLastVerification = queryCount;
        }

        // Time-based trigger
        const elapsed = Date.now() - this.lastVerificationTime;
        if (elapsed >= this.config.intervalMs) {
            return true;
        }

        // Query-count-based trigger
        if (this.queriesSinceLastVerification >= this.config.queryCountThreshold) {
            return true;
        }

        return false;
    }

    trackQuery(): void {
        this.queriesSinceLastVerification++;
    }

    getQueryCount(): number {
        return this.queriesSinceLastVerification;
    }

    // ── Private helpers ─────────────────────────────────────

    private selectBatch(): KnowledgeEntry[] {
        // Select LTM entries, oldest-verified-first
        // Entries that have never been verified come first (no entry in verification_log)
        const rows = this.db
            .prepare(
                `SELECT ke.id FROM knowledge_entries ke
                 WHERE ke.tier = 'LTM' AND ke.is_stale = 0
                 ORDER BY (
                     SELECT MAX(vl.created_at) FROM verification_log vl
                     WHERE vl.knowledge_id = ke.id
                 ) ASC NULLS FIRST
                 LIMIT ?`,
            )
            .all(this.config.batchSize) as { id: string }[];

        const entries: KnowledgeEntry[] = [];
        for (const row of rows) {
            const entry = this.knowledgeStore.get(row.id);
            if (entry) {
                entries.push(entry);
            }
        }
        return entries;
    }

    private getOrGenerateQuestion(entry: KnowledgeEntry): string {
        if (entry.selfTestQuestions && entry.selfTestQuestions.length > 0) {
            // Pick a random existing question
            const idx = Math.floor(Math.random() * entry.selfTestQuestions.length);
            return entry.selfTestQuestions[idx];
        }
        // Generate a default question
        return `Is the following true? ${entry.content}`;
    }

    private async askLocalModel(question: string): Promise<string> {
        const response = await this.llmClient.chat([
            {
                role: 'system',
                content: 'Answer the following question concisely and accurately.',
            },
            { role: 'user', content: question },
        ]);
        return response.content;
    }

    private async judgeContradiction(
        storedFact: string,
        modelAnswer: string,
    ): Promise<boolean> {
        const response = await this.llmClient.chat([
            {
                role: 'system',
                content:
                    'You are a fact-checking judge. Compare the stored fact with the model answer. ' +
                    'Respond with ONLY "CONFIRMS" if the answer confirms the fact, or "CONTRADICTS" if it contradicts the fact.',
            },
            {
                role: 'user',
                content: `Stored fact: ${storedFact}\n\nModel answer: ${modelAnswer}`,
            },
        ]);

        const verdict = response.content.trim().toUpperCase();
        return verdict.includes('CONFIRMS');
    }

    private recordVerification(
        knowledgeId: string,
        question: string,
        modelAnswer: string,
        passed: boolean,
    ): void {
        const id = generateId();
        const now = nowISO();

        this.db
            .prepare(
                `INSERT INTO verification_log
                (id, knowledge_id, question, model_answer, passed, created_at)
                VALUES (?, ?, ?, ?, ?, ?)`,
            )
            .run(id, knowledgeId, question, modelAnswer, passed ? 1 : 0, now);
    }

    private flagAsStale(entryId: string): void {
        this.db
            .prepare(
                `UPDATE knowledge_entries SET is_stale = 1 WHERE id = ?`,
            )
            .run(entryId);

        this.logger.info('SelfVerificationSystem: flagged entry as stale', { entryId });
    }
}

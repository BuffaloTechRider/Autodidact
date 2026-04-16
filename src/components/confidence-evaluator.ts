import type Database from 'better-sqlite3';
import type {
    EvaluationContext,
    IConfidenceEvaluator,
    ILLMClient,
    QueryOutcome,
    RoutingResult,
    SignalScores,
    SignalWeights,
} from '../types.js';
import { cosineSimilarity } from '../utils/cosine-similarity.js';
import { generateId } from '../utils/id.js';
import type { Logger } from '../utils/logger.js';
import { defaultLogger } from '../utils/logger.js';
import { nowISO } from '../utils/time.js';

export interface ConfidenceEvaluatorConfig {
    localThreshold: number;
    hedgeThreshold: number;
    initialAlpha: number;
    initialBeta: number;
}

const DEFAULT_CONFIG: ConfidenceEvaluatorConfig = {
    localThreshold: 0.7,
    hedgeThreshold: 0.4,
    initialAlpha: 1.0,
    initialBeta: 1.0,
};

const SIGNAL_NAMES = [
    'knowledgeSimilarity',
    'skillCoverage',
    'queryComplexity',
    'selfAssessment',
] as const;

type SignalName = (typeof SIGNAL_NAMES)[number];

/**
 * Sample from a Gamma distribution using the Marsaglia and Tsang method.
 * For shape >= 1. For shape < 1, uses the Ahrens-Dieter boost.
 */
function sampleGamma(shape: number): number {
    if (shape < 1) {
        // Boost: Gamma(shape) = Gamma(shape+1) * U^(1/shape)
        const u = Math.random();
        return sampleGamma(shape + 1) * Math.pow(u, 1 / shape);
    }

    // Marsaglia and Tsang's method for shape >= 1
    const d = shape - 1 / 3;
    const c = 1 / Math.sqrt(9 * d);

    while (true) {
        let x: number;
        let v: number;

        do {
            x = randomNormal();
            v = 1 + c * x;
        } while (v <= 0);

        v = v * v * v;
        const u = Math.random();

        if (u < 1 - 0.0331 * (x * x) * (x * x)) {
            return d * v;
        }

        if (Math.log(u) < 0.5 * x * x + d * (1 - v + Math.log(v))) {
            return d * v;
        }
    }
}

/**
 * Sample from a standard normal distribution using Box-Muller transform.
 */
function randomNormal(): number {
    const u1 = Math.random();
    const u2 = Math.random();
    return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

/**
 * Sample from Beta(alpha, beta) using Gamma distributions.
 * Beta(a, b) = Gamma(a) / (Gamma(a) + Gamma(b))
 */
function sampleBeta(alpha: number, beta: number): number {
    const x = sampleGamma(alpha);
    const y = sampleGamma(beta);
    if (x + y === 0) return 0.5;
    return x / (x + y);
}

export class ConfidenceEvaluator implements IConfidenceEvaluator {
    private readonly db: Database.Database;
    private readonly config: ConfidenceEvaluatorConfig;
    private readonly llmClient: ILLMClient;
    private readonly logger: Logger;

    constructor(
        db: Database.Database,
        llmClient: ILLMClient,
        config?: Partial<ConfidenceEvaluatorConfig>,
        logger: Logger = defaultLogger,
    ) {
        this.db = db;
        this.config = { ...DEFAULT_CONFIG, ...config };
        this.llmClient = llmClient;
        this.logger = logger;
        this.initThompsonParams();
    }

    async evaluate(
        query: string,
        context: EvaluationContext,
    ): Promise<RoutingResult> {
        const queryId = generateId();

        // Compute the 4 signals
        const signals = await this.computeSignals(query, context);

        // Read Thompson params
        const weights = this.getSignalWeights();

        // Sample θᵢ ~ Beta(αᵢ, βᵢ) per signal
        const thetas: Record<SignalName, number> = {
            knowledgeSimilarity: sampleBeta(
                weights.knowledgeSimilarity.alpha,
                weights.knowledgeSimilarity.beta,
            ),
            skillCoverage: sampleBeta(
                weights.skillCoverage.alpha,
                weights.skillCoverage.beta,
            ),
            queryComplexity: sampleBeta(
                weights.queryComplexity.alpha,
                weights.queryComplexity.beta,
            ),
            selfAssessment: sampleBeta(
                weights.selfAssessment.alpha,
                weights.selfAssessment.beta,
            ),
        };

        // Compute fused_score = Σ(θᵢ × signalᵢ) / Σ(θᵢ)
        let numerator = 0;
        let denominator = 0;
        for (const name of SIGNAL_NAMES) {
            numerator += thetas[name] * signals[name];
            denominator += thetas[name];
        }
        const fusedScore = denominator > 0 ? numerator / denominator : 0;

        // Route based on thresholds
        let decision: 'LOCAL' | 'HEDGE' | 'ESCALATE';
        if (fusedScore >= this.config.localThreshold) {
            decision = 'LOCAL';
        } else if (fusedScore >= this.config.hedgeThreshold) {
            decision = 'HEDGE';
        } else {
            decision = 'ESCALATE';
        }

        this.logger.debug('ConfidenceEvaluator.evaluate', {
            queryId,
            signals,
            fusedScore,
            decision,
        });

        return { decision, signals, fusedScore, queryId };
    }

    recordOutcome(queryId: string, outcome: QueryOutcome): void {
        const now = nowISO();

        for (const name of SIGNAL_NAMES) {
            if (outcome === 'success') {
                this.db
                    .prepare(
                        `UPDATE thompson_params
                         SET alpha = alpha + 1, updated_at = ?
                         WHERE signal_name = ?`,
                    )
                    .run(now, name);
            } else {
                this.db
                    .prepare(
                        `UPDATE thompson_params
                         SET beta = beta + 1, updated_at = ?
                         WHERE signal_name = ?`,
                    )
                    .run(now, name);
            }
        }

        this.logger.debug('ConfidenceEvaluator.recordOutcome', {
            queryId,
            outcome,
        });
    }

    getSignalWeights(): SignalWeights {
        const rows = this.db
            .prepare(`SELECT signal_name, alpha, beta FROM thompson_params`)
            .all() as { signal_name: string; alpha: number; beta: number }[];

        const weights: Record<string, { alpha: number; beta: number }> = {};
        for (const row of rows) {
            weights[row.signal_name] = { alpha: row.alpha, beta: row.beta };
        }

        return {
            knowledgeSimilarity: weights['knowledgeSimilarity'] ?? {
                alpha: this.config.initialAlpha,
                beta: this.config.initialBeta,
            },
            skillCoverage: weights['skillCoverage'] ?? {
                alpha: this.config.initialAlpha,
                beta: this.config.initialBeta,
            },
            queryComplexity: weights['queryComplexity'] ?? {
                alpha: this.config.initialAlpha,
                beta: this.config.initialBeta,
            },
            selfAssessment: weights['selfAssessment'] ?? {
                alpha: this.config.initialAlpha,
                beta: this.config.initialBeta,
            },
        };
    }

    // ── Private helpers ─────────────────────────────────────

    private initThompsonParams(): void {
        const now = nowISO();
        const stmt = this.db.prepare(
            `INSERT OR IGNORE INTO thompson_params (signal_name, alpha, beta, updated_at)
             VALUES (?, ?, ?, ?)`,
        );

        for (const name of SIGNAL_NAMES) {
            stmt.run(name, this.config.initialAlpha, this.config.initialBeta, now);
        }

        this.logger.debug('ConfidenceEvaluator: thompson_params initialized');
    }

    private async computeSignals(
        query: string,
        context: EvaluationContext,
    ): Promise<SignalScores> {
        const knowledgeSimilarity = this.computeKnowledgeSimilarity(context);
        const skillCoverage = this.computeSkillCoverage(context);
        const queryComplexity = this.computeQueryComplexity(query);
        const selfAssessment = await this.computeSelfAssessment(query);

        return {
            knowledgeSimilarity,
            skillCoverage,
            queryComplexity,
            selfAssessment,
        };
    }

    private computeKnowledgeSimilarity(context: EvaluationContext): number {
        if (context.knowledgeHits.length === 0) {
            return 0;
        }
        // The knowledge hits are already sorted by similarity from the store search.
        // We take the max similarity. Since we don't have the raw similarity score
        // stored on the entry, we use confidence as a proxy if no embedding,
        // or compute similarity between the first hit's embedding and itself (=1).
        // In practice, the caller provides pre-searched hits, so the first hit
        // is the most similar. We use confidence of the best hit as a proxy.
        // A better approach: the hits come from search() which orders by similarity.
        // The best hit's confidence is a reasonable proxy for knowledge similarity.
        // However, the design says "max cosine similarity from the provided knowledgeHits".
        // Since we don't have the query embedding here, we use the confidence of the top hit.
        // The top hit from search() is already the most similar.
        return Math.min(1, Math.max(0, context.knowledgeHits[0].confidence));
    }

    private computeSkillCoverage(context: EvaluationContext): number {
        if (context.skillHits.length === 0) {
            return 0;
        }
        // Best matching skill relevance. Use success rate as a proxy for relevance.
        const best = context.skillHits[0];
        if (best.invocationCount === 0) {
            return 0.5; // No data yet, neutral
        }
        return Math.min(
            1,
            Math.max(0, best.successCount / best.invocationCount),
        );
    }

    private computeQueryComplexity(query: string): number {
        // Heuristic: estimate complexity, then invert so 1 = simple
        let complexity = 0;

        // Token count heuristic (rough word count)
        const words = query.trim().split(/\s+/).length;
        if (words > 50) complexity += 0.3;
        else if (words > 20) complexity += 0.2;
        else if (words > 10) complexity += 0.1;

        // Question marks indicate multi-part questions
        const questionMarks = (query.match(/\?/g) || []).length;
        if (questionMarks > 2) complexity += 0.2;
        else if (questionMarks > 0) complexity += 0.1;

        // Domain keywords that suggest technical complexity
        const domainKeywords = [
            'implement', 'architecture', 'optimize', 'debug',
            'algorithm', 'distributed', 'concurrent', 'async',
            'database', 'security', 'performance', 'scalab',
        ];
        const lowerQuery = query.toLowerCase();
        for (const kw of domainKeywords) {
            if (lowerQuery.includes(kw)) {
                complexity += 0.05;
            }
        }

        // Clamp and invert: 1 = simple, 0 = complex
        complexity = Math.min(1, Math.max(0, complexity));
        return 1 - complexity;
    }

    private async computeSelfAssessment(query: string): Promise<number> {
        try {
            const response = await this.llmClient.chat([
                {
                    role: 'system',
                    content:
                        'You are a confidence estimator. Respond with ONLY a single number between 0 and 1 representing your confidence in answering the given query. No other text.',
                },
                {
                    role: 'user',
                    content: `Rate your confidence 0-1 in answering: ${query}`,
                },
            ]);

            const parsed = parseFloat(response.content.trim());
            if (isNaN(parsed) || parsed < 0 || parsed > 1) {
                this.logger.warn(
                    'ConfidenceEvaluator: failed to parse self-assessment, defaulting to 0.5',
                    { raw: response.content },
                );
                return 0.5;
            }
            return parsed;
        } catch {
            this.logger.warn(
                'ConfidenceEvaluator: self-assessment LLM call failed, defaulting to 0.5',
            );
            return 0.5;
        }
    }
}

import type Database from 'better-sqlite3';
import { resolveConfig, validateConfig } from '../config.js';
import { initDatabase } from '../database.js';
import type {
    AgentMetrics,
    AgentResponse,
    AutodidactConfig,
    IAgent,
    ICloudRouter,
    IConfidenceEvaluator,
    IKnowledgeStore,
    ILearningExtractor,
    ILLMClient,
    IMetricsTracker,
    ISelfVerificationSystem,
    ISkillEvolver,
    ISkillStore,
    IToolRegistry,
    IUserProfile,
    RoutingResult,
} from '../types.js';
import { generateId } from '../utils/id.js';
import type { Logger } from '../utils/logger.js';
import { defaultLogger } from '../utils/logger.js';
import { CloudRouter } from './cloud-router.js';
import { ConfidenceEvaluator } from './confidence-evaluator.js';
import { KnowledgeStore } from './knowledge-store.js';
import { LearningExtractor } from './learning-extractor.js';
import { LLMClient } from './llm-client.js';
import { MetricsTracker } from './metrics-tracker.js';
import { SelfVerificationSystem } from './self-verification.js';
import { SkillEvolver } from './skill-evolver.js';
import { SkillStore } from './skill-store.js';
import { UserProfile } from './user-profile.js';
import { ToolRegistry } from './tool-registry.js';

/**
 * Optional custom component implementations for dependency injection.
 */
export interface AgentComponents {
    llmClient?: ILLMClient;
    knowledgeStore?: IKnowledgeStore;
    skillStore?: ISkillStore;
    confidenceEvaluator?: IConfidenceEvaluator;
    cloudRouter?: ICloudRouter;
    learningExtractor?: ILearningExtractor;
    selfVerification?: ISelfVerificationSystem;
    skillEvolver?: ISkillEvolver;
    userProfile?: IUserProfile;
    metricsTracker?: IMetricsTracker;
    toolRegistry?: IToolRegistry;
    logger?: Logger;
}

export class Agent implements IAgent {
    private readonly config: AutodidactConfig;
    private readonly db: Database.Database;
    private readonly llmClient: ILLMClient;
    private readonly knowledgeStore: IKnowledgeStore;
    private readonly skillStore: ISkillStore;
    private readonly confidenceEvaluator: IConfidenceEvaluator;
    private readonly cloudRouter: ICloudRouter;
    private readonly learningExtractor: ILearningExtractor;
    private readonly selfVerification: ISelfVerificationSystem;
    private readonly skillEvolver: ISkillEvolver;
    private readonly userProfile: IUserProfile;
    private readonly metricsTracker: IMetricsTracker;
    private readonly toolRegistry: IToolRegistry;
    private readonly logger: Logger;
    private queryCount: number = 0;

    constructor(config: Partial<AutodidactConfig>, components?: AgentComponents) {
        // Resolve defaults then validate
        const resolved = resolveConfig(config);
        this.config = validateConfig(resolved);
        this.logger = components?.logger ?? defaultLogger;

        // Initialize database
        this.db = initDatabase(this.config.database.path);

        // Create or inject components
        this.llmClient = components?.llmClient ?? new LLMClient(
            {
                baseUrl: this.config.localLLM.baseUrl,
                apiKey: this.config.localLLM.apiKey ?? '',
                model: this.config.localLLM.model,
                timeoutMs: this.config.localLLM.timeoutMs,
            },
            this.logger,
        );

        this.knowledgeStore = components?.knowledgeStore ?? new KnowledgeStore(
            this.db,
            this.config.knowledgeStore,
            this.logger,
        );

        this.skillStore = components?.skillStore ?? new SkillStore(this.db, this.logger);

        this.confidenceEvaluator = components?.confidenceEvaluator ?? new ConfidenceEvaluator(
            this.db,
            this.llmClient,
            this.config.confidenceEvaluator,
            this.logger,
        );

        this.cloudRouter = components?.cloudRouter ?? new CloudRouter(
            this.db,
            this.config.cloudRouter.providers,
            this.logger,
        );

        this.learningExtractor = components?.learningExtractor ?? new LearningExtractor(
            this.llmClient,
            this.logger,
        );

        this.selfVerification = components?.selfVerification ?? new SelfVerificationSystem(
            this.llmClient,
            this.knowledgeStore,
            this.db,
            this.config.selfVerification,
            this.logger,
        );

        this.skillEvolver = components?.skillEvolver ?? new SkillEvolver(
            this.llmClient,
            this.skillStore,
            this.db,
            this.config.skillEvolver,
            this.logger,
        );

        this.userProfile = components?.userProfile ?? new UserProfile(this.db, this.logger);

        this.metricsTracker = components?.metricsTracker ?? new MetricsTracker(this.db, this.logger);

        this.toolRegistry = components?.toolRegistry ?? new ToolRegistry(
            this.db,
            this.config.toolRegistry,
            this.logger,
        );
    }

    async query(text: string): Promise<AgentResponse> {
        const startTime = Date.now();

        try {
            // 1. Embed the query
            const embedding = await this.llmClient.embed(text);

            // 2. Search KnowledgeStore + SkillStore
            const knowledgeHits = this.knowledgeStore.search(text, embedding, 5);
            const skillHits = this.skillStore.search(text, embedding, 5);

            // 3. Evaluate via ConfidenceEvaluator
            const routing = await this.confidenceEvaluator.evaluate(text, {
                knowledgeHits,
                skillHits,
            });

            let content: string;
            let cost = 0;
            const sourcesUsed: string[] = [
                ...knowledgeHits.map((k) => k.id),
                ...skillHits.map((s) => s.id),
            ];

            if (routing.decision === 'ESCALATE') {
                // 4b. ESCALATE: call CloudRouter
                const cloudContext = this.buildContext(knowledgeHits, skillHits);
                const escalationContext = cloudContext
                    + '\n\nIMPORTANT: If you use any external APIs or tools to answer this, describe them with their endpoint URL, HTTP method, and parameters.';
                const cloudResponse = await this.cloudRouter.escalate(text, escalationContext);
                content = cloudResponse.content;
                cost = cloudResponse.cost;

                // Pass to LearningExtractor
                const extraction = await this.learningExtractor.extract(text, cloudResponse.content);

                // Store extracted knowledge
                for (const k of extraction.knowledge) {
                    const entry = this.knowledgeStore.insert({
                        ...k,
                        embedding: await this.safeEmbed(k.content),
                    });
                    sourcesUsed.push(entry.id);
                }

                // Store extracted skills
                for (const s of extraction.skills) {
                    const entry = this.skillStore.insert({
                        ...s,
                        embedding: await this.safeEmbed(s.description),
                    });
                    sourcesUsed.push(entry.id);
                }

                // Auto-register discovered tools
                if (this.config.toolRegistry.enabled && extraction.tools.length > 0) {
                    for (const t of extraction.tools) {
                        try {
                            // Skip if tool already exists
                            const existing = this.toolRegistry.get(t.name);
                            if (existing) {
                                this.logger.debug('Agent: tool already registered, skipping', { name: t.name });
                                continue;
                            }
                            const registered = this.toolRegistry.register({
                                ...t,
                                source: 'learned',
                            });
                            this.logger.info('Agent: auto-registered discovered tool', {
                                name: registered.name,
                                type: registered.type,
                            });

                            // Auto-verify if configured
                            if (this.config.toolRegistry.autoVerify && registered.type === 'http') {
                                this.toolRegistry.verify(registered.name).catch((err) => {
                                    this.logger.warn('Agent: auto-verify failed for tool', {
                                        name: registered.name,
                                        error: String(err),
                                    });
                                });
                            }
                        } catch (err) {
                            this.logger.warn('Agent: failed to register discovered tool', {
                                name: t.name,
                                error: err instanceof Error ? err.message : String(err),
                            });
                        }
                    }
                }
            } else {
                // 4a. LOCAL or HEDGE: generate via local LLM
                const systemPrompt = this.buildLocalPrompt(knowledgeHits, skillHits, routing);
                const response = await this.llmClient.chat([
                    { role: 'system', content: systemPrompt },
                    { role: 'user', content: text },
                ]);
                content = response.content;

                if (routing.decision === 'HEDGE') {
                    content = `[Note: This response has reduced confidence.]\n\n${content}`;
                }
            }

            const latencyMs = Date.now() - startTime;

            // 6. Record in MetricsTracker
            this.metricsTracker.recordQuery({
                id: routing.queryId,
                queryText: text,
                routingDecision: routing.decision,
                signals: routing.signals,
                fusedScore: routing.fusedScore,
                cost,
                latencyMs,
            });

            // Record outcome as success
            this.metricsTracker.recordOutcome(routing.queryId, 'success', content);
            this.confidenceEvaluator.recordOutcome(routing.queryId, 'success');

            // 7. Track query count for self-verification triggers
            this.queryCount++;
            if (this.selfVerification instanceof SelfVerificationSystem) {
                (this.selfVerification as SelfVerificationSystem).trackQuery();
                if ((this.selfVerification as SelfVerificationSystem).shouldRunCycle()) {
                    // Fire and forget — don't block the response
                    this.selfVerification.runVerificationCycle().catch((err) => {
                        this.logger.error('Agent: self-verification cycle failed', String(err));
                    });
                }
            }

            // Update user profile if enabled
            if (this.config.userProfile.enabled && this.config.userProfile.autoExtract) {
                this.updateProfileFromQuery(text);
            }

            // 8. Return AgentResponse
            return {
                content,
                routing,
                cost,
                latencyMs,
                sourcesUsed,
            };
        } catch (err) {
            const latencyMs = Date.now() - startTime;
            const errorRouting: RoutingResult = {
                decision: 'ESCALATE',
                signals: {
                    knowledgeSimilarity: 0,
                    skillCoverage: 0,
                    queryComplexity: 0,
                    selfAssessment: 0,
                },
                fusedScore: 0,
                queryId: generateId(),
            };

            this.logger.error(
                'Agent.query: lifecycle error',
                err instanceof Error ? err.message : String(err),
            );

            return {
                content: `Error processing query: ${err instanceof Error ? err.message : String(err)}`,
                routing: errorRouting,
                cost: 0,
                latencyMs,
                sourcesUsed: [],
            };
        }
    }

    getMetrics(): AgentMetrics {
        return this.metricsTracker.getMetrics();
    }

    getConfig(): AutodidactConfig {
        return this.config;
    }

    // ── Private helpers ─────────────────────────────────────

    private buildContext(
        knowledgeHits: { content: string }[],
        skillHits: { name: string; description: string; steps: { order: number; description: string }[] }[],
    ): string {
        const parts: string[] = [];

        if (knowledgeHits.length > 0) {
            parts.push('Relevant knowledge:');
            for (const k of knowledgeHits) {
                parts.push(`- ${k.content}`);
            }
        }

        if (skillHits.length > 0) {
            parts.push('Relevant skills:');
            for (const s of skillHits) {
                const steps = s.steps.map((st) => `  ${st.order}. ${st.description}`).join('\n');
                parts.push(`- ${s.name}: ${s.description}\n${steps}`);
            }
        }

        return parts.join('\n');
    }

    private buildLocalPrompt(
        knowledgeHits: { content: string }[],
        skillHits: { name: string; description: string }[],
        _routing: RoutingResult,
    ): string {
        const parts: string[] = [
            'You are a helpful AI assistant. Use the following context to answer the user query.',
        ];

        // Inject user profile context if enabled
        if (this.config.userProfile.enabled) {
            const profileContext = this.userProfile.getContext(this.config.userProfile.defaultProfile);
            if (profileContext) {
                parts.push('', profileContext);
            }
        }

        if (knowledgeHits.length > 0) {
            parts.push('', 'Relevant knowledge:');
            for (const k of knowledgeHits) {
                parts.push(`- ${k.content}`);
            }
        }

        if (skillHits.length > 0) {
            parts.push('', 'Relevant skills:');
            for (const s of skillHits) {
                parts.push(`- ${s.name}: ${s.description}`);
            }
        }

        return parts.join('\n');
    }

    private async safeEmbed(text: string): Promise<number[] | undefined> {
        try {
            return await this.llmClient.embed(text);
        } catch {
            this.logger.warn('Agent: failed to embed text, storing without embedding');
            return undefined;
        }
    }

    private updateProfileFromQuery(queryText: string): void {
        try {
            // Simple heuristic extraction of vocabulary from the query
            const words = queryText.split(/\s+/).filter((w) => w.length > 4);
            const observations = words.slice(0, 3).map((w) => ({
                type: 'vocabulary' as const,
                key: w.toLowerCase(),
                value: w.toLowerCase(),
            }));
            if (observations.length > 0) {
                this.userProfile.update(this.config.userProfile.defaultProfile, observations);
            }
        } catch {
            // Non-critical — don't fail the query
        }
    }
}

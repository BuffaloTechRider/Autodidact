import type Database from 'better-sqlite3';
import type {
    ILLMClient,
    ISkillEvolver,
    ISkillStore,
    SkillEntry,
    SkillReviewResult,
    SkillStep,
} from '../types.js';
import { generateId } from '../utils/id.js';
import type { Logger } from '../utils/logger.js';
import { defaultLogger } from '../utils/logger.js';
import { nowISO } from '../utils/time.js';

export interface SkillEvolverConfig {
    enabled: boolean;
    reviewThreshold: number;
    minSuccessRate: number;
}

const DEFAULT_CONFIG: SkillEvolverConfig = {
    enabled: true,
    reviewThreshold: 10,
    minSuccessRate: 0.6,
};

const EVOLUTION_PROMPT = `You are a skill optimization system. Analyze the skill's performance and rewrite its steps to improve success rate.

Respond with ONLY valid JSON in this exact format:
{
  "steps": [
    { "order": 1, "description": "step description", "input": "what is needed", "output": "what is produced" }
  ],
  "reason": "brief explanation of what was changed and why"
}

Rules:
- Each step must have order, description, input, and output fields.
- Maintain the skill's original purpose.
- Address the failure patterns described.
- Keep the number of steps reasonable.`;

interface RawEvolution {
    steps?: {
        order?: number;
        description?: string;
        input?: string;
        output?: string;
        toolName?: string;
    }[];
    reason?: string;
}

export class SkillEvolver implements ISkillEvolver {
    private readonly db: Database.Database;
    private readonly llmClient: ILLMClient;
    private readonly skillStore: ISkillStore;
    private readonly config: SkillEvolverConfig;
    private readonly logger: Logger;

    constructor(
        llmClient: ILLMClient,
        skillStore: ISkillStore,
        db: Database.Database,
        config?: Partial<SkillEvolverConfig>,
        logger: Logger = defaultLogger,
    ) {
        this.llmClient = llmClient;
        this.skillStore = skillStore;
        this.db = db;
        this.config = { ...DEFAULT_CONFIG, ...config };
        this.logger = logger;
    }

    async reviewSkill(skillId: string): Promise<SkillReviewResult> {
        const skill = this.skillStore.get(skillId);
        if (!skill) {
            return {
                skillId,
                skillName: 'unknown',
                previousVersion: 0,
                action: 'failed',
                reason: 'Skill not found',
            };
        }

        const successRate =
            skill.invocationCount > 0
                ? skill.successCount / skill.invocationCount
                : 1;

        // Check if skill needs evolution
        if (
            skill.invocationCount < this.config.reviewThreshold &&
            successRate >= this.config.minSuccessRate
        ) {
            const result: SkillReviewResult = {
                skillId,
                skillName: skill.name,
                previousVersion: skill.version,
                action: 'kept',
                reason: `Skill does not meet review criteria (invocations: ${skill.invocationCount}, successRate: ${successRate.toFixed(2)})`,
            };
            this.logEvolution(result);
            return result;
        }

        // Attempt evolution via LLM
        try {
            const newSteps = await this.generateEvolvedSteps(skill, successRate);
            if (!newSteps) {
                const result: SkillReviewResult = {
                    skillId,
                    skillName: skill.name,
                    previousVersion: skill.version,
                    action: 'failed',
                    reason: 'LLM failed to generate valid revised steps',
                };
                this.logEvolution(result);
                return result;
            }

            // Create new version in SkillStore
            const newEntry = this.skillStore.insert({
                name: skill.name,
                description: skill.description,
                steps: newSteps,
                tags: skill.tags,
                embedding: skill.embedding ?? undefined,
                metadata: {
                    ...skill.metadata,
                    parentId: skill.id,
                    evolvedFrom: skill.version,
                },
            });

            // Update the new entry's parent_id and version in DB directly
            const newVersion = skill.version + 1;
            this.db
                .prepare(
                    `UPDATE skill_entries SET version = ?, parent_id = ? WHERE id = ?`,
                )
                .run(newVersion, skill.id, newEntry.id);

            const result: SkillReviewResult = {
                skillId,
                skillName: skill.name,
                previousVersion: skill.version,
                action: 'evolved',
                newVersion,
                reason: `Evolved due to ${successRate < this.config.minSuccessRate ? 'low success rate' : 'invocation threshold reached'} (successRate: ${successRate.toFixed(2)}, invocations: ${skill.invocationCount})`,
            };
            this.logEvolution(result);
            return result;
        } catch (err) {
            this.logger.error(
                'SkillEvolver.reviewSkill: evolution failed',
                { skillId, error: err instanceof Error ? err.message : String(err) },
            );
            const result: SkillReviewResult = {
                skillId,
                skillName: skill.name,
                previousVersion: skill.version,
                action: 'failed',
                reason: `Evolution error: ${err instanceof Error ? err.message : String(err)}`,
            };
            this.logEvolution(result);
            return result;
        }
    }

    async checkAndEvolve(): Promise<SkillReviewResult[]> {
        if (!this.config.enabled) {
            return [];
        }

        const results: SkillReviewResult[] = [];

        // Find all skills due for review
        const rows = this.db
            .prepare(
                `SELECT id, invocation_count, success_count, failure_count
                 FROM skill_entries`,
            )
            .all() as {
                id: string;
                invocation_count: number;
                success_count: number;
                failure_count: number;
            }[];

        for (const row of rows) {
            const successRate =
                row.invocation_count > 0
                    ? row.success_count / row.invocation_count
                    : 1;

            const needsReview =
                row.invocation_count >= this.config.reviewThreshold ||
                (row.invocation_count > 0 && successRate < this.config.minSuccessRate);

            if (needsReview) {
                const result = await this.reviewSkill(row.id);
                results.push(result);
            }
        }

        return results;
    }

    // ── Private helpers ─────────────────────────────────────

    private async generateEvolvedSteps(
        skill: SkillEntry,
        successRate: number,
    ): Promise<SkillStep[] | null> {
        const currentSteps = skill.steps
            .map((s) => `${s.order}. ${s.description} (Input: ${s.input} → Output: ${s.output})`)
            .join('\n');

        const response = await this.llmClient.chat([
            { role: 'system', content: EVOLUTION_PROMPT },
            {
                role: 'user',
                content: `Skill: ${skill.name}\nDescription: ${skill.description}\nSuccess rate: ${(successRate * 100).toFixed(1)}%\nInvocation count: ${skill.invocationCount}\n\nCurrent steps:\n${currentSteps}\n\nRewrite the steps to improve success rate.`,
            },
        ]);

        return this.parseSteps(response.content);
    }

    private parseSteps(content: string): SkillStep[] | null {
        try {
            const jsonMatch = content.match(/```(?:json)?\s*([\s\S]*?)```/);
            const jsonStr = jsonMatch ? jsonMatch[1].trim() : content.trim();
            const parsed = JSON.parse(jsonStr) as RawEvolution;

            if (!Array.isArray(parsed.steps) || parsed.steps.length === 0) {
                return null;
            }

            const steps: SkillStep[] = [];
            for (const s of parsed.steps) {
                if (!s.description || s.input === undefined || s.output === undefined) {
                    continue;
                }
                steps.push({
                    order: typeof s.order === 'number' ? s.order : steps.length + 1,
                    description: s.description,
                    input: s.input,
                    output: s.output,
                    ...(s.toolName ? { toolName: s.toolName } : {}),
                });
            }

            return steps.length > 0 ? steps : null;
        } catch {
            this.logger.error('SkillEvolver.parseSteps: JSON parse failed');
            return null;
        }
    }

    private logEvolution(result: SkillReviewResult): void {
        const id = generateId();
        const now = nowISO();

        this.db
            .prepare(
                `INSERT INTO skill_evolution_log
                (id, skill_id, skill_name, previous_version, new_version, action, reason, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)`,
            )
            .run(
                id,
                result.skillId,
                result.skillName,
                result.previousVersion,
                result.newVersion ?? null,
                result.action,
                result.reason,
                now,
            );
    }
}

import type {
    ExtractionResult,
    ILearningExtractor,
    ILLMClient,
    NewKnowledgeEntry,
    NewSkillEntry,
    SelfTestQuestion,
} from '../types.js';
import type { Logger } from '../utils/logger.js';
import { defaultLogger } from '../utils/logger.js';

const EXTRACTION_PROMPT = `You are a knowledge extraction system. Analyze the following cloud LLM response to a user query and extract:

1. **Knowledge entries**: Factual claims or information that can be reused.
2. **Skill entries**: Step-by-step procedures or reasoning patterns.
3. **Self-test questions**: At least one question per knowledge entry to verify it later.

Respond with ONLY valid JSON in this exact format:
{
  "knowledge": [
    {
      "content": "the factual claim",
      "tags": ["tag1", "tag2"],
      "confidence": 0.8
    }
  ],
  "skills": [
    {
      "name": "skill_name",
      "description": "what this skill does",
      "steps": [
        { "order": 1, "description": "step description", "input": "what is needed", "output": "what is produced" }
      ],
      "tags": ["tag1"]
    }
  ],
  "selfTestQuestions": [
    { "knowledgeIndex": 0, "question": "A question to verify the knowledge entry" }
  ]
}

Rules:
- Each knowledge entry must have content, tags, and confidence (0-1).
- Each skill must have at least one step with order, description, input, and output.
- Generate at least one self-test question per knowledge entry.
- knowledgeIndex refers to the zero-based index in the knowledge array.`;

interface RawExtraction {
    knowledge?: {
        content?: string;
        tags?: string[];
        confidence?: number;
    }[];
    skills?: {
        name?: string;
        description?: string;
        steps?: {
            order?: number;
            description?: string;
            input?: string;
            output?: string;
            toolName?: string;
        }[];
        tags?: string[];
    }[];
    selfTestQuestions?: {
        knowledgeIndex?: number;
        question?: string;
    }[];
}

export class LearningExtractor implements ILearningExtractor {
    private readonly llmClient: ILLMClient;
    private readonly logger: Logger;

    constructor(llmClient: ILLMClient, logger: Logger = defaultLogger) {
        this.llmClient = llmClient;
        this.logger = logger;
    }

    async extract(query: string, response: string): Promise<ExtractionResult> {
        const empty: ExtractionResult = { knowledge: [], skills: [], selfTestQuestions: [] };

        try {
            const llmResponse = await this.llmClient.chat([
                { role: 'system', content: EXTRACTION_PROMPT },
                {
                    role: 'user',
                    content: `User query: ${query}\n\nCloud response:\n${response}`,
                },
            ]);

            const parsed = this.parseResponse(llmResponse.content);
            if (!parsed) {
                this.logger.error('LearningExtractor.extract: failed to parse LLM output');
                return empty;
            }

            return this.buildResult(parsed);
        } catch (err) {
            this.logger.error(
                'LearningExtractor.extract: extraction failed',
                err instanceof Error ? err.message : String(err),
            );
            return empty;
        }
    }

    private parseResponse(content: string): RawExtraction | null {
        try {
            // Try to extract JSON from the response (may be wrapped in markdown code blocks)
            const jsonMatch = content.match(/```(?:json)?\s*([\s\S]*?)```/);
            const jsonStr = jsonMatch ? jsonMatch[1].trim() : content.trim();
            return JSON.parse(jsonStr) as RawExtraction;
        } catch {
            this.logger.error('LearningExtractor.parseResponse: JSON parse failed');
            return null;
        }
    }

    private buildResult(raw: RawExtraction): ExtractionResult {
        const knowledge: NewKnowledgeEntry[] = [];
        const skills: NewSkillEntry[] = [];
        const selfTestQuestions: SelfTestQuestion[] = [];

        // Build knowledge entries
        if (Array.isArray(raw.knowledge)) {
            for (const k of raw.knowledge) {
                if (!k.content || typeof k.content !== 'string') continue;
                knowledge.push({
                    content: k.content,
                    source: 'cloud_escalation',
                    confidence: typeof k.confidence === 'number'
                        ? Math.max(0, Math.min(1, k.confidence))
                        : 0.5,
                    tags: Array.isArray(k.tags) ? k.tags.filter((t) => typeof t === 'string') : [],
                });
            }
        }

        // Build skill entries
        if (Array.isArray(raw.skills)) {
            for (const s of raw.skills) {
                if (!s.name || !s.description || !Array.isArray(s.steps) || s.steps.length === 0) continue;

                const steps = s.steps
                    .filter((st) => st.description && st.input !== undefined && st.output !== undefined)
                    .map((st, idx) => ({
                        order: typeof st.order === 'number' ? st.order : idx + 1,
                        description: st.description!,
                        input: st.input!,
                        output: st.output!,
                        ...(st.toolName ? { toolName: st.toolName } : {}),
                    }));

                if (steps.length === 0) continue;

                skills.push({
                    name: s.name,
                    description: s.description,
                    steps,
                    tags: Array.isArray(s.tags) ? s.tags.filter((t) => typeof t === 'string') : [],
                });
            }
        }

        // Build self-test questions — use placeholder IDs since entries aren't stored yet
        if (Array.isArray(raw.selfTestQuestions)) {
            for (const q of raw.selfTestQuestions) {
                if (!q.question || typeof q.question !== 'string') continue;
                const idx = typeof q.knowledgeIndex === 'number' ? q.knowledgeIndex : 0;
                selfTestQuestions.push({
                    knowledgeId: `pending_${idx}`,
                    question: q.question,
                });
            }
        }

        // Ensure at least one self-test question per knowledge entry
        for (let i = 0; i < knowledge.length; i++) {
            const hasQuestion = selfTestQuestions.some((q) => q.knowledgeId === `pending_${i}`);
            if (!hasQuestion) {
                selfTestQuestions.push({
                    knowledgeId: `pending_${i}`,
                    question: `Is the following true? ${knowledge[i].content}`,
                });
            }
        }

        // Attach self-test questions to knowledge entries
        for (let i = 0; i < knowledge.length; i++) {
            const questions = selfTestQuestions
                .filter((q) => q.knowledgeId === `pending_${i}`)
                .map((q) => q.question);
            if (questions.length > 0) {
                knowledge[i].selfTestQuestions = questions;
            }
        }

        return { knowledge, skills, selfTestQuestions };
    }
}

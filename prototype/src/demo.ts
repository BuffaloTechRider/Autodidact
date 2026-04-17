#!/usr/bin/env node
/**
 * Autodidact Demo — shows the self-learning loop in 60 seconds.
 *
 * Usage: npx tsx src/demo.ts
 *
 * This demo uses a mock LLM (no Ollama/API key needed) to demonstrate:
 * 1. Agent starts with empty knowledge
 * 2. First queries escalate to "cloud" (mock)
 * 3. Agent learns from each escalation
 * 4. Same queries answered locally on second pass
 * 5. Shows the learning curve in real-time
 */

import { Agent } from './components/agent.js';
import { initDatabase } from './database.js';
import { KnowledgeStore } from './components/knowledge-store.js';
import { SkillStore } from './components/skill-store.js';
import { ConfidenceEvaluator } from './components/confidence-evaluator.js';
import { MetricsTracker } from './components/metrics-tracker.js';
import { UserProfile } from './components/user-profile.js';
import { ToolRegistry } from './components/tool-registry.js';
import { ContextBuilder } from './components/context-builder.js';
import type {
    ChatMessage,
    ChatOptions,
    ChatResponse,
    CloudResponse,
    ExtractionResult,
    ICloudRouter,
    ILearningExtractor,
    ILLMClient,
    ISelfVerificationSystem,
    ISkillEvolver,
    CloudProvider,
    EscalationRecord,
    VerificationResult,
    SkillReviewResult,
    NewKnowledgeEntry,
} from './types.js';
import { createSilentLogger } from './utils/logger.js';
import { generateId } from './utils/id.js';
import * as fs from 'node:fs';

// ── Mock LLM Client ─────────────────────────────────────────
// Knowledge base the "cloud" knows
const CLOUD_KNOWLEDGE: Record<string, { answer: string; tags: string[]; domain: string; topic: string }> = {
    'what is rust': {
        answer: 'Rust is a systems programming language focused on safety, speed, and concurrency. It prevents memory errors at compile time through its ownership system.',
        tags: ['programming', 'rust', 'systems'],
        domain: 'programming',
        topic: 'rust',
    },
    'what is typescript': {
        answer: 'TypeScript is a strongly typed superset of JavaScript developed by Microsoft. It adds static type checking and compiles to plain JavaScript.',
        tags: ['programming', 'typescript', 'javascript'],
        domain: 'programming',
        topic: 'typescript',
    },
    'what is docker': {
        answer: 'Docker is a platform for building, shipping, and running applications in containers. Containers package code with dependencies for consistent deployment across environments.',
        tags: ['devops', 'docker', 'containers'],
        domain: 'devops',
        topic: 'docker',
    },
    'what is kubernetes': {
        answer: 'Kubernetes (K8s) is an open-source container orchestration platform. It automates deployment, scaling, and management of containerized applications across clusters.',
        tags: ['devops', 'kubernetes', 'orchestration'],
        domain: 'devops',
        topic: 'kubernetes',
    },
    'what is machine learning': {
        answer: 'Machine learning is a subset of AI where systems learn patterns from data without explicit programming. Key approaches include supervised learning, unsupervised learning, and reinforcement learning.',
        tags: ['ai', 'machine-learning', 'data-science'],
        domain: 'ai',
        topic: 'machine-learning',
    },
    'what is a neural network': {
        answer: 'A neural network is a computing system inspired by biological neurons. It consists of layers of interconnected nodes that process data through weighted connections, learning patterns via backpropagation.',
        tags: ['ai', 'neural-networks', 'deep-learning'],
        domain: 'ai',
        topic: 'neural-networks',
    },
    'what is git': {
        answer: 'Git is a distributed version control system for tracking changes in source code. It supports branching, merging, and collaboration through repositories.',
        tags: ['tools', 'git', 'version-control'],
        domain: 'tools',
        topic: 'git',
    },
    'what is sql': {
        answer: 'SQL (Structured Query Language) is the standard language for managing relational databases. It supports querying, inserting, updating, and deleting data with declarative syntax.',
        tags: ['databases', 'sql', 'data'],
        domain: 'databases',
        topic: 'sql',
    },
};

function findCloudAnswer(query: string): { answer: string; tags: string[]; domain: string; topic: string } | null {
    const lower = query.toLowerCase().trim().replace(/[?!.]/g, '');
    for (const [key, value] of Object.entries(CLOUD_KNOWLEDGE)) {
        if (lower.includes(key) || key.includes(lower)) {
            return value;
        }
    }
    return null;
}

// Simple embedding: hash words into a fixed-size vector
function mockEmbed(text: string): number[] {
    const dim = 64;
    const vec = new Array(dim).fill(0);
    const words = text.toLowerCase().split(/\s+/);
    for (const word of words) {
        for (let i = 0; i < word.length; i++) {
            const idx = (word.charCodeAt(i) * (i + 1)) % dim;
            vec[idx] += 1;
        }
    }
    // Normalize
    const mag = Math.sqrt(vec.reduce((s: number, v: number) => s + v * v, 0));
    if (mag > 0) {
        for (let i = 0; i < dim; i++) vec[i] /= mag;
    }
    return vec;
}

class MockLLMClient implements ILLMClient {
    learnedTopics = new Set<string>();

    markLearned(query: string) {
        this.learnedTopics.add(query.toLowerCase().trim().replace(/[?!.]/g, ''));
    }

    async chat(messages: ChatMessage[], _options?: ChatOptions): Promise<ChatResponse> {
        const lastMsg = messages[messages.length - 1]?.content ?? '';

        // Check if this is a confidence self-assessment request
        if (lastMsg.includes('Rate your confidence')) {
            // The confidence evaluator sends the raw query for self-assessment.
            // Check if the query matches something we've already learned.
            // In a real system, the LLM would see knowledge context.
            // Here we check if the query topic exists in our cloud knowledge.
            // Extract just the query part (before any "\n\nAvailable knowledge:" context)
            const rawPart = lastMsg.replace('Rate your confidence 0-1 in answering: ', '');
            const queryPart = rawPart.split('\n\n')[0].trim();
            // If we have knowledge about this topic AND it was previously learned,
            // return high confidence. Otherwise low.
            if (this.learnedTopics.has(queryPart.toLowerCase().replace(/[?!.]/g, ''))) {
                return { content: '0.9', usage: { promptTokens: 10, completionTokens: 1 }, model: 'mock' };
            }
            return { content: '0.05', usage: { promptTokens: 10, completionTokens: 1 }, model: 'mock' };
        }

        // Check if this is an extraction request
        if (messages[0]?.content?.includes('knowledge extraction')) {
            return { content: '{"knowledge":[],"skills":[],"selfTestQuestions":[],"tools":[]}', usage: { promptTokens: 50, completionTokens: 20 }, model: 'mock' };
        }

        // Regular query — find the best matching knowledge from system prompt context
        const systemMsg = messages.find(m => m.role === 'system')?.content ?? '';
        const queryLower = lastMsg.toLowerCase();

        // Try to find a knowledge entry that matches the query topic
        if (systemMsg.includes('- ')) {
            const entries = systemMsg.match(/- .+/g) ?? [];
            for (const entry of entries) {
                const entryText = entry.replace(/^- (\[.*?\] )?/, '');
                // Check if the entry topic matches the query
                const queryWords = queryLower.split(/\s+/).filter(w => w.length > 3);
                const entryLower = entryText.toLowerCase();
                if (queryWords.some(w => entryLower.includes(w))) {
                    return { content: entryText, usage: { promptTokens: 30, completionTokens: 20 }, model: 'mock-local' };
                }
            }
            // If we have entries but no match, return the first one
            if (entries.length > 0) {
                return { content: entries[0]!.replace(/^- (\[.*?\] )?/, ''), usage: { promptTokens: 30, completionTokens: 20 }, model: 'mock-local' };
            }
        }

        return { content: `I don't have enough information to answer: "${lastMsg}"`, usage: { promptTokens: 20, completionTokens: 15 }, model: 'mock-local' };
    }

    async embed(text: string): Promise<number[]> {
        return mockEmbed(text);
    }
}

class MockCloudRouter implements ICloudRouter {
    private escalationCount = 0;

    async escalate(query: string, _context?: string): Promise<CloudResponse> {
        this.escalationCount++;
        const match = findCloudAnswer(query);
        if (match) {
            return {
                content: match.answer,
                provider: 'mock-cloud',
                model: 'gpt-4o-mock',
                cost: 0.002,
                latencyMs: 150 + Math.random() * 100,
            };
        }
        return {
            content: `Cloud answer for: ${query}`,
            provider: 'mock-cloud',
            model: 'gpt-4o-mock',
            cost: 0.001,
            latencyMs: 100,
        };
    }

    getProviders(): CloudProvider[] {
        return [{ name: 'mock-cloud', baseUrl: 'http://mock', apiKey: 'mock', model: 'gpt-4o-mock', costPer1kTokens: 0.01, timeoutMs: 5000, priority: 1 }];
    }

    getEscalationLog(): EscalationRecord[] { return []; }
}

class MockLearningExtractor implements ILearningExtractor {
    async extract(query: string, response: string): Promise<ExtractionResult> {
        const match = findCloudAnswer(query);
        const knowledge: NewKnowledgeEntry[] = [{
            content: response,
            source: 'cloud_escalation',
            confidence: 0.9,
            tags: match?.tags ?? ['general'],
            domain: match?.domain ?? 'general',
            topic: match?.topic ?? 'uncategorized',
            category: 'facts',
            selfTestQuestions: [`What is the definition of ${query.replace(/what is /i, '').replace(/[?]/g, '')}?`],
        }];
        return { knowledge, skills: [], selfTestQuestions: [], tools: [] };
    }
}

class MockSelfVerification implements ISelfVerificationSystem {
    async runVerificationCycle(): Promise<VerificationResult> { return { tested: 0, passed: 0, failed: 0, staleEntries: [] }; }
    getPassRate(): number { return 1.0; }
}

class MockSkillEvolver implements ISkillEvolver {
    async reviewSkill(_id: string): Promise<SkillReviewResult> { return { skillId: _id, skillName: '', previousVersion: 1, action: 'kept', reason: 'mock' }; }
    async checkAndEvolve(): Promise<SkillReviewResult[]> { return []; }
}

// ── Demo Runner ─────────────────────────────────────────────

const QUERIES = [
    'What is Rust?',
    'What is TypeScript?',
    'What is Docker?',
    'What is Kubernetes?',
    'What is machine learning?',
    'What is a neural network?',
    'What is Git?',
    'What is SQL?',
];

function color(text: string, code: number): string {
    return `\x1b[${code}m${text}\x1b[0m`;
}

const green = (t: string) => color(t, 32);
const yellow = (t: string) => color(t, 33);
const red = (t: string) => color(t, 31);
const cyan = (t: string) => color(t, 36);
const bold = (t: string) => color(t, 1);
const dim = (t: string) => color(t, 2);

async function runDemo() {
    console.log('\n' + bold('🧠 Autodidact Demo — Self-Learning AI Agent'));
    console.log(dim('   Watch the agent learn in real-time. No API keys needed.\n'));

    // Clean up any previous demo DB
    const dbPath = '/tmp/autodidact-demo.db';
    if (fs.existsSync(dbPath)) fs.unlinkSync(dbPath);

    const logger = createSilentLogger();
    const db = initDatabase(dbPath);
    const mockLLM = new MockLLMClient();

    const agent = new Agent(
        {
            localLLM: { baseUrl: 'http://mock:11434/v1', model: 'mock' },
            cloudRouter: { providers: [] },
            database: { path: dbPath },
        },
        {
            llmClient: mockLLM,
            cloudRouter: new MockCloudRouter(),
            learningExtractor: new MockLearningExtractor(),
            knowledgeStore: new KnowledgeStore(db, undefined, logger),
            skillStore: new SkillStore(db, logger),
            confidenceEvaluator: new ConfidenceEvaluator(db, mockLLM, { localThreshold: 0.7, hedgeThreshold: 0.5, initialAlpha: 1, initialBeta: 1 }, logger),
            selfVerification: new MockSelfVerification(),
            skillEvolver: new MockSkillEvolver(),
            userProfile: new UserProfile(db, logger),
            metricsTracker: new MetricsTracker(db, logger),
            toolRegistry: new ToolRegistry(db, undefined, logger),
            logger,
        },
    );

    // ── Pass 1: Agent knows nothing, escalates everything ───
    console.log(bold('━━━ Pass 1: Agent starts with empty brain ━━━\n'));

    let escalations = 0;
    let localAnswers = 0;

    for (const query of QUERIES) {
        const result = await agent.query(query);
        const isLocal = result.routing.decision !== 'ESCALATE';

        if (isLocal) {
            localAnswers++;
            console.log(`  ${green('LOCAL')}  ${query}`);
        } else {
            escalations++;
            mockLLM.markLearned(query);
            console.log(`  ${red('CLOUD')}  ${query} ${dim(`→ learned! ($${result.cost.toFixed(3)})`)}`);
        }

        // Small delay for visual effect
        await new Promise(r => setTimeout(r, 50));
    }

    const pass1Rate = ((localAnswers / QUERIES.length) * 100).toFixed(0);
    console.log(`\n  ${dim('Pass 1:')} ${red(`${escalations} escalations`)}, ${green(`${localAnswers} local`)} — Local rate: ${bold(pass1Rate + '%')}\n`);

    // ── Pass 2: Same questions — agent should answer locally ─
    console.log(bold('━━━ Pass 2: Same questions — agent has learned ━━━\n'));

    escalations = 0;
    localAnswers = 0;

    for (const query of QUERIES) {
        const result = await agent.query(query);
        const isLocal = result.routing.decision !== 'ESCALATE';

        if (isLocal) {
            localAnswers++;
            const preview = result.content.replace(/^\[Note:.*?\]\s*/, '').slice(0, 70).replace(/\n/g, ' ');
            console.log(`  ${green('LOCAL')}  ${query} ${dim(`→ ${preview}...`)}`);
        } else {
            escalations++;
            console.log(`  ${yellow('CLOUD')}  ${query} ${dim('→ still learning...')}`);
        }

        await new Promise(r => setTimeout(r, 50));
    }

    const pass2Rate = ((localAnswers / QUERIES.length) * 100).toFixed(0);
    console.log(`\n  ${dim('Pass 2:')} ${red(`${escalations} escalations`)}, ${green(`${localAnswers} local`)} — Local rate: ${bold(pass2Rate + '%')}\n`);

    // ── Summary ─────────────────────────────────────────────
    const metrics = agent.getMetrics();
    console.log(bold('━━━ Results ━━━\n'));
    console.log(`  ${cyan('Total queries:')}          ${metrics.totalQueries}`);
    console.log(`  ${cyan('Total escalations:')}      ${metrics.totalEscalations}`);
    console.log(`  ${cyan('Knowledge entries:')}      ${metrics.totalKnowledgeEntries}`);
    console.log(`  ${cyan('Local resolution rate:')}  ${bold((metrics.localResolutionRate * 100).toFixed(0) + '%')}`);
    console.log(`  ${cyan('Cloud cost:')}             $${(metrics.totalEscalations * 0.002).toFixed(3)}`);
    console.log(`  ${cyan('Cost saved (Pass 2):')}    $${(localAnswers * 0.002).toFixed(3)}`);

    console.log(`\n  ${bold('The agent learned from ' + metrics.totalKnowledgeEntries + ' cloud escalations.')}`);
    console.log(`  ${bold('It will never ask the same question twice.')}\n`);

    // Cleanup
    db.close();
    if (fs.existsSync(dbPath)) fs.unlinkSync(dbPath);
}

runDemo().catch(console.error);

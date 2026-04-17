#!/usr/bin/env node
/**
 * EvoAgent Bedrock Demo — local Ollama + AWS Bedrock cloud escalation.
 *
 * Uses:
 *   - Local: Ollama qwen2.5:7b (chat) + nomic-embed-text (embeddings)
 *   - Cloud: AWS Bedrock Claude (escalation)
 *
 * Prerequisites:
 *   - Ollama running with qwen2.5:7b and nomic-embed-text
 *   - AWS credentials configured (aws configure / AWS_PROFILE / IAM role)
 *
 * Usage: npx tsx src/demo-bedrock.ts
 */

import { Agent } from './components/agent.js';
import { LLMClient } from './components/llm-client.js';
import { BedrockRouter } from './components/bedrock-router.js';
import type { BedrockProviderConfig } from './components/bedrock-router.js';
import { initDatabase } from './database.js';
import { KnowledgeStore } from './components/knowledge-store.js';
import { SkillStore } from './components/skill-store.js';
import { ConfidenceEvaluator } from './components/confidence-evaluator.js';
import { LearningExtractor } from './components/learning-extractor.js';
import { MetricsTracker } from './components/metrics-tracker.js';
import { UserProfile } from './components/user-profile.js';
import { ToolRegistry } from './components/tool-registry.js';
import { SelfVerificationSystem } from './components/self-verification.js';
import { SkillEvolver } from './components/skill-evolver.js';
import type { ChatMessage, ChatOptions, ChatResponse, ILLMClient } from './types.js';
import { defaultLogger } from './utils/logger.js';
import * as fs from 'node:fs';

const BEDROCK_DB = '/tmp/evoagent-bedrock-demo.db';

class DualModelClient implements ILLMClient {
    private chat_: LLMClient;
    private embed_: LLMClient;
    constructor(baseUrl: string, chatModel: string, embedModel: string, timeoutMs: number) {
        this.chat_ = new LLMClient({ baseUrl, apiKey: 'ollama', model: chatModel, timeoutMs });
        this.embed_ = new LLMClient({ baseUrl, apiKey: 'ollama', model: embedModel, timeoutMs });
    }
    async chat(messages: ChatMessage[], options?: ChatOptions): Promise<ChatResponse> {
        return this.chat_.chat(messages, options);
    }
    async embed(text: string): Promise<number[]> {
        return this.embed_.embed(text);
    }
}

function color(text: string, code: number): string { return `\x1b[${code}m${text}\x1b[0m`; }
const green = (t: string) => color(t, 32);
const yellow = (t: string) => color(t, 33);
const red = (t: string) => color(t, 31);
const cyan = (t: string) => color(t, 36);
const bold = (t: string) => color(t, 1);
const dim = (t: string) => color(t, 2);

const QUERIES = [
    'What is the Rust programming language and why is it popular?',
    'Explain what Docker containers are in simple terms.',
    'What is Thompson Sampling and how does it work?',
    'What are the main differences between SQL and NoSQL databases?',
];

const QUERIES_REPHRASED = [
    'Tell me about Rust.',
    'What are Docker containers?',
    'How does Thompson Sampling work?',
    'Compare SQL vs NoSQL.',
];

async function main() {
    console.log('\n' + bold('🧠 EvoAgent — Local Ollama + AWS Bedrock Demo'));
    console.log(dim('   Local: qwen2.5:7b | Cloud: Bedrock Claude | Embeddings: nomic-embed-text\n'));

    // Verify Ollama
    try {
        const res = await fetch('http://localhost:11434/api/tags');
        if (!res.ok) throw new Error();
    } catch {
        console.error(red('Ollama not running. Start with: ollama serve'));
        process.exit(1);
    }

    if (fs.existsSync(BEDROCK_DB)) fs.unlinkSync(BEDROCK_DB);

    const logger = defaultLogger;
    const db = initDatabase(BEDROCK_DB);
    const llm = new DualModelClient('http://localhost:11434/v1', 'qwen2.5:7b', 'nomic-embed-text', 60_000);

    // Bedrock providers — try Claude models in order
    const bedrockProviders: BedrockProviderConfig[] = [
        {
            region: 'us-west-2',
            modelId: 'us.anthropic.claude-3-5-haiku-20241022-v1:0',
            costPer1kInputTokens: 0.001,
            costPer1kOutputTokens: 0.005,
            maxTokens: 2048,
        },
        {
            region: 'us-west-2',
            modelId: 'us.anthropic.claude-sonnet-4-20250514-v1:0',
            costPer1kInputTokens: 0.003,
            costPer1kOutputTokens: 0.015,
            maxTokens: 2048,
        },
    ];

    const bedrockRouter = new BedrockRouter(db, bedrockProviders, logger);

    const agent = new Agent(
        {
            localLLM: { baseUrl: 'http://localhost:11434/v1', apiKey: 'ollama', model: 'qwen2.5:7b', timeoutMs: 60_000 },
            cloudRouter: { providers: [] },
            confidenceEvaluator: { localThreshold: 0.65, hedgeThreshold: 0.4, initialAlpha: 1, initialBeta: 1 },
            database: { path: BEDROCK_DB },
            toolRegistry: { enabled: false, autoVerify: false, decayThreshold: 0.1 },
        },
        {
            llmClient: llm,
            cloudRouter: bedrockRouter,
            knowledgeStore: new KnowledgeStore(db, undefined, logger),
            skillStore: new SkillStore(db, logger),
            confidenceEvaluator: new ConfidenceEvaluator(db, llm, { localThreshold: 0.65, hedgeThreshold: 0.4, initialAlpha: 1, initialBeta: 1 }, logger),
            learningExtractor: new LearningExtractor(llm, logger),
            selfVerification: new SelfVerificationSystem(llm, new KnowledgeStore(db, undefined, logger), db, undefined, logger),
            skillEvolver: new SkillEvolver(llm, new SkillStore(db, logger), db, undefined, logger),
            userProfile: new UserProfile(db, logger),
            metricsTracker: new MetricsTracker(db, logger),
            toolRegistry: new ToolRegistry(db, undefined, logger),
            logger,
        },
    );

    // ── Pass 1 ──────────────────────────────────────────────
    console.log(bold('━━━ Pass 1: Empty brain — escalates to Bedrock Claude ━━━\n'));

    for (const query of QUERIES) {
        console.log(`  ${cyan('Q:')} ${query}`);
        const start = Date.now();
        try {
            const result = await agent.query(query);
            const elapsed = Date.now() - start;
            const route = result.routing.decision;
            const routeColor = route === 'ESCALATE' ? red : route === 'HEDGE' ? yellow : green;
            console.log(`  ${routeColor(route.padEnd(8))} ${dim(`(${elapsed}ms, $${result.cost.toFixed(4)}, score: ${result.routing.fusedScore.toFixed(2)})`)}`);
            const preview = result.content.replace(/^\[Note:.*?\]\s*/, '').slice(0, 120).replace(/\n/g, ' ');
            console.log(`  ${dim('→')} ${preview}...`);
            console.log();
        } catch (err) {
            console.log(`  ${red('ERROR')} ${err instanceof Error ? err.message : String(err)}\n`);
        }
    }

    const m1 = agent.getMetrics();
    console.log(dim(`  Pass 1: ${m1.totalEscalations} escalations, ${m1.totalKnowledgeEntries} facts learned, $${(m1.totalEscalations * 0.003).toFixed(4)} cloud cost\n`));

    // ── Pass 2 ──────────────────────────────────────────────
    console.log(bold('━━━ Pass 2: Rephrased questions — does it remember? ━━━\n'));

    for (const query of QUERIES_REPHRASED) {
        console.log(`  ${cyan('Q:')} ${query}`);
        const start = Date.now();
        try {
            const result = await agent.query(query);
            const elapsed = Date.now() - start;
            const route = result.routing.decision;
            const routeColor = route === 'ESCALATE' ? red : route === 'HEDGE' ? yellow : green;
            console.log(`  ${routeColor(route.padEnd(8))} ${dim(`(${elapsed}ms, $${result.cost.toFixed(4)}, score: ${result.routing.fusedScore.toFixed(2)})`)}`);
            const preview = result.content.replace(/^\[Note:.*?\]\s*/, '').slice(0, 120).replace(/\n/g, ' ');
            console.log(`  ${dim('→')} ${preview}...`);
            console.log();
        } catch (err) {
            console.log(`  ${red('ERROR')} ${err instanceof Error ? err.message : String(err)}\n`);
        }
    }

    // ── Results ─────────────────────────────────────────────
    const m = agent.getMetrics();
    console.log(bold('━━━ Results ━━━\n'));
    console.log(`  ${cyan('Total queries:')}          ${m.totalQueries}`);
    console.log(`  ${cyan('Total escalations:')}      ${m.totalEscalations}`);
    console.log(`  ${cyan('Knowledge entries:')}      ${m.totalKnowledgeEntries}`);
    console.log(`  ${cyan('Local resolution rate:')}  ${bold((m.localResolutionRate * 100).toFixed(0) + '%')}`);
    console.log(`  ${cyan('Confidence calibration:')} ${(m.confidenceCalibration * 100).toFixed(0)}%`);
    console.log();

    db.close();
    if (fs.existsSync(BEDROCK_DB)) fs.unlinkSync(BEDROCK_DB);
}

main().catch(err => {
    console.error(red('Fatal:'), err);
    process.exit(1);
});

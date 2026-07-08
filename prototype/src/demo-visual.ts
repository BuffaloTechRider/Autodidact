#!/usr/bin/env node
/**
 * Autodidact Visual Demo — step-by-step UI for video recording.
 *
 * Shows the thinking process visually:
 *   🧠 THINKING → ☁️ CLOUD CALL → 📚 LEARNING → ✅ ANSWER
 *
 * Usage: npx tsx src/demo-visual.ts
 * Env vars:
 *   LOCAL_MODEL (default: llama3.2)
 *   EMBEDDING_MODEL (default: nomic-embed-text)
 *   AWS_PROFILE (default: acamlops)
 *   BEDROCK_MODEL (default: us.anthropic.claude-3-5-haiku-20241022-v1:0)
 */

import { Agent } from './components/agent.js';
import type { AgentComponents } from './components/agent.js';
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
import type { AgentResponse, ChatMessage, ChatOptions, ChatResponse, ILLMClient } from './types.js';
import { defaultLogger } from './utils/logger.js';
import * as fs from 'node:fs';
import * as readline from 'node:readline';

// ── Terminal colors ──
const C = {
    reset: '\x1b[0m', bold: '\x1b[1m', dim: '\x1b[2m',
    green: '\x1b[32m', yellow: '\x1b[33m', red: '\x1b[31m',
    cyan: '\x1b[36m', magenta: '\x1b[35m', blue: '\x1b[34m',
};
const LINE = '──────────────────────────────────────────────';

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

function banner(text: string): void {
    console.log(`\n${C.cyan}${C.bold}${'═'.repeat(50)}${C.reset}`);
    console.log(`${C.cyan}${C.bold}  ${text}${C.reset}`);
    console.log(`${C.cyan}${C.bold}${'═'.repeat(50)}${C.reset}\n`);
}

function printThinking(score: number, decision: string): void {
    console.log(`${C.dim}${LINE}${C.reset}`);
    console.log(`${C.bold}🧠 THINKING ${C.dim}(local model)${C.reset}`);
    console.log(`${C.dim}${LINE}${C.reset}`);
    const icon = score >= 0.55 ? '✅' : score >= 0.3 ? '⚠️' : '❌';
    const label = score >= 0.55 ? `${C.green}HIGH${C.reset}` : score >= 0.3 ? `${C.yellow}MEDIUM${C.reset}` : `${C.red}LOW${C.reset}`;
    console.log(`  Confidence: ${C.bold}${score.toFixed(3)}${C.reset} ${icon} ${label}`);
    if (decision === 'ESCALATE') {
        console.log(`  Decision: ${C.red}${C.bold}🔴 ESCALATE to cloud${C.reset}`);
    } else {
        console.log(`  Decision: ${C.green}${C.bold}🟢 LOCAL execution${C.reset}`);
    }
}

function printCloudCall(model: string, latencyMs: number, cost: number): void {
    console.log(`${C.dim}${LINE}${C.reset}`);
    console.log(`${C.bold}☁️  CLOUD CALL${C.reset}`);
    console.log(`${C.dim}${LINE}${C.reset}`);
    console.log(`  Model: ${C.cyan}${model}${C.reset}`);
    console.log(`  Latency: ${C.bold}${(latencyMs / 1000).toFixed(1)}s${C.reset}`);
    console.log(`  Cost: ${C.bold}$${cost.toFixed(4)}${C.reset}`);
}

function printLearning(knowledgeCount: number, skillCount: number): void {
    console.log(`${C.dim}${LINE}${C.reset}`);
    console.log(`${C.bold}📚 LEARNING${C.reset}`);
    console.log(`${C.dim}${LINE}${C.reset}`);
    const parts: string[] = [];
    if (knowledgeCount > 0) parts.push(`${C.green}${knowledgeCount} knowledge${C.reset}`);
    if (skillCount > 0) parts.push(`${C.cyan}${skillCount} skills${C.reset}`);
    console.log(`  Extracted: ${parts.join(' + ') || 'processing...'}`);
    console.log(`  ${C.dim}Stored in local memory for future use${C.reset}`);
}

function printLocalDone(latencyMs: number): void {
    console.log(`  Latency: ${C.bold}${(latencyMs / 1000).toFixed(1)}s${C.reset}`);
    console.log(`  Cost: ${C.green}${C.bold}$0.0000${C.reset} ${C.dim}(free — answered from memory)${C.reset}`);
}

function printAnswer(content: string): void {
    console.log(`${C.dim}${LINE}${C.reset}`);
    console.log(`${C.bold}✅ ANSWER${C.reset}`);
    console.log(`${C.dim}${LINE}${C.reset}`);
    console.log(`  ${content.split('\n').join('\n  ')}`);
    console.log();
}

function printMetrics(agent: Agent): void {
    const m = agent.getMetrics();
    const rate = (m.localResolutionRate * 100).toFixed(1);
    console.log(`${C.magenta}${C.bold}╔════════════════════════════════════════════╗${C.reset}`);
    console.log(`${C.magenta}${C.bold}║         LEARNING METRICS DASHBOARD         ║${C.reset}`);
    console.log(`${C.magenta}${C.bold}╠════════════════════════════════════════════╣${C.reset}`);
    console.log(`${C.magenta}║${C.reset}  Total Queries:         ${C.bold}${String(m.totalQueries).padStart(6)}${C.reset}          ${C.magenta}║${C.reset}`);
    console.log(`${C.magenta}║${C.reset}  Escalations:           ${C.bold}${String(m.totalEscalations).padStart(6)}${C.reset}          ${C.magenta}║${C.reset}`);
    console.log(`${C.magenta}║${C.reset}  Local Resolution:  ${C.green}${C.bold}${rate.padStart(6)}%${C.reset}          ${C.magenta}║${C.reset}`);
    console.log(`${C.magenta}║${C.reset}  Knowledge Entries:     ${C.bold}${String(m.totalKnowledgeEntries).padStart(6)}${C.reset}          ${C.magenta}║${C.reset}`);
    console.log(`${C.magenta}║${C.reset}  Skills Learned:    ${C.cyan}${C.bold}${String(m.totalSkillEntries).padStart(6)}${C.reset}          ${C.magenta}║${C.reset}`);
    console.log(`${C.magenta}${C.bold}╚════════════════════════════════════════════╝${C.reset}`);
}

async function queryWithUI(agent: Agent, query: string, prevKnowledge: number, prevSkills: number): Promise<AgentResponse> {
    console.log(`\n${C.cyan}${C.bold}❓ ${query}${C.reset}`);

    const start = Date.now();
    const res = await agent.query(query);
    const elapsed = Date.now() - start;

    // Print thinking
    printThinking(res.routing.fusedScore, res.routing.decision);

    if (res.routing.decision === 'ESCALATE') {
        // Print cloud call info
        const model = process.env['BEDROCK_MODEL'] ?? 'us.anthropic.claude-3-5-haiku-20241022-v1:0';
        printCloudCall(model, elapsed, res.cost);

        // Print learning (diff from before)
        const m = agent.getMetrics();
        const newKnowledge = m.totalKnowledgeEntries - prevKnowledge;
        const newSkills = m.totalSkillEntries - prevSkills;
        printLearning(newKnowledge, newSkills);
    } else {
        printLocalDone(elapsed);
    }

    printAnswer(res.content);
    return res;
}

async function sleep(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
}

async function main(): Promise<void> {
    console.log(`\n${C.bold}${C.cyan}🧠 AUTODIDACT — The Local-First AI Agent That Actually Learns and Evolves${C.reset}`);
    console.log(`${C.dim}Vietnam AI Stars 2026${C.reset}\n`);

    const DB_PATH = '/tmp/autodidact-visual-demo.db';
    if (fs.existsSync(DB_PATH)) fs.unlinkSync(DB_PATH);

    // Verify Ollama
    try {
        const res = await fetch('http://localhost:11434/api/tags');
        if (!res.ok) throw new Error();
    } catch {
        console.error(`${C.red}Ollama not running. Start with: ollama serve${C.reset}`);
        process.exit(1);
    }

    process.env['AWS_PROFILE'] = process.env['AWS_PROFILE'] ?? 'acamlops';

    const localModel = process.env['LOCAL_MODEL'] ?? 'llama3.2';
    const embedModel = process.env['EMBEDDING_MODEL'] ?? 'nomic-embed-text';
    const bedrockModel = process.env['BEDROCK_MODEL'] ?? 'us.anthropic.claude-3-5-haiku-20241022-v1:0';

    const logger = { ...defaultLogger, debug: () => { }, info: () => { } }; // suppress noise

    const db = initDatabase(DB_PATH);
    const llm = new DualModelClient('http://localhost:11434/v1', localModel, embedModel, 120_000);

    const bedrockProviders: BedrockProviderConfig[] = [{
        region: process.env['AWS_REGION'] ?? 'us-east-1',
        modelId: bedrockModel,
        costPer1kInputTokens: 0.001,
        costPer1kOutputTokens: 0.005,
        maxTokens: 2048,
    }];

    const knowledgeStore = new KnowledgeStore(db, undefined, logger);
    const skillStore = new SkillStore(db, logger);

    const components: AgentComponents = {
        llmClient: llm,
        cloudRouter: new BedrockRouter(db, bedrockProviders, logger),
        knowledgeStore,
        skillStore,
        confidenceEvaluator: new ConfidenceEvaluator(db, llm, { localThreshold: 0.55, hedgeThreshold: 0.55, initialAlpha: 1, initialBeta: 1 }, logger),
        learningExtractor: new LearningExtractor(llm, logger),
        selfVerification: new SelfVerificationSystem(llm, knowledgeStore, db, undefined, logger),
        skillEvolver: new SkillEvolver(llm, skillStore, db, undefined, logger),
        userProfile: new UserProfile(db, logger),
        metricsTracker: new MetricsTracker(db, logger),
        toolRegistry: new ToolRegistry(db, undefined, logger),
        logger,
    };

    const agent = new Agent(
        {
            localLLM: { baseUrl: 'http://localhost:11434/v1', apiKey: 'ollama', model: localModel, timeoutMs: 120_000 },
            cloudRouter: { providers: [] },
            confidenceEvaluator: { localThreshold: 0.55, hedgeThreshold: 0.55, initialAlpha: 1, initialBeta: 1 },
            database: { path: DB_PATH },
            toolRegistry: { enabled: false, autoVerify: false, decayThreshold: 0.1 },
        },
        components,
    );

    // ── Act 1: ESCALATE ──
    banner('ACT 1: Empty Brain — The Agent Knows Nothing');
    console.log(`${C.dim}The brain has no memory yet. It must ask the cloud.${C.reset}`);
    let m = agent.getMetrics();
    try {
        await queryWithUI(agent, 'What are the key regulations and compliance requirements for launching a fintech startup in Vietnam?', m.totalKnowledgeEntries, m.totalSkillEntries);
    } catch (err) {
        console.error(`${C.red}Act 1 failed:${C.reset}`, err instanceof Error ? err.message : err);
        console.log(`${C.yellow}Tip: Check Ollama is running and AWS credentials are valid.${C.reset}`);
        process.exit(1);
    }

    await sleep(500);

    // ── Act 2: LOCAL ──
    banner('ACT 2: The Brain Remembers');
    console.log(`${C.dim}A related question — will the brain recognize it?${C.reset}`);
    m = agent.getMetrics();
    await queryWithUI(agent, 'What compliance does a Vietnamese payment app need before launch?', m.totalKnowledgeEntries, m.totalSkillEntries);

    await sleep(500);

    // ── Interactive Mode ──
    banner('INTERACTIVE MODE');
    console.log(`${C.dim}The agent learned from the cloud. Try asking related or new questions.${C.reset}`);
    console.log(`${C.dim}Commands:${C.reset}`);
    console.log(`${C.dim}  • Type any question to ask the agent${C.reset}`);
    console.log(`${C.dim}  • "batch"   — run 12 pre-written queries across 5 domains${C.reset}`);
    console.log(`${C.dim}  • "metrics" — show the learning dashboard${C.reset}`);
    console.log(`${C.dim}  • "quit"    — exit${C.reset}\n`);

    const batchQueries = [
        'Compare PostgreSQL vs MongoDB for a real-time analytics dashboard',
        'Should I use a relational or document database for event tracking?',
        'What are the biggest challenges for AI startups in Southeast Asia?',
        'What obstacles do Vietnamese tech companies face when scaling?',
        'How do I set up a CI/CD pipeline for a TypeScript monorepo?',
        'What is the best way to automate deployment for a Node.js project?',
        'Explain the difference between fine-tuning and retrieval-augmented generation',
        'When should I use RAG instead of fine-tuning a language model?',
        'How do open-source developer tools typically monetize?',
        'What revenue models work for open-source AI frameworks?',
        'What database is better for analytics workloads with complex joins?',
        'What are the regulatory hurdles for fintech in Vietnam?',
    ];

    const rl = readline.createInterface({ input: process.stdin, output: process.stdout });

    const ask = (): void => {
        rl.question(`${C.cyan}${C.bold}You: ${C.reset}`, async (input) => {
            const t = input.trim();
            if (!t || ['quit', 'exit', '/quit', '/exit'].includes(t.toLowerCase())) {
                console.log(`\n${C.dim}Final stats:${C.reset}`);
                printMetrics(agent);
                rl.close();
                db.close();
                return;
            }
            if (t.toLowerCase() === 'metrics') {
                printMetrics(agent);
                ask();
                return;
            }
            if (t.toLowerCase() === 'batch') {
                banner('BATCH: 12 Queries Across 5 Domains');
                let localCount = 0, escalateCount = 0;
                for (let i = 0; i < batchQueries.length; i++) {
                    console.log(`${C.dim}[${i + 1}/${batchQueries.length}]${C.reset}`);
                    const bm = agent.getMetrics();
                    try {
                        const res = await queryWithUI(agent, batchQueries[i], bm.totalKnowledgeEntries, bm.totalSkillEntries);
                        if (res.routing.decision === 'LOCAL' || res.routing.decision === 'HEDGE') localCount++;
                        else escalateCount++;
                    } catch (err) {
                        console.error(`${C.red}  Failed:${C.reset}`, err instanceof Error ? err.message : err);
                    }
                }
                console.log(`\n${C.bold}Batch: ${C.green}${localCount} local${C.reset} / ${C.red}${escalateCount} escalated${C.reset} out of ${batchQueries.length}\n`);
                printMetrics(agent);
                ask();
                return;
            }
            try {
                const qm = agent.getMetrics();
                await queryWithUI(agent, t, qm.totalKnowledgeEntries, qm.totalSkillEntries);
            } catch (err) {
                console.error(`${C.red}Error:${C.reset}`, err instanceof Error ? err.message : err);
            }
            ask();
        });
    };
    ask();
}

main().catch(err => {
    console.error(`${C.red}Demo crashed:${C.reset}`, err);
    process.exit(1);
});

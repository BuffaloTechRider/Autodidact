#!/usr/bin/env node
/**
 * Autodidact Real Model Demo — uses Ollama with real LLM + embeddings.
 *
 * Prerequisites:
 *   - Ollama running with qwen2.5:7b and nomic-embed-text
 *   - npm install
 *
 * Usage: npx tsx src/demo-real.ts
 */

import { Agent } from './components/agent.js';
import { LLMClient } from './components/llm-client.js';
import type { AutodidactConfig, ChatMessage, ChatOptions, ChatResponse, ILLMClient } from './types.js';
import * as fs from 'node:fs';

const DB_PATH = '/tmp/autodidact-real-demo.db';

/**
 * Dual-model LLM client: uses one model for chat, another for embeddings.
 * This is needed because Ollama serves chat (qwen2.5:7b) and embeddings
 * (nomic-embed-text) as separate models on the same endpoint.
 */
class DualModelLLMClient implements ILLMClient {
    private chatClient: LLMClient;
    private embedClient: LLMClient;

    constructor(baseUrl: string, chatModel: string, embedModel: string, timeoutMs: number) {
        this.chatClient = new LLMClient({ baseUrl, apiKey: 'ollama', model: chatModel, timeoutMs });
        this.embedClient = new LLMClient({ baseUrl, apiKey: 'ollama', model: embedModel, timeoutMs });
    }

    async chat(messages: ChatMessage[], options?: ChatOptions): Promise<ChatResponse> {
        return this.chatClient.chat(messages, options);
    }

    async embed(text: string): Promise<number[]> {
        return this.embedClient.embed(text);
    }
}

const QUERIES_PASS1 = [
    'What is the Rust programming language and why is it popular?',
    'Explain what Docker containers are in simple terms.',
    'What is Thompson Sampling and how does it work?',
    'What are the main differences between SQL and NoSQL databases?',
];

const QUERIES_PASS2 = [
    'Tell me about the Rust programming language.',
    'What are Docker containers?',
    'How does Thompson Sampling work?',
    'Compare SQL and NoSQL databases.',
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

async function main() {
    console.log('\n' + bold('🧠 Autodidact Real Model Demo'));
    console.log(dim('   Using Ollama qwen2.5:7b + nomic-embed-text\n'));

    // Verify Ollama is running
    try {
        const res = await fetch('http://localhost:11434/api/tags');
        if (!res.ok) throw new Error('Ollama not responding');
    } catch {
        console.error(red('Error: Ollama is not running. Start it with: ollama serve'));
        process.exit(1);
    }

    // Clean previous demo DB
    if (fs.existsSync(DB_PATH)) fs.unlinkSync(DB_PATH);

    // Configure agent — use qwen2.5:7b for both local and "cloud"
    // In production, cloud would be OpenAI/Anthropic. Here we simulate
    // by using the same Ollama model as a cloud provider.
    const config: Partial<AutodidactConfig> = {
        localLLM: {
            baseUrl: 'http://localhost:11434/v1',
            apiKey: 'ollama',
            model: 'nomic-embed-text',  // embedding model
            timeoutMs: 60_000,
        },
        cloudRouter: {
            providers: [{
                name: 'ollama-cloud-sim',
                baseUrl: 'http://localhost:11434/v1',
                apiKey: 'ollama',
                model: 'qwen2.5:7b',
                costPer1kTokens: 0.01,  // simulated cost
                timeoutMs: 120_000,
                priority: 1,
            }],
        },
        confidenceEvaluator: {
            localThreshold: 0.65,
            hedgeThreshold: 0.4,
            initialAlpha: 1,
            initialBeta: 1,
        },
        database: { path: DB_PATH },
        toolRegistry: { enabled: false, autoVerify: false, decayThreshold: 0.1 },
    };

    console.log(dim('   Initializing agent...\n'));

    const dualLLM = new DualModelLLMClient(
        'http://localhost:11434/v1',
        'qwen2.5:7b',
        'nomic-embed-text',
        60_000,
    );

    const agent = new Agent(config, { llmClient: dualLLM });

    // ── Pass 1 ──────────────────────────────────────────────
    console.log(bold('━━━ Pass 1: Agent starts with empty brain ━━━\n'));

    for (const query of QUERIES_PASS1) {
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

    const metrics1 = agent.getMetrics();
    console.log(dim(`  Pass 1 summary: ${metrics1.totalEscalations} escalations, ${metrics1.totalKnowledgeEntries} facts learned\n`));

    // ── Pass 2 ──────────────────────────────────────────────
    console.log(bold('━━━ Pass 2: Same topics, rephrased — does the agent remember? ━━━\n'));

    for (const query of QUERIES_PASS2) {
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

    // ── Final metrics ───────────────────────────────────────
    const metrics = agent.getMetrics();
    console.log(bold('━━━ Results ━━━\n'));
    console.log(`  ${cyan('Total queries:')}          ${metrics.totalQueries}`);
    console.log(`  ${cyan('Total escalations:')}      ${metrics.totalEscalations}`);
    console.log(`  ${cyan('Knowledge entries:')}      ${metrics.totalKnowledgeEntries}`);
    console.log(`  ${cyan('Local resolution rate:')}  ${bold((metrics.localResolutionRate * 100).toFixed(0) + '%')}`);
    console.log(`  ${cyan('Confidence calibration:')} ${(metrics.confidenceCalibration * 100).toFixed(0)}%`);
    console.log();

    // Cleanup
    if (fs.existsSync(DB_PATH)) fs.unlinkSync(DB_PATH);
}

main().catch(err => {
    console.error(red('Fatal:'), err);
    process.exit(1);
});

import { readFileSync, existsSync } from 'node:fs';
import { resolve } from 'node:path';
import { Agent } from './components/agent.js';
import { resolveConfig } from './config.js';
import { initDatabase } from './database.js';
import { exportAll, importSkill } from './components/skill-format.js';
import { SkillStore } from './components/skill-store.js';
import type { AutodidactConfig } from './types.js';

function loadConfig(): Partial<AutodidactConfig> {
    // Try loading from config file
    const configPath = resolve('autodidact.config.json');
    let fileConfig: Partial<AutodidactConfig> = {};

    if (existsSync(configPath)) {
        try {
            const raw = readFileSync(configPath, 'utf-8');
            fileConfig = JSON.parse(raw) as Partial<AutodidactConfig>;
        } catch (err) {
            console.error(`Failed to parse ${configPath}:`, err instanceof Error ? err.message : err);
        }
    }

    // Override with environment variables
    const envBaseUrl = process.env['EVOAGENT_LOCAL_URL'];
    const envModel = process.env['EVOAGENT_LOCAL_MODEL'];
    const openaiKey = process.env['OPENAI_API_KEY'];
    const anthropicKey = process.env['ANTHROPIC_API_KEY'];

    if (envBaseUrl || envModel) {
        fileConfig.localLLM = {
            ...{ baseUrl: 'http://localhost:11434/v1', model: 'llama3' },
            ...fileConfig.localLLM,
            ...(envBaseUrl ? { baseUrl: envBaseUrl } : {}),
            ...(envModel ? { model: envModel } : {}),
        };
    }

    // Use OpenAI or Anthropic key for local LLM apiKey if set
    const apiKey = openaiKey ?? anthropicKey;
    if (apiKey && fileConfig.localLLM) {
        fileConfig.localLLM.apiKey = apiKey;
    }

    return fileConfig;
}

function printHelp(): void {
    console.log(`
Autodidact CLI — Self-learning AI agent framework

Usage:
  npx tsx src/cli.ts <command> [args]

Commands:
  init                  Create database at configured path
  query <text>          Run a query through the agent
  metrics               Show current metrics
  export-skills         Export all skills to stdout
  import-skill <file>   Import a skill from a Markdown file

Environment Variables:
  EVOAGENT_LOCAL_URL    Local LLM base URL
  EVOAGENT_LOCAL_MODEL  Local LLM model name
  OPENAI_API_KEY        API key for OpenAI-compatible endpoints
  ANTHROPIC_API_KEY     API key for Anthropic endpoints

Config File:
  autodidact.config.json  (in current directory)
`.trim());
}

async function main(): Promise<void> {
    const args = process.argv.slice(2);
    const command = args[0];

    if (!command || command === '--help' || command === '-h') {
        printHelp();
        return;
    }

    const partialConfig = loadConfig();
    const config = resolveConfig(partialConfig);

    switch (command) {
        case 'init': {
            const db = initDatabase(config.database.path);
            db.close();
            console.log(`Database initialized at ${config.database.path}`);
            break;
        }

        case 'query': {
            const text = args.slice(1).join(' ');
            if (!text) {
                console.error('Usage: query <text>');
                process.exit(1);
            }
            const agent = new Agent(partialConfig);
            const response = await agent.query(text);
            console.log('\nRouting:', response.routing.decision);
            console.log('Fused Score:', response.routing.fusedScore.toFixed(3));
            console.log('Cost: $' + response.cost.toFixed(4));
            console.log('Latency:', response.latencyMs + 'ms');
            console.log('\n' + response.content);
            break;
        }

        case 'metrics': {
            const agent = new Agent(partialConfig);
            const metrics = agent.getMetrics();
            console.log(JSON.stringify(metrics, null, 2));
            break;
        }

        case 'export-skills': {
            const db = initDatabase(config.database.path);
            const skillStore = new SkillStore(db);
            const exported = exportAll(skillStore);
            for (const [filename, markdown] of exported) {
                console.log(`--- ${filename} ---`);
                console.log(markdown);
            }
            if (exported.size === 0) {
                console.log('No skills to export.');
            }
            db.close();
            break;
        }

        case 'import-skill': {
            const filePath = args[1];
            if (!filePath) {
                console.error('Usage: import-skill <file>');
                process.exit(1);
            }
            const resolvedPath = resolve(filePath);
            if (!existsSync(resolvedPath)) {
                console.error(`File not found: ${resolvedPath}`);
                process.exit(1);
            }
            const markdown = readFileSync(resolvedPath, 'utf-8');
            const skillEntry = importSkill(markdown);
            const db = initDatabase(config.database.path);
            const skillStore = new SkillStore(db);
            const inserted = skillStore.insert(skillEntry);
            console.log(`Imported skill: ${inserted.name} (id: ${inserted.id}, version: ${inserted.version})`);
            db.close();
            break;
        }

        default:
            console.error(`Unknown command: ${command}`);
            printHelp();
            process.exit(1);
    }
}

main().catch((err) => {
    console.error('Fatal error:', err instanceof Error ? err.message : err);
    process.exit(1);
});

# Autodidact

The open-source self-learning agent framework. Local-first, privacy-first, gets smarter every day.

Autodidact is a TypeScript/Node.js SDK for building AI agents that learn from experience. It wraps a frozen local model in intelligent infrastructure — confidence routing, tiered memory, procedural skills, and a self-verification loop — so the agent improves without fine-tuning, without sending your data to the cloud, and without costing more over time.

```
Day 1:   Agent knows nothing. Escalates 90% of queries to cloud.
Month 3: Agent answers 70% locally. Cloud costs drop. Knowledge grows.
Month 6: Agent handles 90%+ autonomously. Runs almost entirely on local inference.
```

## How It Works

```
User Query
    │
    ▼
┌─────────────────────┐
│ Confidence Evaluator │  ← 4 signals + Thompson Sampling
│ (multi-signal router)│     learns which signals to trust
└────────┬────────────┘
         │
    ┌────┴────┐
    │         │
 confident  uncertain
    │         │
    ▼         ▼
 Local     Cloud Model
 Model     (cheapest first)
    │         │
    │         ├──→ Learn from response
    │         │    ├── Extract facts → Knowledge Store (STM → LTM)
    │         │    ├── Extract procedures → Skill Store
    │         │    └── Generate self-test questions
    │         │
    ▼         ▼
   Response to User
```

The agent escalates once, learns forever. Next time a similar question comes up, it answers locally — no cloud call, no cost, no data leaving your machine.

## Key Features

- **Multi-signal confidence routing** — 4 signals fused via Thompson Sampling (Bayesian bandit). The router learns which signals predict success and self-calibrates over time.
- **Tiered memory (STM/LTM)** — New knowledge enters short-term memory. Frequently accessed knowledge gets promoted to long-term memory. Unused knowledge decays via the Ebbinghaus forgetting curve. Just like a human brain.
- **Procedural skill store** — Learns not just facts but *how to do things*. Multi-step workflows extracted from cloud responses, versioned, with performance metrics.
- **Skill self-improvement** — Skills auto-evolve based on usage outcomes. Underperforming skills get rewritten. Previous versions retained for rollback.
- **Self-verification** — Periodically tests its own knowledge. Stale or incorrect entries get flagged and re-learned from the cloud.
- **Cost-aware cloud routing** — Tries the cheapest model first, fails over to more expensive ones. Tracks every dollar spent.
- **User/team profiling** — Builds a persistent model of preferences, vocabulary, and conventions. Personalizes responses across sessions.
- **Portable skill format** — Export/import skills as Markdown files. Share across agents and ecosystems.
- **Model-agnostic** — Works with any OpenAI-compatible API: Ollama, vLLM, OpenAI, Anthropic, etc. Swap models freely — knowledge persists.
- **Single-file state** — All knowledge, skills, metrics, and config in one SQLite file. Portable, backupable, inspectable.

## Quick Start

```bash
npm install Autodidact
```

```typescript
import { Agent } from 'Autodidact';

const agent = new Agent({
  localLLM: {
    baseUrl: 'http://localhost:11434/v1',  // Ollama
    model: 'llama3.2',
  },
  cloudRouter: {
    providers: [{
      name: 'openai',
      baseUrl: 'https://api.openai.com/v1',
      apiKey: process.env.OPENAI_API_KEY!,
      model: 'gpt-4o-mini',
      costPer1kTokens: 0.15,
      timeoutMs: 30000,
      priority: 1,
    }],
  },
  database: { path: './Autodidact.db' },
});

// First time: escalates to cloud, learns from response
const r1 = await agent.query('How do I deploy to staging?');
console.log(r1.content);       // detailed answer
console.log(r1.routing.decision); // 'ESCALATE'
console.log(r1.cost);          // $0.002

// Later: answers locally from learned knowledge
const r2 = await agent.query('Deploy to staging');
console.log(r2.routing.decision); // 'LOCAL'
console.log(r2.cost);          // $0.00
```

## Architecture

Autodidact is a framework, not an application. Every component is a TypeScript interface you can swap:

| Component | What it does |
|-----------|-------------|
| `ConfidenceEvaluator` | Multi-signal routing with Thompson Sampling |
| `KnowledgeStore` | Tiered STM/LTM with Ebbinghaus decay + vector search |
| `SkillStore` | Versioned procedural memory with performance metrics |
| `LearningExtractor` | Distills cloud responses into reusable knowledge + skills |
| `CloudRouter` | Cost-ordered provider failover |
| `SelfVerificationSystem` | Periodic knowledge validation + stale flagging |
| `SkillEvolver` | Auto-rewrites underperforming skills |
| `UserProfile` | Persistent preference/convention modeling |
| `LLMClient` | OpenAI-compatible API client (local or cloud) |
| `MetricsTracker` | Tracks local resolution rate, cost savings, calibration |

## Metrics That Matter

```typescript
const metrics = agent.getMetrics();

metrics.localResolutionRate   // 0.73 — 73% answered locally
metrics.cumulativeCostAvoided // $142.50 saved
metrics.selfTestPassRate      // 0.91 — 91% of knowledge verified correct
metrics.confidenceCalibration // 0.85 — router makes the right call 85% of the time
metrics.totalKnowledgeEntries // 847 facts learned
metrics.totalSkillEntries     // 23 procedures learned
```

## Configuration

Autodidact works with minimal config (just `localLLM` and `cloudRouter.providers`). Everything else has sensible defaults:

```typescript
const agent = new Agent({
  localLLM: { baseUrl: '...', model: '...' },
  cloudRouter: { providers: [...] },

  // Optional — all have defaults
  knowledgeStore: {
    stmTtlMs: 3_600_000,           // 1 hour STM window
    ltmBaseStabilityHours: 168,     // 7 day LTM base stability
    decayThreshold: 0.1,            // expire below 10% relevance
  },
  confidenceEvaluator: {
    localThreshold: 0.7,            // answer locally above 0.7
    hedgeThreshold: 0.4,            // hedge between 0.4-0.7
  },
  selfVerification: {
    intervalMs: 86_400_000,         // verify every 24 hours
    queryCountThreshold: 50,        // or every 50 queries
    batchSize: 20,
  },
  skillEvolver: {
    reviewThreshold: 10,            // review after 10 invocations
    minSuccessRate: 0.6,            // evolve below 60% success
  },
  userProfile: {
    defaultProfile: 'default',
    autoExtract: true,
  },
  database: { path: './Autodidact.db' },
});
```

## The Vision

**Phase 1** (now): Open-source framework. The self-learning engine that makes any AI agent smarter over time.

**Phase 2**: Full product. Enterprise agent with integrations (Slack, GitHub, Jira, Confluence). The "new employee" that never quits and never forgets.

**Phase 3**: Hive network. Agents teaching agents. Knowledge marketplace with Knowledge Tokens. A master agent in legal teaches your agent GDPR patterns. Your DevOps agent shares deployment skills with the community. The network effect that makes every agent smarter.

## Why Autodidact?

Every other AI agent is stateless. They forget everything after each conversation. They cost the same on day 300 as day 1. They send your data to the cloud every single time.

Autodidact remembers. It learns. It gets cheaper. It keeps your data local. And it gets measurably better every week — you can see the curve.

## Contributing

We welcome contributions. See the [spec docs](.kiro/specs/autodidact-framework/) for the full requirements, design, and implementation plan.

```bash
git clone https://github.com/BuffaloTechRider/Autodidact.git
cd Autodidact
npm install
npm test
```

## License

MIT

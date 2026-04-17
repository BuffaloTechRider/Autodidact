# Contributing to Autodidact

Thanks for your interest in contributing. Autodidact is an ambitious project — we're building the first self-learning AI agent framework with publishable research contributions. We need help from engineers, researchers, and builders.

## What We're Building

Autodidact is a local-first AI agent that learns from every cloud escalation. The core innovations are:
1. Thompson Sampling for confidence-based routing (novel — no published precedent)
2. Ebbinghaus-inspired memory consolidation (cognitive science meets AI)
3. Self-improving procedural memory (closed-loop skill optimization)
4. Autonomous tool discovery (agent learns HOW to do things, not just WHAT)

## How to Contribute

### Good First Issues

Look for issues labeled `good-first-issue`. These are scoped, well-defined tasks with clear acceptance criteria.

### Areas We Need Help

| Area | Skills Needed | Priority |
|------|--------------|----------|
| Python core (ChromaDB/FAISS) | Python, vector DBs | High |
| Benchmark suite | Python, ML evaluation | High |
| Energy scorer research | ML, embeddings, classification | High |
| TypeScript SDK improvements | TypeScript, Node.js | Medium |
| Multi-turn conversation | LLM prompting, context management | Medium |
| Documentation & examples | Technical writing | Medium |
| Tool registry expansion | APIs, HTTP, testing | Medium |
| Self-verification system | LLM evaluation, NLI | Medium |

### For Researchers

If you're interested in the research aspects:
- The Thompson Sampling router needs calibration benchmarks
- The Ebbinghaus decay model needs comparison against flat storage
- The energy scorer needs evaluation against other confidence methods
- The learning curve needs characterization across different domains

We're open to co-authoring papers on any of these contributions.

## Development Setup

### Python Core

```bash
git clone https://github.com/BuffaloTechRider/Autodidact.git
cd Autodidact

# Python setup
python -m venv venv
source venv/bin/activate
pip install -e ".[dev]"

# Run tests
pytest tests/

# Run benchmarks
python -m autodidact.benchmarks --all
```

### TypeScript SDK

```bash
cd prototype/
npm install
npx tsc --noEmit  # type check
npx vitest --run  # tests
```

### Prerequisites

- Python 3.10+
- Node.js 18+ (for TypeScript SDK)
- Ollama (for local model testing) — `ollama pull qwen2.5:7b && ollama pull nomic-embed-text`

## Code Style

### Python
- Black formatter (line length 100)
- Type hints on all public functions
- Docstrings on all public classes and methods
- Pydantic for data validation

### TypeScript
- Strict mode
- ESM modules
- Zod for runtime validation
- No `any` types

## Pull Request Process

1. Fork the repo and create a branch from `main`
2. Write your code with tests
3. Ensure all tests pass (`pytest` / `npx vitest --run`)
4. Update documentation if needed
5. Submit a PR with a clear description of what and why

### PR Title Format

```
feat: add BM25 hybrid search to Python knowledge store
fix: confidence evaluator threshold edge case
docs: add Thompson Sampling algorithm explanation
bench: add LongMemEval retrieval benchmark
refactor: extract signal computation into separate modules
```

### What Makes a Good PR

- Focused on one thing (not a kitchen sink)
- Has tests that prove it works
- Doesn't break existing tests
- Includes a brief explanation of the approach
- References the relevant requirement or issue

## Architecture Decisions

Major design decisions are documented in the spec:
- `.kiro/specs/autodidact-framework/requirements.md` — What we're building
- `.kiro/specs/autodidact-framework/design.md` — How we're building it
- `ROADMAP.md` — Where we're going

If you want to propose a significant architectural change, open a Discussion first.

## Looking for Co-Founders

We're actively looking for technical co-founders who are passionate about:
- Self-learning AI systems
- Local-first / privacy-preserving AI
- Bayesian methods and adaptive systems
- Building open-source developer tools

If this resonates, reach out via GitHub Discussions or open an issue tagged `co-founder`.

## Community

- GitHub Discussions — questions, ideas, proposals
- Issues — bugs, feature requests, good-first-issues
- PRs — code contributions

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

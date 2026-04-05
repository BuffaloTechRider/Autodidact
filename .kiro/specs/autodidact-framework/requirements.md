# Requirements Document

## Introduction

Autodidact is an open-source, local-first AI agent framework that enables developers to build agents which learn and improve over time. The framework operates on a "new employee" analogy: an agent starts with minimal knowledge, escalates uncertain queries to cloud/big models or web search, extracts reusable knowledge and skills from every escalation, and stores them in a persistent memory system. Over time, the agent resolves more queries locally — becoming more autonomous, cheaper to run, and more private.

Phase 1 delivers the core framework/SDK (no UI, no swarm network, no fine-tuning). All improvement comes from the memory and skill infrastructure around a frozen local model. The framework is model-agnostic, privacy-first, and pluggable into any system via a TypeScript/Node.js SDK.

## Glossary

- **Agent**: The main orchestrator that receives queries, routes them through the Confidence_Evaluator, and returns responses — learning from every cloud escalation.
- **Confidence_Evaluator**: The multi-signal routing component that fuses knowledge similarity, skill coverage, query complexity, and local model self-assessment to decide LOCAL, HEDGE, or ESCALATE.
- **Knowledge_Store**: The tiered persistent memory (STM + LTM) that stores factual knowledge entries with vector embeddings for similarity search.
- **STM**: Short-Term Memory tier within the Knowledge_Store. New learnings enter here and are promoted to LTM on repeated access.
- **LTM**: Long-Term Memory tier within the Knowledge_Store. Entries promoted from STM with longer TTL but eventual decay if unused.
- **Skill_Store**: The procedural memory that stores learned prompt chains, tool sequences, and multi-step action patterns with versioning and performance metrics.
- **Learning_Extractor**: The component that distills cloud escalation responses into reusable factual knowledge and procedural skills.
- **Cloud_Router**: The cost-aware model selection component that routes escalations to cloud LLM providers with failover from cheaper to more expensive models.
- **Self_Verification_System**: The component that periodically generates test questions against stored knowledge and flags stale or incorrect entries for refresh.
- **LLM_Client**: The unified client for communicating with any OpenAI-compatible LLM API (local or cloud).
- **Thompson_Sampling**: A Bayesian bandit algorithm used by the Confidence_Evaluator to learn optimal routing weights from outcome feedback.
- **Routing_Decision**: The output of the Confidence_Evaluator — one of LOCAL (answer from memory + local model), HEDGE (answer locally with uncertainty flag), or ESCALATE (cloud model or web search).
- **Learning_Artifact**: A factual claim or procedural skill extracted by the Learning_Extractor from a cloud escalation response.
- **Ebbinghaus_Decay**: The time-based decay pattern applied to memory entries — unused entries lose relevance and eventually expire.
- **Query**: A user question or task submitted to the Agent for processing.
- **Skill_Evolver**: The component that analyzes skill performance metrics and rewrites underperforming skills based on success/failure patterns.
- **User_Profile**: A persistent model of the user's or team's preferences, vocabulary, conventions, and interaction patterns that personalizes agent behavior across sessions.
- **Skill_Format**: The portable, open-standard format for skill definitions that enables interoperability and sharing across agent ecosystems.

## Requirements

### Requirement 1: Multi-Signal Confidence Evaluation and Routing

**User Story:** As a developer, I want the framework to intelligently route queries based on multiple confidence signals, so that the agent answers locally when confident and escalates only when necessary.

#### Acceptance Criteria

1. WHEN a Query is received, THE Confidence_Evaluator SHALL compute confidence signals from knowledge store similarity, skill coverage, query complexity, and local model self-assessment.
2. WHEN all confidence signals have been computed, THE Confidence_Evaluator SHALL fuse the signals into a Routing_Decision of LOCAL, HEDGE, or ESCALATE.
3. WHEN the Routing_Decision is LOCAL, THE Agent SHALL generate a response using the local model augmented with retrieved knowledge and skills without contacting any cloud provider.
4. WHEN the Routing_Decision is HEDGE, THE Agent SHALL generate a response using the local model and attach an uncertainty flag indicating reduced confidence.
5. WHEN the Routing_Decision is ESCALATE, THE Agent SHALL forward the Query to the Cloud_Router for cloud model processing.
6. THE Confidence_Evaluator SHALL use Thompson_Sampling to maintain and update per-signal routing weights based on outcome feedback from resolved queries.
7. WHEN a resolved query outcome is recorded, THE Confidence_Evaluator SHALL update the Thompson_Sampling parameters (alpha and beta) for each signal that contributed to the Routing_Decision.
8. THE Confidence_Evaluator SHALL expose the individual signal scores and the fused Routing_Decision as structured metadata alongside every response.

### Requirement 2: Tiered Knowledge Store with STM and LTM

**User Story:** As a developer, I want knowledge to be stored in a tiered memory system with natural decay, so that frequently-used knowledge persists while stale knowledge expires automatically.

#### Acceptance Criteria

1. WHEN the Learning_Extractor produces a new factual knowledge entry, THE Knowledge_Store SHALL insert the entry into STM with an initial TTL and usage count of zero.
2. WHILE a knowledge entry resides in STM, THE Knowledge_Store SHALL promote the entry to LTM when the entry is accessed within the configured promotion window.
3. WHILE a knowledge entry resides in STM and has not been accessed within the configured promotion window, THE Knowledge_Store SHALL expire and remove the entry.
4. WHILE a knowledge entry resides in LTM, THE Knowledge_Store SHALL apply Ebbinghaus_Decay to reduce the entry relevance score over time based on elapsed time since last access.
5. WHILE a knowledge entry resides in LTM and the relevance score falls below the configured decay threshold, THE Knowledge_Store SHALL expire and remove the entry.
6. WHEN a knowledge entry is accessed for retrieval, THE Knowledge_Store SHALL update the entry usage count and last-accessed timestamp.
7. THE Knowledge_Store SHALL store vector embeddings for each knowledge entry and support cosine similarity search for retrieval.
8. THE Knowledge_Store SHALL persist all knowledge entries and metadata (source, confidence, usage count, timestamps, tags) in SQLite via better-sqlite3.

### Requirement 3: Skill Store for Procedural Memory

**User Story:** As a developer, I want the framework to store learned procedures and action patterns, so that the agent can reuse multi-step workflows without re-escalating to the cloud.

#### Acceptance Criteria

1. WHEN the Learning_Extractor produces a new procedural skill, THE Skill_Store SHALL store the skill with its prompt chain, tool sequence, and action pattern.
2. THE Skill_Store SHALL version each skill entry and retain previous versions for rollback.
3. THE Skill_Store SHALL track performance metrics (success rate, average latency, invocation count) for each skill version.
4. WHEN a skill is invoked and completes, THE Skill_Store SHALL update the performance metrics for the invoked skill version.
5. WHEN a Query is received, THE Skill_Store SHALL support retrieval of matching skills by semantic similarity and tag-based lookup.
6. THE Skill_Store SHALL persist all skill entries and metadata in SQLite via better-sqlite3.

### Requirement 4: Learning Extraction from Cloud Escalations

**User Story:** As a developer, I want the framework to automatically extract reusable knowledge and skills from every cloud escalation, so that the agent learns and avoids repeating the same escalation.

#### Acceptance Criteria

1. WHEN the Cloud_Router returns a response for an escalated Query, THE Learning_Extractor SHALL analyze the response and extract factual claims as knowledge entries.
2. WHEN the Cloud_Router returns a response for an escalated Query, THE Learning_Extractor SHALL analyze the response and extract reasoning patterns or step-by-step procedures as skill entries.
3. WHEN the Learning_Extractor produces knowledge entries, THE Learning_Extractor SHALL assign a source attribution, initial confidence score, and relevant tags to each entry.
4. WHEN the Learning_Extractor produces skill entries, THE Learning_Extractor SHALL structure each skill as an ordered sequence of steps with input/output descriptions.
5. WHEN the Learning_Extractor completes extraction, THE Learning_Extractor SHALL generate self-test questions for each extracted knowledge entry to enable future self-verification.
6. IF the Learning_Extractor fails to parse or extract from a cloud response, THEN THE Learning_Extractor SHALL log the failure with the original response and continue processing without crashing.

### Requirement 5: Cost-Aware Cloud Routing with Failover

**User Story:** As a developer, I want cloud escalations to use the cheapest viable model first and fail over to more expensive models only when needed, so that the framework minimizes cloud API costs.

#### Acceptance Criteria

1. THE Cloud_Router SHALL maintain an ordered list of cloud model providers ranked by cost (cheapest first).
2. WHEN an escalation is requested, THE Cloud_Router SHALL attempt the cheapest available model provider first.
3. IF a cloud model provider returns an error or fails to respond within the configured timeout, THEN THE Cloud_Router SHALL attempt the next provider in the cost-ordered list.
4. IF all configured cloud model providers fail, THEN THE Cloud_Router SHALL return an error result with details of each provider failure.
5. WHEN a cloud escalation completes, THE Cloud_Router SHALL record the cost, latency, and provider used for the escalation.
6. THE Cloud_Router SHALL communicate with any OpenAI-compatible API (including Ollama, vLLM, OpenAI, and Anthropic endpoints) via the LLM_Client.

### Requirement 6: Self-Verification of Stored Knowledge

**User Story:** As a developer, I want the framework to periodically validate stored knowledge, so that stale or incorrect entries are flagged for refresh and the knowledge base stays accurate.

#### Acceptance Criteria

1. THE Self_Verification_System SHALL periodically select a batch of knowledge entries from the Knowledge_Store for verification based on a configurable schedule.
2. WHEN a batch of knowledge entries is selected for verification, THE Self_Verification_System SHALL generate test questions for each entry (or use previously generated self-test questions from the Learning_Extractor).
3. WHEN a test question is generated, THE Self_Verification_System SHALL submit the question to the local model and compare the response against the stored knowledge entry.
4. IF the local model response contradicts or fails to confirm the stored knowledge entry, THEN THE Self_Verification_System SHALL flag the entry for refresh by marking it as stale.
5. WHEN a knowledge entry is flagged as stale, THE Self_Verification_System SHALL queue the entry for re-escalation to the Cloud_Router to obtain updated information.
6. THE Self_Verification_System SHALL record the self-test pass rate as a metric for knowledge quality monitoring.

### Requirement 7: Model-Agnostic Local LLM Integration

**User Story:** As a developer, I want the framework to work with any OpenAI-compatible local model, so that I can swap models freely without losing accumulated knowledge or skills.

#### Acceptance Criteria

1. THE LLM_Client SHALL communicate with any LLM endpoint that implements the OpenAI chat completions API (including Ollama, vLLM, and OpenAI-compatible servers).
2. THE LLM_Client SHALL accept a base URL and API key as configuration, with no hard-coded provider assumptions.
3. WHEN the LLM_Client sends a request and receives no response within the configured timeout, THE LLM_Client SHALL return a timeout error without blocking the calling thread.
4. IF the LLM_Client receives a malformed or unexpected response from the LLM endpoint, THEN THE LLM_Client SHALL return a structured error with the raw response for debugging.
5. THE LLM_Client SHALL validate all request and response payloads using Zod schemas.
6. WHEN the local model is swapped for a different model, THE Knowledge_Store and Skill_Store SHALL retain all previously stored entries without data loss.

### Requirement 8: Agent Orchestration and Query Lifecycle

**User Story:** As a developer, I want a single Agent entry point that orchestrates the full query lifecycle from routing through response and learning, so that I can integrate the framework with minimal boilerplate.

#### Acceptance Criteria

1. WHEN a Query is submitted to the Agent, THE Agent SHALL pass the Query through the Confidence_Evaluator to obtain a Routing_Decision.
2. WHEN the Routing_Decision is LOCAL or HEDGE, THE Agent SHALL retrieve relevant knowledge from the Knowledge_Store and matching skills from the Skill_Store, then generate a response via the local LLM_Client.
3. WHEN the Routing_Decision is ESCALATE, THE Agent SHALL forward the Query to the Cloud_Router, receive the cloud response, and pass the response to the Learning_Extractor for knowledge and skill extraction.
4. WHEN the Learning_Extractor completes extraction from an escalated Query, THE Agent SHALL store the extracted knowledge entries in the Knowledge_Store and skill entries in the Skill_Store.
5. WHEN a Query lifecycle completes, THE Agent SHALL record the outcome (routing decision, response quality, latency, cost) and feed the outcome back to the Confidence_Evaluator for Thompson_Sampling updates.
6. THE Agent SHALL expose a programmatic API (TypeScript functions and types) for submitting queries, configuring components, and retrieving metrics.
7. IF any component in the query lifecycle throws an unhandled error, THEN THE Agent SHALL catch the error, log diagnostic information, and return a structured error response to the caller.

### Requirement 9: Observability and Improvement Metrics

**User Story:** As a developer, I want the framework to track key performance metrics, so that I can measure how the agent improves over time and identify areas for optimization.

#### Acceptance Criteria

1. THE Agent SHALL track and expose the local resolution rate (percentage of queries resolved without cloud escalation) as a running metric.
2. THE Agent SHALL track and expose the knowledge growth rate (new knowledge entries added per time period) as a running metric.
3. THE Agent SHALL track and expose the cumulative cloud API cost avoided (estimated cost savings from local resolutions) as a running metric.
4. THE Agent SHALL track and expose the self-test pass rate (percentage of self-verification tests passed) from the Self_Verification_System.
5. THE Agent SHALL track and expose the confidence calibration accuracy (percentage of Routing_Decisions that led to successful outcomes) as a running metric.
6. THE Agent SHALL persist all metrics in SQLite so that historical trends survive process restarts.

### Requirement 10: Framework Configuration and Extensibility

**User Story:** As a developer, I want the framework to be configurable with sensible defaults and extensible through well-defined interfaces, so that I can customize behavior without modifying framework internals.

#### Acceptance Criteria

1. THE Agent SHALL accept a configuration object at initialization that specifies settings for all components (Confidence_Evaluator thresholds, Knowledge_Store TTLs, Cloud_Router provider list, Self_Verification_System schedule).
2. THE Agent SHALL apply sensible default values for all configuration settings so that the framework operates correctly with minimal configuration.
3. THE Agent SHALL validate the configuration object using Zod schemas at initialization and return descriptive errors for invalid configurations.
4. THE Agent SHALL define TypeScript interfaces for each pluggable component (LLM_Client, Knowledge_Store, Skill_Store, Cloud_Router) so that developers can provide custom implementations.
5. WHEN a custom component implementation is provided, THE Agent SHALL use the custom implementation in place of the default without requiring changes to other components.

### Requirement 11: Skill Self-Improvement Through Usage Analysis

**User Story:** As a developer, I want skills to automatically improve based on their usage outcomes, so that the agent's procedural knowledge gets refined without manual intervention.

#### Acceptance Criteria

1. WHEN a skill's invocation count reaches the configured review threshold (default: 10 invocations), THE Skill_Evolver SHALL analyze the skill's success/failure patterns and execution traces.
2. WHEN the Skill_Evolver determines that a skill's success rate has fallen below the configured minimum (default: 0.6), THE Skill_Evolver SHALL generate a revised version of the skill's prompt chain using the local LLM with the failure patterns as context.
3. WHEN the Skill_Evolver generates a revised skill version, THE Skill_Store SHALL store the revision as a new version while retaining the previous version for rollback.
4. WHEN a revised skill version is created, THE Skill_Store SHALL reset the new version's performance metrics to zero while preserving the previous version's metrics.
5. IF the Skill_Evolver fails to generate a valid revised skill, THEN THE Skill_Evolver SHALL log the failure and retain the current skill version unchanged.
6. THE Skill_Evolver SHALL also trigger a review when a skill's success rate drops below the configured minimum regardless of invocation count.

### Requirement 12: User and Team Profile Modeling

**User Story:** As a developer, I want the framework to build a persistent model of the user's or team's preferences and patterns, so that the agent personalizes its responses and actions across sessions.

#### Acceptance Criteria

1. THE Agent SHALL maintain a User_Profile that persists across sessions and stores observed preferences, vocabulary, conventions, and interaction patterns.
2. WHEN the Agent processes a Query, THE Agent SHALL update the User_Profile with any observed preferences or patterns (e.g., preferred response format, domain-specific terminology, tool preferences).
3. WHEN generating a response, THE Agent SHALL incorporate relevant User_Profile context into the local LLM prompt to personalize the output.
4. THE User_Profile SHALL be stored in SQLite alongside other framework state and survive process restarts.
5. THE Agent SHALL expose an API for reading, updating, and resetting the User_Profile programmatically.
6. THE User_Profile SHALL support multiple named profiles so that the framework can model different users or teams within the same deployment.

### Requirement 13: Portable Open-Standard Skill Format

**User Story:** As a developer, I want skills to be stored in a portable, open-standard format, so that skills can be shared, exported, and imported across agent ecosystems.

#### Acceptance Criteria

1. THE Skill_Store SHALL serialize each skill entry in a self-contained Markdown-based Skill_Format with YAML frontmatter for metadata and Markdown body for the procedure.
2. THE Skill_Store SHALL support exporting individual skills or all skills to the Skill_Format as standalone files.
3. THE Skill_Store SHALL support importing skills from Skill_Format files into the local Skill_Store.
4. THE Skill_Format SHALL include all fields necessary to reconstruct the skill: name, description, tags, steps (with input/output), version, and performance metrics.
5. WHEN importing a skill that conflicts with an existing skill by name, THE Skill_Store SHALL create a new version of the existing skill rather than overwriting it.

### Requirement 14: Query-Count-Based Verification Triggers

**User Story:** As a developer, I want self-verification to trigger based on query count in addition to time intervals, so that actively-used agents verify knowledge more frequently.

#### Acceptance Criteria

1. THE Self_Verification_System SHALL support triggering verification cycles based on a configurable query count threshold (default: every 50 queries) in addition to the time-based schedule.
2. WHEN the query count since the last verification cycle reaches the configured threshold, THE Self_Verification_System SHALL initiate a verification cycle.
3. WHEN a verification cycle is triggered by query count, THE Self_Verification_System SHALL reset the query counter and the time-based schedule timer.
4. THE Self_Verification_System SHALL expose the current query count since last verification as a metric.

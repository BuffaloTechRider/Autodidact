import Database from 'better-sqlite3';

const SCHEMA_V1 = `
-- ── Knowledge Store ─────────────────────────────────────────
CREATE TABLE IF NOT EXISTS knowledge_entries (
  id            TEXT PRIMARY KEY,
  content       TEXT NOT NULL,
  source        TEXT NOT NULL,
  confidence    REAL NOT NULL DEFAULT 0.5,
  tags          TEXT NOT NULL DEFAULT '[]',
  embedding     BLOB,
  tier          TEXT NOT NULL DEFAULT 'STM',
  usage_count   INTEGER NOT NULL DEFAULT 0,
  created_at    TEXT NOT NULL,
  last_accessed TEXT NOT NULL,
  promoted_at   TEXT,
  is_stale      INTEGER NOT NULL DEFAULT 0,
  self_test_questions TEXT NOT NULL DEFAULT '[]',
  metadata      TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_knowledge_tier ON knowledge_entries(tier);
CREATE INDEX IF NOT EXISTS idx_knowledge_stale ON knowledge_entries(is_stale);
CREATE INDEX IF NOT EXISTS idx_knowledge_last_accessed ON knowledge_entries(last_accessed);

-- ── Skill Store ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS skill_entries (
  id              TEXT PRIMARY KEY,
  name            TEXT NOT NULL,
  description     TEXT NOT NULL,
  steps           TEXT NOT NULL,
  tags            TEXT NOT NULL DEFAULT '[]',
  embedding       BLOB,
  version         INTEGER NOT NULL DEFAULT 1,
  parent_id       TEXT,
  success_count   INTEGER NOT NULL DEFAULT 0,
  failure_count   INTEGER NOT NULL DEFAULT 0,
  total_latency_ms INTEGER NOT NULL DEFAULT 0,
  invocation_count INTEGER NOT NULL DEFAULT 0,
  created_at      TEXT NOT NULL,
  updated_at      TEXT NOT NULL,
  metadata        TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_skill_name ON skill_entries(name);
CREATE INDEX IF NOT EXISTS idx_skill_version ON skill_entries(name, version);

-- ── Thompson Sampling Parameters ────────────────────────────
CREATE TABLE IF NOT EXISTS thompson_params (
  signal_name TEXT PRIMARY KEY,
  alpha       REAL NOT NULL DEFAULT 1.0,
  beta        REAL NOT NULL DEFAULT 1.0,
  updated_at  TEXT NOT NULL
);

-- ── Cloud Escalation Log ────────────────────────────────────
CREATE TABLE IF NOT EXISTS escalation_log (
  id          TEXT PRIMARY KEY,
  query_id    TEXT NOT NULL,
  provider    TEXT NOT NULL,
  model       TEXT NOT NULL,
  cost        REAL NOT NULL DEFAULT 0.0,
  latency_ms  INTEGER NOT NULL,
  success     INTEGER NOT NULL,
  error       TEXT,
  created_at  TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_escalation_query ON escalation_log(query_id);

-- ── Query Outcome Log ───────────────────────────────────────
CREATE TABLE IF NOT EXISTS query_log (
  id              TEXT PRIMARY KEY,
  query_text      TEXT NOT NULL,
  routing_decision TEXT NOT NULL,
  signals         TEXT NOT NULL,
  fused_score     REAL NOT NULL,
  outcome         TEXT,
  response_text   TEXT,
  cost            REAL NOT NULL DEFAULT 0.0,
  latency_ms      INTEGER NOT NULL,
  created_at      TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_query_routing ON query_log(routing_decision);
CREATE INDEX IF NOT EXISTS idx_query_created ON query_log(created_at);

-- ── Metrics ─────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS metrics (
  id              TEXT PRIMARY KEY,
  metric_name     TEXT NOT NULL,
  metric_value    REAL NOT NULL,
  period_start    TEXT NOT NULL,
  period_end      TEXT NOT NULL,
  created_at      TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_metrics_name ON metrics(metric_name);
CREATE INDEX IF NOT EXISTS idx_metrics_period ON metrics(period_start, period_end);

-- ── Self-Verification Log ───────────────────────────────────
CREATE TABLE IF NOT EXISTS verification_log (
  id              TEXT PRIMARY KEY,
  knowledge_id    TEXT NOT NULL,
  question        TEXT NOT NULL,
  model_answer    TEXT NOT NULL,
  passed          INTEGER NOT NULL,
  created_at      TEXT NOT NULL,
  FOREIGN KEY (knowledge_id) REFERENCES knowledge_entries(id)
);

CREATE INDEX IF NOT EXISTS idx_verification_knowledge ON verification_log(knowledge_id);

-- ── User Profiles ───────────────────────────────────────────
CREATE TABLE IF NOT EXISTS user_profiles (
  name            TEXT PRIMARY KEY,
  preferences     TEXT NOT NULL DEFAULT '{}',
  vocabulary      TEXT NOT NULL DEFAULT '[]',
  conventions     TEXT NOT NULL DEFAULT '[]',
  interaction_count INTEGER NOT NULL DEFAULT 0,
  created_at      TEXT NOT NULL,
  updated_at      TEXT NOT NULL
);

-- ── Skill Evolution Log ─────────────────────────────────────
CREATE TABLE IF NOT EXISTS skill_evolution_log (
  id              TEXT PRIMARY KEY,
  skill_id        TEXT NOT NULL,
  skill_name      TEXT NOT NULL,
  previous_version INTEGER NOT NULL,
  new_version     INTEGER,
  action          TEXT NOT NULL,
  reason          TEXT NOT NULL,
  created_at      TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_evolution_skill ON skill_evolution_log(skill_id);
`;

export function initDatabase(dbPath: string): Database.Database {
    const db = new Database(dbPath);

    // Enable WAL mode for concurrent reads
    db.pragma('journal_mode = WAL');

    const currentVersion = db.pragma('user_version', { simple: true }) as number;

    if (currentVersion < 1) {
        db.exec(SCHEMA_V1);
        db.pragma('user_version = 1');
    }

    return db;
}

"""SQLite schema initialization for Autodidact."""

import sqlite3


SCHEMA_SQL = """
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS knowledge_entries (
    id                TEXT PRIMARY KEY,
    content           TEXT NOT NULL,
    source            TEXT NOT NULL CHECK(source IN ('cloud_escalation','manual','self_verification')),
    confidence        REAL NOT NULL DEFAULT 0.5,
    tags              TEXT NOT NULL DEFAULT '[]',
    embedding         BLOB,
    tier              TEXT NOT NULL DEFAULT 'STM' CHECK(tier IN ('STM','LTM')),
    usage_count       INTEGER NOT NULL DEFAULT 0,
    created_at        TEXT NOT NULL,
    last_accessed     TEXT NOT NULL,
    promoted_at       TEXT,
    metadata          TEXT NOT NULL DEFAULT '{}',
    domain            TEXT NOT NULL DEFAULT 'general',
    topic             TEXT NOT NULL DEFAULT 'uncategorized',
    category          TEXT NOT NULL DEFAULT 'facts'
                      CHECK(category IN ('facts','events','discoveries','preferences','advice')),
    valid_from        TEXT NOT NULL,
    valid_to          TEXT,
    verbatim_response TEXT
);

CREATE INDEX IF NOT EXISTS idx_ke_tier ON knowledge_entries(tier);
CREATE INDEX IF NOT EXISTS idx_ke_last_accessed ON knowledge_entries(last_accessed);
CREATE INDEX IF NOT EXISTS idx_ke_domain ON knowledge_entries(domain);
CREATE INDEX IF NOT EXISTS idx_ke_topic ON knowledge_entries(topic);
CREATE INDEX IF NOT EXISTS idx_ke_category ON knowledge_entries(category);
CREATE INDEX IF NOT EXISTS idx_ke_domain_topic ON knowledge_entries(domain, topic);
CREATE INDEX IF NOT EXISTS idx_ke_domain_topic_cat ON knowledge_entries(domain, topic, category);
CREATE INDEX IF NOT EXISTS idx_ke_valid_from ON knowledge_entries(valid_from);
CREATE INDEX IF NOT EXISTS idx_ke_valid_to ON knowledge_entries(valid_to);
CREATE INDEX IF NOT EXISTS idx_ke_source ON knowledge_entries(source);

CREATE TABLE IF NOT EXISTS thompson_params (
    signal_name       TEXT PRIMARY KEY,
    alpha             REAL NOT NULL DEFAULT 1.0,
    beta_param        REAL NOT NULL DEFAULT 1.0,
    updated_at        TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS energy_scorer_examples (
    id                TEXT PRIMARY KEY,
    query_text        TEXT NOT NULL,
    query_embedding   BLOB NOT NULL,
    outcome           TEXT NOT NULL CHECK(outcome IN ('pass','fail')),
    created_at        TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_es_outcome ON energy_scorer_examples(outcome);

CREATE TABLE IF NOT EXISTS energy_scorer_model (
    id                INTEGER PRIMARY KEY CHECK(id = 1),
    weights           BLOB NOT NULL,
    bias              REAL NOT NULL,
    example_count     INTEGER NOT NULL,
    trained_at        TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS query_log (
    id                TEXT PRIMARY KEY,
    session_id        TEXT,
    query_text        TEXT NOT NULL,
    routing_decision  TEXT NOT NULL,
    signals           TEXT NOT NULL,
    fusion_weights    TEXT NOT NULL,
    fused_score       REAL NOT NULL,
    outcome           TEXT,
    response_text     TEXT,
    cost              REAL NOT NULL DEFAULT 0.0,
    latency_ms        INTEGER NOT NULL,
    provider          TEXT,
    created_at        TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_ql_routing ON query_log(routing_decision);
CREATE INDEX IF NOT EXISTS idx_ql_created ON query_log(created_at);
CREATE INDEX IF NOT EXISTS idx_ql_session ON query_log(session_id);
CREATE INDEX IF NOT EXISTS idx_ql_outcome ON query_log(outcome);
"""


def init_database(db_path: str = "autodidact.db") -> sqlite3.Connection:
    """Initialize the SQLite database with the Autodidact schema."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA_SQL)
    return conn

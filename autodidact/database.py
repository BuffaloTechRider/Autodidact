"""SQLite schema initialization for Autodidact."""

import sqlite3


SCHEMA_SQL = """
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS knowledge_entries (
    id                TEXT PRIMARY KEY,
    content           TEXT NOT NULL,
    question          TEXT,
    source            TEXT NOT NULL CHECK(source IN ('cloud_escalation','manual','self_verification')),
    confidence        REAL NOT NULL DEFAULT 0.5,
    tags              TEXT NOT NULL DEFAULT '[]',
    embedding         BLOB,
    answer_embedding  BLOB,
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

-- ── Experiment tables (v0.1 ablation experiment) ──────────────────

CREATE TABLE IF NOT EXISTS experiment_results (
    id                                TEXT PRIMARY KEY,
    run_id                            TEXT NOT NULL,
    query_index                       INTEGER NOT NULL,
    query_id                          TEXT NOT NULL,
    query_text                        TEXT NOT NULL,
    ground_truth                      TEXT NOT NULL,
    category                          TEXT NOT NULL,

    -- Signals (0..1 unless null)
    knowledge_similarity              REAL NOT NULL,
    query_classification              REAL NOT NULL,
    energy_scorer                     REAL,
    grounded_self_assessment          REAL NOT NULL,
    logprob_uncertainty               REAL NOT NULL,
    self_consistency                  REAL NOT NULL,

    -- Per-signal latency (ms)
    latency_knowledge_similarity      INTEGER NOT NULL,
    latency_query_classification      INTEGER NOT NULL,
    latency_energy_scorer             INTEGER,
    latency_grounded_self_assessment  INTEGER NOT NULL,
    latency_logprob_uncertainty       INTEGER NOT NULL,
    latency_self_consistency          INTEGER NOT NULL,

    -- Model outputs
    local_answer                      TEXT NOT NULL,
    local_avg_logprob                 REAL,
    cloud_answer                      TEXT NOT NULL,

    -- RouteLLM baseline scores (frozen classifiers applied per query)
    routellm_no_memory                REAL NOT NULL,
    routellm_plus_ks                  REAL NOT NULL,

    -- Retrieval quality proxy (1 if any top-5 hit shares the query's MMLU-Pro category, else 0)
    retrieval_recall_at_5             INTEGER NOT NULL DEFAULT 0,

    -- Diagnostic: how the grounded_self_assessment signal was extracted per query.
    -- One of 'logprob_softmax', 'text_hard', 'neutral'. Helps the memo stratify
    -- AUROC by extraction mode so we can separate "signal genuinely works" from
    -- "signal was mostly neutral fill-in".
    gsa_extraction_mode               TEXT,

    -- Correctness (cloud-determined)
    local_correct                     INTEGER NOT NULL,
    cloud_correct                     INTEGER NOT NULL,
    judge_used                        INTEGER NOT NULL DEFAULT 0,

    -- Cost accounting
    cost_cloud_answer_usd             REAL NOT NULL DEFAULT 0.0,
    cost_judge_usd                    REAL NOT NULL DEFAULT 0.0,
    cost_local_usd                    REAL NOT NULL DEFAULT 0.0,

    -- Error surface (populated when a row was recorded with a partial failure)
    error_info                        TEXT,

    created_at                        TEXT NOT NULL,
    UNIQUE(run_id, query_index)
);

CREATE INDEX IF NOT EXISTS idx_er_run ON experiment_results(run_id);
CREATE INDEX IF NOT EXISTS idx_er_run_category ON experiment_results(run_id, category);

-- RouteLLM baseline training data (cached labels for the disjoint 1000-query training set)
CREATE TABLE IF NOT EXISTS routellm_training_rows (
    id                   TEXT PRIMARY KEY,
    training_seed        INTEGER NOT NULL,
    local_model          TEXT NOT NULL DEFAULT '',
    query_id             TEXT NOT NULL,
    query_text           TEXT NOT NULL,
    query_embedding      BLOB NOT NULL,
    knowledge_similarity REAL NOT NULL,
    local_answer         TEXT NOT NULL,
    cloud_answer         TEXT NOT NULL,
    local_correct        INTEGER NOT NULL,
    cloud_correct        INTEGER NOT NULL,
    local_is_sufficient  INTEGER NOT NULL,
    cost_cloud_usd       REAL NOT NULL DEFAULT 0.0,
    cost_judge_usd       REAL NOT NULL DEFAULT 0.0,
    created_at           TEXT NOT NULL,
    UNIQUE(training_seed, local_model, query_id)
);

CREATE INDEX IF NOT EXISTS idx_rltr_seed ON routellm_training_rows(training_seed);
CREATE INDEX IF NOT EXISTS idx_rltr_seed_model ON routellm_training_rows(training_seed, local_model);

-- ── Document store (R9: cold start fix) ────────────────────────────
-- Separate from knowledge_entries per AD-002. Documents answer
-- "what do the source materials say?"; knowledge_entries answer
-- "have I been asked this before?"

CREATE TABLE IF NOT EXISTS document_chunks (
    id                TEXT PRIMARY KEY,
    content           TEXT NOT NULL,
    source_file       TEXT NOT NULL,
    chunk_index       INTEGER NOT NULL,
    embedding         BLOB,
    tags              TEXT NOT NULL DEFAULT '[]',
    created_at        TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_dc_source_file ON document_chunks(source_file);
CREATE INDEX IF NOT EXISTS idx_dc_created_at ON document_chunks(created_at);

-- ── FTS5 full-text search for document chunks (v1.5: hybrid retrieval) ──
-- External-content FTS5 table kept in sync via triggers.
-- Enables BM25 keyword search alongside vector similarity.
CREATE VIRTUAL TABLE IF NOT EXISTS document_chunks_fts USING fts5(
    content,
    source_file,
    content='document_chunks',
    content_rowid='rowid',
    tokenize='porter unicode61'
);

CREATE TRIGGER IF NOT EXISTS dc_fts_insert AFTER INSERT ON document_chunks BEGIN
    INSERT INTO document_chunks_fts(rowid, content, source_file)
    VALUES (new.rowid, new.content, new.source_file);
END;

CREATE TRIGGER IF NOT EXISTS dc_fts_delete AFTER DELETE ON document_chunks BEGIN
    INSERT INTO document_chunks_fts(document_chunks_fts, rowid, content, source_file)
    VALUES ('delete', old.rowid, old.content, old.source_file);
END;

CREATE TRIGGER IF NOT EXISTS dc_fts_update AFTER UPDATE ON document_chunks BEGIN
    INSERT INTO document_chunks_fts(document_chunks_fts, rowid, content, source_file)
    VALUES ('delete', old.rowid, old.content, old.source_file);
    INSERT INTO document_chunks_fts(rowid, content, source_file)
    VALUES (new.rowid, new.content, new.source_file);
END;
"""


def init_database(db_path: str = "autodidact.db") -> sqlite3.Connection:
    """Initialize the SQLite database with the Autodidact schema."""
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row

    # Backfill migrations FIRST, before executescript -
    # Reason: SCHEMA_SQL includes CREATE INDEX on columns we added later
    # (e.g. routellm_training_rows.local_model). If the table already exists
    # in an older DB without that column, the CREATE INDEX inside the
    # executescript call will fail. Run ALTERs first so the columns exist
    # before indexes reference them. Each ALTER is idempotent via try/except.
    try:
        conn.execute("ALTER TABLE knowledge_entries ADD COLUMN answer_embedding BLOB")
        conn.commit()
    except sqlite3.OperationalError:
        # Table doesn't exist yet, or column already present. Both are fine —
        # executescript below will handle the table-doesn't-exist case.
        pass
    try:
        conn.execute(
            "ALTER TABLE routellm_training_rows ADD COLUMN local_model TEXT NOT NULL DEFAULT ''"
        )
        conn.commit()
    except sqlite3.OperationalError:
        pass

    # UNIQUE constraint migration. The original schema had
    # UNIQUE(training_seed, query_id). The cross-model Task 14 setup needs
    # UNIQUE(training_seed, local_model, query_id). SQLite can't change a
    # UNIQUE constraint in-place — we detect the old shape and rebuild.
    try:
        schema_row = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='routellm_training_rows'"
        ).fetchone()
        if schema_row is not None:
            sql = schema_row["sql"] or ""
            has_new_constraint = "UNIQUE(training_seed, local_model, query_id)" in sql
            table_exists = True
        else:
            has_new_constraint = False
            table_exists = False

        if table_exists and not has_new_constraint:
            # Rebuild: copy rows with valid local_model tags only. Orphans
            # (empty local_model, from pre-migration aborted runs) are dropped
            # — they can't be matched to a model.
            conn.executescript("""
                BEGIN;
                CREATE TABLE routellm_training_rows_new (
                    id                   TEXT PRIMARY KEY,
                    training_seed        INTEGER NOT NULL,
                    local_model          TEXT NOT NULL DEFAULT '',
                    query_id             TEXT NOT NULL,
                    query_text           TEXT NOT NULL,
                    query_embedding      BLOB NOT NULL,
                    knowledge_similarity REAL NOT NULL,
                    local_answer         TEXT NOT NULL,
                    cloud_answer         TEXT NOT NULL,
                    local_correct        INTEGER NOT NULL,
                    cloud_correct        INTEGER NOT NULL,
                    local_is_sufficient  INTEGER NOT NULL,
                    cost_cloud_usd       REAL NOT NULL DEFAULT 0.0,
                    cost_judge_usd       REAL NOT NULL DEFAULT 0.0,
                    created_at           TEXT NOT NULL,
                    UNIQUE(training_seed, local_model, query_id)
                );
                INSERT INTO routellm_training_rows_new
                SELECT id, training_seed, local_model, query_id, query_text,
                       query_embedding, knowledge_similarity, local_answer,
                       cloud_answer, local_correct, cloud_correct,
                       local_is_sufficient, cost_cloud_usd, cost_judge_usd,
                       created_at
                FROM routellm_training_rows
                WHERE local_model != '';
                DROP TABLE routellm_training_rows;
                ALTER TABLE routellm_training_rows_new RENAME TO routellm_training_rows;
                COMMIT;
            """)
    except sqlite3.OperationalError:
        # Any unexpected shape is fine — executescript below will run and
        # CREATE TABLE IF NOT EXISTS handles fresh DBs.
        pass

    conn.executescript(SCHEMA_SQL)

    # Backfill FTS5 for existing document_chunks (users upgrading from v1.0).
    try:
        fts_count = conn.execute(
            "SELECT COUNT(*) FROM document_chunks_fts"
        ).fetchone()[0]
        chunk_count = conn.execute(
            "SELECT COUNT(*) FROM document_chunks"
        ).fetchone()[0]
        if chunk_count > 0 and fts_count == 0:
            conn.execute(
                "INSERT INTO document_chunks_fts(rowid, content, source_file) "
                "SELECT rowid, content, source_file FROM document_chunks"
            )
            conn.commit()
    except sqlite3.OperationalError:
        pass

    return conn

-- CORE SQLITE SCHEMA (shareable reference)
-- Derived from the actual /mnt/data/schema supplied 15 October 2026.
-- Not a schema migration, not a complete production database dump.
-- No personal records or authentication tokens are included.
--
-- Scope: raw uploaded conversation chunks, explicit chunk-character tags,
-- entity/name cards, durable atomic facts, episodes, Wilson retrieval
-- configuration, FTS5 search, BGE-small 384-d vector indexes and an optional
-- source-summaries placeholder.
--
--
-- Requirements: SQLite with FTS5 and sqlite-vec extension (vec0) LOADED
-- on this SQLite CONNECTION before executing CREATE VIRTUAL TABLE ... vec0.
-- This file defines structure only; populated records, model embeddings,
-- retrieval logic and source processing live in application code.
--
-- Important maintenance rules drawn from the live implementation:
--   * chunks_fts is external-content FTS5 and is maintained by load_file.py
--     (NOT by a SQLite trigger in this schema).
--   * memory_facts_fts is maintained by triggers below.
--   * vectors are generated in Python and inserted by matching rowid:
--       chunks_vec.rowid = chunks.rowid
--       memory_facts_vec.rowid = memory_facts.id
--   * retrieval_chunk_vec_sources.active=1 makes source vectors eligible.
--   * source_summaries is a structural placeholder; no summarisation
--     pipeline is specified here.
--   * Empty configuration tables need application-specific rows to operate.

PRAGMA foreign_keys = ON;

-- =====================================================================
-- 1. Raw uploaded chat sources and literal retrieval
-- =====================================================================
CREATE TABLE chunks (
  id            TEXT PRIMARY KEY,
  source        TEXT NOT NULL,
  source_ord    INTEGER NOT NULL DEFAULT 0,
  batch         TEXT,
  path          TEXT,
  loaded_at     TEXT,
  event_date    TEXT,
  yymm          TEXT,
  body          TEXT NOT NULL
);

CREATE INDEX idx_chunks_source_ord ON chunks(source_ord DESC);

CREATE INDEX idx_chunks_source     ON chunks(source);

CREATE INDEX idx_chunks_yymm       ON chunks(yymm);

CREATE INDEX idx_chunks_event_date ON chunks(event_date);

CREATE TABLE chunk_chars (
  chunk_id  TEXT NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
  name      TEXT NOT NULL,
  PRIMARY KEY (chunk_id, name)
);

CREATE INDEX idx_chunk_chars_name ON chunk_chars(name);

CREATE VIRTUAL TABLE chunks_fts USING fts5(
  body,
  content='chunks',
  content_rowid='rowid'
);


-- =====================================================================
-- 2. People, names and source associations
-- =====================================================================
CREATE TABLE entities (
  id              INTEGER PRIMARY KEY AUTOINCREMENT,
  canonical_name  TEXT NOT NULL COLLATE NOCASE UNIQUE,
  entity_type     TEXT NOT NULL,                 -- 'character' | 'theme' | 'event' | 'place' | 'concept' | …
  aliases         TEXT,                          -- JSON array of alternate names
  description     TEXT,                          -- short stable profile / definition
  first_ord       INTEGER,
  last_ord        INTEGER,
  mention_count   INTEGER DEFAULT 0,
  importance      REAL DEFAULT 0.5,
  updated_at      TEXT
);

CREATE INDEX idx_ent_type ON entities(entity_type);

CREATE INDEX idx_ent_name ON entities(canonical_name);

CREATE INDEX idx_ent_importance ON entities(importance DESC);

CREATE TABLE entity_sources (
  entity_id       INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
  source          TEXT NOT NULL,
  prominence      TEXT DEFAULT 'mentioned',      -- 'central' | 'major' | 'mentioned'
  notes           TEXT,
  PRIMARY KEY (entity_id, source)
);

CREATE INDEX idx_es_source ON entity_sources(source);


-- =====================================================================
-- 3. Episode grouping and durable atomic facts
-- =====================================================================
CREATE TABLE retrieval_episodes (
  id                 INTEGER PRIMARY KEY AUTOINCREMENT,
  title              TEXT    NOT NULL,
  stream             TEXT    NOT NULL DEFAULT 'mixed',
  event_date         TEXT,
  entities           TEXT,
  canonical_fact_id  INTEGER,
  status             TEXT    NOT NULL DEFAULT 'live',
  notes              TEXT,
  created_at         TEXT    NOT NULL,
  updated_at         TEXT    NOT NULL
);

CREATE INDEX idx_episodes_stream
  ON retrieval_episodes(stream, status);

CREATE INDEX idx_episodes_date
  ON retrieval_episodes(event_date);

CREATE UNIQUE INDEX idx_episodes_title
  ON retrieval_episodes(title);

CREATE TABLE memory_facts (
  id                  INTEGER PRIMARY KEY AUTOINCREMENT,
  category            TEXT,                      -- relationship | timeline | preference | event | belief
  fact                TEXT NOT NULL,
  entities            TEXT,                      -- JSON array of names
  supporting_sources  TEXT,                      -- JSON array of sources / ords
  confidence          REAL DEFAULT 0.8,
  created_at          TEXT,
  first_seen_ord      INTEGER,
  last_seen_ord       INTEGER,
  updated_at          TEXT,
  episode_id          INTEGER REFERENCES retrieval_episodes(id)
);

CREATE INDEX idx_mf_category ON memory_facts(category);

CREATE INDEX idx_mf_episode
  ON memory_facts(episode_id);

CREATE VIRTUAL TABLE memory_facts_fts USING fts5(
  fact,
  category,
  entities,
  content='memory_facts',
  content_rowid='id'
);

CREATE TRIGGER memory_facts_ai AFTER INSERT ON memory_facts BEGIN
  INSERT INTO memory_facts_fts(rowid, fact, category, entities)
    VALUES (new.id, new.fact, new.category, new.entities);
END;

CREATE TRIGGER memory_facts_ad AFTER DELETE ON memory_facts BEGIN
  INSERT INTO memory_facts_fts(memory_facts_fts, rowid, fact, category, entities)
    VALUES('delete', old.id, old.fact, old.category, old.entities);
END;

CREATE TRIGGER memory_facts_au
AFTER UPDATE OF fact, category, entities ON memory_facts BEGIN
  INSERT INTO memory_facts_fts(memory_facts_fts, rowid, fact, category, entities)
    VALUES('delete', old.id, old.fact, old.category, old.entities);
  INSERT INTO memory_facts_fts(rowid, fact, category, entities)
    VALUES (new.id, new.fact, new.category, new.entities);
END;


-- =====================================================================
-- 4. Configuration for retrieval, streams, aliases, disambiguation
-- =====================================================================
CREATE TABLE fact_categories (
  name         TEXT PRIMARY KEY COLLATE NOCASE,
  description  TEXT NOT NULL,
  sort_ord     INTEGER NOT NULL,
  active       INTEGER NOT NULL DEFAULT 1,
  created_at   TEXT NOT NULL,
  stream       TEXT NOT NULL DEFAULT 'mixed'
);

CREATE INDEX idx_fact_categories_stream
  ON fact_categories(stream);

CREATE TABLE fact_category_aliases (
  legacy     TEXT PRIMARY KEY COLLATE NOCASE,
  canonical  TEXT NOT NULL COLLATE NOCASE,
  created_at TEXT NOT NULL
);

CREATE TABLE fact_category_streams (
  category TEXT NOT NULL,
  stream   TEXT NOT NULL,
  active   INTEGER NOT NULL DEFAULT 1,
  PRIMARY KEY (category, stream)
);

CREATE TABLE retrieval_streams (
  name        TEXT PRIMARY KEY,
  description TEXT NOT NULL,
  sort_ord    INTEGER NOT NULL,
  active      INTEGER NOT NULL DEFAULT 1,
  created_at  TEXT NOT NULL
);

CREATE TABLE retrieval_stream_cues (
  stream       TEXT NOT NULL,
  pattern      TEXT NOT NULL,
  active       INTEGER NOT NULL DEFAULT 1,
  PRIMARY KEY (stream, pattern)
);

CREATE TABLE retrieval_page_stubs (
  canonical_name TEXT PRIMARY KEY,
  notes          TEXT,
  active         INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE retrieval_pins (
  fact_id    INTEGER NOT NULL,
  stream     TEXT NOT NULL DEFAULT '*',
  weight     REAL NOT NULL DEFAULT 2.0,
  notes      TEXT,
  active     INTEGER NOT NULL DEFAULT 1,
  created_at TEXT NOT NULL,
  PRIMARY KEY (fact_id, stream)
);

CREATE TABLE retrieval_pin_star_allow (
  fact_id INTEGER PRIMARY KEY,
  notes   TEXT
);

CREATE TRIGGER trg_pin_star_ins
    BEFORE INSERT ON retrieval_pins
    WHEN lower(trim(NEW.stream)) = '*'
     AND NEW.fact_id NOT IN (SELECT fact_id FROM retrieval_pin_star_allow)
    BEGIN
      SELECT RAISE(ABORT,
        'star pin not on allow-list; use an allowed named stream');
    END;

CREATE TRIGGER trg_pin_star_upd
    BEFORE UPDATE OF stream, fact_id ON retrieval_pins
    WHEN lower(trim(NEW.stream)) = '*'
     AND NEW.fact_id NOT IN (SELECT fact_id FROM retrieval_pin_star_allow)
    BEGIN
      SELECT RAISE(ABORT, 'star pin not on allow-list');
    END;

CREATE TABLE retrieval_query_stems (
  id        INTEGER PRIMARY KEY AUTOINCREMENT,
  fact_id   INTEGER NOT NULL,
  stream    TEXT    NOT NULL DEFAULT '*',
  stem      TEXT    NOT NULL,
  notes     TEXT,
  active    INTEGER NOT NULL DEFAULT 1
);

CREATE UNIQUE INDEX idx_query_stems_uniq
  ON retrieval_query_stems(fact_id, stream, stem);


-- =====================================================================
-- 5. Embeddings and vector-search source gate
-- =====================================================================
-- Requires sqlite-vec: use sqlite_vec.load(connection) in Python,
-- or .load <path-to-vec0> in the sqlite3 CLI.
CREATE VIRTUAL TABLE memory_facts_vec USING vec0(embedding float[384]);

CREATE TABLE retrieval_chunk_vec_sources (
  source    TEXT PRIMARY KEY,
  active    INTEGER NOT NULL DEFAULT 1,
  notes     TEXT,
  added_at  TEXT
);

CREATE VIRTUAL TABLE chunks_vec USING vec0(embedding float[384]);


-- =====================================================================
-- 6. Optional source-summary navigation placeholder
-- =====================================================================
-- Kept with its existing columns as a placeholder/navigation layer.
-- An ordinary raw source load may seed a pending segment=0 row.
CREATE TABLE IF NOT EXISTS "source_summaries" (
  source            TEXT    NOT NULL,
  segment           INTEGER NOT NULL DEFAULT 0,
  source_ord        INTEGER NOT NULL,
  chunk_count       INTEGER,
  chunk_start       INTEGER,
  chunk_end         INTEGER,
  characters        TEXT,
  title             TEXT,
  short_summary     TEXT,
  full_summary      TEXT,
  key_themes        TEXT,
  key_events        TEXT,
  consistency_notes TEXT,
  status            TEXT    NOT NULL DEFAULT 'pending',
  enriched_at       TEXT,
  model             TEXT,
  generated_at      TEXT,
  PRIMARY KEY (source, segment)
);

CREATE INDEX idx_source_summaries_ord
  ON source_summaries (source_ord, source, segment);


#!/usr/bin/env python3
"""Public CORE-ONLY reference extract of the production text loader.

Source: loaded vector-aware load_file.py (15 October 2026). The core
chunking, date heuristics, FTS maintenance, upsert/cleanup, sqlite-vec
embedding and activation are copied from the supplied implementation.

REDACTION: protected seed-name lists and source-specific examples have
been removed. Consequently name tagging is DISABLED here (seed names
are deliberately unavailable); the production loader has that list.
This is a reference for code review, NOT a drop-in loader.

Core schema: chunks, chunk_chars, chunks_fts, chunks_vec, entities,
entity_sources, retrieval_chunk_vec_sources, source_summaries placeholder.
No private data, auth tokens or personal identifiers are included.
"""

from __future__ import annotations

import hashlib
import os
import json
import re
import sqlite3
import sys
import traceback
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DB_PATH = Path(os.environ.get("ARCHIVE_DB_PATH", "archive.sqlite"))

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200
UPSERT_BATCH = 500
VECTOR_BATCH = 64
VECTOR_MODEL = os.environ.get("EMBEDDING_MODEL") or "BAAI/bge-small-en-v1.5"

# REDACTED: production uses a fixed protected name-seed list.
# Publication code deliberately contains NO names or configured tag values.
CHARACTER_NAMES: tuple[str, ...] = ()
LOAD_EXACT_EXTRA: tuple[str, ...] = ()
CORE_SET: set[str] = set()

_GIVEN_SPLIT = re.compile(r"[\s(_]+")

# --- event_date heuristics ---
DATE_PATTERNS = [
    (re.compile(
        r"\b(\d{1,2})(?:st|nd|rd|th)?\s+"
        r"(January|February|March|April|May|June|July|August|September|October|November|December)"
        r"\s+(\d{4})\b",
        re.I,
    ), "dmy_named"),
    (re.compile(
        r"\b(January|February|March|April|May|June|July|August|September|October|November|December)"
        r"(?:\s+(\d{1,2})(?:st|nd|rd|th)?,?)?\s+(\d{4})\b",
        re.I,
    ), "mdy_named"),
    (re.compile(r"\b(20\d{2})-(\d{2})-(\d{2})\b"), "iso"),
    (re.compile(r"\b(\d{1,2})/(\d{1,2})/(20\d{2}|\d{2})\b"), "dmy_slash"),
]

MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
}


def source_ord_from_name(source: str) -> int:
    m = re.match(r"^Chat(\d+)$", source, re.I)
    if m:
        return int(m.group(1))
    m = re.match(r"^part(\d+)$", source, re.I)
    if m:
        return int(m.group(1))
    return 0


def chunk_text(text: str) -> list[str]:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if not text.strip():
        return []
    chunks: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + CHUNK_SIZE, n)
        if end < n:
            cut = text.rfind("\n", start, end)
            if cut < start + CHUNK_SIZE // 2:
                cut = text.rfind(" ", start, end)
            if cut > start:
                end = cut
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= n:
            break
        start = max(end - CHUNK_OVERLAP, start + 1)
    return chunks


def extract_event_date(text: str) -> str | None:
    for cre, kind in DATE_PATTERNS:
        m = cre.search(text)
        if not m:
            continue
        try:
            if kind == "dmy_named":
                d, mon, y = int(m.group(1)), MONTHS[m.group(2).lower()], int(m.group(3))
                return f"{y:04d}-{mon:02d}-{d:02d}"
            if kind == "mdy_named":
                mon = MONTHS[m.group(1).lower()]
                y = int(m.group(3))
                d = int(m.group(2)) if m.group(2) else 1
                return f"{y:04d}-{mon:02d}-{d:02d}"
            if kind == "iso":
                return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
            if kind == "dmy_slash":
                d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
                if y < 100:
                    y += 2000
                return f"{y:04d}-{mo:02d}-{d:02d}"
        except (ValueError, KeyError):
            continue
    return None


def yymm_from_event_date(ed: str | None) -> str | None:
    if not ed or len(ed) < 7:
        return None
    try:
        y, m = int(ed[0:4]), int(ed[5:7])
        return f"{y % 100:02d}{m:02d}"
    except ValueError:
        return None


def given_name(canonical: str) -> str:
    s = (canonical or "").strip()
    if not s:
        return ""
    return _GIVEN_SPLIT.split(s, maxsplit=1)[0]


def chars_in_text(text: str) -> list[str]:
    """Exact canonicals only. Bare '[PERSON]' is not tagged (collision)."""
    found = []
    for name in CHARACTER_NAMES + LOAD_EXACT_EXTRA:
        if re.search(rf"\b{re.escape(name)}\b", text):
            found.append(name)
    return found


def load_character_index(db: sqlite3.Connection) -> list[dict[str, Any]]:
    if not table_exists(db, "entities"):
        return []
    try:
        rows = db.execute(
            """
            SELECT canonical_name, COALESCE(description, '') AS description
            FROM entities
            WHERE entity_type = 'character'
            """
        ).fetchall()
    except sqlite3.Error:
        return []
    out: list[dict[str, Any]] = []
    for r in rows:
        name = r[0]
        desc = (r[1] or "").lower()
        warning = (
            "ambiguous name-row" in desc
            or "do not treat this card as a person" in desc
            or "collision warning" in desc
        )
        out.append(
            {
                "canonical_name": name,
                "given": given_name(name),
                "warning": warning,
            }
        )
    return out


def exact_canonical_in_text(text: str, index: list[dict[str, Any]]) -> list[str]:
    """Qualified names that appear verbatim (e.g. '[PERSON] ([PERSON])')."""
    found: list[str] = []
    seen: set[str] = set()
    for rec in index:
        if rec.get("warning"):
            continue
        canon = rec["canonical_name"]
        if canon.lower() in CORE_SET:
            continue
        if " " not in canon and "(" not in canon:
            continue
        if re.search(rf"\b{re.escape(canon)}\b", text, re.I):
            key = canon.lower()
            if key not in seen:
                seen.add(key)
                found.append(canon)
    return found


def find_ambiguous_mentions(
    text: str,
    index: list[dict[str, Any]],
) -> list[tuple[str, list[str]]]:
    """
    Given-names that appear as a bare word and map to 2+ real cards,
    or a bare word that is not itself an exact canonical.

    Returns [(surface, [candidate canonicals]), ...]
    Seed-protected canonicals are excluded when a private seed list is configured.
    """
    people = [r for r in index if not r.get("warning")]
    by_given: dict[str, list[str]] = {}
    exact_lower = {r["canonical_name"].lower() for r in people}
    for r in people:
        g = (r.get("given") or "").strip()
        if not g:
            continue
        by_given.setdefault(g.lower(), []).append(r["canonical_name"])

    flagged: list[tuple[str, list[str]]] = []
    seen: set[str] = set()
    for g_low, members in sorted(by_given.items()):
        if g_low in CORE_SET:
            continue
        if not re.search(rf"\b{re.escape(g_low)}\b", text, re.I):
            continue
        if g_low in seen:
            continue
        seen.add(g_low)
        # Bare word is an exact canonical only if a card is exactly that word
        # AND there is no cluster (unique).
        unique_exact = g_low in exact_lower and len(members) == 1
        if unique_exact:
            continue
        flagged.append((members[0].split()[0] if members else g_low, members))
    return flagged


def ambiguous_note(flags: list[tuple[str, list[str]]]) -> str | None:
    if not flags:
        return None
    parts = []
    names = []
    for surface, members in flags:
        names.append(surface)
        who = ", ".join(members) if members else "(no card)"
        parts.append(f"{surface} (could be {who})")
    listed = "; ".join(parts)
    label = "name" if len(names) == 1 else "names"
    return (
        f"AMBIGUOUS {label} in source text — not tagged at load: {listed}. "
        "Entity disambiguation is deliberately deferred."
    )


def make_id(source: str, index: int, body: str) -> str:
    h = hashlib.sha1(body.encode("utf-8", errors="replace")).hexdigest()[:12]
    return f"{source}_{index}_{h}"


def connect() -> sqlite3.Connection:
    if not DB_PATH.exists():
        raise SystemExit(f"DB not found: {DB_PATH} — create schema first")
    db = sqlite3.connect(str(DB_PATH))
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA synchronous=NORMAL")
    db.execute("PRAGMA foreign_keys=ON")
    return db


def table_exists(db: sqlite3.Connection, name: str) -> bool:
    row = db.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name = ?",
        (name,),
    ).fetchone()
    return row is not None


def load_vector_extension(db: sqlite3.Connection) -> None:
    import sqlite_vec
    db.enable_load_extension(True)
    try:
        sqlite_vec.load(db)
    finally:
        db.enable_load_extension(False)
    schema = db.execute("SELECT sql FROM sqlite_master WHERE name='chunks_vec'").fetchone()
    if not schema or "float[384]" not in schema[0].lower().replace(" ", ""):
        raise RuntimeError("Expected chunks_vec embedding float[384]; check vector schema")
    if not table_exists(db, "retrieval_chunk_vec_sources"):
        raise RuntimeError("Missing retrieval_chunk_vec_sources")


def upsert_batch(
    db: sqlite3.Connection,
    rows: list[tuple],
    char_rows: list[tuple[str, str]],
    vectors_ready: bool = False,
) -> None:
    """Core ingestion. Preserve rowids on updates; invalidate changed embeddings."""
    if not rows:
        return
    ids = [r[0] for r in rows]
    qmarks = ",".join("?" * len(ids))
    previous = {
        cid: (rid, body)
        for rid, cid, body in db.execute(
            f"SELECT rowid,id,body FROM chunks WHERE id IN ({qmarks})", ids
        )
    }
    try:
        # FTS is explicitly maintained by this loader, not by a DB trigger.
        for cid, old in previous.items():
            rid, old_body = old
            db.execute(
                "INSERT INTO chunks_fts(chunks_fts,rowid,body) VALUES('delete',?,?)",
                (rid, old_body),
            )
        db.execute(f"DELETE FROM chunk_chars WHERE chunk_id IN ({qmarks})", ids)
        # UPDATE, rather than REPLACE, retains rowid for unchanged chunk IDs.
        db.executemany(
            """
            INSERT INTO chunks
              (id,source,source_ord,batch,path,loaded_at,event_date,yymm,body)
            VALUES (?,?,?,?,?,?,?,?,?)
            ON CONFLICT(id) DO UPDATE SET
              source=excluded.source, source_ord=excluded.source_ord,
              batch=excluded.batch, path=excluded.path,
              loaded_at=excluded.loaded_at, event_date=excluded.event_date,
              yymm=excluded.yymm, body=excluded.body
            """, rows,
        )
        for cid, *_rest, body in rows:
            rid = db.execute("SELECT rowid FROM chunks WHERE id=?", (cid,)).fetchone()[0]
            if vectors_ready and cid in previous and previous[cid][1] != body:
                db.execute("DELETE FROM chunks_vec WHERE rowid=?", (rid,))
            db.execute("INSERT INTO chunks_fts(rowid,body) VALUES (?,?)", (rid,body))
        if char_rows:
            db.executemany(
                "INSERT OR IGNORE INTO chunk_chars(chunk_id,name) VALUES (?,?)",char_rows
            )
        db.commit()
    except Exception:
        db.rollback()
        raise


def prune_obsolete_chunks(db: sqlite3.Connection, source: str, current_ids: set[str],
                          vectors_ready: bool) -> int:
    """Remove chunks no longer present on reload, including their search indexes."""
    old = db.execute(
        "SELECT rowid,id,body FROM chunks WHERE source=?", (source,)
    ).fetchall()
    stale = [(rid,cid,body) for rid,cid,body in old if cid not in current_ids]
    if not stale:
        return 0
    try:
        for rid,cid,body in stale:
            if vectors_ready:
                db.execute("DELETE FROM chunks_vec WHERE rowid=?", (rid,))
            db.execute(
                "INSERT INTO chunks_fts(chunks_fts,rowid,body) VALUES('delete',?,?)",
                (rid,body),
            )
            db.execute("DELETE FROM chunk_chars WHERE chunk_id=?",(cid,))
            db.execute("DELETE FROM chunks WHERE rowid=?",(rid,))
        db.commit()
    except Exception:
        db.rollback()
        raise
    return len(stale)


def populate_source_vectors(db: sqlite3.Connection, source: str) -> None:
    """Best-effort, resumable embedding stage; same model/preprocessing as backfill."""
    import sqlite_vec
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(VECTOR_MODEL, local_files_only=True)
    if model.get_sentence_embedding_dimension() != 384:
        raise RuntimeError(f"Unexpected embedding dimension for {VECTOR_MODEL}")
    count = db.execute(
        "SELECT COUNT(*) FROM chunks c LEFT JOIN chunks_vec v ON c.rowid=v.rowid "
        "WHERE c.source=? AND v.rowid IS NULL", (source,)
    ).fetchone()[0]
    print(f"Vector indexing: {count} missing for {source}; model={VECTOR_MODEL}",flush=True)
    last = 0
    done = 0
    t0 = time.monotonic()
    while True:
        batch = db.execute(
            "SELECT c.rowid,c.body FROM chunks c "
            "LEFT JOIN chunks_vec v ON c.rowid=v.rowid "
            "WHERE c.source=? AND v.rowid IS NULL AND c.rowid>? "
            "ORDER BY c.rowid LIMIT ?", (source,last,VECTOR_BATCH)
        ).fetchall()
        if not batch:
            break
        last = batch[-1][0]
        texts = [(body or "").strip()[:2000] or " " for _,body in batch]
        embeddings = model.encode(texts,normalize_embeddings=True,show_progress_bar=False)
        try:
            with db:
                for (rid,_),vec in zip(batch,embeddings):
                    db.execute("INSERT INTO chunks_vec(rowid,embedding) VALUES (?,?)",
                               (rid,sqlite_vec.serialize_float32([float(v) for v in vec])))
        except Exception:
            db.rollback()
            raise
        done += len(batch)
        if done % 640 < len(batch):
            print(f"  Embedded {done}/{count}; {time.monotonic()-t0:.0f}s",flush=True)
    missing = db.execute(
        "SELECT COUNT(*) FROM chunks c LEFT JOIN chunks_vec v ON c.rowid=v.rowid "
        "WHERE c.source=? AND v.rowid IS NULL",(source,)
    ).fetchone()[0]
    if missing:
        raise RuntimeError(f"{missing} vectors still missing for {source}")
    with db:
        db.execute(
            "INSERT INTO retrieval_chunk_vec_sources(source,active) VALUES (?,1) "
            "ON CONFLICT(source) DO UPDATE SET active=1",(source,)
        )
    print(f"Vector indexing complete: {done} new embeddings; source active.",flush=True)


def refresh_entities(db: sqlite3.Connection) -> None:
    """
    Re-aggregate volume fields from live chunk_chars onto EXISTING entity cards.

    Never INSERT a new entities row from a chunk_chars name — that is how a
    short '[PERSON]' card would get minted at load time. Only update mention_count /
    first_ord / last_ord when cc.name already matches a canonical_name.
    importance and description are never touched.
    """
    if not table_exists(db, "entities"):
        raise RuntimeError("table 'entities' missing")
    if not table_exists(db, "entity_sources"):
        raise RuntimeError("table 'entity_sources' missing")

    db.execute(
        """
        UPDATE entities
        SET
          mention_count = (
            SELECT COUNT(*) FROM chunk_chars cc WHERE cc.name = entities.canonical_name
          ),
          first_ord = (
            SELECT MIN(c.source_ord)
            FROM chunk_chars cc
            JOIN chunks c ON c.id = cc.chunk_id
            WHERE cc.name = entities.canonical_name
          ),
          last_ord = (
            SELECT MAX(c.source_ord)
            FROM chunk_chars cc
            JOIN chunks c ON c.id = cc.chunk_id
            WHERE cc.name = entities.canonical_name
          ),
          updated_at = datetime('now')
        WHERE entity_type = 'character'
          AND EXISTS (
            SELECT 1 FROM chunk_chars cc WHERE cc.name = entities.canonical_name
          )
        """
    )

    db.execute(
        """
        INSERT INTO entity_sources (entity_id, source, prominence)
        SELECT
          e.id,
          c.source,
          CASE
            WHEN COUNT(*) >= 20 THEN 'central'
            WHEN COUNT(*) >= 5  THEN 'major'
            ELSE 'mentioned'
          END
        FROM chunk_chars cc
        JOIN chunks c ON c.id = cc.chunk_id
        JOIN entities e ON e.canonical_name = cc.name COLLATE NOCASE
        GROUP BY e.id, c.source
        ON CONFLICT(entity_id, source) DO UPDATE SET
          prominence = excluded.prominence
        """
    )
    db.commit()


def upsert_source_summary_basic(
    db: sqlite3.Connection,
    source: str,
    source_ord: int,
    chunk_count: int,
    character_names: list[str],
    consistency_notes: str | None = None,
) -> None:
    """Insert/update the OPTIONAL segment-0 navigation placeholder.

    Adapted from the original loader: the non-core progress queue was
    intentionally removed. An already-completed summary is preserved.
    This step performs no summarisation.
    """
    if not table_exists(db, "source_summaries"):
        return
    chars_json = json.dumps(sorted(set(character_names)), ensure_ascii=False)
    db.execute(
        """
        INSERT INTO source_summaries (
            source, segment, source_ord, chunk_count, chunk_start, chunk_end,
            characters, consistency_notes, status, generated_at
        ) VALUES (?, 0, ?, ?, 0,
                  CASE WHEN ? < 1 THEN NULL ELSE ? - 1 END,
                  ?, ?, 'pending', datetime('now'))
        ON CONFLICT(source, segment) DO UPDATE SET
            source_ord = excluded.source_ord,
            chunk_count = excluded.chunk_count,
            characters = CASE WHEN source_summaries.status='done'
                              THEN source_summaries.characters
                              ELSE excluded.characters END,
            consistency_notes = CASE WHEN source_summaries.status='done'
                                     THEN source_summaries.consistency_notes
                                     ELSE excluded.consistency_notes END,
            status = CASE WHEN source_summaries.status='done'
                          THEN 'done' ELSE 'pending' END
        """,
        (source, source_ord, chunk_count, chunk_count, chunk_count,
         chars_json, consistency_notes),
    )
    db.commit()



def try_core_metadata_sync(
    db: sqlite3.Connection,
    source: str,
    source_ord: int,
    chunk_count: int,
    character_names: list[str],
    consistency_notes: str | None = None,
) -> None:
    """Core metadata refresh and optional source-summary placeholder.

    The production loader also updates a separate progress queue, omitted
    here because that table is deliberately absent from the core schema.
    """
    try:
        refresh_entities(db)
    except sqlite3.Error as exc:
        print(f"Core entity metadata refresh skipped: {type(exc).__name__}")
    try:
        upsert_source_summary_basic(db, source, source_ord, chunk_count,
                                    character_names, consistency_notes)
    except sqlite3.Error as exc:
        print(f"Source-summary placeholder skipped: {type(exc).__name__}")



def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    path = Path(sys.argv[1]).expanduser().resolve()
    if not path.is_file():
        raise SystemExit(f"Not a file: {path}")

    source = sys.argv[2] if len(sys.argv) > 2 else path.stem
    source = re.sub(r"\s+", "", source)
    sord = source_ord_from_name(source)
    loaded_at = datetime.now(timezone.utc).isoformat()
    batch = source

    print(f"Reading {path} ...")
    text = path.read_text(encoding="utf-8", errors="replace")
    chunks = chunk_text(text)
    print(f"Created {len(chunks)} chunks")
    print(f"loaded_at : {loaded_at}")
    print(f"batch     : {batch}")
    print(f"source_ord: {sord}")

    rows: list[tuple] = []
    char_rows: list[tuple[str, str]] = []
    all_names_this_file: set[str] = set()
    dated = 0

    for i, body in enumerate(chunks):
        cid = make_id(source, i, body)
        ed = extract_event_date(body)
        if ed:
            dated += 1
        ym = yymm_from_event_date(ed)
        rows.append(
            (cid, source, sord, batch, str(path), loaded_at, ed, ym, body)
        )
        names = chars_in_text(body)
        for name in names:
            char_rows.append((cid, name))
            all_names_this_file.add(name)

    print(f"chunks with event_date: {dated} / {len(chunks)}")
    print(f"characters tagged    : {sorted(all_names_this_file)}")

    # --- CORE PATH (must succeed) ---
    db = connect()
    total = 0
    vectors_ready = False
    try:
        load_vector_extension(db)
        vectors_ready = True
    except Exception as exc:
        db.close()
        raise SystemExit(
            f"sqlite-vec unavailable: {exc}. Aborting before chunk changes "
            "to avoid stale embeddings; install sqlite-vec and retry."
        )
    # Mark this source inactive until ingestion and embedding verification finish.
    if vectors_ready:
        with db:
            db.execute(
                "INSERT INTO retrieval_chunk_vec_sources(source,active) VALUES (?,0) "
                "ON CONFLICT(source) DO UPDATE SET active=0",(source,)
            )
    try:
        for start in range(0, len(rows), UPSERT_BATCH):
            sl = slice(start, start + UPSERT_BATCH)
            batch_rows = rows[sl]
            id_set = {r[0] for r in batch_rows}
            batch_chars = [cr for cr in char_rows if cr[0] in id_set]
            seen = set()
            unique_rows = []
            for r in batch_rows:
                if r[0] in seen:
                    continue
                seen.add(r[0])
                unique_rows.append(r)
            upsert_batch(db, unique_rows, batch_chars, vectors_ready=vectors_ready)
            total += len(unique_rows)
            print(
                f"  Batch {start // UPSERT_BATCH + 1}: "
                f"upserted {len(unique_rows)} (running total {total})"
            )
    except Exception:
        db.close()
        print("ERROR: core chunk load failed — nothing enrichment-related was attempted.")
        raise

    removed = prune_obsolete_chunks(db, source, {r[0] for r in rows}, vectors_ready)
    if removed:
        print(f"Removed {removed} obsolete chunks for reloaded source {source}")
    n = db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    print(f"Core load done. Total chunks in store: {n}")

    index = load_character_index(db)
    qualified = exact_canonical_in_text(text, index)
    if qualified:
        all_names_this_file.update(qualified)
        print(f"exact canonicals in text: {qualified}")

    flags = find_ambiguous_mentions(text, index)
    note = ambiguous_note(flags)
    if note:
        print(f"AMBIGUOUS (not tagged): {note}")

    # --- CORE METADATA PATH (best-effort) ---
    try_core_metadata_sync(
        db, source, sord, total, sorted(all_names_this_file),
        consistency_notes=note,
    )

    try:
        if table_exists(db, "entities"):
            n_ent = db.execute(
                "SELECT COUNT(*) FROM entities WHERE entity_type = 'character'"
            ).fetchone()[0]
            print(f"Character entities: {n_ent}")
    except Exception:
        pass

    # --- VECTOR PATH (best-effort; raw chunks remain committed on failure) ---
    if vectors_ready:
        try:
            populate_source_vectors(db, source)
        except Exception as exc:
            print(f"WARNING: vector indexing incomplete for {source}: {exc}")
            print("Source remains inactive for vector retrieval; rerun loader or backfill.")
            traceback.print_exc(file=sys.stderr)
    db.close()
    print("Done.")


if __name__ == "__main__":
    main()

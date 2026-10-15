#!/usr/bin/env python3
"""CORE-ONLY, SANITISED TECHNICAL REFERENCE — NOT A DEPLOYABLE SERVER.

Derived from local_chat.py plus the Wilson v3.4.3 raw-chunk-vector patch,
15 October 2026. This contains no project names, private conversations,
personal examples, identity seed lists, activity alias catalogue, or secrets.

The retrieval algorithms are excerpted from the actual implementation.
A few interfaces are abbreviated where production code also depends on
excluded non-core data stores. Those abbreviations are marked explicitly.
No numerical thresholds have been tuned or rewritten for publication.

Core schema: chunks, chunk_chars, chunks_fts, chunks_vec, entities,
entity_sources, memory_facts, memory_facts_fts, memory_facts_vec,
retrieval_episodes, retrieval_streams, retrieval_pins, category tables,
source_summaries (read-only placeholder), retrieval_chunk_vec_sources.
"""
from __future__ import annotations
import os, re, json, sqlite3, hmac, time, logging, math
from pathlib import Path
from typing import Any
from datetime import datetime, timezone
from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

DB_PATH = Path(os.environ.get("ARCHIVE_DB_PATH", "archive.sqlite"))
RAG_MCP_TOKEN = os.environ.get("RAG_MCP_TOKEN", "")
mcp = FastMCP("Archive Core Reference")

# Retained REAL technical settings; deployment-specific model env keys
# are renamed for publication (private environment keys for the vector and CE stages).
_VEC_MODEL_NAME = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
_CE_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L6-v2"
_VEC_DIM = 384
_VEC_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
_VEC_KNN = 80
_CHUNK_VEC_KNN = 24
_V35_CHUNK_VEC_K = 80
_V35_SEMANTIC_COS = 0.55
_V35_SEMANTIC_CE = 0.62
_RRF_K = 60
_V2_MAX_BYTES = 17000
_V2_SUMMARY_POOL = 700
_V2_CE_SUMMARY_POOL = 18
_V3_BUDGET_MS = 2000
_V3_FTS_STAGE_CAP_MS = 750
_V3_EVIDENCE_RESERVE_MS = 450
_VEC_MODEL = _CE_MODEL = None
_VEC_ERROR = _CE_ERROR = None
_VEC_READY = False

# REDACTED: original server contains an explicit private activity alias
# catalogue and a protected global pinned fact ID. Not reproduced.
# The actual general-purpose algorithms remain below.
_V2_ACTIVITY_ALIASES = {}  # REDACTED deployment configuration
_V2_STOP = set()            # REDACTED original lexical stop-word configuration
_V2_ELLIPSIS = re.compile(
    r"\b(this time|last time|those|they|them|that one|still|again|tomorrow|"
    r"yesterday|the other|not many|as before|same as)\b", re.I)
_V34_REF_RE = re.compile(r"\b((?:[1-3]\s+)?[a-z][a-z]+)\s+(\d{1,3})\s*:\s*(\d{1,3})(?:[-–]\d{1,3})?\b", re.I)
_V34_CAP_RE = re.compile(r"\b(?:[A-Z][a-z]+\s+){1,2}[A-Z][a-z]+\b")

_GIVEN_SPLIT = re.compile(r"[\s(_]+")

def log_tool(fn):
    """Publication-only no-op decorator; production logs redacted args."""
    return fn

def require_auth(auth_token: str = "") -> str | None:
    # Require a configured secret and a matching token for every tool.
    if not RAG_MCP_TOKEN or not hmac.compare_digest(auth_token or "", RAG_MCP_TOKEN):
        return _err("authentication failed")
    return None

def _dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str, separators=(",", ":"))

def _err(message: str, **detail) -> str:
    return _dumps({"ok": False, "error": message, **detail})

def _ok(detail: dict) -> str:
    return _dumps({"ok": True, **detail})

def _clamp_limit(value, lo=1, hi=100):
    return max(lo, min(hi, int(value)))

def _row_to_dict(row):
    return dict(row) if row is not None else None

def _clamp(value, lo, hi):
    return min(hi, max(lo, value))

def _parse_name_list(raw):
    if isinstance(raw, list): return [str(x).strip() for x in raw if str(x).strip()]
    if not raw: return []
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list): return [str(x).strip() for x in parsed if str(x).strip()]
    except (ValueError, TypeError): pass
    return [p.strip() for p in str(raw).split(",") if p.strip()]

def connect() -> sqlite3.Connection:
    db = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    db.row_factory = sqlite3.Row
    _try_load_sqlite_vec(db)
    return db

def _try_load_sqlite_vec(db):
    global _VEC_READY
    try:
        import sqlite_vec
        db.enable_load_extension(True)
        sqlite_vec.load(db)
        db.enable_load_extension(False)
        _VEC_READY = True
        return True
    except Exception:
        try: db.enable_load_extension(False)
        except Exception: pass
        _VEC_READY = False
        return False

def _vec_table_ready(db):
    if not _VEC_READY: return False
    try: db.execute("SELECT 1 FROM memory_facts_vec LIMIT 0"); return True
    except sqlite3.Error: return False

# _load_retrieval_streams (source-derived)
def _load_retrieval_streams(db: sqlite3.Connection | None = None) -> list[str]:
    """Live stream names from retrieval_streams (or retrieve_streams)."""
    closer = False
    if db is None:
        try:
            db = connect()
            closer = True
        except Exception:
            return []
    try:
        rows = []
        for tbl in ("retrieval_streams", "retrieve_streams"):
            try:
                rows = db.execute(
                    f"SELECT name FROM {tbl} "
                    "WHERE COALESCE(active,1)=1 ORDER BY sort_ord, name"
                ).fetchall()
                if rows:
                    break
            except sqlite3.Error:
                rows = []
        names = []
        for r in rows:
            n = str(r[0] if not isinstance(r, sqlite3.Row) else r["name"]).strip().lower()
            if n:
                names.append(n)
        return names
    finally:
        if closer and db is not None:
            db.close()


# _load_stream_category_map (source-derived)
def _load_stream_category_map(
    db: sqlite3.Connection | None = None,
) -> dict[str, set[str]]:
    """stream → {category,...} from fact_categories.stream. Empty if unmigrated."""
    mapping: dict[str, set[str]] = {}
    closer = False
    if db is None:
        try:
            db = connect()
            closer = True
        except Exception:
            db = None
    if db is None:
        return mapping
    try:
        rows = []
        try:
            rows = db.execute(
                "SELECT category, stream FROM fact_category_streams"
            ).fetchall()
        except sqlite3.Error:
            rows = []
        if not rows:
            try:
                rows = db.execute(
                    "SELECT name, stream FROM fact_categories "
                    "WHERE COALESCE(active,1)=1"
                ).fetchall()
            except sqlite3.Error:
                rows = []
        for r in rows:
            if isinstance(r, sqlite3.Row):
                keys = r.keys()
                cat = r["category"] if "category" in keys else r["name"]
                st = r["stream"]
            else:
                cat, st = r[0], (r[1] if len(r) > 1 else "mixed")
            cat = str(cat or "").strip().lower()
            st = str(st or "mixed").strip().lower()
            if cat:
                mapping.setdefault(st, set()).add(cat)
        return mapping
    finally:
        if closer:
            db.close()


# _load_retrieval_pins (source-derived)
def _load_retrieval_pins(
    db: sqlite3.Connection | None = None,
) -> list[dict[str, Any]]:
    closer = False
    if db is None:
        try:
            db = connect()
            closer = True
        except Exception:
            db = None
    pins: list[dict[str, Any]] = []
    if db is None:
        return pins
    try:
        try:
            rows = db.execute(
                "SELECT fact_id, stream, weight, notes FROM retrieval_pins "
                "WHERE COALESCE(active,1)=1"
            ).fetchall()
        except sqlite3.Error:
            rows = []
        for r in rows:
            pins.append({
                "fact_id": int(r["fact_id"] if isinstance(r, sqlite3.Row) else r[0]),
                "stream": str(r["stream"] if isinstance(r, sqlite3.Row) else r[1] or "*").lower(),
                "weight": float(r["weight"] if isinstance(r, sqlite3.Row) else r[2] or 2.0),
                "notes": str(r["notes"] if isinstance(r, sqlite3.Row) else r[3] or ""),
            })
        return pins
    finally:
        if closer:
            db.close()


# _load_stream_cues (source-derived)
def _load_stream_cues(
    db: sqlite3.Connection | None = None,
) -> list[tuple[str, re.Pattern[str]]]:
    """(stream, compiled pattern) from retrieval_stream_cues."""
    closer = False
    if db is None:
        try:
            db = connect()
            closer = True
        except Exception:
            db = None
    out: list[tuple[str, re.Pattern[str]]] = []
    if db is None:
        return out
    try:
        try:
            rows = db.execute(
                "SELECT stream, pattern FROM retrieval_stream_cues "
                "WHERE COALESCE(active,1)=1"
            ).fetchall()
        except sqlite3.Error:
            rows = []
        for r in rows:
            if isinstance(r, sqlite3.Row):
                st, pat = r["stream"], r["pattern"]
            else:
                st, pat = r[0], r[1]
            st = str(st or "").strip().lower()
            pat = str(pat or "").strip()
            if not st or not pat:
                continue
            try:
                wrapped = pat if pat.startswith(r"\b") or pat.startswith("(?") else (
                    r"\b(?:" + pat + r")\b"
                )
                out.append((st, re.compile(wrapped, re.I)))
            except re.error:
                continue
        return out
    finally:
        if closer:
            db.close()


# _load_page_stubs (source-derived)
def _load_page_stubs(db: sqlite3.Connection | None = None) -> set[str]:
    """Canonical names that are page-only stubs."""
    closer = False
    if db is None:
        try:
            db = connect()
            closer = True
        except Exception:
            db = None
    names: set[str] = set()
    if db is None:
        return names
    try:
        try:
            rows = db.execute(
                "SELECT canonical_name FROM retrieval_page_stubs "
                "WHERE COALESCE(active,1)=1"
            ).fetchall()
        except sqlite3.Error:
            rows = []
        for r in rows:
            n = str(r[0] if not isinstance(r, sqlite3.Row) else r["canonical_name"]).strip()
            if n:
                names.add(n)
        return names
    finally:
        if closer:
            db.close()


# _exact_canonical_in_turn (source-derived)
def _exact_canonical_in_turn(name: str, turn: str) -> bool:
    if not name or not turn:
        return False
    return bool(re.search(
        r"(?<!\w)" + re.escape(name) + r"(?!\w)",
        turn,
        re.I,
    ))


# _filter_live_names (source-derived)
def _filter_live_names(names: list[str], turn: str, index: list[dict[str, Any]] | None = None) -> list[str]:
    """Drop page stubs and collision-warning cards unless the exact canonical is in the turn."""
    stubs = {n.lower() for n in _load_page_stubs()}
    by_lower = {}
    if index is None:
        index = load_character_index()
    for rec in index or []:
        by_lower[(rec.get("canonical_name") or "").lower()] = rec
    kept: list[str] = []
    for name in names:
        rec = by_lower.get(name.lower()) or {}
        desc = rec.get("description") or ""
        stub = name.lower() in stubs
        warn = is_collision_warning(desc)
        if stub or warn:
            if _exact_canonical_in_turn(name, turn):
                kept.append(name)
            continue
        kept.append(name)
    return kept


# infer_stream (source-derived)
def infer_stream(turn: str, explicit: str = "") -> tuple[str, bool]:
    """Return (stream, inferred?). Unknown explicit → mixed."""
    allowed = set(_load_retrieval_streams())
    # Also accept any stream already assigned on fact_categories.
    for st in _load_stream_category_map():
        allowed.add(st)
    want = (explicit or "").strip().lower()
    if want in ("", "auto", "infer"):
        want = ""
    if want and (not allowed or want in allowed or want == "mixed"):
        return want, False
    hits: list[str] = []
    seen: set[str] = set()
    for name, rx in _load_stream_cues():
        if name in seen:
            continue
        if rx.search(turn or ""):
            hits.append(name)
            seen.add(name)
    if len(hits) == 1:
        return hits[0], True
    if hits:
        return "mixed", True
    return "mixed", True


# given_name (source-derived)
def given_name(canonical: str) -> str:
    """First token only. '[PERSON] ([PERSON])' → '[PERSON]'; '[PERSON] Q' → '[PERSON]'.

    Qualifiers after space, underscore or '(' never become the given name,
    so a query for [PERSON] does not pull [PERSON] ([PERSON]).
    """
    s = (canonical or "").strip()
    if not s:
        return ""
    return _GIVEN_SPLIT.split(s, maxsplit=1)[0]


# is_collision_warning (source-derived)
def is_collision_warning(description: str) -> bool:
    d = (description or "").lower()
    return (
        "ambiguous name-row" in d
        or "do not treat this card as a person" in d
        or "collision warning" in d
        or "this row is only a collision warning" in d
    )


# _alias_list (source-derived)
def _alias_list(aliases: Any) -> list[str]:
    if not aliases:
        return []
    if isinstance(aliases, list):
        return [str(a).strip() for a in aliases if str(a).strip()]
    s = str(aliases).strip()
    if not s:
        return []
    try:
        parsed = json.loads(s)
        if isinstance(parsed, list):
            return [str(a).strip() for a in parsed if str(a).strip()]
    except Exception:
        pass
    return [p.strip() for p in re.split(r"[,;]", s) if p.strip()]


# load_character_index (source-derived)
def load_character_index() -> list[dict[str, Any]]:
    """Every character row: canonical, given name, aliases, warning flag."""
    db = connect()
    try:
        cols = {r[1] for r in db.execute("PRAGMA table_info(entities)").fetchall()}
        extra = ", aliases" if "aliases" in cols else ""
        rows = db.execute(
            f"""
            SELECT canonical_name, description{extra}
            FROM entities
            WHERE entity_type = 'character'
            """
        ).fetchall()
        out: list[dict[str, Any]] = []
        for r in rows:
            name = r["canonical_name"]
            aliases = _alias_list(r["aliases"]) if extra else []
            desc = r["description"] or ""
            out.append(
                {
                    "canonical_name": name,
                    "description": desc,
                    "aliases": aliases,
                    "given": given_name(name),
                    "warning": is_collision_warning(desc),
                }
            )
        return out
    except sqlite3.Error:
        return []
    finally:
        db.close()


# name_clusters (source-derived)
def name_clusters(
    index: list[dict[str, Any]] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """given-name (lower) → members who share that given name. Size ≥ 2 only."""
    index = index if index is not None else load_character_index()
    clusters: dict[str, list[dict[str, Any]]] = {}
    for rec in index:
        g = (rec.get("given") or "").lower()
        if not g:
            continue
        clusters.setdefault(g, []).append(rec)
    return {k: v for k, v in clusters.items() if len(v) > 1}


# resolve_to_canonical (source-derived)
def resolve_to_canonical(
    name: str,
    index: list[dict[str, Any]] | None = None,
) -> tuple[str | None, str]:
    """Map a surface person-name onto entities.canonical_name.

    Order: exact canonical, exact alias, unique given-name among real
    (non-warning) cards. Ambiguous given-names ([PERSON] / [PERSON] ([PERSON])) return
    None so we never invent a short card and never pick the wrong twin.
    """
    raw = (name or "").strip()
    if not raw:
        return None, "empty"
    index = index if index is not None else load_character_index()
    people = [r for r in index if not r.get("warning")]
    low = raw.lower()

    for r in people:
        if r["canonical_name"].lower() == low:
            return r["canonical_name"], "exact"
    for r in people:
        for alias in r.get("aliases") or []:
            if alias.lower() == low:
                return r["canonical_name"], "alias"

    given = given_name(raw).lower()
    hits = [r for r in people if (r.get("given") or "").lower() == given]
    if len(hits) == 1:
        return hits[0]["canonical_name"], "unique-given"
    if len(hits) > 1:
        return None, "ambiguous:" + ",".join(h["canonical_name"] for h in hits)
    return None, "unknown"


# _name_variants (source-derived)
def _name_variants(canon: str) -> set[str]:
    """Matchable surface forms of a canonical name. First token is included.

    Qualifier tokens after the first ('[PERSON]' in '[PERSON] ([PERSON])') are NOT added,
    so a query for [PERSON] does not resolve to [PERSON] ([PERSON]).
    """
    variants = {
        canon,
        canon.replace("(", "").replace(")", "").strip(),
        canon.replace(" ", "_"),
        canon.replace("_", " "),
        re.sub(r"[^\w\s]", " ", canon).strip(),
        given_name(canon),
    }
    return {v for v in variants if v}


# candidate_names (source-derived)
def candidate_names(query: str) -> list[str]:
    """
    Resolve names mentioned in the query.

    - Exact / near-exact canonical matches (spaces, parentheses, underscores).
    - First-token given-name match ALWAYS expands to the full collision
      cluster (so '[PERSON]' yields both [PERSON] and [PERSON] ([PERSON])).
    - A query for the qualifier ('[PERSON]') does not pull '[PERSON] ([PERSON])'.
    """
    index = load_character_index()
    if not index:
        return []

    q = query or ""
    found: set[str] = set()
    by_lower = {rec["canonical_name"].lower(): rec for rec in index}

    # 1. Exact / near-exact canonical matches
    for rec in index:
        canon = rec["canonical_name"]
        for v in _name_variants(canon):
            pattern = r"(?<!\w)" + re.escape(v) + r"(?!\w)"
            if re.search(pattern, q, re.I):
                found.add(canon)
                break
        for alias in rec.get("aliases") or []:
            pattern = r"(?<!\w)" + re.escape(alias) + r"(?!\w)"
            if alias and re.search(pattern, q, re.I):
                found.add(canon)
                break

    # 2. Given-name clustering — always, not only when step 1 is empty.
    #    A single-word token that equals a given-name of a 2+ cluster
    #    pulls every member of that cluster.
    clusters = name_clusters(index)
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9\-']*", q)
    for tok in tokens:
        members = clusters.get(tok.lower())
        if members:
            for rec in members:
                found.add(rec["canonical_name"])

    # 3. Fallback: case-insensitive lookup of unmatched single-word names
    if not found:
        for tok in tokens:
            rec = by_lower.get(tok.lower())
            if rec:
                found.add(rec["canonical_name"])

    return sorted(found)


# _collision_banner (source-derived)
def _collision_banner(
    resolved_names: list[str],
    index: list[dict[str, Any]] | None = None,
) -> list[str]:
    """One line per given-name that has two or more people."""
    if not resolved_names:
        return []
    index = index if index is not None else load_character_index()
    clusters = name_clusters(index)
    lines: list[str] = []
    seen: set[str] = set()
    for name in resolved_names:
        g = given_name(name).lower()
        if not g or g in seen:
            continue
        members = clusters.get(g)
        if not members:
            continue
        seen.add(g)
        labels = []
        for rec in members:
            label = rec["canonical_name"]
            if rec.get("warning"):
                label += " (ambiguous — not a person)"
            labels.append(label)
        display_given = members[0].get("given") or name.split()[0]
        lines.append(f"- {display_given} → " + "; ".join(labels))
    return lines


# _clerk_ce_enabled (source-derived)
def _clerk_ce_enabled() -> bool:
    raw = (os.environ.get("USE_CROSS_ENCODER") or "1").strip().lower()
    return raw not in ("0", "off", "false", "no")


# _clerk_cross_encoder (source-derived)
def _clerk_cross_encoder():
    """Lazy query-time load. Never imported at enrich/load."""
    global _CE_MODEL, _CE_ERROR
    if _CE_MODEL is not None:
        return _CE_MODEL
    if _CE_ERROR is not None:
        return None
    if not _clerk_ce_enabled():
        _CE_ERROR = RuntimeError("cross-encoder disabled")
        return None
    try:
        import logging
        for _name in (
            "httpx",
            "httpcore",
            "huggingface_hub",
            "huggingface_hub.file_download",
            "huggingface_hub.utils",
            "sentence_transformers",
            "transformers",
        ):
            logging.getLogger(_name).setLevel(logging.ERROR)
        os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        from sentence_transformers import CrossEncoder
        try:
            _CE_MODEL = CrossEncoder(_CE_MODEL_NAME, local_files_only=True)
        except Exception:
            _CE_MODEL = CrossEncoder(_CE_MODEL_NAME, local_files_only=False)
    except Exception as exc:
        _CE_ERROR = exc
        return None
    return _CE_MODEL


# _clerk_vec_encoder (source-derived)
def _clerk_vec_encoder():
    """Lazy embedder. Query-time and write-time. Soft-fail."""
    global _VEC_MODEL, _VEC_ERROR
    if _VEC_MODEL is not None:
        return _VEC_MODEL
    if _VEC_ERROR is not None:
        return None
    try:
        import logging
        for _name in (
            "httpx",
            "httpcore",
            "huggingface_hub",
            "sentence_transformers",
            "transformers",
        ):
            logging.getLogger(_name).setLevel(logging.ERROR)
        os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        from sentence_transformers import SentenceTransformer
        try:
            _VEC_MODEL = SentenceTransformer(_VEC_MODEL_NAME, local_files_only=True)
        except Exception:
            _VEC_MODEL = SentenceTransformer(_VEC_MODEL_NAME, local_files_only=False)
    except Exception as exc:
        _VEC_ERROR = exc
        return None
    return _VEC_MODEL


# _embed_texts (source-derived)
def _embed_texts(texts: list[str], query: bool = False) -> list[list[float]] | None:
    model = _clerk_vec_encoder()
    if model is None:
        return None
    payload = []
    for t in texts:
        body = (t or "").strip()
        if query:
            body = _VEC_QUERY_PREFIX + body
        payload.append(body[:2000] if body else " ")
    try:
        vecs = model.encode(payload, normalize_embeddings=True, show_progress_bar=False)
    except Exception:
        return None
    out = []
    for v in vecs:
        out.append([float(x) for x in list(v)[:_VEC_DIM]])
    return out


# _vec_blob (source-derived)
def _vec_blob(vec: list[float]) -> bytes:
    try:
        from sqlite_vec import serialize_float32
        return serialize_float32(vec)
    except Exception:
        import struct
        return struct.pack("%sf" % len(vec), *vec)


# _facts_vec_upsert (source-derived)
def _facts_vec_upsert(db: sqlite3.Connection, fact_id: int, body: str) -> bool:
    """Embed one fact body and upsert memory_facts_vec. Soft-fail."""
    if not _vec_table_ready(db):
        return False
    vecs = _embed_texts([body or ""], query=False)
    if not vecs:
        return False
    blob = _vec_blob(vecs[0])
    try:
        db.execute("DELETE FROM memory_facts_vec WHERE rowid = ?", (int(fact_id),))
        db.execute(
            "INSERT INTO memory_facts_vec(rowid, embedding) VALUES (?, ?)",
            (int(fact_id), blob),
        )
        return True
    except sqlite3.Error:
        return False


# _vec_knn (source-derived)
def _vec_knn(db: sqlite3.Connection, turn: str, k: int = _VEC_KNN) -> list[tuple[int, float]]:
    """Return [(fact_id, distance), ...] nearest to the turn. Empty on miss."""
    if not turn.strip() or not _vec_table_ready(db):
        return []
    vecs = _embed_texts([turn], query=True)
    if not vecs:
        return []
    blob = _vec_blob(vecs[0])
    try:
        rows = db.execute(
            """
            SELECT rowid, distance
            FROM memory_facts_vec
            WHERE embedding MATCH ?
            ORDER BY distance
            LIMIT ?
            """,
            (blob, int(k)),
        ).fetchall()
    except sqlite3.Error:
        try:
            rows = db.execute(
                """
                SELECT rowid, distance
                FROM memory_facts_vec
                WHERE embedding MATCH ? AND k = ?
                ORDER BY distance
                """,
                (blob, int(k)),
            ).fetchall()
        except sqlite3.Error:
            return []
    out = []
    for r in rows:
        try:
            out.append((int(r[0]), float(r[1])))
        except (TypeError, ValueError, IndexError):
            continue
    return out


# _chunks_vec_ready (source-derived)
def _chunks_vec_ready(db: sqlite3.Connection) -> bool:
    if not _VEC_READY:
        return False
    try:
        db.execute("SELECT 1 FROM chunks_vec LIMIT 0")
        return True
    except sqlite3.Error:
        return False


# _chunks_knn (source-derived)
def _chunks_knn(db: sqlite3.Connection, turn: str, k: int = _CHUNK_VEC_KNN) -> list[tuple[int, float]]:
    if not turn.strip() or not _chunks_vec_ready(db):
        return []
    vecs = _embed_texts([turn], query=True)
    if not vecs:
        return []
    blob = _vec_blob(vecs[0])
    try:
        rows = db.execute(
            """
            SELECT rowid, distance
            FROM chunks_vec
            WHERE embedding MATCH ?
            ORDER BY distance
            LIMIT ?
            """,
            (blob, int(k)),
        ).fetchall()
    except sqlite3.Error:
        try:
            rows = db.execute(
                """
                SELECT rowid, distance
                FROM chunks_vec
                WHERE embedding MATCH ? AND k = ?
                ORDER BY distance
                """,
                (blob, int(k)),
            ).fetchall()
        except sqlite3.Error:
            return []
    out = []
    for r in rows:
        try:
            out.append((int(r[0]), float(r[1])))
        except (TypeError, ValueError, IndexError):
            continue
    return out


# _v2_parse_payload (source-derived)
def _v2_parse_payload(payload: str) -> dict[str, Any]:
    """Bare text or {turn, context, mode, limit, include_chunks, diagnostic}.

    `context` is explicit prior dialogue, NEVER used as a replacement for turn.
    It is not persisted server-side; no cross-chat state can leak into retrieval.
    """
    raw = (payload or "").strip()
    if raw.startswith("{"):
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON payload: {exc.msg}") from exc
        if not isinstance(data, dict):
            raise ValueError("JSON payload must be an object")
    else:
        data = {"turn": raw}
    turn = str(data.get("turn") or data.get("query") or "").strip()[:1500]
    if not turn:
        raise ValueError("turn is required")
    context = data.get("context") or data.get("prior_turns") or ""
    if isinstance(context, list):
        context = " ".join(str(x) for x in context[-3:])
    context = str(context).strip()[-750:]
    mode = str(data.get("mode") or "fast").lower()
    if mode not in ("fast", "deep"):
        raise ValueError("mode must be 'fast' or 'deep'")
    return {
        "turn": turn,
        "context": context,
        "mode": mode,
        "limit": _clamp_limit(data.get("limit") or (8 if mode == "fast" else 12), 3, 16),
        "include_chunks": bool(data.get("include_chunks", mode == "deep")),
        "diagnostic": bool(data.get("diagnostic") or data.get("debug")),
        "use_ce": bool(data.get("use_ce", True)),
        "use_vec": bool(data.get("use_vec", True)),
    }


# _v2_tokens (source-derived)
def _v2_tokens(s: str, names: list[str] | None = None) -> list[str]:
    name_tokens = _clerk_name_tokens(names or [])
    out = []
    seen = set()
    for t in re.findall(r"[a-z0-9]+(?:'[a-z]+)?", (s or "").lower()):
        if len(t) < 3 or t in _V2_STOP or t in name_tokens or t in seen:
            continue
        seen.add(t)
        out.append(t)
    return out[:14]


# _v2_term_score (source-derived)
def _v2_term_score(text: str, toks: list[str]) -> float:
    hay = set(re.findall(r"[a-z0-9]+(?:'[a-z]+)?", (text or "").lower()))
    return float(sum(1 for t in toks if t in hay))


# _v2_ce_scores (source-derived)
def _v2_ce_scores(query: str, texts: list[str], enabled: bool) -> tuple[list[float] | None, str]:
    """Rescale MiniLM logits to bounded evidence bonuses, never mix raw logits."""
    if not enabled or not _clerk_ce_enabled():
        return None, "disabled"
    model = _clerk_cross_encoder()
    if model is None:
        return None, "unavailable"
    try:
        import math
        raw = model.predict([(query[:500], t[:900]) for t in texts],
                            batch_size=8, show_progress_bar=False)
        return [1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, float(x)))))
                for x in raw], "used"
    except Exception:
        return None, "failed"


# _v2_summary_matches (source-derived)
def _v2_summary_matches(db: sqlite3.Connection, intents: list[dict],
                        names: list[str], deep: bool, use_ce: bool) -> tuple[list[dict], str]:
    """Look across segment=0 and windows, grouping windows by source.

    In-memory lexical scan over hundreds of summaries (not 222k raw chunks).
    Cross-encoder only reranks the short candidate shortlist.
    """
    rows = db.execute(
        "SELECT source, segment, source_ord, title, short_summary, "
        "full_summary, key_themes, key_events, characters "
        "FROM source_summaries WHERE status = 'done' "
        "ORDER BY source_ord DESC LIMIT ?", (_V2_SUMMARY_POOL,)
    ).fetchall()
    scored = []
    primary = intents[0]["query"] if intents else ""
    for r in rows:
        source = r["source"] or ""
        title = r["title"] or ""
        short = r["short_summary"] or ""
        theme = (r["key_themes"] or "") + " " + (r["key_events"] or "")
        full = (r["full_summary"] or "")[:2200]
        chars = (r["characters"] or "")
        val = 0.0
        signals = []
        for it in intents:
            ts = it["tokens"]
            # A named activity should win over ambient mentions of people.
            overlap = (3.5 * _v2_term_score(title, ts)
                       + 2.7 * _v2_term_score(short, ts)
                       + 2.1 * _v2_term_score(theme, ts)
                       + 0.7 * _v2_term_score(full, ts))
            q = it["query"].lower().strip()
            if len(q) >= 5 and q in (title + " " + short + " " + theme).lower():
                overlap += 8
            if it["kind"] == "activity" and overlap:
                overlap *= 1.25
            if overlap * it["weight"] > val:
                val = overlap * it["weight"]
                signals = [it["kind"]]
        if val <= 0:
            continue
        # Names are modest supporting evidence, never the whole story.
        names_here = sum(1 for n in names if re.search(
            r"(?<!\w)" + re.escape(n) + r"(?!\w)", chars, re.I))
        val += min(names_here, 2) * 0.6
        val += min(max(0, int(r["source_ord"] or 0)) / 4000.0, 0.6)
        scored.append({
            "source": source, "segment": int(r["segment"]),
            "source_ord": r["source_ord"], "title": title[:150],
            "summary": (short or full)[:500 if deep else 330],
            "score": val, "reason": "+".join(signals),
            "_ce_text": " ".join([title, short, theme, full[:550]])[:1100],
        })
    scored.sort(key=lambda x: -x["score"])
    ce_status = "skipped"
    if scored:
        pool = scored[:_V2_CE_SUMMARY_POOL]
        ce, ce_status = _v2_ce_scores(primary, [r["_ce_text"] for r in pool], use_ce)
        if ce is not None:
            for r, rank in zip(pool, ce):
                r["score"] += rank * 5.0
                r["reason"] += "+ce"
            scored.sort(key=lambda x: -x["score"])
    # Return a useful whole-chat parent plus its relevant windows when present.
    by_source: dict[str, list[dict]] = {}
    for row in scored:
        by_source.setdefault(row["source"], []).append(row)
    grouped = []
    for src, members in by_source.items():
        members.sort(key=lambda x: -x["score"])
        best = members[0]
        whole = next((r for r in members if r["segment"] == 0), None)
        windows = [r for r in members if r["segment"] != 0]
        windows.sort(key=lambda x: -x["score"])
        grouped.append({
            "source": src, "score": round(best["score"], 2),
            "whole": {k: whole[k] for k in ("segment", "title", "summary")}
                     if whole else None,
            "windows": [{k: r[k] for k in ("segment", "title", "summary", "score")}
                        for r in windows[:3 if deep else 2]],
            "reason": best["reason"],
        })
    grouped.sort(key=lambda x: -x["score"])
    return grouped[:8 if deep else 5], ce_status


# _v3_queries (source-derived)
def _v3_queries(args, names):
    turn, context = args["turn"], args["context"]
    elliptical = len(turn.split()) < 10 or bool(_V2_ELLIPSIS.search(turn))
    primary = (context + " " + turn).strip() if context and elliptical else turn
    toks = _v2_tokens(primary, names)
    if not toks:
        toks = _v2_tokens(primary)
    # Full phrase, narrow AND, broad OR: complementary retrieval paths.
    phrases = re.findall(r"[\w']+", primary.lower())
    queries = []
    if 2 <= len(phrases) <= 7:
        queries.append('"' + ' '.join(phrases).replace('"', '') + '"')
    if len(toks) >= 2:
        queries.append(' AND '.join('"'+x+'"' for x in toks[:3]))
    # Exact structured references should not disappear when a long question's
    # first three lexical tokens are unrelated to the reference digits.
    for kind, phrase, words in _v34_anchors(primary):
        if kind == 'reference' and len(words) >= 3:
            queries.insert(0, ' AND '.join('"'+w+'"' for w in words[:3]))
    if toks:
        queries.append(' OR '.join('"'+x+'"' for x in toks[:8]))
    return primary, toks, list(dict.fromkeys(queries))


# _v3_overlap (source-derived)
def _v3_overlap(text, tokens):
    words = set(re.findall(r"[a-z0-9]+", (text or '').lower()))
    if not tokens: return 0.0
    return sum(1 for x in tokens if x in words) / max(1, len(tokens))


# _v33_windows (source-derived)
def _v33_windows(body, tokens, query, width=850, max_windows=3):
    """Independently score bounded, overlapping evidence windows.

    Anchor candidates on every term occurrence and query bigram; keep the
    highest-scoring non-overlapping windows, not the parent chunk's opening.
    """
    if not body: return []
    clean=re.sub(r"\s+",' ',body)
    low=clean.lower()
    terms=list(dict.fromkeys(t.lower() for t in tokens if len(t)>1))
    query_words=re.findall(r'[a-z0-9]+',query.lower())
    bigrams=[' '.join(query_words[i:i+2]) for i in range(len(query_words)-1)]
    spans=[]
    for term in terms:
        spans.extend((m.start(),m.end()) for m in re.finditer(r'(?<!\w)'+re.escape(term)+r'(?!\w)',low))
    for phrase in bigrams:
        spans.extend((m.start(),m.end()) for m in re.finditer(r'(?<!\w)'+re.escape(phrase)+r'(?!\w)',low))
    if not spans: return [(clean[:width],0.,0.)]
    # Limit work for very large raw sources while retaining distributed coverage.
    spans=sorted(set(spans))
    if len(spans)>450:
        stride=max(1,len(spans)//450)
        spans=spans[::stride][:450]
    centers={max(0,min(len(clean)-width,max(0,(a+b)//2-width//2))) for a,b in spans}
    ranked=[]
    for start in centers:
        window=clean[start:start+width]; wl=window.lower()
        present={t for t in terms if re.search(r'(?<!\w)'+re.escape(t)+r'(?!\w)',wl)}
        coverage=len(present)/max(1,len(terms))
        phrase_hits=sum(1 for p in bigrams if p in wl)
        # Density rewards coherent local discussion over scattered mentions.
        matches=sum(len(re.findall(r'(?<!\w)'+re.escape(t)+r'(?!\w)',wl)) for t in present)
        density=min(1.,matches/max(1,len(window.split()))*9)
        score=coverage*14+min(3,phrase_hits)*3+density*2
        ranked.append((score,start,window,coverage))
    ranked.sort(key=lambda x:(-x[0],x[1]))
    selected=[]
    for score,start,window,coverage in ranked:
        if any(abs(start-old)<width//2 for old,_,_,_ in selected): continue
        selected.append((start,window,score,coverage))
        if len(selected)>=max_windows: break
    return [(w,score,coverage) for _,w,score,coverage in selected]


# _v34_anchors (source-derived)
def _v34_anchors(query):
    """Extract named phrases, quoted language and chapter:verse references.

    These are *not* claims or topic labels. They let general retrieval preserve
    discriminating expressions that standard stop-word stripping can lose.
    """
    found=[]; seen=set()
    for match in _V34_REF_RE.finditer(query):
        words=tuple(re.findall(r'[a-z0-9]+',match.group(0).lower()))
        key=('reference',words)
        if key not in seen: found.append(('reference',match.group(0),words));seen.add(key)
    for match in re.finditer(r'["“”]([^"“”]{3,75})["“”]',query):
        words=tuple(re.findall(r'[a-z0-9]+',match.group(1).lower()))
        if len(words)<2: continue
        key=('quoted',words)
        if key not in seen: found.append(('quoted',match.group(1),words));seen.add(key)
    for match in _V34_CAP_RE.finditer(query):
        words=tuple(re.findall(r'[a-z0-9]+',match.group(0).lower()))
        # Suppress ordinary sentence-initial constructions such as "Did [PERSON]".
        if words[0] in ('did','does','what','how','when','where','why','was',
                        'were','have','has','will','could','would','can','is','are'):
            continue
        key=('named',words)
        if key not in seen: found.append(('named',match.group(0),words));seen.add(key)
    return found[:6]


# _v34_anchor_score (source-derived)
def _v34_anchor_score(text, anchors):
    """Reward an exact expression, or a near expression with an inserted word.

    A reference is stronger than a generic capitalised expression. Presence
    of an isolated word (e.g. '[PLACE]') is NOT treated as an anchor match.
    """
    if not text or not anchors:return 0.0, False
    low=re.sub(r'[^a-z0-9]+',' ',text.lower()).strip()
    natural=text.lower()
    best=0.0; exact=False
    for kind, _phrase, words in anchors:
        if len(words)<2:continue
        joined=' '.join(words)
        target=low if kind=='reference' else natural
        joined_pat=r'\s+'.join(re.escape(w) for w in words) if kind!='reference' else re.escape(joined)
        if re.search(r'(?<!\w)'+joined_pat+r'(?!\w)',target):
            best=max(best,22.0 if kind=='reference' else (12.0 if kind=='quoted' else 9.0));exact=True
            continue
        if kind=='reference':continue  # chapter and verse must be contiguous
        # Allow near phrases such as 'Oh [PLACE] in [PLACE]'. Require full ordered
        # coverage, with at most two intervening tokens per adjacent pair.
        pat=r'(?<!\w)'+r'(?:\s+\w+){0,2}\s+'.join(re.escape(w) for w in words)+r'(?!\w)'
        if re.search(pat,natural):
            best=max(best,14.0 if kind=='reference' else 6.0)
    return best,exact


# _v35_chunk_vectors (source-derived)
def _v35_chunk_vectors(db, query, chunks_raw, k=_V35_CHUNK_VEC_K):
    """Add active raw-chunk KNN results to FTS candidates by stable chunk ID.

    Rowid is the vec0 locator, but chunk ID is Wilson's deduplication key.
    No new database writes. The opt-in/source-active table is compulsory so
    an in-progress reload cannot contribute partially indexed evidence.
    Returns (active KNN matches, new semantic-only chunk IDs).
    """
    knn = _chunks_knn(db, query, k)
    if not knn:
        return 0, 0
    ids = [int(rid) for rid, _dist in knn]
    marks = ','.join('?' for _ in ids)
    rows = db.execute(
        'SELECT c.rowid AS rid,c.id,c.source,c.source_ord,c.body '
        'FROM chunks c JOIN retrieval_chunk_vec_sources a ON a.source=c.source '
        f'WHERE c.rowid IN ({marks}) AND a.active=1', ids
    ).fetchall()
    found = {int(r['rid']): dict(r) for r in rows}
    n_active = n_only = 0
    for rank, (rid, distance) in enumerate(knn, 1):
        r = found.get(int(rid))
        if r is None:
            continue
        cid = r['id']
        already = cid in chunks_raw
        if not already:
            # The normal FTS collector exposes exactly these fields.
            chunks_raw[cid] = {'row': {field: r[field] for field in
                              ('id', 'body', 'source', 'source_ord')}, 'routes': []}
            n_only += 1
        item = chunks_raw[cid]
        # All stored embeddings and query embeddings are L2-normalised.
        # sqlite-vec's default vec0 distance is Euclidean distance.
        cosine = max(-1.0, min(1.0, 1.0 - float(distance)**2 / 2.0))
        item['vector_rank'] = rank
        item['vector_cosine'] = cosine
        n_active += 1
    return n_active, n_only


# _v35_semantic_grounded (source-derived)
def _v35_semantic_grounded(candidate):
    """Semantic-only evidence needs BOTH close BGE and a strong MiniLM vote.

    Query-time CE is a relevance signal, NOT a factual entailment verdict.
    Lexical/anchored evidence continues to use Wilson's normal thresholds.
    With CE unavailable, semantic-only candidates cannot bypass lexical gates.
    Exact structured references must actually occur in the excerpt.
    """
    return (bool(candidate.get('vector_rank'))
            and candidate.get('vector_cosine', -1.0) >= _V35_SEMANTIC_COS
            and candidate.get('_ce_score', 0.0) >= _V35_SEMANTIC_CE
            and (not candidate.get('reference_required') or candidate.get('anchor_exact')))


# _v34_rerank_passages (source-derived)
def _v34_rerank_passages(query,candidates,enabled,deadline):
    """Give independent source/chunk windows a chance at the CE shortlist.

    v3.3 took the highest 24 raw windows, which could be dominated by several
    nearly identical snippets of the same long chunk. Prefer best window from
    each chunk first, then fill from other windows, with a per-source cap.
    """
    if not enabled or not candidates or time.perf_counter()>=deadline-.20:
        return 'budget-skipped'
    ordered=sorted(candidates,key=lambda x:-x['score'])
    chosen=[]; seen_chunks=set(); source_counts={}
    # Reserve a handful of CE seats for semantic-only raw chunks. Without this,
    # the lexical top 30 usually excludes them before their relevance is tested.
    semantic=sorted((c for c in candidates
                     if c.get('vector_rank') and not c.get('fts_routes')
                     and c.get('vector_cosine', -1.0) >= _V35_SEMANTIC_COS
                     and (not c.get('reference_required') or c.get('anchor_exact'))),
                    key=lambda c:(-c['vector_cosine'], c['vector_rank']))
    for c in semantic:
        if len(chosen)>=6:break
        if c['chunk_id'] in seen_chunks or source_counts.get(c['source'],0)>=2:
            continue
        chosen.append(c);seen_chunks.add(c['chunk_id'])
        source_counts[c['source']]=source_counts.get(c['source'],0)+1
    for round_ in (0,1):
        for c in ordered:
            if len(chosen)>=30:break
            key=c['chunk_id']; source=c['source']
            if round_==0 and key in seen_chunks:continue
            if c in chosen:continue
            if source_counts.get(source,0)>=3:continue
            chosen.append(c); seen_chunks.add(key)
            source_counts[source]=source_counts.get(source,0)+1
        if len(chosen)>=30:break
    scores,status=_v2_ce_scores(query,[x['_text'] for x in chosen],True)
    if scores is not None:
        for c,value in zip(chosen,scores):
            c['_ce_score']=float(value)
            c['score']+=9.0*value
            c['why']+='+ce'
    return status


# _v34_select_passages (source-derived)
def _v34_select_passages(candidates, tokens, deep):
    """Rank passages, filter weak followers, diversify sources/chunks.

    Dynamic filtering is relative to the best score while keeping an absolute
    floor. An exact match for a structured reference can survive if it is not
    the strongest semantic passage.
    """
    for c in candidates:
        c['score']+=c.get('coverage',0)*14
    ranked=sorted(candidates,key=lambda x:-x['score'])
    if not ranked:return []
    best=ranked[0]['score']
    cutoff=max(14.0,best*(0.58 if deep else 0.65))
    strict_short_name=len(tokens)<=2 and any(c.get('anchor_exact') for c in ranked)
    chosen=[]; used_chunks=set(); source_counts={}
    for phase in (0,1):
        for c in ranked:
            if len(chosen)>=(8 if deep else 4):break
            if strict_short_name and not c.get('anchor_exact'):continue
            semantic=_v35_semantic_grounded(c)
            if c['score']<cutoff and not (c.get('anchor_exact') and c['score']>=14) and not semantic:continue
            if len(tokens)>=3 and c.get('coverage',0)<0.34 and not c.get('anchor_exact') and not semantic:continue
            # Do not admit a low-coverage vector hit merely because its rank
            # gave it a good fusion score: the CE must independently agree.
            if not c.get('fts_routes') and c.get('coverage',0)<0.34 and not c.get('anchor_exact') and not semantic:continue
            if c['chunk_id'] in used_chunks:continue
            if (source_counts.get(c['source'],0)>0) != bool(phase):continue
            if source_counts.get(c['source'],0)>=2:continue
            c['semantic_validated']=semantic
            chosen.append({k:(round(v,2) if k=='score' else v)
                           for k,v in c.items() if not k.startswith('_')})
            used_chunks.add(c['chunk_id']);source_counts[c['source']]=source_counts.get(c['source'],0)+1
        if len(chosen)>=(8 if deep else 4):break
    # v3.4.1: a source can be strongly ranked yet lose *all* excerpts to the
    # relative cutoff / strict short-name filter. Recover only independently
    # grounded local windows; never fall back to an arbitrary top-ranked chunk.
    if not chosen:
        grounded = [c for c in ranked
                    if c['score'] >= 14.0 and
                    (c.get('anchor_exact') or c.get('coverage',0) >= .60
                     or _v35_semantic_grounded(c))]
        # An explicit structured/name anchor is stronger than diffuse coverage.
        grounded.sort(key=lambda c:(not bool(c.get('anchor_exact')), -c['score']))
        for c in grounded:
            if len(chosen)>= (2 if deep else 1):break
            if c['chunk_id'] in used_chunks:continue
            if chosen and c['source']==chosen[0]['source']:continue
            c['semantic_validated']=_v35_semantic_grounded(c)
            chosen.append({k:(round(v,2) if k=='score' else v)
                           for k,v in c.items() if not k.startswith('_')})
            used_chunks.add(c['chunk_id'])
    return chosen


# _v34_intent (source-derived)
def _v34_intent(turn):
    """Classify evidence shape without conflating every 'how' with a history."""
    low=turn.lower()
    longitudinal=bool(re.search(r"\b(develop\w*|evolv\w*|chang\w*|history|matur\w*|progress\w*|"
                                r"timeline|compar\w*|throughout)\b|\bover (?:the |a )?(?:years?|months?|time)\b|"
                                r"\bacross (?:the )?(?:years?|months?|time)\b",low))
    verification=(bool(re.search(r"\b(did|does|was|were|has|have|is|are|could)\b",low))
                  and '?' in turn and not low.startswith(('how ','why ')))
    return 'longitudinal' if longitudinal else ('verification' if verification else 'lookup')


# _v34_assess_evidence (source-derived)
def _v34_assess_evidence(turn,tokens,facts,passages):
    """Heuristic triage, not an entailment judgement.

    Reject ungrounded source-count reasoning: many thematically weak chunks do
    not together establish a longitudinal narrative. Relevant text still needs
    inspection, especially for reports of people's beliefs or motives.
    """
    intent=_v34_intent(turn)
    ordinary=[x for x in facts if not x.get('pinned')]
    cover=lambda s:_v3_overlap(s,tokens)
    best=max([cover(p.get('excerpt','')) for p in passages]+
             [cover(f.get('fact','')) for f in ordinary]+[0.])
    qualified=[p for p in passages if (p.get('coverage',0)>=.34
              or p.get('anchor_exact') or p.get('semantic_validated'))]
    distinct=len({p['source'] for p in qualified})
    if not qualified and not ordinary:status='insufficient-evidence'
    elif intent=='verification':
        status='candidate-evidence' if best>=.60 and bool(qualified or ordinary) else 'insufficient-evidence'
    elif intent=='longitudinal':
        status='candidate-evidence' if (distinct>=2 and best>=.25) or (distinct>=1 and len(ordinary)>=2) else 'limited-evidence'
    else:
        status='candidate-evidence' if best>=.34 else 'weak-evidence'
    return status,round(best,3),intent,distinct


# _v3_collect (source-derived)
def _v3_collect(db, table, queries, limit=110, deadline=None, min_candidates=12):
    out={}; errors=[]
    if table == 'facts':
        sql = ('SELECT m.id AS id,m.fact AS body,m.category AS category,'
               'm.confidence AS confidence,m.episode_id AS episode_id,'
               'm.supporting_sources AS supporting_sources,m.entities AS entities,'
               'bm25(memory_facts_fts) AS rank FROM memory_facts_fts '
               'JOIN memory_facts m ON m.id=memory_facts_fts.rowid '
               'WHERE memory_facts_fts MATCH ? ORDER BY rank LIMIT ?')
    else:
        sql = ('SELECT c.id AS id,c.body AS body,c.source AS source,'
               'c.source_ord AS source_ord,bm25(chunks_fts) AS rank '
               'FROM chunks_fts JOIN chunks c ON c.rowid=chunks_fts.rowid '
               'WHERE chunks_fts MATCH ? ORDER BY rank LIMIT ?')
    for qi,q in enumerate(queries):
        # Run every retrieval route (including broad OR) while within the FTS
        # deadline. Candidate count alone does not establish candidate quality;
        # skipping the final route can hide relevant evidence from reranking.
        if deadline is not None and time.perf_counter() >= deadline:
            break
        try:
            for rank,r in enumerate(db.execute(sql,(q,limit)).fetchall(),1):
                key=r['id']; rr=out.setdefault(key, {'row':dict(r),'routes':[]})
                rr['routes'].append((qi,rank))
        except sqlite3.Error as exc:
            errors.append(f'{table}:{str(exc)[:110]}')
    return out,errors


# _v3_rerank (source-derived)
def _v3_rerank(query, candidates, enabled, deadline):
    # Only spend inference time while inside the remaining soft budget.
    if not enabled or not candidates or time.perf_counter() >= deadline - .18:
        return 'budget-skipped'
    top=sorted(candidates,key=lambda x:-x['score'])[:24]
    scores,status=_v2_ce_scores(query,[x['_text'] for x in top],True)
    if scores is not None:
        for row,value in zip(top,scores):
            row['score'] += 8.0*value
            row['why'] += '+ce'
    return status


# _v3_retrieve (source-derived)
def _v3_retrieve(args):
    import time
    start=time.perf_counter(); deadline=start+_V3_BUDGET_MS/1000
    turn=args['turn']; context=args['context']; deep=args['mode']=='deep'
    names=_filter_live_names(candidate_names(turn),turn)
    if not names and context and (len(turn.split())<10 or _V2_ELLIPSIS.search(turn)):
        names=_filter_live_names(candidate_names(context),context)[:5]
    stream,inferred=infer_stream(turn)
    query,tokens,queries=_v3_queries(args,names)
    anchors=_v34_anchors(query)
    timings={'planning_ms':round((time.perf_counter()-start)*1000,1)}
    errors=[]; db=connect()
    try:
        t=time.perf_counter()
        # The evidence pool gets a guaranteed opportunity before broad fact searches.
        # Bound candidate generation independently of the overall 2-second budget.
        fts_deadline=min(deadline-_V3_EVIDENCE_RESERVE_MS/1000,
                         t+_V3_FTS_STAGE_CAP_MS/1000)
        chunks_raw,e=_v3_collect(db,'chunks',queries,65,
                                  deadline=fts_deadline,min_candidates=18); errors+=e
        facts_raw,e=_v3_collect(db,'facts',queries,65,
                                 deadline=fts_deadline,min_candidates=18); errors+=e
        timings['fts_ms']=round((time.perf_counter()-t)*1000,1)
        t=time.perf_counter(); vector_count=0; chunk_vector_count=0; chunk_vector_only=0
        # Give primary raw evidence first chance at the vector/time budget.
        # This uses the same query text/prefix/model as the fact-vector route.
        if args['use_vec'] and time.perf_counter()<deadline-.65:
            try:
                chunk_vector_count,chunk_vector_only=_v35_chunk_vectors(db,query,chunks_raw)
            except Exception as exc:
                errors.append('chunk_vector:'+type(exc).__name__)
        if args['use_vec'] and time.perf_counter()<deadline-.65:
            try:
                for rank,(fid,dist) in enumerate(_vec_knn(db,query,80),1):
                    if fid not in facts_raw:
                        row=db.execute('SELECT id,fact AS body,category,confidence,episode_id,supporting_sources,entities FROM memory_facts WHERE id=?',(fid,)).fetchone()
                        if row: facts_raw[fid]={'row':dict(row),'routes':[]}
                    if fid in facts_raw: facts_raw[fid]['vector_rank']=rank; vector_count+=1
            except Exception as exc: errors.append('vector:'+type(exc).__name__)
        timings['vector_ms']=round((time.perf_counter()-t)*1000,1)
        t=time.perf_counter()
        pins={int(p['fact_id']) for p in _load_retrieval_pins(db) if p['stream'] in ('*','mixed',stream)}  # Private forced-pin ID removed for publication
        for fid in pins:
            if fid not in facts_raw:
                row=db.execute('SELECT id,fact AS body,category,confidence,episode_id,supporting_sources,entities FROM memory_facts WHERE id=?',(fid,)).fetchone()
                if row: facts_raw[fid]={'row':dict(row),'routes':[]}
        fact_candidates=[]
        for fid,item in facts_raw.items():
            r=item['row']; body=r['body'] or ''; pinned=fid in pins
            if (r.get('category') or '').lower()=='page-layer' and stream not in ('page','mixed') and not pinned: continue
            lexical=_v3_overlap(body,tokens)
            rrf=sum(1/(50+rank) for _,rank in item['routes'])
            if item.get('vector_rank'): rrf+=1/(50+item['vector_rank'])
            anchor_bonus,anchor_exact=_v34_anchor_score(body,anchors)
            score=lexical*17+rrf*350+anchor_bonus+(1000 if pinned else 0)
            if not pinned and score<1.5: continue
            fact_candidates.append({'id':fid,'category':r.get('category'),'fact':body[:530 if deep else 350],
                'confidence':r.get('confidence'),'episode_id':r.get('episode_id'),
                'supporting_sources':r.get('supporting_sources'),'pinned':pinned,
                'score':score,'anchor_exact':anchor_exact,'why':'pin' if pinned else 'fts/vec','_text':body[:1100]})
        chunk_candidates=[]
        for item in chunks_raw.values():
            r=item['row']; body=r['body'] or ''
            # Score independent local windows and keep the strongest, so a
            # multi-topic source cannot win solely on scattered keywords.
            windows=_v33_windows(body,tokens,query,width=850,max_windows=3)
            rrf=sum(1/(50+rank) for _,rank in item['routes'])
            vec_rank=item.get('vector_rank')
            cosine=item.get('vector_cosine',-1.0)
            if vec_rank: rrf+=1/(50+vec_rank)
            fts_routes=len(item['routes'])
            for wi,(window,local_score,coverage) in enumerate(windows):
                anchor_bonus,anchor_exact=_v34_anchor_score(window,anchors)
                anchor_penalty=8 if anchors and not anchor_bonus and any(a[0] in ('named','reference') for a in anchors) else 0
                # RRF unifies literal and semantic retrieval; cosine is a small
                # *candidate* bonus, not proof that this excerpt answers the turn.
                score=local_score+rrf*240+anchor_bonus-anchor_penalty
                if vec_rank: score+=14.0*max(0.0,cosine)
                if score<2:continue
                route=('fts+vec' if vec_rank and fts_routes else
                       'vec' if vec_rank else 'fts')
                chunk_candidates.append({'source':r['source'],'chunk_id':r['id'],
                    'source_ord':r['source_ord'],'excerpt':window[:750],
                    'score':score,'why':route+'+local-window'+('+anchor' if anchor_bonus else ''),
                    'coverage':round(coverage,3),'anchor_exact':anchor_exact,
                    'fts_routes':fts_routes,'vector_rank':vec_rank,
                    'vector_similarity':round(cosine,3) if vec_rank else None,
                    'vector_cosine':cosine,'reference_required':any(a[0]=='reference' for a in anchors),
                    '_text':window[:850],'_window':wi})
        timings['fusion_ms']=round((time.perf_counter()-t)*1000,1)
        t=time.perf_counter()
        # Rerank primary evidence FIRST; the fact reranker may be skipped if
        # necessary, but it cannot starve the passages that substantiate claims.
        chunk_ce=_v34_rerank_passages(query,chunk_candidates,args['use_ce'],deadline)
        fact_ce=_v3_rerank(query,fact_candidates,args['use_ce'],deadline)
        timings['rerank_ms']=round((time.perf_counter()-t)*1000,1)
        t=time.perf_counter()
        # Relevance-ranked results must not be obscured by mandatory system
        # context. Keep facts backward-compatible (pins appended, never first),
        # and also expose clean channels for consumers that understand v3.4.
        fact_candidates.sort(key=lambda x:(x['pinned'],-x['score']))
        strict_short_fact=len(tokens)<=2 and any(x.get('anchor_exact') for x in fact_candidates if not x['pinned'])
        facts=[]; pinned_context=[]; ordinary_facts=[]; episodes={}
        for f in fact_candidates:
            if strict_short_fact and not f['pinned'] and not f.get('anchor_exact'):continue
            if not f['pinned'] and f['score'] < 17.0:continue
            ep=f['episode_id']
            if ep and not f['pinned'] and episodes.get(ep,0)>=(3 if deep else 2):continue
            if ep and not f['pinned']:episodes[ep]=episodes.get(ep,0)+1
            item={k:(round(v,2) if k=='score' else v) for k,v in f.items() if not k.startswith('_')}
            if f['pinned']:
                pinned_context.append(item)
            elif len(ordinary_facts)<args['limit']:
                ordinary_facts.append(item)
        facts=(ordinary_facts+pinned_context)[:args['limit']]
        # Local-source diversity and strength gating are applied consistently
        # to excerpts and the source list below.
        passages=_v34_select_passages(chunk_candidates,tokens,deep)
        chunk_candidates.sort(key=lambda x:-x['score'])
        # Summaries: fetch whole-chat narrative for the discovered primary evidence,
        # and use lexical summary discovery as a fallback for weak raw coverage.
        sources=[]
        source_scores={}
        strict_short_source=len(tokens)<=2 and any(c.get('anchor_exact') for c in chunk_candidates)
        for c in chunk_candidates:
            if strict_short_source and not c.get('anchor_exact'):continue
            if (c['score']<max(14,chunk_candidates[0]['score']*.58)
                    and not _v35_semantic_grounded(c)):continue
            if (len(tokens)>=3 and c.get('coverage',0)<0.34
                    and not c.get('anchor_exact') and not _v35_semantic_grounded(c)):continue
            if not c.get('fts_routes') and c.get('coverage',0)<0.34 and not c.get('anchor_exact') and not _v35_semantic_grounded(c):continue
            src=c['source'];source_scores[src]=max(source_scores.get(src,0),c['score'])
        for f in ordinary_facts:
            try: refs=json.loads(f['supporting_sources'] or '[]')
            except (ValueError,TypeError):refs=[]
            for ref in refs:
                if isinstance(ref,str):source_scores.setdefault(ref,3.0)
        selected=sorted(source_scores,key=lambda s:-source_scores[s])[:(10 if deep else 6)]
        if selected:
            ph=','.join('?' for _ in selected)
            rows=db.execute('SELECT source,segment,source_ord,title,short_summary,full_summary FROM source_summaries '
                f'WHERE source IN ({ph}) AND status="done" ORDER BY segment ASC',selected).fetchall()
            grouped={}
            for r in rows:grouped.setdefault(r['source'],[]).append(r)
            for src in selected:
                rs=grouped.get(src,[]);whole=next((r for r in rs if r['segment']==0),None)
                windows=[{'segment':r['segment'],'title':r['title'],'summary':(r['short_summary'] or r['full_summary'] or '')[:350]}
                    for r in rs if r['segment']!=0][:2]
                sources.append({'source':src,'score':round(source_scores[src],2),
                    'source_ord':whole['source_ord'] if whole else (rs[0]['source_ord'] if rs else None),
                    'whole':({'segment':0,'title':whole['title'],'summary':(whole['short_summary'] or whole['full_summary'] or '')[:480]} if whole else None),
                    'windows':windows,'reason':'raw/fact evidence'})
        if (not sources or (len(sources)<3 and len(tokens)>2)) and time.perf_counter()<deadline-.12:
            # Existing generic summary scan remains a fallback, not a hard gate.
            intents=[{'query':query,'tokens':tokens,'weight':1.0,'kind':'current'}]
            fallback,_=_v2_summary_matches(db,intents,names,deep,False)
            known={x['source'] for x in sources}
            sources.extend(x for x in fallback if x['source'] not in known)
        timings['assembly_ms']=round((time.perf_counter()-t)*1000,1)
    finally:db.close()
    timings['total_ms']=round((time.perf_counter()-start)*1000,1)
    support_status,support_coverage,support_intent,support_sources=_v34_assess_evidence(turn,tokens,ordinary_facts,passages)
    out={'ok':True,'version':'wilson-v3.4.3-experimental','mode':args['mode'],'turn':turn,
        'context_used':bool(context and (len(turn.split())<10 or _V2_ELLIPSIS.search(turn))),
        'stream':stream,'stream_inferred':inferred,'names':names,'collisions':_collision_banner(names),
        'activities':[],'facts':facts,'relevance_facts':ordinary_facts,
        'pinned_context':pinned_context,'sources':sources[:(8 if deep else 5)],
        'passages':passages if args['include_chunks'] else [],
        'retrieval_status':{'facts_fts':bool(facts_raw),'facts_vector':vector_count>0,
            'chunks_vector':chunk_vector_count>0,'chunk_vectors_added':chunk_vector_only,
            'fact_cross_encoder':fact_ce,'summary_cross_encoder':'not-used',
            'passage_cross_encoder':chunk_ce,'errors':errors}}
    out['evidence_assessment']={'status':support_status,'max_query_coverage':support_coverage,
        'intent':support_intent,'distinct_passage_sources':support_sources,
        'heuristic_only':True,
        'note':'Candidate evidence is not entailment or verification; inspect original passages.'}
    if args['diagnostic']:
        out['diagnostic']={'queries':queries,'candidates':{'facts':len(facts_raw),'chunks':len(chunks_raw),
            'vectors':vector_count,'chunk_vectors':chunk_vector_count,'chunk_vector_only':chunk_vector_only,
            'semantic_passages_validated':sum(1 for p in passages if p.get('semantic_validated'))},
            'timings':timings,'query_anchors':[{'kind':k,'phrase':p} for k,p,_ in anchors],
            'budget_ms':_V3_BUDGET_MS,'budget_exceeded':timings['total_ms']>_V3_BUDGET_MS,
            'limitations':['two-second soft budget: in-flight SQLite/model calls cannot be interrupted',
                'summary provenance is navigational; verify exact claims against raw excerpts',
                'context is supplied by caller, not persisted by retrieval',
                'pinned context is metadata, not evidence of the query claim']}
    # v3.4.2: raw excerpts are the primary evidence, not expendable padding.
    # Previous releases deleted passages FIRST when source summaries, duplicate
    # fact channels and diagnostic metadata exceeded the connector byte cap.
    # Keep the original candidate count to distinguish filtering from clipping.
    _selected_passages=len(out['passages'])
    _output_pruned={'source_windows':0,'source_summaries':0,'sources':0,
                    'fact_duplicates':0,'passages':0}
    _prune_steps=0
    while len(_dumps(out).encode('utf-8'))>(_V2_MAX_BYTES-(450 if args['diagnostic'] else 0)) and _prune_steps<200:
        _prune_steps+=1
        # The two parallel fact channels duplicate content in 'facts'. Consumers
        # still get ranked facts and the mandatory pinned_context separately.
        if out.get('relevance_facts'):
            out['relevance_facts']=[];_output_pruned['fact_duplicates']+=1
        elif any(x.get('windows') for x in out['sources']):
            for x in out['sources']:
                if x.get('windows'):
                    x['windows']=[];_output_pruned['source_windows']+=1
        elif any(x.get('whole') and len(x['whole'].get('summary',''))>140 for x in out['sources']):
            for x in out['sources']:
                if x.get('whole') and len(x['whole'].get('summary',''))>140:
                    x['whole']['summary']=x['whole']['summary'][:140]
                    _output_pruned['source_summaries']+=1
        elif len(out['sources'])>2:
            out['sources'].pop();_output_pruned['sources']+=1
        elif len(out['facts'])>2:
            # Trim the least relevant ordinary fact first, then redundant pins;
            # pinned_context remains separately available to the caller.
            ix=next((i for i in range(len(out['facts'])-1,-1,-1)
                     if not out['facts'][i].get('pinned')),len(out['facts'])-1)
            out['facts'].pop(ix);_output_pruned['fact_duplicates']+=1
        elif len(out.get('pinned_context',[]))>1:
            out['pinned_context'].pop();_output_pruned['fact_duplicates']+=1
        elif len(out['passages'])>1:
            out['passages'].pop();_output_pruned['passages']+=1
        else:
            out['truncated']=True;break
    if args['diagnostic']:
        out['diagnostic']['passage_output']={
            'selected_before_size_limit':_selected_passages,
            'returned':len(out['passages']),
            'pruned':_output_pruned,
            'reason_if_empty':('no-passage-candidates-selected' if not _selected_passages
                               else 'output-size-limit' if not out['passages'] else None)}
    return out


# Minimal lexical helper for the source-derived _v2_tokens routine.
def _clerk_name_tokens(names):
    return {w.lower() for name in names for w in re.findall(r"[A-Za-z0-9']{3,}", name)}

# Core fact listing, with the same ID-first / filtered-search design.
@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
def memory_facts_list(query: str="", category: str="", fact_id: int=0,
                      limit: int=20, auth_token: str="") -> str:
    denied=require_auth(auth_token)
    if denied: return denied
    db=connect()
    try:
        if fact_id:
            row=db.execute("SELECT * FROM memory_facts WHERE id=?",(int(fact_id),)).fetchone()
            return _ok({"facts":[dict(row)] if row else []})
        if query:
            term=' OR '.join('"'+x+'"' for x in re.findall(r"[A-Za-z0-9]{3,}",query)[:8])
            rows=db.execute("SELECT m.* FROM memory_facts_fts f JOIN memory_facts m "
                            "ON m.id=f.rowid WHERE memory_facts_fts MATCH ? "
                            "ORDER BY bm25(memory_facts_fts) LIMIT ?",(term,limit)).fetchall()
        else:
            rows=db.execute("SELECT * FROM memory_facts WHERE (?='' OR category=?) "
                            "ORDER BY id DESC LIMIT ?",(category,category,limit)).fetchall()
        return _ok({"facts":[dict(r) for r in rows]})
    finally: db.close()

ALLOWED_PROMINENCE = frozenset({"central", "major", "mentioned"})


# entity_sources_list: core-schema tool (source-derived).
@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
def entity_sources_list(
    name: str = "",
    source: str = "",
    prominence: str = "",
    limit: int = 80,
    auth_token: str = "",
) -> str:
    """List entity↔source links with prominence (central / major / mentioned).
    Filter by canonical name, source, or prominence.
    Pass auth_token matching the server-side secret.
    """
    denied = require_auth(auth_token)
    if denied:
        return denied
    limit = _clamp_limit(limit, 1, 200)
    db = connect()
    try:
        where = ["1=1"]
        params: list[Any] = []
        if name.strip():
            where.append("e.canonical_name COLLATE NOCASE = ?")
            params.append(name.strip())
        if source.strip():
            where.append("es.source = ?")
            params.append(source.strip())
        if prominence.strip():
            where.append("es.prominence = ?")
            params.append(prominence.strip().lower())
        sql = f"""
            SELECT e.canonical_name AS name, e.id AS entity_id,
                   es.source, es.prominence, es.notes
            FROM entity_sources es
            JOIN entities e ON e.id = es.entity_id
            WHERE {" AND ".join(where)}
            ORDER BY e.canonical_name, es.source
            LIMIT ?
        """
        params.append(limit)
        rows = [_row_to_dict(r) for r in db.execute(sql, params).fetchall()]
        return _ok({"n": len(rows), "links": rows})
    except sqlite3.Error as e:
        return _err(str(e))
    finally:
        db.close()

# entity_sources_set: core-schema tool (source-derived).
@mcp.tool(annotations=ToolAnnotations(readOnlyHint=False))
def entity_sources_set(
    name: str,
    source: str,
    prominence: str = "mentioned",
    notes: str = "",
    auth_token: str = "",
) -> str:
    """Upsert an entity_sources row. Resolves entity_id by canonical_name.
    prominence must be central, major, or mentioned.
    Pass auth_token matching the server-side secret.
    """
    denied = require_auth(auth_token)
    if denied:
        return denied
    canon = (name or "").strip()
    src = (source or "").strip()
    if not canon or not src:
        return _err("name and source are required")
    prom = (prominence or "mentioned").strip().lower()
    if prom not in ALLOWED_PROMINENCE:
        return _err(
            f"invalid prominence '{prom}'",
            allowed=sorted(ALLOWED_PROMINENCE),
        )
    db = connect()
    try:
        ent = db.execute(
            """
            SELECT id, canonical_name FROM entities
            WHERE canonical_name COLLATE NOCASE = ?
            """,
            (canon,),
        ).fetchone()
        if not ent:
            return _err(f"no entity named '{canon}'")
        db.execute(
            """
            INSERT INTO entity_sources (entity_id, source, prominence, notes)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(entity_id, source) DO UPDATE SET
              prominence = excluded.prominence,
              notes = excluded.notes
            """,
            (ent["id"], src, prom, notes.strip() or None),
        )
        db.commit()
        row = db.execute(
            """
            SELECT e.canonical_name AS name, e.id AS entity_id,
                   es.source, es.prominence, es.notes
            FROM entity_sources es
            JOIN entities e ON e.id = es.entity_id
            WHERE es.entity_id = ? AND es.source = ?
            """,
            (ent["id"], src),
        ).fetchone()
        return _ok({"link": _row_to_dict(row)})
    except sqlite3.Error as e:
        return _err(str(e))
    finally:
        db.close()

# get_memory_fact: core-schema tool (source-derived).
@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
def get_memory_fact(fact_id: int, auth_token: str = "") -> str:
    """Fetch one memory fact by numeric ID (read-only diagnostic lookup).

    Executes only a parameterised SELECT; never accepts SQL or modifies data.
    Pass auth_token matching the server-side secret.
    """
    denied = require_auth(auth_token)
    if denied:
        return denied
    db = connect()
    try:
        row = db.execute(
            """
            SELECT id, category, fact, entities, supporting_sources,
                   confidence, first_seen_ord, last_seen_ord,
                   created_at, updated_at
            FROM memory_facts WHERE id = ?
            """,
            (int(fact_id),),
        ).fetchone()
        return _ok({"n": 1 if row else 0,
                    "facts": [_row_to_dict(row)] if row else []})
    except sqlite3.Error as e:
        return _err(str(e))
    finally:
        db.close()

# Core source navigation. Production sources_list additionally reads an
# excluded operational queue; this public version relies only on chunks and
# the retained (optional) source_summaries placeholder.
@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
def sources_list(limit: int=40, auth_token: str="") -> str:
    denied=require_auth(auth_token)
    if denied: return denied
    db=connect()
    try:
        rows=db.execute("SELECT c.source,MAX(c.source_ord) AS source_ord,"
            "COUNT(*) AS chunk_count, s.title, s.short_summary, s.status "
            "FROM chunks c LEFT JOIN source_summaries s "
            "ON s.source=c.source AND s.segment=0 "
            "GROUP BY c.source ORDER BY source_ord DESC LIMIT ?",(limit,)).fetchall()
        return _ok({"sources":[dict(r) for r in rows]})
    finally: db.close()

@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
def stats(auth_token: str="") -> str:
    """Core metrics; peripheral production counters omitted."""
    denied=require_auth(auth_token)
    if denied: return denied
    db=connect()
    try:
        out={}
        for table in ('chunks','chunk_chars','entities','memory_facts',
                      'retrieval_episodes','source_summaries',
                      'retrieval_chunk_vec_sources','chunks_vec','memory_facts_vec'):
            try: out[table]=int(db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
            except sqlite3.Error: out[table]=None
        return _ok({"counts":out})
    finally: db.close()

@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
def entities(limit: int=40, auth_token: str="") -> str:
    """Public variant omits production image BLOB flags and URLs."""
    denied=require_auth(auth_token)
    if denied: return denied
    db=connect()
    try:
        rows=db.execute("SELECT canonical_name,entity_type,description,"
            "first_ord,last_ord,mention_count,importance FROM entities "
            "ORDER BY importance DESC, mention_count DESC LIMIT ?",(limit,)).fetchall()
        return _ok({"entities":[dict(r) for r in rows]})
    finally: db.close()

# Core fact write pathway: data -> memory_facts insert -> automatic fact-FTS
# trigger -> best-effort BGE vector upsert. Public code omits production's
# private name-resolution and category-normalisation policy; do not deploy.
@mcp.tool(annotations=ToolAnnotations(readOnlyHint=False))
def fact_insert(fact: str, category: str, entities: str="",
                supporting_sources: str="", confidence: float=0.8,
                auth_token: str="") -> str:
    denied=require_auth(auth_token)
    if denied: return denied
    db=connect()
    try:
        ts=datetime.now(timezone.utc).isoformat()
        cur=db.execute("INSERT INTO memory_facts "
            "(category,fact,entities,supporting_sources,confidence,created_at,updated_at) "
            "VALUES(?,?,?,?,?,?,?)",(category,fact,entities,supporting_sources,confidence,ts,ts))
        fid=cur.lastrowid
        _facts_vec_upsert(db,fid,fact)
        db.commit()
        return _ok({"inserted_id":fid})
    except sqlite3.Error as exc:
        db.rollback(); return _err(str(exc))
    finally: db.close()


# Core update pathway (adapted): the production implementation additionally
# validates names, category aliases, episode links and provenance before commit.
# SQL fact FTS triggers update automatically; vector refresh is explicit.
@mcp.tool(annotations=ToolAnnotations(readOnlyHint=False))
def memory_facts_update(fact_id: int, fact: str="", category: str="",
                        auth_token: str="") -> str:
    denied=require_auth(auth_token)
    if denied: return denied
    db=connect()
    try:
        row=db.execute("SELECT fact,category FROM memory_facts WHERE id=?",
                       (int(fact_id),)).fetchone()
        if row is None: return _err("unknown fact ID")
        new_fact=fact.strip() if fact.strip() else row['fact']
        new_category=category.strip() if category.strip() else row['category']
        db.execute("UPDATE memory_facts SET fact=?,category=?,updated_at=? WHERE id=?",
                   (new_fact,new_category,datetime.now(timezone.utc).isoformat(),fact_id))
        if new_fact != row['fact']:
            _facts_vec_upsert(db,fact_id,new_fact)
        db.commit()
        return _ok({"updated_id":fact_id})
    except sqlite3.Error as exc:
        db.rollback();return _err(str(exc))
    finally: db.close()

# Episode operations on retrieval_episodes and memory_facts. Production has
# separate list/get/add/attach/propose branches, with explicit add as the
# minting operation; fact writes do not automatically mint an episode.
# Lightweight original raw-FTS search route (production exposes more parsing,
# name collision handling and memory-context assembly).
@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
def search(query: str, n_results: int=6, auth_token: str="") -> str:
    denied=require_auth(auth_token)
    if denied: return denied
    db=connect()
    try:
        terms=re.findall(r"[A-Za-z0-9]{3,}",query)
        fts=' OR '.join('"'+term+'"' for term in terms[:8])
        if not fts: return _ok({"hits":[]})
        rows=db.execute("SELECT c.id,c.source,c.source_ord,c.body "
            "FROM chunks_fts JOIN chunks c ON c.rowid=chunks_fts.rowid "
            "WHERE chunks_fts MATCH ? ORDER BY bm25(chunks_fts) LIMIT ?",
            (fts,n_results)).fetchall()
        return _ok({"hits":[dict(r) for r in rows]})
    finally: db.close()

# now_iso (source-derived)
def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

# _fact_trunc (source-derived)
def _fact_trunc(text: str, limit: int = 480) -> str:
    t = re.sub(r"\s+", " ", (text or "").strip())
    if len(t) > limit:
        return t[:limit].rstrip() + "…"
    return t

# _fact_dedupe_key (source-derived)
def _fact_dedupe_key(fact: str) -> str:
    """First-line key used to collapse duplicate fact records."""
    t = re.sub(r"\s+", " ", (fact or "").strip().lower())
    t = re.sub(r"[*_`#]+", "", t)
    t = re.split(r"\s+---+\s+|\s+—\s+", t, maxsplit=1)[0]
    return t[:96]

# _episode_title_from_fact (source-derived)
def _episode_title_from_fact(fact: str) -> str:
    """First-line title, capped. Used when minting a singleton episode."""
    t = re.sub(r"\s+", " ", (fact or "").strip())
    t = re.sub(r"[*_`#]+", "", t)
    t = re.split(r"\s+---+\s+|\s+—\s+", t, maxsplit=1)[0]
    t = t.strip(" .;:")
    return (t[:80] or "untitled episode")

# _episodes_ready (source-derived)
def _episodes_ready(db: sqlite3.Connection) -> bool:
    try:
        db.execute("SELECT id FROM retrieval_episodes LIMIT 0")
        db.execute("SELECT episode_id FROM memory_facts LIMIT 0")
        return True
    except sqlite3.Error:
        return False

# _lookup_episode (source-derived)
def _lookup_episode(db: sqlite3.Connection, episode_id: int) -> dict[str, Any] | None:
    try:
        row = db.execute(
            "SELECT id, title, stream, event_date, canonical_fact_id, status, notes "
            "FROM retrieval_episodes WHERE id = ?",
            (int(episode_id),),
        ).fetchone()
    except sqlite3.Error:
        return None
    return dict(row) if row else None


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=False))
def reserved_b(payload: str="", auth_token: str="") -> str:
    """Episode operations on core schema: list, get, add, attach, propose.

    Duplicate-title add returns existing episode. Attach does not remint.
    Production-only personal examples were removed from this docstring.
    """
    denied = require_auth(auth_token)
    if denied:
        return denied
    raw = (payload or "").strip()
    op = "list"
    args: dict[str, Any] = {}
    if raw.startswith("{") or raw.startswith("["):
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            return _err("payload is not JSON")
        if isinstance(obj, dict):
            args = obj
            op = str(obj.get("op") or "list").strip().lower()
        else:
            return _err("payload must be an object")
    elif raw:
        args = {"query": raw}
        op = "list"
    db = connect()
    try:
        if not _episodes_ready(db):
            return _err(
                "episodes table missing",
                hint="Episode tables are required.",
            )
        if op == "list":
            q = str(args.get("query") or "").strip()
            limit = _clamp_limit(int(args.get("limit") or 20), 1, 80)
            sql = (
                "SELECT e.id, e.title, e.stream, e.event_date, "
                "e.canonical_fact_id, e.status, e.notes, "
                "(SELECT COUNT(*) FROM memory_facts f "
                " WHERE f.episode_id = e.id) AS n_facts "
                "FROM retrieval_episodes e WHERE 1=1"
            )
            params: list[Any] = []
            if q:
                sql += " AND (e.title LIKE ? OR ifnull(e.notes,'') LIKE ?)"
                params.extend([f"%{q}%", f"%{q}%"])
            sql += " ORDER BY e.id DESC LIMIT ?"
            params.append(limit)
            rows = [dict(r) for r in db.execute(sql, params).fetchall()]
            return _ok({"n": len(rows), "episodes": rows})
        if op == "get":
            eid = args.get("episode_id") or args.get("id")
            title = str(args.get("title") or "").strip()
            row = None
            if eid not in (None, ""):
                try:
                    row = _lookup_episode(db, int(eid))
                except (TypeError, ValueError):
                    return _err("episode_id must be an integer")
            elif title:
                found = db.execute(
                    "SELECT id, title, stream, event_date, "
                    "canonical_fact_id, status, notes "
                    "FROM retrieval_episodes WHERE title = ?",
                    (title,),
                ).fetchone()
                row = dict(found) if found else None
            else:
                return _err("episode_id or title is required")
            if not row:
                return _err("episode not found")
            facts = [
                {
                    "id": int(f["id"]),
                    "category": f["category"],
                    "fact": _fact_trunc(f["fact"] or "", 280),
                    "canonical": (
                        row.get("canonical_fact_id") is not None
                        and int(f["id"]) == int(row["canonical_fact_id"])
                    ),
                }
                for f in db.execute(
                    "SELECT id, category, fact FROM memory_facts "
                    "WHERE episode_id = ? ORDER BY id",
                    (int(row["id"]),),
                ).fetchall()
            ]
            out = dict(row)
            out["n_facts"] = len(facts)
            out["facts"] = facts
            return _ok({"episode": out})
        if op == "add":
            title = str(args.get("title") or "").strip()
            if not title:
                return _err("title is required")
            stream = str(args.get("stream") or "mixed").strip().lower() or "mixed"
            event_date = str(args.get("event_date") or "").strip() or None
            ents = args.get("entities") or []
            if isinstance(ents, str):
                ents = _parse_name_list(ents)
            can = args.get("canonical_fact_id")
            try:
                can_i = int(can) if can not in (None, "") else None
            except (TypeError, ValueError):
                return _err("canonical_fact_id must be an integer")
            ts = now_iso()
            existed = False
            try:
                cur = db.execute(
                    """
                    INSERT INTO retrieval_episodes (
                      title, stream, event_date, entities, canonical_fact_id,
                      status, notes, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, 'live', ?, ?, ?)
                    """,
                    (
                        title,
                        stream,
                        event_date,
                        json.dumps(ents, ensure_ascii=False),
                        can_i,
                        str(args.get("notes") or ""),
                        ts,
                        ts,
                    ),
                )
                eid = int(cur.lastrowid)
            except sqlite3.IntegrityError:
                found = db.execute(
                    "SELECT id FROM retrieval_episodes WHERE title = ?",
                    (title,),
                ).fetchone()
                if not found:
                    return _err("title already exists but could not be loaded")
                eid = int(found["id"])
                existed = True
            attached = []
            for fid in args.get("fact_ids") or []:
                try:
                    i = int(fid)
                except (TypeError, ValueError):
                    continue
                db.execute(
                    "UPDATE memory_facts SET episode_id = ? WHERE id = ?",
                    (eid, i),
                )
                attached.append(i)
            if can_i and can_i not in attached:
                db.execute(
                    "UPDATE memory_facts SET episode_id = ? WHERE id = ?",
                    (eid, can_i),
                )
                attached.append(can_i)
            db.commit()
            return _ok({
                "episode_id": eid,
                "attached": attached,
                "existed": existed,
            })
        if op == "attach":
            try:
                eid = int(args.get("episode_id"))
            except (TypeError, ValueError):
                return _err("episode_id is required")
            if not _lookup_episode(db, eid):
                return _err(f"no episode {eid}")
            attached = []
            for fid in args.get("fact_ids") or []:
                try:
                    i = int(fid)
                except (TypeError, ValueError):
                    continue
                db.execute(
                    "UPDATE memory_facts SET episode_id = ? WHERE id = ?",
                    (eid, i),
                )
                attached.append(i)
            db.execute(
                "UPDATE retrieval_episodes SET updated_at = ? WHERE id = ?",
                (now_iso(), eid),
            )
            db.commit()
            return _ok({"episode_id": eid, "attached": attached})
        if op == "propose":
            limit = _clamp_limit(int(args.get("limit") or 12), 1, 40)
            rows = db.execute(
                "SELECT id, fact, episode_id, category, entities FROM memory_facts "
                "WHERE confidence > 0 ORDER BY id"
            ).fetchall()
            buckets: dict[str, list[int]] = {}
            cats: dict[str, set[str]] = {}
            for r in rows:
                if r["episode_id"]:
                    continue
                key = _fact_dedupe_key(r["fact"] or "")
                if not key:
                    continue
                buckets.setdefault(key, []).append(int(r["id"]))
                cats.setdefault(key, set()).add((r["category"] or "other").lower())
            proposed = []
            for key, ids in sorted(buckets.items(), key=lambda kv: -len(kv[1])):
                if len(ids) < 2:
                    continue
                key_cats = cats.get(key) or set()
                if "page-layer" in key_cats and (key_cats - {"page-layer"}):
                    continue
                proposed.append({
                    "title": _episode_title_from_fact(
                        next(
                            (r["fact"] for r in rows if int(r["id"]) == ids[0]),
                            key,
                        )
                    ),
                    "fact_ids": ids[:12],
                    "n": len(ids),
                    "categories": sorted(key_cats),
                })
                if len(proposed) >= limit:
                    break
            return _ok({"n": len(proposed), "candidates": proposed})
        return _err(f"unknown op '{op}'", hint="list | get | add | attach | propose")
    except sqlite3.Error as e:
        return _err(str(e))
    finally:
        db.close()

# Production reserved_a is a separate, fact-first relevance clerk. It uses the
# core fact FTS/vector indexes, stream cues, pins and episode grouping; it also
# has non-core response branches deliberately NOT copied into this version.
# The complete core Wilson v3.4.3 retrieval implementation is _v3_retrieve.
@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
def reserved_c(payload: str="", auth_token: str="") -> str:
    denied=require_auth(auth_token)
    if denied: return denied
    try: return _dumps(_v3_retrieve(_v2_parse_payload(payload)))
    except (ValueError,sqlite3.Error) as exc: return _err(str(exc))

# No enrichment, private image routes, custom watch, peripheral stores or
# unrestricted SQL-administration tool are included here.

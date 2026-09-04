#!/usr/bin/env python3
"""coop — multi-instance work history + learning DB (CWK coop pattern, 2026-07-17).

Every AI instance that works on BJT's codebases (Claude Code, Codex CLI,
a human in a terminal) records what it did (`log`), what it learned
(`learn`), and can leave handoff notes (`note`) and todos (`todo`) for the
next session. Stateless models forget; the coop is the shared, durable,
retrievable record that lets them grow across sessions — CWK: models must
retrieve past learnings when they hit a problem, or there is no growth loop.

Agents may also address generic mailbox messages to one another.  Coop owns
delivery, threads, artifact references and claim state; the agents own the
judgment about whether, when and how to collaborate.

Ground truth lives on ONE machine (the always-on server) at
$HOME/forrest-db/coop.db — deliberately outside the Dropbox CloudStorage
tree (live SQLite + cloud sync don't mix; CLAUDE.md Dropbox caution).
Satellite machines reach it over SSH; they do not carry their own copy.

`brief` prints the session-start card: which machine this is, whether it is
the ground-truth server or a satellite, git sync state (dirty / unpushed /
no-origin), open notes+todos, and the freshest entries. Wired as a Claude
Code SessionStart hook (.claude/settings.json) so every terminal session
re-grounds automatically; Codex sessions get it via AGENTS.md. The Forrest
SDK sessions run with setting_sources=[] — the SDK's documented isolation
mode (CLAUDE.md rule 9) — so this terminal hook can never leak into WebUI
turns. Separately, the backend exposes
the same storage API to Admin and app-Forrest body-work turns; that is an
explicit coop surface, not filesystem-setting inheritance.

Stdlib only (sqlite3/argparse), runs on macOS system python3. Short-lived
CLI processes over one WAL file — concurrent writers are safe via
busy_timeout, no daemon, no server.

Env: COOP_DB_PATH (default $HOME/forrest-db/coop.db, override for tests),
     COOP_GROUND_TRUTH_HOST (default the declared server below),
     COOP_AGENT (default agent id; CLAUDECODE=1 implies claude-code).
"""

import argparse
from contextlib import contextmanager
import datetime as dt
import fcntl
import json
import os
import re
import socket
import sqlite3
import subprocess
import sys
import uuid

GROUND_TRUTH_HOST = "Jeongtaeks-Mac-Studio"  # BJT's Mac Studio — the always-on server
DEFAULT_DB = os.path.join(os.path.expanduser("~"), "forrest-db", "coop.db")
KINDS = ("history", "learning", "note", "todo")
STATUSES = ("open", "read", "done", "superseded")
REVIEW_STATUSES = (
    "open", "claimed", "changes_requested", "approved", "superseded",
)
REVIEW_VERDICTS = ("approved", "changes_requested")
MESSAGE_STATUSES = ("open", "claimed", "done", "superseded")
COOP_SCHEMA_VERSION = 2

# ---------------------------------------------------------------- storage

FTS_OK = False  # set by connect(); sqlite3.Connection forbids ad-hoc attributes


@contextmanager
def _managed_work_result_guard(
    intent,
    *,
    target_message_id=None,
    fallback_message_id=None,
):
    """Fence a managed result against a concurrent generation rotation."""
    if intent != "work_result" or os.environ.get("FORREST_WORKSPACE") != "1":
        yield
        return
    runtime_raw = os.environ.get("FORREST_WORK_RUNTIME_PATH", "").strip()
    token = os.environ.get("FORREST_WORK_SESSION_TOKEN", "").strip()
    generation = os.environ.get("FORREST_WORK_GENERATION", "").strip()
    if not runtime_raw or not token or not generation.isdigit():
        raise ValueError("managed work_result is missing its runtime fence")
    lock_path = runtime_raw + ".lock"
    with open(lock_path, "a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            with open(runtime_raw, encoding="utf-8") as runtime_handle:
                descriptor = json.load(runtime_handle)
            if (
                not isinstance(descriptor, dict)
                or descriptor.get("session_token") != token
                or int(descriptor.get("generation") or 0) != int(generation)
                or descriptor.get("cancel_requested")
                or descriptor.get("generation_sealed")
                or descriptor.get("containment") != "codex_seatbelt_generation"
            ):
                raise ValueError(
                    "managed work_result belongs to an expired Forrest generation",
                )
            expected_raw = (
                descriptor.get("completion_message_id")
                or descriptor.get("message_id")
                or fallback_message_id
            )
            try:
                expected_message_id = int(expected_raw or 0)
            except (TypeError, ValueError):
                expected_message_id = 0
            if (
                target_message_id is not None
                and expected_message_id > 0
                and int(target_message_id) != expected_message_id
            ):
                raise ValueError(
                    "managed work_result targets a different active Forrest request",
                )
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

_TABLE_SCHEMA = """
CREATE TABLE IF NOT EXISTS entries (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  kind TEXT NOT NULL CHECK (kind IN ('history','learning','note','todo')),
  project TEXT NOT NULL,
  agent TEXT NOT NULL,
  machine TEXT NOT NULL,
  title TEXT NOT NULL,
  body TEXT NOT NULL DEFAULT '',
  slug TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT '',
  commit_sha TEXT NOT NULL DEFAULT '',
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  project TEXT NOT NULL,
  thread_id TEXT NOT NULL,
  reply_to INTEGER REFERENCES messages(id),
  intent TEXT NOT NULL,
  sender_agent TEXT NOT NULL,
  recipient_agent TEXT NOT NULL,
  claimed_by TEXT NOT NULL DEFAULT '',
  machine TEXT NOT NULL,
  subject TEXT NOT NULL,
  body TEXT NOT NULL DEFAULT '',
  artifact_refs TEXT NOT NULL DEFAULT '[]',
  status TEXT NOT NULL DEFAULT 'open'
    CHECK (status IN ('open','claimed','done','superseded')),
  source_key TEXT NOT NULL DEFAULT '',
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS reviews (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  project TEXT NOT NULL,
  requester_agent TEXT NOT NULL,
  reviewer_agent TEXT NOT NULL,
  reviewer_model TEXT NOT NULL DEFAULT '',
  claimed_by TEXT NOT NULL DEFAULT '',
  machine TEXT NOT NULL,
  title TEXT NOT NULL,
  body TEXT NOT NULL DEFAULT '',
  base_sha TEXT NOT NULL DEFAULT '',
  head_sha TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'open'
    CHECK (status IN ('open','claimed','changes_requested','approved','superseded')),
  last_error TEXT NOT NULL DEFAULT '',
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS review_rounds (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  review_id INTEGER NOT NULL REFERENCES reviews(id) ON DELETE CASCADE,
  reviewer_agent TEXT NOT NULL,
  reviewer_model TEXT NOT NULL DEFAULT '',
  head_sha TEXT NOT NULL,
  verdict TEXT NOT NULL
    CHECK (verdict IN ('approved','changes_requested')),
  body TEXT NOT NULL DEFAULT '',
  created_at TEXT NOT NULL
);
"""

_INDEX_SCHEMA = """
CREATE INDEX IF NOT EXISTS idx_entries_kind_created ON entries(kind, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_entries_project ON entries(project);
CREATE INDEX IF NOT EXISTS idx_messages_recipient_status
  ON messages(recipient_agent, status, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_messages_project_thread
  ON messages(project, thread_id, id);
CREATE UNIQUE INDEX IF NOT EXISTS idx_messages_source_key
  ON messages(source_key) WHERE source_key != '';
CREATE INDEX IF NOT EXISTS idx_reviews_project_created
  ON reviews(project, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_reviews_reviewer_status
  ON reviews(reviewer_agent, status, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_review_rounds_review
  ON review_rounds(review_id, id);
"""

_FTS_SCHEMA = """
CREATE VIRTUAL TABLE IF NOT EXISTS entries_fts
  USING fts5(title, body, content='entries', content_rowid='id');
CREATE TRIGGER IF NOT EXISTS entries_ai AFTER INSERT ON entries BEGIN
  INSERT INTO entries_fts(rowid, title, body) VALUES (new.id, new.title, new.body);
END;
CREATE TRIGGER IF NOT EXISTS entries_ad AFTER DELETE ON entries BEGIN
  INSERT INTO entries_fts(entries_fts, rowid, title, body)
    VALUES ('delete', old.id, old.title, old.body);
END;
CREATE TRIGGER IF NOT EXISTS entries_au AFTER UPDATE ON entries BEGIN
  INSERT INTO entries_fts(entries_fts, rowid, title, body)
    VALUES ('delete', old.id, old.title, old.body);
  INSERT INTO entries_fts(rowid, title, body) VALUES (new.id, new.title, new.body);
END;
"""


def _table_columns(conn, table):
    return {row[1] for row in conn.execute("PRAGMA table_info(%s)" % table)}


def _migrate_schema(conn):
    """Advance the central ledger before indexes or writers use new columns.

    The DB predates explicit schema versions, so version 0 can mean either a
    fresh database or any historical versionless shape.  Each migration must
    therefore be introspection-safe and idempotent; only the server-owned
    canonical client grows this list. Older vendored clients may still write
    during a fleet rollout, so additive NOT NULL columns must always provide a
    DEFAULT and every migration must preserve explicit legacy INSERT lists.
    """
    observed = conn.execute("PRAGMA user_version").fetchone()[0]
    if observed > COOP_SCHEMA_VERSION:
        raise RuntimeError(
            "coop DB schema %d is newer than this client supports (%d)"
            % (observed, COOP_SCHEMA_VERSION)
        )
    if observed == COOP_SCHEMA_VERSION:
        return

    conn.execute("BEGIN IMMEDIATE")
    try:
        # Another short-lived CLI may have completed the migration while this
        # connection waited for the write lock. Re-read under ownership.
        current = conn.execute("PRAGMA user_version").fetchone()[0]
        if current > COOP_SCHEMA_VERSION:
            raise RuntimeError(
                "coop DB schema %d is newer than this client supports (%d)"
                % (current, COOP_SCHEMA_VERSION)
            )
        if current < 1:
            # ``source_key`` is required by the legacy-review projection and
            # its unique index. CREATE TABLE IF NOT EXISTS cannot add it to a
            # versionless pre-existing messages table.
            if "source_key" not in _table_columns(conn, "messages"):
                conn.execute(
                    "ALTER TABLE messages ADD COLUMN "
                    "source_key TEXT NOT NULL DEFAULT ''"
                )
            conn.execute("PRAGMA user_version=1")
        if current < 2:
            # Memory L6 realignment (2026-08-16): the history entry is
            # written BEFORE its commit and the head is backfilled after —
            # so the sha needs a typed home instead of free-text prose.
            if "commit_sha" not in _table_columns(conn, "entries"):
                conn.execute(
                    "ALTER TABLE entries ADD COLUMN "
                    "commit_sha TEXT NOT NULL DEFAULT ''"
                )
            conn.execute("PRAGMA user_version=2")
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def connect(path=None):
    global FTS_OK
    db_path = path or os.environ.get("COOP_DB_PATH") or DEFAULT_DB
    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=5000")
    try:
        # Tables must exist before a versionless DB can be introspected, while
        # indexes must wait until migrations have supplied every referenced
        # column.
        conn.executescript(_TABLE_SCHEMA)
        _migrate_schema(conn)
        conn.executescript(_INDEX_SCHEMA)
    except Exception:
        conn.close()
        raise
    try:  # FTS5 ships with macOS/homebrew sqlite; degrade to LIKE search without it
        conn.executescript(_FTS_SCHEMA)
        FTS_OK = True
    except sqlite3.OperationalError:
        FTS_OK = False
    _migrate_legacy_reviews(conn)
    return conn


# ---------------------------------------------------------------- context


def detect_machine():
    return os.environ.get("COOP_MACHINE") or socket.gethostname().split(".")[0]


def detect_agent(cli_value=None):
    if cli_value:
        return cli_value
    if os.environ.get("COOP_AGENT"):
        return os.environ["COOP_AGENT"]
    if os.environ.get("CLAUDECODE"):
        return "claude-code"
    return "human"


def detect_project(cli_value=None, cwd=None):
    if cli_value:
        return cli_value
    try:
        top = subprocess.run(
            ["git", "-C", cwd or os.getcwd(), "rev-parse", "--show-toplevel"],
            capture_output=True, text=True, timeout=10,
        )
        if top.returncode == 0 and top.stdout.strip():
            return os.path.basename(top.stdout.strip())
    except Exception:
        pass
    return os.path.basename(cwd or os.getcwd())


def now_iso():
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def make_slug(title, created):
    stem = re.sub(r"[^0-9A-Za-z가-힣]+", "-", title.lower()).strip("-")[:48].strip("-")
    return "%s-%s" % (created[:10], stem or "entry")


def read_body(value):
    if value == "-":
        return sys.stdin.read().strip()
    return value or ""


# ---------------------------------------------------------------- library
# Every SQL statement lives in this storage API.  The CLI and backend route
# both call these functions (backend/app/routes/coop.py loads this module by
# path), so schema, filtering, ranking and defaults stay single-source.


def add_entry(conn, kind, title, body="", project=None, agent=None,
              machine=None, slug=None, commit_sha=None):
    """Insert one entry; returns (id, slug).

    Memory L6 ordering: a history/learning entry is written BEFORE its git
    commit (so the *why* survives a crash between commit and record), then
    the head is backfilled via ``set_commit_sha``. ``commit_sha`` here is for
    replay-safe machine writers and retroactive records that already know
    their head.
    """
    if kind not in KINDS:
        raise ValueError("kind must be one of: %s" % ", ".join(KINDS))
    created = now_iso()
    status = "open" if kind in ("note", "todo") else ""
    final_slug = slug or make_slug(title, created)
    cur = conn.execute(
        "INSERT INTO entries (kind, project, agent, machine, title, body, slug, status,"
        " commit_sha, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        (kind, project or detect_project(), agent or detect_agent(),
         machine or detect_machine(), title, body or "", final_slug, status,
         commit_sha or "", created, created),
    )
    conn.commit()
    return cur.lastrowid, final_slug


def set_commit_sha(conn, entry_id, commit_sha):
    """Backfill the git head onto a history/learning entry after the commit.

    Returns True if a row changed. Notes/todos have no commit identity and
    are refused so a typo'd id cannot silently decorate the wrong kind.
    """
    sha = (commit_sha or "").strip()
    if not sha:
        raise ValueError("commit_sha must be non-empty")
    cur = conn.execute(
        "UPDATE entries SET commit_sha=?, updated_at=? "
        "WHERE id=? AND kind IN ('history','learning')",
        (sha, now_iso(), entry_id),
    )
    conn.commit()
    return cur.rowcount > 0


def set_status(conn, entry_id, new_status):
    """Set a note/todo status; returns True if a row changed."""
    if new_status not in STATUSES:
        raise ValueError("status must be one of: %s" % ", ".join(STATUSES))
    cur = conn.execute(
        "UPDATE entries SET status=?, updated_at=? WHERE id=? AND kind IN ('note','todo')",
        (new_status, now_iso(), entry_id),
    )
    conn.commit()
    return cur.rowcount > 0


def get_entry(conn, entry_id):
    """Return one entry row, or None."""
    return conn.execute(
        "SELECT * FROM entries WHERE id=?", (entry_id,),
    ).fetchone()


def get_entry_by_slug(conn, project, slug):
    """Return one project-scoped entry with ``slug``, or None.

    Bootstrap-style callers need a replay-safe way to record a finished unit:
    a process can die after SQLite commits the Coop row but before its own
    state machine checkpoints success.  Looking up the deterministic slug on
    replay avoids publishing the same history entry twice while keeping all
    SQL inside this shared storage module.
    """
    return conn.execute(
        "SELECT * FROM entries WHERE project=? AND slug=? ORDER BY id LIMIT 1",
        (project, slug),
    ).fetchone()


def _entry_filters(kind=None, project=None, status=None, alias=""):
    prefix = (alias + ".") if alias else ""
    where, params = [], []
    for column, value in (("kind", kind), ("project", project),
                          ("status", status)):
        if value is not None:
            where.append("%s%s=?" % (prefix, column))
            params.append(value)
    return where, params


def search_rows(conn, query, kind=None, limit=10, *, project=None, status=None):
    """Search with every structured filter applied *before* LIMIT.

    The old HTTP route filtered project/status in Python after this function
    had already clipped the global result set.  A matching project could then
    look empty merely because newer hits from another project occupied the
    limit.  Keeping the complete predicate here fixes that correctness bug and
    preserves one SQL owner.
    """
    filters, filter_params = _entry_filters(
        kind, project, status, alias="e",
    )
    structured_sql = (" AND " + " AND ".join(filters)) if filters else ""
    if FTS_OK:
        fts_query = " ".join('"%s"' % t.replace('"', "") for t in query.split())
        try:
            rows = conn.execute(
                "SELECT e.* FROM entries_fts f JOIN entries e ON e.id=f.rowid"
                " WHERE entries_fts MATCH ?" + structured_sql
                + " ORDER BY rank LIMIT ?",
                [fts_query, *filter_params, limit],
            ).fetchall()
            if rows:
                return rows
        except sqlite3.OperationalError:
            pass
    like = "%" + query + "%"
    return conn.execute(
        "SELECT e.* FROM entries e WHERE (e.title LIKE ? OR e.body LIKE ?)"
        + structured_sql + " ORDER BY e.id DESC LIMIT ?",
        [like, like, *filter_params, limit],
    ).fetchall()


def list_entries(conn, *, query=None, kind=None, project=None, status=None,
                 limit=100):
    """Return newest/ranked entries under one provider-neutral filter API."""
    if query:
        return search_rows(
            conn, query, kind, limit, project=project, status=status,
        )
    where, params = _entry_filters(kind, project, status)
    sql = "SELECT * FROM entries"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY id DESC LIMIT ?"
    return conn.execute(sql, [*params, limit]).fetchall()


def entry_counts(conn):
    """Return dashboard counts without exposing SQL to presentation layers."""
    counts = {kind: 0 for kind in KINDS}
    for row in conn.execute(
        "SELECT kind, COUNT(*) AS n FROM entries GROUP BY kind",
    ):
        counts[row["kind"]] = row["n"]
    counts["total"] = sum(counts.values())
    counts["open"] = conn.execute(
        "SELECT COUNT(*) FROM entries WHERE status='open'",
    ).fetchone()[0]
    return counts


def list_projects(conn):
    """Return distinct project names in stable display order."""
    return [row[0] for row in conn.execute(
        "SELECT project FROM ("
        " SELECT project FROM entries"
        " UNION SELECT project FROM messages"
        " UNION SELECT project FROM reviews"
        ") ORDER BY project",
    ).fetchall()]


# ---------------------------------------------------------------- mailbox
# The mailbox is the deterministic substrate for agent collaboration.  It
# guarantees delivery, addressing, threading, exact artifact references and
# claim ownership; it deliberately does NOT decide when an agent should ask a
# peer, whom it should ask, or how it should use the reply.


def _normalize_intent(intent):
    value = (intent or "request").strip().lower()
    if not re.fullmatch(r"[a-z][a-z0-9_.-]{0,63}", value):
        raise ValueError(
            "intent must match [a-z][a-z0-9_.-]{0,63}",
        )
    return value


def _normalize_artifact_refs(artifact_refs):
    refs = []
    seen = set()
    for raw in artifact_refs or []:
        value = str(raw).strip()
        if not value or value in seen:
            continue
        if len(value) > 2000:
            raise ValueError("artifact ref must be at most 2000 characters")
        refs.append(value)
        seen.add(value)
    return refs


def _insert_message(conn, subject, *, recipient_agent, sender_agent,
                    intent="request", body="", artifact_refs=None,
                    project=None, machine=None, thread_id=None, reply_to=None,
                    status="open", claimed_by="", source_key="",
                    created_at=None, updated_at=None, ignore_conflict=False):
    if not subject.strip():
        raise ValueError("message subject must not be empty")
    if not recipient_agent.strip() or not sender_agent.strip():
        raise ValueError("message sender and recipient must not be empty")
    if status not in MESSAGE_STATUSES:
        raise ValueError(
            "message status must be one of: %s" % ", ".join(MESSAGE_STATUSES),
        )
    created = created_at or now_iso()
    updated = updated_at or created
    final_thread = thread_id or ("coop-" + uuid.uuid4().hex)
    refs_json = json.dumps(
        _normalize_artifact_refs(artifact_refs), ensure_ascii=False,
        separators=(",", ":"),
    )
    cur = conn.execute(
        ("INSERT OR IGNORE" if ignore_conflict else "INSERT")
        + " INTO messages (project, thread_id, reply_to, intent,"
        " sender_agent, recipient_agent, claimed_by, machine, subject, body,"
        " artifact_refs, status, source_key, created_at, updated_at)"
        " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            project or detect_project(), final_thread, reply_to,
            _normalize_intent(intent), sender_agent, recipient_agent,
            claimed_by, machine or detect_machine(), subject.strip(), body or "",
            refs_json, status, source_key or "", created, updated,
        ),
    )
    return cur.lastrowid, final_thread


def add_message(conn, subject, *, recipient_agent, sender_agent=None,
                intent="request", body="", artifact_refs=None, project=None,
                machine=None, thread_id=None, reply_to=None):
    """Send one durable message; returns ``(id, thread_id)``."""
    result = _insert_message(
        conn, subject,
        recipient_agent=recipient_agent,
        sender_agent=sender_agent or detect_agent(),
        intent=intent,
        body=body,
        artifact_refs=artifact_refs,
        project=project,
        machine=machine,
        thread_id=thread_id,
        reply_to=reply_to,
    )
    conn.commit()
    return result


def add_message_once(conn, subject, *, source_key, recipient_agent,
                     sender_agent=None, intent="request", body="",
                     artifact_refs=None, project=None, machine=None,
                     thread_id=None, reply_to=None):
    """Idempotently append one source-keyed message and return ``(row, new)``.

    This is the generic mailbox boundary for crash-recovered producers.  It
    never reopens or rewrites an older message: a correction request is a new
    addressable row in the same thread, while the earlier result remains
    immutable collaboration history.
    """
    key = str(source_key or "").strip()
    if not key:
        raise ValueError("source_key must not be empty")
    conn.execute("BEGIN IMMEDIATE")
    try:
        existing = conn.execute(
            "SELECT * FROM messages WHERE source_key=?",
            (key,),
        ).fetchone()
        if existing is not None:
            conn.commit()
            return existing, False
        message_id, _thread = _insert_message(
            conn,
            subject,
            recipient_agent=recipient_agent,
            sender_agent=sender_agent or detect_agent(),
            intent=intent,
            body=body,
            artifact_refs=artifact_refs,
            project=project,
            machine=machine,
            thread_id=thread_id,
            reply_to=reply_to,
            source_key=key,
        )
        conn.commit()
        return get_message(conn, message_id), True
    except Exception:
        conn.rollback()
        raise


def get_message(conn, message_id):
    return conn.execute(
        "SELECT * FROM messages WHERE id=?", (message_id,),
    ).fetchone()


def message_artifact_refs(row):
    try:
        value = json.loads(row["artifact_refs"] or "[]")
    except (TypeError, json.JSONDecodeError):
        return []
    return [str(item) for item in value] if isinstance(value, list) else []


def list_messages(conn, *, query=None, project=None, status=None,
                  recipient_agent=None, sender_agent=None, thread_id=None,
                  limit=100):
    where, params = [], []
    for column, value in (
        ("project", project), ("status", status),
        ("recipient_agent", recipient_agent), ("sender_agent", sender_agent),
        ("thread_id", thread_id),
    ):
        if value is not None:
            where.append("%s=?" % column)
            params.append(value)
    if query:
        where.append(
            "(subject LIKE ? OR body LIKE ? OR intent LIKE ? OR artifact_refs LIKE ?)",
        )
        like = "%" + query + "%"
        params.extend((like, like, like, like))
    sql = "SELECT * FROM messages"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY id DESC LIMIT ?"
    return conn.execute(sql, [*params, limit]).fetchall()


def list_active_root_messages(conn, *, project, intents):
    """Return every open/claimed root for an exact project + intent set.

    This is deliberately separate from the human-facing recent-message window:
    admission guards must not forget an old claimed unit merely because newer
    mailbox traffic pushed it past a display LIMIT.
    """
    normalized = tuple(_normalize_intent(intent) for intent in intents)
    if not normalized:
        return []
    placeholders = ",".join("?" for _ in normalized)
    return conn.execute(
        "SELECT * FROM messages"
        " WHERE project=? AND reply_to IS NULL"
        " AND status IN ('open','claimed')"
        " AND intent IN (" + placeholders + ")"
        " ORDER BY id DESC",
        (project, *normalized),
    ).fetchall()


def list_thread_messages(conn, thread_id):
    """Return one thread's full transcript, oldest first.

    WorkUnit projections (project_work / project_workflow) fold lifecycle
    truth from the complete thread; the SQL shape lives here so the coop
    schema keeps one owner (Memory L6).
    """
    return conn.execute(
        "SELECT * FROM messages WHERE thread_id=? ORDER BY id",
        (thread_id,),
    ).fetchall()


def list_root_messages_with_artifact(conn, *, project, artifact_ref):
    """Return roots whose artifact_refs may contain one exact ref, newest first.

    LIKE is a candidate prefilter only — callers must re-check membership
    with `message_artifact_refs` (a wildcard match is not exact identity).
    """
    return conn.execute(
        "SELECT * FROM messages WHERE project=? AND reply_to IS NULL "
        "AND artifact_refs LIKE ? ORDER BY id DESC",
        (project, "%" + artifact_ref + "%"),
    ).fetchall()


def project_message_high_water(conn):
    """Return {project: max message id} — the outbox sweep's cheap cursor map."""
    rows = conn.execute(
        "SELECT project, MAX(id) AS max_id FROM messages GROUP BY project",
    ).fetchall()
    return {str(row["project"]): int(row["max_id"] or 0) for row in rows}


def message_counts(conn, *, project=None, recipient_agent=None):
    where, params = [], []
    if project is not None:
        where.append("project=?")
        params.append(project)
    if recipient_agent is not None:
        where.append("recipient_agent=?")
        params.append(recipient_agent)
    clause = (" WHERE " + " AND ".join(where)) if where else ""
    counts = {status: 0 for status in MESSAGE_STATUSES}
    for row in conn.execute(
        "SELECT status, COUNT(*) AS n FROM messages" + clause
        + " GROUP BY status",
        params,
    ):
        counts[row["status"]] = row["n"]
    counts["total"] = sum(counts.values())
    counts["attention"] = counts["open"] + counts["claimed"]
    return counts


def claim_message(conn, message_id, agent):
    """Atomically claim one open message addressed to ``agent``."""
    conn.execute("BEGIN IMMEDIATE")
    try:
        row = get_message(conn, message_id)
        if row is None:
            raise ValueError("no message with id %d" % message_id)
        if row["recipient_agent"] != agent:
            raise ValueError(
                "message #%d is addressed to %s, not %s" % (
                    message_id, row["recipient_agent"], agent,
                )
            )
        if row["status"] != "open":
            raise ValueError(
                "message #%d is %s, not open" % (message_id, row["status"]),
            )
        cur = conn.execute(
            "UPDATE messages SET status='claimed', claimed_by=?, updated_at=?"
            " WHERE id=? AND status='open'",
            (agent, now_iso(), message_id),
        )
        if cur.rowcount != 1:  # pragma: no cover - BEGIN IMMEDIATE serializes
            raise ValueError("message #%d was claimed concurrently" % message_id)
        conn.commit()
        return get_message(conn, message_id)
    except Exception:
        conn.rollback()
        raise


def release_message(conn, message_id, agent):
    cur = conn.execute(
        "UPDATE messages SET status='open', claimed_by='', updated_at=?"
        " WHERE id=? AND status='claimed' AND claimed_by=?",
        (now_iso(), message_id, agent),
    )
    conn.commit()
    return cur.rowcount == 1


def finish_message(conn, message_id, agent, status="done"):
    if status not in ("done", "superseded"):
        raise ValueError("finished message status must be done or superseded")
    conn.execute("BEGIN IMMEDIATE")
    try:
        row = get_message(conn, message_id)
        if row is None:
            raise ValueError("no message with id %d" % message_id)
        if agent not in (row["sender_agent"], row["recipient_agent"]):
            raise ValueError(
                "agent %s is not part of message #%d" % (agent, message_id),
            )
        if row["status"] not in ("open", "claimed"):
            raise ValueError(
                "message #%d is already %s" % (message_id, row["status"]),
            )
        cur = conn.execute(
            "UPDATE messages SET status=?, claimed_by='', updated_at=?"
            " WHERE id=? AND status IN ('open','claimed')",
            (status, now_iso(), message_id),
        )
        if cur.rowcount != 1:  # pragma: no cover - BEGIN IMMEDIATE serializes
            raise ValueError("message #%d changed concurrently" % message_id)
        conn.commit()
        return True
    except Exception:
        conn.rollback()
        raise


def reply_message(conn, message_id, *, sender_agent, body, subject=None,
                  intent="response", artifact_refs=None):
    """Reply in-thread and close the addressed parent atomically."""
    conn.execute("BEGIN IMMEDIATE")
    try:
        parent = get_message(conn, message_id)
        if parent is None:
            raise ValueError("no message with id %d" % message_id)
        if parent["recipient_agent"] != sender_agent:
            raise ValueError(
                "message #%d is addressed to %s, not %s" % (
                    message_id, parent["recipient_agent"], sender_agent,
                )
            )
        if parent["status"] not in ("open", "claimed"):
            raise ValueError(
                "message #%d is %s, not replyable" % (
                    message_id, parent["status"],
                )
            )
        if parent["status"] == "claimed" and parent["claimed_by"] != sender_agent:
            raise ValueError(
                "message #%d is claimed by %s" % (
                    message_id, parent["claimed_by"],
                )
            )
        reply_subject = subject or ("Re: " + parent["subject"])
        reply_id, thread_id = _insert_message(
            conn, reply_subject,
            recipient_agent=parent["sender_agent"],
            sender_agent=sender_agent,
            intent=intent,
            body=body,
            artifact_refs=artifact_refs,
            project=parent["project"],
            thread_id=parent["thread_id"],
            reply_to=message_id,
        )
        conn.execute(
            "UPDATE messages SET status='done', claimed_by='', updated_at=?"
            " WHERE id=?",
            (now_iso(), message_id),
        )
        conn.commit()
        return get_message(conn, reply_id), thread_id
    except Exception:
        conn.rollback()
        raise


def _migrate_legacy_reviews(conn):
    """Project old review rows into the generic mailbox, idempotently.

    Legacy tables remain read-only compatibility data.  ``source_key`` keeps
    every connect() replay-safe while preserving the original request/round
    timestamps and exact Git references.
    """
    reviews = conn.execute("SELECT * FROM reviews ORDER BY id").fetchall()
    for review in reviews:
        request_key = "legacy-review:%d:request" % review["id"]
        existing = conn.execute(
            "SELECT id FROM messages WHERE source_key=?", (request_key,),
        ).fetchone()
        if existing is None:
            refs = ["git-head:" + review["head_sha"]]
            if review["base_sha"]:
                refs.insert(
                    0,
                    "git-range:%s..%s" % (
                        review["base_sha"], review["head_sha"],
                    ),
                )
            legacy_status = review["status"]
            status = (
                legacy_status if legacy_status in ("open", "claimed", "superseded")
                else "done"
            )
            _insert_message(
                conn, review["title"],
                recipient_agent=review["reviewer_agent"],
                sender_agent=review["requester_agent"],
                intent="peer_review",
                body=review["body"],
                artifact_refs=refs,
                project=review["project"],
                machine=review["machine"],
                thread_id="legacy-review-%d" % review["id"],
                status=status,
                claimed_by=review["claimed_by"] if status == "claimed" else "",
                source_key=request_key,
                created_at=review["created_at"],
                updated_at=review["updated_at"],
                ignore_conflict=True,
            )
            request_id = conn.execute(
                "SELECT id FROM messages WHERE source_key=?", (request_key,),
            ).fetchone()["id"]
        else:
            request_id = existing["id"]
            current = get_message(conn, request_id)
            if current["updated_at"] < review["updated_at"]:
                legacy_status = review["status"]
                status = (
                    legacy_status
                    if legacy_status in ("open", "claimed", "superseded")
                    else "done"
                )
                conn.execute(
                    "UPDATE messages SET status=?, claimed_by=?, updated_at=?"
                    " WHERE id=?",
                    (
                        status,
                        review["claimed_by"] if status == "claimed" else "",
                        review["updated_at"],
                        request_id,
                    ),
                )
        rounds = conn.execute(
            "SELECT * FROM review_rounds WHERE review_id=? ORDER BY id",
            (review["id"],),
        ).fetchall()
        for result in rounds:
            source_key = "legacy-review:%d:round:%d" % (
                review["id"], result["id"],
            )
            if conn.execute(
                "SELECT 1 FROM messages WHERE source_key=?", (source_key,),
            ).fetchone() is not None:
                continue
            _insert_message(
                conn, "Re: %s [%s]" % (review["title"], result["verdict"]),
                recipient_agent=review["requester_agent"],
                sender_agent=result["reviewer_agent"],
                intent="review_feedback",
                body=result["body"],
                artifact_refs=["git-head:" + result["head_sha"]],
                project=review["project"],
                machine=review["machine"],
                thread_id="legacy-review-%d" % review["id"],
                reply_to=request_id,
                status="done",
                source_key=source_key,
                created_at=result["created_at"],
                updated_at=result["created_at"],
                ignore_conflict=True,
            )
    conn.commit()


# ------------------------------------------------ legacy review compatibility
# Kept so older databases and /api/coop/reviews readers remain intact while
# connect() projects every row into the generic mailbox.  No CLI command or
# model runner creates/dispatches this specialized workflow anymore.


def add_review(conn, title, *, body="", project=None, requester_agent=None,
               reviewer_agent=None, reviewer_model="", machine=None,
               base_sha="", head_sha=""):
    """Create one legacy review row and return its id."""
    if not title.strip():
        raise ValueError("review title must not be empty")
    if not reviewer_agent or not reviewer_agent.strip():
        raise ValueError("legacy review reviewer_agent must not be empty")
    if not head_sha.strip():
        raise ValueError("review head_sha must not be empty")
    created = now_iso()
    cur = conn.execute(
        "INSERT INTO reviews (project, requester_agent, reviewer_agent,"
        " reviewer_model, machine, title, body, base_sha, head_sha, status,"
        " created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?,'open',?,?)",
        (
            project or detect_project(), requester_agent or detect_agent(),
            reviewer_agent, reviewer_model or "", machine or detect_machine(),
            title.strip(), body or "", base_sha or "", head_sha,
            created, created,
        ),
    )
    conn.commit()
    return cur.lastrowid


def get_review(conn, review_id):
    return conn.execute(
        "SELECT * FROM reviews WHERE id=?", (review_id,),
    ).fetchone()


def list_review_rounds(conn, review_id):
    return conn.execute(
        "SELECT * FROM review_rounds WHERE review_id=? ORDER BY id",
        (review_id,),
    ).fetchall()


def list_reviews(conn, *, query=None, project=None, status=None,
                 reviewer_agent=None, requester_agent=None, limit=100):
    where, params = [], []
    for column, value in (
        ("project", project), ("status", status),
        ("reviewer_agent", reviewer_agent),
        ("requester_agent", requester_agent),
    ):
        if value is not None:
            where.append("%s=?" % column)
            params.append(value)
    if query:
        where.append("(title LIKE ? OR body LIKE ?)")
        like = "%" + query + "%"
        params.extend((like, like))
    sql = "SELECT * FROM reviews"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY id DESC LIMIT ?"
    return conn.execute(sql, [*params, limit]).fetchall()


def review_counts(conn, *, project=None):
    counts = {status: 0 for status in REVIEW_STATUSES}
    where = " WHERE project=?" if project is not None else ""
    params = [project] if project is not None else []
    for row in conn.execute(
        "SELECT status, COUNT(*) AS n FROM reviews" + where
        + " GROUP BY status",
        params,
    ):
        counts[row["status"]] = row["n"]
    counts["total"] = sum(counts.values())
    counts["attention"] = counts["open"] + counts["claimed"] + counts["changes_requested"]
    return counts


def claim_review(conn, review_id, reviewer_agent):
    """Atomically claim an open request for its named reviewer."""
    conn.execute("BEGIN IMMEDIATE")
    try:
        row = get_review(conn, review_id)
        if row is None:
            raise ValueError("no review with id %d" % review_id)
        if row["reviewer_agent"] != reviewer_agent:
            raise ValueError(
                "review #%d targets %s, not %s" % (
                    review_id, row["reviewer_agent"], reviewer_agent,
                )
            )
        if row["status"] != "open":
            raise ValueError(
                "review #%d is %s, not open" % (review_id, row["status"])
            )
        cur = conn.execute(
            "UPDATE reviews SET status='claimed', claimed_by=?, last_error='',"
            " updated_at=? WHERE id=? AND status='open'",
            (reviewer_agent, now_iso(), review_id),
        )
        if cur.rowcount != 1:  # pragma: no cover - BEGIN IMMEDIATE serializes
            raise ValueError("review #%d was claimed concurrently" % review_id)
        conn.commit()
        return get_review(conn, review_id)
    except Exception:
        conn.rollback()
        raise


def release_review(conn, review_id, reviewer_agent, error):
    """Return a failed dispatch to the durable queue without losing context."""
    cur = conn.execute(
        "UPDATE reviews SET status='open', claimed_by='', last_error=?,"
        " updated_at=? WHERE id=? AND status='claimed' AND claimed_by=?",
        ((error or "review dispatch failed")[:4000], now_iso(), review_id,
         reviewer_agent),
    )
    conn.commit()
    return cur.rowcount == 1


def submit_review(conn, review_id, reviewer_agent, verdict, body="",
                  reviewer_model=""):
    """Persist a verdict for the exact claimed head and advance lifecycle."""
    if verdict not in REVIEW_VERDICTS:
        raise ValueError("verdict must be one of: %s" % ", ".join(REVIEW_VERDICTS))
    conn.execute("BEGIN IMMEDIATE")
    try:
        row = get_review(conn, review_id)
        if row is None:
            raise ValueError("no review with id %d" % review_id)
        if row["status"] != "claimed" or row["claimed_by"] != reviewer_agent:
            raise ValueError(
                "review #%d is not claimed by %s" % (review_id, reviewer_agent)
            )
        created = now_iso()
        conn.execute(
            "INSERT INTO review_rounds (review_id, reviewer_agent, reviewer_model,"
            " head_sha, verdict, body, created_at) VALUES (?,?,?,?,?,?,?)",
            (
                review_id, reviewer_agent,
                reviewer_model or row["reviewer_model"], row["head_sha"],
                verdict, body or "", created,
            ),
        )
        conn.execute(
            "UPDATE reviews SET status=?, claimed_by='', last_error='',"
            " updated_at=? WHERE id=?",
            (verdict, created, review_id),
        )
        conn.commit()
        return get_review(conn, review_id)
    except Exception:
        conn.rollback()
        raise


def resubmit_review(conn, review_id, requester_agent, head_sha, body=None):
    """Re-open a changes-requested review at a new exact Git head."""
    if not head_sha.strip():
        raise ValueError("review head_sha must not be empty")
    conn.execute("BEGIN IMMEDIATE")
    try:
        row = get_review(conn, review_id)
        if row is None:
            raise ValueError("no review with id %d" % review_id)
        if row["requester_agent"] != requester_agent:
            raise ValueError(
                "review #%d belongs to requester %s" % (
                    review_id, row["requester_agent"],
                )
            )
        if row["status"] not in ("open", "changes_requested"):
            raise ValueError(
                "review #%d is %s, not open/changes_requested" % (
                    review_id, row["status"],
                )
            )
        if row["head_sha"] == head_sha:
            raise ValueError("review #%d already covers %s" % (review_id, head_sha[:12]))
        next_body = row["body"] if body is None else body
        conn.execute(
            "UPDATE reviews SET head_sha=?, body=?, status='open', claimed_by='',"
            " last_error='', updated_at=? WHERE id=?",
            (head_sha, next_body, now_iso(), review_id),
        )
        conn.commit()
        return get_review(conn, review_id)
    except Exception:
        conn.rollback()
        raise


# ---------------------------------------------------------------- commands


def cmd_add(args, kind):
    conn = connect()
    entry_id, slug = add_entry(
        conn, kind, args.title, body=read_body(args.body),
        project=args.project, agent=args.agent, slug=args.slug,
        commit_sha=getattr(args, "sha", None),
    )
    print("coop #%d %s recorded — %s" % (entry_id, kind, slug))
    if kind in ("history", "learning") and not getattr(args, "sha", None):
        print("(entry-first flow: after the commit lands, backfill with "
              "`coop.py sha %d <head>`)" % entry_id)


def cmd_sha(args):
    conn = connect()
    try:
        changed = set_commit_sha(conn, args.id, args.commit_sha)
    except ValueError as exc:
        sys.exit(str(exc))
    if not changed:
        sys.exit("no history/learning entry with id %d" % args.id)
    print("coop #%d sha=%s" % (args.id, args.commit_sha))


def cmd_status(args):
    new_status = "done" if args.cmd == "done" else args.status
    conn = connect()
    try:
        changed = set_status(conn, args.id, new_status)
    except ValueError as exc:
        sys.exit(str(exc))
    if not changed:
        sys.exit("no note/todo with id %d" % args.id)
    print("coop #%d -> %s" % (args.id, new_status))


def format_row(r):
    flag = (" [%s]" % r["status"]) if r["status"] not in ("", "done") else ""
    return "#%-4d %-8s %s  %s · %s · %s%s" % (
        r["id"], r["kind"], r["created_at"][:16].replace("T", " "),
        r["title"], r["project"], r["agent"], flag,
    )


def cmd_recent(args):
    conn = connect()
    rows = list_entries(
        conn,
        kind=args.kind,
        project=None if args.all_projects else detect_project(args.project),
        limit=args.n,
    )
    if not rows:
        print("coop: no entries yet")
    for r in rows:
        print(format_row(r))


def cmd_search(args):
    conn = connect()
    rows = list_entries(
        conn,
        query=args.query,
        kind=args.kind,
        project=None if args.all_projects else detect_project(args.project),
        limit=args.n,
    )
    if not rows:
        print("coop: no match for %r" % args.query)
    for r in rows:
        print(format_row(r))


def cmd_show(args):
    conn = connect()
    r = get_entry(conn, args.id)
    if not r:
        sys.exit("no entry with id %d" % args.id)
    print(format_row(r))
    print("slug: %s · machine: %s · updated: %s" % (r["slug"], r["machine"], r["updated_at"]))
    if r["commit_sha"]:
        print("commit: %s" % r["commit_sha"])
    if r["body"]:
        print("\n" + r["body"])


def format_message(row):
    reply = (" reply:%s" % row["reply_to"]) if row["reply_to"] else ""
    return "#%-4d message  %s  %s -> %s · %s · %s [%s]%s" % (
        row["id"], row["created_at"][:16].replace("T", " "),
        row["sender_agent"], row["recipient_agent"], row["intent"],
        row["subject"], row["status"], reply,
    )


def cmd_send(args):
    conn = connect()
    try:
        message_id, thread_id = add_message(
            conn, args.subject,
            recipient_agent=args.to,
            sender_agent=detect_agent(args.agent),
            intent=args.intent,
            body=read_body(args.body),
            artifact_refs=args.ref,
            project=detect_project(args.project),
        )
    except ValueError as exc:
        sys.exit(str(exc))
    finally:
        conn.close()
    print(
        "coop message #%d sent to %s · thread %s" % (
            message_id, args.to, thread_id,
        )
    )


def cmd_inbox(args):
    agent = detect_agent(args.agent)
    conn = connect()
    try:
        rows = list_messages(
            conn,
            project=None if args.all_projects else detect_project(args.project),
            recipient_agent=agent,
            limit=args.n,
        )
    finally:
        conn.close()
    if not args.all:
        rows = [row for row in rows if row["status"] in ("open", "claimed")]
    if not rows:
        print("coop: inbox empty for %s" % agent)
    for row in rows:
        print(format_message(row))


def cmd_messages(args):
    conn = connect()
    try:
        rows = list_messages(
            conn,
            query=args.query,
            project=None if args.all_projects else detect_project(args.project),
            status=args.status,
            thread_id=args.thread,
            limit=args.n,
        )
    finally:
        conn.close()
    if not rows:
        print("coop: no messages")
    for row in rows:
        print(format_message(row))


def cmd_message_show(args):
    conn = connect()
    try:
        row = get_message(conn, args.id)
    finally:
        conn.close()
    if row is None:
        sys.exit("no message with id %d" % args.id)
    print(format_message(row))
    print(
        "thread: %s · project: %s · machine: %s · updated: %s" % (
            row["thread_id"], row["project"], row["machine"], row["updated_at"],
        )
    )
    refs = message_artifact_refs(row)
    if refs:
        print("refs: " + ", ".join(refs))
    if row["body"]:
        print("\n" + row["body"])


def cmd_message_claim(args):
    agent = detect_agent(args.agent)
    conn = connect()
    try:
        row = claim_message(conn, args.id, agent)
    except ValueError as exc:
        sys.exit(str(exc))
    finally:
        conn.close()
    print("coop message #%d claimed by %s" % (args.id, row["claimed_by"]))


def cmd_message_release(args):
    agent = detect_agent(args.agent)
    conn = connect()
    try:
        changed = release_message(conn, args.id, agent)
    finally:
        conn.close()
    if not changed:
        sys.exit("message #%d is not claimed by %s" % (args.id, agent))
    print("coop message #%d released to open" % args.id)


def cmd_reply(args):
    agent = detect_agent(args.agent)
    refs = list(args.ref)
    result_message_id = ""
    if args.intent == "work_result" and os.environ.get("FORREST_WORKSPACE") == "1":
        task_id = os.environ.get("FORREST_WORK_TASK_ID", "").strip()
        generation = os.environ.get("FORREST_WORK_GENERATION", "").strip()
        root_message_id = os.environ.get("FORREST_WORK_UNIT_MESSAGE_ID", "").strip()
        result_message_id = os.environ.get(
            "FORREST_WORK_RESULT_MESSAGE_ID",
            root_message_id,
        ).strip()
        if not task_id or not generation.isdigit() or int(generation) <= 0:
            sys.exit("managed work_result is missing its Forrest generation identity")
        refs.extend((
            "forrest-work-task:" + task_id,
            "forrest-work-generation:" + generation,
        ))
        refs = list(dict.fromkeys(refs))
    try:
        with _managed_work_result_guard(
            args.intent,
            target_message_id=args.id,
            fallback_message_id=result_message_id or None,
        ):
            conn = connect()
            try:
                row, thread_id = reply_message(
                    conn, args.id,
                    sender_agent=agent,
                    body=read_body(args.body),
                    subject=args.subject,
                    intent=args.intent,
                    artifact_refs=refs,
                )
            finally:
                conn.close()
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        sys.exit(str(exc))
    print(
        "coop reply #%d sent to %s · thread %s" % (
            row["id"], row["recipient_agent"], thread_id,
        )
    )


def cmd_message_done(args):
    agent = detect_agent(args.agent)
    conn = connect()
    try:
        finish_message(conn, args.id, agent, status=args.status)
    except ValueError as exc:
        sys.exit(str(exc))
    finally:
        conn.close()
    print("coop message #%d -> %s" % (args.id, args.status))


# ---------------------------------------------------------------- brief


def git_line(cwd):
    def run(*argv):
        p = subprocess.run(["git", "-C", cwd] + list(argv),
                           capture_output=True, text=True, timeout=15)
        return p.returncode, p.stdout.strip()

    rc, top = run("rev-parse", "--show-toplevel")
    if rc != 0:
        return "Repo    : (not a git repository)", []
    rc, branch = run("rev-parse", "--abbrev-ref", "HEAD")
    rc, dirty = run("status", "--porcelain")
    dirty_n = len([l for l in dirty.splitlines() if l.strip()])
    warns = []
    if dirty_n:
        warns.append("%d uncommitted file(s) — commit before ending the session" % dirty_n)
    rc, remotes = run("remote")
    if not remotes:
        sync = "NO ORIGIN"
        warns.append("no git origin — repo is unsynced (ground-truth bare repo missing?)")
    else:
        rc, counts = run("rev-list", "--left-right", "--count", "@{u}...HEAD")
        if rc != 0:
            sync = "origin: no upstream for %s" % branch
            warns.append("branch has no upstream — `git push -u origin %s`" % branch)
        else:
            behind, ahead = (int(x) for x in counts.split())
            sync = "origin: in sync" if not (ahead or behind) else \
                "origin: %d ahead / %d behind" % (ahead, behind)
            if ahead:
                warns.append("%d unpushed commit(s) — push (satellites pull from origin)" % ahead)
            if behind:
                warns.append("%d commit(s) behind origin — pull before working" % behind)
    state = "clean" if not dirty_n else "%d dirty" % dirty_n
    return "Repo    : %s @ %s — %s | %s" % (
        os.path.basename(top), branch, state, sync), warns


def cmd_brief(args):
    try:
        machine = detect_machine()
        truth = os.environ.get("COOP_GROUND_TRUTH_HOST", GROUND_TRUTH_HOST)
        print("━━ coop session brief ━━")
        if machine == truth:
            print("Machine : %s — GROUND-TRUTH server (coop DB + git origin live here)" % machine)
        else:
            print("Machine : %s — SATELLITE (ground truth: %s)" % (machine, truth))
            print("  ⚠ You are NOT on the ground-truth machine. Pull before working;"
                  " commit+push after EVERY unit; never let the sides diverge.")
        repo_line, warns = git_line(os.getcwd())
        print(repo_line)
        for w in warns:
            print("  ⚠ " + w)
        conn = connect()
        project = detect_project(args.project)
        agent = detect_agent(args.agent)
        open_rows = list_entries(conn, status="open", limit=5)
        if open_rows:
            print("Open    :")
            for r in open_rows:
                print("  " + format_row(r))
        recent = list_entries(conn, project=project, limit=args.n)
        if recent:
            print("Recent  : (project %s)" % project)
            for r in recent:
                print("  " + format_row(r))
        inbox = [
            row for row in list_messages(
                conn, project=project, recipient_agent=agent, limit=20,
            )
            if row["status"] in ("open", "claimed")
        ][:5]
        if inbox:
            print("Inbox   : (agent %s)" % agent)
            for row in inbox:
                print("  " + format_message(row))
        conn.close()
        print("Rules   : per work unit → `coop.py log` BEFORE the commit,"
              " then commit+push, then `coop.py sha <id> <head>`;"
              " non-obvious lesson → `coop.py learn`;"
              " optional collaboration → `coop.py send/inbox/reply`;"
              " before deep debugging → `coop.py search <kw>`.")
        print("━━━━━━━━")
    except Exception as exc:  # a broken brief must never block a session start
        db_path = os.path.abspath(
            os.environ.get("COOP_DB_PATH") or DEFAULT_DB,
        )
        retry = "python3 tools/coop/coop.py brief"
        if args.agent:
            retry += " --agent %s" % args.agent
        print("━━ ⚠ COOP BRIEF UNAVAILABLE ━━")
        print("Inbox and WorkUnits were NOT loaded; do not treat this as an empty inbox.")
        print("Database: %s" % db_path)
        print("Reason  : %s" % exc)
        print(
            "Action  : grant this app read/write access to the database and "
            "its parent directory, then rerun:"
        )
        print("          " + retry)
        print("Do not start project work until the retry prints the normal Coop brief.")
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    return 0


# ---------------------------------------------------------------- main


def main(argv=None):
    p = argparse.ArgumentParser(prog="coop", description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_writer(name):
        s = sub.add_parser(name)
        s.add_argument("title")
        s.add_argument("-b", "--body", default="", help="entry body; '-' reads stdin")
        s.add_argument("--slug")
        s.add_argument("--project")
        s.add_argument("--agent")
        if name in ("log", "learn"):
            s.add_argument("--sha", help="git head when already known "
                           "(machine writers / retroactive records)")
        return s

    for name in ("log", "learn", "note", "todo"):
        add_writer(name)

    s = sub.add_parser("sha")
    s.add_argument("id", type=int)
    s.add_argument("commit_sha")

    s = sub.add_parser("done")
    s.add_argument("id", type=int)
    s = sub.add_parser("status")
    s.add_argument("id", type=int)
    s.add_argument("status", choices=STATUSES)
    s = sub.add_parser("show")
    s.add_argument("id", type=int)

    for name in ("recent", "search"):
        s = sub.add_parser(name)
        if name == "search":
            s.add_argument("query")
        s.add_argument("-n", type=int, default=10)
        s.add_argument("--kind", choices=KINDS)
        s.add_argument("--project")
        s.add_argument("--all-projects", action="store_true",
                       default=(name == "search"))

    s = sub.add_parser("brief")
    s.add_argument("-n", type=int, default=5)
    s.add_argument("--project")
    s.add_argument("--agent")

    s = sub.add_parser("send")
    s.add_argument("subject")
    s.add_argument("--to", required=True, help="recipient agent id")
    s.add_argument("--intent", default="request")
    s.add_argument("-b", "--body", default="", help="message body; '-' reads stdin")
    s.add_argument("--ref", action="append", default=[],
                   help="artifact reference; repeat for more than one")
    s.add_argument("--project")
    s.add_argument("--agent")

    s = sub.add_parser("inbox")
    s.add_argument("-n", type=int, default=20)
    s.add_argument("--all", action="store_true", help="include completed messages")
    s.add_argument("--all-projects", action="store_true")
    s.add_argument("--project")
    s.add_argument("--agent")

    s = sub.add_parser("messages")
    s.add_argument("-n", type=int, default=50)
    s.add_argument("--query")
    s.add_argument("--status", choices=MESSAGE_STATUSES)
    s.add_argument("--thread")
    s.add_argument("--all-projects", action="store_true")
    s.add_argument("--project")

    s = sub.add_parser("message-show")
    s.add_argument("id", type=int)

    s = sub.add_parser("message-claim")
    s.add_argument("id", type=int)
    s.add_argument("--agent")

    s = sub.add_parser("message-release")
    s.add_argument("id", type=int)
    s.add_argument("--agent")

    s = sub.add_parser("reply")
    s.add_argument("id", type=int)
    s.add_argument("-b", "--body", required=True, help="reply body; '-' reads stdin")
    s.add_argument("--subject")
    s.add_argument("--intent", default="response")
    s.add_argument("--ref", action="append", default=[])
    s.add_argument("--agent")

    s = sub.add_parser("message-done")
    s.add_argument("id", type=int)
    s.add_argument("--status", choices=("done", "superseded"), default="done")
    s.add_argument("--agent")

    args = p.parse_args(argv)
    kind_map = {"log": "history", "learn": "learning", "note": "note", "todo": "todo"}
    if args.cmd in kind_map:
        cmd_add(args, kind_map[args.cmd])
    elif args.cmd == "sha":
        cmd_sha(args)
    elif args.cmd in ("done", "status"):
        cmd_status(args)
    elif args.cmd == "recent":
        cmd_recent(args)
    elif args.cmd == "search":
        cmd_search(args)
    elif args.cmd == "show":
        cmd_show(args)
    elif args.cmd == "brief":
        cmd_brief(args)
    elif args.cmd == "send":
        cmd_send(args)
    elif args.cmd == "inbox":
        cmd_inbox(args)
    elif args.cmd == "messages":
        cmd_messages(args)
    elif args.cmd == "message-show":
        cmd_message_show(args)
    elif args.cmd == "message-claim":
        cmd_message_claim(args)
    elif args.cmd == "message-release":
        cmd_message_release(args)
    elif args.cmd == "reply":
        cmd_reply(args)
    elif args.cmd == "message-done":
        cmd_message_done(args)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""coop — multi-instance work history + learning DB (CWK coop pattern, 2026-07-17).

Every AI instance that works on BJT's codebases (Claude Code, Codex CLI,
a human in a terminal) records what it did (`log`), what it learned
(`learn`), and can leave handoff letters (`note`) and assigned todos (`todo`)
for the next session. Stateless models forget; the coop is the shared, durable,
retrievable record that lets them grow across sessions — CWK: models must
retrieve past learnings when they hit a problem, or there is no growth loop.

Letters are Memory L6's past-self → future-self handoffs (schema v3): a
dedicated table with a six-state machine — unread → read → working → done,
plus abandoned / superseded. `note` writes one (optionally `--to` an agent;
without a recipient it is for any future session of the project). The
session ritual is `letter read` (the only command that marks letters read),
`letter start <id>` when acting on one, `letter ack <id> --body …` when it
shipped. Nothing here polls, claims or notifies: every transition is an
explicit command in an instance session (Memory L6: 아빠 is the coordinator).

Learnings are history's typed sister table (schema v4, Memory L6
`learning_entries`): history asks what a session did, a learning what a
framework taught when it blocked the work. `learn <slug>` records one — the
slug names the symptom, `--scope` the framework or language, and typed fields
hold the problem, hypothesis, resolution and source. A learning written before
v4 stays a free-body entries row, readable and searchable as before.

Agents may also address generic mailbox messages to one another.  Coop owns
delivery, threads, artifact references and claim state; the agents own the
judgment about whether, when and how to collaborate.

Ground truth lives on ONE machine (the always-on server) at
$HOME/forrest-db/coop.db — deliberately outside the Dropbox CloudStorage
tree (live SQLite + cloud sync don't mix; CLAUDE.md Dropbox caution).
Satellite machines reach it over SSH; they do not carry their own copy.

`brief` prints the session-start card without changing any state: which
machine this is, whether it is the ground-truth server or a satellite, git
sync state (dirty / unpushed / no-origin), the agent's unread and working
letters, the todos assigned to it, and the freshest entries. Wired as a Claude
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
# The CLI/filter vocabulary. ``note`` stays a word people and docs use, but
# since schema v3 a note is a row in ``letters``; ``entries`` keeps only the
# ENTRY_KINDS below plus the frozen pre-v3 note rows (see _migrate_schema).
# Since schema v4 a new ``learning`` is a typed row in ``learning_entries``;
# the entries kind stays readable for the free-body learnings written before
# it, and no writer adds another (add_entry refuses the kind).
KINDS = ("history", "learning", "note", "todo")
ENTRY_KINDS = ("history", "learning", "todo")
# A typed learning's text columns, in index order: the FTS index and the
# LIKE fallback both search exactly these.
LEARNING_TEXT_FIELDS = (
    "scope", "slug", "problem", "hypothesis", "resolution", "source",
)
# Todo statuses (``done``/``status`` commands, /api/coop/entries/{id}/status).
STATUSES = ("open", "read", "done", "superseded")
LETTER_STATUSES = (
    "unread", "read", "working", "done", "abandoned", "superseded",
)
ACTIVE_LETTER_STATUSES = ("unread", "read", "working")
# action -> (statuses it may leave, status it enters). Anything else is an
# illegal transition and is refused, never coerced.
LETTER_TRANSITIONS = {
    "read": (("unread",), "read"),
    "start": (("read",), "working"),
    "ack": (("working",), "done"),
    "abandon": (ACTIVE_LETTER_STATUSES, "abandoned"),
    "supersede": (ACTIVE_LETTER_STATUSES, "superseded"),
}
REVIEW_STATUSES = (
    "open", "claimed", "changes_requested", "approved", "superseded",
)
REVIEW_VERDICTS = ("approved", "changes_requested")
MESSAGE_STATUSES = ("open", "claimed", "done", "superseded")
ACTIVE_MESSAGE_STATUSES = ("open", "claimed")
COOP_SCHEMA_VERSION = 4
BRIEF_ITEM_LIMIT = 10

# ---------------------------------------------------------------- storage

FTS_OK = False  # set by connect(); sqlite3.Connection forbids ad-hoc attributes


class LetterNotFound(LookupError):
    """No letter carries the requested id."""


class LetterTransitionError(ValueError):
    """The letter's state or address does not admit the requested move."""


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
  assignee TEXT NOT NULL DEFAULT '',
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

# Created by the v3 migration inside its own transaction (not by the
# pre-migration _TABLE_SCHEMA script), so the table, the copied notes and the
# version stamp commit together or not at all.
_LETTERS_TABLE = """
CREATE TABLE IF NOT EXISTS letters (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  project TEXT NOT NULL,
  author_agent TEXT NOT NULL,
  recipient_agent TEXT
    CHECK (recipient_agent IS NULL OR recipient_agent != ''),
  machine TEXT NOT NULL,
  subject TEXT NOT NULL,
  body TEXT NOT NULL DEFAULT '',
  status TEXT NOT NULL DEFAULT 'unread'
    CHECK (status IN ('unread','read','working','done','abandoned','superseded')),
  worker_agent TEXT NOT NULL DEFAULT '',
  ack_body TEXT NOT NULL DEFAULT '',
  superseded_by INTEGER REFERENCES letters(id),
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
)
"""

# Created by the v4 migration inside its own transaction, as the letters table
# is. Memory L6 names the columns: "scope 가 framework 나 언어 …, slug 은
# 증상-이름, typed 컬럼이 problem / hypothesis / resolution / source 를 들어".
# The lesson does not say which are required; Forrest requires the slug, the
# problem and the resolution (add_learning), and these CHECKs hold that for
# every writer.
_LEARNING_ENTRIES_TABLE = """
CREATE TABLE IF NOT EXISTS learning_entries (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  project TEXT NOT NULL,
  author_agent TEXT NOT NULL,
  scope TEXT NOT NULL DEFAULT '',
  slug TEXT NOT NULL CHECK (slug != ''),
  problem TEXT NOT NULL CHECK (problem != ''),
  hypothesis TEXT NOT NULL DEFAULT '',
  resolution TEXT NOT NULL CHECK (resolution != ''),
  source TEXT NOT NULL DEFAULT '',
  commit_sha TEXT NOT NULL DEFAULT '',
  created_at TEXT NOT NULL
)
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
CREATE INDEX IF NOT EXISTS idx_letters_project_status
  ON letters(project, status, id);
CREATE INDEX IF NOT EXISTS idx_letters_recipient_status
  ON letters(recipient_agent, status, id);
CREATE INDEX IF NOT EXISTS idx_learning_entries_project
  ON learning_entries(project, id);
"""

# The two external-content FTS5 indexes, each a (table name, statements) pair
# that _ensure_fts_index runs one statement at a time, in one transaction with
# the rebuild. entries_fts indexes free-body entries (title, body); the typed
# learnings' index covers LEARNING_TEXT_FIELDS.
_FTS_SCHEMA = (
    """
CREATE VIRTUAL TABLE IF NOT EXISTS entries_fts
  USING fts5(title, body, content='entries', content_rowid='id')
""",
    """
CREATE TRIGGER IF NOT EXISTS entries_ai AFTER INSERT ON entries BEGIN
  INSERT INTO entries_fts(rowid, title, body) VALUES (new.id, new.title, new.body);
END
""",
    """
CREATE TRIGGER IF NOT EXISTS entries_ad AFTER DELETE ON entries BEGIN
  INSERT INTO entries_fts(entries_fts, rowid, title, body)
    VALUES ('delete', old.id, old.title, old.body);
END
""",
    """
CREATE TRIGGER IF NOT EXISTS entries_au AFTER UPDATE ON entries BEGIN
  INSERT INTO entries_fts(entries_fts, rowid, title, body)
    VALUES ('delete', old.id, old.title, old.body);
  INSERT INTO entries_fts(rowid, title, body) VALUES (new.id, new.title, new.body);
END
""",
)

_LEARNING_FTS_SCHEMA = (
    """
CREATE VIRTUAL TABLE IF NOT EXISTS learning_entries_fts
  USING fts5(scope, slug, problem, hypothesis, resolution, source,
             content='learning_entries', content_rowid='id')
""",
    """
CREATE TRIGGER IF NOT EXISTS learning_entries_ai AFTER INSERT ON learning_entries BEGIN
  INSERT INTO learning_entries_fts(rowid, scope, slug, problem, hypothesis,
    resolution, source)
    VALUES (new.id, new.scope, new.slug, new.problem, new.hypothesis,
    new.resolution, new.source);
END
""",
    """
CREATE TRIGGER IF NOT EXISTS learning_entries_ad AFTER DELETE ON learning_entries BEGIN
  INSERT INTO learning_entries_fts(learning_entries_fts, rowid, scope, slug,
    problem, hypothesis, resolution, source)
    VALUES ('delete', old.id, old.scope, old.slug, old.problem, old.hypothesis,
    old.resolution, old.source);
END
""",
    """
CREATE TRIGGER IF NOT EXISTS learning_entries_au AFTER UPDATE ON learning_entries BEGIN
  INSERT INTO learning_entries_fts(learning_entries_fts, rowid, scope, slug,
    problem, hypothesis, resolution, source)
    VALUES ('delete', old.id, old.scope, old.slug, old.problem, old.hypothesis,
    old.resolution, old.source);
  INSERT INTO learning_entries_fts(rowid, scope, slug, problem, hypothesis,
    resolution, source)
    VALUES (new.id, new.scope, new.slug, new.problem, new.hypothesis,
    new.resolution, new.source);
END
""",
)


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
        if current < 3:
            # Memory L6 (2026-09-23): letters get their own table and state
            # machine, and todos get an assignee.
            conn.execute(_LETTERS_TABLE)
            if "assignee" not in _table_columns(conn, "entries"):
                conn.execute(
                    "ALTER TABLE entries ADD COLUMN "
                    "assignee TEXT NOT NULL DEFAULT ''"
                )
            # An existing todo had no assignee: it stays anyone's (every
            # brief lists it), as every open todo was listed under v2.
            # Each note becomes the letter with the SAME id, so "coop #N"
            # keeps naming it. open had no reader yet (unread); the other
            # v2 statuses exist in both vocabularies; anything unexpected
            # surfaces as unread rather than disappearing. The entries row is
            # kept byte-for-byte as a frozen forensic copy: no v3 reader
            # shows it or writes it, and a rollback to a v2 client only needs
            # `PRAGMA user_version=2`. OR IGNORE makes a re-run after such a
            # rollback keep each letter's live state instead of reverting it.
            conn.execute(
                "INSERT OR IGNORE INTO letters (id, project, author_agent,"
                " recipient_agent, machine, subject, body, status, created_at,"
                " updated_at)"
                " SELECT id, project, agent, NULL, machine, title, body,"
                " CASE status WHEN 'read' THEN 'read' WHEN 'done' THEN 'done'"
                " WHEN 'superseded' THEN 'superseded' ELSE 'unread' END,"
                " created_at, updated_at"
                " FROM entries WHERE kind='note' ORDER BY id"
            )
            conn.execute("PRAGMA user_version=3")
        if current < 4:
            # Memory L6 (EV v3 · 2026-06-05): learning gets history's sister
            # table — "history 는 forensic 책임, learning 은 같은 벽에 부딪힌
            # 자매면 누구든 끌어 쓰는 cross-session 참조". Nothing is copied or
            # rewritten: a pre-v4 learning keeps its entries row and free
            # body, and a v3 client ignores this table, so a rollback is only
            # `PRAGMA user_version=3`. Its FTS index is built after the
            # migration (_ensure_fts_index), as entries' is, so a SQLite
            # without FTS5 still migrates and searches with LIKE.
            conn.execute(_LEARNING_ENTRIES_TABLE)
            conn.execute("PRAGMA user_version=4")
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def _ensure_fts_index(conn, table, statements):
    """Build one external-content FTS index once, complete, or not at all.

    An FTS5 table created beside rows that already exist indexes none of
    them, and a search that then finds only newer rows hides the older ones
    (Memory L6 re-audit F2, 2026-09-08, reproduced on ``entries_fts``). So
    the index, its triggers and a rebuild from its content table commit in
    one transaction: a cut or a failure leaves no index, and the next connect
    builds it again. Raises sqlite3.OperationalError without FTS5, which
    connect() treats as the LIKE fallback.
    """
    probe = "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?"
    if conn.execute(probe, (table,)).fetchone() is not None:
        return
    conn.execute("BEGIN IMMEDIATE")
    try:
        # Another short-lived CLI may have built it while this one waited.
        if conn.execute(probe, (table,)).fetchone() is None:
            for statement in statements:
                conn.execute(statement)
            conn.execute("INSERT INTO %s(%s) VALUES('rebuild')" % (table, table))
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
        _ensure_fts_index(conn, "entries_fts", _FTS_SCHEMA)
        _ensure_fts_index(conn, "learning_entries_fts", _LEARNING_FTS_SCHEMA)
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
    """The project a checkout belongs to: its repository's name.

    A linked worktree (`.claude/worktrees/<x>`, a WorkUnit workspace) shares
    its repository's common git dir, so every checkout of one repository names
    the same project; its own folder name would scope briefs and letters to a
    project nobody else reads.
    """
    if cli_value:
        return cli_value
    try:
        common = subprocess.run(
            ["git", "-C", cwd or os.getcwd(), "rev-parse",
             "--path-format=absolute", "--git-common-dir"],
            capture_output=True, text=True, timeout=10,
        )
        git_dir = common.stdout.strip() if common.returncode == 0 else ""
        if git_dir and os.path.basename(git_dir) == ".git":
            return os.path.basename(os.path.dirname(git_dir))
    except Exception:
        pass
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
              machine=None, slug=None, commit_sha=None, assignee=None):
    """Insert one entry; returns (id, slug).

    Memory L6 ordering: a history/learning entry is written BEFORE its git
    commit (so the *why* survives a crash between commit and record), then
    the head is backfilled via ``set_commit_sha``. ``commit_sha`` here is for
    replay-safe machine writers and retroactive records that already know
    their head.

    A todo carries an optional ``assignee``; an unassigned todo is anyone's
    (every brief lists it). A note is not an entry since schema v3 —
    ``add_letter`` owns it — and a new learning is not one since schema v4:
    ``add_learning`` writes it typed (the entries kind stays readable for the
    learnings written before).
    """
    if kind == "note":
        raise ValueError(
            "a note is a letter since coop schema v3 — use add_letter()",
        )
    if kind == "learning":
        raise ValueError(
            "a learning is typed since coop schema v4 — use add_learning()",
        )
    if kind not in ENTRY_KINDS:
        raise ValueError("kind must be one of: %s" % ", ".join(ENTRY_KINDS))
    if assignee is not None and kind != "todo":
        raise ValueError("only a todo has an assignee")
    created = now_iso()
    author = agent or detect_agent()
    status = "open" if kind == "todo" else ""
    final_assignee = (assignee or "").strip() if kind == "todo" else ""
    final_slug = slug or make_slug(title, created)
    cur = conn.execute(
        "INSERT INTO entries (kind, project, agent, machine, title, body, slug, status,"
        " commit_sha, assignee, created_at, updated_at)"
        " VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        (kind, project or detect_project(), author,
         machine or detect_machine(), title, body or "", final_slug, status,
         commit_sha or "", final_assignee, created, created),
    )
    conn.commit()
    return cur.lastrowid, final_slug


def set_commit_sha(conn, entry_id, commit_sha):
    """Backfill the git head onto a history entry or a learning after the commit.

    Returns True if a row changed. A learning is a typed row since schema v4
    and an entries row before it; the shared id space names at most one of
    them. Todos, letters (and the frozen pre-v3 note rows) have no commit
    identity and are refused so a typo'd id cannot silently decorate the
    wrong kind.
    """
    sha = (commit_sha or "").strip()
    if not sha:
        raise ValueError("commit_sha must be non-empty")
    changed = conn.execute(
        "UPDATE entries SET commit_sha=?, updated_at=? "
        "WHERE id=? AND kind IN ('history','learning')",
        (sha, now_iso(), entry_id),
    ).rowcount
    if not changed:
        changed = conn.execute(
            "UPDATE learning_entries SET commit_sha=? WHERE id=?",
            (sha, entry_id),
        ).rowcount
    conn.commit()
    return changed > 0


def set_status(conn, entry_id, new_status):
    """Set a todo status; returns True if a row changed.

    A note became a letter with its own state machine in schema v3
    (``transition_letter``); its frozen entries row is never rewritten.
    """
    if new_status not in STATUSES:
        raise ValueError("status must be one of: %s" % ", ".join(STATUSES))
    cur = conn.execute(
        "UPDATE entries SET status=?, updated_at=? WHERE id=? AND kind='todo'",
        (new_status, now_iso(), entry_id),
    )
    conn.commit()
    return cur.rowcount > 0


def assign_todo(conn, entry_id, assignee):
    """(Re)assign one todo, or unassign it with None; True if a row changed."""
    value = "" if assignee is None else assignee.strip()
    if assignee is not None and not value:
        raise ValueError("assignee must be non-empty (None unassigns)")
    cur = conn.execute(
        "UPDATE entries SET assignee=?, updated_at=? WHERE id=? AND kind='todo'",
        (value, now_iso(), entry_id),
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


def _entry_filters(kind=None, project=None, status=None, assignee=None,
                   alias=""):
    """Structured predicates for every live ``entries`` reader.

    The first predicate is unconditional: since schema v3 a note lives in
    ``letters``, and its original entries row is a frozen forensic copy that
    must never be presented as current state (its status stopped moving at
    the migration). ``status`` is one value or a sequence of them.
    """
    prefix = (alias + ".") if alias else ""
    where, params = ["%skind != 'note'" % prefix], []
    for column, value in (("kind", kind), ("project", project)):
        if value is not None:
            where.append("%s%s=?" % (prefix, column))
            params.append(value)
    if assignee is not None:
        assignees = (assignee,) if isinstance(assignee, str) else tuple(assignee)
        where.append("%sassignee IN (%s)" % (prefix, ",".join("?" * len(assignees))))
        params.extend(assignees)
    if status is not None:
        statuses = (status,) if isinstance(status, str) else tuple(status)
        where.append("%sstatus IN (%s)" % (prefix, ",".join("?" * len(statuses))))
        params.extend(statuses)
    return where, params


def _fts_match(query):
    """The FTS5 MATCH expression for a free-text query: every term, quoted."""
    return " ".join('"%s"' % term.replace('"', "") for term in query.split())


def search_rows(conn, query, kind=None, limit=10, *, project=None, status=None,
                assignee=None):
    """Search with every structured filter applied *before* LIMIT.

    The old HTTP route filtered project/status in Python after this function
    had already clipped the global result set.  A matching project could then
    look empty merely because newer hits from another project occupied the
    limit.  Keeping the complete predicate here fixes that correctness bug and
    preserves one SQL owner.
    """
    filters, filter_params = _entry_filters(
        kind, project, status, assignee, alias="e",
    )
    structured_sql = (" AND " + " AND ".join(filters)) if filters else ""
    if FTS_OK:
        try:
            rows = conn.execute(
                "SELECT e.* FROM entries_fts f JOIN entries e ON e.id=f.rowid"
                " WHERE entries_fts MATCH ?" + structured_sql
                + " ORDER BY rank LIMIT ?",
                [_fts_match(query), *filter_params, limit],
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
                 assignee=None, limit=100):
    """Return newest/ranked entries under one provider-neutral filter API."""
    if query:
        return search_rows(
            conn, query, kind, limit, project=project, status=status,
            assignee=assignee,
        )
    where, params = _entry_filters(kind, project, status, assignee)
    sql = "SELECT * FROM entries"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY id DESC LIMIT ?"
    return conn.execute(sql, [*params, limit]).fetchall()


def entry_counts(conn):
    """Return dashboard counts without exposing SQL to presentation layers.

    Live entry kinds only: notes are counted by ``letter_counts`` since
    schema v3, and their frozen entries rows are not current state.
    """
    counts = {kind: 0 for kind in ENTRY_KINDS}
    for row in conn.execute(
        "SELECT kind, COUNT(*) AS n FROM entries WHERE kind != 'note'"
        " GROUP BY kind",
    ):
        counts[row["kind"]] = row["n"]
    counts["total"] = sum(counts.values())
    counts["open"] = conn.execute(
        "SELECT COUNT(*) FROM entries WHERE status='open' AND kind != 'note'",
    ).fetchone()[0]
    return counts


def list_projects(conn):
    """Return distinct project names in stable display order."""
    return [row[0] for row in conn.execute(
        "SELECT project FROM ("
        " SELECT project FROM entries"
        " UNION SELECT project FROM letters"
        " UNION SELECT project FROM learning_entries"
        " UNION SELECT project FROM messages"
        " UNION SELECT project FROM reviews"
        ") ORDER BY project",
    ).fetchall()]


# ---------------------------------------------------------------- letters
# Memory L6 handoffs: past-self → future-self notes with their own six-state
# machine (LETTER_TRANSITIONS). The table shares one id space with
# ``entries`` and ``learning_entries`` (_next_ledger_id), so a bare
# "coop #N" names exactly one row. Every move is an explicit command; nothing
# here polls, claims or notifies.


def _next_ledger_id(conn):
    """Next id in the id space ``letters`` and ``learning_entries`` share
    with ``entries``.

    Runs inside the caller's write transaction. A pre-v3 note already has an
    entries id and kept it as its letter id; a new letter or learning takes
    the next id past every entry, letter and learning ever issued
    (AUTOINCREMENT sequences included, so a deleted row's number is never
    reused).
    """
    row = conn.execute(
        "SELECT MAX(n) FROM ("
        " SELECT MAX(id) AS n FROM entries"
        " UNION ALL SELECT MAX(id) FROM letters"
        " UNION ALL SELECT MAX(id) FROM learning_entries"
        " UNION ALL SELECT seq FROM sqlite_sequence"
        " WHERE name IN ('entries','letters','learning_entries'))",
    ).fetchone()
    return int(row[0] or 0) + 1


def _reserve_entry_id(conn, ledger_id):
    """Raise the entries AUTOINCREMENT floor past a letter's or learning's id.

    Otherwise the next entry — written by this module or by any other writer
    relying on AUTOINCREMENT — would reuse that number, and a
    `coop.py done <id>` typed from a letter's receipt could close an
    unrelated todo. It also keeps a v3 client, after a rollback, from
    reusing a learning's id: its own allocation reads this floor. SQLite
    documents ordinary UPDATE/INSERT on sqlite_sequence; the value only ever
    grows here.
    """
    conn.execute(
        "UPDATE sqlite_sequence SET seq=? WHERE name='entries' AND seq<?",
        (ledger_id, ledger_id),
    )
    conn.execute(
        "INSERT INTO sqlite_sequence (name, seq) SELECT 'entries', ?"
        " WHERE NOT EXISTS (SELECT 1 FROM sqlite_sequence WHERE name='entries')",
        (ledger_id,),
    )


def add_letter(conn, subject, *, body="", recipient_agent=None,
               author_agent=None, project=None, machine=None):
    """Write one unread letter and return its id.

    ``recipient_agent=None`` addresses any future session of ``project``.
    """
    text = (subject or "").strip()
    if not text:
        raise ValueError("letter subject must not be empty")
    recipient = (recipient_agent or "").strip() or None
    author = (author_agent or "").strip() or detect_agent()
    scope = project or detect_project()
    host = machine or detect_machine()
    created = now_iso()
    conn.execute("BEGIN IMMEDIATE")
    try:
        letter_id = _next_ledger_id(conn)
        conn.execute(
            "INSERT INTO letters (id, project, author_agent, recipient_agent,"
            " machine, subject, body, status, created_at, updated_at)"
            " VALUES (?,?,?,?,?,?,?,'unread',?,?)",
            (letter_id, scope, author, recipient, host, text, body or "",
             created, created),
        )
        _reserve_entry_id(conn, letter_id)
        conn.commit()
        return letter_id
    except Exception:
        conn.rollback()
        raise


def get_letter(conn, letter_id):
    """Return one letter row, or None."""
    return conn.execute(
        "SELECT * FROM letters WHERE id=?", (letter_id,),
    ).fetchone()


def _check_transition(row, action, agent):
    """Refuse ``action`` unless ``agent`` may apply it to ``row`` now.

    Addressing mirrors the mailbox: reading, starting and acking belong to
    the recipient (anyone, for a letter to any session); the author may also
    withdraw or replace a letter it wrote. Like a claimed message, a working
    letter belongs to the agent that started it: only that worker may ack it,
    and only the worker or the author may abandon or replace it.
    """
    if row["status"] == "working" and agent != row["worker_agent"]:
        if action == "ack" or row["author_agent"] != agent:
            raise LetterTransitionError(
                "letter #%d is being worked on by %s, not %s" % (
                    row["id"], row["worker_agent"], agent,
                )
            )
    addressed = row["recipient_agent"] is None or row["recipient_agent"] == agent
    if action in ("abandon", "supersede"):
        addressed = addressed or row["author_agent"] == agent
    if not addressed:
        raise LetterTransitionError(
            "letter #%d is addressed to %s, not %s" % (
                row["id"], row["recipient_agent"], agent,
            )
        )
    sources, target = LETTER_TRANSITIONS[action]
    if row["status"] not in sources:
        raise LetterTransitionError(
            "letter #%d is %s; `%s` moves %s → %s" % (
                row["id"], row["status"], action, "/".join(sources), target,
            )
        )


def transition_letter(conn, letter_id, action, *, agent, body=None,
                      superseded_by=None):
    """Apply one explicit state-machine move; return the updated row.

    ``ack`` requires a non-empty ``body`` (what shipped); ``supersede`` may
    name the letter that replaces this one. Raises LetterNotFound,
    LetterTransitionError (illegal state or address) or ValueError (input).
    """
    if action not in LETTER_TRANSITIONS:
        raise ValueError(
            "letter action must be one of: %s" % ", ".join(LETTER_TRANSITIONS),
        )
    actor = (agent or "").strip()
    if not actor:
        raise ValueError("a letter transition needs the acting agent")
    ack_body = None
    if action == "ack":
        ack_body = (body or "").strip()
        if not ack_body:
            raise ValueError("letter ack requires a non-empty body: say what shipped")
    elif body is not None:
        raise ValueError("only ack takes a body")
    if superseded_by is not None and action != "supersede":
        raise ValueError("only supersede names a superseding letter")
    target = LETTER_TRANSITIONS[action][1]
    conn.execute("BEGIN IMMEDIATE")
    try:
        row = get_letter(conn, letter_id)
        if row is None:
            raise LetterNotFound("no letter with id %d" % letter_id)
        _check_transition(row, action, actor)
        if superseded_by is not None:
            if superseded_by == letter_id:
                raise ValueError("a letter cannot supersede itself")
            replacement = get_letter(conn, superseded_by)
            if replacement is None:
                raise ValueError("no superseding letter with id %d" % superseded_by)
            if replacement["status"] not in ACTIVE_LETTER_STATUSES:
                # A replacement that is itself closed or replaced would leave
                # the handoff with nowhere to go (and allow A→B→A cycles).
                raise ValueError(
                    "letter #%d is %s and cannot replace another"
                    % (superseded_by, replacement["status"]),
                )
        cur = conn.execute(
            "UPDATE letters SET status=?, worker_agent=?, ack_body=?,"
            " superseded_by=?, updated_at=? WHERE id=? AND status=?",
            (
                target,
                actor if action == "start" else row["worker_agent"],
                row["ack_body"] if ack_body is None else ack_body,
                row["superseded_by"] if superseded_by is None else superseded_by,
                now_iso(), letter_id, row["status"],
            ),
        )
        if cur.rowcount != 1:  # pragma: no cover - BEGIN IMMEDIATE serializes
            raise LetterTransitionError("letter #%d changed concurrently" % letter_id)
        conn.commit()
        return get_letter(conn, letter_id)
    except Exception:
        conn.rollback()
        raise


def read_letters(conn, *, agent, project=None, letter_ids=None):
    """Mark letters read and return them: the `letter read` ritual.

    With ids, exactly those letters; each must be unread and addressed to
    ``agent``, and one refusal rolls the whole call back. Without ids, every
    unread letter addressed to ``agent`` or to any session (in ``project``
    when given), oldest first. Only an explicit read changes state — `brief`
    and `letter list` never mark anything read.
    """
    actor = (agent or "").strip()
    if not actor:
        raise ValueError("reading letters needs the reading agent")
    conn.execute("BEGIN IMMEDIATE")
    try:
        if letter_ids:
            rows = []
            for letter_id in dict.fromkeys(letter_ids):
                row = get_letter(conn, letter_id)
                if row is None:
                    raise LetterNotFound("no letter with id %d" % letter_id)
                _check_transition(row, "read", actor)
                rows.append(row)
        else:
            rows = conn.execute(
                "SELECT * FROM letters WHERE status='unread'"
                " AND (recipient_agent=? OR recipient_agent IS NULL)"
                " AND (? IS NULL OR project=?) ORDER BY id",
                (actor, project, project),
            ).fetchall()
        stamp = now_iso()
        for row in rows:
            conn.execute(
                "UPDATE letters SET status='read', updated_at=?"
                " WHERE id=? AND status='unread'",
                (stamp, row["id"]),
            )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    return [get_letter(conn, row["id"]) for row in rows]


def list_letters(conn, *, query=None, project=None, status=None,
                 addressed_to=None, worker_agent=None, newest_first=False,
                 limit=100):
    """Return letters, active ones first, then newest (or newest only).

    ``addressed_to`` keeps letters whose recipient is that agent or any
    session; ``status`` is one value or a sequence. Every predicate applies
    before LIMIT, so an old unread letter cannot hide behind newer settled
    ones. Each query term must appear in the subject, body or ack body.
    """
    where, params = [], []
    if project is not None:
        where.append("project=?")
        params.append(project)
    if status is not None:
        statuses = (status,) if isinstance(status, str) else tuple(status)
        where.append("status IN (%s)" % ",".join("?" * len(statuses)))
        params.extend(statuses)
    if addressed_to is not None:
        where.append("(recipient_agent=? OR recipient_agent IS NULL)")
        params.append(addressed_to)
    if worker_agent is not None:
        where.append("worker_agent=?")
        params.append(worker_agent)
    for term in (query or "").split():
        like = "%" + term + "%"
        where.append("(subject LIKE ? OR body LIKE ? OR ack_body LIKE ?)")
        params.extend((like, like, like))
    sql = "SELECT * FROM letters"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += (
        " ORDER BY id DESC LIMIT ?" if newest_first
        else " ORDER BY status IN ('unread','read','working') DESC, id DESC LIMIT ?"
    )
    return conn.execute(sql, [*params, limit]).fetchall()


def letter_counts(conn, *, project=None, addressed_to=None):
    """Per-status letter counts plus ``total`` and ``active``."""
    counts = {status: 0 for status in LETTER_STATUSES}
    for row in conn.execute(
        "SELECT status, COUNT(*) AS n FROM letters"
        " WHERE (? IS NULL OR project=?)"
        " AND (? IS NULL OR recipient_agent=? OR recipient_agent IS NULL)"
        " GROUP BY status",
        (project, project, addressed_to, addressed_to),
    ):
        counts[row["status"]] = row["n"]
    counts["total"] = sum(counts[status] for status in LETTER_STATUSES)
    counts["active"] = sum(counts[status] for status in ACTIVE_LETTER_STATUSES)
    return counts


# ---------------------------------------------------------------- learnings
# Memory L6 typed learnings (schema v4), history's sister table. History is
# the forensic record of what a session did; a learning is the cross-session
# reference any sister who hits the same wall can pull. One ledger id space
# with entries and letters (_next_ledger_id).


def add_learning(conn, slug, *, problem, resolution, hypothesis="", source="",
                 scope="", author_agent=None, project=None, commit_sha=None):
    """Write one typed learning and return its id.

    ``slug`` names the symptom and ``scope`` the framework or language. The
    lesson names the columns but not which are required; a learning without
    its problem or its resolution teaches nothing, so both are required and
    the rest are optional. Every text is stripped. ``commit_sha`` is for a
    writer that already knows its head; the entry-first flow backfills it
    with ``set_commit_sha``.
    """
    text = {
        name: (value or "").strip()
        for name, value in (
            ("slug", slug), ("problem", problem), ("resolution", resolution),
            ("hypothesis", hypothesis), ("source", source), ("scope", scope),
        )
    }
    for name in ("slug", "problem", "resolution"):
        if not text[name]:
            raise ValueError("a learning needs a non-empty %s" % name)
    author = (author_agent or "").strip() or detect_agent()
    owner = project or detect_project()
    created = now_iso()
    conn.execute("BEGIN IMMEDIATE")
    try:
        learning_id = _next_ledger_id(conn)
        conn.execute(
            "INSERT INTO learning_entries (id, project, author_agent, scope,"
            " slug, problem, hypothesis, resolution, source, commit_sha,"
            " created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (learning_id, owner, author, text["scope"], text["slug"],
             text["problem"], text["hypothesis"], text["resolution"],
             text["source"], (commit_sha or "").strip(), created),
        )
        _reserve_entry_id(conn, learning_id)
        conn.commit()
        return learning_id
    except Exception:
        conn.rollback()
        raise


def get_learning(conn, learning_id):
    """Return one typed learning row, or None."""
    return conn.execute(
        "SELECT * FROM learning_entries WHERE id=?", (learning_id,),
    ).fetchone()


def list_learnings(conn, *, query=None, project=None, scope=None, limit=100):
    """Return typed learnings: FTS-ranked for a query, newest otherwise.

    ``project`` and ``scope`` apply before LIMIT, as every entries filter
    does. A query searches every LEARNING_TEXT_FIELDS column; when FTS5 is
    unavailable or its index finds nothing, each query term must appear in
    one of them (LIKE).
    """
    where, params = [], []
    for column, value in (("project", project), ("scope", scope)):
        if value is not None:
            where.append("l.%s=?" % column)
            params.append(value)
    terms = (query or "").split()
    if terms and FTS_OK:
        fts_sql = (
            "SELECT l.* FROM learning_entries_fts f"
            " JOIN learning_entries l ON l.id=f.rowid"
            " WHERE learning_entries_fts MATCH ?"
            + "".join(" AND " + clause for clause in where)
            + " ORDER BY rank LIMIT ?"
        )
        try:
            rows = conn.execute(fts_sql, [_fts_match(query), *params, limit]).fetchall()
            if rows:
                return rows
        except sqlite3.OperationalError:
            pass
    for term in terms:
        where.append("(%s)" % " OR ".join(
            "l.%s LIKE ?" % column for column in LEARNING_TEXT_FIELDS
        ))
        params.extend(["%" + term + "%"] * len(LEARNING_TEXT_FIELDS))
    sql = "SELECT l.* FROM learning_entries l"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY l.id DESC LIMIT ?"
    return conn.execute(sql, [*params, limit]).fetchall()


def learning_counts(conn, *, project=None):
    """The typed learnings' ``total`` (in ``project`` when given)."""
    total = conn.execute(
        "SELECT COUNT(*) FROM learning_entries WHERE (? IS NULL OR project=?)",
        (project, project),
    ).fetchone()[0]
    return {"total": total}


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
    """``status`` is one value or a sequence of them; like every other
    predicate it applies before the LIMIT."""
    where, params = [], []
    for column, value in (
        ("project", project),
        ("recipient_agent", recipient_agent), ("sender_agent", sender_agent),
        ("thread_id", thread_id),
    ):
        if value is not None:
            where.append("%s=?" % column)
            params.append(value)
    if status is not None:
        statuses = (status,) if isinstance(status, str) else tuple(status)
        where.append("status IN (%s)" % ",".join("?" * len(statuses)))
        params.extend(statuses)
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
# Read-only: the reviews/review_rounds tables are kept so older databases and
# the /api/coop/reviews readers remain intact while connect() projects every
# row into the generic mailbox.  Nothing writes them any more — the writer
# family (add/claim/submit/resubmit/release) had no CLI command or route and
# left with the 2026-09-24 refinement; a legacy ledger already holds its rows.


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


# ---------------------------------------------------------------- commands


def cmd_add(args, kind):
    conn = connect()
    try:
        entry_id, slug = add_entry(
            conn, kind, args.title, body=read_body(args.body),
            project=args.project, agent=args.agent, slug=args.slug,
            commit_sha=getattr(args, "sha", None),
            assignee=getattr(args, "assignee", None),
        )
        row = get_entry(conn, entry_id)
    except ValueError as exc:
        sys.exit(str(exc))
    finally:
        conn.close()
    suffix = (" (assigned to %s)" % row["assignee"]) if kind == "todo" else ""
    print("coop #%d %s recorded — %s%s" % (entry_id, kind, slug, suffix))
    if kind == "history" and not getattr(args, "sha", None):
        print("(entry-first flow: after the commit lands, backfill with "
              "`coop.py sha %d <head>`)" % entry_id)


def cmd_learn(args):
    """`learn` writes a typed learning (schema v4); '-' reads a field from stdin."""
    conn = connect()
    try:
        learning_id = add_learning(
            conn, args.slug,
            problem=read_body(args.problem),
            resolution=read_body(args.resolution),
            hypothesis=read_body(args.hypothesis),
            source=read_body(args.source),
            scope=args.scope,
            author_agent=args.agent, project=args.project, commit_sha=args.sha,
        )
        row = get_learning(conn, learning_id)
    except ValueError as exc:
        sys.exit(str(exc))
    finally:
        conn.close()
    print("coop #%d learning recorded — %s" % (learning_id, row["slug"]))
    if not args.sha:
        print("(entry-first flow: after the commit lands, backfill with "
              "`coop.py sha %d <head>`)" % learning_id)


def cmd_note(args):
    """`note` keeps its name and habit; since schema v3 it writes a letter."""
    conn = connect()
    try:
        letter_id = add_letter(
            conn, args.title, body=read_body(args.body),
            recipient_agent=args.to, author_agent=args.agent,
            project=args.project,
        )
        row = get_letter(conn, letter_id)
    except ValueError as exc:
        sys.exit(str(exc))
    finally:
        conn.close()
    print("coop letter #%d recorded → %s" % (
        letter_id, row["recipient_agent"] or "any future session of %s" % row["project"],
    ))
    print("(read with `coop.py letter read`; then `letter start %d`, and "
          "`letter ack %d --body …` when it ships)" % (letter_id, letter_id))


def _not_a_todo(conn, entry_id):
    """Explain a todo command that matched nothing — often a letter id."""
    letter = get_letter(conn, entry_id)
    if letter is not None:
        return (
            "coop #%d is a letter (%s), not a todo — letters move with "
            "`coop.py letter read|start|ack|abandon|supersede %d`"
            % (entry_id, letter["status"], entry_id)
        )
    return "no todo with id %d" % entry_id


def cmd_assign(args):
    if args.unassign == bool(args.assignee):
        sys.exit("give an assignee, or --unassign")
    conn = connect()
    try:
        changed = assign_todo(conn, args.id, None if args.unassign else args.assignee)
        if not changed:
            sys.exit(_not_a_todo(conn, args.id))
    except ValueError as exc:
        sys.exit(str(exc))
    finally:
        conn.close()
    if args.unassign:
        print("coop todo #%d is unassigned (anyone's)" % args.id)
    else:
        print("coop todo #%d assigned to %s" % (args.id, args.assignee.strip()))


def cmd_sha(args):
    conn = connect()
    try:
        changed = set_commit_sha(conn, args.id, args.commit_sha)
    except ValueError as exc:
        sys.exit(str(exc))
    if not changed:
        sys.exit("no history entry or learning with id %d" % args.id)
    print("coop #%d sha=%s" % (args.id, args.commit_sha))


def cmd_status(args):
    new_status = "done" if args.cmd == "done" else args.status
    conn = connect()
    try:
        changed = set_status(conn, args.id, new_status)
        if not changed:
            sys.exit(_not_a_todo(conn, args.id))
    except ValueError as exc:
        sys.exit(str(exc))
    finally:
        conn.close()
    print("coop #%d -> %s" % (args.id, new_status))


def format_row(r):
    flag = (" [%s]" % r["status"]) if r["status"] not in ("", "done") else ""
    who = r["agent"]
    if r["kind"] == "todo" and r["assignee"]:
        who = "%s -> %s" % (r["agent"], r["assignee"])
    return "#%-4d %-8s %s  %s · %s · %s%s" % (
        r["id"], r["kind"], r["created_at"][:16].replace("T", " "),
        r["title"], r["project"], who, flag,
    )


def format_letter(row):
    worker = (
        " by %s" % row["worker_agent"]
        if row["status"] == "working" and row["worker_agent"] else ""
    )
    return "#%-4d %-8s %s  %s · %s · %s -> %s [%s%s]" % (
        row["id"], "letter", row["created_at"][:16].replace("T", " "),
        row["subject"], row["project"], row["author_agent"],
        row["recipient_agent"] or "any session", row["status"], worker,
    )


def print_letter(row):
    print(format_letter(row))
    print("machine: %s · updated: %s" % (row["machine"], row["updated_at"]))
    if row["body"]:
        print("\n" + row["body"])
    if row["ack_body"]:
        print("\nack: " + row["ack_body"])
    if row["superseded_by"]:
        print("\nsuperseded by letter #%d" % row["superseded_by"])


def format_learning(row):
    symptom = "%s: %s" % (row["scope"], row["slug"]) if row["scope"] else row["slug"]
    return "#%-4d %-8s %s  %s · %s · %s" % (
        row["id"], "learning", row["created_at"][:16].replace("T", " "),
        symptom, row["project"], row["author_agent"],
    )


def print_learning(row):
    print(format_learning(row))
    if row["commit_sha"]:
        print("commit: %s" % row["commit_sha"])
    for field in ("problem", "hypothesis", "resolution", "source"):
        if row[field]:
            print("\n%s: %s" % (field, row[field]))


def cmd_recent(args):
    """Newest entries, typed learnings and letters, interleaved by their
    shared ledger id."""
    conn = connect()
    project = None if args.all_projects else detect_project(args.project)
    lines = []
    if args.kind != "note":
        lines.extend(
            (r["id"], format_row(r))
            for r in list_entries(conn, kind=args.kind, project=project, limit=args.n)
        )
    if args.kind in (None, "learning"):
        lines.extend(
            (row["id"], format_learning(row))
            for row in list_learnings(conn, project=project, limit=args.n)
        )
    if args.kind in (None, "note"):
        lines.extend(
            (row["id"], format_letter(row))
            for row in list_letters(
                conn, project=project, newest_first=True, limit=args.n,
            )
        )
    conn.close()
    lines.sort(key=lambda item: item[0], reverse=True)
    if not lines:
        print("coop: no entries yet")
    for _id, line in lines[:args.n]:
        print(line)


def cmd_search(args):
    conn = connect()
    project = None if args.all_projects else detect_project(args.project)
    rows = [] if args.kind == "note" else list_entries(
        conn, query=args.query, kind=args.kind, project=project, limit=args.n,
    )
    # Retrieval-first ("what did this framework teach?"): the typed
    # learnings print ahead of the entries a query also matched.
    learnings = list_learnings(
        conn, query=args.query, project=project, limit=args.n,
    ) if args.kind in (None, "learning") else []
    letters = list_letters(
        conn, query=args.query, project=project, limit=args.n,
    ) if args.kind in (None, "note") else []
    conn.close()
    if not rows and not learnings and not letters:
        print("coop: no match for %r" % args.query)
    for row in learnings:
        print(format_learning(row))
    for r in rows:
        print(format_row(r))
    for row in letters:
        print(format_letter(row))


def cmd_show(args):
    conn = connect()
    r = get_entry(conn, args.id)
    letter = get_letter(conn, args.id)
    learning = get_learning(conn, args.id)
    conn.close()
    if r is None and letter is None and learning is None:
        sys.exit("no entry, learning or letter with id %d" % args.id)
    if learning is not None:
        print_learning(learning)
        return
    if letter is not None:
        print_letter(letter)
        if r is not None and r["kind"] == "note":
            print("\n(migrated from note entry #%d at coop schema v3; that entries "
                  "row is a frozen copy, left at status '%s')" % (r["id"], r["status"]))
            return
    if r is None:
        return
    print(format_row(r))
    print("slug: %s · machine: %s · updated: %s" % (r["slug"], r["machine"], r["updated_at"]))
    if r["commit_sha"]:
        print("commit: %s" % r["commit_sha"])
    if r["body"]:
        print("\n" + r["body"])


def cmd_letter_read(args):
    agent = detect_agent(args.agent)
    conn = connect()
    try:
        rows = read_letters(
            conn,
            agent=agent,
            project=None if args.all_projects else detect_project(args.project),
            letter_ids=args.ids,
        )
    except (LetterNotFound, ValueError) as exc:
        sys.exit(str(exc))
    finally:
        conn.close()
    if not rows:
        print("coop: no unread letters for %s" % agent)
        return
    for row in rows:
        print(format_letter(row))
        for line in (row["body"] or "(no body)").splitlines():
            print("    " + line)
    print("(now read; `coop.py letter start <id>` when you act on one, "
          "`coop.py letter ack <id> --body …` when it ships)")


def cmd_letter_move(args):
    agent = detect_agent(args.agent)
    conn = connect()
    try:
        row = transition_letter(
            conn, args.id, args.letter_cmd, agent=agent,
            body=read_body(args.body) if args.letter_cmd == "ack" else None,
            superseded_by=getattr(args, "by", None),
        )
    except (LetterNotFound, ValueError) as exc:
        sys.exit(str(exc))
    finally:
        conn.close()
    print("coop letter #%d -> %s" % (row["id"], row["status"]))


def cmd_letter_list(args):
    agent = detect_agent(args.agent)
    conn = connect()
    try:
        rows = list_letters(
            conn,
            project=None if args.all_projects else detect_project(args.project),
            status=None if args.all else ACTIVE_LETTER_STATUSES,
            addressed_to=agent,
            limit=args.n,
        )
    finally:
        conn.close()
    if not rows:
        print("coop: no letters for %s" % agent)
    for row in rows:
        print(format_letter(row))


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
            status=None if args.all else ACTIVE_MESSAGE_STATUSES,
            recipient_agent=agent,
            limit=args.n,
        )
    finally:
        conn.close()
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


def _print_brief_letters(conn, project, agent):
    """Unread letters addressed to ``agent`` (or any session) and the ones it
    is working on. A letter read but never started is counted, not listed,
    so it cannot silently drop out of every session card."""
    unread = list_letters(
        conn, project=project, status="unread", addressed_to=agent,
        limit=BRIEF_ITEM_LIMIT + 1,
    )
    working = list_letters(
        conn, project=project, status="working", worker_agent=agent,
        limit=BRIEF_ITEM_LIMIT + 1,
    )
    parked = letter_counts(conn, project=project, addressed_to=agent)["read"]
    if not (unread or working or parked):
        return

    def count(rows):  # rows is a LIMIT + 1 probe
        return "%d+" % BRIEF_ITEM_LIMIT if len(rows) > BRIEF_ITEM_LIMIT else str(len(rows))

    hint = " — `coop.py letter read` prints and marks them read" if unread else ""
    print("Letters : (%s in %s) %s unread · %s working%s" % (
        agent, project, count(unread), count(working), hint,
    ))
    for row in unread[:BRIEF_ITEM_LIMIT] + working[:BRIEF_ITEM_LIMIT]:
        print("  " + format_letter(row))
    if len(unread) > BRIEF_ITEM_LIMIT or len(working) > BRIEF_ITEM_LIMIT:
        print("  … more — `coop.py letter list`")
    if parked:
        print("  (%d read but not started — `coop.py letter list`)" % parked)


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
        # Read-only by contract: only an explicit `letter read` marks a
        # letter read, so the card lists what is waiting without moving it.
        _print_brief_letters(conn, project, agent)
        todos = list_entries(
            conn, kind="todo", status=("open", "read"), project=project,
            assignee=(agent, ""), limit=BRIEF_ITEM_LIMIT + 1,
        )
        if todos:
            print("Todos   : (%s's and unassigned, in %s)" % (agent, project))
            for r in todos[:BRIEF_ITEM_LIMIT]:
                print("  " + format_row(r))
            if len(todos) > BRIEF_ITEM_LIMIT:
                print("  … more — `coop.py recent --kind todo -n 50`")
        # A learning is typed since schema v4, no longer an entry: interleave
        # both by their shared ledger id so a fresh learning stays on the card.
        recent = [
            (r["id"], format_row(r))
            for r in list_entries(conn, project=project, limit=args.n)
        ] + [
            (row["id"], format_learning(row))
            for row in list_learnings(conn, project=project, limit=args.n)
        ]
        recent.sort(key=lambda item: item[0], reverse=True)
        if recent:
            print("Recent  : (project %s)" % project)
            for _id, line in recent[:args.n]:
                print("  " + line)
        inbox = list_messages(
            conn, project=project, status=ACTIVE_MESSAGE_STATUSES,
            recipient_agent=agent, limit=5,
        )
        if inbox:
            print("Inbox   : (agent %s)" % agent)
            for row in inbox:
                print("  " + format_message(row))
        conn.close()
        print("Rules   : per work unit → `coop.py log` BEFORE the commit,"
              " then commit+push, then `coop.py sha <id> <head>`;"
              " non-obvious lesson → `coop.py learn <symptom-slug>"
              " --problem … --resolution …`;"
              " handoff → `coop.py note` (a letter: `letter read` →"
              " `letter start <id>` → `letter ack <id> --body …`);"
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
        if name == "log":
            s.add_argument("--sha", help="git head when already known "
                           "(machine writers / retroactive records)")
        if name == "todo":
            s.add_argument("--assignee", help="who should do it (default: you)")
        return s

    for name in ("log", "todo"):
        add_writer(name)

    s = sub.add_parser(
        "learn",
        help="record a typed learning (schema v4)",
        description=(
            "Record what a framework taught when it blocked the work "
            "(Memory L6 learning_entries). The lesson names the columns - "
            "scope, slug, problem, hypothesis, resolution, source - but not "
            "which are required; Forrest requires --problem and --resolution "
            "and leaves the rest optional. '-' as a value reads that field "
            "from stdin."
        ),
    )
    s.add_argument("slug", help="the symptom's name, e.g. fts-lost-after-alter")
    s.add_argument("--problem", required=True,
                   help="required: what broke, as it showed itself")
    s.add_argument("--resolution", required=True,
                   help="required: what fixed it, and the rule to keep")
    s.add_argument("--hypothesis", default="",
                   help="optional: what you suspected on the way")
    s.add_argument("--source", default="",
                   help="optional: where the answer came from (docs, issue, commit)")
    s.add_argument("--scope", default="",
                   help="optional: the framework or language (sqlite, tauri, mps)")
    s.add_argument("--sha", help="git head when already known "
                   "(machine writers / retroactive records)")
    s.add_argument("--project")
    s.add_argument("--agent")

    s = sub.add_parser("note", help="write a handoff letter (schema v3)")
    s.add_argument("title", help="letter subject")
    s.add_argument("-b", "--body", default="", help="letter body; '-' reads stdin")
    s.add_argument("--to", help="recipient agent; omit for any future "
                   "session of the project")
    s.add_argument("--project")
    s.add_argument("--agent")

    s = sub.add_parser("assign", help="(re)assign a todo, or --unassign it")
    s.add_argument("id", type=int)
    s.add_argument("assignee", nargs="?")
    s.add_argument("--unassign", action="store_true")

    s = sub.add_parser("letter", help="Memory L6 handoff letters")
    letter_sub = s.add_subparsers(dest="letter_cmd", required=True)
    s = letter_sub.add_parser(
        "read", help="print unread letters addressed to you (or the given "
        "ids) and mark them read — the only command that does",
    )
    s.add_argument("ids", type=int, nargs="*")
    s.add_argument("--project")
    s.add_argument("--all-projects", action="store_true")
    s.add_argument("--agent")
    for name, help_text in (
        ("start", "read → working: you are acting on it"),
        ("ack", "working → done: it shipped (body required)"),
        ("abandon", "unread/read/working → abandoned"),
        ("supersede", "unread/read/working → superseded"),
    ):
        s = letter_sub.add_parser(name, help=help_text)
        s.add_argument("id", type=int)
        s.add_argument("--agent")
        if name == "ack":
            s.add_argument("-b", "--body", required=True,
                           help="what shipped; '-' reads stdin")
        if name == "supersede":
            s.add_argument("--by", type=int, help="the letter that replaces it")
    s = letter_sub.add_parser("list", help="letters addressed to you (read-only)")
    s.add_argument("-n", type=int, default=20)
    s.add_argument("--all", action="store_true",
                   help="include done/abandoned/superseded")
    s.add_argument("--all-projects", action="store_true")
    s.add_argument("--project")
    s.add_argument("--agent")

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
    kind_map = {"log": "history", "todo": "todo"}
    if args.cmd in kind_map:
        cmd_add(args, kind_map[args.cmd])
    elif args.cmd == "learn":
        cmd_learn(args)
    elif args.cmd == "note":
        cmd_note(args)
    elif args.cmd == "assign":
        cmd_assign(args)
    elif args.cmd == "letter":
        if args.letter_cmd == "read":
            cmd_letter_read(args)
        elif args.letter_cmd == "list":
            cmd_letter_list(args)
        else:
            cmd_letter_move(args)
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

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
shipped. A take from a terminal (`take`, `message-claim`) is a take the
office records: an observed session that owes an outcome within three hours,
discharged by the taker's own `done`/`release`/`abandon` (Coop Quest
"adopting-a-take", "an-outcome-is-owed"), which carry `--for-dad`, the two
or three Korean lines Dad reads of it (Master Chief
"the-worker-and-its-report"; the office requires them where the close
reaches him).

Learnings are history's typed sister table (schema v4, Memory L6
`learning_entries`): history asks what a session did, a learning what a
framework taught when it blocked the work. `learn <slug>` records one — the
slug names the symptom, `--scope` the framework or language, and typed fields
hold the problem, hypothesis, resolution and source. A learning written before
v4 stays a free-body entries row, readable and searchable as before.

Agents may also address generic mailbox messages to one another.  Coop owns
delivery, threads, artifact references and claim state; the agents own the
judgment about whether, when and how to collaborate.

Schema v5 (Coop Quest track 1, 2026-11-06) gives todos the lesson's
lifecycle — ``open → taken → done | abandoned``, ``release`` back to open, the
``open → done`` fast path, the taker and closer on the row, the first close as
the record (`take`, `release`, `done`, `abandon`) — records every todo, letter
and message transition in the append-only ``coop_events`` table (`events`),
lets a learning be superseded (`learn --supersedes`), keeps history
``(project, slug)`` unique, counts a letter's reads, keeps one live letter per
author and address, refuses a letter that points at a handoff file, and
checks every stored identity against the roster (CANONICAL_AGENT_IDS).

Ground truth lives on ONE machine (the always-on server) at
$HOME/forrest-db/coop.db — deliberately outside the Dropbox CloudStorage
tree (live SQLite + cloud sync don't mix; CLAUDE.md Dropbox caution).
Satellite writers use the registered HTTPS compatibility gateway; fenced
local replicas remain read-only. There is no persistent SSH write tunnel.
connect() enforces this: on any other machine, the default location
opens only with an office's owner file or fence (a read-only replica),
and otherwise raises LedgerUnavailable instead of creating a ledger.

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

Stdlib only. bjtCoop owns this source and exports a pinned single-file bundle
for existing callers. An adopted ledger routes mutations to its private
loopback office. Reads remain local and writes fail closed when the office
cannot acknowledge them. Isolated, unadopted ledgers retain the original
in-process storage API for migration and tests.

Env: COOP_DB_PATH (default $HOME/forrest-db/coop.db, override for tests),
     COOP_GROUND_TRUTH_HOST (explicit override; otherwise the live Forrest role),
     COOP_MACHINE (default the short hostname),
     COOP_AGENT (default agent id; CLAUDECODE=1 implies claude-code).
"""

import argparse
import functools
import inspect
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
import time
import uuid
from pathlib import Path
from urllib import error as urlerror, request as urlrequest

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
# Todo statuses. The lifecycle is Coop Quest "lifecycles-and-the-first-close":
# ``open → taken → done | abandoned``, ``release`` back to open, and the fast
# path ``open → done``. ``read`` and ``superseded`` are legacy values written
# before schema v5: still readable, listable and filterable, never written
# again (``read`` is taken like ``open``; ``superseded`` is terminal).
STATUSES = ("open", "taken", "done", "abandoned", "read", "superseded")
TODO_OPEN_STATUSES = ("open", "read")
TODO_ACTIVE_STATUSES = ("open", "taken", "read")
TODO_TERMINAL_STATUSES = ("done", "abandoned", "superseded")
TODO_LEGACY_STATUSES = ("read", "superseded")
LETTER_STATUSES = (
    "unread", "read", "working", "done", "abandoned", "superseded",
)
ACTIVE_LETTER_STATUSES = ("unread", "read", "working")
# action -> (statuses it may leave, status it enters). Anything else is an
# illegal transition and is refused, never coerced. ``ack`` closes from
# ``read`` or ``working`` ("skipping working is fine", Coop Quest
# "a-letter-carries-itself"); a later ``read`` of a live letter is no
# transition but bumps ``read_count`` (transition_letter, read_letters).
LETTER_TRANSITIONS = {
    "read": (("unread",), "read"),
    "start": (("read",), "working"),
    "ack": (("read", "working"), "done"),
    "abandon": (ACTIVE_LETTER_STATUSES, "abandoned"),
    "supersede": (ACTIVE_LETTER_STATUSES, "superseded"),
}
REVIEW_STATUSES = (
    "open", "claimed", "changes_requested", "approved", "superseded",
)
REVIEW_VERDICTS = ("approved", "changes_requested")
MESSAGE_STATUSES = ("open", "claimed", "done", "superseded")
ACTIVE_MESSAGE_STATUSES = ("open", "claimed")
# The intents a writer may put on a message. Each is read by some code:
# dispatch threads and `coop.py send`/`reply` use request/response, the
# legacy review projection peer_review/review_feedback, project work its
# three modes plus work_result/work_revision. Coop Quest
# "seven-tables-and-a-roster" dropped a column no reader processed ("a field
# nobody processes is not neutral"); here the readers exist, so the set is
# closed at write time instead, and old rows keep whatever they say.
MESSAGE_INTENTS = (
    "request", "response", "peer_review", "review_feedback", "work_result",
    "work_revision", "project_implement", "project_review", "project_refine",
)
# The roster (Coop Quest "seven-tables-and-a-roster"): one id per surface,
# checked by require_canonical_identity at every write that stores an
# identity; reads stay lenient so older spellings still list and filter.
# ``dad`` is BJT at a keyboard (the engine already reserves it: a model
# cannot act as Dad); the two terminal surfaces carry their product names;
# the WebUI Brains are ``<brain>-forrest``; ``forrest`` is the app's own
# door (coop_runtime.APP_AGENT_ID). A new surface joins this tuple first.
CANONICAL_AGENT_IDS = (
    "dad",
    "claude-code", "codex",
    "claude-forrest", "opus-forrest", "fable-forrest", "codex-forrest",
    "ollama-forrest",
    "forrest",
)
# Dispatch's receiver-to-mailbox mapping also bounds work-over settlement.
# Keep it here so the standalone compatibility export uses the same source.
DISPATCH_WORKER_MAILBOXES = {
    "claude": "claude-code", "codex": "codex", "ollama": "ollama-forrest",
}
# The six handoff-document stems the letter checkpoint knows (Coop Quest
# "a-letter-carries-itself"): a path-shaped mention is refused, a bare name
# passes.
HANDOFF_STEMS = (
    "handoff", "handover", "session-notes", "session-state",
    "new-session-reminder", "context-dump",
)
COOP_SCHEMA_VERSION = 6
BRIEF_ITEM_LIMIT = 10
# Settled client receipt files (committed / rejected) are deleted after a
# day, checked at most hourly; uncertain ones stay until reconciled (Coop
# Quest "the-owner-file-fails-closed").
RECEIPT_RETENTION_SECONDS = 86400
RECEIPT_PRUNE_INTERVAL_SECONDS = 3600

# ---------------------------------------------------------------- storage

FTS_OK = False  # set by connect(); sqlite3.Connection forbids ad-hoc attributes


class LetterNotFound(LookupError):
    """No letter carries the requested id."""


class LetterTransitionError(ValueError):
    """The letter's state or address does not admit the requested move."""


class LedgerUnavailable(RuntimeError):
    """A satellite reached the default ledger and no office fences it."""


class LifecycleError(ValueError):
    """A todo move the lifecycle does not admit: wrong state, wrong actor,
    or the row changed under the write."""


class UnknownIdentityError(ValueError):
    """A write tried to store an identity the roster does not name."""


class HandoffPointerError(ValueError):
    """A letter body points at a handoff file instead of carrying the handoff."""


@contextmanager
def _managed_work_result_guard(
    intent,
    *,
    target_message_id=None,
    fallback_message_id=None,
    environ=None,
):
    """Fence a managed result against a concurrent generation rotation."""
    # A remote writer holds this fence in the office process, through commit.
    # Holding it here too would deadlock; timing out here must not release the
    # office's guard while its write is still in flight.
    if environ is None and owner_path().exists():
        yield
        return
    environ = os.environ if environ is None else environ
    if intent != "work_result" or environ.get("FORREST_WORKSPACE") != "1":
        yield
        return
    runtime_raw = environ.get("FORREST_WORK_RUNTIME_PATH", "").strip()
    token = environ.get("FORREST_WORK_SESSION_TOKEN", "").strip()
    generation = environ.get("FORREST_WORK_GENERATION", "").strip()
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

# Created by the v5 migration (Coop Quest "seven-tables-and-a-roster":
# "coop_events — one status change of a todo, request or letter; append
# only, never deleted"). The two guard triggers make the table append-only
# at the schema: no UPDATE or DELETE succeeds, for any writer.
_COOP_EVENTS_TABLE = """
CREATE TABLE IF NOT EXISTS coop_events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  kind TEXT NOT NULL,
  entry_id INTEGER NOT NULL,
  event TEXT NOT NULL,
  from_status TEXT NOT NULL DEFAULT '',
  to_status TEXT NOT NULL DEFAULT '',
  actor TEXT NOT NULL,
  reason TEXT NOT NULL DEFAULT '',
  payload TEXT NOT NULL DEFAULT '{}',
  at TEXT NOT NULL
)
"""
_COOP_EVENTS_GUARDS = (
    "CREATE TRIGGER IF NOT EXISTS coop_events_no_update BEFORE UPDATE ON coop_events"
    " BEGIN SELECT RAISE(ABORT, 'coop_events is append-only'); END",
    "CREATE TRIGGER IF NOT EXISTS coop_events_no_delete BEFORE DELETE ON coop_events"
    " BEGIN SELECT RAISE(ABORT, 'coop_events is append-only'); END",
)

_INDEX_SCHEMA = """
CREATE INDEX IF NOT EXISTS idx_entries_kind_created ON entries(kind, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_coop_events_entry ON coop_events(kind, entry_id, id);
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
        if current < 5:
            # Coop Quest track 1 (2026-11-06): a todo keeps its taker and
            # closer, every transition is an append-only coop_events row, a
            # learning can be superseded, a letter counts its reads, and
            # history (project, slug) is unique. Every column is additive
            # with a default; the only rewrite is the slug of a later
            # duplicate history row, each with an `edit` event carrying the
            # previous slug (_dedupe_history_slugs). A v4 client ignores the
            # columns, so a rollback is `PRAGMA user_version=4`.
            for column, ddl in (
                ("taken_by", "TEXT NOT NULL DEFAULT ''"),
                ("taken_at", "TEXT NOT NULL DEFAULT ''"),
                ("closed_by", "TEXT NOT NULL DEFAULT ''"),
                ("closed_at", "TEXT NOT NULL DEFAULT ''"),
                ("result_summary", "TEXT NOT NULL DEFAULT ''"),
            ):
                if column not in _table_columns(conn, "entries"):
                    conn.execute(
                        "ALTER TABLE entries ADD COLUMN %s %s" % (column, ddl),
                    )
            for column, ddl in (
                ("read_count", "INTEGER NOT NULL DEFAULT 0"),
                ("last_read_at", "TEXT NOT NULL DEFAULT ''"),
            ):
                if column not in _table_columns(conn, "letters"):
                    conn.execute(
                        "ALTER TABLE letters ADD COLUMN %s %s" % (column, ddl),
                    )
            # A letter past unread was read at least once; the exact time is
            # known only while it is still `read` (its last update).
            conn.execute(
                "UPDATE letters SET read_count=1 WHERE read_count=0"
                " AND status IN ('read','working','done')"
            )
            conn.execute(
                "UPDATE letters SET last_read_at=updated_at"
                " WHERE last_read_at='' AND status='read'"
            )
            if "superseded_by" not in _table_columns(conn, "learning_entries"):
                conn.execute(
                    "ALTER TABLE learning_entries ADD COLUMN superseded_by"
                    " INTEGER REFERENCES learning_entries(id)"
                )
            conn.execute(_COOP_EVENTS_TABLE)
            for statement in _COOP_EVENTS_GUARDS:
                conn.execute(statement)
            _dedupe_history_slugs(conn)
            conn.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_entries_history_slug"
                " ON entries(project, slug) WHERE kind='history'"
            )
            conn.execute("PRAGMA user_version=5")
        if current < 6:
            # Master Chief `the-coop-tool-and-a-direct-request` (2026-11-08,
            # BJT "퀘스트대로"): "`add`는 그 자체로 읽히는 투두를 만들어. 범위,
            # 제목, 본문, 우선순위." One additive column with a default, so a
            # v5 client's INSERT list still fits; a rollback is only
            # `PRAGMA user_version=5`.
            if "priority" not in _table_columns(conn, "entries"):
                conn.execute("ALTER TABLE entries ADD COLUMN priority TEXT NOT NULL DEFAULT ''")
            conn.execute("PRAGMA user_version=6")
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def _dedupe_history_slugs(conn):
    """Make every later duplicate of a history (project, slug) unique.

    Coop Quest "seven-tables-and-a-roster": history_entries' (scope, slug)
    must be unique. The first row of a group keeps its slug (get_entry_by_slug
    already answered with it: the lowest id); each later one gets `-2`,
    `-3`, … and an `edit` event in coop_events carrying the previous slug,
    so the rename is history, not an overwrite. Runs inside the v5
    migration's transaction. Live ledger on 2026-11-06: five groups, ten
    rows renamed.
    """
    groups = conn.execute(
        "SELECT project, slug FROM entries WHERE kind='history'"
        " GROUP BY project, slug HAVING COUNT(*) > 1 ORDER BY project, slug",
    ).fetchall()
    for group in groups:
        rows = conn.execute(
            "SELECT id, slug FROM entries WHERE kind='history'"
            " AND project=? AND slug=? ORDER BY id",
            (group["project"], group["slug"]),
        ).fetchall()
        taken = {
            row[0] for row in conn.execute(
                "SELECT slug FROM entries WHERE kind='history' AND project=?",
                (group["project"],),
            )
        }
        suffix = 2
        for row in rows[1:]:
            while True:
                candidate = "%s-%d" % (row["slug"], suffix)
                suffix += 1
                if candidate not in taken:
                    break
            taken.add(candidate)
            conn.execute(
                "UPDATE entries SET slug=? WHERE id=?", (candidate, row["id"]),
            )
            _record_event(
                conn, "history", row["id"], "edit", "migration",
                reason="history (project, slug) made unique at coop schema v5",
                payload={"fields": ["slug"], "previous": {"slug": row["slug"]}},
            )


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


class CoopConnection(sqlite3.Connection):
    """Connection-local routing; ownership never lives in a process-global flag."""

    office_writer = False
    office_owned = False
    db_path = ""
    office_batch = False

    def execute(self, sql, parameters=()):
        if self.office_batch and sql.strip().upper().startswith("BEGIN"):
            return super().execute("SELECT 1")
        return super().execute(sql, parameters)

    def commit(self):
        if not self.office_batch:
            super().commit()

    def rollback(self):
        if not self.office_batch:
            super().rollback()


def owner_path(path=None):
    return Path(str(path or os.environ.get("COOP_DB_PATH") or DEFAULT_DB) + ".owner.json")


def _satellite_refusal(db_path):
    truth = ground_truth_host()
    return LedgerUnavailable(
        "COOP LEDGER UNAVAILABLE: %s is a satellite (ground truth: %s) and no "
        "Coop office fences %s, so this client will not create, read or write "
        "a local ledger. Restore a fenced replica or run Coop on %s."
        % (detect_machine(), truth, db_path, truth)
    )


def connect(path=None, *, _office_writer=False):
    global FTS_OK
    db_path = os.path.abspath(path or os.environ.get("COOP_DB_PATH") or DEFAULT_DB)
    canonical = os.path.realpath(db_path) == os.path.realpath(DEFAULT_DB)
    if _office_writer and canonical and not is_local_office_owner():
        raise _satellite_refusal(db_path)
    # The fleet holds fenced, read-only replicas of the one office's ledger
    # (Memory L10). A client on a satellite that finds neither an owner file
    # nor a fence at the default location would otherwise start a private
    # ledger nobody reads; it fails closed instead, like an absent office.
    satellite = (
        not _office_writer
        and not is_local_office_owner()
        and os.path.realpath(db_path) == os.path.realpath(DEFAULT_DB)
        and not owner_path(db_path).exists()
    )
    if satellite and not os.path.exists(db_path):
        raise _satellite_refusal(db_path)
    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
    conn = sqlite3.connect(db_path, factory=CoopConnection)
    try:
        conn.db_path = db_path
        conn.office_writer = _office_writer
        conn.row_factory = sqlite3.Row
        owned = owner_path(db_path).exists() or conn.execute(
            "SELECT 1 FROM sqlite_master WHERE name='coop_office' AND type='table'",
        ).fetchone() is not None
        conn.office_owned = owned
        if satellite and not owned:
            raise _satellite_refusal(db_path)
        # Every protected row mutation, including a statement prepared before
        # cutover, must resolve this function on THAT connection. Legacy clients
        # have no such function and SQLite rejects their write.
        conn.create_function("coop_writer_authorized", 0,
            lambda: int(conn.office_writer and (not canonical or is_local_office_owner())))
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA busy_timeout=5000")
        if owned and not _office_writer:
            conn.execute("PRAGMA query_only=ON")
            FTS_OK = conn.execute(
                "SELECT count(*) FROM sqlite_master WHERE type='table' "
                "AND name IN ('entries_fts','learning_entries_fts')",
            ).fetchone()[0] == 2
            return conn
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA busy_timeout=5000")
        # Tables must exist before a versionless DB can be introspected, while
        # indexes must wait until migrations have supplied every referenced
        # column.
        conn.executescript(_TABLE_SCHEMA)
        _migrate_schema(conn)
        conn.executescript(_INDEX_SCHEMA)
        try:  # FTS5 ships with macOS/homebrew sqlite; degrade to LIKE search without it
            _ensure_fts_index(conn, "entries_fts", _FTS_SCHEMA)
            _ensure_fts_index(conn, "learning_entries_fts", _LEARNING_FTS_SCHEMA)
            FTS_OK = True
        except sqlite3.OperationalError:
            FTS_OK = False
        _migrate_legacy_reviews(conn)
        return conn
    except BaseException:
        conn.close()
        raise


# ---------------------------------------------------------------- context


def detect_machine():
    return os.environ.get("COOP_MACHINE") or socket.gethostname().split(".")[0]


def _family_ledger():
    """Read one canonical role declaration; the export is only its fallback."""
    explicit = os.environ.get("FORREST_FAMILY_LEDGER", "").strip()
    live = Path(explicit).expanduser() if explicit else Path.home() / "bjtPersonalProjects/bjtForrest/config/family-ledger.json"
    for path in (live, Path(__file__).with_name("family-ledger.json")):
        try:
            value = json.loads(path.read_text())
            roles, machines = value["roles"], value["machines"]
            if (type(value.get("schema_version")) is int and value["schema_version"] == 1
                    and isinstance(roles, dict) and set(roles) == {"forrest", "workbench"}
                    and isinstance(machines, dict)
                    and all(isinstance(role, str) and role in machines for role in roles.values())
                    and all(isinstance(machine, dict) and isinstance(machine.get("hostnames"), list)
                        and machine["hostnames"] and all(isinstance(name, str) and name for name in machine["hostnames"])
                        for machine in machines.values())):
                return value
        except (OSError, ValueError, KeyError, TypeError):
            pass
    raise LedgerUnavailable("Coop cannot resolve the canonical Forrest role")


def ground_truth_host():
    override = os.environ.get("COOP_GROUND_TRUTH_HOST", "").strip()
    return override or _family_ledger()["roles"]["forrest"]


def is_ground_truth_machine():
    if os.environ.get("COOP_GROUND_TRUTH_HOST", "").strip():
        return detect_machine().casefold().rstrip(".") == ground_truth_host().casefold().rstrip(".")
    family = _family_ledger()
    role = family["roles"]["forrest"]
    names = {role, *family["machines"][role].get("hostnames", [])}
    return detect_machine().casefold().rstrip(".") in {name.casefold().rstrip(".") for name in names}



def ark_profile():
    """A local independent fork; canonical role ownership remains unchanged."""
    family = _family_ledger()
    machine = detect_machine().casefold().rstrip('.')
    matches = [name for name, row in family['machines'].items()
               if machine in {value.casefold().rstrip('.') for value in [name, *row.get('hostnames', [])]}]
    if len(matches) != 1 or matches[0] == family['roles']['forrest']:
        return None
    path = Path(os.environ.get('FORREST_HOST_PROFILE', Path.home()/'.config/forrest/host-profile.json')).expanduser()
    try:
        value = json.loads(path.read_text())
    except FileNotFoundError:
        if os.environ.get('FORREST_ARK_BRANCH'):
            raise LedgerUnavailable('running Ark lost its branch profile') from None
        return None
    except (OSError, ValueError):
        raise LedgerUnavailable('local host profile is unreadable') from None
    if not isinstance(value, dict):
        raise LedgerUnavailable('local host profile is invalid')
    if value.get('ark') is not True:
        if os.environ.get('FORREST_ARK_BRANCH'):
            raise LedgerUnavailable('running Ark lost its branch authority')
        return None
    if (value.get('version') != 1
            or value.get('machine') != matches[0] or not isinstance(value.get('branch_id'), str)
            or not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_-]{7,127}', value['branch_id'])
            or not isinstance(value.get('members', {}), dict) or not isinstance(value.get('writers', {}), dict)
            or os.environ.get('FORREST_ARK_BRANCH', value['branch_id']) != value['branch_id']):
        raise LedgerUnavailable('local Ark profile does not own this branch')
    return value


def is_local_office_owner():
    profile = ark_profile()
    return is_ground_truth_machine() or bool(profile and profile.get('members', {}).get('coop') == 'run')


def require_family_writer(db_path, *, outward=False, now=None, writer="coop-native"):
    """An Ark owns only its local fork; outward copied work defaults held."""
    profile = ark_profile()
    if profile:
        if profile.get('members', {}).get('coop') != 'run':
            raise LedgerUnavailable('Ark Coop member is held')
        if outward and profile.get('writers', {}).get(writer) != 'run':
            raise LedgerUnavailable('Ark outward writer is held: ' + writer)
        return
    if os.path.realpath(db_path) != os.path.realpath(DEFAULT_DB):
        return  # Separate ledgers/tests are not replicas of the canonical DB.
    if not is_ground_truth_machine():
        raise _satellite_refusal(db_path)
    if not outward:
        return
    family = _family_ledger()
    if family['roles'].get('forrest') == family['roles'].get('workbench'):
        return
    path = Path(os.environ.get('FORREST_LEDGER_CONFIRMATION', Path.home()/'.cache/forrest/ledger-confirm.json')).expanduser()
    try:
        raw = json.loads(path.read_text())
    except FileNotFoundError:
        return  # An unenrolled current installation keeps its existing behavior.
    except (OSError, ValueError):
        raise LedgerUnavailable('Coop notifier holds: origin confirmation unreadable') from None
    try:
        stamp=dt.datetime.fromisoformat(raw['confirmed_at'].replace('Z','+00:00'))
        age=((now or dt.datetime.now(dt.timezone.utc))-stamp).total_seconds()
        if stamp.tzinfo is None or not 0 <= age <= 1800:
            raise ValueError('expired')
    except (ValueError, KeyError, TypeError, AttributeError):
        raise LedgerUnavailable('Coop notifier holds: no valid origin confirmation within 30 minutes') from None


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


# ---------------------------------------------------------------- a take's origin
# Coop Quest "adopting-a-take": a CLI call declares the chat it runs in and
# the Mac it ran on, so the engine records a terminal take against that chat
# "without asking the model to remember anything". An exported reference
# always wins over detection. Claude Code registers every live chat as
# `~/.claude/sessions/<pid>.json` (with its `sessionId`), and a CLI call typed
# in that chat is a descendant of that process, so the parent chain is walked
# up to MAX_ANCESTORS levels; Codex has no per-process registry and exports
# `codex:<thread id>` itself.

SESSION_ENV = "FORREST_COOP_SESSION"
HOST_ENV = "FORREST_COOP_HOST"
MAX_ANCESTORS = 16


def _parent(pid):
    """The parent pid through `ps -o ppid=`, or None when it cannot be read."""
    try:
        out = subprocess.run(
            ["/bin/ps", "-o", "ppid=", "-p", str(pid)],
            capture_output=True, text=True, timeout=2,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    return int(out) if out.isdigit() else None


def detect_claude_session(registry=None, pid=None, parent=None):
    """`claude:<sessionId>` of the Claude Code chat this process runs in, else None.

    Coop Quest "adopting-a-take": "`detect_claude_session`은 `ps -o ppid=`로
    부모 사슬을 거슬러 올라가. 최대 열여섯 단계, 등록 파일이 있는 pid를 만날
    때까지. 그리고 `claude:<sessionId>`를 돌려줘. 훅도, 플래그도, 프롬프트 속
    지시도 없어." ``registry`` and ``parent`` are injectable so a test walks
    a fake process tree against a fake registry; the walk stops at pid 1.
    """
    registry = Path(registry) if registry else Path.home() / ".claude" / "sessions"
    if not registry.is_dir():
        return None
    parent = parent or _parent
    pid = os.getpid() if pid is None else pid
    for _ in range(MAX_ANCESTORS):
        entry = registry / ("%d.json" % pid)
        if entry.is_file():
            try:
                session = json.loads(entry.read_text()).get("sessionId")
            except (OSError, ValueError, AttributeError):
                return None
            return "claude:" + session if isinstance(session, str) and session else None
        pid = parent(pid)
        if not pid or pid <= 1:
            return None
    return None


def detect_session():
    """The chat reference a take declares: the exported one, else the detected one."""
    exported = os.environ.get(SESSION_ENV, "").strip()
    return exported or detect_claude_session() or ""


def fleet_label(machine=None):
    """This Mac's fleet label from the family ledger, else its own short name."""
    own = machine or detect_machine()
    name = own.casefold().rstrip(".")
    try:
        family = _family_ledger()
    except LedgerUnavailable:
        return own
    for label, row in family.get("machines", {}).items():
        names = {label, *row.get("hostnames", [])}
        if name in {value.casefold().rstrip(".") for value in names}:
            return row.get("fleet_label") or label
    return own


def declared_host():
    """The Mac the command ran on: `FORREST_COOP_HOST`, else the fleet label.

    Coop Quest "adopting-a-take": "ssh로 들어가서 쳤다면, 밝히는 건 호출한 쪽
    맥이 아니라 명령이 떨어진 맥이야."
    """
    return os.environ.get(HOST_ENV, "").strip() or fleet_label()


def declare_origin():
    """What a take declares about itself; `_office_writer` fills it on the way
    to the office: the chat reference (or ''), the Mac and the working directory."""
    return {"session": detect_session(), "host": declared_host(), "workspace": os.getcwd()}


def now_iso():
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def make_slug(title, created):
    stem = re.sub(r"[^0-9A-Za-z가-힣]+", "-", title.lower()).strip("-")[:48].strip("-")
    return "%s-%s" % (created[:10], stem or "entry")


def read_body(value):
    if value == "-":
        return sys.stdin.read().strip()
    return value or ""


def _field(row, name, default=""):
    """A column a ledger older than this client may not carry yet."""
    return row[name] if name in row.keys() else default


# ---------------------------------------------------------------- checkpoints
# Coop Quest "seven-tables-and-a-roster" and "a-letter-carries-itself": a
# rule the schema cannot check lives in the helper every write passes
# through, refuses before anything is written, and carries the fix.


def require_canonical_identity(value, *, role="actor"):
    """Refuse to store ``value`` as an identity unless the roster names it.

    Coop Quest "seven-tables-and-a-roster": identity columns are plain TEXT,
    so the roster is checked at the write gate, in the helpers every write
    passes through; reads stay lenient so every old spelling still lists.
    ``role`` names the field the value would fill (actor, assignee,
    recipient). The refusal carries the fix — the field, the roster and the
    derivation rule — so the next correct call is in the message.
    """
    text = value.strip() if isinstance(value, str) else value
    if text in CANONICAL_AGENT_IDS:
        return text
    hint = ""
    if text in ("human", "unknown", "", None):
        hint = (
            "\n  this session has no identity: pass --agent <id> or set"
            " COOP_AGENT=<id> (Claude Code sets claude-code itself; BJT at a"
            " keyboard is dad)"
        )
    elif text == "any":
        hint = (
            "\n  'any' is not an identity: a todo for anyone has no assignee,"
            " a letter for anyone has no --to"
        )
    raise UnknownIdentityError(
        "refusing to write %r as the %s: not a canonical coop identity.%s\n"
        "  canonical ids: %s\n"
        "  rule: one id per surface — terminal CLIs by product name"
        " (claude-code, codex), WebUI Brains <brain>-forrest, the app's own"
        " door forrest, BJT dad; a new surface joins CANONICAL_AGENT_IDS in"
        " bjtCoop coop/compat.py first."
        % (value, role, hint, ", ".join(CANONICAL_AGENT_IDS))
    )


def assert_no_handoff_pointer(body):
    """Refuse a letter body that points at a handoff file.

    Coop Quest "a-letter-carries-itself": the body IS the handoff. Each of
    HANDOFF_STEMS is looked for with `.md`, case-insensitively; only a
    path-shaped mention (a `/` or `~` right before the stem) is refused — a
    bare name is talk about the pattern and passes. There is deliberately
    no override: the caller who would reach for one is the pressed session
    this checkpoint exists for.
    """
    lowered = (body or "").lower()
    for stem in HANDOFF_STEMS:
        needle = stem + ".md"
        start = 0
        while (hit := lowered.find(needle, start)) != -1:
            start = hit + 1
            if hit > 0 and lowered[hit - 1] in "/~":
                raise HandoffPointerError(
                    "refusing to write a letter that points at '%s.md' — the"
                    " letter body IS the handoff. Put the state, the open work"
                    " and the constraints in the body; a path may locate"
                    " material too big for the body, never the handoff itself."
                    % stem
                )


def _record_event(conn, kind, entry_id, event, actor, *, from_status="",
                  to_status="", reason="", payload=None):
    """Append one coop_events row inside the caller's write transaction.

    ``kind`` is todo, history, letter, learning or message; ``entry_id`` the
    row's id in that kind's id space (messages have their own). Nothing
    updates or deletes a row here: the table's guard triggers refuse both.
    """
    conn.execute(
        "INSERT INTO coop_events (kind, entry_id, event, from_status,"
        " to_status, actor, reason, payload, at) VALUES (?,?,?,?,?,?,?,?,?)",
        (
            kind, int(entry_id), event, from_status or "", to_status or "",
            actor, reason or "",
            json.dumps(payload or {}, ensure_ascii=False, sort_keys=True,
                       separators=(",", ":")),
            now_iso(),
        ),
    )


def list_events(conn, entry_id, *, kind=None, limit=200):
    """Every recorded transition of one id, oldest first (read-only).

    Entries, letters and learnings share one id space, messages have their
    own, so without ``kind`` a message's events print beside an entry's
    with the kind column telling them apart.
    """
    where, params = ["entry_id=?"], [int(entry_id)]
    if kind is not None:
        where.append("kind=?")
        params.append(kind)
    return conn.execute(
        "SELECT * FROM coop_events WHERE " + " AND ".join(where)
        + " ORDER BY id LIMIT ?",
        [*params, limit],
    ).fetchall()


def event_payload(row):
    try:
        value = json.loads(row["payload"] or "{}")
    except (TypeError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


# ---------------------------------------------------------------- library
# Every SQL statement lives in this storage API.  The CLI and backend route
# both call these functions (backend/app/routes/coop.py loads this module by
# path), so schema, filtering, ranking and defaults stay single-source.


def add_entry(conn, kind, title, body="", project=None, agent=None,
              machine=None, slug=None, commit_sha=None, assignee=None,
              priority=None):
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
    # MC `the-coop-tool-and-a-direct-request`: a todo's fourth field is its
    # priority, a short word of the writer's (the lesson names no scale).
    if priority is not None and kind != "todo":
        raise ValueError("only a todo has a priority")
    final_priority = (priority or "").strip() if kind == "todo" else ""
    if len(final_priority) > 24:
        raise ValueError("priority is a short word (at most 24 characters)")
    created = now_iso()
    author = require_canonical_identity(agent or detect_agent())
    status = "open" if kind == "todo" else ""
    final_assignee = (assignee or "").strip() if kind == "todo" else ""
    if final_assignee:
        require_canonical_identity(final_assignee, role="assignee")
    final_slug = slug or make_slug(title, created)
    scope = project or detect_project()
    try:
        cur = conn.execute(
            "INSERT INTO entries (kind, project, agent, machine, title, body, slug, status,"
            " commit_sha, assignee, priority, created_at, updated_at)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (kind, scope, author,
             machine or detect_machine(), title, body or "", final_slug, status,
             commit_sha or "", final_assignee, final_priority, created, created),
        )
    except sqlite3.IntegrityError as exc:
        # history (project, slug) is unique since schema v5; a failed INSERT
        # must not leave Python's implicit transaction open on this
        # connection (CLAUDE.md: a 10 ms "database is locked").
        conn.rollback()
        existing = get_entry_by_slug(conn, scope, final_slug)
        if kind == "history" and existing is not None:
            raise ValueError(
                "history slug %r already exists in %s as coop #%d"
                % (final_slug, scope, existing["id"])
            ) from exc
        raise
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


def set_status(conn, entry_id, new_status, *, agent=None, reason=None,
               result=None, commit_sha=None, force=False, for_dad=None):
    """The legacy status door, routed to the lifecycle verbs; True if a todo moved.

    ``open`` releases a take (``reason`` required), ``taken`` takes,
    ``done`` closes (or amends a closed row's outcome) and ``abandoned``
    gives up (``reason`` required). ``read`` and ``superseded`` are legacy
    values: readable and listable, never written again. Returns False only
    when the id is not a todo; a move the lifecycle refuses raises
    LifecycleError. A note became a letter with its own state machine in
    schema v3 (``transition_letter``); its frozen entries row is never
    rewritten. ``for_dad`` rides to the verbs (see ``done_todo``).
    """
    if new_status not in STATUSES:
        raise ValueError("status must be one of: %s" % ", ".join(STATUSES))
    if new_status in TODO_LEGACY_STATUSES:
        raise LifecycleError(
            "%r is a legacy todo status, readable but never written again:"
            " a todo moves with take, release, done or abandon" % new_status
        )
    row = get_entry(conn, entry_id)
    if row is None or row["kind"] != "todo":
        return False
    if new_status == "open":
        release_todo(conn, entry_id, agent=agent, reason=reason or "",
                     for_dad=for_dad)
    elif new_status == "taken":
        take_todo(conn, entry_id, agent=agent, force=force)
    elif new_status == "done":
        done_todo(conn, entry_id, agent=agent, result=result,
                  commit_sha=commit_sha, force=force, for_dad=for_dad)
    else:
        abandon_todo(conn, entry_id, agent=agent, reason=reason or "",
                     force=force, for_dad=for_dad)
    return True


def _todo_row(conn, entry_id):
    """The todo row, inside the caller's write transaction, or a refusal
    that explains a letter id (the two share one id space)."""
    row = get_entry(conn, entry_id)
    if row is None or row["kind"] != "todo":
        raise LifecycleError(_not_a_todo(conn, entry_id))
    return row


def _letters_mentioning(conn, entry_id):
    """Ids of live letters whose subject or body says ``#<entry_id>``.

    Coop Quest "a-letter-carries-itself": marking a todo done returns the
    live letters that mention it, so the CLI can tell the session to ack
    the letter that asked for the work.
    """
    like = "%#" + str(int(entry_id)) + "%"
    pattern = re.compile(r"(?<![0-9#])#%d(?![0-9])" % int(entry_id))
    rows = conn.execute(
        "SELECT id, subject, body FROM letters"
        " WHERE status IN ('unread','read','working')"
        " AND (subject LIKE ? OR body LIKE ?) ORDER BY id",
        (like, like),
    ).fetchall()
    return [
        row["id"] for row in rows
        if pattern.search(row["subject"] or "") or pattern.search(row["body"] or "")
    ]


def take_todo(conn, entry_id, *, agent=None, force=False, origin=None):
    """``open → taken`` as one conditional UPDATE; returns the row.

    Coop Quest "lifecycles-and-the-first-close": inside one BEGIN IMMEDIATE
    the read refuses a terminal row (never re-taken, ``force`` included) and
    a foreign take without ``force``; the UPDATE's WHERE decides
    open-or-mine and its rowcount is the verdict (0 rows: changed under this
    take — re-read and retry). Re-taking one's own row only refreshes
    ``taken_at``. A forced foreign take is recorded as a ``takeover`` event
    carrying the previous owner and time, never silently. The legacy
    ``read`` status is taken like ``open``.

    ``origin`` is what the taking CLI declared about itself
    (`declare_origin`: chat reference, Mac, working directory; Coop Quest
    "adopting-a-take"). The domain keeps the reference and the Mac in the
    take's event; the office's engine records the session, binding and
    outcome obligation the take owes in this same transaction.
    """
    actor = require_canonical_identity(agent or detect_agent())
    conn.execute("BEGIN IMMEDIATE")
    try:
        row = _todo_row(conn, entry_id)
        if row["status"] in TODO_TERMINAL_STATUSES:
            raise LifecycleError(
                "todo #%d is %r — terminal rows are never re-taken"
                % (entry_id, row["status"])
            )
        foreign = (
            row["status"] == "taken"
            and bool(row["taken_by"])
            and row["taken_by"] != actor
        )
        if foreign and not force:
            raise LifecycleError(
                "todo #%d is already taken by %s (since %s) — --force to take"
                " it over" % (entry_id, row["taken_by"], row["taken_at"])
            )
        stamp = now_iso()
        if force:
            guard = "status IN ('open','read','taken')"
            params = (actor, stamp, stamp, entry_id)
        else:
            guard = "(status IN ('open','read') OR taken_by=?)"
            params = (actor, stamp, stamp, entry_id, actor)
        cur = conn.execute(
            "UPDATE entries SET status='taken', taken_by=?, taken_at=?,"
            " updated_at=? WHERE id=? AND kind='todo' AND " + guard,
            params,
        )
        if cur.rowcount == 0:
            raise LifecycleError(
                "todo #%d changed under this take — re-read and retry" % entry_id
            )
        payload = (
            {"previous": row["taken_by"], "previous_at": row["taken_at"]}
            if foreign else {}
        )
        payload.update(_declared(origin))
        _record_event(
            conn, "todo", entry_id, "takeover" if foreign else "take", actor,
            from_status=row["status"], to_status="taken",
            payload=payload or None,
        )
        conn.commit()
        return get_entry(conn, entry_id)
    except Exception:
        conn.rollback()
        raise


def _declared(origin):
    """The chat reference and the Mac a take declared, for its event payload."""
    if not isinstance(origin, dict):
        return {}
    return {key: origin[key] for key in ("session", "host")
            if isinstance(origin.get(key), str) and origin[key].strip()}


def release_todo(conn, entry_id, *, agent=None, reason, evidence=None,
                 for_dad=None):
    """``taken → open`` with a recorded reason; returns the row.

    Coop Quest "lifecycles-and-the-first-close": there is no auto-reap. A
    stale take is shown ("[taken by X, Nh ago]") and released only by a
    named decision — the taker, BJT or the chief — with a reason, through
    this verb, and the event log keeps who decided. ``for_dad`` is Dad's
    Korean lines, carried to the office (see ``done_todo``).
    """
    actor = require_canonical_identity(agent or detect_agent())
    text = (reason or "").strip()
    if not text:
        raise ValueError(
            "a release needs a reason (--reason): releasing a take is a"
            " named decision, never a timer"
        )
    conn.execute("BEGIN IMMEDIATE")
    try:
        row = _todo_row(conn, entry_id)
        if row["status"] != "taken":
            raise LifecycleError(
                "todo #%d is %s, not taken — nothing to release"
                % (entry_id, row["status"])
            )
        cur = conn.execute(
            "UPDATE entries SET status='open', taken_by='', taken_at='',"
            " updated_at=? WHERE id=? AND status='taken'",
            (now_iso(), entry_id),
        )
        if cur.rowcount == 0:
            raise LifecycleError(
                "todo #%d changed under this release — re-read and retry" % entry_id
            )
        payload = {"previous": row["taken_by"], "previous_at": row["taken_at"]}
        if evidence:
            payload["evidence"] = evidence
        _record_event(
            conn, "todo", entry_id, "release", actor, from_status="taken",
            to_status="open", reason=text, payload=payload,
        )
        conn.commit()
        return get_entry(conn, entry_id)
    except Exception:
        conn.rollback()
        raise


def _amend_outcome(conn, kind, row, *, result_summary, commit_sha, actor):
    """Amend a closed row's outcome; its closer, time and state never change.

    Coop Quest "lifecycles-and-the-first-close": a second ``done`` may fill
    or correct ``result_summary`` and ``commit_sha`` only, as an ``edit``
    event carrying the amender and the previous values; an identical repeat
    writes nothing. Returns the sorted names of the fields that changed.
    """
    changes = {
        name: value for name, value in
        (("result_summary", result_summary), ("commit_sha", commit_sha))
        if value is not None and value != row[name]
    }
    if not changes:
        return []
    conn.execute(
        "UPDATE entries SET result_summary=COALESCE(?, result_summary),"
        " commit_sha=COALESCE(?, commit_sha), updated_at=? WHERE id=?",
        (changes.get("result_summary"), changes.get("commit_sha"), now_iso(),
         row["id"]),
    )
    fields = sorted(changes)
    _record_event(
        conn, kind, row["id"], "edit", actor,
        from_status=row["status"], to_status=row["status"],
        payload={"fields": fields,
                 "previous": {name: row[name] for name in fields}},
    )
    return fields


def _message_envelope_states(conn, todo_id, message_id):
    """Prove carrier kinds/states from payloads; None means no safe proof.

    Followups carry message_id; task/reply_review contracts carry reply_ids
    and replies; invitations carry reply_review.replies. Inspect every
    generation, including a replacement envelope after an explicit resend.
    A malformed payload on this work cannot prove absence of another carrier.
    """
    tables = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    )}
    if not {"office_envelopes", "office_dispatches"} <= tables:
        return None
    carriers = []
    for envelope in conn.execute(
        "SELECT e.kind,e.payload,e.state FROM office_envelopes e"
        " JOIN office_dispatches d ON d.id=e.dispatch_id WHERE d.item_id=? ORDER BY e.rowid",
        (todo_id,),
    ):
        try:
            payload = json.loads(envelope["payload"])
            if envelope["kind"] == "followup":
                ids = [payload["message_id"]]
            else:
                if envelope["kind"] == "invitation":
                    payload = payload.get("reply_review", {})
                elif envelope["kind"] not in ("task", "reply_review"):
                    return None
                ids = payload.get("reply_ids", []) + [r["id"] for r in payload.get("replies", [])]
            if any(type(mid) is not int for mid in ids):
                return None
        except (ValueError, TypeError, KeyError, AttributeError):
            return None
        if message_id in ids:
            carriers.append({"kind": envelope["kind"], "state": envelope["state"]})
    return carriers or None


def _settle_finished_messages(conn, *, actor, verb, todo_id=None, left_open=None):
    """Settle proven work threads inside the caller's close/start transaction.

    Our messages merge Quest requests and replies. Applying L31's envelope
    rule to these rows is the Chief's adaptation: superseded means work-over,
    not that a reply was read or answered (L25). No outcome is fabricated.
    office_work is the durable link; prose, subjects and dispatch age are not.
    Ambiguous shared threads and broken/cross-thread reply chains stay open.
    A standalone legacy ledger without office_work has no proven links.
    Only a dispatch worker mailbox matching the root is ours to settle.
    A Chief, Soul or Dad take never makes that mailbox ours. L03 preserves the
    first close; L26 keeps post-close conversations deliverable. Unknown
    close times use the stricter post-close carrier test. Exclusions are
    listed in left_open if supplied.
    """
    if not conn.in_transaction:
        raise RuntimeError("message settlement needs the close/start transaction")
    actor = require_canonical_identity(actor)
    if not conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='office_work'"
    ).fetchone():
        return []
    candidates = conn.execute(
        "WITH RECURSIVE linked(todo_id,root_id,thread_id,recipient,closed_at,message_id) AS ("
        " SELECT w.item_id,m.id,m.thread_id,m.recipient_agent,"
        " COALESCE(NULLIF(e.closed_at,''),(SELECT at FROM coop_events c"
        " WHERE c.kind='todo' AND c.entry_id=e.id"
        " AND c.to_status IN ('done','abandoned') AND c.from_status!=c.to_status"
        " ORDER BY c.id LIMIT 1)),m.id FROM office_work w"
        " JOIN entries e ON e.id=w.item_id JOIN messages m ON m.id=w.message_id"
        " WHERE e.kind='todo' AND e.status IN ('done','abandoned')"
        " AND (? IS NULL OR e.id=?) AND NOT EXISTS ("
        " SELECT 1 FROM office_work other JOIN messages r ON r.id=other.message_id"
        " WHERE r.thread_id=m.thread_id AND other.item_id!=w.item_id)"
        " UNION SELECT l.todo_id,l.root_id,l.thread_id,l.recipient,l.closed_at,m.id FROM linked l"
        " JOIN messages m ON m.reply_to=l.message_id AND m.thread_id=l.thread_id)"
        " SELECT m.*,l.todo_id,l.root_id,l.recipient,"
        " julianday(l.closed_at) AS close_day,julianday(m.created_at) AS message_day FROM linked l"
        " JOIN messages m ON m.id=l.message_id"
        " WHERE m.status IN ('open','claimed') ORDER BY m.id",
        (todo_id, todo_id),
    ).fetchall()
    has_obligations = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='office_obligations'"
    ).fetchone() is not None
    stamp = now_iso()
    settled = []
    for row in candidates:
        reason = None
        carriers = None
        if has_obligations and conn.execute(
            "SELECT 1 FROM office_obligations WHERE message_id=? AND status!='closed'",
            (row["id"],),
        ).fetchone():
            reason = "owed-claim"
        elif row["recipient_agent"] != row["recipient"]:
            reason = "different-recipient"
        elif row["recipient_agent"] not in DISPATCH_WORKER_MAILBOXES.values():
            reason = "non-worker-recipient"
        elif row["close_day"] is not None and row["message_day"] is None:
            reason = "unproven-message-time"
        elif row["close_day"] is None or row["message_day"] > row["close_day"]:
            carriers = _message_envelope_states(conn, row["todo_id"], row["id"])
            if carriers is None:
                reason = "unproven-envelope-link"
            # These are the terminal states used by dispatch.pending_envelopes.
            # never_delivered is retryable in place and is deliberately absent.
            elif any(carrier["state"] not in ("delivered", "processed", "settled", "withdrawn", "undelivered") for carrier in carriers):
                reason = "deliverable-envelope"
        if reason:
            if left_open is not None:
                left_open.append({"message_id": row["id"], "todo_id": row["todo_id"], "reason": reason})
            continue
        conn.execute(
            "UPDATE messages SET status='superseded',claimed_by='',updated_at=?"
            " WHERE id=? AND status IN ('open','claimed')", (stamp, row["id"]),
        )
        _record_event(
            conn, "message", row["id"], "supersede", actor,
            from_status=row["status"], to_status="superseded", reason="work-over",
            payload={"todo_id": row["todo_id"], "verb": verb,
                     "root_message_id": row["root_id"],
                     # Keep stored states verbatim: settled is not delivered.
                     **({"carriers": carriers} if carriers is not None else {}),
                     **({"close_time": "unproven"} if row["close_day"] is None else {})},
        )
        settled.append(row["id"])
    return settled


def done_todo(conn, entry_id, *, agent=None, result=None, commit_sha=None,
              force=False, evidence=None, reply_outcomes=None, for_dad=None):
    """Close a todo as done; the first close is the record.

    Coop Quest "lifecycles-and-the-first-close": ``open → done`` is the
    legal fast path (an implicit take, so the row names its doer);
    ``taken → done`` only by the taker unless ``force``. A second ``done``
    goes to _amend_outcome (result_summary and commit_sha only, as an
    ``edit`` event; closer, time and status never change). An abandoned
    todo — or a legacy superseded one — becomes done only with ``force``
    (BJT's call), and that ``done`` event carries ``previous_status`` for
    ever.

    ``reply_outcomes`` answers the replies on the todo's thread, one
    ``{message_id, disposition: applied|not_applied, reason}`` each (Coop
    Quest "every-reply-answered": "답은 네 영수증에 붙어 저장되고"); the
    domain stores them in the ``done`` event, the office's engine validates
    them against the thread when the todo is bound.

    ``for_dad`` is what Dad reads of the close — "아빠한테 쓰는 한국어 두세
    줄, 많아야 네 줄" (Master Chief "the-worker-and-its-report") — carried
    through the office's engine, which requires it when the close reports
    taken work and writes his row (its ``_dad_line``); the domain itself
    writes no row for Dad and keeps the English ``result`` as the record.

    Returns ``{"todo": row, "changed": bool, "amended": [fields],
    "letters": [ids]}`` — the ids are the live letters that mention
    ``#<id>``, so the CLI can name the letter that asked for this work.
    """
    actor = require_canonical_identity(agent or detect_agent())
    summary = (result or "").strip()
    sha = (commit_sha or "").strip()
    conn.execute("BEGIN IMMEDIATE")
    try:
        row = _todo_row(conn, entry_id)
        amended = []
        changed = False
        if row["status"] == "done":
            amended = _amend_outcome(
                conn, "todo", row, result_summary=summary or None,
                commit_sha=sha or None, actor=actor,
            )
        else:
            if row["status"] in TODO_TERMINAL_STATUSES and not force:
                raise LifecycleError(
                    "todo #%d is %s — the first close is the record;"
                    " --force (BJT's call) records it done with its previous"
                    " status in the event" % (entry_id, row["status"])
                )
            foreign = (
                row["status"] == "taken"
                and bool(row["taken_by"])
                and row["taken_by"] != actor
            )
            if foreign and not force:
                raise LifecycleError(
                    "todo #%d is taken by %s — only the taker closes it"
                    " (--force to override)" % (entry_id, row["taken_by"])
                )
            stamp = now_iso()
            if row["status"] == "taken" and row["taken_by"]:
                taken_by, taken_at = row["taken_by"], row["taken_at"]
            else:
                taken_by, taken_at = actor, stamp
            cur = conn.execute(
                "UPDATE entries SET status='done', closed_by=?, closed_at=?,"
                " taken_by=?, taken_at=?,"
                " result_summary=CASE WHEN ?='' THEN result_summary ELSE ? END,"
                " commit_sha=CASE WHEN ?='' THEN commit_sha ELSE ? END,"
                " updated_at=? WHERE id=? AND status=?",
                (actor, stamp, taken_by, taken_at, summary, summary, sha, sha,
                 stamp, entry_id, row["status"]),
            )
            if cur.rowcount == 0:
                raise LifecycleError(
                    "todo #%d changed under this close — re-read and retry"
                    % entry_id
                )
            payload = {}
            if row["status"] in TODO_TERMINAL_STATUSES:
                payload["previous_status"] = row["status"]
            if foreign:
                payload["previous_taker"] = row["taken_by"]
            if evidence:
                payload["evidence"] = evidence
            if reply_outcomes:
                payload["reply_outcomes"] = reply_outcomes
            _record_event(
                conn, "todo", entry_id, "done", actor,
                from_status=row["status"], to_status="done",
                payload=payload or None,
            )
            changed = True
        if changed:
            _settle_finished_messages(conn, actor=actor, verb="done", todo_id=entry_id)
        letters = _letters_mentioning(conn, entry_id)
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    return {
        "todo": get_entry(conn, entry_id), "changed": changed or bool(amended),
        "amended": amended, "letters": letters,
    }


def abandon_todo(conn, entry_id, *, agent=None, reason, force=False,
                 evidence=None, for_dad=None):
    """``open | taken → abandoned``: the terminal give-up, with a reason.

    Coop Quest "lifecycles-and-the-first-close": the taker abandons; anyone
    else needs ``force``. A done todo is never abandoned — its first close
    is the record (amend the outcome instead) — and an abandoned one stays
    abandoned unless BJT forces it done. ``for_dad`` is Dad's Korean lines,
    carried to the office (see ``done_todo``).
    """
    actor = require_canonical_identity(agent or detect_agent())
    text = (reason or "").strip()
    if not text:
        raise ValueError("abandoning a todo needs a reason (--reason)")
    conn.execute("BEGIN IMMEDIATE")
    try:
        row = _todo_row(conn, entry_id)
        if row["status"] in TODO_TERMINAL_STATUSES:
            raise LifecycleError(
                "todo #%d is %s — the first close is the record"
                % (entry_id, row["status"])
            )
        foreign = (
            row["status"] == "taken"
            and bool(row["taken_by"])
            and row["taken_by"] != actor
        )
        if foreign and not force:
            raise LifecycleError(
                "todo #%d is taken by %s — only the taker abandons it"
                " (--force to override)" % (entry_id, row["taken_by"])
            )
        stamp = now_iso()
        cur = conn.execute(
            "UPDATE entries SET status='abandoned', closed_by=?, closed_at=?,"
            " updated_at=? WHERE id=? AND status=?",
            (actor, stamp, stamp, entry_id, row["status"]),
        )
        if cur.rowcount == 0:
            raise LifecycleError(
                "todo #%d changed under this abandon — re-read and retry"
                % entry_id
            )
        payload = {"previous_taker": row["taken_by"]} if foreign else {}
        if evidence:
            payload["evidence"] = evidence
        _record_event(
            conn, "todo", entry_id, "abandon", actor,
            from_status=row["status"], to_status="abandoned", reason=text,
            payload=payload or None,
        )
        _settle_finished_messages(conn, actor=actor, verb="abandon", todo_id=entry_id)
        conn.commit()
        return get_entry(conn, entry_id)
    except Exception:
        conn.rollback()
        raise


def close_todo(conn, entry_id, *, status, actor, result=None, evidence=None,
               commit_sha=None, force=False):
    """The engine's one door to a todo outcome (dispatch's settle and facade).

    ``status`` is ``done``, ``abandoned`` or ``open`` (a release); ``result``
    is the result summary for done and the reason for abandoned and open;
    ``evidence`` rides in the event payload. ``actor`` is stored as the
    closer, so it must be a roster id. Returns done_todo's dict for done,
    ``{"todo": row}`` otherwise; refusals are LifecycleError.
    """
    if status == "done":
        return done_todo(conn, entry_id, agent=actor, result=result,
                         commit_sha=commit_sha, force=force, evidence=evidence)
    if status == "abandoned":
        row = abandon_todo(conn, entry_id, agent=actor, reason=result or "",
                           force=force, evidence=evidence)
    elif status == "open":
        row = release_todo(conn, entry_id, agent=actor, reason=result or "",
                           evidence=evidence)
    else:
        raise ValueError("close_todo status must be done, abandoned or open")
    return {"todo": row}


def assign_todo(conn, entry_id, assignee):
    """(Re)assign one todo, or unassign it with None; True if a row changed.

    The assignee is checked against the roster (an unassigned todo is
    anyone's, so ``any`` is never written); the change is an ``assign``
    event with the previous assignee.
    """
    value = "" if assignee is None else assignee.strip()
    if assignee is not None and not value:
        raise ValueError("assignee must be non-empty (None unassigns)")
    if value:
        require_canonical_identity(value, role="assignee")
    conn.execute("BEGIN IMMEDIATE")
    try:
        row = get_entry(conn, entry_id)
        if row is None or row["kind"] != "todo":
            conn.rollback()
            return False
        cur = conn.execute(
            "UPDATE entries SET assignee=?, updated_at=? WHERE id=? AND kind='todo'",
            (value, now_iso(), entry_id),
        )
        _record_event(
            conn, "todo", entry_id, "assign", detect_agent(),
            from_status=row["status"], to_status=row["status"],
            payload={"assignee": value, "previous": row["assignee"]},
        )
        conn.commit()
        return cur.rowcount > 0
    except Exception:
        conn.rollback()
        raise


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
                   alias="", taken_by=None):
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
    if taken_by is not None:
        where.append("%staken_by=?" % prefix)
        params.append(taken_by)
    if status is not None:
        statuses = (status,) if isinstance(status, str) else tuple(status)
        where.append("%sstatus IN (%s)" % (prefix, ",".join("?" * len(statuses))))
        params.extend(statuses)
    return where, params


def _fts_match(query):
    """The FTS5 MATCH expression for a free-text query: every term, quoted."""
    return " ".join('"%s"' % term.replace('"', "") for term in query.split())


def search_rows(conn, query, kind=None, limit=10, *, project=None, status=None,
                assignee=None, taken_by=None):
    """Search with every structured filter applied *before* LIMIT.

    The old HTTP route filtered project/status in Python after this function
    had already clipped the global result set.  A matching project could then
    look empty merely because newer hits from another project occupied the
    limit.  Keeping the complete predicate here fixes that correctness bug and
    preserves one SQL owner.
    """
    filters, filter_params = _entry_filters(
        kind, project, status, assignee, alias="e", taken_by=taken_by,
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
                 assignee=None, taken_by=None, limit=100):
    """Return newest/ranked entries under one provider-neutral filter API."""
    if query:
        return search_rows(
            conn, query, kind, limit, project=project, status=status,
            assignee=assignee, taken_by=taken_by,
        )
    where, params = _entry_filters(kind, project, status, assignee,
                                   taken_by=taken_by)
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
    author = require_canonical_identity((author_agent or "").strip() or detect_agent())
    if recipient is not None:
        require_canonical_identity(recipient, role="recipient")
    assert_no_handoff_pointer(body or "")
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
        # One live letter per brain (Coop Quest "a-letter-carries-itself"):
        # the new letter replaces every other live letter this author wrote
        # in this project to the same address (Forrest keeps `--to`, so the
        # address is part of "the same brain's letter"). Each old one keeps
        # its reads and start, points at the new id and gets its own event,
        # so the chain reads both ways.
        stale = conn.execute(
            "SELECT id, status FROM letters WHERE project=? AND author_agent=?"
            " AND recipient_agent IS ? AND status IN ('unread','read','working')"
            " AND id != ? ORDER BY id",
            (scope, author, recipient, letter_id),
        ).fetchall()
        for old in stale:
            conn.execute(
                "UPDATE letters SET status='superseded', superseded_by=?,"
                " updated_at=? WHERE id=? AND status=?",
                (letter_id, created, old["id"], old["status"]),
            )
            _record_event(
                conn, "letter", old["id"], "supersede", author,
                from_status=old["status"], to_status="superseded",
                reason="a newer letter from the same author to the same address",
                payload={"superseded_by": letter_id},
            )
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
    if action == "read" and row["status"] in ("read", "working"):
        return  # a later read: no transition, read_count bumps
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
    require_canonical_identity(actor)
    ack_body = None
    if action == "ack":
        ack_body = (body or "").strip()
        if not ack_body:
            raise ValueError("letter ack requires a non-empty body: say what shipped")
        assert_no_handoff_pointer(ack_body)
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
        stamp = now_iso()
        if action == "read" and row["status"] != "unread":
            # A later read of a live letter: the count and time move, the
            # state does not (Coop Quest "a-letter-carries-itself").
            conn.execute(
                "UPDATE letters SET read_count=read_count+1, last_read_at=?"
                " WHERE id=?",
                (stamp, letter_id),
            )
            conn.commit()
            return get_letter(conn, letter_id)
        bump = 1 if action == "read" else 0
        cur = conn.execute(
            "UPDATE letters SET status=?, worker_agent=?, ack_body=?,"
            " superseded_by=?, read_count=read_count+?,"
            " last_read_at=CASE WHEN ?=1 THEN ? ELSE last_read_at END,"
            " updated_at=? WHERE id=? AND status=?",
            (
                target,
                actor if action == "start" else row["worker_agent"],
                row["ack_body"] if ack_body is None else ack_body,
                row["superseded_by"] if superseded_by is None else superseded_by,
                bump, bump, stamp,
                stamp, letter_id, row["status"],
            ),
        )
        if cur.rowcount != 1:  # pragma: no cover - BEGIN IMMEDIATE serializes
            raise LetterTransitionError("letter #%d changed concurrently" % letter_id)
        _record_event(
            conn, "letter", letter_id, action, actor,
            from_status=row["status"], to_status=target,
            payload=(
                {"superseded_by": superseded_by} if superseded_by is not None
                else None
            ),
        )
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
    require_canonical_identity(actor)
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
            if row["status"] == "unread":
                # The first read is the transition; later reads only count
                # (Coop Quest "a-letter-carries-itself": reading is a write).
                conn.execute(
                    "UPDATE letters SET status='read', read_count=read_count+1,"
                    " last_read_at=?, updated_at=? WHERE id=? AND status='unread'",
                    (stamp, stamp, row["id"]),
                )
                _record_event(
                    conn, "letter", row["id"], "read", actor,
                    from_status="unread", to_status="read",
                )
            else:
                conn.execute(
                    "UPDATE letters SET read_count=read_count+1, last_read_at=?"
                    " WHERE id=?",
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
                 scope="", author_agent=None, project=None, commit_sha=None,
                 supersedes=None):
    """Write one typed learning and return its id.

    ``slug`` names the symptom and ``scope`` the framework or language. The
    lesson names the columns but not which are required; a learning without
    its problem or its resolution teaches nothing, so both are required and
    the rest are optional. Every text is stripped. ``commit_sha`` is for a
    writer that already knows its head; the entry-first flow backfills it
    with ``set_commit_sha``.

    ``supersedes`` names the learning this one corrects (Coop Quest
    "seven-tables-and-a-roster": the old row is never overwritten; its
    ``superseded_by`` points at the fix so "we used to think" stays
    readable). An already superseded learning is refused: supersede the
    one that replaced it.
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
    author = require_canonical_identity((author_agent or "").strip() or detect_agent())
    owner = project or detect_project()
    created = now_iso()
    conn.execute("BEGIN IMMEDIATE")
    try:
        learning_id = _next_ledger_id(conn)
        if supersedes is not None:
            old = get_learning(conn, supersedes)
            if old is None:
                raise ValueError("no learning with id %d to supersede" % supersedes)
            if _field(old, "superseded_by"):
                raise ValueError(
                    "learning #%d is already superseded by #%d; supersede that one"
                    % (supersedes, old["superseded_by"])
                )
        conn.execute(
            "INSERT INTO learning_entries (id, project, author_agent, scope,"
            " slug, problem, hypothesis, resolution, source, commit_sha,"
            " created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (learning_id, owner, author, text["scope"], text["slug"],
             text["problem"], text["hypothesis"], text["resolution"],
             text["source"], (commit_sha or "").strip(), created),
        )
        _reserve_entry_id(conn, learning_id)
        if supersedes is not None:
            conn.execute(
                "UPDATE learning_entries SET superseded_by=? WHERE id=?",
                (learning_id, supersedes),
            )
            _record_event(
                conn, "learning", supersedes, "supersede", author,
                payload={"superseded_by": learning_id},
            )
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


def _normalize_intent(intent, *, strict=True):
    """The intent a message carries; a write must name one of MESSAGE_INTENTS.

    Reads (``strict=False``) stay lenient: a filter may ask for any old
    value, and the rows written before the set was closed keep theirs.
    """
    value = (intent or "request").strip().lower()
    if not re.fullmatch(r"[a-z][a-z0-9_.-]{0,63}", value):
        raise ValueError(
            "intent must match [a-z][a-z0-9_.-]{0,63}",
        )
    if strict and value not in MESSAGE_INTENTS:
        raise ValueError(
            "intent %r is not one the mailbox processes; use one of: %s"
            % (value, ", ".join(MESSAGE_INTENTS))
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
    if reply_to is not None and cur.rowcount and getattr(conn, 'office_writer', False):
        # Only the owning office has this business transition. Exported
        # standalone clients route their public writes to that same owner.
        from . import engine
        engine.answer_parked_from_reply(conn, cur.lastrowid)
    return cur.lastrowid, final_thread


def add_message(conn, subject, *, recipient_agent, sender_agent=None,
                intent="request", body="", artifact_refs=None, project=None,
                machine=None, thread_id=None, reply_to=None):
    """Send one durable message; returns ``(id, thread_id)``."""
    sender = require_canonical_identity(sender_agent or detect_agent())
    require_canonical_identity(recipient_agent, role="recipient")
    result = _insert_message(
        conn, subject,
        recipient_agent=recipient_agent,
        sender_agent=sender,
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
    sender = require_canonical_identity(sender_agent or detect_agent())
    require_canonical_identity(recipient_agent, role="recipient")
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
            sender_agent=sender,
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
    normalized = tuple(_normalize_intent(intent, strict=False) for intent in intents)
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


def claim_message(conn, message_id, agent, origin=None):
    """Atomically claim one open message addressed to ``agent``.

    ``origin`` is what the claiming CLI declared (`declare_origin`); the
    event keeps its chat reference and Mac, and the office's engine records
    the observed session and the outcome obligation a terminal claim owes
    (Coop Quest "adopting-a-take").
    """
    require_canonical_identity(agent)
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
        _record_event(
            conn, "message", message_id, "claim", agent,
            from_status="open", to_status="claimed",
            payload=_declared(origin) or None,
        )
        conn.commit()
        return get_message(conn, message_id)
    except Exception:
        conn.rollback()
        raise


def release_message(conn, message_id, agent, for_dad=None):
    """Release one's own claim; True if the message went back to open.
    ``for_dad`` is Dad's Korean lines, carried to the office (see
    ``done_todo``), which requires them when the release reports a claim."""
    require_canonical_identity(agent)
    conn.execute("BEGIN IMMEDIATE")
    try:
        cur = conn.execute(
            "UPDATE messages SET status='open', claimed_by='', updated_at=?"
            " WHERE id=? AND status='claimed' AND claimed_by=?",
            (now_iso(), message_id, agent),
        )
        if cur.rowcount == 1:
            _record_event(
                conn, "message", message_id, "release", agent,
                from_status="claimed", to_status="open",
            )
        conn.commit()
        return cur.rowcount == 1
    except Exception:
        conn.rollback()
        raise


def finish_message(conn, message_id, agent, status="done", for_dad=None):
    """Finish a message one is part of, as ``done`` or ``superseded``.
    ``for_dad`` is Dad's Korean lines, carried to the office (see
    ``done_todo``), which requires them when the finish reports a claim."""
    if status not in ("done", "superseded"):
        raise ValueError("finished message status must be done or superseded")
    require_canonical_identity(agent)
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
        _record_event(
            conn, "message", message_id,
            "done" if status == "done" else "supersede", agent,
            from_status=row["status"], to_status=status,
        )
        conn.commit()
        return True
    except Exception:
        conn.rollback()
        raise


def reply_message(conn, message_id, *, sender_agent, body, subject=None,
                  intent="response", artifact_refs=None):
    """Reply in-thread and close the addressed parent atomically."""
    require_canonical_identity(sender_agent)
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
        _record_event(
            conn, "message", message_id, "reply", sender_agent,
            from_status=parent["status"], to_status="done",
            payload={"reply_id": reply_id},
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
            # Sent only when given, so an office still on schema v5 (no
            # `priority` argument) keeps taking this client's other writes.
            **({"priority": args.priority} if getattr(args, "priority", None) else {}),
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
            supersedes=args.supersedes,
        )
        row = get_learning(conn, learning_id)
    except ValueError as exc:
        sys.exit(str(exc))
    finally:
        conn.close()
    print("coop #%d learning recorded — %s" % (learning_id, row["slug"]))
    if args.supersedes is not None:
        print("(supersedes learning #%d; it stays readable, pointing here)"
              % args.supersedes)
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


def _parse_reply_outcome(text):
    """`<message id>:<applied|not_applied>:<one-line reason>` → the lesson's shape."""
    parts = text.split(":", 2)
    if (len(parts) != 3 or not parts[0].strip().isdigit()
            or parts[1].strip() not in ("applied", "not_applied")
            or not parts[2].strip()):
        raise ValueError(
            "--reply-outcome takes <message id>:<applied|not_applied>:<one-line"
            " reason>, got %r" % text
        )
    return {"message_id": int(parts[0]), "disposition": parts[1].strip(),
            "reason": parts[2].strip()}


def _owed_line(row, entry_id):
    """The outcome a take through the office owes, when the office answered one.

    Coop Quest "an-outcome-is-owed": the take and the debt are written
    together; the taker's own `done --result` / `release` / `abandon` (a
    claim's `reply` / `message-done` / `message-release`) is the report that
    discharges it, and the chief reviews it after three hours otherwise.
    """
    owed = row["obligation"] if "obligation" in row.keys() else None
    if not owed:
        return ""
    return ("  owes an outcome by %s — your own done/release/abandon --for-dad"
            " \"…\" (reply or message-done for a claim) is its report; otherwise"
            " the chief reviews #%d then" % (owed["due_at"], entry_id))


def cmd_status(args):
    """`take`, `release`, `done`, `abandon`, and the legacy `status` door.

    Coop Quest "lifecycles-and-the-first-close": every verb is a function
    that checks the current state, moves it and appends an event; `done`
    also names the live letters that mention the todo so the session acks
    the letter that asked for the work. A take the office records owes an
    outcome ("adopting-a-take"); the CLI says so and names the report.
    """
    agent = detect_agent(getattr(args, "agent", None))
    force = bool(getattr(args, "force", False))
    dad = read_body(getattr(args, "for_dad", None))
    conn = connect()
    try:
        if args.cmd == "done":
            outcome = done_todo(
                conn, args.id, agent=agent, result=args.result,
                commit_sha=args.sha, force=force,
                reply_outcomes=[_parse_reply_outcome(text)
                                for text in (getattr(args, "reply_outcome", None) or [])],
                for_dad=dad,
            )
            if outcome["amended"]:
                print("coop #%d is already done — amended %s (an `edit` event;"
                      " closer, time and status unchanged)"
                      % (args.id, ", ".join(outcome["amended"])))
            elif not outcome["changed"]:
                print("coop #%d is already done — an identical repeat writes"
                      " nothing" % args.id)
            else:
                print("coop #%d -> done" % args.id)
            if outcome["letters"]:
                print("(live letters mentioning #%d: %s — `coop.py letter ack"
                      " <id> --body …` if this was their work)" % (
                          args.id,
                          ", ".join("#%d" % n for n in outcome["letters"]),
                      ))
        elif args.cmd == "take":
            row = take_todo(conn, args.id, agent=agent, force=force)
            print("coop #%d -> taken by %s" % (args.id, row["taken_by"]))
            if _owed_line(row, args.id):
                print(_owed_line(row, args.id))
        elif args.cmd == "release":
            release_todo(conn, args.id, agent=agent, reason=args.reason,
                         for_dad=dad)
            print("coop #%d -> open (released)" % args.id)
        elif args.cmd == "abandon":
            abandon_todo(conn, args.id, agent=agent, reason=args.reason,
                         force=force, for_dad=dad)
            print("coop #%d -> abandoned" % args.id)
        else:
            changed = set_status(
                conn, args.id, args.status, agent=agent, reason=args.reason,
                result=args.result, commit_sha=args.sha, force=force,
                for_dad=dad,
            )
            if not changed:
                sys.exit(_not_a_todo(conn, args.id))
            print("coop #%d -> %s" % (args.id, args.status))
    except ValueError as exc:  # LifecycleError, UnknownIdentityError included
        sys.exit(str(exc))
    finally:
        conn.close()


def cmd_events(args):
    """`events <id>`: the append-only transition log of one row."""
    conn = connect()
    try:
        rows = list_events(conn, args.id, kind=args.kind)
    finally:
        conn.close()
    if not rows:
        print("coop: no events for #%d" % args.id)
        return
    for row in rows:
        print(format_event(row))


def format_event(row):
    move = (
        "%s → %s" % (row["from_status"], row["to_status"])
        if row["from_status"] or row["to_status"] else ""
    )
    extra = []
    if row["reason"]:
        extra.append(row["reason"])
    payload = event_payload(row)
    if payload:
        extra.append(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return "%s  %-8s #%-5d %-9s %s · %s%s" % (
        row["at"][:16].replace("T", " "), row["kind"], row["entry_id"],
        row["event"], move, row["actor"],
        (" · " + " · ".join(extra)) if extra else "",
    )


def _age(stamp):
    """`3h ago` for an ISO stamp; `?` when it cannot be read."""
    try:
        then = dt.datetime.fromisoformat(stamp)
    except (TypeError, ValueError):
        return "?"
    if then.tzinfo is None:
        then = then.astimezone()
    seconds = max(0, (dt.datetime.now(dt.timezone.utc) - then).total_seconds())
    if seconds < 3600:
        return "%dm ago" % (seconds // 60)
    if seconds < 48 * 3600:
        return "%dh ago" % (seconds // 3600)
    return "%dd ago" % (seconds // 86400)


def format_row(r):
    # A take is shown, never reaped (Coop Quest "lifecycles-and-the-first-close":
    # "[taken by X, Nh ago]"); done rows carry no flag, as before.
    if r["status"] == "taken":
        flag = " [taken by %s, %s]" % (
            _field(r, "taken_by") or "?", _age(_field(r, "taken_at")),
        )
    elif r["status"] not in ("", "done"):
        flag = " [%s]" % r["status"]
    else:
        flag = ""
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
    if _field(row, "read_count", 0):
        print("read %d time(s) · last read: %s" % (
            row["read_count"], _field(row, "last_read_at") or "?",
        ))
    if row["body"]:
        print("\n" + row["body"])
    if row["ack_body"]:
        print("\nack: " + row["ack_body"])
    if row["superseded_by"]:
        print("\nsuperseded by letter #%d" % row["superseded_by"])


def format_learning(row):
    symptom = "%s: %s" % (row["scope"], row["slug"]) if row["scope"] else row["slug"]
    superseded = _field(row, "superseded_by")
    return "#%-4d %-8s %s  %s · %s · %s%s" % (
        row["id"], "learning", row["created_at"][:16].replace("T", " "),
        symptom, row["project"], row["author_agent"],
        (" [superseded by #%d]" % superseded) if superseded else "",
    )


def print_learning(row):
    print(format_learning(row))
    if row["commit_sha"]:
        print("commit: %s" % row["commit_sha"])
    if _field(row, "superseded_by"):
        print("superseded by learning #%d (kept as what we used to think)"
              % row["superseded_by"])
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
    if _owed_line(row, args.id):
        print(_owed_line(row, args.id))


def cmd_message_release(args):
    agent = detect_agent(args.agent)
    conn = connect()
    try:
        changed = release_message(conn, args.id, agent, for_dad=read_body(args.for_dad))
    except ValueError as exc:
        sys.exit(str(exc))
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
        finish_message(conn, args.id, agent, status=args.status,
                       for_dad=read_body(args.for_dad))
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
        truth = ground_truth_host()
        print("━━ coop session brief ━━")
        if is_ground_truth_machine():
            print("Machine : %s — GROUND-TRUTH service (Coop DB lives here; code origin follows workbench)" % machine)
        elif ark_profile():
            print("Machine : %s — ARK independent branch %s (canonical role unchanged)" % (machine, ark_profile()['branch_id']))
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
        # The agent's and unassigned active todos, plus the ones it has
        # taken whoever they were assigned to; a take is shown, never reaped.
        todos = {
            row["id"]: row for row in list_entries(
                conn, kind="todo", status=TODO_ACTIVE_STATUSES, project=project,
                assignee=(agent, ""), limit=BRIEF_ITEM_LIMIT + 1,
            )
        }
        for row in list_entries(
            conn, kind="todo", status="taken", project=project,
            taken_by=agent, limit=BRIEF_ITEM_LIMIT + 1,
        ):
            todos.setdefault(row["id"], row)
        todos = [todos[key] for key in sorted(todos, reverse=True)]
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
        if isinstance(exc, LedgerUnavailable):
            print("Action  : run it on %s instead, for example over SSH:"
                  % ground_truth_host())
        else:
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
            s.add_argument("--priority", help="a short word for its priority (MC L26)")
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
    s.add_argument("--supersedes", type=int, metavar="ID",
                   help="the learning this one corrects (it stays readable,"
                        " pointing here)")
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

    # The todo lifecycle (Coop Quest "lifecycles-and-the-first-close"):
    # open → taken → done | abandoned, release back to open, and the fast
    # path `done` on an open todo. `status` is the legacy door routed to
    # the same verbs. A close of taken work reports to Dad, who reads two
    # or three Korean lines of it (Master Chief "the-worker-and-its-report"),
    # so the closing verbs carry `--for-dad`; the office requires it there.
    def add_for_dad(s):
        s.add_argument("--for-dad", metavar="LINES",
                       help="two or three Korean lines to Dad, at most four (cut"
                            " at 1,200 characters): what was done, where it"
                            " landed, whether he must act; required when the"
                            " close reports taken work; '-' reads stdin")

    s = sub.add_parser("take", help="open → taken (your name on it)")
    s.add_argument("id", type=int)
    s.add_argument("--force", action="store_true",
                   help="take over someone else's take (recorded as a takeover)")
    s.add_argument("--agent")
    s = sub.add_parser("release", help="taken → open, with a reason")
    s.add_argument("id", type=int)
    s.add_argument("--reason", required=True,
                   help="why the take is released (a named decision)")
    add_for_dad(s)
    s.add_argument("--agent")
    s = sub.add_parser("done", help="open|taken → done; again: amend the outcome")
    s.add_argument("id", type=int)
    s.add_argument("--result", help="result summary (what shipped); the report"
                   " of a take the office records")
    s.add_argument("--sha", help="the commit that closed it")
    s.add_argument("--reply-outcome", action="append", default=[],
                   metavar="ID:applied|not_applied:REASON",
                   help="answer one reply on the todo's thread; repeat for each"
                        " (a bound todo's done must answer every reply)")
    s.add_argument("--force", action="store_true",
                   help="close someone else's take, or record an abandoned"
                        " todo done (BJT's call; the event keeps the reversal)")
    add_for_dad(s)
    s.add_argument("--agent")
    s = sub.add_parser("abandon", help="open|taken → abandoned, with a reason")
    s.add_argument("id", type=int)
    s.add_argument("--reason", required=True, help="why the work is given up")
    s.add_argument("--force", action="store_true",
                   help="abandon someone else's take")
    add_for_dad(s)
    s.add_argument("--agent")
    s = sub.add_parser("status", help="legacy door to the same verbs")
    s.add_argument("id", type=int)
    s.add_argument("status", choices=STATUSES)
    s.add_argument("--reason")
    s.add_argument("--result")
    s.add_argument("--sha")
    s.add_argument("--force", action="store_true")
    add_for_dad(s)
    s.add_argument("--agent")
    s = sub.add_parser("events", help="the append-only transition log of one id")
    s.add_argument("id", type=int)
    s.add_argument("--kind", choices=("todo", "history", "letter", "learning",
                                      "message"))
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
    add_for_dad(s)
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
    add_for_dad(s)
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
    elif args.cmd in ("take", "release", "done", "abandon", "status"):
        cmd_status(args)
    elif args.cmd == "events":
        cmd_events(args)
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


# This file is the generated, standalone compatibility bundle. bjtCoop owns
# its source; callers keep their established imports and CLI vocabulary.
WRITE_METHODS = (
    "add_entry", "set_commit_sha", "set_status", "assign_todo",
    "take_todo", "release_todo", "done_todo", "abandon_todo", "close_todo",
    "add_letter", "transition_letter", "read_letters", "add_learning",
    "add_message", "add_message_once", "claim_message", "release_message",
    "finish_message", "reply_message",
)
WORKSPACE_FENCE_KEYS = (
    "FORREST_WORKSPACE", "FORREST_WORK_RUNTIME_PATH", "FORREST_WORK_SESSION_TOKEN",
    "FORREST_WORK_GENERATION", "FORREST_WORK_UNIT_MESSAGE_ID",
)


class OfficeCallError(RuntimeError):
    """A write was not acknowledged; receipt_id is the reconciliation key."""

    def __init__(self, receipt):
        self.receipt = receipt
        super().__init__("Coop %s (HTTP %s, receipt %s): %s" % (
            receipt["verdict"], receipt.get("http_status", "unknown"),
            receipt["request_id"], receipt.get("error", "office unavailable"),
        ))


class OfficeRejected(OfficeCallError, ValueError):
    pass


def _read_owner(path):
    with owner_path(path).open(encoding="utf-8") as handle:
        owner = json.load(handle)
    if (not isinstance(owner, dict) or owner.get("protocol") != 1
            or not re.fullmatch(r"http://127\.0\.0\.1:[0-9]{1,5}", owner.get("url", ""))
            or not isinstance(owner.get("token"), str) or len(owner["token"]) < 32
            or not isinstance(owner.get("instance"), str)):
        raise ValueError("invalid Coop owner descriptor")
    return owner


def operator_token(db_path=None):
    token = os.environ.get("COOP_OPERATOR_TOKEN", "").strip()
    if not token:
        path = Path(os.environ.get("COOP_OPERATOR_KEY_FILE", str(db_path or DEFAULT_DB) + ".operator.key")).expanduser()
        status = path.stat()
        if status.st_uid != os.getuid() or status.st_mode & 0o077:
            raise LedgerUnavailable("Coop operator key must be private to its owner")
        token = path.read_text().strip()
    if len(token) < 32 or any(char.isspace() for char in token):
        raise LedgerUnavailable("Coop operator bearer is unavailable")
    return token


def _office_target(db_path):
    profile = ark_profile()
    if profile and profile.get('members', {}).get('coop') != 'run':
        raise LedgerUnavailable('Ark Coop is held; canonical routing is not a fallback')
    if os.path.realpath(db_path) != os.path.realpath(DEFAULT_DB) or is_local_office_owner():
        owner = _read_owner(db_path)
        return owner['url'] + '/rpc', owner['token'], owner['instance']
    family = _family_ledger()
    member = family.get('members', {}).get('coop', {})
    machine = family['machines'][family['roles']['forrest']]
    dns = machine.get('magic_dns')
    path = member.get('compat_path')
    if (member.get('host') != '@forrest'
            or not isinstance(dns, str) or not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9.-]*', dns)
            or path != '/api/coop/compat'):
        raise LedgerUnavailable('Coop has no registered HTTPS compatibility gateway')
    port = member.get('https_port', 443)
    if type(port) is not int or not 1 <= port <= 65535:
        raise LedgerUnavailable('Coop HTTPS port is invalid')
    origin = 'https://' + dns.lower().rstrip('.') + ((':' + str(port)) if port != 443 else '')
    return origin + path, operator_token(db_path), None


def _save_receipt(path, receipt):
    directory = Path(str(path) + ".receipts")
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    destination = directory / (receipt["request_id"] + ".json")
    temporary = destination.with_suffix(".tmp")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(receipt, handle, ensure_ascii=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)


def _prune_receipts(db_path, *, now=None):
    """Delete settled receipt files older than a day, at most once an hour.

    Coop Quest "the-owner-file-fails-closed": settled receipts (committed or
    rejected) are deleted after a day; the sweep runs at most hourly (537
    files had piled up in three days); an uncertain receipt stays until
    someone reconciles it. A file whose verdict cannot be read is kept too:
    it is not known to be settled. The hourly stamp is `.pruned-at` in the
    receipts directory. Never raises; returns how many files went.
    """
    directory = Path(str(db_path) + ".receipts")
    stamp = directory / ".pruned-at"
    moment = time.time() if now is None else now
    removed = 0
    try:
        if (stamp.exists()
                and moment - stamp.stat().st_mtime < RECEIPT_PRUNE_INTERVAL_SECONDS):
            return 0
        with os.scandir(directory) as found:
            for entry in found:
                if not entry.name.endswith(".json"):
                    continue
                try:
                    if moment - entry.stat().st_mtime < RECEIPT_RETENTION_SECONDS:
                        continue
                    with open(entry.path, encoding="utf-8") as handle:
                        verdict = json.load(handle).get("verdict")
                    if verdict in ("committed", "rejected"):
                        os.unlink(entry.path)
                        removed += 1
                except (OSError, ValueError, AttributeError):
                    continue
        stamp.touch()
        os.utime(stamp, (moment, moment))
    except OSError:
        pass
    return removed


class _NoRedirect(urlrequest.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def office_call(method, params=None, *, path=None, mutating=True, request_id=None):
    """Call the sole writer once. Never replay or fall back to raw SQLite.

    An uncertain request is reconciled using office_receipt, not repeated
    with a fresh identity. The original receipt is durable before dispatch.
    """
    db_path = os.path.abspath(path or os.environ.get("COOP_DB_PATH") or DEFAULT_DB)
    rid = request_id or str(uuid.uuid4())
    if str(uuid.UUID(rid)) != rid:
        raise ValueError("invalid request id")
    receipt = {"request_id": rid, "method": method, "verdict": "uncertain"}
    if mutating:
        _save_receipt(db_path, receipt)

    def settle():
        if mutating:
            _save_receipt(db_path, receipt)
            _prune_receipts(db_path)

    status = None
    try:
        endpoint, token, instance = _office_target(db_path)
        payload = {"request_id": rid, "method": method, "params": params or {},
                   "workspace_fence": {k: os.environ[k] for k in WORKSPACE_FENCE_KEYS if k in os.environ}}
        req = urlrequest.Request(
            endpoint, data=json.dumps(payload).encode(),
            headers={"Authorization": "Bearer " + token,
                     "Content-Type": "application/json"}, method="POST",
        )
        opener = urlrequest.build_opener(urlrequest.ProxyHandler({}), _NoRedirect())
        try:
            response = opener.open(req, timeout=15)
        except urlerror.HTTPError as exc:
            response = exc
        with response:
            status = response.status
            if status in (401, 403, 422):
                receipt.update(verdict="rejected", http_status=status)
            raw = response.read(2 * 1024 * 1024 + 1)
        try:
            domain = json.loads(raw)
        except (ValueError, UnicodeDecodeError):
            domain = None
        valid = (isinstance(domain, dict) and domain.get("request_id") == rid
                 and isinstance(domain.get("instance"), str) and bool(domain["instance"])
                 and (instance is None or domain["instance"] == instance))
        if valid and status == 200 and domain.get("verdict") == "committed" and "result" in domain:
            receipt.update(verdict="committed", http_status=status)
            settle()
            return domain["result"]
        receipt.update(http_status=status,
                       verdict="rejected" if status in (401, 403, 422) else "uncertain",
                       error="office did not return a valid committed verdict")
        if valid and domain.get("verdict") == "rejected":
            receipt.update(verdict="rejected", error=domain.get("error", "rejected"))
            error_kind = domain.get("error_type")
            settle()
            if error_kind in _DOMAIN_ERRORS:
                raise _DOMAIN_ERRORS[error_kind](receipt["error"])
    except (OSError, ValueError, LedgerUnavailable) as exc:
        if isinstance(exc, tuple(_DOMAIN_ERRORS.values())):
            raise
        receipt.update(error=str(exc), http_status=status)
    settle()
    raise (OfficeRejected if receipt["verdict"] == "rejected" else OfficeCallError)(receipt)


# The office answers a domain refusal with the exception's class name; the
# client raises the same class so a caller sees one vocabulary either side
# of the process boundary.
_DOMAIN_ERRORS = {
    "LetterNotFound": LetterNotFound,
    "LetterTransitionError": LetterTransitionError,
    "LifecycleError": LifecycleError,
    "UnknownIdentityError": UnknownIdentityError,
    "HandoffPointerError": HandoffPointerError,
}


def _office_writer(fn):
    signature = inspect.signature(fn)

    @functools.wraps(fn)
    def routed(conn, *args, **kwargs):
        owned = (getattr(conn, "office_owned", False)
                 or owner_path(getattr(conn, "db_path", None)).exists())
        if getattr(conn, "office_writer", False) or not owned:
            return fn(conn, *args, **kwargs)
        bound = signature.bind(conn, *args, **kwargs)
        bound.apply_defaults()
        params = dict(bound.arguments)
        params.pop("conn")
        # Resolve caller-local identity before crossing the process boundary;
        # a take's origin (its chat, its Mac, its cwd) is caller-local too.
        for key, default in (("project", detect_project), ("agent", detect_agent),
                             ("author_agent", detect_agent), ("sender_agent", detect_agent),
                             ("machine", detect_machine), ("origin", declare_origin)):
            if key in params and not params[key]:
                # None on read_letters means all projects, not caller project.
                if fn.__name__ == "read_letters" and key == "project":
                    continue
                params[key] = default()
        # Schema v6's `priority` is sent only when given: an office still on
        # v5 knows no such argument, and absent means none on a v6 office.
        if params.get("priority") is None:
            params.pop("priority", None)
        return office_call(fn.__name__, params, path=conn.db_path)
    return routed


for _name in WRITE_METHODS:
    globals()[_name] = _office_writer(globals()[_name])


if __name__ == "__main__":
    try:
        main()
    except (OfficeCallError, LedgerUnavailable) as exc:
        sys.exit(str(exc))

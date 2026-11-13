#!/usr/bin/env python3
"""Open a Forrest terminal body: the Soul Vault identity card, then the body's context.

The repository copy serves the general development body and is also vendored
as ``.forrest/session_start.py`` in every registered repository. Claude Code
and Codex run the appropriate copy at SessionStart.

Identity has one source (Memory L14). The hook runs the Soul Vault client of
the family's main checkout, ``soul-vault bootstrap --harness <agent> --soul
<vault> --hook``, and prints what it prints: the identity card (the identity
note whole, then the bundle to read in full before any task), or one refusal
line, ``identity not loaded: <code>: <cause>``, with the instruction to stop.
This file reads no vault note and holds no location of one, so a worktree, a
WorkUnit clone or a registered repository cannot serve an older copy of the
Soul, and no half of an identity is ever assembled here.

The hook is not fail-open for identity. A client that is missing, cannot
start, times out or prints nothing, and a session whose Soul the store cannot
name, all end as that same refusal. A refusal is printed alone: no overlay
that invites work, and no Coop letter is read, because reading marks a handoff
letter read and a session that will stop must not consume one. The hook still
exits 0: a harness may drop the output of a hook that failed, and the refusal
is exactly the text the model must see.

After a card the hook adds only what the body owns: the terminal-context
header, the selected Soul's registry coordinates, a registered repository's
project overlay, the Coop handoff letters and the live Coop card. Those parts
stay fail-open and conspicuous (an unavailable Coop database is a warning in
the model-visible output), and under a host's context limit they yield first.
The card is never clipped or demoted.

The bootstrap is a SessionStart product. Any other hook event exits silently
and writes nothing: bundles 22-23 shipped a ``UserPromptSubmit`` hook for a
terminal-session receipt that has had no reader since 2026-09-18, and Claude
Code snapshots its hooks when a session opens, so a session older than bundle
24 keeps calling this script on every prompt. Printing there would re-inject
the card and the whole context into the model's context each turn.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
import secrets
import tempfile
import time


_PROJECT_BRIEF_MAX_BYTES = 16 * 1024
_COOP_MAX_BYTES = 16 * 1024
_RUNTIME_CONTEXT_FILENAME = "runtime-context.json"
_RUNTIME_CONTEXT_SCHEMA_VERSION = 1
_HOOK_PAYLOAD_MAX_BYTES = 1024 * 1024
# The Soul Vault client of the family's main checkout, found the way the Coop
# client finds the family ledger: an explicit path, else the checkout under
# the home directory. Never the repository this script sits in.
_SOUL_VAULT_CLIENT_ENV = "FORREST_SOUL_VAULT_CLIENT"
_MAIN_CHECKOUT = ("bjtPersonalProjects", "bjtForrest")
_CARD_FIRST_WORDS = "[Soul Vault] identity loaded:"
_REFUSAL_FIRST_WORDS = "identity not loaded:"
# A harness that kills a slow hook opens the session with no text at all,
# neither card nor refusal. Codex gives this hook 30 seconds (.codex/hooks.json
# here and in every registered repository), so the whole run stays inside 25
# and the client's share inside 8: its two calls take about 0.2 seconds.
_HOOK_DEADLINE_SECONDS = 25
_IDENTITY_DEADLINE_SECONDS = 8
_COOP_DEADLINE_SECONDS = 20
# Claude Code keeps at most this many characters of SessionStart stdout in
# context. Above it the whole output is written to a tool-results file and only
# a 2,000-character preview reaches the model (observed on Claude Code 2.1.270,
# 2026-09-14; tracked upstream as anthropics/claude-code#44086). Codex hooks run
# with additionalContextLimit=0 and keep the full bootstrap.
# Claude Code caps a hook's plain stdout at 10,000 characters; over the cap the
# model gets the file path plus a preview of the first 2,000 characters
# (code.claude.com/docs/en/hooks, "Over the limit"). The cap counts
# characters, not bytes: on 2026-10-31 an 8,614-character, 15,507-byte
# bootstrap rode whole (that session's hook_success attachment). The host is
# JavaScript, so `_host_chars` counts UTF-16 units like its `.length`; the
# margin covers only the unknown >/>= boundary and the host's own wrapper.
_HOST_CONTEXT_MAX_CHARS = {"claude-code": 10_000}
_HOST_CONTEXT_MARGIN_CHARS = 200


def _host_chars(text: str) -> int:
    """The host's count of ``text``: UTF-16 code units, as JavaScript's ``.length``."""
    return len(text.encode("utf-16-le")) // 2
_DEMOTABLE_LABEL_PREFIX = "PROJECT_BRIEF.md"
_CONTEXT_CLOSE = "</forrest-terminal-context>"
_FORREST_PROTOCOL_BEGIN = "<!-- BEGIN:forrest-project-protocol -->"
_FORREST_PROTOCOL_END = "<!-- END:forrest-project-protocol -->"
_GUARDED_PATHS = (
    "AGENTS.md",
    "CLAUDE.md",
    ".claude/settings.json",
    ".codex/hooks.json",
    ".forrest/session_start.py",
    "tools/coop/coop.py",
)


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _append_managed_hook_alive(
    journal: Path,
    *,
    task_id: str,
    generation: int,
    agent: str,
    session_id: str,
) -> None:
    """Fsync one idempotent SessionStart liveness marker.

    The marker contains no conversation or command body.  Completion later
    requires this positive proof in addition to complete Pre/Post tool pairs,
    so a missing or broken hook installation can no longer look like an empty,
    therefore supposedly safe, journal.
    """
    journal.parent.mkdir(parents=True, exist_ok=True)
    lock_path = journal.with_suffix(journal.suffix + ".lock")
    session_digest = _sha256(session_id.encode("utf-8"))
    event_id = _sha256(
        f"{task_id}:{generation}:{agent}:{session_digest}:hook_alive".encode("utf-8")
    )
    event = {
        "schema_version": 1,
        "task_id": task_id,
        "generation": generation,
        "phase": "hook_alive",
        "event_id": event_id,
        "agent": agent,
        "session_digest": session_digest,
        "recorded_at": _now_iso(),
    }
    with lock_path.open("a+b") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        try:
            try:
                existing = journal.read_text(encoding="utf-8").splitlines()
            except FileNotFoundError:
                existing = []
            for line in existing:
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError("managed tool journal is incomplete") from exc
                if isinstance(value, dict) and value.get("event_id") == event_id:
                    return
            encoded = (
                json.dumps(event, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                + "\n"
            ).encode("utf-8")
            fd = os.open(str(journal), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
            try:
                view = memoryview(encoded)
                while view:
                    written = os.write(fd, view)
                    if written <= 0:
                        raise OSError("short managed hook liveness write")
                    view = view[written:]
                os.fsync(fd)
            finally:
                os.close(fd)
        finally:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)


def _managed_block(content: bytes, relative: str) -> bytes:
    text = content.decode("utf-8")
    if (
        text.count(_FORREST_PROTOCOL_BEGIN) != 1
        or text.count(_FORREST_PROTOCOL_END) != 1
    ):
        raise ValueError(f"{relative} lacks one complete Forrest protocol block")
    start = text.index(_FORREST_PROTOCOL_BEGIN)
    end = text.index(_FORREST_PROTOCOL_END, start) + len(_FORREST_PROTOCOL_END)
    return text[start:end].encode("utf-8")


def _managed_hooks(content: bytes, relative: str) -> bytes:
    parsed = json.loads(content.decode("utf-8"))
    hooks = parsed.get("hooks") if isinstance(parsed, dict) else None
    expected_agent = "claude-code" if relative.startswith(".claude") else "codex"
    event_names = (
        ("SessionStart", "PreToolUse", "PostToolUse", "PostToolUseFailure")
        if relative.startswith(".claude")
        else ("SessionStart", "PreToolUse", "PostToolUse")
    )
    managed: dict[str, dict[str, object]] = {}
    for event_name in event_names:
        groups = hooks.get(event_name) if isinstance(hooks, dict) else None
        found: list[dict[str, object]] = []
        for group in groups if isinstance(groups, list) else []:
            nested = group.get("hooks") if isinstance(group, dict) else None
            for item in nested if isinstance(nested, list) else []:
                if not isinstance(item, dict) or item.get("type") != "command":
                    continue
                command = item.get("command")
                if not isinstance(command, str):
                    continue
                if event_name == "SessionStart":
                    matches = (
                        ".forrest/session_start.py" in command
                        and f"--agent {expected_agent}" in command
                    )
                else:
                    matches = "FORREST_WORK_HOOK" in command
                if matches:
                    found.append(item)
        if len(found) != 1:
            raise ValueError(f"{relative} lacks one Forrest {event_name} hook")
        managed[event_name] = found[0]
    return json.dumps(
        managed,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _read_hook_payload() -> dict[str, object]:
    """Read the one hook payload without blocking manual invocations."""
    if sys.stdin.isatty():
        return {}
    raw = sys.stdin.buffer.read(_HOOK_PAYLOAD_MAX_BYTES + 1)
    if len(raw) > _HOOK_PAYLOAD_MAX_BYTES:
        raise ValueError("SessionStart payload is too large")
    if not raw.strip():
        return {}
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("SessionStart payload is invalid")
    return payload


def _record_managed_session_identity(payload: dict[str, object]) -> None:
    """Persist only the exact external session id for a managed generation."""
    target_raw = os.environ.get("FORREST_WORK_SESSION_FILE", "").strip()
    task_id = os.environ.get("FORREST_WORK_TASK_ID", "").strip()
    generation_raw = os.environ.get("FORREST_WORK_GENERATION", "").strip()
    token = os.environ.get("FORREST_WORK_SESSION_TOKEN", "").strip()
    journal_raw = os.environ.get("FORREST_WORK_TOOL_JOURNAL", "").strip()
    if not target_raw:
        return
    if not task_id or not generation_raw.isdigit() or not token or not journal_raw:
        raise ValueError("managed session identity is missing its Forrest fence")
    session_id = str(
        payload.get("session_id")
        or payload.get("sessionId")
        or payload.get("thread_id")
        or payload.get("threadId")
        or ""
    ).strip()
    if not session_id or len(session_id) > 200:
        raise ValueError("managed SessionStart payload has no exact session id")
    target = Path(target_raw).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "schema_version": 1,
        "task_id": task_id,
        "generation": int(generation_raw),
        "agent": os.environ.get("FORREST_WORK_AGENT", ""),
        "session_id": session_id,
        "token_digest": hashlib.sha256(token.encode()).hexdigest(),
    }
    encoded = (
        json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")
    temp = target.with_name(f".{target.name}.{secrets.token_hex(8)}.tmp")
    fd = os.open(str(temp), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        view = memoryview(encoded)
        while view:
            written = os.write(fd, view)
            if written <= 0:  # pragma: no cover - os.write contract
                raise OSError("short managed session identity write")
            view = view[written:]
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(temp, target)
    _append_managed_hook_alive(
        Path(journal_raw).expanduser().resolve(),
        task_id=task_id,
        generation=int(generation_raw),
        agent=str(record["agent"]),
        session_id=session_id,
    )


def _payload_text(payload: dict[str, object], *keys: str) -> str:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def verify_integration(repo_root: Path) -> list[str]:
    """Return drift failures for the sibling-owned SessionStart kit."""
    manifest_path = repo_root / ".forrest" / "project.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        integration = manifest.get("integration") if isinstance(manifest, dict) else None
        guards = integration.get("guard_hashes") if isinstance(integration, dict) else None
        binding = manifest.get("context_binding") if isinstance(manifest, dict) else None
        slug = manifest.get("slug") if isinstance(manifest, dict) else None
        guarded_paths = _GUARDED_PATHS
        version = integration.get("bundle_version") if isinstance(integration, dict) else None
        if type(version) is int and version >= 30:
            guarded_paths += ("tools/coop/family-ledger.json", "tools/coop/manifest.json")
        if not isinstance(guards, dict) or set(guards) != set(guarded_paths):
            raise ValueError("manifest guard_hashes are unavailable or incomplete")
        if (
            not isinstance(binding, dict)
            or binding.get("host_kind") != "git_repository"
            or not isinstance(slug, str)
            or binding.get("context_key") != f"app:{slug}"
        ):
            raise ValueError("manifest typed context_binding is invalid")
    except (
        FileNotFoundError,
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        ValueError,
    ) as exc:
        return [f"{manifest_path}: {exc}"]

    failures: list[str] = []
    for relative in guarded_paths:
        path = repo_root / relative
        try:
            content = path.read_bytes()
            if relative in {"AGENTS.md", "CLAUDE.md"}:
                guarded = _managed_block(content, relative)
            elif relative in {".claude/settings.json", ".codex/hooks.json"}:
                guarded = _managed_hooks(content, relative)
            else:
                guarded = content
            if _sha256(guarded) != guards[relative]:
                failures.append(f"{relative}: Forrest-managed content changed")
        except (
            FileNotFoundError,
            OSError,
            UnicodeDecodeError,
            json.JSONDecodeError,
            ValueError,
        ) as exc:
            failures.append(f"{relative}: {exc}")
    return failures


def _runtime_context_path() -> Path:
    explicit = os.environ.get("FORREST_RUNTIME_CONTEXT", "").strip()
    if explicit:
        return Path(explicit).expanduser().resolve()
    db_root = os.environ.get("FORREST_DB_ROOT", "").strip()
    root = Path(db_root).expanduser() if db_root else Path.home() / "forrest-db"
    return root.resolve() / _RUNTIME_CONTEXT_FILENAME


def _registration_retired(repo_root: Path) -> bool:
    """Whether Forrest has retired the registration this repository records.

    Retirement writes no Git commit, so the committed manifest cannot carry
    this and the backend's runtime descriptor is the only source. Every
    uncertainty — no descriptor, no key, no manifest — answers False, so a
    terminal never announces a retirement that did not happen.
    """
    try:
        manifest = json.loads(
            (repo_root / ".forrest" / "project.json").read_text(encoding="utf-8"),
        )
        task_id = manifest.get("task_id") if isinstance(manifest, dict) else None
        if not isinstance(task_id, str) or not task_id:
            return False
        payload = json.loads(
            _runtime_context_path().read_text(encoding="utf-8"),
        )
        retired = payload.get("retired_projects") if isinstance(payload, dict) else None
        return isinstance(retired, list) and task_id in retired
    except (
        FileNotFoundError,
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
    ):
        return False


def _profile_projection(
    value: object,
    *,
    expected_soul_id: str | None,
) -> dict[str, str] | None:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return None
    if not isinstance(value, dict):
        return None
    profile = {
        key: str(value.get(key) or "").strip()
        for key in ("soul_id", "display_name")
    }
    if not profile["soul_id"] or not profile["display_name"]:
        return None
    if expected_soul_id and profile["soul_id"] != expected_soul_id:
        return None
    return profile


class _IdentityRefused(Exception):
    """Identity did not load.

    Carries a code from Soul Vault's closed failure set with the observed
    cause, or the client's own refusal text, which is passed on verbatim.
    """

    def __init__(self, code: str = "", cause: object = "", *, said: str = "") -> None:
        super().__init__(said or f"{code}: {cause}")
        self.code = code
        self.cause = " ".join(str(cause).split())
        self.said = said

    def text(self, agent: str) -> str:
        """The whole refusal a session reads: one line, then the instruction to stop."""
        if self.said:
            return self.said if self.said.endswith("\n") else self.said + "\n"
        # The words of the bundled client's ``render.refusal_text``. The hook
        # says them only where the client could not: it is missing, it did not
        # start, it did not answer, or it could not be asked for one Soul.
        return (
            f"{_REFUSAL_FIRST_WORDS} {self.code}: {self.cause}\n"
            "\n"
            f"STOP. The soul vault did not load for {agent}. Do not answer as "
            "a generic assistant and do not invent an identity. Tell the owner exactly the line "
            "above (in Korean if he speaks Korean), and do nothing else until he fixes it or "
            "tells you otherwise.\n"
        )


def _selected_soul() -> tuple[dict[str, str] | None, str, list[str]]:
    """The registry coordinates of the Soul this session is for, and who gave them.

    Managed project work belongs to the Soul fixed on its source conversation.
    The fenced supervisor pins that Soul from the exact WorkUnit descriptor,
    and the pin outranks the runtime descriptor, which names only the default
    terminal Soul. A pin without complete coordinates is refused: a WorkUnit
    is never answered by the default Soul in its place.

    An ordinary terminal still opens when the backend has published no
    descriptor. Identity does not wait on a service; the missing coordinates
    are a warning.
    """
    work_soul_id = os.environ.get("FORREST_WORK_SOUL_ID", "").strip()
    work_profile = os.environ.get("FORREST_WORK_SOUL_PROFILE", "").strip()
    if work_soul_id or work_profile:
        profile = (
            _profile_projection(work_profile, expected_soul_id=work_soul_id)
            if work_soul_id
            else None
        )
        if profile is None:
            raise _IdentityRefused(
                "soul_not_found",
                "this managed WorkUnit does not name its Soul completely (soul id "
                f"{work_soul_id or 'missing'}, registry coordinates "
                f"{'unreadable' if work_profile else 'missing'}); the default Soul is "
                "not loaded in its place",
            )
        return profile, f"managed WorkUnit Soul {work_soul_id}", []

    descriptor = _runtime_context_path()
    try:
        payload = json.loads(descriptor.read_text(encoding="utf-8"))
        if (
            not isinstance(payload, dict)
            or payload.get("schema_version") != _RUNTIME_CONTEXT_SCHEMA_VERSION
        ):
            raise ValueError("unsupported descriptor")
    except (
        FileNotFoundError,
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        ValueError,
    ) as exc:
        return None, str(descriptor), [
            "ACTIVE RUNTIME CONTEXT UNAVAILABLE: "
            f"{descriptor} ({exc}). The selected Soul's registry coordinates "
            "are unknown until the Forrest backend publishes this descriptor."
        ]
    soul_id = payload.get("soul_id")
    profile = _profile_projection(
        payload.get("soul_profile"),
        expected_soul_id=soul_id if isinstance(soul_id, str) else None,
    )
    if profile is None:
        return None, str(descriptor), [
            "ACTIVE SOUL ROUTING UNAVAILABLE: runtime descriptor predates "
            "the canonical routing projection."
        ]
    return profile, str(descriptor), []


def _soul_vault_client() -> Path:
    explicit = os.environ.get(_SOUL_VAULT_CLIENT_ENV, "").strip()
    if explicit:
        return Path(explicit).expanduser()
    return Path.home().joinpath(*_MAIN_CHECKOUT, "tools", "soul-vault")


def _last_line(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else ""


def _run_client(
    client: Path, arguments: list[str], deadline: float,
) -> subprocess.CompletedProcess[str]:
    """Run the client once. Every way it cannot answer is a named refusal."""
    if not client.is_file():
        raise _IdentityRefused("service_unavailable", f"no Soul Vault client at {client}")
    try:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise subprocess.TimeoutExpired([str(client)], _IDENTITY_DEADLINE_SECONDS)
        return subprocess.run(
            [str(client), *arguments],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=remaining,
        )
    except subprocess.TimeoutExpired:
        raise _IdentityRefused(
            "service_unavailable",
            f"the Soul Vault client {client} did not answer within "
            f"{_IDENTITY_DEADLINE_SECONDS} seconds",
        ) from None
    except OSError as exc:
        raise _IdentityRefused(
            "service_unavailable",
            f"the Soul Vault client {client} could not start "
            f"({type(exc).__name__}: {exc})",
        ) from None


def _soul_vault(
    client: Path, agent: str, selected: dict[str, str] | None, deadline: float,
) -> str:
    """The store's name for the selected Soul's vault, taken from data alone.

    The client lists the souls its store holds (``health``). Nothing published
    yet joins a registry ``soul_id`` to a vault name, so the rule is the one
    that needs no join: the store's only soul vault, and only while it answers
    to the selected Soul's registry display name. A store with several soul
    vaults, and a session for a Soul that vault does not belong to (a guest,
    a Soul the record does not hold), are refused by name. A name is never
    answered from a neighbour's vault.
    """
    done = _run_client(client, ["health", "--harness", agent], deadline)
    if done.returncode != 0:
        said = _last_line(done.stderr)
        code, separator, cause = said.partition(": ")
        if separator and re.fullmatch(r"[a-z_]+", code):
            raise _IdentityRefused(code, cause)  # the client's own named failure
        raise _IdentityRefused(
            "internal_error",
            f"the Soul Vault client {client} could not list the store's souls "
            f"(exit {done.returncode}{': ' + said if said else ''})",
        )
    try:
        souls = [
            (str(soul["name"]), str(soul["display"]))
            for soul in json.loads(done.stdout)["souls"]
            if soul["kind"] == "soul"
        ]
    except (ValueError, KeyError, TypeError) as exc:
        problem = (
            f"{type(exc).__name__}: {exc}" if done.stdout.strip() else "it printed nothing"
        )
        raise _IdentityRefused(
            "internal_error",
            f"the list of souls from the Soul Vault client {client} could not be "
            f"read ({problem})",
        ) from None
    who = (
        f"Soul {selected['soul_id']} ({selected['display_name']})"
        if selected is not None
        else "the default Soul"
    )
    if len(souls) != 1:
        names = ", ".join(name for name, _display in souls) or "none"
        raise _IdentityRefused(
            "soul_not_found",
            f"the store holds {len(souls)} soul vaults ({names}), and nothing "
            f"published says which one {who} owns",
        )
    name, display = souls[0]
    if selected is not None and selected["display_name"] != display:
        raise _IdentityRefused(
            "soul_not_found",
            f"this session is for {who}, and the store's only soul vault, {name}, "
            f"answers to {display}; another Soul's identity is not loaded in its place",
        )
    return name


def _identity_card(agent: str) -> tuple[str, dict[str, str] | None, str, list[str]]:
    """Load identity through the client, or raise the refusal. Nothing else.

    Returns the card exactly as the client printed it, the selected Soul's
    registry coordinates, where those came from, and warnings about them.
    """
    selected, source, warnings = _selected_soul()
    client = _soul_vault_client()
    deadline = time.monotonic() + _IDENTITY_DEADLINE_SECONDS
    vault = _soul_vault(client, agent, selected, deadline)
    done = _run_client(
        client, ["bootstrap", "--harness", agent, "--soul", vault, "--hook"], deadline,
    )
    if done.stdout.startswith(_REFUSAL_FIRST_WORDS):
        raise _IdentityRefused(said=done.stdout)
    if done.returncode != 0 or not done.stdout.startswith(_CARD_FIRST_WORDS):
        said = _last_line(done.stderr)
        raise _IdentityRefused(
            "internal_error",
            f"the Soul Vault client {client} printed neither a card nor a refusal "
            f"(exit {done.returncode}{': ' + said if said else ''})",
        )
    return done.stdout, selected, source, warnings


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except (FileNotFoundError, OSError, UnicodeDecodeError):
        return ""


def _clip(text: str, max_bytes: int, *, source: Path | str) -> str:
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return text
    notice = (
        f"\n\n[bounded at {max_bytes} bytes; read the complete file from {source}]"
    ).encode("utf-8")
    if len(notice) >= max_bytes:
        return notice[:max_bytes].decode("utf-8", errors="ignore")
    clipped = encoded[: max_bytes - len(notice)].decode(
        "utf-8",
        errors="ignore",
    ).rstrip()
    return clipped + notice.decode("utf-8")


def _project_overlay(repo_root: Path) -> tuple[str, list[str]]:
    brief_path = repo_root / "PROJECT_BRIEF.md"
    brief = _read(brief_path)
    if not brief:
        return "", [f"Project overlay is unavailable: {brief_path}"]
    return (
        "--- PROJECT_BRIEF.md — app-owned project overlay "
        f"({brief_path}) ---\n"
        + _clip(brief, _PROJECT_BRIEF_MAX_BYTES, source=brief_path),
        [],
    )


def _coop_brief(
    repo_root: Path, agent: str, deadline: float | None = None,
) -> tuple[str, list[str]]:
    """Perform the letter-read ritual, then load the read-only session card.

    Both commands use the generated client, so an adopted ledger's read
    transition belongs to the office. Starting or completing work is separate.
    Keep successful letter output even if the following brief cannot load.
    Reading marks a letter read, so this runs only after identity loaded.
    """
    client = repo_root / "tools" / "coop" / "coop.py"
    if not client.is_file():
        return "", [f"COOP BRIEF UNAVAILABLE: client missing: {client}"]
    parts: list[str] = []
    warnings: list[str] = []
    own_deadline = time.monotonic() + _COOP_DEADLINE_SECONDS
    deadline = own_deadline if deadline is None else min(deadline, own_deadline)
    for command, label, failure in (
        (["letter", "read"], "Coop handoff letters", "COOP LETTERS UNAVAILABLE"),
        (["brief"], "live Coop session card", "COOP BRIEF UNAVAILABLE"),
    ):
        argv = ["python3", str(client), *command, "--agent", agent]
        try:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise subprocess.TimeoutExpired(argv, _COOP_DEADLINE_SECONDS)
            proc = subprocess.run(
                argv, cwd=repo_root, capture_output=True, text=True, timeout=remaining,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            warnings.append(f"{failure}: {exc}")
            continue
        output = proc.stdout.strip()
        if proc.returncode != 0 or not output:
            detail = (proc.stderr or output or f"exit {proc.returncode}: no output").strip()
            warnings.append(f"{failure}: {detail[:500]}")
            continue
        parts.append(f"--- {label} ---\n{output}")
    return "\n\n".join(parts), warnings


def _preserve_coop_context(text: str, agent: str) -> str:
    """A read letter cannot be recovered by rerunning the unread-only ritual.

    Preserve the complete output in a private temporary file before replacing
    it with a read-first pointer. This is a delivery copy, not session memory.
    The Coop database remains its durable owner.
    """
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", prefix="forrest-coop-context-", suffix=".txt",
            delete=False,
        ) as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
            source = Path(handle.name).resolve()
    except OSError as exc:
        return (
            "--- COOP HANDOFF UNAVAILABLE ---\n"
            f"The complete startup output could not be preserved: {exc}\n"
            "Letters may already be marked read. Before your first reply, use "
            f"`python3 tools/coop/coop.py letter list --all --agent {agent}` to find their IDs and "
            "`python3 tools/coop/coop.py show <id>` to read their bodies."
        )
    return (
        "--- Coop handoff and session card not inlined (host limit) ---\n"
        "Read the complete output with the Read tool before your first reply:\n"
        f"{source}\n"
        "Delivered letters are already marked read; rerunning `letter read` will not repeat them."
    )


def _split_section(section: str) -> tuple[str, str, str]:
    """Return (label, source, body) for one ``--- label (source) ---`` section."""
    first, _, body = section.partition("\n")
    head = first.strip()
    if not (head.startswith("--- ") and head.endswith(" ---")):
        return head, "", body
    inner = head[4:-4]
    if inner.endswith(")") and " (" in inner:
        label, _, source = inner[:-1].rpartition(" (")
        return label, source, body
    return inner, "", body


def _demotable(section: str) -> bool:
    label, source, _body = _split_section(section)
    return (
        label.startswith(_DEMOTABLE_LABEL_PREFIX)
        and bool(source)
        and Path(source).is_absolute()
    )


def _read_first_block(demoted: list[tuple[str, str, int]]) -> str:
    lines = [
        "--- project overlay not inlined (host limit) ---",
        "Read each one with the Read tool before your first reply, in this order:",
    ]
    for index, (label, source, chars) in enumerate(demoted, start=1):
        lines.append(f"{index}. {source} — {label} ({chars:,} chars)")
    return "\n".join(lines)


def _assemble(
    header: str,
    warnings: list[str],
    demoted: list[tuple[str, str, int]],
    sections: list[str],
    coop: str,
    *,
    limit: int | None,
    coop_read_first: str = "",
) -> str:
    # Both read-first pointers come straight after the header, ahead of
    # everything a last-resort cut may take.
    parts = [header]
    if coop_read_first:
        parts.append(coop_read_first)
    if demoted and limit is not None:
        parts.append(_read_first_block(demoted))
    if warnings:
        parts.append("--- bootstrap warnings ---\n" + "\n".join(warnings))
    parts.extend(sections)
    if coop:
        parts.append(coop)
    parts.append(_CONTEXT_CLOSE)
    return "\n\n".join(parts)


def _fit_host_context(
    agent: str,
    card: str,
    header: str,
    warnings: list[str],
    sections: list[str],
    coop: str,
) -> str:
    """Fit the hook's own context into what the host injects behind the card.

    The card is the client's text. It rides first and whole and is never
    clipped, demoted or reordered, so only what this hook owns yields: the
    project overlay first, named in a read-first block, then the Coop context,
    kept complete in a read-first file. Both pointers sit before the inline
    sections. The header and the registry coordinates stay inline.
    """
    limit = _HOST_CONTEXT_MAX_CHARS.get(agent)
    demoted: list[tuple[str, str, int]] = []
    coop_read_first = ""
    if len(coop.encode("utf-8")) > _COOP_MAX_BYTES:
        coop_read_first, coop = _preserve_coop_context(coop, agent), ""
    if limit is None:
        return _assemble(
            header, warnings, demoted, sections, coop, limit=None,
            coop_read_first=coop_read_first,
        )
    # What the card leaves of the host's window, less the newline that
    # separates the context from it and the one that ends the output.
    room = limit - _host_chars(card) - 2
    budget = room - _HOST_CONTEXT_MARGIN_CHARS
    sections = list(sections)

    def current() -> str:
        return _assemble(
            header, warnings, demoted, sections, coop, limit=limit,
            coop_read_first=coop_read_first,
        )

    while _host_chars(current()) > budget:
        index = next(
            (i for i in range(len(sections) - 1, -1, -1) if _demotable(sections[i])),
            None,
        )
        if index is None:
            break
        label, source, body = _split_section(sections[index])
        demoted.insert(0, (label, source, len(body)))
        del sections[index]
    # A short Coop card costs less than its recovery pointer; keep it inline.
    if _host_chars(current()) > budget and len(coop) > 512:
        coop_read_first, coop = _preserve_coop_context(coop, agent), ""
    if _host_chars(current()) > room and coop:
        coop_read_first, coop = _preserve_coop_context(coop, agent), ""
    rendered = current()
    if _host_chars(rendered) > room:
        # Last resort: never hand the host something it will replace with a
        # preview. The visible cut takes the tail of the hook's own text and
        # never a read-first pointer: a letter already marked read is reachable
        # only through its pointer. When the card leaves no room for the
        # header and the pointers, nothing is cut and the output goes whole
        # into the file the host keeps for it, card first.
        closer = f"\n\n[bounded at the host limit]\n\n{_CONTEXT_CLOSE}"
        pointers = _assemble(
            header, [], demoted, [], "", limit=limit, coop_read_first=coop_read_first,
        )
        keep = room - len(closer)
        if keep >= len(pointers) - len(_CONTEXT_CLOSE):
            rendered = rendered[:keep].rstrip() + closer
    return rendered


def _body_context(
    agent: str,
    repo_root: Path,
    body: str,
    card: str,
    profile: dict[str, str] | None,
    profile_source: str,
    warnings: list[str],
    deadline: float,
) -> str:
    """What the hook itself owns, fitted behind an identity card that loaded."""
    overlay = ""
    if body == "root":
        standing = f"""This is Forrest's terminal channel for the general development body at {repo_root}.

- Repository instructions and engineering notes are body overlays; the Soul
  loaded by the card above is the identity owner.
- This body is not a registered Git development body and has no PROJECT_BRIEF
  or project binder."""
    elif body == "registered-app":
        overlay, overlay_warnings = _project_overlay(repo_root)
        warnings.extend(overlay_warnings)
        retired = _registration_retired(repo_root)
        drift = verify_integration(repo_root)
        if drift and not retired:
            warnings.append(
                "INTEGRATION DRIFT: stop project work and restore/upgrade the "
                "Forrest kit: " + "; ".join(drift)
            )

        # A retired registration keeps its kit on purpose — retirement
        # re-scopes a surface instead of erasing it — so the honest thing is
        # to keep loading identity while saying plainly that the binder is
        # gone. Claiming to be a registered development body here would be the
        # one thing this file must never do.
        standing = (
            f"""This is Forrest's terminal channel for {repo_root}.
Forrest RETIRED this development project's registration. The repository and its kit stay, and
Forrest still works here with you, but the binder is gone:

- No canonical home conversation — nothing here lands in a Forrest project room.
- Work results no longer return to Forrest automatically; the Coop ledger below
  is still the shared record.
- Kit drift is informational now, not something to restore. Register the Git
  development body again from the WebUI if you want the binder back."""
            if retired
            else f"""This is Forrest's terminal channel for the registered Git development body at {repo_root}.

- PROJECT_BRIEF.md is this project's bounded overlay; Git is the code source of truth."""
        )
    else:
        raise ValueError(f"unsupported Forrest terminal body: {body}")
    host_limit = _HOST_CONTEXT_MAX_CHARS.get(agent)
    host_line = (
        f"\n- Host limit: {agent} keeps at most {host_limit:,} characters of this"
        "\n  bootstrap in context; the card rides whole, and what else does not"
        "\n  fit is named for you to Read before your first reply."
        if host_limit is not None
        else ""
    )
    # A bootstrap catalog (Memory L1): where identity came from and what this
    # host keeps. The isolation and Coop rules live in CLAUDE.md and in the
    # shared boundary, which the identity bundle carries.
    header = f"""<forrest-terminal-context>
{standing}

- Identity: the Soul Vault card above; registry coordinates from {profile_source}.
- Agent surface: {agent}{host_line}
""".rstrip()
    sections: list[str] = []
    if profile is not None:
        sections.append(
            "--- Soul registry — canonical routing identity "
            "(runtime canonical registry projection) ---\n"
            f"soul_id: {profile['soul_id']}\n"
            f"display_name: {profile['display_name']}"
        )
    if overlay:
        sections.append(overlay)
    # Last, so nothing above can fail after a handoff letter was marked read.
    coop, coop_warnings = _coop_brief(repo_root, agent, deadline)
    warnings.extend(coop_warnings)
    return _fit_host_context(agent, card, header, warnings, sections, coop)


def render(
    agent: str,
    repo_root: Path,
    *,
    body: str = "registered-app",
    notices: list[str] | None = None,
) -> str:
    """Everything SessionStart prints: the refusal alone, or the card and then the body.

    ``notices`` are the caller's own warnings about the hook payload. They
    join the body's warnings behind a card and are dropped with a refusal,
    which is printed alone.
    """
    deadline = time.monotonic() + _HOOK_DEADLINE_SECONDS
    try:
        card, profile, profile_source, warnings = _identity_card(agent)
    except _IdentityRefused as refused:
        return refused.text(agent)
    except Exception as exc:  # a defect is the refusal too, never a session without identity
        return _IdentityRefused("internal_error", f"{type(exc).__name__}: {exc}").text(agent)
    try:
        context = _body_context(
            agent, repo_root, body, card, profile, profile_source,
            [*(notices or []), *warnings], deadline,
        )
    except Exception as exc:  # the body's own context stays fail-open behind the card
        context = (
            "<forrest-terminal-context>\n"
            f"FORREST SESSION CONTEXT UNAVAILABLE: {exc}\n"
            "Identity loaded above. Do not assume the body overlay or Coop context loaded.\n"
            "</forrest-terminal-context>"
        )
    return f"{card}\n{context}\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", default="codex")
    parser.add_argument(
        "--body",
        choices=("registered-app", "root"),
        default="registered-app",
        help="terminal body whose non-identity overlay should be loaded",
    )
    parser.add_argument(
        "--verify-integration",
        action="store_true",
        help="CI/build gate: verify the sibling-owned Forrest kit and exit",
    )
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    if args.verify_integration:
        failures = verify_integration(repo_root)
        if failures:
            print("Forrest integration drift:")
            for failure in failures:
                print(f"- {failure}")
            return 2
        print("Forrest integration verified")
        return 0
    if hasattr(sys.stdout, "reconfigure"):
        # The card carries the identity note as the client wrote it: UTF-8,
        # whatever the session's locale.
        sys.stdout.reconfigure(encoding="utf-8")
    notices: list[str] = []
    try:
        hook_payload = _read_hook_payload()
    except Exception as exc:
        hook_payload = {}
        notices.append(f"FORREST SESSION PAYLOAD UNAVAILABLE: {exc}")
    event_name = (
        _payload_text(hook_payload, "hook_event_name", "hookEventName")
        or "SessionStart"
    )
    if event_name != "SessionStart":
        # A session opened before bundle 24 still holds the retired prompt
        # hook in its snapshot. Anything printed here would enter the model's
        # context on every prompt, and a non-zero exit would erase the prompt.
        return 0
    try:
        _record_managed_session_identity(hook_payload)
    except Exception as exc:
        # The managed session record remains fail-open. A missing record
        # disables exact conversation resume, so Forrest opens a new model
        # turn instead of guessing `--last`/`--continue`.
        notices.append(f"FORREST MANAGED SESSION ID UNAVAILABLE: {exc}")
    sys.stdout.write(
        render(str(args.agent), repo_root, body=str(args.body), notices=notices),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

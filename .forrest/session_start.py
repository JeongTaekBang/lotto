#!/usr/bin/env python3
"""Load the selected Forrest Soul into a Forrest terminal body.

The repository copy serves the general development body and is also vendored
as ``.forrest/session_start.py`` in every registered repository. Claude Code
and Codex run the appropriate copy at SessionStart. The identity itself is
never copied into a repository: each run reads the selected Soul vault fresh,
then adds only the body-specific context and the live Coop card.

The hook is deliberately fail-open. A broken optional identity note must not
prevent a terminal from opening, while a missing soul entry, missing
instructions file, an unusable existing core note, or unavailable Coop
database is made conspicuous in the model-visible output.

The bootstrap is a SessionStart product. Any other hook event exits silently
and writes nothing: bundles 22-23 shipped a ``UserPromptSubmit`` hook for a
terminal-session receipt that has had no reader since 2026-09-18, and Claude
Code snapshots its hooks when a session opens, so a session older than bundle
24 keeps calling this script on every prompt. Printing there would re-inject
the whole identity into the model's context each turn.
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


_IDENTITY_MAX_BYTES = 48 * 1024
_PROJECT_BRIEF_MAX_BYTES = 16 * 1024
_COOP_MAX_BYTES = 16 * 1024
_SKIP_PREFIXES = ("_", ".")
_RUNTIME_CONTEXT_FILENAME = "runtime-context.json"
_RUNTIME_CONTEXT_SCHEMA_VERSION = 1
_HOOK_PAYLOAD_MAX_BYTES = 1024 * 1024
# Claude Code keeps at most this many characters of SessionStart stdout in
# context. Above it the whole output is written to a tool-results file and only
# a 2,000-character preview reaches the model (observed on Claude Code 2.1.270,
# 2026-09-14; tracked upstream as anthropics/claude-code#44086). Codex hooks run
# with additionalContextLimit=0 and keep the full bootstrap.
_HOST_CONTEXT_MAX_CHARS = {"claude-code": 10_000}
_HOST_CONTEXT_MARGIN_CHARS = 1_000
_DEMOTABLE_LABEL_PREFIXES = ("core/", "shared core/", "PROJECT_BRIEF.md")
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
        if not isinstance(guards, dict) or set(guards) != set(_GUARDED_PATHS):
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
    for relative in _GUARDED_PATHS:
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


def _shared_root(value: object, warnings: list[str]) -> Path | None:
    """Read a published shared owner without guessing from the Soul root."""
    if value is None:
        return None  # Additive descriptor field; older descriptors have none.
    try:
        if not isinstance(value, str) or not Path(value).is_absolute():
            raise ValueError("expected an absolute shared vault path")
        return Path(value).resolve()
    except (OSError, RuntimeError, ValueError) as exc:
        warnings.append(f"SHARED SOUL CORE ROOT UNAVAILABLE: {exc}")
        return None


def _identity_source() -> tuple[
    Path,
    Path | None,
    dict[str, str] | None,
    str | None,
    Path | None,
    str,
    list[str],
]:
    """Resolve the same active identity pointers the WebUI backend published."""
    # Managed project work belongs to the Soul fixed on its source
    # conversation. The fenced supervisor pins these three values from the
    # exact WorkUnit descriptor. They outrank the global runtime-context file,
    # which describes only the default terminal surface.
    work_root = os.environ.get("FORREST_WORK_SOUL_VAULT_ROOT", "").strip()
    work_entry = os.environ.get("FORREST_WORK_SOUL_ENTRY", "").strip()
    work_soul_id = os.environ.get("FORREST_WORK_SOUL_ID", "").strip()
    if work_root or work_entry or work_soul_id:
        root = Path(work_root or ".").expanduser().resolve()
        source = f"managed WorkUnit Soul {work_soul_id or '<missing>'}"
        if not (work_root and work_entry and work_soul_id):
            return root, None, None, None, None, source, [
                "MANAGED WORKUNIT SOUL IDENTITY INCOMPLETE: refusing to mix "
                "it with the global Soul.",
            ]
        entry = Path(work_entry).expanduser().resolve()
        if entry != root and root not in entry.parents:
            return root, None, None, None, None, source, [
                "MANAGED WORKUNIT SOUL ENTRY ESCAPED ITS VAULT: identity "
                "entry was not loaded.",
            ]
        profile = _profile_projection(
            os.environ.get("FORREST_WORK_SOUL_PROFILE"),
            expected_soul_id=work_soul_id,
        )
        shared = os.environ.get("FORREST_WORK_SHARED_INSTRUCTIONS", "").strip()
        work_warnings: list[str] = []
        if profile is None:
            work_warnings.append(
                "MANAGED WORKUNIT SOUL ROUTING UNAVAILABLE: registry routing "
                "projection was not loaded."
            )
        if not shared:
            work_warnings.append(
                "MANAGED WORKUNIT SHARED SOUL INSTRUCTIONS UNAVAILABLE."
            )
        shared_root = _shared_root(
            os.environ.get("FORREST_WORK_SHARED_VAULT_ROOT"), work_warnings,
        )
        return root, entry, profile, shared or None, shared_root, source, work_warnings

    descriptor = _runtime_context_path()
    warnings: list[str] = []
    try:
        payload = json.loads(descriptor.read_text(encoding="utf-8"))
        raw_root = payload.get("vault_root") if isinstance(payload, dict) else None
        raw_soul = payload.get("soul_entry") if isinstance(payload, dict) else None
        raw_soul_id = payload.get("soul_id") if isinstance(payload, dict) else None
        if (
            not isinstance(payload, dict)
            or payload.get("schema_version") != _RUNTIME_CONTEXT_SCHEMA_VERSION
            or not isinstance(raw_root, str)
            or not raw_root.strip()
        ):
            raise ValueError("unsupported or incomplete descriptor")
        root = Path(raw_root).expanduser().resolve()
        soul = (
            Path(raw_soul).expanduser().resolve()
            if isinstance(raw_soul, str) and raw_soul.strip()
            else None
        )
        soul_id = raw_soul_id if isinstance(raw_soul_id, str) else None
        profile = _profile_projection(
            payload.get("soul_profile"),
            expected_soul_id=soul_id,
        )
        shared_raw = payload.get("shared_instructions")
        shared = shared_raw.strip() if isinstance(shared_raw, str) else ""
        if profile is None:
            warnings.append(
                "ACTIVE SOUL ROUTING UNAVAILABLE: runtime descriptor predates "
                "the canonical routing projection."
            )
        if not shared:
            warnings.append(
                "ACTIVE SHARED SOUL INSTRUCTIONS UNAVAILABLE: runtime "
                "descriptor predates the shared projection."
            )
        shared_root = _shared_root(payload.get("shared_vault_root"), warnings)
        return root, soul, profile, shared or None, shared_root, str(descriptor), warnings
    except (
        FileNotFoundError,
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        ValueError,
    ) as exc:
        warnings.append(
            "ACTIVE RUNTIME CONTEXT UNAVAILABLE: "
            f"{descriptor} ({exc}). Terminal identity is using a fallback; "
            "do not assume WebUI/terminal parity until the Forrest backend "
            "publishes this descriptor."
        )

    raw = os.environ.get("FORREST_VAULT_ROOT", "").strip()
    root = (
        Path(raw).expanduser().resolve()
        if raw
        else (Path.home() / "Forrest").resolve()
    )
    explicit_soul = os.environ.get("FORREST_SOUL_ENTRY", "").strip()
    if explicit_soul:
        configured = Path(explicit_soul).expanduser()
        soul = (
            configured.resolve()
            if configured.is_absolute()
            else (root / configured).resolve()
        )
    else:
        soul = None
    fallback = "FORREST_VAULT_ROOT" if raw else "default ~/Forrest"
    return root, soul, None, None, None, fallback, warnings


def _soul_entry(root: Path, configured: Path | None = None) -> Path:
    if configured is not None:
        return configured
    explicit = os.environ.get("FORREST_SOUL_ENTRY", "").strip()
    if explicit:
        configured = Path(explicit).expanduser()
        return (
            configured.resolve()
            if configured.is_absolute()
            else (root / configured).resolve()
        )
    candidates = list(dict.fromkeys((
        root / f"{root.name}.md",
        root / f"{root.name.casefold()}.md",
        root / "soul.md",
        root / "forrest.md",
    )))
    return next((path for path in candidates if path.is_file()), candidates[0])


def _owned_path(path: Path, root: Path) -> Path | None:
    """Resolve a terminal identity path without crossing its Soul vault."""
    try:
        owner = root.resolve()
        resolved = path.resolve()
    except (OSError, RuntimeError):
        return None
    if resolved != owner and owner not in resolved.parents:
        return None
    return resolved


def _read(path: Path, *, owner_root: Path | None = None) -> str:
    read_path = _owned_path(path, owner_root) if owner_root is not None else path
    if read_path is None:
        return ""
    try:
        return read_path.read_text(encoding="utf-8").strip()
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


def _identity_sections(
    root: Path,
    configured_soul: Path | None = None,
    profile: dict[str, str] | None = None,
    shared_instructions: str | None = None,
    shared_root: Path | None = None,
) -> tuple[list[str], list[str]]:
    entry = _soul_entry(root, configured_soul)
    profile_body = "\n".join(
        f"{key}: {profile.get(key, '')}"
        for key in ("soul_id", "display_name")
    ) if profile is not None else ""
    candidates: list[tuple[str, str, str, str | None]] = [
        (
            f"{entry.name} — named soul entry",
            str(entry),
            _read(entry, owner_root=root),
            f"CRITICAL: active soul entry is unavailable: {entry}",
        ),
        (
            "Soul registry — canonical routing identity",
            "runtime canonical registry projection",
            profile_body,
            "CRITICAL: active Soul registry routing is unavailable",
        ),
        (
            "shared-instructions.md — registry-wide inheritance",
            "runtime canonical shared-instructions projection",
            (shared_instructions or "").strip(),
            "CRITICAL: shared Soul instructions are unavailable",
        ),
    ]
    core_warnings: list[str] = []
    shared_start = len(candidates)
    if shared_root is not None:
        shared_candidates, shared_warnings = _core_candidates(
            shared_root, kind="shared soul",
        )
        candidates.extend(shared_candidates)
        core_warnings.extend(shared_warnings)
    shared_end = len(candidates)
    candidates.append((
        "instructions.md — soul operational delta",
        str(root / "instructions.md"),
        _read(root / "instructions.md", owner_root=root),
        "CRITICAL: active soul instructions are unavailable: "
        f"{root / 'instructions.md'}",
    ))
    local_candidates, local_warnings = _core_candidates(root)
    candidates.extend(local_candidates)
    core_warnings.extend(local_warnings)

    rendered_candidates = [
        f"--- {label} ({source}) ---\n{body}" if body else ""
        for label, source, body, _warning in candidates
    ]
    # Budget priority differs from inheritance order. Shared core may use
    # only bytes left after the pre-existing Soul spine, while still appearing
    # before its overrides. Counting a two-byte separator per section against
    # budget + 2 also handles a missing entry/profile without an off-by-two.
    spine_bytes = sum(
        len(rendered.encode("utf-8")) + 2
        for index, rendered in enumerate(rendered_candidates)
        if rendered and not shared_start <= index < shared_end
    )
    shared_budget = max(0, _IDENTITY_MAX_BYTES + 2 - spine_bytes)
    sections: list[str] = []
    warnings: list[str] = core_warnings
    used = 0
    shared_omission_warned = False
    for index, (label, source, body, missing_warning) in enumerate(candidates):
        if not body:
            if missing_warning is not None:
                warnings.append(missing_warning)
            continue
        rendered = rendered_candidates[index]
        is_shared = shared_start <= index < shared_end
        separator_bytes = 2 if sections else 0
        remaining = _IDENTITY_MAX_BYTES - used - separator_bytes
        if is_shared:
            remaining = min(remaining, shared_budget - 2)
        if remaining <= 0:
            if is_shared:
                if not shared_omission_warned:
                    warnings.append(
                        "Shared core byte budget omitted file(s) to preserve "
                        f"the Soul's own instructions/core; read {shared_root / 'core'} "
                        "for the complete shared layer."
                    )
                    shared_omission_warned = True
                continue
            omitted = [
                candidate_label
                for candidate_label, _source, _body, _warning in candidates[index:]
            ]
            warnings.append(
                "Identity byte budget omitted later file(s): "
                + ", ".join(omitted[:8])
                + (f" (+{len(omitted) - 8} more)" if len(omitted) > 8 else "")
            )
            break
        bounded = _clip(rendered, remaining, source=source)
        sections.append(bounded)
        used += separator_bytes + len(bounded.encode("utf-8"))
        if is_shared:
            shared_budget -= len(bounded.encode("utf-8")) + 2
            if bounded != rendered:
                warnings.append(
                    f"Shared core note bounded to preserve Soul overrides: {source}"
                )
            continue
        if bounded != rendered:
            omitted = [
                candidate_label
                for candidate_label, _source, _body, _warning
                in candidates[index + 1:]
            ]
            if omitted:
                warnings.append(
                    "Identity byte budget omitted later file(s): "
                    + ", ".join(omitted[:8])
                    + (f" (+{len(omitted) - 8} more)" if len(omitted) > 8 else "")
                )
            break
    return sections, warnings


def _prompt_mode(text: str) -> str | None:
    """Stdlib-only copy of the shared scalar contract in utils/prompt_notes.

    This file is vendored into repositories that cannot import the backend.
    Cross-host tests keep the note-authoring contract identical.
    """
    lines = text.strip().removeprefix("\ufeff").splitlines()
    if not lines or lines[0].strip() != "---":
        return None
    try:
        end = next(i for i in range(1, len(lines)) if lines[i].strip() == "---")
    except StopIteration:
        return "on-demand"
    values = [line.partition(":")[2].split("#", 1)[0].strip()
              for line in lines[1:end] if line.startswith("prompt:")]
    if not values:
        return None
    if len(values) != 1:
        return "on-demand"
    if re.fullmatch(r'''(?:inline|'inline'|"inline")''', values[0]):
        return "inline"
    return "on-demand"


def _core_candidates(
    root: Path, *, kind: str = "active soul",
) -> tuple[list[tuple[str, str, str, str | None]], list[str]]:
    """Fresh, sorted, owner-contained core reads for both inheritance layers."""
    candidates: list[tuple[str, str, str, str | None]] = []
    warnings: list[str] = []
    core = root / "core"
    if not core.is_dir():
        return candidates, warnings
    resolved_core = _owned_path(core, root)
    if resolved_core is None:
        return candidates, [
            f"CRITICAL: {kind} core directory escaped its vault: {core}",
        ]
    try:
        core_paths = sorted(resolved_core.rglob("*.md"))
    except OSError as exc:
        return candidates, [
            f"CRITICAL: {kind} core could not be enumerated: {core} ({exc})",
        ]
    for path in core_paths:
        if path.name.startswith(_SKIP_PREFIXES):
            continue
        rel = path.relative_to(root.resolve()).as_posix()
        label = f"shared {rel}" if kind == "shared soul" else rel
        body = _read(path, owner_root=root)
        if kind == "shared soul" and body and _prompt_mode(body) != "inline":
            body = f"On-demand note. Read `{path}` when needed."
        candidates.append((
            label,
            str(path),
            body,
            f"CRITICAL: {kind} core note is unavailable or outside its vault: {path}",
        ))
    return candidates, warnings


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


def _coop_brief(repo_root: Path, agent: str) -> tuple[str, list[str]]:
    client = repo_root / "tools" / "coop" / "coop.py"
    if not client.is_file():
        return "", [f"COOP BRIEF UNAVAILABLE: client missing: {client}"]
    try:
        proc = subprocess.run(
            ["python3", str(client), "brief", "--agent", agent],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=20,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return "", [f"COOP BRIEF UNAVAILABLE: {exc}"]
    output = proc.stdout.strip()
    if proc.returncode != 0:
        detail = (proc.stderr or output or f"exit {proc.returncode}").strip()
        return "", [f"COOP BRIEF UNAVAILABLE: {detail[:500]}"]
    if not output:
        return "", ["COOP BRIEF UNAVAILABLE: client returned no session card"]
    return (
        "--- live Coop session card ---\n"
        + _clip(output, _COOP_MAX_BYTES, source=client),
        [],
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
        label.startswith(_DEMOTABLE_LABEL_PREFIXES)
        and bool(source)
        and Path(source).is_absolute()
    )


def _clip_chars(text: str, max_chars: int, *, source: str) -> str:
    if len(text) <= max_chars:
        return text
    notice = f"\n\n[bounded at {max_chars} characters; rerun or read {source}]"
    if len(notice) >= max_chars:
        return notice[:max_chars]
    return text[: max_chars - len(notice)].rstrip() + notice


def _read_first_block(demoted: list[tuple[str, str, int]], limit: int) -> str:
    lines = [
        "--- always-on self not inlined (host limit) ---",
        f"Claude Code keeps at most {limit:,} characters of SessionStart output in",
        "context; a larger bootstrap is replaced by a 2,000-character preview. The",
        "files below are part of the always-on identity the WebUI loads every turn.",
        "Read each one with the Read tool before your first reply, in this order,",
        "and do not answer from this bootstrap alone:",
    ]
    for index, (label, source, chars) in enumerate(demoted, start=1):
        lines.append(f"{index}. {source} — {label} ({chars:,} chars)")
    lines.append(
        "index/, playbook/, memory/, and library/ stay on demand as "
        "instructions.md directs."
    )
    return "\n".join(lines)


def _assemble(
    header: str,
    warnings: list[str],
    demoted: list[tuple[str, str, int]],
    sections: list[str],
    coop: str,
    *,
    limit: int | None,
) -> str:
    parts = [header]
    if warnings:
        parts.append("--- bootstrap warnings ---\n" + "\n".join(warnings))
    if demoted and limit is not None:
        parts.append(_read_first_block(demoted, limit))
    parts.extend(sections)
    if coop:
        parts.append(coop)
    parts.append("</forrest-terminal-context>")
    return "\n\n".join(parts)


def _fit_host_context(
    agent: str,
    header: str,
    warnings: list[str],
    sections: list[str],
    coop: str,
) -> str:
    """Keep the bootstrap inside what the host actually injects.

    Later layers yield first (project overlay, then core notes in reverse
    load order); each demoted file is named in a read-first block placed
    before the inline sections so that even a preview still carries the map.
    The Soul entry, registry, shared boundary, and instructions stay inline.
    """
    limit = _HOST_CONTEXT_MAX_CHARS.get(agent)
    demoted: list[tuple[str, str, int]] = []
    if limit is None:
        return _assemble(header, warnings, demoted, sections, coop, limit=None)
    budget = limit - _HOST_CONTEXT_MARGIN_CHARS
    sections = list(sections)

    def current() -> str:
        return _assemble(header, warnings, demoted, sections, coop, limit=limit)

    while len(current()) > budget:
        index = next(
            (i for i in range(len(sections) - 1, -1, -1) if _demotable(sections[i])),
            None,
        )
        if index is None:
            break
        label, source, body = _split_section(sections[index])
        demoted.insert(0, (label, source, len(body)))
        del sections[index]
    if len(current()) > budget and coop:
        excess = len(current()) - budget
        coop = _clip_chars(
            coop,
            max(0, len(coop) - excess),
            source="`python3 tools/coop/coop.py brief --agent " + agent + "`",
        )
    rendered = current()
    if len(rendered) > limit:
        # Last resort: never hand the host something it will replace with a
        # preview. The cut is visible and names where the rest lives.
        closer = "\n\n[bounded at the host limit; read instructions.md and core/ from the vault]\n\n</forrest-terminal-context>"
        rendered = rendered[: limit - len(closer)].rstrip() + closer
    return rendered


def render(agent: str, repo_root: Path, *, body: str = "registered-app") -> str:
    (
        vault,
        configured_soul,
        soul_profile,
        shared_instructions,
        shared_root,
        identity_source,
        warnings,
    ) = _identity_source()
    identity, identity_warnings = _identity_sections(
        vault,
        configured_soul,
        soul_profile,
        shared_instructions,
        shared_root,
    )
    warnings.extend(identity_warnings)
    coop, coop_warnings = _coop_brief(repo_root, agent)
    warnings.extend(coop_warnings)
    overlay = ""
    if body == "root":
        standing = f"""This is Forrest's terminal channel for the general development body at {repo_root}.

- Repository instructions and engineering notes are body overlays; the
  selected Soul vault below is the identity owner.
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
        "\n  bootstrap in context; always-on files that do not fit are listed"
        "\n  first for you to Read before your first reply."
        if host_limit is not None
        else ""
    )
    header = f"""<forrest-terminal-context>
{standing}

- Persistent Soul and Brain are orthogonal; this terminal app is an execution
  surface, never the identity source.
- Identity below was read fresh from the selected Soul vault at {vault}.
- Active identity pointers came from {identity_source}.
- Coop is the shared work ledger. If its card is unavailable, do not mistake that
  for an empty inbox; restore access and rerun the brief before project work.
- Keep WebUI identity injection separate. This terminal bootstrap must never be
  enabled as a setting source inside Forrest's WebUI SDK turns.
- Agent surface: {agent}{host_line}
""".rstrip()
    sections = list(identity)
    if overlay:
        sections.append(overlay)
    return _fit_host_context(agent, header, warnings, sections, coop)

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
    try:
        hook_payload = _read_hook_payload()
    except Exception as exc:
        hook_payload = {}
        print(f"FORREST SESSION PAYLOAD UNAVAILABLE: {exc}")
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
        # SessionStart remains fail-open. A missing identity proof disables
        # exact conversation resume, so Forrest opens a new model turn instead
        # of guessing `--last`/`--continue`.
        print(f"FORREST MANAGED SESSION ID UNAVAILABLE: {exc}")
    try:
        print(render(str(args.agent), repo_root, body=str(args.body)))
    except Exception as exc:  # SessionStart must remain fail-open.
        print(
            "<forrest-terminal-context>\n"
            f"FORREST SESSION BOOTSTRAP UNAVAILABLE: {exc}\n"
            "Open the terminal, but do not assume identity or Coop context loaded.\n"
            "</forrest-terminal-context>"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

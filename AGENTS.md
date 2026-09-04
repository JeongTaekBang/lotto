<!-- BEGIN:forrest-project-protocol -->
## Forrest project bridge

This existing repository is registered through Forrest as the Git development
body `lotto`. It is not a runtime sibling app, Soul, or Brain. Its durable
descriptor remains `git_repository` / `app:lotto`; a future runtime sibling
requires a separate host kind, pull adapter, and system folder. Git and the
working tree remain the source of code truth; Forrest WebUI injects a verified
read-only project view into its canonical conversation.

That canonical-home conversation owns the immutable `soul_id`; a managed
WorkUnit carries it, while either terminal Brain reads only the selected Soul
vault plus this project's bounded brief at SessionStart. The eligible source
conversation normally becomes home, but an unavailable/already-bound or
Council/global source receives a server-minted default-Soul home. Editable
`source_ref` text is never identity authority. Without a managed WorkUnit, an
ordinary terminal uses the current runtime-selected/default Soul and does not
pin, bind, or adopt project identity.
It pins only that selected Soul in a metadata receipt for bounded idle return
to the Soul's `memory/inbox/`; full transcript Adopt stays explicit.

Confirm startup output contains `<forrest-terminal-context>`. If it did not
run, execute the shared bootstrap explicitly:

    python3 .forrest/session_start.py --agent codex

Before body work, run:

    python3 .forrest/session_start.py --verify-integration

If it fails, stop and restore/upgrade the Forrest-owned protocol, hooks, or
client. `.forrest/project.json` owns the typed `context_binding`; Forrest
resolves it to one canonical home conversation, so do not derive or rewrite
that address from a display label, path, or cwd.

On the ground-truth host the client needs read/write access to
`$HOME/forrest-db/coop.db` and its parent directory. `COOP BRIEF UNAVAILABLE`
means the inbox and WorkUnits were not loaded, not that they are empty.
Request/allow access and rerun the command before starting project work.

Retrieve relevant history before deep diagnosis, then record a verified unit
with `coop.py log` and reusable lessons with `coop.py learn`. The generic
mailbox (`send`, `inbox`, `message-claim`, `reply`, `message-done`) is available
when another Forrest, Codex, or Claude builder would materially help. Coop is
logically scoped by this Git root's `lotto` project name and does not replace
Git history or the repository's existing engineering rules.

After a non-trivial implementation is verified, committed, and pushed, Codex
should normally send one `peer_review` request for the exact `git-head` to
`claude-code`. First search `coop.py messages --query "<full-sha>"`; skip the
handoff when the change is trivial and never duplicate a request or result for
the same head. Claude replies in-thread, Codex evaluates the feedback, and
review-only work never requests another review.

If `brief`/`inbox` shows `project_implement`, `project_review`, or
`project_refine`, inspect that WorkUnit with `message-show <id>`, claim it, and
follow its objective and acceptance criteria. After verification, commit and
push the exact candidate. A managed WorkUnit (`FORREST_WORKSPACE=1`) stays in
its generation-specific private clone and uses its local staging remote with
`git push origin HEAD:$FORREST_TARGET_BRANCH`; it must not replace that remote
or push the registered project's real origin. Forrest promotes only the current
verified generation. An ordinary session follows the repository's existing
push rule. Then reply with `--intent work_result`, a concise
result body, and `--ref "git-head:$(git rev-parse HEAD)"`. A reply without the
result Git ref is incomplete. Codex uses `--agent codex`; Claude Code can rely
on its automatic `claude-code` identity. Forrest writes a valid result to the
canonical home JSONL before acknowledging it; do not manually paste or import
the returned work into chat.
<!-- END:forrest-project-protocol -->

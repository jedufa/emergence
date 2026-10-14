# EIEL Context Epoch and Bounded Output Policy

**Document role:** Normative operating policy, introduced in V1.9 and retained in V1.11.

See the internal `AGENTS.md` (not distributed; see [contributor guidance](../../CONTRIBUTING.md)) for repository-wide scope and model-use defaults.

## Purpose

This policy reduces Codex context growth without weakening repository authority.
It does not create a Context Manager, memory service, scheduler, model router,
or transcript archive.

The repository owns truth. The explicit operator instruction owns direct
intent. A bound Work Package owns its formal contract. A Session owns retained
progress. The conversation owns nothing permanent.

## Normal unit of work

One Codex conversation should execute one bounded operator instruction, Work
Package, or tightly coupled authority phase:

- implementation;
- bounded repair;
- review;
- qualification;
- merge or consolidation.

A Session normally corresponds to one Context Epoch. A fresh conversation is
recommended when the objective or authority role changes.

## Epoch boundary

Checkpoint and hand off when any of these occurs:

- the package objective is complete;
- implementation reaches `IMPLEMENTATION_COMPLETE`;
- promotion is requested;
- Review or Qualification Authority becomes the next authority;
- owned scope changes materially;
- the first compaction occurs for ordinary work;
- the package advisory tool-call threshold is reached;
- a clean checkpoint exists and continued work would carry mostly history.

Workspace only recommends the boundary. It does not end conversations or launch
another model.

## Initial working set

A Work Package may define a `read_contract` with:

- `entry_files`;
- `required_documents`;
- `search_roots`;
- `excluded_roots`;
- `known_symbols`;
- `maximum_initial_files`;
- `broad_search_allowed`.

Start with the declared files. Expand only when the initial contract is
insufficient, and record why. The read contract limits initial exploration; it
does not fabricate authority or prohibit justified inspection.

## Bounded command output

A Work Package may define an `output_policy`. Default behavior is compact:

- quiet tests and failure summaries;
- short Git status or diff statistics;
- selected JSON fields;
- bounded log tails;
- reason required before broad recursive output.

For large output, redirect to a file and inspect a bounded summary. Do not place
full logs, giant JSON documents, broad recursive searches, or raw campaign trees
into the conversation merely because the terminal is willing to do so.

## Advisory execution budget

A Work Package may define an `execution_budget`. Counters are advisory and must
never be invented. When telemetry is unavailable, Workspace reports
`NOT_OBSERVED`.

The initial policy supports:

- expected tool calls;
- a handoff recommendation threshold;
- maximum compactions;
- expected duration;
- advisory context-health policy.

No hard termination, model routing, or automatic conversation creation exists.

## Context Receipt

A Context Receipt is an immutable projection of the minimum working set needed
to continue. It binds:

- repository and commit;
- Work Package path, version, and digest when a package is bound;
- Work and Session IDs;
- authority role and epoch;
- initial read set and file fences;
- output and budget policy;
- current progress, blockers, validation, and next action;
- authority and critical-file identities;
- context-health observations.

It excludes raw command output, search history, reasoning traces, complete tool
transcripts, copies of authority documents, and old conversation summaries.

Without a Work Package, a receipt preserves an explicitly unscoped snapshot and
makes no managed acceptance claim. A receipt may warn on unrelated Git HEAD
drift. It marks itself stale or invalid when package, Session, Work, authority,
or declared critical files change incompatibly; that prevents trusting the
receipt, not operator-authorized work.

## Standard commands

```bash
python -m tools.eiel start --human --output /tmp/eiel-start.json
python -m tools.eiel sessionManager checkpoint SES-... \
  --tool-call-count 42 \
  --compaction-count 1 \
  --epoch-boundary-reason "implementation complete"
python -m tools.eiel sessionManager context-receipt SES-... --attach --human
python -m tools.eiel handoff --work-id WORK-... --session-id SES-...
```

## Stop rule

The objective is not to preserve conversations. The objective is to make them
disposable while preserving deterministic engineering continuity.

## V1.11 compact opening and authority-read discipline

The optional compact recovery front door is:

```bash
python -m tools.eiel start --human --output /tmp/eiel-start.json
```

Use `--full` only for forensic inspection. When exact machine data must remain
available without entering the conversation, use:

```bash
python -m tools.eiel start --output /tmp/eiel-start-full.json
```

The console remains compact while the file contains the complete deterministic
Workspace payload.

Global repository instructions already supplied to the active Codex context do
not need to be shell-read again merely because they are authority. Re-read them
only when their digest changed, scope changed, the active instruction context
did not supply them, or an exact disputed section is required. Work Package
`required_documents` should normally list task-specific documents, not duplicate
`AGENTS.md` or the complete operator guide.

The compact projection reports exact byte and line counts for the initial files
named by the Work Package. Large or missing files produce an advisory warning;
they do not silently expand authority and do not block justified inspection.

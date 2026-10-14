# EIEL Codex Operator Tooling V1.17

> Repository-wide scope and model-use defaults: the internal `AGENTS.md` (not distributed; see [contributor guidance](../../CONTRIBUTING.md)).

**Document role:** Manager ownership and detailed reference. Use `OPERATOR_QUICKSTART.md` for ordinary daily operation.

Deterministic, local, file-backed tools for onboarding, tracking, resuming, and
closing Codex work without reconstructing the laboratory from chat memory.

## Optional recovery front door

```text
./tools/bin/eiel start
```

This delegates to the read-only Workspace. It reports Bootstrap and manager
state without granting or revoking execution authority. Stale receipts and
records from other repository lanes are exposed as administrative obligations;
they do not block unrelated work explicitly authorized by the operator.

The repository-local launcher resolves the worktree `.venv/` first and then
the shared CUDA environment. Use `./tools/bin/emergence-python -m <module>`
for other Python commands from a worktree.

No Work Item, Work Package, or Session is required to begin authorized work.
Use them when their recovery, provenance, coordination, or formal acceptance
value justifies the record.

Top-level recovery shortcuts are:

```text
python -m tools.eiel start
python -m tools.eiel resume
python -m tools.eiel status
python -m tools.eiel audit
python -m tools.eiel handoff
```

They delegate only to `python -m tools.eiel workspace`.

V1.11 makes `start`, `resume`, and `status` display a bounded structured
projection by default. The projection preserves repository/Bootstrap identity,
selected Work and Session, Work Package digest, exact next action, initial file
fences, output policy, context health, and Context Receipt identity without
embedding complete actor records.

```text
python -m tools.eiel start --full
python -m tools.eiel start --output /tmp/eiel-start-full.json
python -m tools.eiel workspace open --compact
```

Direct `workspace open` remains full by default for compatibility. `--output`
always stores the full deterministic payload even when the console is compact.

## Tools and ownership

| Tool | Owns | Does not own |
|---|---|---|
| `bootstrap` | repository, environment, authority-preflight and validation readiness | setup, installation, edits or execution |
| `workspace` | consolidated read-only context, consistency audit and resume handoff | record mutation or automatic selection |
| `workManager` | durable work portfolio, dependencies, Work Package binding and review-gated completion | execution or scientific priority |
| `sessionManager` | exact Codex attempt, authority/model assignment, package binding, promotion stop and review verdict | work completion or Codex launching |
| `missionPlanner` | structure of already-authorized scientific/technical intent | programme authority or campaign execution |
| `qualificationManager` | bounded technical authority lookup and records | campaign execution or scientific qualification |
| `resourceManager` | read-only resource observation and advisory admission | reservation, scheduling or process control |
| `evidenceManager` | procedural evidence assessment through the existing EDL | mechanics changes or scientific truth |
| `bridgeManager` | optional explicit ChatGPT transport | repository authority or ordinary operation |
| `token` | WRA-1 token snapshots, immutable attribution artifacts and exports through `ccusage` | Codex log parsing, costs, Work/Session lifecycle authority or estimator learning |
| `assignment` | deterministic compilation of explicit Dispatch Requests into immutable Assignments | work selection, project truth or execution |
| `scheduler` | optional single-worker launch, Lease heartbeat, Result collection and restart reconciliation | work creation, authority, acceptance or manual operation |

## Work Item Resource Accounting (WRA-1)

V1.17 adds automatic token evidence without adding another manager. `ccusage`
parses Codex telemetry; EIEL binds normalized cumulative snapshots and deltas to
the Work Item, Session, Context Epoch and Codex thread. Work creation adds a
static estimate. Session lifecycle boundaries capture START/END snapshots, and
Work completion writes an immutable `token_usage.json`.

```text
python -m tools.eiel token capture-start SES-...
python -m tools.eiel token capture-end SES-...
python -m tools.eiel token finalize-work WORK-... --correction-note "..."
python -m tools.eiel token-report --output-dir /tmp/eiel-token-report
```

Missing or invalid telemetry produces explicit `UNAVAILABLE` evidence and never
invents counters or blocks valid engineering completion. Work records retain only
the estimate, artifact reference and estimation error; raw telemetry is never
stored. Corrections create new immutable artifacts. Estimator evolution remains
a manual offline review followed by human-approved edits to
`tools/actors/estimator_rules.md`.

See `tools/actors/WRA1_WORK_ITEM_RESOURCE_ACCOUNTING.md`.

## Time scales

```text
Bootstrap      one current repository/environment check
Work item      days to months
Codex session  minutes to days
Mission        the lifetime of an authorized technical/scientific objective
```

A work item may own many sessions. A session may be resumed by the original or a
replacement Codex. Mission records are optional for purely mechanical work and
mandatory only when the work supports a mission requiring that authority.

## Multi-model engineering authority

The stable process abstraction is engineering authority, not a model name:

```text
Architecture Authority
  -> Engineering Authority
     -> Implementation Authority
        -> Review Authority
```

The human selects the current model for each session. A session may record the
actual model and reasoning level. The core managers do not route models or start
terminals. The optional Scheduler may launch only the explicit worker profile
recorded by the human in a Dispatch Request.

The current recommended mapping may change without changing the workflow:

| Authority | Typical purpose | Preferred model today |
|---|---|---|
| Architecture Authority | architecture, scientific uncertainty, deep audit | GPT-5.6 Ultra |
| Engineering Authority | programme planning and deterministic Work Packages | GPT-5.5 High |
| Implementation Authority | bounded implementation | GPT-5.3 Spark High |
| Review Authority | independent verification, repair decision and acceptance | GPT-5.5 High |

## Work Package: an optional formal engineering contract

A simple Programme document may describe 20–50 Work Items, dependencies,
manual parallel groups, consolidation, integration and final review. It is
planning guidance, not another manager or executable schema.

Each implementable Work Item may bind one repository-native YAML or JSON Work
Package. The package is deliberately small and immutable while sessions use it:

```text
Work ID
Goal
Scope
Owned files
Forbidden files
Dependencies
Acceptance criteria
Tests
Evidence
Stop conditions
Recommended authority
Recommended model
Uncertainty class
```

Mutable lifecycle state remains in Work Manager and Session Manager. It does
not belong in the Work Package.

Create Work from a package:

```text
python -m tools.eiel workManager create \
  --work-package work_packages/WORK-042.yaml \
  --repo-root /tmp/emergence-example/user/emergence \
  --status ACTIVE
```

Work Manager requires a Git-tracked, committed, clean source and records the
repository-relative path, package version, source commit, canonical contract
SHA-256, source-file SHA-256 and normalized-document SHA-256. YAML and
JSON are normalized to the same canonical JSON contract before hashing. Duplicate
keys, YAML aliases/merges, ambiguous scalar coercion, repository escape, case
mismatch and symlink traversal fail closed.

The canonical contract digest covers goal, scope, file fences, dependencies,
acceptance, tests, evidence, stop conditions, authority and uncertainty. Display
title, current model recommendation and source formatting are advisory or
non-material. Once a Session has used a package, a material revision must use a
new repository path and higher `package_version`; historical Sessions retain the
old digest. The tooling never guesses that a wording change inside a material
field is merely cosmetic.

File fences are lexical. A repository-relative literal owns exactly that path;
recursive ownership requires an explicit glob such as `tools/new_component/**`.
Changing a file into a directory never grants descendant authority. A delivery
session may complete only from a clean worktree whose committed changes stay
within `owned_files`, outside `forbidden_files`, and do not alter any current or
historical Work Package source path.

Examples:

```text
tools/actors/examples/programme.example.md
tools/actors/examples/work_package.example.yaml
```

## Promotion and review

Implementation Authority must not redesign architecture. If uncertainty is
higher than the approved package permits, stop and record a promotion:

```text
python -m tools.eiel sessionManager request-promotion SES-... \
  --requested-class U4 \
  --reason "Shared ownership contract is unresolved." \
  --affected tools/shared_contract.py \
  --safe-state "Bounded changes are committed and focused tests pass."
```

That managed Session cannot be resumed as though the package were unchanged.
Repair or supersede the record when it remains useful. The operator may still
authorize a replacement or independent bounded task directly.

A package-bound delivery session ends with a typed outcome such as
`IMPLEMENTATION_COMPLETE`. Work is still open. A separate Review Authority
session records one verdict:

```text
ACCEPT
CHANGES_REQUIRED
REJECT
PROMOTE
BLOCKED
```

The review is a separate Session and authority attempt. It may use the same model
family, but it must use a different Codex identity and explicitly target the
latest delivery Session. The Review Session binds the implementation Session,
implementation commit, package digest, implementing/reviewing identities,
tests reviewed, evidence reviewed and known limitations.

```text
python -m tools.eiel sessionManager start WORK-042 \
  --role REVIEW_AUTHORITY \
  --review-target-session SES-IMPLEMENTATION-042 \
  --codex-id review-context-042 ...
```

`CHANGES_REQUIRED` may authorize a bounded repair inside the same package. The
repair is a new delivery Session; its commit makes every earlier review stale.
A new Review Session must target that repair. `PROMOTE` requires a revised
package because the engineering contract or uncertainty has changed.

Only an accepted review of the latest delivery may close package-bound Work:

```text
python -m tools.eiel workManager complete WORK-042 --review-session SES-REVIEW-042
```

Ordinary force flags cannot bypass promotion or review. Corrupted historical
state has a separate administrative reconciliation path that requires a reason
and explicit acknowledgement, records permanent history, ends in
`DISCONTINUED` or `OBSOLETE`, and never implies acceptance or releases
dependencies.

Unpackaged V1 work retains its existing lifecycle behavior.

## Normal flow

```text
1. python -m tools.eiel start
2. Run the exact bootstrap command if requested.
3. python -m tools.eiel start --bootstrap-record BST-...
4. Select WORK-... or SES-... explicitly when several tracks exist.
5. Resume or start the session. For package-bound work, record the authority and actual model.
6. Use mission/qualification/resource/evidence managers only when the work needs them.
7. Execute campaigns through existing campaign tooling.
8. Complete the delivery session, then use a separate Review Authority session.
9. Checkpoint, interrupt, or request promotion before stopping.
10. Run python -m tools.eiel audit periodically and python -m tools.eiel handoff before transfer.
```

## Bootstrap refresh

Workspace validates the bootstrap against the current Git root and HEAD. Receipts older than 24 hours produce a refresh recommendation. A strict caller
can enforce an age limit with `--max-bootstrap-age-hours`. A paused session can
be rebound safely:

```text
python -m tools.eiel sessionManager resume SES-... \
  --bootstrap-record BST-NEW \
  --codex-id <current-codex-id>
```

When Workspace finds a stale receipt from the same worktree, its next action
now uses `bootstrap refresh BST-...`. That command retains the previous
non-secret task classification inputs, verifies the live worktree identity,
and records a fresh receipt; it refuses a receipt from another repository lane.

The previous bootstrap IDs remain in `bootstrap_history`.

## State and integrity

Default state root:

```text
${XDG_STATE_HOME:-$HOME/.local/state}/eiel/actors
```

Override with `EIEL_ACTOR_STATE_ROOT` or `--state-root`.

Records use canonical JSON envelopes, stable IDs, SHA-256 seals, append-only
revision history, atomic writes, and file locks. `python -m tools.eiel audit` checks current
records, histories, indexes, work dependencies, Work Package digests, review
independence, lifecycle consistency, and cross-manager references without
repairing anything automatically.

## Optional bridge

`python -m tools.eiel bridgeManager` runs only when explicitly invoked. Removing `bridge/` must
not block Bootstrap, Workspace, work/session recovery, missions, qualification,
resources, evidence, campaign execution, or Librarian publication.

See `tools/actors/CODEX_OPERATOR_GUIDE.md` for commands and examples.

## Context Epoch and Context Receipt

Introduced in V1.9 and retained in V1.11, the Context Epoch model adds no Context Manager. Work Packages may optionally define:

- `read_contract`;
- `output_policy`;
- `execution_budget`.

Session Manager records one bounded Context Epoch and observed counters. Workspace
projects the initial working set and context health. A Context Receipt carries
only decision-relevant continuation state and remains bound to the exact package,
Work, Session, authority role, repository and critical files.

```bash
python -m tools.eiel sessionManager context-receipt SES-... --attach
```

See `tools/actors/CONTEXT_EPOCH_POLICY.md`, `tools/actors/CONTEXT_EFFICIENCY_AB_PROTOCOL.md`, and
`tools/actors/V1_9_NEXT_SLICE_HANDOFF.md`.


## Optional Assignment Linker and Scheduler

Manual operation remains the primary path. No ordinary command consults the
Scheduler, and no Work becomes automated merely because it is eligible.

```text
Human Dispatch Request
  -> Assignment Linker compiles immutable Assignment
  -> Scheduler claims with temporary Lease
  -> fresh worker executes one bounded authority phase
  -> structured Worker Result
  -> Session Manager records execution history
```

The Scheduler never asks Workspace what work to invent, never edits Work or
qualification truth directly, never completes Work, and never parses prose. V1 scheduling is limited to implementation, review, and repair; higher-authority phases remain manual. Its
local runtime file is disposable. Physical removal of `tools/scheduler/` must
leave all manual manager commands working. See `tools/scheduler/README.md`.

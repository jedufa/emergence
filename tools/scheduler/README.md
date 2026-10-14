# EIEL Assignment Linker and Optional Worker Scheduler V1.10.3

**Document role:** Optional Scheduler reference. Ignore this document during ordinary manual operation.

## Purpose

The Scheduler is optional operational infrastructure for explicitly authorized,
bounded Work while the human is away. It extends the existing operator tooling;
it does not replace Bootstrap, Workspace, Work Manager, Session Manager,
Qualification, Resource, Evidence, the Librarian, or manual Codex use.

## Manual-first invariant

The normal path remains:

```text
python -m tools.eiel start
  -> Bootstrap / Workspace
  -> Work / Session
  -> manually started Codex
```

Removing `tools/scheduler/` disables only the optional `assignment` and
`scheduler` commands. A prepared-but-unstarted Session can be detached and
resumed manually:

```bash
python -m tools.eiel sessionManager resume SES-... \
  --take-over-prepared \
  --codex-id '<actual-codex-thread-id>'
```

No Scheduler runtime process or runtime database is required for that takeover.

## Ownership

```text
Repository and managers  own project truth
Assignment Linker        compiles existing truth only
Scheduler                owns one process execution attempt only
Worker                   executes one bounded authority phase only
Human                     owns authorization, model choice, and Work acceptance
```

The Scheduler never creates Work, chooses scientific priority, repairs
Bootstrap, changes authority, accepts Work, or interprets worker prose.

## V1 scope

Unattended scheduling supports only:

- `implementation`
- `review`
- `repair`

Architecture and Engineering Authority remain manual. Worker sandbox profiles
are limited to `read-only` and `workspace-write`. Resource admission is deferred
until assessments can be exactly bound to one Work/profile/start window.

## Durable records

Only four scheduler-facing record types exist:

- **Dispatch Request**: explicit human authorization.
- **Assignment**: immutable, digest-protected launch material.
- **Lease**: temporary ownership of one Assignment execution attempt.
- **Worker Result**: structured, digest-bound result.

Runtime state and bounded logs are disposable. Durable engineering history
remains in Work, Session, Assignment, Lease, and Worker Result records.

## Safety model

Before claim and worker launch, V1.10.3 requires:

- enabled, unchanged Dispatch authorization;
- an exact link-intent digest covering Dispatch revision, Work and package,
  repository/worktree/branch/HEAD, authority phase, Session/review target,
  worker profile, Lease policy, automation policy, and Bootstrap digest;
- one unique active Assignment per Work/package/phase/Session/HEAD;
- immutable READY Assignment material;
- clean worktree;
- exact branch and HEAD;
- exact Bootstrap record digest;
- live Work Package re-hash;
- live Context Receipt validation, including entry/authority files;
- active, unexpired Lease with immutable worker ownership.

Any relevant change creates or requires a new Assignment.

## Worker startup

The Scheduler persists a `LAUNCHING` row before process creation. A tiny
ACK-gated wrapper publishes its PID/start identity and cannot execute worker code
until the Scheduler revalidates Dispatch, Assignment, Lease, repository truth,
and mechanically activates the exact prepared Session. This closes the
post-`Popen` invisible-worker window without adding a daemon.

For real Codex execution, a provisional process identity is used only during
startup. The ACK-gated wrapper copies the first valid `thread.started` identity
into a tiny durable sidecar before rotating human-readable logs. That identity
must be captured before Lease expiry and bound to the Lease and Session. A
conflicting second thread identity fails closed. Review fails closed when a real
thread identity is absent or matches the implementation thread.

A worker must:

1. receive one immutable Assignment and active Lease;
2. operate in the frozen worktree, branch, and scope;
3. execute only the assigned authority phase;
4. after a commit, create and attach a fresh verify-only Bootstrap for new HEAD;
5. complete, block, or checkpoint and pause/interrupt the Session;
6. write one structured Worker Result;
7. exit.

## Crash and process containment

All Assignment/Lease lifecycle transitions use one per-Assignment lock. Result
submission is idempotent and converges after a crash between durable writes.
Exact duplicate replay completes missing terminal transitions rather than
returning early. The shared actor store treats immutable history as its
write-ahead recovery source, promoting a contiguous orphan revision and
rebuilding a torn index on the next bounded read/write.

Scheduler startup reconciles `LAUNCHING`, `CLAIMED`, and `RUNNING` attempts
before requesting new work. A surviving wrapper is adopted, heartbeated, and
followed to a structured result or deterministic worker-loss transition.

The wrapper and actual worker each own a process group. Launch callback failure,
heartbeat failure, timeout, Scheduler termination, ordinary failure, and normal
leader exit terminate all descendants with bounded SIGTERM then SIGKILL
fallback. Stdout and stderr are bounded while execution is live, not merely
truncated afterward. Terminal process rows are removed from disposable runtime
state, and old run directories are pruned.

## Authorization and late-result policy

Disabling a Dispatch Request supersedes PREPARING, READY, CLAIMED, and
pre-ACK RUNNING attempts. The ACK gate prevents worker code from starting after
revocation. A worker with a durably acknowledged launch is not silently killed
merely because authorization is later disabled; explicit Lease revocation is a
separate future decision.

V1 rejects work that executes through or after Lease expiry. A structured result
may be submitted after expiry only when its immutable execution timestamps prove
that it started after claim and ended strictly before expiry, and every bound
Assignment, Lease, Session, Work Package, Bootstrap, repository, branch, and
worker identity still matches. The expired Lease is closed without revival.

When a worker is gone but a result file is visible, reconciliation uses one
bounded settlement path: two stable observations, bounded validation retries,
then either submission or explicit `RESULT_INVALID` with the original file and
validation errors preserved. Partial and zero-byte files remain `RESULT_PENDING`
instead of being mistaken for immediate worker loss.

## Authority transitions

Current delivery state takes precedence over old resumable Sessions:

```text
PROMOTION_REQUIRED -> no Assignment
READY_TO_ACCEPT     -> no Assignment
REVIEW_REQUIRED     -> review only
CHANGES_REQUIRED    -> repair only
otherwise           -> eligible current-package resumable Session
```

Old, superseded, or wrong-package Sessions cannot bypass review, repair, or
promotion. Manually active Sessions make their Work ineligible for Scheduler
selection; a deterministic candidate-level rejection does not stop later valid
Work from being considered.

## Complete safe operator flow

The host capability probe comes first for real Codex execution:

```bash
python -m tools.eiel assignment codex-capability --human
```

Authorize one bounded phase. A structured request file is preferred when many
profile fields are required; the flags below show the minimal manual form:

```bash
python -m tools.eiel assignment authorize \
  --work-id WORK-... \
  --repo-root /path/to/worktree \
  --phase implementation \
  --launcher codex_exec \
  --model '<human-selected-model>' \
  --human
```

Inspect deterministic eligibility, then run one bounded attempt:

```bash
python -m tools.eiel assignment candidates --eligible-only --human
python -m tools.eiel scheduler run-once --human
python -m tools.eiel scheduler status --human
```

The foreground polling loop remains single-worker:

```bash
python -m tools.eiel scheduler start \
  --max-assignments 3 \
  --poll-seconds 300
```

`start` occupies its terminal. Use a second terminal for `status`, `pause`,
`resume`, or `stop`. These controls affect Scheduler runtime only; they do not
change Work or Session truth.

```bash
python -m tools.eiel scheduler status --human
python -m tools.eiel scheduler pause --human
python -m tools.eiel scheduler resume --human
python -m tools.eiel scheduler stop --human
```

Manual fallback requires no Scheduler process:

```bash
python -m tools.eiel start --human
python -m tools.eiel sessionManager resume SES-... \
  --take-over-prepared \
  --codex-id '<actual-codex-thread-id>' \
  --human
```

Expired Leases and interrupted Sessions remain durable for inspection and
reconciliation. Do not restart or reinterpret them from prose alone.

## Non-goals

V1.10.3 does not implement multiple workers, automatic Work generation,
automatic model routing, automatic Work completion, worktree mutation, resource
reservations, retry engines, programme waves, an event bus, a daemon, or prose
parsing.

## Remaining host limitation

Fake Worker and repository mechanics are fully testable in a transfer archive.
Real unattended execution remains host-qualified only after one disposable
`codex exec` conformance smoke proves the installed CLI, actual thread identity,
sandbox, structured output, process-group termination, and failure behavior.

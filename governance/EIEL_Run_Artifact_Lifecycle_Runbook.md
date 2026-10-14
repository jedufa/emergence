# EIEL Run Artifact Lifecycle Contract V1

This runbook describes how campaign and production runs choose, enforce, and
close an artifact lifecycle. It is an integration contract over the existing
campaign compiler, QRR front door, execution runner, evidence closeout,
retention guard, and prune actuator. It is not a new manager, scheduler,
database, or execution authority.

## Current default: report-only finalized runs

The operator's 2026-10-11 instruction supersedes the permanent certificate,
JSON/Markdown capsule pairs, audit ZIPs, terminal checkpoints, cold archives,
and durable per-run receipt defaults described in older sections below.
[EIEL Report-Only Run Retention](EIEL_Report_Only_Run_Retention.md) is the
current closeout contract: retain one self-contained Markdown report per
finalized run and reclaim all other run-owned material. Existing completed
runs are explicitly included; active and unfinished runs remain protected.
Finalized failure and explicit abandonment also require report and cleanup.

Use the existing canonical lifecycle and exact-root retirement helper; do not
add a storage manager or delete by glob, age, or apparent importance. Exact
inventory, manifests, reservations, journals and receipts protect deletion
while it is in flight. They are temporary crash-recovery material and must be
removed once retirement and Markdown finalization are verified. Embed final
cleanup status and accounting in the Markdown report; do not keep a second
permanent JSON receipt. Publish the report outside the retired root first.

Legacy certificate procedures below remain useful for an active bounded gate
or recovery transaction, but never impose permanent extra run outputs. Resolve
historical holds and references explicitly; do not relabel a live obligation
as closed. Optional selected checkpoints belong only in
`/tmp/emergence-example/interesting-ckpt` and may never become a required dependency.
The user may delete that directory at any time; reports must independently
explain how to regenerate inputs and selected states.

## Ownership and flow

```text
campaign spec / QRR request
        |
        v
compiler + request normalizer
        |  strict lifecycle contract, bound into plan identity
        v
preflight + RunStorageGuard
        |  free-disk admission and bounded runtime checks
        v
campaign/QRR execution
        |  checkpoints, passport, closeout evidence
        v
disposition manifest + storage receipt
        |
        v
existing retention/prune actuator
```

The Work Manager remains portfolio and lifecycle bookkeeping. The Qualification
and Evidence Managers retain their existing interpretation and evidence
authority. The campaign compiler/runner and QRR own execution. The existing
retention/prune tooling is the only destructive actuator.

## Time authority and contract placement

For active Lifecycle V2 / `EIEL_FULL_EVOLUTION_V1` runs, absolute ticks are
the only execution, checkpoint, restore, reporting, qualification, and science
horizon authority. The active request must identify
`tick_native_execution_route_v1` and declare `total_ticks` plus its bounded
owner, dispatch, refill, source, and event-window controls. A 160-tick refill
or source span is engineering chunking; it is not a generation or qualification
horizon.

Generation-shaped fields are accepted only by the explicitly selected
`legacy_generation_route_v1` compatibility route or by a declared heritage
turnover adapter. They must not be copied into active Lifecycle V2 inputs. See
the [Tick-Native Authority and Generation-Era Supersession Index](EIEL_Tick_Native_Authority_and_Generation_Era_Supersession_Index.md)
before reusing an older campaign, report, qualification receipt, or example.

New campaign inputs may carry an `artifact_lifecycle` object. Its canonical
shape is `eiel_run_artifact_lifecycle_contract_v1`:

```yaml
artifact_lifecycle:
  schema_version: eiel_run_artifact_lifecycle_contract_v1
  lifecycle_profile: CHECKPOINT_TRANSIT
  storage:
    total_max_bytes: 10737418240
    scratch_max_bytes: 1073741824
    checkpoint_max_bytes: 4294967296
    durable_max_bytes: 4294967296
    emergency_reserve_bytes: 536870912
  observability:
    output_profile: compact_default
    raw_tick_trace_policy: forbidden
    debug_event_trace_policy: forbidden
    ring_buffer:
      enabled: false
      max_bytes: 1
  checkpoints:
    schedule: [0, 2500, 5000]
    time_authority: absolute_tick_v1
    roles:
      - identity: checkpoint-000000000
        tick: 0
        role: intermediate
        retention_class: R1
      - identity: checkpoint-000002500
        tick: 2500
        role: intermediate
        retention_class: R1
      - identity: checkpoint-000005000
        tick: 5000
        role: terminal
        retention_class: R4
  closeout:
    required_types: [run_passport, artifact_disposition_manifest, storage_disposition_receipt]
    compact_replacement_verification_required: false
  retention:
    default_class: R1
    decision_evidence_class: null
    audit_hold_binding: not_required
    qualification_hold_binding: not_required
    unknown_artifact_policy: RX_BLOCKS_AUTO_DELETE
  prune:
    automatic_prune_allowed: true
    required_predicates:
      - valid_disposition_manifest
      - unknown_artifacts_resolved
      - no_live_references
      - audit_hold_released_or_not_required
      - qualification_hold_released_or_not_required
      - replacement_verified_or_not_required
      - explicit_checkpoint_roles_resolved
      - storage_disposition_receipt
```

The compiler validates exact nested fields, quota partitioning, checkpoint
role coverage, retention policy, and prune predicates. A contract digest is
included in the compiled plan payload, so changing lifecycle intent changes
the plan identity. For a tick-native request, every schedule entry and role
uses the same absolute-tick coordinate and the terminal role equals
`total_ticks`. A legacy lifecycle contract may retain generation roles only
when its enclosing execution request explicitly selects
`legacy_generation_route_v1`.

Campaigns without lifecycle data remain accepted through a conservative
derived contract. The derived contract preserves the existing output behavior,
forbids raw/debug traces for checkpoint-transit defaults, and never enables
automatic deletion. This compatibility does not authorize generation-shaped
fields on a new tick-native request. Legacy runs do not become auto-prunable
merely because they are recompiled.

## Profile selection

| Profile | Use | Output binding | Trace policy | Release posture |
| --- | --- | --- | --- | --- |
| `CHECKPOINT_TRANSIT` | Long evolution/production runs | `compact_default` | Raw and debug/event traces forbidden | Automatic exact R0-R2 release at clean terminal closeout; terminal/replay material remains durable |
| `SCIENCE_COMPACT` | Decision-bearing science campaigns | `compact_default` | Traces forbidden unless explicitly escalated | Automatic exact R0-R2 release after replacement predicates; Class A evidence remains held/durable |
| `QUALIFICATION` | Evidence-driven qualification runs | `compact_default` | Compact by default; replay/debug observability is explicit and evidence-driven | Automatic exact operational R0-R2 release; qualification-critical R3-R5 remains protected/durable |
| `DEBUG_FORENSIC` | Debug/engineering investigations | `debug_event_trace` | Bounded raw trace, event trace, and optional ring buffer | Explicit quota and diagnostic closeout; automatic prune disabled until semantic review |

`DEBUG_FORENSIC` cannot opt into automatic prune before diagnostic closeout.
The other explicit new-run profiles default to automatic release, but only for
exact manifest-authorized R0-R2 rows after their own last-consumer and
replacement predicates are satisfied. Qualification and audit holds continue
to protect R3-R5 evidence; RX remains preserved. Legacy-derived contracts stay
non-prunable until an accepted reconstructed disposition exists.
An output profile different from the lifecycle default, or a trace policy that
is not forbidden for `CHECKPOINT_TRANSIT`/`SCIENCE_COMPACT`/`QUALIFICATION`, must carry a
non-empty `observability.override_reason`. This is an explicit contract
declaration, not an automatic qualification decision. Qualification evidence
obligations remain authoritative; if compact output is insufficient, the
qualification contract must explicitly escalate before execution.
For `SCIENCE_COMPACT`, any applied release still requires a valid disposition
manifest, resolved RX inventory, released holds, verified replacement where
required, explicit checkpoint roles, and a storage disposition receipt.

## Managed zones and closeout artifacts

The lifecycle inventory recognizes these zones without requiring an all-at-once
writer migration:

- `control/`: manifests, passports, contracts, plans, and receipts;
- `checkpoints/`: role-bound checkpoint bundles;
- `durable/`: accepted durable material;
- `scratch/`: temporary traces and intermediate material;
- `cache/`: reproducible accelerators;
- `quarantine/`: material that is not safe to classify as releasable;
- `closeout/`: compact evidence and closeout outputs.

Unclassified paths are `RX` and block automatic cleanup. Large inventories may
be deterministically aggregated by artifact family, but the manifest retains
bytes, hashes, owner/family, retention class, release predicate, hold state,
replacement identity, and checkpoint/run association.

At execution start the runner writes the lifecycle contract and a compact
`run_passport.json`. At terminal closeout it writes
`artifact_disposition_manifest.json` and a dry-run
`storage_disposition_receipt.json`; the passport is then updated with those
references. Evidence closeout records verified compact replacements rather
than silently treating a ZIP as a replacement for raw material.

## Storage enforcement

`RunStorageGuard` is a deterministic observer, not a manager. Campaign and QRR
preflight use the existing free-disk admission path and add lifecycle quota
checks. The runner checks the guard at bounded lifecycle points (before the
campaign and around batches), tracking scratch, checkpoints, durable bytes,
total bytes, free filesystem space, and the emergency reserve.

An exceeded quota, emergency-reserve breach, or checkpoint-transit trace
violation first invokes the bounded canonical pre-admission reclamation pass
through the existing lifecycle pointer queue and exact prune actuator. The
guard retries admission once after release. Only a remaining irreducible
shortfall produces `storage_budget_failure_receipt.json` and fails closed. The
guard never deletes data itself, never silently disables declared debug or
qualification observability, and never sends ordinary transient evidence to
HDD as a pressure-relief workaround.

## Retention and pruning

Decision-evidence Class A remains protected until audit release. Class B
diagnostics remain protected until the reducer closeout/replacement predicate
is released. Class C/R0-R2 material can be selected only when the lifecycle
disposition says it is eligible. Qualification and live-reference holds remain
fail-closed.

The prune helper is dry-run-first:

```bash
./tools/bin/emergence-python tools/scripts/eiel_runtime_storage_prune.py \
  <run-root> --disposition-manifest <run-root>/artifact_disposition_manifest.json
```

`--apply` is rejected without a valid machine-readable lifecycle disposition
manifest whose selected paths are explicitly eligible. Applied cleanup writes
`storage_disposition_receipt.json` as transaction authority while cleanup is
pending; finalized report-only retirement folds its accounting into Markdown
and removes the temporary receipt. This implementation
does not perform broad cleanup or delete historical evidence.

## Explicit gate-boundary Evidence Certificate closeout

For a coherent evidence obligation that closes before its containing
programme, use a fresh `eiel_evidence_certificate_request_v1` with the exact
producer inventory, bound programme-plan hash, producing/current gate, named
future consumers and last use, live-state predicates, forensic/asset and exact
qualification-retention declarations, validation/review receipts, semantic
decision, limitations, and reproduction recipe.
The programme-plan hash is derived from programme ID, plan ID, and the ordered
gate graph; `closed_gates` is the exact ordered prefix through the current gate,
and `closure_receipt_sha256` binds the current gate-ledger/closure authority.

```bash
./tools/bin/emergence-python \
  tools/scripts/eiel_evidence_certificate_reclaim.py close-gate \
  --request <fresh-gate-closure-request.json> \
  --evidence-root <exact-bounded-gate-evidence-root> \
  --output-root <same-root>/control/evidence-reclamation/<gate>

# Only after reviewing the exact dry-run and with the same bound request:
./tools/bin/emergence-python \
  tools/scripts/eiel_evidence_certificate_reclaim.py close-gate \
  --request <fresh-gate-closure-request.json> \
  --evidence-root <exact-bounded-gate-evidence-root> \
  --output-root <same-root>/control/evidence-reclamation/<gate> \
  --apply
```

The first call seals a compact certificate, compiles exact rows into the
existing disposition manifest, and writes a crash-recoverable intent journal.
The second call rejects changed requests/programme plans/certificates/manifests,
revalidates exact object identity, atomically stages authorized objects before
deletion, and writes the existing durable storage disposition receipt plus a
final certificate revision. A named future consumer, live state, active
forensic hold, asset, exact qualification/provenance authority, or unknown
policy remains protected.

Certificate-bearing manifests fail closed in the generic prune CLI. The
gate-close apply path alone supplies the bound request/certificate/plan/intent,
holds the canonical reservation fence, uses descriptor-pinned in-root staging,
and reconciles post-delete progress interruptions without under-reporting
released bytes.

This seam never infers a semantic gate decision from orchestration, never
invokes the archive worker for ordinary releasable R0-R2 evidence, and does not
weaken the storage guard's reserve. Run it at each coherent gate/evidence
closure; programme final closeout is not the first release point.

### Legacy compact capsule transaction (superseded for final retention)

For a `/tmp/emergence-example/runtime` evidence root, gate closeout first publishes a
compact, immutable `SEALED_PRE_RELEASE` pair under
`/tmp/emergence-example/storage/_permanent_evidences` before any raw apply:

```text
<content-bound capsule>.json  machine-verifiable certificate and disposition
<content-bound capsule>.md    conclusions, steps, limits, and reproduction guide
```

That pre-release capsule binds the exact certificate, identities, conclusions,
reproduction procedure, and selected source accounting, and was the legacy
prerequisite for certificate-based release. Final report-only retirement uses
the verified Markdown report and canonical retirement transaction instead. After an applied release has a verified
storage receipt, closeout publishes a second immutable `FINAL` pair. In the legacy path the
`FINAL` capsule additionally
binds the exact released rows, byte totals, receipt, and release time. A crash
between those phases resumes from the durable transaction records; it never
rewrites either immutable publication with a different identity.

Together the phase-appropriate capsule binds the
source/request/plan/runtime/environment/RNG identities, exact sealed-inventory
digest and accounting,
criteria and measurements, decision, limitations and no-claims, and
reproduction commands and verification rules. Only the `FINAL` phase binds the
final storage receipt. Neither phase contains raw
checkpoint, source-import, event-history, profiler-stream, or simulation
payload. The exact inventory is bound by digest plus byte/class/disposition
accounting rather than duplicated row-by-row. The Markdown reading copy should
normally remain only a few kilobytes; the complete JSON/Markdown/marker
publication is capped at 256 KiB and the whole root at 512 MiB. Publication is
atomic, content-bound, idempotent, and subject to those checked-in limits. If
publication or verification fails, raw apply does not begin.

`DEBUG_FORENSIC` remains non-prunable under its plan-bound lifecycle contract.
Its raw rows become eligible only through an explicit diagnostic closeout that
binds the source manifest, accounts for every affected row, records an accepted
substantive review and conclusion, closes every named consumer and forensic
hold, declares no unresolved anomaly, and proves that audit, qualification,
replay, writer, resume, reducer, and live-reference obligations no longer need
the exact bytes. The certificate-derived release authority does not relabel or
weaken the original profile. R4, R5, RX, assets, and active holds remain
protected.

## Legacy evidence reconstruction

Roots created before `eiel_artifact_disposition_manifest_v1` are not
permanent-retention exemptions. They are a separate, bounded review lane:

```bash
./tools/bin/emergence-python \
  tools/scripts/eiel_legacy_evidence_reconstruct.py \
  <exact-legacy-root> \
  --evidence-reference <report-or-receipt> \
  --write
```

The command seals bounded file/directory segments, hashes them, records local
and supplied evidence references, and writes an
`artifact_disposition_manifest.json` whose
`legacy_reconstruction.status` is `REQUIRES_REVIEW`. It never deletes or
archives bytes. Aggregate parents such as `/tmp/emergence-example/storage`, `evidence_raw`, and
`campaigns` are rejected; process one historical root at a time. Use
`--max-objects` and `--max-bytes` to keep the reconstruction pass bounded.

The reviewer must answer the last-consumer questions (active Work/resume,
branch/investigation/reducer/qualification references, durable replacement,
replay need, and reproducibility) and create a complete
`eiel_legacy_evidence_review_receipt_v1` containing one decision for every
reconstructed row. Each row must become exactly one of `KEEP_ACTIVE`,
`HOLD_DIAGNOSTIC`, `RELEASE`, `RETAIN_DURABLE`, or `RX`; an original manifest
is not required, but an accepted review receipt is. Each decision records its
consumer references, reproducibility claim, live-reference status, replacement
identity, and reason; an empty consumer list is an explicit finding, not an
omission.

Promote the reviewed candidate with:

```bash
./tools/bin/emergence-python \
  tools/scripts/eiel_legacy_evidence_reconstruct.py \
  <exact-legacy-root> \
  --review-receipt <review-receipt.json> \
  --write
```

Only an accepted reconstructed manifest can enter the existing exact
prune/archive paths. `RELEASE` rows are still re-hashed and checked for live
writers/reservations by the prune actuator. `RETAIN_DURABLE` rows may be
registered for asynchronous cold-storage archival; `RX` remains preserved,
but only for the explicitly unresolved exact rows. A missing original
manifest therefore means `REQUIRES_REVIEW`, never an automatic delete and
never an unbounded permanent-retention decision.

## Terminal exact-root retirement

After every covered object has passed its normal manifest/reconstruction
disposition and the whole programme is terminal, an exact programme-owned
runtime root may be removed with a fresh
`eiel_runtime_root_retirement_request_v1`. This is the final actuator of the
same lifecycle, not a discovery, age-based cleanup, or archive path.

```bash
./tools/bin/emergence-python \
  tools/scripts/eiel_evidence_certificate_reclaim.py retire-roots \
  --request <fresh-terminal-retirement-request.json> \
  --output-root <external-durable-control-root>

# Only after writing an accepted root_retirement_review.json that binds the
# exact request and sealed inventory hashes:
./tools/bin/emergence-python \
  tools/scripts/eiel_evidence_certificate_reclaim.py retire-roots \
  --request <fresh-terminal-retirement-request.json> \
  --output-root <same-external-durable-control-root> \
  --apply
```

The request enumerates each exact strict child of an allowed
`causyl-runtime` namespace, binds the compact programme authority, terminal
decisions, per-root lifecycle/reconstruction receipts, zero RX/live-reference
state, released audit and qualification holds, and accepted semantic review.
The dry run rejects processes, storage reservations, symlinks, hard links,
overlapping roots, and an output path that contains or is contained by any
target. It seals every regular file by hash plus the root device/inode.

Apply reacquires the reservation fence, revalidates authority hashes and exact
inventory, atomically renames each target to a hidden same-filesystem stage,
and deletes only descriptor-pinned sealed members. Its external intent journal
supports safe replay after staging or partial deletion; an extra or changed
member fails closed. A durable applied receipt is accepted idempotently only
when neither the source nor its authenticated stage survives. The actuator
does not invoke archival, wait for slow storage, scan a parent namespace, or
make scientific/qualification claims.

## Slow-tier non-blocking contract

Slow HDD and cold-storage work is never part of the foreground execution,
validation, checkpoint-release, closeout, or Codex critical path. The
foreground path may seal the exact lifecycle manifest, publish its durable
archive pointer, and record the background worker or supervisor identity. It
must then continue every separable action without copying, destination hashing,
polling, or waiting for the slow-tier transfer inline.

The bounded `eiel_runtime_storage_archive_worker.py` remains the archive
actuator and owns transfer, destination verification, retry, and authorized
source release. Its elapsed time is reported as storage/archive latency,
separate from simulation, ACS, validation, and ratchet decision timing. A
consumer that truly requires cold bytes may wait only at that declared
consumer boundary; the producer GPU and unrelated work remain released. Do not
introduce a second queue, watcher, scheduler, or storage manager to implement
this rule.

When a bounded terminal reseal replaces a run manifest, registration moves
older exact same-run archive pointers into the managed `superseded/` audit
subdirectory. The background worker therefore sees only the current manifest
authority and does not retry an obsolete pointer indefinitely.

The fast release actuator runs independently of slow storage and consumes the
same pointer queue in release-only mode. It may release only exact,
manifest-authorized R0-R2 rows and never opens the cold destination. The slow
actuator archives only explicit R3-R5 `RETAIN_DURABLE` rows. Failed or busy
pointers rotate to the back of the bounded queue so one poison item cannot
starve unrelated releases. If the readable cold destination already contains
a different sealed object, both objects are preserved and the new one is
written beneath a deterministic content-addressed version path; neither is
overwritten.

Useful profiling data is an optional selected asset, not a reason to preserve
the raw diagnostic tree. Prefer a compact profile summary. Any exact replay or
profile asset admitted to slow storage must be R3-R5, explicitly justified,
and within the per-object and aggregate cold-diagnostic budgets in
`_spec/storage/v1/eiel_data2_storage_availability_v1.json`. R0-R2 raw profiler
streams are direct-release material after their last consumer closes.

## Canonical `/tmp/emergence-example/storage` availability floor

Every managed `/tmp/emergence-example/storage` run loads the checked-in
`EIEL_DATA2_STORAGE_AVAILABILITY_V1` policy and binds its digest plus the
observed filesystem identity into storage snapshots and reservations. The
protected-free floor is the maximum of the measured execution reserve, the
lifecycle reserve, and any explicitly configured stricter fraction. The
reformed policy has no global percentage or 100 GiB floor. Its reserve derives
from the remaining output envelope, active checkpoint window, writer/reducer
staging, quarantine allowance, a 4 GiB filesystem reserve, and a bounded 5%
uncertainty margin capped at 8 GiB. Admission adds all other active bounded
reservations and the candidate's remaining projection to that floor. It is
fail-closed if policy or filesystem observation is unavailable or if projected
output would cross the resulting requirement.

After managed Work completion has durably sealed the accepted integration
identity, Work Manager requires durable certificate/report/reproduction
references, then invokes the existing exact prune actuator for explicitly
bound `storage_evidence_roots`. The resulting immutable
`eiel_storage_merge_closeout_v1` receipt records retained/released bytes and
reasons. Direct Git merges are reconciled with `eiel storage
reconcile-merged-work`; Git hooks are not the authority. See
`EIEL_Storage_Policy_Reform_Procedure.md` for the complete pressure,
merge-closeout, hold, and historical-evidence policy.

This is a guarantee for canonical managed writers: it preserves usable free
space instead of reserving a growing raw-evidence collection. It cannot prevent
an unmanaged process from filling the filesystem; such a writer must be moved
behind the same admission policy or an operating-system quota before claiming
a machine-wide guarantee.

## Operator checklist

1. Choose the profile from the run's purpose, not from available disk.
2. Bind the lifecycle contract before execution and confirm the compiled plan
   hash/contract digest in preflight output.
3. Confirm checkpoint schedule and roles cover every requested absolute tick;
   for a tick-native run, reject generation-addressed roles.
4. For science and qualification, bind the relevant audit/qualification hold
   before generating decision-bearing evidence.
5. Treat RX, live references, Class A, and unreleased Class B as protected.
6. Review the disposition dry run and replacement/hold receipts before any
   `--apply` operation.
7. Finalize the self-contained Markdown report with verified cleanup status;
   remove temporary passports, manifests, receipts, journals, and capsules
   after their transaction and recovery obligations close.

No scientific semantics, runtime state-transition semantics, or qualification
interpretation are changed by this contract. It controls declared storage,
observability, provenance, retention, and cleanup authority around the run.

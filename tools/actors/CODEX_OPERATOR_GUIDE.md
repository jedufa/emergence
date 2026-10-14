# EIEL V1.19 Codex Operator Procedure

> Repository-wide scope and model-use defaults: the internal `AGENTS.md` (not distributed; see [contributor guidance](../../CONTRIBUTING.md)).

**Document role:** Detailed operational reference. Use `OPERATOR_QUICKSTART.md` for the daily path.

## The rule

The explicit operator instruction is the primary operational authority. The
commands below recover and record work; they are not approval ceremonies. A
missing, stale, incomplete, unreviewed, or unaccepted workflow record is an
administrative obligation, not by itself a reason to stop authorized work.

Use this when a Codex conversation begins, resumes after interruption, changes
authority phase, or loses confidence in repository state:

```text
python -m tools.eiel start
```

Workspace may recover the context or recommend a reconciliation command. Never
adopt an unrelated Work or Session merely because it is open. If the projection
is unavailable or administratively stale, perform bounded direct verification,
record the limitation, and continue unless the underlying condition is a real
technical, safety, resource, concurrency, access, or claim-truth blocker.

A “conversation” here means a Codex thread or bounded Context Epoch, not every
chat message or terminal command. Repeated `start`/`resume` calls are unnecessary
while the same Session and repository state remain understood.

The ordinary shortcut is compact structured JSON in V1.11. It is intended for
the working Codex context, not for forensic inspection.

```text
python -m tools.eiel start --full
python -m tools.eiel start --output /tmp/eiel-start-full.json
```

`--full` displays the complete Workspace record graph. `--output` preserves that
full payload on disk while leaving the console compact. Direct
`workspace open` remains full unless `--compact` is requested.


## Context-efficient epoch

For a package-bound Session, Workspace now shows:

- the initial files and search roots;
- excluded roots and known symbols;
- compact command-output policy;
- advisory execution budget;
- current context health;
- a compact Context Receipt.

Read the declared initial set before searching elsewhere. Global instructions
already supplied to the active Codex context do not need a ritual full-file
reread when their digest and scope are unchanged. `required_documents` should
name task-specific authority not already present in the instruction context.
Redirect broad output to a file and inspect a bounded summary. At an implementation, promotion, review,
or qualification boundary, checkpoint the Session and continue in a fresh
conversation.

```text
python -m tools.eiel sessionManager checkpoint SES-... \
  --tool-call-count 40 \
  --compaction-count 1 \
  --epoch-boundary-reason "implementation complete"
python -m tools.eiel sessionManager context-receipt SES-... --attach --human
```

Manual or externally supplied counters were introduced in V1.9 and are retained in V1.11. Missing telemetry is shown
as `NOT_OBSERVED`; Workspace never fabricates it. See
`tools/actors/CONTEXT_EPOCH_POLICY.md`.

## 1. Bootstrap when requested

Typical implementation bootstrap:

```text
python -m tools.eiel bootstrap inspect \
  --repo-root ${HOME}/emergence \
  --task-mode implementation \
  --topic "<exact task>" \
  --operator-instruction-ref "<exact operator instruction>" \
  --edit-authorized \
  --write-scope tools \
  --expected-validation "pytest -q <tests>" \
  --gpu-required auto
```

Bootstrap is read-first and fail-closed. It never creates a virtual environment,
installs packages, changes source, launches a campaign, or guesses.

A receipt is bound to:

- Git worktree root and HEAD;
- Python/Torch/CUDA environment;
- task classification and topic;
- operator instruction;
- write scope and validation plan;
- Knowledge Librarian preflight.

Workspace reports repository or Git-HEAD mismatch as stale recovery state. It
reports receipt age and recommends refresh after 24 hours. A corrupt or
incompatible receipt may prevent Workspace from trusting that receipt, but does
not revoke the live operator instruction.

## 2. Open or resume

```text
python -m tools.eiel start --bootstrap-record BST-...
```

Possible outcomes:

- `WORKSPACE_READY`: use relevant guidance or proceed directly within the operator instruction.
- multiple Sessions or Work Items: select one only if it matches the request.
- `SESSION_BOOTSTRAP_REFRESH_REQUIRED`: optionally bind the current BST receipt to repair Session recovery.
- `WORK_BLOCKED_BY_DEPENDENCY`: inspect and complete the blocker.
- `PROMOTION_REQUIRED`: do not resume the stopped implementation; return to Engineering Authority.
- `WORK_REVIEW_REQUIRED`: start a Review Authority session; prefer an independent identity when immediately available, otherwise use the same-identity reduced-assurance fallback.
- `WORK_CHANGES_REQUIRED`: start bounded repair work or promote the package.
- `WORK_READY_TO_ACCEPT`: complete Work with the named accepted review session.
- `NO_OPEN_WORK`: proceed on the authorized instruction; create Work only if useful.

Select explicitly:

```text
python -m tools.eiel start --bootstrap-record BST-... --work-id WORK-...
python -m tools.eiel start --bootstrap-record BST-... --session-id SES-...
```

## 3. Durable work

```text
python -m tools.eiel workManager list --open-only --view priority
python -m tools.eiel workManager inspect WORK-...
python -m tools.eiel workManager graph WORK-...
python -m tools.eiel workManager next WORK-...
```

Create one bounded work item:

```text
python -m tools.eiel workManager create \
  --title "W32 qualification" \
  --summary "Qualify the admitted W32 route" \
  --work-type QUALIFICATION \
  --priority P1 \
  --next-action "Create the bounded qualification plan."
```

For multi-model development, prefer a repository-native Work Package. The
Engineering Authority writes the package; Work Manager binds it by relative
path, version and SHA-256:

```text
python -m tools.eiel workManager create \
  --work-package work_packages/WORK-042.yaml \
  --repo-root ${HOME}/emergence \
  --status ACTIVE \
  --next-action "Execute the exact bound Work Package."
```

The package contains only the stable engineering contract:

```text
work_id
goal
scope
owned_files
forbidden_files
dependencies
acceptance_criteria
tests
evidence
stop_conditions
recommended_authority
recommended_model
uncertainty_class
```

YAML and JSON authoring are both supported. Active binding requires the package
to be Git-tracked, committed and clean. The loader rejects duplicate keys, YAML
aliases/merges, ambiguous scalar coercion, absolute/traversal paths, incorrect
case, symlink traversal and oversized contracts. It validates the package,
normalizes it to canonical JSON, and hashes the effective material contract.
Equivalent YAML and JSON contracts therefore have the same identity.

Every executable package needs non-empty scope, ownership, acceptance criteria,
tests, evidence and stop conditions. Literal ownership is exact; use `dir/**`
when recursive subtree ownership is intended. Filesystem type changes never
widen or shrink the contract.

Material fields are goal, scope, owned/forbidden files, dependencies,
acceptance, tests, evidence, stop conditions, authority, uncertainty and the
effective optional contracts. Set-like lists are canonicalized, and omitted
optional contracts hash identically to explicit defaults. A change to material
content after a Session exists requires a higher `package_version` at a new path
such as `WORK-042-v2.yaml`. The display title, model recommendation and formatting
are advisory/non-material, but changed source bytes still require an explicit
rebind before new execution. The tooling does not guess that wording changes
inside material fields are cosmetic.

Do not put mutable status or current phase in the package. Work Manager and
Session Manager already own that state. Historical Sessions keep their original
binding and remain auditable.

A simple `programme.md` may describe execution order, manual parallel groups,
consolidation and integration Work Items. It is not a scheduler or state engine.
See `tools/actors/examples/programme.example.md`.

When work discovers a prerequisite, retain the original track:

```text
python -m tools.eiel workManager block WORK-W32-CAMPAIGN \
  --blocked-by WORK-W32-QUALIFICATION \
  --reason "The required route is not qualified."
```

Only `COMPLETED` satisfies a dependency. `DISCONTINUED`, `OBSOLETE`, and
`SUPERSEDED` are terminal but not successful.

## 4. Codex sessions

### Codex identity

For manual Codex work, use the actual conversation/thread ID or `--codex-url`
when available. If the host does not expose one, use a stable explicit label and
record that it is operator-supplied. Reopening a terminal does not create a new
identity when the same conversation continues. A new conversation uses a new
identity, and a Review Authority Session must not reuse the implementation
identity.

Start:

```text
python -m tools.eiel sessionManager start WORK-... \
  --bootstrap-record BST-... \
  --codex-id "<conversation-or-session-id>" \
  --repo-root ${HOME}/emergence \
  --stage implementation \
  --current-step "audit ownership" \
  --next-action "Implement the first bounded slice."
```

For a package-bound implementation session, record the actual human-selected
role and model:

```text
python -m tools.eiel sessionManager start WORK-042 \
  --bootstrap-record BST-... \
  --codex-id "<codex-session-id>" \
  --repo-root ${HOME}/emergence-worktrees/work-042 \
  --role IMPLEMENTATION_AUTHORITY \
  --model "GPT-5.3 Spark High" \
  --reasoning-level HIGH
```

If `--role` is omitted, the Work Package recommendation is used. The model
recommendation remains advisory; `--model` records what the human actually
selected.

Checkpoint meaningful progress:

```text
python -m tools.eiel sessionManager checkpoint SES-... \
  --current-stage validation \
  --current-step "run focused tests" \
  --completed-step "implementation complete" \
  --success-command "python -m py_compile ..." \
  --success-result passed \
  --next-action "Run the workflow tests." \
  --next-command "pytest -q tests/test_..."
```


### Planned and abrupt stops

Before a planned stop, checkpoint the exact continuation and attach a Context
Receipt:

```text
python -m tools.eiel sessionManager context-receipt SES-... --attach --human
```

After an abrupt stop, the repository and latest durable Session checkpoint
survive. Uncheckpointed conversational progress does not. Run `eiel start`,
inspect the retained checkpoint, and do not assume later unrecorded reasoning or
commands survived.

Before interruption:

```text
python -m tools.eiel sessionManager interrupt SES-... \
  --interrupted-during "qualification evidence review" \
  --next-action "Resume at comparison row 18." \
  --next-command "python -m tools.eiel qualificationManager check ..."
```

Resume after another track changed the repository:

```text
python -m tools.eiel sessionManager resume SES-... \
  --bootstrap-record BST-CURRENT \
  --codex-id "<current-id>" \
  --takeover-reason "original Codex unavailable"
```

The original Codex ID, participants, takeovers, old bootstrap IDs, repository
HEADs, and immutable revisions remain available. Completing a session never
completes the work item automatically.

### Promotion stop

Implementation Authority must not redesign architecture. When the real work is
more uncertain than the approved package:

```text
python -m tools.eiel sessionManager request-promotion SES-... \
  --requested-class U4 \
  --reason "Two incompatible ownership contracts are possible." \
  --affected tools/shared_contract.py \
  --safe-state "Bounded changes are committed and focused tests pass."
```

This leaves that managed Session `BLOCKED` with outcome `PROMOTION_REQUESTED`.
It cannot be resumed as though its package were unchanged. Repair or supersede
the record when useful; the operator may authorize a separate bounded task
directly.

### Review gate and reduced-assurance fallback

Finish bounded implementation:

```text
python -m tools.eiel sessionManager complete SES-IMPLEMENTATION-...
```

For an Implementation Authority package this records
`IMPLEMENTATION_COMPLETE`. It does not accept Work.

Start a fresh review targeting the exact latest delivery Session. Prefer a
different Codex identity when one is immediately available. Otherwise, use the
same identity and record `REVIEWER_INDEPENDENCE_UNAVAILABLE`; it remains a
substantive review and can accept and release the Work when its evidence passes.

```text
python -m tools.eiel sessionManager start WORK-042 \
  --bootstrap-record BST-... \
  --codex-id "<review-id>" \
  --repo-root ${HOME}/emergence \
  --role REVIEW_AUTHORITY \
  --review-target-session SES-IMPLEMENTATION-... \
  --model "GPT-5.5 High" \
  --reasoning-level HIGH
```

Accept only after binding the exact commit, tests and evidence reviewed:

```text
python -m tools.eiel sessionManager complete SES-REVIEW-... \
  --review-verdict ACCEPT \
  --tests-reviewed "pytest -q tests/test_feature.py" \
  --evidence-reviewed "clean implementation commit and focused test output" \
  --known-limitation "bounded package scope only"
```

A legal repair cycle is:

```text
IMPLEMENTATION_COMPLETE
  -> CHANGES_REQUIRED --repair-disposition WITHIN_PACKAGE
  -> new bounded delivery Session
  -> new Review Session targeting the repair
  -> ACCEPT
```

For `CHANGES_REQUIRED`, provide at least one `--review-finding` and
`--repair-disposition WITHIN_PACKAGE`. If review discovers changed scope,
architecture, authority or uncertainty, use `PROMOTE` with
`REVISED_PACKAGE_REQUIRED`; do not disguise it as repair. Any new delivery
commit invalidates earlier reviews.

Other verdicts are `REJECT` and `BLOCKED`. The implementer cannot act as its own
reviewer. Work completion names the accepted review explicitly:

```text
python -m tools.eiel workManager complete WORK-042 --review-session SES-REVIEW-...
```

Unpackaged legacy Work remains compatible with the earlier completion flow.

### Administrative reconciliation

Ordinary `--force` behavior never bypasses a package promotion or review gate.
For genuinely corrupted historical state, use the separate fail-loud path:

```text
python -m tools.eiel workManager reconcile WORK-... \
  --disposition DISCONTINUE \
  --reason "Historical actor state is irrecoverably inconsistent." \
  --acknowledge-no-acceptance
```

This preserves history, records active Sessions and operator reasoning, never
sets `COMPLETED`, never implies acceptance, and never releases dependencies.
Reconcile the affected Session separately using the exact next action returned.

## 5. Mission and guardrails only when needed

Mission recovery and authority inspection remain available through the exact
compatibility commands:

```text
python -m tools.eiel missionPlanner preflight MIS-...
python -m tools.eiel missionPlanner status MIS-...
python -m tools.eiel missionPlanner history MIS-...
python -m tools.eiel missionPlanner next MIS-...
python -m tools.eiel missionPlanner handoff MIS-...
```

The normal guardrail path is:

```text
python -m tools.eiel missionPlanner status MIS-...
python -m tools.eiel qualificationManager check --mission MIS-... --store
python -m tools.eiel resourceManager assess --mission MIS-...
python -m tools.scripts.eiel_campaign preflight CAMPAIGN_SPEC.json
python -m tools.scripts.eiel_campaign run CAMPAIGN_SPEC.json
python -m tools.eiel evidenceManager assess campaign_evidence_manifest.json --mission MIS-... --deterministic
```

None of these four actor CLIs launches the campaign. Qualification decisions are technical.
Resource decisions are advisory. Evidence decisions are procedural and
profile-scoped. None is a general scientific claim.

For CUDA admission, ordinary functional or determinism diagnostics may opt into
sharing when every current owner has a declared peak allocation and the
projected peak fits the configured usable capacity:

```text
python -m tools.eiel resourceManager assess --device cuda \
  --gpu-admission-mode shared_functional --peak-vram-gib <peak-including-transients>
```

Shared admission returns `ADMIT_WITH_LIMITS` and marks bandwidth, SM,
performance, VRAM-margin, residency-timing, checkpoint/resume, and formal
qualification evidence as non-gating until an exclusive rerun is made.
Use `--gpu-admission-mode exclusive` (or `--exclusive-gpu`) for critical
measurements and formal qualification. Configure a host-specific usable CUDA
capacity with `--usable-vram-gib` when the driver-reported total is not the
safe allocation ceiling.

## 6. Audit and handoff

Run periodically and before transfer:

```text
python -m tools.eiel audit
python -m tools.eiel handoff --bootstrap-record BST-... --session-id SES-...
```

Audit checks:

- record seals, histories and indexes;
- dependency cycles and unresolved blockers;
- terminal/open work-session inconsistencies;
- missing work, bootstrap, mission and manager references;
- multiple active sessions without explicit parallel acknowledgement.
- Work Package existence, digest and cross-worktree binding;
- authority roles, promotion stops and review independence;
- accepted review linkage for completed package-bound Work.

It never repairs state automatically.

The consolidated handoff contains the current repository and bootstrap, work,
session, mission, exact next action, record hashes, and optional bridge status.
It is the preferred input for a weaker replacement Codex.

## 7. Optional bridge

Use only when transport to an existing ChatGPT thread is explicitly wanted:

```text
python -m tools.eiel bridgeManager status
python -m tools.eiel bridgeManager list --json
```

The bridge never starts automatically. Its absence never blocks ordinary work.

## Never infer permission from exit code alone

Always inspect:

```text
status
reason_codes
problems
warnings
next_actions
related_records
```

A command succeeding mechanically does not create programme, qualification,
resource, evidence, or scientific authority.

## Authority reminder

Never treat exit code `0` alone as permission to run. Never treat an exit code
as execution permission or denial. The explicit
operator instruction supplies operational authority. Work Packages,
`ACTIVE_GOAL.md`, and manager records supply additional scope and provenance
when relevant; stale administration must not be represented as current
authority. Specific scientific, qualification, acceptance, and publication
claims still require the exact governing evidence. Reports that rely on an
Evidence Preflight retain this heading:

```text
## Evidence Preflight
```

## Operational continuity

`workManager continue WORK-...` is the deterministic administrative recovery
entry point. For DISCONTINUED, OBSOLETE, or SUPERSEDED Work it creates or
reuses a successor, retains predecessor coordinates, and rewires unsuccessful
dependencies to their own successors. It does not reopen immutable history or
copy a Work Package contract to a different Work ID. Rebind a new package
before claiming repository delivery.

`workManager refresh` automatically moves BLOCKED Work to READY only when all
retained dependencies are COMPLETED and no machine-readable hold remains.
Unsuccessful terminal Work never satisfies a dependency. A real unavailable
route, unsafe runtime, active writer/lease conflict, missing source, or
unsupported claim remains a blocker and must be reported precisely.

The high-level continuation result classifies its own administrative repair as
`SOFT`, scopes it to `action` or `work`, sets
`operator_intervention_required: false`, and records the reconciliation action.
Only a concrete technical or evidence-integrity limitation may be reported as
`HARD`; a lifecycle enum or CLI refusal alone is never that reason.

EIEL may require evidence, preserve history, reject false claims, and protect
user work. It may not require ceremonial human permission for a safe
administrative action Codex can perform itself. If a safe provenance-preserving
sequence would let the work continue, Codex executes it rather than returning
that sequence to the operator as homework.


## Work Item Resource Accounting (WRA-1)

WRA-1 is automatic instrumentation. Work creation records a static estimate;
Session start/resume captures cumulative START telemetry; completion, terminal
outcomes, and promotion boundaries capture END telemetry; Work completion
attaches an immutable aggregate. `ccusage` owns Codex telemetry parsing. EIEL
owns only Work/Session attribution.

Manual recovery and export commands are:

```text
python -m tools.eiel token capture-start SES-...
python -m tools.eiel token capture-end SES-...
python -m tools.eiel token finalize-work WORK-... --correction-note "..."
python -m tools.eiel token-report --output-dir /tmp/eiel-token-report
```

Never paste raw Codex logs into actor state. Never replace `UNAVAILABLE`
evidence with a guess. Estimator review is offline and advisory; only a human
may approve changes to `tools/actors/estimator_rules.md`. See
`tools/actors/WRA1_WORK_ITEM_RESOURCE_ACCOUNTING.md`.

## Optional unattended Assignments

Ignore this section during ordinary manual work. The Scheduler is explicit-only
and never runs because Work merely exists.

The operator may authorize one bounded phase:

```text
python -m tools.eiel assignment authorize \
  --work-id WORK-... \
  --repo-root /path/to/worktree \
  --phase implementation \
  --launcher codex_exec \
  --model <human-selected-model>

python -m tools.eiel assignment candidates --eligible-only
python -m tools.eiel scheduler run-once
```

For several sequential items while away, use the foreground single-worker loop:

```text
python -m tools.eiel scheduler start --max-assignments 3
```

The worker verifies the immutable Assignment, Lease, Bootstrap, repository HEAD,
Work Package digest and Context Receipt before editing. Failure stops that
unattended Assignment; it does not revoke manual operator-authorized work. A
structured Worker Result is required. Scheduler loss is mechanical history,
not an engineering verdict.

Returning to manual work requires no Scheduler runtime. Run `python -m tools.eiel start`. If a Session was prepared but no worker started, use `sessionManager resume SES-... --take-over-prepared --codex-id <actual-thread-id>`. Expired Leases and interrupted Sessions remain repository-visible and manually recoverable.

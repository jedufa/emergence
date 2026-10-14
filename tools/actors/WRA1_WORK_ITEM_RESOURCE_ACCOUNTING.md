# EIEL Work Item Resource Accounting (WRA-1)

## Status

Implemented in EIEL Operator Tooling V1.17.

WRA-1 is instrumentation, not a new authority layer. It automatically binds
Codex token telemetry to durable Work Items and Sessions while keeping parsing
responsibility in commodity tooling.

## Purpose

WRA-1 answers one bounded question:

> How many tokens did the Codex work attributed to this completed Work Item consume?

V1 measures tokens only. It does not calculate money, credits, quota percentage,
or plan consumption. Those values are commercially mutable; token counts are the
historical evidence.

## Authority boundary

```text
ccusage
  owns: How many tokens did this Codex thread use?

EIEL
  owns: Which Work Item, Session, and Context Epoch did those tokens belong to?
```

EIEL does not parse Codex logs. It invokes the external `ccusage` CLI using the
focused Codex-session JSON interface, normalizes only supported token counters,
and stores no raw telemetry response.

`token` is a core utility, not a manager. It owns no Work or Session lifecycle
truth and creates no engineering authority.

## Lifecycle

```text
Create Work Item
  -> add static resource_estimate
Start or bind Session epoch/thread
  -> capture cumulative START snapshot
Do engineering normally
Complete, terminate, or promote Session phase
  -> capture cumulative END snapshot
  -> compute immutable Session delta
Complete Work Item
  -> aggregate current Session artifacts
  -> write immutable token_usage.json
  -> retain only estimate/artifact/error projection in Work
Export periodically
  -> eiel token-report
Review estimator periodically
  -> offline report -> human approval -> estimator_rules.md
```

## Work estimate

Every newly created Work Item contains:

```json
{
  "resource_estimate": {
    "schema_version": "eiel_resource_estimate_v1",
    "estimated_tokens": {
      "low": null,
      "expected": null,
      "high": null
    },
    "confidence": "none",
    "method": "static"
  }
}
```

The range may be supplied at creation:

```bash
python -m tools.eiel work create \
  --work-id WORK-123 \
  --title "Bounded implementation" \
  --subsystem operator-tooling \
  --estimated-tokens-low 2000000 \
  --estimated-tokens-expected 3000000 \
  --estimated-tokens-high 4000000 \
  --estimate-confidence low
```

WRA-1 does not execute estimator rules. Initial values are deliberately rough so
prediction evidence can begin accumulating before anyone builds another oracle
out of three examples and optimism. The creation-time estimate is manager-owned
and cannot be patched later; corrections belong in reviewed estimator rules for
future Work Items, not in a retrospective rewrite of the original prediction.

## Automatic capture

Normal Session start, Scheduler thread binding, and Session resume capture a
START snapshot for the current Context Epoch and Codex thread. Session completion
captures END and computes the delta. Terminal Session outcomes also close an open
resource segment. A promotion request closes the current authority-phase segment;
a later resumed epoch gets its own START snapshot.

Normal Work completion automatically writes the Work-level aggregate.

Every measured interval is non-overlapping. Ending a Context Epoch closes its
active token segment before any resumed epoch captures a new START. If an older
record lacks the END snapshot from an already-ended epoch, WRA-1 records that
segment as `UNAVAILABLE`; it does not query a later cumulative total and pretend
the boundary was preserved.

A Codex thread may own only one open WRA-1 interval across the actor state root.
Before reusing a thread for another Work Item, close the previous Context Epoch
or terminal Session. This prevents the same cumulative token interval from being
attributed to two Work Items.

Manual commands remain available for recovery and correction:

```bash
python -m tools.eiel token capture-start SES-...
python -m tools.eiel token capture-end SES-...
python -m tools.eiel token finalize-work WORK-... \
  --correction-note "operator-confirmed attribution correction"
```

`EIEL_CCUSAGE_COMMAND` may point to an installed command or test wrapper. The
same value can be supplied per invocation with `--ccusage-command` on capture
commands.

## External query contract

When `ccusage codex session --help` supports a thread filter, WRA-1 invokes the
equivalent of:

```bash
ccusage codex session \
  --id <codex-thread-id> \
  --json \
  --no-cost \
  --offline
```

Current `ccusage` releases do not expose `--id`.  In that case WRA-1 invokes
the same structured, offline, no-cost session query without `--id`, then accepts
only a row whose returned session ID exactly matches the EIEL-bound Codex thread
ID.  A missing or non-matching row remains `UNAVAILABLE`; WRA-1 never treats a
sole returned row as proof of attribution.

The implementation:

- passes the thread ID explicitly when the installed interface supports it, or
  exact-matches it from the returned structured session collection otherwise;
- requests structured JSON;
- disables network price lookup;
- excludes cost fields;
- records the `ccusage` version when available;
- hashes the response bytes for provenance;
- retains only normalized token counters and selected model names;
- never accepts negative or malformed counters as zero.

Missing executable, timeout, non-zero exit, malformed JSON, absent thread,
missing counters, or decreasing cumulative counters produce explicit
`UNAVAILABLE` evidence. They never invent usage and do not invalidate otherwise
valid engineering lifecycle completion.

## Artifact layout

Artifacts live below the actor state root:

```text
resource-accounting/
  sessions/
    SES-.../
      EPOCH-.../
        *.start_snapshot.json
        *.end_snapshot.json
        *.session_token_usage.json
  work/
    WORK-.../
      *.token_usage.json
```

The Work record stores only:

```json
{
  "resource_accounting": {
    "estimate": {},
    "artifact": {},
    "estimation_error": {},
    "artifact_history": []
  }
}
```

Raw counters remain in immutable artifacts, not mutable Work state.

## Artifact status

- `MEASURED`: every currently attributed segment has a valid delta.
- `PARTIAL`: at least one segment is measured and at least one is unavailable.
- `UNAVAILABLE`: no trustworthy measured segment exists.

Corrections create new immutable artifacts. The previous Work artifact reference
moves to `artifact_history`; it is never overwritten. An explicit Session END
correction likewise retains the prior Session artifact and re-samples `ccusage`
without accepting hand-entered counters. The operator must confirm in the
correction note that the current cumulative telemetry represents the intended
historical boundary; otherwise an unreconstructable interval remains
`UNAVAILABLE` rather than being cosmetically rescued.

## Export

Export all attached token artifacts from completed Work Items:

```bash
python -m tools.eiel token-report \
  --output-dir /tmp/eiel-token-report \
  --human \
  --output /tmp/eiel-token-report-command.json
```

Output:

```text
/tmp/eiel-token-report/
  token_report_index.json
  WORK-123/token_usage.json
  WORK-124/token_usage.json
```

The actor-state repository remains the source evidence. Export is a deterministic
copy for analysis, not a second database.

## Estimator review

Estimator evolution is manual and reviewed:

1. Export recent completed Work artifacts.
2. Run an offline Codex review using `WRA1_ESTIMATOR_REVIEW_PROMPT.md`.
3. Produce an Estimator Review Report.
4. A human accepts, rejects, or edits proposed rules.
5. Update `estimator_rules.md` only after approval.

No automatic self-learning or estimator mutation exists in V1.

## Explicitly out of scope

WRA-1 does not implement:

- a Codex log parser;
- costs, credits, prices, or quota percentages;
- a database or service;
- automatic model routing;
- automatic estimator learning;
- automatic changes to `estimator_rules.md`;
- CPU/GPU, wall-clock, shell-output, or review-iteration accounting beyond
  existing fields recorded when already available;
- scientific or engineering acceptance decisions based on token use.

Later resource dimensions require separate evidence and an explicit versioned
change. Tokens are the first dimension, not a secret new fitness function for
software development.

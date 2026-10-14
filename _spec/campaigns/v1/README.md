# Campaign V1 time authority and lifecycle binding

## Active Lifecycle V2 / EIEL_FULL_EVOLUTION_V1 input

Active Lifecycle V2 inputs are tick-native. Copy this execution shape and bind
the remaining campaign fields through the canonical compiler:

```yaml
execution:
  backend: cuda_resident
  execution_route_identity: tick_native_execution_route_v1
  total_ticks: 5000
  start_next_tick: 0
  checkpoint_ticks: [2500, 5000]
  resident_owner_span_ticks: 1120
  native_dispatch_window_ticks: 1120
  refill_chunk_ticks: 160
  source_window_ticks: 160
  event_window_capacity: 65536
```

`total_ticks` is the horizon. Owner, dispatch, refill, source, and event
controls bound resident engineering work and do not define a biological or
qualification generation. Reporting, checkpoint, science, reducer, and
archive schedules are independent absolute-tick schedules.

The active lifecycle checkpoint shape is also tick-addressed:

```yaml
artifact_lifecycle:
  checkpoints:
    authority: absolute_tick
    checkpoint_ticks: [2500, 5000]
    roles:
      - identity: checkpoint-tick-000002500
      checkpoint_tick: 2500
        role: intermediate
        retention_class: R1
      - identity: checkpoint-tick-000005000
      checkpoint_tick: 5000
        role: terminal
        retention_class: R4
```

The canonical complete example is
[`tw_rngv2/science_representative_5000tick_v1.yaml`](tw_rngv2/science_representative_5000tick_v1.yaml).
Its observational `no_claims` ceiling still applies; compiling a 5,000-tick
request is not qualification evidence.

## Legacy compatibility input

Generation-shaped input is valid only when the execution object explicitly
selects `legacy_generation_route_v1`. It remains available for existing legacy
campaigns and readers; it is not a template for Lifecycle V2:

```yaml
execution:
  backend: cpu
  execution_route_identity: legacy_generation_route_v1
  generations: 14
  ticks_per_generation: 80
```

Historical files may retain their original fields and filenames. Their current
interpretation is controlled by the
[Tick-Native Authority and Generation-Era Supersession Index](../../../governance/EIEL_Tick_Native_Authority_and_Generation_Era_Supersession_Index.md),
not by copyable text inside a sealed snapshot.

## Lifecycle binding

Every `eiel_campaign_v1` input is compiled by
`engine.campaigns.compiler.compile_campaign`. New inputs may include
`artifact_lifecycle`, an exact `eiel_run_artifact_lifecycle_contract_v1`
object. The compiler validates it and binds its canonical digest into
`CampaignPlanV1.plan_sha256`.

The lifecycle object is intentionally small and has these required sections:

```text
schema_version
lifecycle_profile
storage
  total_max_bytes, scratch_max_bytes, checkpoint_max_bytes,
  durable_max_bytes, emergency_reserve_bytes
observability
  output_profile, raw_tick_trace_policy, debug_event_trace_policy, ring_buffer
  optional override_reason when escalating from the profile default
checkpoints
  schedule, time_authority, roles[identity, tick, role, retention_class]
closeout
  required_types, compact_replacement_verification_required
retention
  default_class, decision_evidence_class, audit_hold_binding,
  qualification_hold_binding, unknown_artifact_policy
prune
  automatic_prune_allowed, required_predicates
```

For tick-native execution, the schedule must cover the requested absolute
checkpoint ticks and terminal tick, every role uses `tick`, and
`time_authority` is `absolute_tick_v1`. Legacy generation roles remain schema
compatible only inside an explicitly selected legacy route. Checkpoint roles
are explicit (`intermediate`, `terminal`, or `qualification_required`) and are
carried into the run passport and disposition inventory.

Profile defaults and operator guidance are in
[`governance/EIEL_Run_Artifact_Lifecycle_Runbook.md`](../../../governance/EIEL_Run_Artifact_Lifecycle_Runbook.md).
The four profiles are:

- `CHECKPOINT_TRANSIT` for long checkpoint-centric evolution;
- `SCIENCE_COMPACT` for compact decision-bearing campaigns (the normal
  `compact_default` output; `compact_funnel` is an explicit escalation);
- `QUALIFICATION` for evidence-driven qualification output (the normal
  `compact_default` output; full replay/debug observability is an explicit,
  reason-bound escalation);
- `DEBUG_FORENSIC` for quota-bounded engineering diagnostics.

Campaigns that predate this field receive a conservative derived contract for
plan/preflight compatibility. The derived contract does not enable automatic
prune. Existing campaign paths and receipts remain valid; the lifecycle
passport, disposition manifest, and storage receipt are additive run artifacts.

Current B136 science authority is not established by the bounded envelope.
Long-horizon qualification requires evidence from at least 5,000 absolute
ticks on the exact B136 W32/P256/C512 route plus the governing continuation,
memory, event, checkpoint, and resource-safety obligations.

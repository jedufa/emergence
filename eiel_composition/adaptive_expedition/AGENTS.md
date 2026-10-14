# Adaptive Expedition implementation boundary

This package is the additive, no-claims outer science coordinator.  It is not a
scheduler, provider, evidence authority, lifecycle manager, or mutable runtime
controller.

## Non-negotiable dependency rules

- Coordinator and recovery code consume `ExpeditionPolicyV1`; they must not
  import or interpret Adaptive Illumination, MAP-Elites, niches, novelty,
  distances, thresholds, quotas, replicate counts, evolvability, or phenotype
  names.
- Policy implementations are pure.  They may use standard-library code and
  low-dependency EIEL contracts, but not filesystem, network, wall-clock,
  ambient randomness, providers, simulation/runtime packages, lifecycle
  writers, or execution front doors.
- Runtime execution enters only through the canonical Science Request front
  door.  No expedition module may call a provider, Launchpad child, Resource
  Manager actuator, or storage cleanup path directly.
- Absolute ticks are the only scientific time coordinate.  Generation-shaped
  fields are forbidden from requests, StateRefs, decisions, journals, reports,
  and branch identity.
- Snapshots are observation-only unless a separately validated exact resolver
  proves a restorable state contract.

## Scientific and storage boundaries

- `EvolutionaryAtlasV1` is compact epistemic memory.  It never grants
  branchability or storage retention.
- `RetainedParentBankV1` contains only exact or exactly replayable StateRefs.
  Same-niche states may coexist when the bound policy proves distinct marginal
  lineage, genotype, structure, latent-state, control-history, comparator, or
  historical value.
- Parent-bank eviction removes a reference or hold; it never deletes bytes.
  Existing lifecycle manifests and actuators remain authoritative.
- Named phenotype authority is allowed only through an explicit, manifest-bound
  bounded probe contract.  Otherwise phenomena remain opaque evidence IDs.

## Programme fence

Implement and validate G0-G5, then seal
`ENGINEERING_G5_COMPLETE_AWAITING_HUMAN_REVIEW`.  Do not launch 24-hour,
72-hour, week-scale, PREQUAL, or Qualification work in this programme.

# Tests Topology (Canonical)

This repository uses a topic-based test layout under `tests/`.

## Determinism

- Core:
  - `tests/determinism/test_static_rng_scan.py`
  - `tests/determinism/test_checkpoint_rng_authority.py`
  - `tests/determinism/test_no_rng_islands_runtime.py`
- P0:
  - `tests/determinism/p0/test_p0c_torch_rng_migration.py`
  - `tests/determinism/p0/test_p0d_numpy_python_rng_migration.py`
  - `tests/determinism/p0/test_p0e_rng_budget_enforcement.py`
- Harness:
  - Recommended semantic entrypoints (no phase labels):
    - `tests/determinism/harnesses/determinism_foundations_e2e.sh`
    - `tests/determinism/harnesses/checkpoint_resume_determinism_e2e.sh`
    - `tests/determinism/harnesses/checkpoint_resume_forensics_e2e.sh`
  - Canonical historical entrypoints:
  - `tests/determinism/harnesses/determinism_foundations_e2e.sh`
    - `tests/determinism/harnesses/checkpoint_resume_determinism_e2e.sh`
    - `tests/determinism/harnesses/checkpoint_resume_forensics_e2e.sh`

## Evolvability

- Phase A:
  - `tests/evolvability/phase_a/test_p1a_chemistry_foundation.py`
  - `tests/evolvability/phase_a/test_p1a2_maintenance_delayed_process.py`
  - `tests/evolvability/phase_a/test_p1a3_chemistry_functional_coupling.py`
  - `tests/evolvability/phase_a/test_chemistry_intent_validation.py`
- Phase B:
  - `tests/evolvability/phase_b/test_b1_modular_genome_canonicalization.py`
  - `tests/evolvability/phase_b/test_b2_module_interfaces_budgets.py`
  - `tests/evolvability/phase_b/test_b3_locality_constraints.py`

## Control Plane

- Core:
  - `tests/control_plane/test_control_plane_timeline_freeze.py`
  - `tests/control_plane/test_control_plane_inbox.py`

## Harnesses

- `tests/harnesses/phase3_resume/`
- `tests/harnesses/phase4_functional/`

## Integration

- `tests/integration/metrics/`
- `tests/integration/phase5b_e2e/`
- `tests/integration/branching/`
- `tests/integration/ui_prereqs/`

## Legacy Paths

- Deprecated harness shims remain for compatibility:
  - `tests/_legacy/shims/determinism_foundations_legacy/evolvability_phase0_determinism_e2e.sh`
  - `tests/_legacy/shims/checkpoint_resume_legacy/phase3_resume_determinism_e2e.sh`
- Legacy deterministic artifacts are archived outside canonical test trees.

## Migration Note

This Emergence project test tree has been pruned to keep engine- and
experiment-local coverage.

Removed in the first migration pass:
- Causyl/WebUI-facing `tests/control_plane/webui/`
- scheduler/service tests under `tests/control_plane/`
- donor `tests/tools/` suite

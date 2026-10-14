# Experiment Registry & Validation Harness (v1)

This folder contains a minimal, boring system for defining and running **reproducible** experiment matrices.

It is infrastructure, not intelligence:
- No agent-level access.
- No adaptive tuning.
- No outcome optimization.

## Concepts (quick)

### Experiment Registry
An experiment registry is a **declarative YAML file** that answers:

"What did we run?"

It defines:
- the axes (e.g. dev-coupling off/on)
- the seeds
- the fixed CLI defaults (generations/steps/pop/etc.)
- how to name runs (stable run_id/run_tag)

### Validation Harness
The harness is a runner that:
- reads a registry YAML
- expands the full matrix deterministically
- executes runs via `scripts/utils/run_evo.py`
- writes provenance (registry hash, git commit, python + library versions)
- validates that required artifacts exist after each run

## Where files live

Registry files:

```
_spec/registry/v1/*.yaml
```

Harness code:

```
experiments/harness/run_experiments.py
```

## Running an experiment

### 1) Dry-run (preview)

This expands the matrix, writes `matrix.csv` + `manifest.json`, prints commands, and runs nothing.

```
python -m experiments.harness.run_experiments \
  --registry _spec/registry/v1/validation_2x2_devcoupling_latecoh_smoke.yaml \
  --runs_root runs \
  --dry_run
```

### 2) Execute

```
python -m experiments.harness.run_experiments \
  --registry _spec/registry/v1/validation_2x2_devcoupling_latecoh_smoke.yaml \
  --runs_root runs
```

Outputs go under:

```
runs/experiments/<experiment_name>/v<registry_version>/<registry_hash>__<git_sha>/<timestamp>__<batch_id>/
  manifest.json
  matrix.csv
  summary.json
  runs/
    <run_id>/
      manifest.json
      run_config.csv
      config.json
      metrics.csv
      ...
```

## Nox integration

This repo includes optional nox sessions:
- `nox -s validation_smoke_registry`
- `nox -s validation_quick_registry`
- `nox -s validation_nightly_registry`

They call the harness with a specific registry file.

## Registry format (v1)

See the example files in `_spec/registry/v1/`.

Notes:
- Axis values are **labels** (`off/on`), not booleans.
- Each label maps to a list of CLI flags (may be empty `[]`).
- Seeds must be unique integers.
- `matrix_order` makes ordering deterministic forever.
- `expected_artifacts` is used for post-run correctness checks.

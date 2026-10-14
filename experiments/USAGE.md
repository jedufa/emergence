# How to use the Experiment Registry & Validation Harness (v1)

This document is intentionally practical. It tells you exactly what to run and what artifacts you should expect.

## Quick glossary

- **Registry**: A YAML file that declares an experiment (axes, seeds, flags, naming).
- **Harness**: The runner that expands the registry into a run matrix, executes the runs, and writes reproducibility metadata.
- **Batch**: One harness invocation. A batch contains `matrix.csv`, `manifest.json`, `summary.json`, and a `runs/` folder with the run directories.

## Where things live

- Registry files: `_spec/registry/v1/*.yaml`
- Harness code: `experiments/harness/run_experiments.py`
- Outputs: under `runs/experiments/...`

## The three built-in v1 registries

These registries run the same 2×2 grid (dev-coupling × late-life coherence pressure) at different scales:

- Smoke: `_spec/registry/v1/validation_2x2_devcoupling_latecoh_smoke.yaml`
- Quick: `_spec/registry/v1/validation_2x2_devcoupling_latecoh_quick.yaml`
- Nightly: `_spec/registry/v1/validation_2x2_devcoupling_latecoh_nightly.yaml`

## Run from the command line

### 1) Dry-run (preview the matrix safely)

Writes `manifest.json` and `matrix.csv`, prints commands, runs nothing.

```bash
  python -m experiments.harness.run_experiments \
  --registry _spec/registry/v1/validation_2x2_devcoupling_latecoh_smoke.yaml \
  --runs_root runs \
  --dry_run
```

### 2) Execute

```bash
  python -m experiments.harness.run_experiments \
  --registry _spec/registry/v1/validation_2x2_devcoupling_latecoh_smoke.yaml \
  --runs_root runs
```

### 3) Overwrite behavior (evidence protection)

The harness refuses to write into an existing batch directory.

If you *really* need to overwrite (usually you don’t), use:

```bash
... --overwrite
```

## Run via nox

These sessions are thin wrappers that just call the harness:

```bash
nox -s validation_smoke_registry
nox -s validation_quick_registry
nox -s validation_nightly_registry
```

Extra smoke test (graceful Ctrl-C => interrupt checkpoint):

```bash
nox -s smoke_interrupt_checkpoint
```

## Ctrl-C behavior (and why it still works under the harness)

- The **simulation loop** handles SIGINT/SIGTERM by requesting a stop and then writing an
  **interrupt checkpoint at the end of the current generation**.
- The harness launches runs via `scripts/utils/run_evo.py`, which forwards Ctrl-C to the
  child process group. If you interrupt a batch, the harness writes `summary.json` with
  `status="interrupted"` and stops launching additional runs.

So: Ctrl-C remains a safe "stop and leave me a checkpoint" behavior, even when you run
through the registry.

## Why some commands use `python -m ...` instead of `python path/to/file.py`

This repo uses absolute imports like `from scripts.utils.run_evo import ...`.

- Running a module (`python -m experiments.runs.smoke`) keeps the repo root on `sys.path`.
- Running a file path directly can set `sys.path[0]` to the script's folder and break
  absolute imports in subtle ways.

Nox is just the launcher menu. The harness and run scripts are normal Python modules.

## What outputs are guaranteed

Every batch folder contains:

- `manifest.json` (batch-level provenance)
- `matrix.csv` (one row per run, fully resolved CLI)
- `summary.json` (per-run status: ok/failed/missing artifacts)
- `runs/` directory containing run directories

Every **run directory** must contain (project-mandatory correctness checks):

- `manifest.json`
- `run_config.csv`
- `config.json`
- `metrics.csv`

If any required file is missing, the harness marks that run as `missing_artifacts`.

## How to add your own experiment

1) Copy an existing registry YAML.
2) Change:
   - `name`
   - `axes` (values and flags)
   - `seeds`
   - `cli_defaults`
   - `naming.run_id_format` and `run_tag_prefix`
3) Run with `--dry_run` first.

## Troubleshooting

### "PyYAML is required"
Install it:

```bash
python -m pip install pyyaml
```

This repo also includes it in `engine/requirements.txt`, so nox installs will include it.

### "Refusing to write into existing directory"
This is intentional evidence protection. Either:
- delete the old batch folder, or
- run again (you’ll get a new batch_id), or
- use `--overwrite` if you’re absolutely sure.

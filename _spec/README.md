# Emergence Spec

This folder is the Causyl-facing specification surface for the Emergence
project.

Purpose:
- define what experiments exist
- define named scenarios, matrices, and campaigns
- provide stable project-level references that the plane can use without
  needing to know old EIEL repo internals

Current source of truth during migration:
- `registry/`
- `../experiments/harness/`
- `../experiments/harnesses/`

Current canonical spec surface:
- `registry/` is the Causyl-facing reference point for named Emergence
  experiment definitions
- `registry/v1/patch_jungle_v0_smoke.yaml` is the Step 14 Patch/Jungle V0
  smoke matrix for deterministic evaluation-harness verification only.

Step 14 dry-run check:

```bash
python -m _runner.run_registry_batch \
  --registry v1/patch_jungle_v0_smoke.yaml \
  --runs_root /tmp/emergence_step14_matrix \
  --batch_tag step14 \
  --dry_run \
  --overwrite
```

This command materializes and validates the 10-scenario smoke matrix without
starting long experiments.

Near-term intent:
- move or mirror named Emergence experiment definitions here
- keep this folder focused on experiment definitions, not launch mechanics

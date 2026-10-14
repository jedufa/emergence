# Emergence Runner

This folder is the Causyl integration seam for launching Emergence work.

Purpose:
- define how Causyl launches Emergence experiments
- provide a narrow adapter boundary between the plane and old EIEL internals
- keep launch semantics separate from the experiment definitions themselves

Current implementation inputs during migration:
- `../tools/core/run_job.py`
- `../tools/entrypoints/run_registry_case.py`
- `../tools/manifest/`
- `../engine/main.py`

Materialized runner seam:
- `run_registry_case.py` is now the project-local adapter for launching one
  Emergence registry case
- `run_registry_batch.py` is now the project-local adapter for launching a full
  registry batch or a selected subset
- both resolve registry references against `../_spec/registry/` and then
  delegate into the retained Emergence harness layer

Near-term intent:
- keep the `_runner/` entrypoints narrow and explicit as the first stable
  launch seams
- make the launch contract explicit: inputs, outputs, artifact paths, and
  expected provenance

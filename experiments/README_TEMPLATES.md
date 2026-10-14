Harness Test Templates (Nightly / Quick / Smoke)

These are drop-in templates matching the registry-driven harness pattern used by the spatial ecology tests.

EOM-1 evidence requirement:
- Before creating or adapting a harness, run:
  `python -m tools.evidence.librarian preflight --topic "<feature-or-campaign-topic>" --source-root tools/evidence/librarian/catalog_source/00-catalog --json`
- Preserve the receipt path with the generated report or handoff.
- The report must start with an Evidence Preflight section listing authority consulted, evidence IDs, claims consulted, limitations reviewed, unknown authority, catalog build ID, and maintained source digest.
- If authority is unknown or ambiguous, do only bounded repository fallback and record why.

How to use (copy/edit):
1) Rename files:
   - Replace "REPLACE_ME_FEATURE" in filenames with your feature name.
2) Edit the YAML registries:
   - Update intent.question
   - Replace flags under axes.scenario.flags.baseline/variant
   - Update expected_artifacts to what your feature MUST produce
3) Edit runner scripts:
   - Update module docstrings and registry paths if you renamed them
4) Run locally:
   - python experiments/runs/<feature>_registry_smoke.py
   - python experiments/runs/<feature>_registry_quick.py
   - python experiments/runs/<feature>_registry_nightly.py

Tip:
- Keep "baseline" vs "variant" as a 2-scenario A/B test for new features.
- Add more axes only when necessary (and log why).

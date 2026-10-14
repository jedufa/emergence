# Golden References

This directory stores deterministic replay references in one co-located layout.

## Tiers

- `mini`: fast unit-scale deterministic sentinels.
- `macro`: integration-scale deterministic checks.
- `mega`: high-confidence replay parity checks.

## Profiles

- `quick`: 6 cases (core mini).
- `standard`: 16 cases (all mini + selected macro).
- `nightly`: all 24 cases.

Catalog: `golden_suite_catalog.json`

Runner:

```bash
# default = quick
./tests/determinism/golden_references/run_golden_suite.py
./tests/determinism/golden_references/run_golden_suite.py quick
./tests/determinism/golden_references/run_golden_suite.py standard
./tests/determinism/golden_references/run_golden_suite.py nightly
```

Legacy flag form is still supported:

```bash
python tests/determinism/golden_references/run_golden_suite.py --profile quick
```

Micro-case folders (reference material) contain:

- `reference_input.json`: deterministic input payload for the micro case.
- `expected_output.json`: expected deterministic output payload/hash metadata.
- `expected_rng_delta.json`: expected per-stream/domain RNG delta map.

Design goal: keep references and tests together so replay drift is easy to audit.

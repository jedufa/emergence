# Qualification mechanical review

This package turns deterministic qualification evidence review into maintained,
read-only tooling. It validates a declarative JSON spec and emits exactly one
canonical JSON receipt. `MECHANICAL_REVIEW` reports PASS/FAIL with every check,
error, warning, rule hash, and input hash. `SEMANTIC_REVIEW_REQUIRED` contains
only questions created by the qualification procedure's risk triggers and the
minimum evidence pointers supplied for those questions.

The tool reads JSON and filesystem metadata only. Specs cannot contain commands
(the schema is closed) and the implementation never invokes a shell, controls a
process, or imports evidence as code. Validators are read-only; the CLI writes
only the explicit `--output` receipt target.

Run the maintained example:

```bash
python -m tools.qualification_review \
  tools/qualification_review/examples/review_spec.v1.json \
  --now 2026-08-14T10:00:00Z \
  --output /tmp/qualification-review.json
```

A PASS exit is zero; mechanical FAIL is one. The receipt may still require
semantic review: mechanical integrity does not decide evidence meaning or grant
authority.

## Specification sections

- `packet_validations`: validate packets against explicitly versioned JSON Schema.
- `hash_graphs`: verify exact file hashes and JSON-pointer hash bindings; cycles
  fail when forbidden.
- `json_assertions`: apply closed, data-only JSON-pointer assertions.
- `inventories`: compare a recursive evidence tree to an exact path set and
  optional per-file hashes.
- `freshness_checks`: bind reusable PASS receipts to JSON timestamps or file
  mtimes and explicit maximum ages.
- `authority_transitions`: prove mechanical preconditions and emit semantic
  questions only for authority, dependency, inheritance, evidence-condition,
  limitation/no-claims, or irreversible-decision triggers.
- `expected_negative_receipts`: require the exact nonzero exit, exact reason set
  (and order by default), and exact state assertions.
- `process_convergence`: verify declared owner/PID observations, terminal source
  publication, then a later exact absence snapshot. It does not inspect or kill
  live processes.

All paths are relative to the spec directory and must remain beneath it.
Absolute paths, escapes, and symlink evidence are rejected. Each check ID must
be globally unique.

Packet schemas may use local `$defs`. V1 does not retrieve remote or external
schema references; compose required definitions into the versioned schema file.

## Receipt reuse

Set `freshness_contract.reusable` and include at least one freshness check. A
prior green receipt may be supplied with `--reuse-receipt`. Reuse occurs only if
the receipt self-hash, schema, spec hash, packaged validator/rule/schema hashes,
the exact spec-derived input path set, every input hash, and declared validity
window still match. Otherwise the tool performs a fresh review. Reuse never
carries a semantic approval.

The v1 receipt schema and risk rules are versioned under `schemas/` and `rules/`.
Adding an operator, validator, or risk rule requires a new compatible asset
version or a deliberate schema-version change; arbitrary extension fields fail
closed.
